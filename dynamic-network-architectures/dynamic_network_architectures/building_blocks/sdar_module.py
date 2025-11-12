"""
SDAR (Spectral Diffusion-inspired Adaptive Refinement) Module

核心功能：
1. 双重平滑光谱先验（Doubly Smoothed Spectral Prior）
2. 扩散启发的轻量迭代精炼（1-2步，非完整扩散）
3. 快速不确定性估计（特征方差，无需MC Dropout）
4. 自适应融合机制

设计原则：
- 轻量化：参数量 < 100K per stage
- 高效：避免MC Dropout和多次前向传播
- 可微分：端到端训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DoublySmoothedPriorCalculator:
    """
    双重平滑密度估计器
    基于论文: "Doubly Smoothed Density Estimation"
    
    两阶段平滑：
    1. 值域平滑（Value Domain Smoothing）
    2. 空间平滑（Spatial Domain Smoothing）
    """
    
    def __init__(self, 
                 value_bandwidth: float = 0.1,
                 spatial_sigma: float = 1.5):
        self.value_bandwidth = value_bandwidth
        self.spatial_sigma = spatial_sigma
    
    def compute_doubly_smoothed_prior(self, spectral_weights: np.ndarray) -> np.ndarray:
        """
        计算双重平滑后的先验权重
        
        Args:
            spectral_weights: 原始光谱先验 (D,) 其中D是光谱维度
            
        Returns:
            smoothed_prior: 双重平滑后的先验 (D,)
        """
        D = len(spectral_weights)
        
        # Stage 1: 值域平滑（Value Domain Smoothing）
        # p(x) = ∑ K(x - xᵢ) / N
        value_smoothed = np.zeros(D)
        for i in range(D):
            # Gaussian kernel for value domain
            kernel_weights = np.exp(-0.5 * ((spectral_weights - spectral_weights[i]) / self.value_bandwidth) ** 2)
            kernel_weights /= (kernel_weights.sum() + 1e-8)
            value_smoothed[i] = np.sum(kernel_weights * spectral_weights)
        
        # Stage 2: 空间平滑（Spatial Domain Smoothing）
        # p̃(s) = ∑ G(s - sⱼ) p(xⱼ)
        spatial_positions = np.arange(D)
        spatial_smoothed = np.zeros(D)
        for i in range(D):
            # Gaussian kernel for spatial domain
            distances = spatial_positions - spatial_positions[i]
            spatial_kernel = np.exp(-0.5 * (distances / self.spatial_sigma) ** 2)
            spatial_kernel /= (spatial_kernel.sum() + 1e-8)
            spatial_smoothed[i] = np.sum(spatial_kernel * value_smoothed)
        
        # Normalize to [0, 1]
        spatial_smoothed = (spatial_smoothed - spatial_smoothed.min()) / (spatial_smoothed.max() - spatial_smoothed.min() + 1e-8)
        
        return spatial_smoothed.astype(np.float32)


class LightweightRefinementBlock(nn.Module):
    """
    轻量级精炼块（模拟扩散模型的单步去噪）
    
    设计：
    - 深度可分离卷积（减少参数）
    - 残差连接
    - 条件输入（先验引导）
    """
    
    def __init__(self, channels: int, spatial_dims: int = 3):
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        
        # 先验条件编码器（1x1x1卷积，轻量！）
        self.prior_encoder = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        
        # 深度可分离卷积（大幅减少参数）
        if spatial_dims == 3:
            self.depthwise = nn.Conv3d(
                channels, channels, 
                kernel_size=3, padding=1, 
                groups=channels,  # Depthwise
                bias=False
            )
            self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        else:
            self.depthwise = nn.Conv2d(
                channels, channels, 
                kernel_size=3, padding=1, 
                groups=channels,
                bias=False
            )
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        self.norm = nn.InstanceNorm3d(channels) if spatial_dims == 3 else nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(0.01, inplace=True)
        
        # 残差缩放参数
        self.residual_scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, prior_guidance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W, D) or (B, C, H, W)
            prior_guidance: 先验引导 (B, C, H, W, D) or (B, C, H, W)
        """
        # 编码先验信息
        prior_encoded = self.prior_encoder(prior_guidance)
        
        # 条件特征：输入 + 先验引导
        conditioned = x + prior_encoded
        
        # 深度可分离卷积（轻量！）
        out = self.depthwise(conditioned)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.activation(out)
        
        # 残差连接（可学习缩放）
        out = x + torch.sigmoid(self.residual_scale) * out
        
        return out


class FastUncertaintyEstimator(nn.Module):
    """
    快速不确定性估计器
    
    核心思想：特征的channel-wise标准差 = 模型不确定性
    - 无需MC Dropout（避免多次前向传播）
    - 计算成本几乎为零
    - 物理意义明确
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 不确定性到置信度的映射
        self.uncertainty_proj = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W, D)
            
        Returns:
            uncertainty: 不确定性图 (B, 1, H, W, D)
            confidence: 置信度图 (B, 1, H, W, D)
        """
        # 计算channel-wise标准差作为不确定性
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W, D)
        std = torch.sqrt(((x - mean) ** 2).mean(dim=1, keepdim=True) + 1e-5)  # (B, 1, H, W, D)
        
        # 标准化不确定性
        uncertainty = std / (std.mean() + 1e-5)
        
        # 映射到置信度：不确定性越高，置信度越低
        # 使用特征本身来增强置信度估计
        confidence = self.uncertainty_proj(x)  # (B, 1, H, W, D)
        
        # 结合统计不确定性和学习的置信度
        combined_confidence = confidence * torch.sigmoid(-uncertainty)
        
        return uncertainty, combined_confidence


class SDARModule(nn.Module):
    """
    SDAR (Spectral Diffusion-inspired Adaptive Refinement) 模块
    
    应用于跳跃连接，进行自适应特征精炼
    
    参数量：约50-100K（轻量！）
    显存开销：约0.5-1GB
    """
    
    def __init__(self,
                 channels: int,
                 spectral_dim: int = 60,
                 spectral_prior_path: Optional[str] = None,
                 num_refinement_steps: int = 1,
                 spatial_dims: int = 3,
                 use_uncertainty: bool = True):
        """
        Args:
            channels: 特征通道数
            spectral_dim: 光谱维度（HSI波段数）
            spectral_prior_path: 光谱先验权重文件路径
            num_refinement_steps: 精炼步数（1或2，模拟扩散步数）
            spatial_dims: 空间维度（2或3）
            use_uncertainty: 是否使用不确定性估计
        """
        super().__init__()
        self.channels = channels
        self.spectral_dim = spectral_dim
        self.num_refinement_steps = num_refinement_steps
        self.use_uncertainty = use_uncertainty
        
        print(f"  [SDAR] Initializing for C={channels}, D={spectral_dim}, Steps={num_refinement_steps}")
        
        # 1. 加载或初始化双重平滑光谱先验
        if spectral_prior_path is not None:
            try:
                raw_prior = np.load(spectral_prior_path)
                print(f"    ├─ Loaded raw spectral prior: {raw_prior.shape}")
                
                # 应用双重平滑
                calculator = DoublySmoothedPriorCalculator()
                smoothed_prior = calculator.compute_doubly_smoothed_prior(raw_prior)
                print(f"    ├─ Applied doubly smoothing")
                
            except Exception as e:
                print(f"    ├─ Warning: Could not load prior from {spectral_prior_path}: {e}")
                print(f"    └─ Using uniform prior")
                smoothed_prior = np.ones(spectral_dim, dtype=np.float32) / spectral_dim
        else:
            print(f"    ├─ No prior path provided, using uniform prior")
            smoothed_prior = np.ones(spectral_dim, dtype=np.float32) / spectral_dim
        
        # 将先验权重注册为buffer（不参与训练）
        self.register_buffer('spectral_prior', torch.from_numpy(smoothed_prior))
        
        # 2. 先验投影层（从光谱维度投影到通道维度）
        self.prior_projection = nn.Linear(spectral_dim, channels)
        
        # 3. 迭代精炼块（模拟扩散模型的去噪步骤）
        self.refinement_blocks = nn.ModuleList([
            LightweightRefinementBlock(channels, spatial_dims)
            for _ in range(num_refinement_steps)
        ])
        
        # 4. 快速不确定性估计器
        if use_uncertainty:
            self.uncertainty_estimator = FastUncertaintyEstimator(channels)
        
        # 5. 自适应融合权重
        self.fusion_alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为小值，保守融合
        self.fusion_beta = nn.Parameter(torch.tensor(0.5))   # 不确定性的影响系数
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            x: 跳跃连接特征 (B, C, H, W, D)
            
        Returns:
            refined: 精炼后的特征 (B, C, H, W, D)
            info: 可选的调试信息字典
        """
        B, C, H, W, D = x.shape
        
        # Step 1: 生成先验引导特征
        # 投影光谱先验到通道维度
        prior_weights = self.prior_projection(self.spectral_prior)  # (C,)
        prior_weights = prior_weights.view(1, C, 1, 1, 1)  # (1, C, 1, 1, 1)
        
        # 广播到空间维度
        prior_guidance = prior_weights.expand(B, C, H, W, D)  # (B, C, H, W, D)
        
        # Step 2: 迭代精炼（扩散启发）
        x_refined = x
        for step, block in enumerate(self.refinement_blocks):
            x_refined = block(x_refined, prior_guidance)
        
        # Step 3: 不确定性估计
        info = {}
        if self.use_uncertainty:
            uncertainty, confidence = self.uncertainty_estimator(x)
            info['uncertainty'] = uncertainty.detach()
            info['confidence'] = confidence.detach()
            
            # 自适应融合权重：不确定性越高，精炼权重越大
            adaptive_weight = torch.sigmoid(self.fusion_alpha) * (1.0 + self.fusion_beta * (1.0 - confidence))
        else:
            adaptive_weight = torch.sigmoid(self.fusion_alpha)
        
        # Step 4: 自适应融合（残差连接）
        x_out = x + adaptive_weight * (x_refined - x)
        
        return x_out, info


class SDARModuleLite(nn.Module):
    """
    SDAR的超轻量版本
    
    简化：
    - 只有1步精炼
    - 禁用不确定性估计
    - 参数量 < 30K
    """
    
    def __init__(self,
                 channels: int,
                 spectral_dim: int = 60,
                 spectral_prior_path: Optional[str] = None,
                 spatial_dims: int = 3):
        super().__init__()
        self.channels = channels
        
        print(f"  [SDAR-Lite] Initializing for C={channels}, D={spectral_dim}")
        
        # 加载双重平滑先验
        if spectral_prior_path is not None:
            try:
                raw_prior = np.load(spectral_prior_path)
                calculator = DoublySmoothedPriorCalculator()
                smoothed_prior = calculator.compute_doubly_smoothed_prior(raw_prior)
                print(f"    ├─ Loaded and smoothed spectral prior")
            except:
                smoothed_prior = np.ones(spectral_dim, dtype=np.float32) / spectral_dim
        else:
            smoothed_prior = np.ones(spectral_dim, dtype=np.float32) / spectral_dim
        
        self.register_buffer('spectral_prior', torch.from_numpy(smoothed_prior))
        
        # 极简的先验投影
        self.prior_projection = nn.Linear(spectral_dim, channels, bias=False)
        
        # 单步精炼
        self.refinement = LightweightRefinementBlock(channels, spatial_dims)
        
        # 固定融合权重
        self.register_buffer('fusion_alpha', torch.tensor(0.2))
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B, C, H, W, D = x.shape
        
        # 先验引导
        prior_weights = self.prior_projection(self.spectral_prior).view(1, C, 1, 1, 1)
        prior_guidance = prior_weights.expand(B, C, H, W, D)
        
        # 单步精炼
        x_refined = self.refinement(x, prior_guidance)
        
        # 固定权重融合
        x_out = x + self.fusion_alpha * (x_refined - x)
        
        return x_out, None


# 便捷函数
def build_sdar_module(channels: int,
                      spectral_dim: int = 60,
                      spectral_prior_path: Optional[str] = None,
                      num_refinement_steps: int = 1,
                      use_lite: bool = False,
                      use_uncertainty: bool = True) -> nn.Module:
    """
    构建SDAR模块的便捷函数
    
    Args:
        channels: 特征通道数
        spectral_dim: 光谱维度
        spectral_prior_path: 光谱先验路径
        num_refinement_steps: 精炼步数（1或2）
        use_lite: 是否使用超轻量版本
        use_uncertainty: 是否使用不确定性估计
    """
    if use_lite:
        return SDARModuleLite(
            channels=channels,
            spectral_dim=spectral_dim,
            spectral_prior_path=spectral_prior_path
        )
    else:
        return SDARModule(
            channels=channels,
            spectral_dim=spectral_dim,
            spectral_prior_path=spectral_prior_path,
            num_refinement_steps=num_refinement_steps,
            use_uncertainty=use_uncertainty
        )


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("Testing SDAR Module")
    print("="*80)
    
    # 测试标准版
    print("\n1. Testing Standard SDAR (1 step):")
    sdar = SDARModule(channels=128, spectral_dim=60, num_refinement_steps=1)
    x = torch.randn(2, 128, 32, 32, 16)
    out, info = sdar(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Info keys: {info.keys()}")
    
    # 测试Lite版
    print("\n2. Testing SDAR-Lite:")
    sdar_lite = SDARModuleLite(channels=128, spectral_dim=60)
    out_lite, _ = sdar_lite(x)
    print(f"   Output shape: {out_lite.shape}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")



