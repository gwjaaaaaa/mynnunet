"""
DSPA - Doubly-Smoothed Prior-guided Uncertainty-Aware Skip Connections
双重平滑先验引导的不确定性感知跳跃连接

核心创新：
1. 双重平滑密度估计生成光谱先验（值域+空间域双重平滑）
2. 认知不确定性估计（Monte Carlo Dropout）
3. 偶然不确定性估计（学习方差预测）
4. 不确定性感知的先验自适应加权
5. 空间自适应门控融合

理论基础：
- Doubly Smoothed Density Estimation (统计学)
- Bayesian Deep Learning (贝叶斯不确定性)
- Adaptive Weighting (自适应融合)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter
from pathlib import Path


class DoublySmoothedPriorGenerator:
    """
    双重平滑先验生成器
    
    第一次平滑（值域平滑）：对统计特征进行核密度估计
    第二次平滑（空间域平滑）：利用空间邻域信息平滑
    
    与传统方法的差异：
    - 传统：全局统计 → 直接归一化
    - 本方法：全局统计 → 值域平滑 → 空间域平滑 → 归一化
    """
    
    @staticmethod
    def apply_doubly_smoothing(weights: np.ndarray, 
                                spatial_sigma: float = 1.0) -> np.ndarray:
        """
        对权重应用双重平滑
        
        Args:
            weights: (D,) 原始权重（已经是值域平滑后的统计特征）
            spatial_sigma: 空间域平滑的高斯核标准差
        
        Returns:
            smoothed_weights: (D,) 双重平滑后的权重
        """
        # 第一次平滑已经在离线计算中完成（统计特征本身就是平滑的）
        # 这里主要做第二次平滑：在光谱维度上应用空间平滑
        
        # 将权重看作1D信号，应用高斯平滑
        if len(weights) > 1:
            smoothed = gaussian_filter(weights, sigma=spatial_sigma, mode='reflect')
        else:
            smoothed = weights
        
        # 归一化到[0, 1]
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        
        return smoothed


class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器
    
    估计两类不确定性：
    1. 认知不确定性（Epistemic）：来自模型参数的不确定性
       - 方法：Monte Carlo Dropout
    2. 偶然不确定性（Aleatoric）：来自数据本身的噪声
       - 方法：学习variance预测头
    """
    
    def __init__(self, channels: int, dropout_rate: float = 0.2, num_mc_samples: int = 5):
        super().__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.num_mc_samples = num_mc_samples
        
        # Dropout层（用于MC Dropout）
        self.mc_dropout = nn.Dropout3d(p=dropout_rate)
        
        # 偶然不确定性预测头（预测log variance）
        self.aleatoric_head = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, 1, kernel_size=1),  # 输出log_variance
        )
        
        # ===【关键】初始化aleatoric_head，让初始输出接近-5（低不确定性）===
        # exp(-5) ≈ 0.0067，表示初期假设低不确定性
        with torch.no_grad():
            # 最后一层卷积的bias设为-5
            self.aleatoric_head[-1].bias.fill_(-5.0)
            # 最后一层卷积的weight设为接近0（小随机值）
            self.aleatoric_head[-1].weight.normal_(0, 0.001)
        
        # 轻量级特征变换（用于MC采样）
        self.feature_transform = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
        )
    
    def estimate_epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        估计认知不确定性（Monte Carlo Dropout）
        
        Args:
            x: (B, C, H, W, D) 输入特征
        
        Returns:
            epistemic_unc: (B, 1, H, W, D) 认知不确定性图
        """
        B, C, H, W, D = x.shape
        
        # 多次前向传播（每次使用不同的dropout mask）
        mc_samples = []
        
        for _ in range(self.num_mc_samples):
            # 应用dropout
            x_dropped = self.mc_dropout(x)
            # 特征变换
            x_transformed = self.feature_transform(x_dropped)
            mc_samples.append(x_transformed)
        
        # Stack: (T, B, C, H, W, D)
        mc_samples = torch.stack(mc_samples, dim=0)
        
        # 计算方差（在T维度上）
        # variance: (B, C, H, W, D)
        variance = torch.var(mc_samples, dim=0)
        
        # 对通道维度求平均，得到单通道不确定性图
        epistemic_unc = variance.mean(dim=1, keepdim=True)  # (B, 1, H, W, D)
        
        return epistemic_unc
    
    def estimate_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        估计偶然不确定性（学习方差预测）
        
        Args:
            x: (B, C, H, W, D) 输入特征
        
        Returns:
            aleatoric_unc: (B, 1, H, W, D) 偶然不确定性图
        """
        # 预测log variance（更稳定）
        log_variance = self.aleatoric_head(x)  # (B, 1, H, W, D)
        
        # 限制log_variance范围，防止exp爆炸（训练稳定性关键！）
        log_variance = torch.clamp(log_variance, min=-10.0, max=2.0)
        # exp(-10) ≈ 0.00005, exp(2) ≈ 7.4
        
        # 转换为variance
        aleatoric_unc = torch.exp(log_variance)
        
        # 二次限制（冗余保护）
        aleatoric_unc = torch.clamp(aleatoric_unc, min=1e-6, max=10.0)
        
        return aleatoric_unc
    
    def forward(self, x: torch.Tensor, 
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的不确定性估计
        
        Args:
            x: (B, C, H, W, D) 输入特征
            training: 是否训练模式（影响是否使用MC Dropout）
        
        Returns:
            epistemic_unc: (B, 1, H, W, D) 认知不确定性
            aleatoric_unc: (B, 1, H, W, D) 偶然不确定性
            total_unc: (B, 1, H, W, D) 总不确定性
        """
        # 估计两类不确定性
        if training:
            # 训练时跳过MC Dropout（节省显存）
            # 使用零占位符（不影响梯度，因为不会用于loss计算）
            B, C, H, W, D = x.shape
            epistemic_unc = torch.zeros(B, 1, H, W, D, device=x.device, dtype=x.dtype)
        else:
            # 验证/推理时使用完整MC Dropout
            epistemic_unc = self.estimate_epistemic_uncertainty(x)
        
        # Aleatoric uncertainty仍然计算（用于训练正则化）
        aleatoric_unc = self.estimate_aleatoric_uncertainty(x)
        
        # 总不确定性
        if training:
            # 训练时只用aleatoric
            total_unc = aleatoric_unc
        else:
            # 验证时用完整公式
            total_unc = torch.sqrt(epistemic_unc ** 2 + aleatoric_unc ** 2)
        
        return epistemic_unc, aleatoric_unc, total_unc


class SpatialAdaptiveGating(nn.Module):
    """
    空间自适应门控模块
    
    根据不确定性图和先验权重，生成空间自适应的门控值
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 门控生成网络
        self.gate_network = nn.Sequential(
            # 输入：特征 + 不确定性（C+1个通道）
            nn.Conv3d(channels + 1, channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()  # 输出[0, 1]范围的门控值
        )
    
    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        生成空间自适应门控
        
        Args:
            x: (B, C, H, W, D) 输入特征
            uncertainty: (B, 1, H, W, D) 不确定性图
        
        Returns:
            gate: (B, 1, H, W, D) 空间门控图
        """
        # 拼接特征和不确定性
        combined = torch.cat([x, uncertainty], dim=1)  # (B, C+1, H, W, D)
        
        # 生成门控
        gate = self.gate_network(combined)  # (B, 1, H, W, D)
        
        return gate


class DSPAModule(nn.Module):
    """
    完整的DSPA模块
    
    工作流程：
    1. 加载双重平滑先验权重
    2. 估计特征的不确定性（认知+偶然）
    3. 不确定性感知的先验自适应加权
    4. 空间自适应门控融合
    5. 输出加权特征和不确定性图
    """
    
    def __init__(self, 
                 channels: int, 
                 spectral_dim: int = 60,
                 stage_id: int = 0,
                 spectral_weights_path: Optional[str] = None,
                 use_uncertainty: bool = True,
                 use_spatial_gating: bool = True,
                 dropout_rate: float = 0.2,
                 num_mc_samples: int = 5,
                 spatial_sigma: float = 1.5):
        """
        Args:
            channels: 特征通道数
            spectral_dim: 原始光谱维度（用于加载先验）
            stage_id: 当前stage的ID（用于日志）
            spectral_weights_path: 光谱先验权重文件路径
            use_uncertainty: 是否使用不确定性估计
            use_spatial_gating: 是否使用空间自适应门控
            dropout_rate: MC Dropout的dropout率
            num_mc_samples: MC采样次数
            spatial_sigma: 空间域平滑的sigma
        """
        super().__init__()
        self.channels = channels
        self.spectral_dim = spectral_dim
        self.stage_id = stage_id
        self.use_uncertainty = use_uncertainty
        self.use_spatial_gating = use_spatial_gating
        self.spatial_sigma = spatial_sigma
        
        # === 1. 加载并处理双重平滑先验权重 ===
        if spectral_weights_path and Path(spectral_weights_path).exists():
            # 加载原始权重
            raw_weights = np.load(spectral_weights_path)
            
            # 应用双重平滑
            smoothed_weights = DoublySmoothedPriorGenerator.apply_doubly_smoothing(
                raw_weights, spatial_sigma=spatial_sigma
            )
            
            # 投影到当前通道数
            projected_weights = self._project_weights(smoothed_weights, channels)
            
            print(f"  [DSPA Stage {stage_id}] 加载双重平滑先验权重")
            print(f"    原始维度: ({len(raw_weights)},) → 投影: ({len(projected_weights)},)")
            print(f"    空间平滑sigma: {spatial_sigma}")
        else:
            # 如果没有权重文件，使用均匀权重
            projected_weights = np.ones(channels)
            print(f"  [DSPA Stage {stage_id}] 使用均匀权重（未找到先验文件）")
        
        # 注册为buffer（不参与训练）
        self.register_buffer('prior_weights', torch.from_numpy(projected_weights).float())
        
        # === 2. 不确定性估计器 ===
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                channels=channels,
                dropout_rate=dropout_rate,
                num_mc_samples=num_mc_samples
            )
            print(f"    ✓ 不确定性估计: MC samples={num_mc_samples}, dropout={dropout_rate}")
        else:
            self.uncertainty_estimator = None
            print(f"    ✗ 不确定性估计: 禁用")
        
        # === 3. 空间自适应门控 ===
        if use_spatial_gating and use_uncertainty:
            self.spatial_gating = SpatialAdaptiveGating(channels)
            print(f"    ✓ 空间自适应门控")
        else:
            self.spatial_gating = None
            print(f"    ✗ 空间自适应门控")
        
        # === 4. 可学习的权重调节参数 ===
        # alpha: 控制不确定性对先验的影响程度（初始值设小，让训练更稳定）
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 1.0 → 0.1
        
        # beta: 控制门控的强度（初始值设小）
        if self.spatial_gating is not None:
            self.beta = nn.Parameter(torch.tensor(0.1))  # 0.5 → 0.1
        
        # === 5. Fallback特征变换（不确定性高时使用） ===
        self.fallback_transform = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        
        print(f"  [DSPA Stage {stage_id}] 初始化完成 (Channels={channels})")
    
    def _project_weights(self, weights: np.ndarray, target_dim: int) -> np.ndarray:
        """
        将权重从原始维度投影到目标维度
        
        Args:
            weights: (D_orig,) 原始权重
            target_dim: 目标维度
        
        Returns:
            projected: (target_dim,) 投影后的权重
        """
        original_dim = len(weights)
        
        if original_dim == target_dim:
            return weights
        
        # 使用线性插值
        x_old = np.linspace(0, original_dim - 1, original_dim)
        x_new = np.linspace(0, original_dim - 1, target_dim)
        projected = np.interp(x_new, x_old, weights)
        
        # 归一化
        projected = (projected - projected.min()) / (projected.max() - projected.min() + 1e-8)
        
        return projected
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            x: (B, C, H, W, D) skip connection特征
        
        Returns:
            weighted_x: (B, C, H, W, D) 加权后的特征
            info: dict 包含不确定性等中间信息
                - 'total_uncertainty': (B, 1, H, W, D)
                - 'epistemic_uncertainty': (B, 1, H, W, D)
                - 'aleatoric_uncertainty': (B, 1, H, W, D)
                - 'confidence': (B, 1, H, W, D)
                - 'gate': (B, 1, H, W, D) 如果使用门控
        """
        B, C, H, W, D = x.shape
        identity = x
        
        info = {}
        
        # === Step 1: 估计不确定性 ===
        if self.use_uncertainty and self.uncertainty_estimator is not None:
            epistemic_unc, aleatoric_unc, total_unc = self.uncertainty_estimator(
                x, training=self.training
            )
            
            # 更稳定的归一化方式（使用平均值和标准差）
            unc_mean = total_unc.mean()
            unc_std = total_unc.std() + 1e-8
            total_unc_norm = (total_unc - unc_mean) / unc_std
            
            # 使用sigmoid计算置信度（更平滑、更稳定）
            # sigmoid(-x) 将不确定性高→置信度低
            confidence = torch.sigmoid(-total_unc_norm)
            
            info['total_uncertainty'] = total_unc
            info['epistemic_uncertainty'] = epistemic_unc
            info['aleatoric_uncertainty'] = aleatoric_unc
            info['confidence'] = confidence
        else:
            # 如果不使用不确定性，假设全部确定（置信度=1）
            confidence = torch.ones(B, 1, H, W, D, device=x.device)
            info['total_uncertainty'] = torch.zeros_like(confidence)
            info['confidence'] = confidence
        
        # === Step 2: 不确定性感知的先验加权 ===
        # prior_weights: (C,) → (1, C, 1, 1, 1)
        prior_weights = self.prior_weights.view(1, -1, 1, 1, 1)
        
        # 调整方案：让先验权重占主导，不确定性影响变小
        # alpha初始=0.1，训练时会自动调整
        # 使用加法而非乘法，更稳定
        adaptive_weights = prior_weights * (1.0 + self.alpha * (confidence - 0.5))
        # confidence在0.5左右时，权重≈prior_weights
        # confidence高时(>0.5)，权重略增；低时(<0.5)，权重略减
        
        # 应用自适应权重
        weighted_x = x * adaptive_weights
        
        # === Step 3: 简化的Fallback机制 ===
        # 训练初期：让fallback影响更小，主要靠先验
        # 只在不确定性极高时才使用fallback（阈值0.3）
        if self.use_uncertainty:
            low_confidence_mask = (confidence < 0.3).float()
            if low_confidence_mask.sum() > 0:  # 只在有低置信区域时计算
                fallback_x = self.fallback_transform(x)
                # 只在低置信区域混合，其他区域保持weighted_x
                weighted_x = (1 - low_confidence_mask) * weighted_x + \
                            low_confidence_mask * (0.7 * weighted_x + 0.3 * fallback_x)
        
        # === Step 4: 空间自适应门控（简化版） ===
        if self.spatial_gating is not None:
            gate = self.spatial_gating(weighted_x, info['total_uncertainty'])
            info['gate'] = gate
            
            # 简化门控：让残差占主导（beta初始=0.1）
            # output = identity + beta * gate * (weighted_x - identity)
            # 相当于：大部分是原始特征，少部分是加权特征
            output = identity + self.beta * gate * (weighted_x - identity)
        else:
            # 简单残差连接（也让残差占主导）
            # 初期让DSPA影响小，避免破坏已学习的特征
            output = 0.9 * identity + 0.1 * weighted_x
        
        return output, info


class DSPAModuleLite(nn.Module):
    """
    DSPA的轻量级版本
    
    简化：
    1. 不使用MC Dropout（只用aleatoric uncertainty）
    2. 不使用空间门控
    3. MC采样次数减少
    
    适用于：快速实验、资源受限场景
    """
    
    def __init__(self, 
                 channels: int, 
                 spectral_dim: int = 60,
                 stage_id: int = 0,
                 spectral_weights_path: Optional[str] = None,
                 spatial_sigma: float = 1.5):
        super().__init__()
        self.channels = channels
        self.stage_id = stage_id
        
        # 加载双重平滑先验
        if spectral_weights_path and Path(spectral_weights_path).exists():
            raw_weights = np.load(spectral_weights_path)
            smoothed_weights = DoublySmoothedPriorGenerator.apply_doubly_smoothing(
                raw_weights, spatial_sigma=spatial_sigma
            )
            projected_weights = self._project_weights(smoothed_weights, channels)
        else:
            projected_weights = np.ones(channels)
        
        self.register_buffer('prior_weights', torch.from_numpy(projected_weights).float())
        
        # 轻量级不确定性估计（只用aleatoric）
        self.uncertainty_head = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=1),
        )
        
        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        print(f"  [DSPA-Lite Stage {stage_id}] Channels={channels}, 双重平滑先验")
    
    def _project_weights(self, weights: np.ndarray, target_dim: int) -> np.ndarray:
        """权重投影"""
        original_dim = len(weights)
        if original_dim == target_dim:
            return weights
        
        x_old = np.linspace(0, original_dim - 1, original_dim)
        x_new = np.linspace(0, original_dim - 1, target_dim)
        projected = np.interp(x_new, x_old, weights)
        projected = (projected - projected.min()) / (projected.max() - projected.min() + 1e-8)
        
        return projected
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        轻量级前向传播
        
        Args:
            x: (B, C, H, W, D)
        
        Returns:
            output: (B, C, H, W, D)
            info: dict
        """
        B, C, H, W, D = x.shape
        
        # 估计不确定性（只用aleatoric）
        log_variance = self.uncertainty_head(x)
        uncertainty = torch.exp(log_variance).clamp(min=1e-6, max=10.0)
        
        # 置信度
        uncertainty_norm = (uncertainty - uncertainty.min()) / \
                          (uncertainty.max() - uncertainty.min() + 1e-8)
        confidence = 1.0 / (1.0 + uncertainty_norm)
        
        # 自适应加权
        prior_weights = self.prior_weights.view(1, -1, 1, 1, 1)
        adaptive_weights = prior_weights * (confidence ** self.alpha)
        
        # 应用权重
        output = x * adaptive_weights
        
        info = {
            'total_uncertainty': uncertainty,
            'confidence': confidence
        }
        
        return output, info


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("测试DSPA模块")
    print("="*80)
    
    # 创建测试数据
    B, C, H, W, D = 2, 64, 32, 32, 15
    x = torch.randn(B, C, H, W, D)
    
    # 创建模拟的先验权重
    spectral_weights = np.random.rand(60)
    spectral_weights = (spectral_weights - spectral_weights.min()) / \
                      (spectral_weights.max() - spectral_weights.min())
    np.save('/tmp/test_doubly_smoothed_weights.npy', spectral_weights)
    
    print("\n1. 测试完整版DSPA:")
    print("-" * 80)
    dspa_full = DSPAModule(
        channels=C,
        spectral_dim=60,
        stage_id=0,
        spectral_weights_path='/tmp/test_doubly_smoothed_weights.npy',
        use_uncertainty=True,
        use_spatial_gating=True,
        num_mc_samples=3
    )
    
    print(f"\n输入形状: {x.shape}")
    output_full, info_full = dspa_full(x)
    print(f"输出形状: {output_full.shape}")
    print(f"不确定性图形状: {info_full['total_uncertainty'].shape}")
    print(f"置信度图形状: {info_full['confidence'].shape}")
    if 'gate' in info_full:
        print(f"门控图形状: {info_full['gate'].shape}")
    
    print("\n2. 测试轻量级DSPA:")
    print("-" * 80)
    dspa_lite = DSPAModuleLite(
        channels=C,
        spectral_dim=60,
        stage_id=1,
        spectral_weights_path='/tmp/test_doubly_smoothed_weights.npy'
    )
    
    output_lite, info_lite = dspa_lite(x)
    print(f"输出形状: {output_lite.shape}")
    print(f"不确定性图形状: {info_lite['total_uncertainty'].shape}")
    
    print("\n3. 参数统计:")
    print("-" * 80)
    full_params = sum(p.numel() for p in dspa_full.parameters() if p.requires_grad)
    lite_params = sum(p.numel() for p in dspa_lite.parameters() if p.requires_grad)
    print(f"完整版可训练参数: {full_params:,}")
    print(f"轻量版可训练参数: {lite_params:,}")
    print(f"参数减少: {(1 - lite_params/full_params)*100:.1f}%")
    
    print("\n✓ 测试完成！")
    print("="*80)

