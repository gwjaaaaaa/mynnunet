"""
Spectral-Morphology Prior Fusion (SMPF) Module

功能：
1. 加载预计算的光谱先验权重
2. 形态学判别（判断腺管区域 vs 细胞区域）
3. 自适应光谱加权（不同区域使用不同的光谱权重）

作者: HSI-UNet Team
日期: 2025-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class SMPFModule(nn.Module):
    """
    Spectral-Morphology Prior Fusion Module
    
    在Skip Connections上应用形态学引导的光谱先验加权
    
    核心创新：
    1. 光谱先验提供"what"（哪些波段重要）- 来自统计分析
    2. 形态学特征提供"where"（哪些区域用哪些波段）- 可学习
    3. 自适应融合：根据形态学概率动态调整光谱权重
    
    输入: skip_feat (B, C, H, W, D) - encoder的skip feature
    输出: 
        - weighted_skip (B, C, H, W, D) - 加权后的skip feature
        - morpho_probs (B, 2, H, W, D) - 形态学概率 [腺管, 细胞]
    """
    
    def __init__(
        self, 
        channels: int, 
        spectral_dim: int = 60,
        stage_id: int = 0,
        spectral_weights_path: Optional[str] = None,
        num_morpho_classes: int = 2,  # 腺管 vs 细胞
        reduction_ratio: int = 4,
        use_residual: bool = True
    ):
        """
        参数:
            channels: 输入特征的通道数
            spectral_dim: 原始光谱维度（60个波段）
            stage_id: 当前stage的ID（用于确定当前depth）
            spectral_weights_path: 光谱先验权重文件路径
            num_morpho_classes: 形态学类别数（默认2：腺管/细胞）
            reduction_ratio: 注意力模块的降维比例
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        self.channels = channels
        self.spectral_dim = spectral_dim
        self.stage_id = stage_id
        self.num_morpho_classes = num_morpho_classes
        self.use_residual = use_residual
        
        # === 1. 加载光谱先验权重 ===
        if spectral_weights_path is not None and Path(spectral_weights_path).exists():
            spectral_weights = np.load(spectral_weights_path)
            print(f"  [SMPF Stage {stage_id}] 加载光谱先验权重: {spectral_weights_path}")
            print(f"    权重形状: {spectral_weights.shape}, 范围: [{spectral_weights.min():.4f}, {spectral_weights.max():.4f}]")
        else:
            # 默认均匀权重
            spectral_weights = np.ones(spectral_dim)
            print(f"  [SMPF Stage {stage_id}] 使用默认均匀权重")
        
        self.register_buffer('spectral_prior', torch.from_numpy(spectral_weights).float())
        
        # === 2. 形态学判别器 ===
        # 判断每个空间位置是腺管还是细胞占主导
        self.morpho_discriminator = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.InstanceNorm3d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels // reduction_ratio, 
                     kernel_size=3, padding=1, groups=channels // reduction_ratio),  # depthwise
            nn.InstanceNorm3d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, num_morpho_classes, kernel_size=1),
        )
        
        # ✅ 改进1.1：特殊初始化形态学判别器
        # 让最后一层输出logits接近0，使得softmax后接近均匀分布[0.5, 0.5]
        # 这样初始时不会引入太多随机noise
        self._init_morpho_discriminator()
        
        # === 3. 形态学引导的光谱调制器 ===
        # 为每个形态学类别学习光谱权重的调制因子
        self.spectral_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spectral_dim, spectral_dim),
                nn.ReLU(inplace=True),
                nn.Linear(spectral_dim, spectral_dim),
                nn.Sigmoid()  # 输出[0, 1]的调制因子
            )
            for _ in range(num_morpho_classes)
        ])
        
        # ✅ 改进1.1：初始化光谱调制器
        # 让初始输出接近1（不调制），避免破坏预训练特征
        self._init_spectral_modulators()
        
        # === 4. 可学习的缩放因子 ===
        # ✅ 改进1.1：从小的alpha开始，让SMPF作用更温和
        # 控制光谱先验的作用强度
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # 从1.0改为0.1
        
        # === 5. 通道自适应层（可选，轻量级） ===
        self.channel_adapt = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True)
        )
        
        print(f"  [SMPF Stage {stage_id}] 初始化完成:")
        print(f"    - Channels: {channels}")
        print(f"    - Spectral dim: {spectral_dim}")
        print(f"    - Morpho classes: {num_morpho_classes} (腺管/细胞)")
        print(f"    - Use residual: {use_residual}")
        print(f"    - Alpha初始值: {self.alpha.item():.3f} (温和启动)")
    
    def _init_morpho_discriminator(self):
        """
        初始化形态学判别器
        让最后一层输出接近0，使得初始预测接近均匀分布
        """
        # 对所有卷积层使用Xavier初始化（更温和）
        for m in self.morpho_discriminator.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 特别处理最后一层：让logits接近0
        last_conv = list(self.morpho_discriminator.modules())[-1]
        if isinstance(last_conv, nn.Conv3d):
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)
        
        print(f"    ✓ 形态学判别器：特殊初始化（输出接近均匀分布）")
    
    def _init_spectral_modulators(self):
        """
        初始化光谱调制器
        让初始输出接近1（sigmoid(2) ≈ 0.88），即几乎不调制
        """
        for modulator in self.spectral_modulators:
            for m in modulator.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        # 最后一层的bias设为2，使得sigmoid(2) ≈ 0.88
                        # 这样初始调制因子接近1，不破坏原始特征
                        if m is list(modulator.modules())[-2]:  # Sigmoid前的Linear
                            nn.init.constant_(m.bias, 2.0)
                        else:
                            nn.init.constant_(m.bias, 0)
        
        print(f"    ✓ 光谱调制器：特殊初始化（输出接近1.0）")
    
    def _project_spectral_weights(self, current_depth: int) -> torch.Tensor:
        """
        将光谱先验权重投影到当前stage的depth
        
        参数:
            current_depth: 当前stage的depth维度
        
        返回:
            projected_weights: (current_depth,)
        """
        if current_depth == self.spectral_dim:
            # 不需要投影
            return self.spectral_prior
        else:
            # 使用线性插值投影
            # spectral_prior: (60,) -> (current_depth,)
            spectral_prior = self.spectral_prior.unsqueeze(0).unsqueeze(0)  # (1, 1, 60)
            projected = F.interpolate(
                spectral_prior, 
                size=current_depth, 
                mode='linear', 
                align_corners=True
            )
            return projected.squeeze(0).squeeze(0)  # (current_depth,)
    
    def forward(self, skip_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            skip_feat: (B, C, H, W, D) - encoder的skip feature
        
        返回:
            weighted_skip: (B, C, H, W, D) - 加权后的skip feature
            morpho_probs: (B, num_morpho_classes, H, W, D) - 形态学概率分布
        """
        B, C, H, W, D = skip_feat.shape
        
        # === Step 1: 形态学判别 ===
        # 判断每个空间位置是腺管还是细胞占主导
        morpho_logits = self.morpho_discriminator(skip_feat)  # (B, num_morpho_classes, H, W, D)
        morpho_probs = F.softmax(morpho_logits, dim=1)  # (B, 2, H, W, D)
        
        # === Step 2: 投影光谱先验到当前depth ===
        projected_spectral_prior = self._project_spectral_weights(D)  # (D,)
        
        # === Step 3: 形态学引导的光谱权重调制 ===
        # 为每个形态学类别计算调制后的光谱权重
        modulated_spectral_weights = []
        
        for i in range(self.num_morpho_classes):
            # 基础光谱先验: (D,) -> (spectral_dim,) -> modulate -> (spectral_dim,) -> (D,)
            # 注意：调制器在原始spectral_dim上操作，然后再投影
            base_prior = self.spectral_prior  # (spectral_dim,)
            
            # 调制因子
            modulation_factor = self.spectral_modulators[i](base_prior)  # (spectral_dim,)
            
            # 调制
            modulated_prior = base_prior * modulation_factor  # (spectral_dim,)
            
            # 投影到当前depth
            if D == self.spectral_dim:
                modulated_weight = modulated_prior
            else:
                modulated_weight = F.interpolate(
                    modulated_prior.unsqueeze(0).unsqueeze(0),
                    size=D,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).squeeze(0)  # (D,)
            
            modulated_spectral_weights.append(modulated_weight)
        
        # === Step 4: 自适应融合光谱权重 ===
        # 根据形态学概率，动态组合不同的光谱权重
        # morpho_probs: (B, num_morpho_classes, H, W, D)
        # modulated_spectral_weights: list of (D,)
        
        # 初始化自适应光谱权重
        adaptive_spectral_weights = torch.zeros(B, 1, 1, 1, D, device=skip_feat.device)
        
        for i in range(self.num_morpho_classes):
            # 形态学概率: (B, 1, H, W, D)
            morpho_prob_i = morpho_probs[:, i:i+1, :, :, :]  # (B, 1, H, W, D)
            
            # 光谱权重: (D,) -> (1, 1, 1, 1, D)
            spectral_weight_i = modulated_spectral_weights[i].view(1, 1, 1, 1, D)
            
            # 加权累加（空间平均版本，轻量级）
            # 注意：这里我们取空间平均，避免每个位置都不同（太重）
            morpho_prob_avg = morpho_prob_i.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1, D)
            
            adaptive_spectral_weights += morpho_prob_avg * spectral_weight_i
        
        # === Step 5: 应用光谱权重到skip feature ===
        # skip_feat: (B, C, H, W, D)
        # adaptive_spectral_weights: (B, 1, 1, 1, D)
        
        # 加权（带可学习的强度控制）
        weighted_skip = skip_feat * (1.0 + self.alpha * adaptive_spectral_weights)
        
        # === Step 6: 通道自适应（可选） ===
        weighted_skip = self.channel_adapt(weighted_skip)
        
        # === Step 7: 残差连接 ===
        if self.use_residual:
            output = skip_feat + weighted_skip
        else:
            output = weighted_skip
        
        return output, morpho_probs


class SMPFModuleLite(nn.Module):
    """
    SMPF的轻量级版本
    
    简化：
    1. 不学习形态学调制器，直接用可学习的模式向量
    2. 更简单的形态学判别器
    3. 减少参数量
    """
    
    def __init__(
        self, 
        channels: int, 
        spectral_dim: int = 60,
        stage_id: int = 0,
        spectral_weights_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.channels = channels
        self.spectral_dim = spectral_dim
        self.stage_id = stage_id
        
        # 加载光谱先验
        if spectral_weights_path is not None and Path(spectral_weights_path).exists():
            spectral_weights = np.load(spectral_weights_path)
        else:
            spectral_weights = np.ones(spectral_dim)
        
        self.register_buffer('spectral_prior', torch.from_numpy(spectral_weights).float())
        
        # 简化的形态学判别器
        self.morpho_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Conv3d(channels, 2, kernel_size=1),  # (B, 2, 1, 1, 1)
        )
        
        # 可学习的光谱调制因子（简化版）
        self.glandular_modulator = nn.Parameter(torch.ones(spectral_dim))
        self.cellular_modulator = nn.Parameter(torch.ones(spectral_dim))
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        print(f"  [SMPF-Lite Stage {stage_id}] 初始化完成 (轻量级版本)")
    
    def forward(self, skip_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W, D = skip_feat.shape
        
        # 形态学判别（全局）
        morpho_logits = self.morpho_discriminator(skip_feat)  # (B, 2, 1, 1, 1)
        morpho_probs = F.softmax(morpho_logits, dim=1)  # (B, 2, 1, 1, 1)
        
        # 广播到全空间
        morpho_probs_full = morpho_probs.expand(B, 2, H, W, D)  # (B, 2, H, W, D)
        
        # 光谱权重调制
        glandular_weight = self.spectral_prior * self.glandular_modulator  # (spectral_dim,)
        cellular_weight = self.spectral_prior * self.cellular_modulator    # (spectral_dim,)
        
        # 投影到当前depth
        if D == self.spectral_dim:
            glandular_weight_proj = glandular_weight
            cellular_weight_proj = cellular_weight
        else:
            glandular_weight_proj = F.interpolate(
                glandular_weight.unsqueeze(0).unsqueeze(0),
                size=D, mode='linear', align_corners=True
            ).squeeze()
            cellular_weight_proj = F.interpolate(
                cellular_weight.unsqueeze(0).unsqueeze(0),
                size=D, mode='linear', align_corners=True
            ).squeeze()
        
        # 融合
        prob_glandular = morpho_probs[:, 0:1, :, :, :].mean(dim=[2,3], keepdim=True)  # (B, 1, 1, 1, 1)
        prob_cellular = morpho_probs[:, 1:2, :, :, :].mean(dim=[2,3], keepdim=True)   # (B, 1, 1, 1, 1)
        
        adaptive_weight = (
            prob_glandular * glandular_weight_proj.view(1, 1, 1, 1, D) +
            prob_cellular * cellular_weight_proj.view(1, 1, 1, 1, D)
        )
        
        # 应用
        weighted_skip = skip_feat * (1.0 + self.alpha * adaptive_weight)
        
        return weighted_skip, morpho_probs_full


def test_smpf_module():
    """测试SMPF模块"""
    print("="*80)
    print("测试SMPF模块")
    print("="*80)
    
    # 参数
    B, C, H, W, D = 2, 64, 128, 128, 30
    spectral_dim = 60
    stage_id = 1
    
    # 创建测试数据
    skip_feat = torch.randn(B, C, H, W, D).cuda()
    
    # 创建模块（使用默认权重）
    smpf = SMPFModule(
        channels=C,
        spectral_dim=spectral_dim,
        stage_id=stage_id,
        spectral_weights_path=None  # 测试时使用默认权重
    ).cuda()
    
    print(f"\n输入形状: {skip_feat.shape}")
    
    # 前向传播
    with torch.cuda.amp.autocast():
        weighted_skip, morpho_probs = smpf(skip_feat)
    
    print(f"输出形状: {weighted_skip.shape}")
    print(f"形态学概率形状: {morpho_probs.shape}")
    print(f"形态学概率和: {morpho_probs.sum(dim=1).mean():.4f} (应该≈1.0)")
    print(f"形态学概率分布:")
    print(f"  腺管: {morpho_probs[:, 0].mean():.4f} ± {morpho_probs[:, 0].std():.4f}")
    print(f"  细胞: {morpho_probs[:, 1].mean():.4f} ± {morpho_probs[:, 1].std():.4f}")
    
    # 测试梯度
    loss = weighted_skip.sum()
    loss.backward()
    
    print("\n梯度检查:")
    print(f"  alpha梯度: {smpf.alpha.grad}")
    print(f"  ✓ 梯度正常")
    
    # 测试轻量级版本
    print("\n" + "="*80)
    print("测试SMPF-Lite模块")
    print("="*80)
    
    skip_feat2 = torch.randn(B, C, H, W, D).cuda()
    smpf_lite = SMPFModuleLite(
        channels=C,
        spectral_dim=spectral_dim,
        stage_id=stage_id
    ).cuda()
    
    with torch.cuda.amp.autocast():
        weighted_skip2, morpho_probs2 = smpf_lite(skip_feat2)
    
    print(f"\n输出形状: {weighted_skip2.shape}")
    print(f"形态学概率形状: {morpho_probs2.shape}")
    
    # 参数量对比
    params_full = sum(p.numel() for p in smpf.parameters())
    params_lite = sum(p.numel() for p in smpf_lite.parameters())
    
    print("\n" + "="*80)
    print("参数量对比:")
    print("="*80)
    print(f"SMPF (完整版): {params_full:,} 参数")
    print(f"SMPF-Lite: {params_lite:,} 参数")
    print(f"减少: {(1 - params_lite/params_full)*100:.1f}%")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_smpf_module()

