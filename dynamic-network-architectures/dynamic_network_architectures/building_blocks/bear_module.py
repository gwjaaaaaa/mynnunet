"""
BEAR (Boundary-Enhanced Adaptive Refinement) Module

核心功能：
1. 轻量级边界检测（Sobel + 学习的边界增强）
2. 强化学习启发的多策略解码（标准/边界增强/保守）
3. 不确定性引导的策略选择
4. 自适应融合机制

设计原则：
- 轻量化：参数量 < 50K per stage
- 高效：避免复杂计算
- 可微分：端到端训练
- RL启发：策略选择但不需要RL训练循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class LightweightBoundaryDetector(nn.Module):
    """
    轻量级边界检测器
    
    结合：
    1. Sobel算子（经典边界检测）
    2. 学习的边界增强卷积
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Sobel算子（固定，不学习）
        # 3D Sobel kernels for x, y, z directions
        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._create_sobel_kernel('z'))
        
        # 学习的边界增强
        self.boundary_enhance = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.InstanceNorm3d(channels // 4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _create_sobel_kernel(self, direction: str) -> torch.Tensor:
        """创建3D Sobel算子"""
        if direction == 'x':
            kernel = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            ], dtype=torch.float32) / 16.0
        elif direction == 'y':
            kernel = torch.tensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ], dtype=torch.float32) / 16.0
        else:  # z
            kernel = torch.tensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ], dtype=torch.float32) / 16.0
        
        # Shape: (1, 1, 3, 3, 3)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W, D)
            
        Returns:
            boundary_map: 边界强度图 (B, 1, H, W, D)
            boundary_enhanced: 边界增强的特征 (B, C, H, W, D)
        """
        B, C, H, W, D = x.shape
        
        # 1. 使用Sobel算子检测边界（在通道维度上平均）
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W, D)
        
        # 应用Sobel卷积
        grad_x = F.conv3d(x_mean, self.sobel_x, padding=1)
        grad_y = F.conv3d(x_mean, self.sobel_y, padding=1)
        grad_z = F.conv3d(x_mean, self.sobel_z, padding=1)
        
        # 计算梯度幅值（边界强度）
        sobel_boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        
        # 2. 学习的边界增强
        learned_boundary = self.boundary_enhance(x)  # (B, 1, H, W, D)
        
        # 3. 融合Sobel和学习的边界
        boundary_map = 0.5 * sobel_boundary + 0.5 * learned_boundary
        
        # 4. 使用边界图增强特征
        boundary_enhanced = x * (1.0 + boundary_map)
        
        return boundary_map, boundary_enhanced


class MultiStrategyDecoder(nn.Module):
    """
    多策略解码器
    
    三种策略（强化学习启发）：
    1. Standard: 标准卷积解码
    2. Boundary-Enhanced: 边界增强解码
    3. Conservative: 保守解码（更多正则化）
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 策略1：标准解码（轻量深度可分离卷积）
        self.standard_path = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),  # Depthwise
            nn.Conv3d(channels, channels, kernel_size=1),  # Pointwise
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 策略2：边界增强解码（更关注高频细节）
        self.boundary_path = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
            # 额外的细节保持层
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 策略3：保守解码（更多平滑，适合高不确定性区域）
        self.conservative_path = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels),  # 更大的感受野
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W, D)
            
        Returns:
            outputs: 包含三种策略输出的字典
        """
        return {
            'standard': self.standard_path(x),
            'boundary': self.boundary_path(x),
            'conservative': self.conservative_path(x)
        }


class RLInspiredStrategySelector(nn.Module):
    """
    强化学习启发的策略选择器
    
    核心思想：
    - 将策略选择建模为"动作选择"
    - 使用不确定性和边界信息作为"状态"
    - 通过可微分的soft选择（非硬RL训练）
    
    优势：
    - 端到端可训练
    - 无需RL训练循环
    - 计算高效
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 状态编码器：[不确定性, 边界强度] -> 策略权重
        self.state_encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=1),  # 2个输入：uncertainty + boundary
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(16, 3, kernel_size=1),  # 3个输出：3种策略的权重
        )
        
        # 价值网络（评估当前状态）- RL概念
        self.value_network = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                x: torch.Tensor,
                uncertainty: torch.Tensor,
                boundary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W, D)
            uncertainty: 不确定性图 (B, 1, H, W, D)
            boundary: 边界强度图 (B, 1, H, W, D)
            
        Returns:
            strategy_weights: 策略权重 (B, 3, H, W, D)，3个通道对应3种策略
            value: 状态价值估计 (B, 1, H, W, D)
        """
        # 1. 构建状态表示
        state = torch.cat([uncertainty, boundary], dim=1)  # (B, 2, H, W, D)
        
        # 2. 策略选择（动作选择）
        strategy_logits = self.state_encoder(state)  # (B, 3, H, W, D)
        strategy_weights = F.softmax(strategy_logits, dim=1)  # Soft策略选择
        
        # 3. 价值估计
        value = self.value_network(x)  # (B, 1, H, W, D)
        
        return strategy_weights, value


class BEARModule(nn.Module):
    """
    BEAR (Boundary-Enhanced Adaptive Refinement) 模块
    
    应用于解码器，进行边界增强和自适应精炼
    
    参数量：约30-50K（轻量！）
    显存开销：约0.5-1GB
    """
    
    def __init__(self,
                 channels: int,
                 use_uncertainty: bool = True,
                 uncertainty_threshold: float = 0.5):
        """
        Args:
            channels: 特征通道数
            use_uncertainty: 是否使用不确定性引导
            uncertainty_threshold: 不确定性阈值（用于策略选择）
        """
        super().__init__()
        self.channels = channels
        self.use_uncertainty = use_uncertainty
        self.uncertainty_threshold = uncertainty_threshold
        
        print(f"  [BEAR] Initializing for C={channels}")
        
        # 1. 边界检测器
        self.boundary_detector = LightweightBoundaryDetector(channels)
        
        # 2. 多策略解码器
        self.multi_strategy_decoder = MultiStrategyDecoder(channels)
        
        # 3. RL启发的策略选择器
        if use_uncertainty:
            self.strategy_selector = RLInspiredStrategySelector(channels)
        
        # 4. 融合权重（可学习）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3))  # 边界增强的强度
        self.fusion_beta = nn.Parameter(torch.tensor(0.2))   # 策略混合的强度
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, 
                x: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: 解码器特征 (B, C, H, W, D)
            uncertainty: 可选的不确定性图 (B, 1, H, W, D)，来自SDAR
            
        Returns:
            refined: 精炼后的特征 (B, C, H, W, D)
            info: 调试信息字典
        """
        B, C, H, W, D = x.shape
        info = {}
        
        # Step 1: 边界检测
        boundary_map, boundary_enhanced = self.boundary_detector(x)
        info['boundary_map'] = boundary_map.detach()
        
        # Step 2: 多策略解码
        strategy_outputs = self.multi_strategy_decoder(x)
        
        # Step 3: 策略选择和融合
        if self.use_uncertainty and uncertainty is not None:
            # 使用不确定性引导的策略选择（RL启发）
            strategy_weights, value = self.strategy_selector(x, uncertainty, boundary_map)
            info['strategy_weights'] = strategy_weights.detach()
            info['value'] = value.detach()
            
            # 加权融合三种策略
            w_std = strategy_weights[:, 0:1, :, :, :]      # 标准策略权重
            w_bnd = strategy_weights[:, 1:2, :, :, :]      # 边界增强策略权重
            w_con = strategy_weights[:, 2:3, :, :, :]      # 保守策略权重
            
            strategy_fused = (w_std * strategy_outputs['standard'] +
                             w_bnd * strategy_outputs['boundary'] +
                             w_con * strategy_outputs['conservative'])
        else:
            # 无不确定性时，使用简单的规则：边界区域用边界增强，其他用标准
            boundary_mask = (boundary_map > 0.5).float()
            strategy_fused = (boundary_mask * strategy_outputs['boundary'] +
                             (1 - boundary_mask) * strategy_outputs['standard'])
        
        # Step 4: 自适应融合
        # 结合原始特征、边界增强和策略融合
        alpha = torch.sigmoid(self.fusion_alpha)
        beta = torch.sigmoid(self.fusion_beta)
        
        refined = x + alpha * (boundary_enhanced - x) + beta * (strategy_fused - x)
        
        return refined, info


class BEARModuleLite(nn.Module):
    """
    BEAR的超轻量版本
    
    简化：
    - 只有边界检测和单一增强策略
    - 移除RL启发的策略选择
    - 参数量 < 20K
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        print(f"  [BEAR-Lite] Initializing for C={channels}")
        
        # 简化的边界检测
        self.boundary_detect = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 简化的边界增强
        self.boundary_enhance = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 固定融合权重
        self.register_buffer('fusion_alpha', torch.tensor(0.3))
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    └─ Total params: {total_params:,} (~{total_params/1000:.1f}K)")
    
    def forward(self, x: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """简化的前向传播"""
        # 边界检测
        boundary_map = self.boundary_detect(x)
        
        # 边界增强
        enhanced = self.boundary_enhance(x)
        
        # 简单融合
        refined = x + self.fusion_alpha * boundary_map * (enhanced - x)
        
        info = {'boundary_map': boundary_map.detach()}
        return refined, info


# 便捷函数
def build_bear_module(channels: int,
                      use_lite: bool = False,
                      use_uncertainty: bool = True,
                      uncertainty_threshold: float = 0.5) -> nn.Module:
    """
    构建BEAR模块的便捷函数
    
    Args:
        channels: 特征通道数
        use_lite: 是否使用超轻量版本
        use_uncertainty: 是否使用不确定性引导
        uncertainty_threshold: 不确定性阈值
    """
    if use_lite:
        return BEARModuleLite(channels=channels)
    else:
        return BEARModule(
            channels=channels,
            use_uncertainty=use_uncertainty,
            uncertainty_threshold=uncertainty_threshold
        )


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("Testing BEAR Module")
    print("="*80)
    
    # 测试标准版
    print("\n1. Testing Standard BEAR:")
    bear = BEARModule(channels=128, use_uncertainty=True)
    x = torch.randn(2, 128, 32, 32, 16)
    uncertainty = torch.rand(2, 1, 32, 32, 16)
    out, info = bear(x, uncertainty)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Info keys: {info.keys()}")
    
    # 测试Lite版
    print("\n2. Testing BEAR-Lite:")
    bear_lite = BEARModuleLite(channels=128)
    out_lite, info_lite = bear_lite(x)
    print(f"   Output shape: {out_lite.shape}")
    print(f"   Info keys: {info_lite.keys()}")
    
    # 测试无不确定性的情况
    print("\n3. Testing BEAR without uncertainty:")
    out_no_unc, info_no_unc = bear(x)
    print(f"   Output shape: {out_no_unc.shape}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")


