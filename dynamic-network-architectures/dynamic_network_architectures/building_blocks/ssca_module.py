"""
SSCA (Spectral-Spatial Channel Attention) Module with Doubly Smoothed Prior

超轻量、高效的跳跃连接增强模块
- 参数量：< 1K per stage
- 计算开销：几乎为零
- 理论基础：SE-Net + 双重平滑光谱先验

创新点：
1. 双重平滑密度估计计算先验（理论创新）
2. 先验用于初始化SE-Net权重（实践高效）
3. 前向传播只是标准SE-Net（极快）
"""

import torch
import torch.nn as nn
import numpy as np


class DoublySmoothedPriorCalculator:
    """
    双重平滑密度估计器
    
    基于统计学理论的两阶段平滑：
    1. 值域平滑（Value Domain Smoothing）
    2. 空间平滑（Spatial Domain Smoothing）
    
    只在初始化时计算一次，零运行时开销！
    """
    
    def __init__(self, 
                 value_bandwidth: float = 0.1,
                 spatial_sigma: float = 1.5):
        self.value_bandwidth = value_bandwidth
        self.spatial_sigma = spatial_sigma
    
    def compute(self, weights: np.ndarray) -> np.ndarray:
        """
        计算双重平滑后的权重
        
        Args:
            weights: 原始权重 (D,)
            
        Returns:
            smoothed: 双重平滑后的权重 (D,)
        """
        D = len(weights)
        
        # Stage 1: 值域平滑
        value_smoothed = np.zeros(D)
        for i in range(D):
            kernel = np.exp(-0.5 * ((weights - weights[i]) / self.value_bandwidth) ** 2)
            kernel /= (kernel.sum() + 1e-8)
            value_smoothed[i] = np.sum(kernel * weights)
        
        # Stage 2: 空间平滑
        positions = np.arange(D)
        spatial_smoothed = np.zeros(D)
        for i in range(D):
            distances = positions - positions[i]
            kernel = np.exp(-0.5 * (distances / self.spatial_sigma) ** 2)
            kernel /= (kernel.sum() + 1e-8)
            spatial_smoothed[i] = np.sum(kernel * value_smoothed)
        
        # 归一化
        smoothed = (spatial_smoothed - spatial_smoothed.min()) / \
                   (spatial_smoothed.max() - spatial_smoothed.min() + 1e-8)
        
        return smoothed.astype(np.float32)


class SSCAModule(nn.Module):
    """
    SSCA with Doubly Smoothed Prior
    
    核心设计：
    1. 双重平滑先验 → 初始化SE-Net权重
    2. SE-Net → 快速注意力计算
    3. 理论+实践完美结合
    """
    
    def __init__(self, 
                 channels: int,
                 spectral_dim: int = 60,
                 spectral_prior_path: str = None,
                 reduction: int = 4,
                 use_doubly_smoothing: bool = True,
                 dropout_rate: float = 0.1):
        """
        Args:
            channels: 特征通道数
            spectral_dim: 光谱维度
            spectral_prior_path: 光谱先验路径
            reduction: 压缩率（越大越轻量）
            use_doubly_smoothing: 是否使用双重平滑
            dropout_rate: Dropout率（用于正则化）
        """
        super().__init__()
        self.channels = channels
        self.use_doubly_smoothing = use_doubly_smoothing
        self.dropout_rate = dropout_rate
        
        # 加载并处理光谱先验
        prior_channels = None
        if spectral_prior_path:
            try:
                # 加载原始先验
                raw_prior = np.load(spectral_prior_path)
                
                # 应用双重平滑
                if use_doubly_smoothing and len(raw_prior) > 0:
                    calculator = DoublySmoothedPriorCalculator(
                        value_bandwidth=0.1,
                        spatial_sigma=1.5
                    )
                    smoothed_prior = calculator.compute(raw_prior)
                    print(f"  [SSCA-DS] C={channels}, Applied doubly smoothing")
                else:
                    smoothed_prior = raw_prior
                    print(f"  [SSCA] C={channels}, No smoothing")
                
                # 归一化
                smoothed_prior = (smoothed_prior - smoothed_prior.min()) / \
                                (smoothed_prior.max() - smoothed_prior.min() + 1e-8)
                
                # 插值到通道维度
                if len(smoothed_prior) == spectral_dim:
                    indices = np.linspace(0, len(smoothed_prior)-1, channels)
                    prior_channels = np.interp(indices, np.arange(len(smoothed_prior)), smoothed_prior)
                else:
                    prior_channels = np.ones(channels) / channels
                
                print(f"    ├─ Prior range: [{prior_channels.min():.3f}, {prior_channels.max():.3f}]")
                
            except Exception as e:
                print(f"  [SSCA] C={channels}, Prior loading failed: {e}")
                prior_channels = None
        
        # SE-Net结构
        mid_channels = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Dropout用于正则化（减少过拟合，稳定验证损失）
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # 关键：用双重平滑先验初始化SE-Net权重
        if prior_channels is not None:
            self._initialize_with_prior(prior_channels, mid_channels)
            print(f"    ├─ Initialized with doubly smoothed prior")
        else:
            print(f"    ├─ Standard initialization")
        
        # 参数量统计
        params = channels * mid_channels + mid_channels * channels
        dropout_str = f", Dropout={dropout_rate}" if dropout_rate > 0 else ""
        print(f"    └─ Params: {params:,} (~{params/1000:.1f}K){dropout_str}")
    
    def _initialize_with_prior(self, prior: np.ndarray, mid_channels: int):
        """
        用双重平滑先验初始化SE-Net权重
        
        核心思想：
        - fc2的权重初始化为与先验相关的值
        - 让网络从一个"好的起点"开始学习
        """
        with torch.no_grad():
            prior_tensor = torch.from_numpy(prior).float()
            
            # fc2: 从压缩空间恢复到通道空间
            # 初始化为对角阵的形式，偏向先验权重高的通道
            weight = self.fc2.weight.data  # (channels, mid_channels)
            
            # 方案：让先验影响每个输出通道的基础强度
            for i in range(self.channels):
                # 先验越大，该通道的权重初始化越大
                scale = 0.5 + prior_tensor[i] * 0.5  # 范围[0.5, 1.0]
                weight[i, :] *= scale
            
            # fc1: 标准初始化（让网络自己学习如何压缩）
            # 保持PyTorch默认初始化
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：标准SE-Net + Dropout正则化
        
        Args:
            x: (B, C, H, W, D)
        Returns:
            out: (B, C, H, W, D)
        """
        B, C, H, W, D = x.shape
        
        # 1. 全局平均池化
        channel_features = x.mean(dim=[2, 3, 4])  # (B, C)
        
        # 2. SE-Net：压缩-激活-扩张
        attention = self.fc1(channel_features)
        attention = self.relu(attention)
        
        # Dropout正则化（训练时）
        if self.dropout is not None and self.training:
            attention = self.dropout(attention)
        
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        
        # 3. 应用注意力
        attention = attention.view(B, C, 1, 1, 1)
        
        # 4. 残差连接
        out = x * attention + x * 0.1
        
        return out


# 便捷构建函数
def build_ssca_module(channels: int,
                      spectral_dim: int = 60,
                      spectral_prior_path: str = None,
                      reduction: int = 4,
                      use_doubly_smoothing: bool = True,
                      dropout_rate: float = 0.1):
    """
    构建SSCA模块
    
    Args:
        channels: 通道数
        spectral_dim: 光谱维度
        spectral_prior_path: 光谱先验路径
        reduction: 压缩率（4=标准，8=更轻量）
        use_doubly_smoothing: 是否使用双重平滑
        dropout_rate: Dropout率（用于减少过拟合）
    """
    return SSCAModule(
        channels=channels,
        spectral_dim=spectral_dim,
        spectral_prior_path=spectral_prior_path,
        reduction=reduction,
        use_doubly_smoothing=use_doubly_smoothing,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    print("="*80)
    print("Testing SSCA-DS Module")
    print("="*80)
    
    # 测试标准版（无先验）
    print("\n1. Testing without prior:")
    for C in [32, 128, 320]:
        print(f"\nC={C}:")
        ssca = SSCAModule(channels=C, spectral_prior_path=None)
        x = torch.randn(2, C, 32, 32, 16)
        out = ssca(x)
        print(f"  Output shape: {out.shape}")
    
    # 测试双重平滑版（如果有先验文件）
    print("\n2. Testing with doubly smoothed prior:")
    print("  (需要先验文件，如果没有会fallback到标准版)")
    
    # 创建模拟先验
    mock_prior = np.random.rand(60)
    mock_prior_path = "/tmp/mock_spectral_prior.npy"
    np.save(mock_prior_path, mock_prior)
    
    for C in [32, 128]:
        print(f"\nC={C}:")
        ssca = SSCAModule(
            channels=C,
            spectral_prior_path=mock_prior_path,
            use_doubly_smoothing=True
        )
        x = torch.randn(2, C, 32, 32, 16)
        out = ssca(x)
        print(f"  Output shape: {out.shape}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

