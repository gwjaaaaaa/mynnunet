"""
HSI-UNet with SSFA-Plus
在PlainConvUNet基础上添加：
1. SSFA-Plus: Spectral-Spatial Fusion Attention (Skip Connections)
"""
from typing import Union, Type, List, Tuple
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


# ============================================================================
# SSFA-Plus模块 - 轻量级实现，确保不改变shape
# ============================================================================

class ChannelAttention(nn.Module):
    """通道注意力 - 不改变shape"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力 - 不改变shape"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class SSFAModule(nn.Module):
    """
    SSFA-Plus模块 - 轻量级实现
    输入输出shape完全相同
    """
    def __init__(self, channels, use_prior=True):
        super().__init__()
        self.channels = channels
        self.use_prior = use_prior
        
        # 通道和空间注意力
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
        # 光谱先验权重（可学习）
        if use_prior:
            # 尝试加载预计算的prior
            try:
                prior_path = '/data/CXY/g/szy/spectral_prior_weights.npy'
                if Path(prior_path).exists():
                    prior = np.load(prior_path)
                    # 缩放到[0.7, 1.3]避免过度抑制
                    prior = prior * 0.6 + 0.7
                else:
                    prior = np.ones(20)
                    
                self.spectral_prior = nn.Parameter(
                    torch.tensor(prior, dtype=torch.float32),
                    requires_grad=True
                )
            except:
                # 如果加载失败，使用均匀权重
                self.spectral_prior = nn.Parameter(
                    torch.ones(20, dtype=torch.float32),
                    requires_grad=True
                )
        else:
            self.register_buffer('spectral_prior', torch.ones(20))
    
    def forward(self, x):
        """
        x: (B, C, H, W, D)
        return: (B, C, H, W, D) 相同shape
        """
        identity = x
        
        # 1. 应用光谱先验（如果深度维度匹配）
        if x.size(-1) == self.spectral_prior.size(0):
            prior = self.spectral_prior.view(1, 1, 1, 1, -1)
            x = x * prior
        elif x.size(-1) > 1 and self.spectral_prior.size(0) > x.size(-1):
            # 如果depth变小了，对prior做平均池化
            indices = torch.linspace(0, self.spectral_prior.size(0) - 1, 
                                    x.size(-1), device=x.device).long()
            prior = self.spectral_prior[indices].view(1, 1, 1, 1, -1)
            x = x * prior
        
        # 2. 通道注意力
        ca = self.channel_att(x)
        x = x * ca
        
        # 3. 空间注意力  
        sa = self.spatial_att(x)
        x = x * sa
        
        # 4. Residual
        x = x + identity
        
        return x


class HSIUNet(AbstractDynamicNetworkArchitectures):
    """
    HSI-UNet with SSFA-Plus
    在PlainConvUNet基础上添加：
    1. SSFA-Plus on skip connections
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        """
        HSI-UNet with SSFA-Plus
        参数与PlainConvUNet完全相同
        """
        super().__init__()

        # 保持与PlainConvUNet相同的keys
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        
        self.n_stages = n_stages
        
        # 使用原始的PlainConvEncoder和UNetDecoder
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        
        self.decoder = UNetDecoder(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision, 
            nonlin_first=nonlin_first
        )
        
        # 为每个stage添加SSFA模块
        self.ssfa_modules = nn.ModuleList()
        for i in range(n_stages):
            # 只对前5个stage添加SSFA（后面的stage特征图太小）
            if i < 5:
                self.ssfa_modules.append(
                    SSFAModule(features_per_stage[i], use_prior=True)
                )
            else:
                self.ssfa_modules.append(nn.Identity())
        
        print(f"✓ HSI-UNet initialized:")
        print(f"  - SSFA-Plus: {sum(1 for m in self.ssfa_modules if not isinstance(m, nn.Identity))} stages")

    def forward(self, x):
        """
        前向传播：
        1. Encoder
        2. SSFA-Plus on skip connections
        3. Decoder
        """
        # 1. Encoder
        skips = self.encoder(x)
        
        # 2. 对每个skip应用SSFA-Plus（特征增强）
        enhanced_skips = []
        for i, skip in enumerate(skips):
            if i < len(self.ssfa_modules):
                enhanced_skip = self.ssfa_modules[i](skip)
                enhanced_skips.append(enhanced_skip)
            else:
                enhanced_skips.append(skip)
        
        # 3. Decoder
        return self.decoder(enhanced_skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + \
               self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == "__main__":
    """
    测试HSI-UNet with SSFA-Plus能否正常工作
    """
    print("="*70)
    print("Testing HSI-UNet with SSFA-Plus")
    print("="*70)
    
    # 使用Dataset518的实际配置
    model = HSIUNet(
        input_channels=1,
        n_stages=7,
        features_per_stage=[32, 64, 128, 256, 320, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3,3,3]]*7,
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,1], [2,2,1], [2,2,1], [2,2,1]],
        n_conv_per_stage=[2]*7,
        num_classes=2,
        n_conv_per_stage_decoder=[2]*6,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True
    )
    
    print(f"\n✓ Model created successfully")
    
    # 测试前向传播
    x = torch.rand(1, 1, 256, 256, 20)
    print(f"✓ Input shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Number of outputs (deep supervision): {len(outputs)}")
    print(f"  Output shape: {outputs[0].shape}")
    
    # 测试反向传播
    x_grad = torch.rand(1, 1, 256, 256, 20, requires_grad=True)
    outputs_grad = model(x_grad)
    loss = outputs_grad[0].sum()
    loss.backward()
    print(f"✓ Backward pass successful")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ssfa_params = sum(p.numel() for m in model.ssfa_modules for p in m.parameters())
    
    print(f"\n✓ Parameter Statistics:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  SSFA-Plus: {ssfa_params:,} ({ssfa_params/total_params*100:.2f}%)")
    print(f"  Baseline: {total_params - ssfa_params:,}")
    
    print("\n" + "="*70)
    print("✓ All tests passed! HSI-UNet with SSFA-Plus is ready!")
    print("="*70)

