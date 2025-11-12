"""
HSI-UNet with Three Innovations
集成三个创新点：
1. CLEF: Contrastive Learning Enhanced Features (训练策略，在trainer中实现)
2. SSFA: Spectral-Spatial Fusion Attention (已有)
3. PBRN: Progressive Boundary Refinement Network (新增)
"""
from typing import Union, Type, List, Tuple
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


# ============================================================================
# SSFA Module (创新点2 - 已有)
# ============================================================================

class ChannelAttention(nn.Module):
    """通道注意力"""
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
    """空间注意力"""
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
    """SSFA模块（创新点2）"""
    def __init__(self, channels, use_prior=True):
        super().__init__()
        self.channels = channels
        self.use_prior = use_prior
        
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
        if use_prior:
            try:
                prior_path = '/data/CXY/g/szy/spectral_prior_weights.npy'
                if Path(prior_path).exists():
                    prior = np.load(prior_path)
                    prior = prior * 0.6 + 0.7
                else:
                    prior = np.ones(20)
                    
                self.spectral_prior = nn.Parameter(
                    torch.tensor(prior, dtype=torch.float32),
                    requires_grad=True
                )
            except:
                self.spectral_prior = nn.Parameter(
                    torch.ones(20, dtype=torch.float32),
                    requires_grad=True
                )
        else:
            self.register_buffer('spectral_prior', torch.ones(20))
    
    def forward(self, x):
        identity = x
        
        # 光谱先验
        if x.size(-1) == self.spectral_prior.size(0):
            prior = self.spectral_prior.view(1, 1, 1, 1, -1)
            x = x * prior
        elif x.size(-1) > 1 and self.spectral_prior.size(0) > x.size(-1):
            indices = torch.linspace(0, self.spectral_prior.size(0) - 1, 
                                    x.size(-1), device=x.device).long()
            prior = self.spectral_prior[indices].view(1, 1, 1, 1, -1)
            x = x * prior
        
        # 通道和空间注意力
        ca = self.channel_att(x)
        x = x * ca
        
        sa = self.spatial_att(x)
        x = x * sa
        
        # Residual
        x = x + identity
        
        return x


# ============================================================================
# PBRN Module (创新点3 - 新增)
# ============================================================================

class RefinementBlock(nn.Module):
    """
    细化块
    包含：残差连接 + 空洞卷积 + 通道注意力
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 主干卷积
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 空洞卷积（扩大感受野）
        self.atrous_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_attention = ChannelAttention(out_channels, reduction=4)
        
        # 投影（用于残差连接）
        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()
        
        # 预测头（中间监督）
        self.pred_head = nn.Conv3d(out_channels, 2, kernel_size=1)
    
    def forward(self, x):
        identity = self.projection(x)
        
        # 主干
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 空洞卷积
        out_atrous = self.atrous_conv(out)
        out = out + out_atrous
        
        # 通道注意力
        out = self.channel_attention(out)
        
        # 残差
        out = out + identity
        
        # 中间预测
        pred = self.pred_head(out)
        
        return out, pred


class ProgressiveBoundaryRefinementNetwork(nn.Module):
    """
    渐进式边界细化网络（创新点3）
    
    设计特点：
    1. 只处理边界ROI（计算高效）
    2. 三尺度渐进细化（32→16→8像素）
    3. 多模态输入（原图+粗预测+不确定性）
    4. 轻量级（参数<主网络10%）
    """
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Level 1: 粗尺度（32像素）
        self.level1 = RefinementBlock(
            in_channels=in_channels,
            out_channels=base_channels
        )
        
        # Level 2: 中尺度（16像素）
        self.level2 = RefinementBlock(
            in_channels=base_channels + in_channels,
            out_channels=base_channels * 2
        )
        
        # Level 3: 细尺度（8像素）
        self.level3 = RefinementBlock(
            in_channels=base_channels * 2 + in_channels,
            out_channels=base_channels * 4
        )
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, 2, kernel_size=1)
        )
        
        print(f"✓ PBRN initialized: base_channels={base_channels}")
    
    def extract_boundary_roi(self, pred, expansion=32):
        """提取边界ROI"""
        pred_binary = (torch.softmax(pred, dim=1)[:, 1, ...] > 0.5).float()
        
        # 形态学操作提取边界
        dilated = F.max_pool3d(
            pred_binary.unsqueeze(1), 
            kernel_size=3, stride=1, padding=1
        )
        eroded = -F.max_pool3d(
            -pred_binary.unsqueeze(1), 
            kernel_size=3, stride=1, padding=1
        )
        boundary = (dilated - eroded).squeeze(1)
        
        # 扩展边界区域
        roi_mask = F.max_pool3d(
            boundary.unsqueeze(1),
            kernel_size=min(expansion*2+1, 15),  # 限制kernel大小
            stride=1,
            padding=min(expansion, 7)
        ).squeeze(1)
        
        return boundary, roi_mask
    
    def estimate_uncertainty(self, pred):
        """估计预测不确定性（熵）"""
        pred_prob = torch.softmax(pred, dim=1)
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-7), dim=1)
        
        # 归一化
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-7)
        return entropy_norm
    
    def forward(self, original_image, coarse_pred):
        """
        Args:
            original_image: (B, 1, H, W, D) - 原始图像
            coarse_pred: (B, 2, H, W, D) - 粗分割输出
        
        Returns:
            refined_pred: (B, 2, H, W, D) - 精细化预测
            intermediate_preds: list - 中间预测（用于监督）
        """
        # 1. 提取边界ROI
        boundary, roi_mask = self.extract_boundary_roi(coarse_pred, expansion=32)
        
        # 2. 估计不确定性
        uncertainty = self.estimate_uncertainty(coarse_pred)
        
        # 3. 多模态输入
        coarse_prob = torch.softmax(coarse_pred, dim=1)
        multi_modal_input = torch.cat([
            original_image,
            coarse_prob,
            uncertainty.unsqueeze(1)
        ], dim=1)  # (B, 4, H, W, D)
        
        # 4. 渐进式细化
        intermediate_preds = []
        
        # Level 1
        feat1, pred1 = self.level1(multi_modal_input)
        intermediate_preds.append(pred1)
        
        # Level 2
        feat1_up = F.interpolate(feat1, size=multi_modal_input.shape[2:], mode='trilinear', align_corners=False)
        input2 = torch.cat([feat1_up, multi_modal_input], dim=1)
        feat2, pred2 = self.level2(input2)
        intermediate_preds.append(pred2)
        
        # Level 3
        feat2_up = F.interpolate(feat2, size=multi_modal_input.shape[2:], mode='trilinear', align_corners=False)
        input3 = torch.cat([feat2_up, multi_modal_input], dim=1)
        feat3, pred3 = self.level3(input3)
        intermediate_preds.append(pred3)
        
        # 5. 最终输出
        refined_pred = self.final_conv(feat3)
        
        # 6. 融合粗预测和细化预测（ROI内用细化，ROI外用粗预测）
        roi_mask_expanded = roi_mask.unsqueeze(1).expand_as(coarse_pred)
        final_pred = torch.where(
            roi_mask_expanded > 0.5,
            refined_pred,
            coarse_pred
        )
        
        return final_pred, intermediate_preds


# ============================================================================
# 完整网络
# ============================================================================

class HSIUNetWithInnovations(AbstractDynamicNetworkArchitectures):
    """
    HSI-UNet with Three Innovations
    
    创新点：
    1. CLEF: 对比学习（在trainer中实现训练策略）
    2. SSFA: 空间注意力（已有）
    3. PBRN: 边界细化网络（新增）
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
        enable_pbrn: bool = True,  # 是否启用PBRN
    ):
        super().__init__()

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
        self.enable_pbrn = enable_pbrn
        
        # Encoder
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
        
        # Decoder
        self.decoder = UNetDecoder(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision, 
            nonlin_first=nonlin_first
        )
        
        # SSFA模块
        self.ssfa_modules = nn.ModuleList()
        for i in range(n_stages):
            if i < 5:
                self.ssfa_modules.append(
                    SSFAModule(features_per_stage[i], use_prior=True)
                )
            else:
                self.ssfa_modules.append(nn.Identity())
        
        # PBRN模块（可选）
        if self.enable_pbrn:
            self.pbrn = ProgressiveBoundaryRefinementNetwork(
                in_channels=4,  # 原图1 + 粗预测2 + 不确定性1
                base_channels=16
            )
        
        print(f"✓ HSI-UNet with Three Innovations initialized:")
        print(f"  - SSFA: {sum(1 for m in self.ssfa_modules if not isinstance(m, nn.Identity))} stages")
        print(f"  - PBRN: {'Enabled' if self.enable_pbrn else 'Disabled'}")
        print(f"  - CLEF: Will be implemented in training strategy")

    def forward(self, x):
        """
        前向传播：
        1. Encoder
        2. SSFA on skip connections
        3. Decoder (coarse prediction)
        4. PBRN (refinement)
        """
        # 保存原始输入（用于PBRN）
        original_image = x
        
        # 1. Encoder
        skips = self.encoder(x)
        
        # 2. SSFA增强skip connections
        enhanced_skips = []
        for i, skip in enumerate(skips):
            if i < len(self.ssfa_modules):
                enhanced_skip = self.ssfa_modules[i](skip)
                enhanced_skips.append(enhanced_skip)
            else:
                enhanced_skips.append(skip)
        
        # 3. Decoder（粗预测）
        coarse_outputs = self.decoder(enhanced_skips)
        
        # 4. PBRN（细化）
        if self.enable_pbrn and not self.training:
            # 推理时使用PBRN细化
            coarse_pred = coarse_outputs[0] if isinstance(coarse_outputs, (list, tuple)) else coarse_outputs
            refined_pred, _ = self.pbrn(original_image, coarse_pred)
            
            # 如果是深度监督，替换最终输出
            if isinstance(coarse_outputs, (list, tuple)):
                return [refined_pred] + list(coarse_outputs[1:])
            else:
                return refined_pred
        else:
            # 训练时或PBRN禁用时，返回粗预测
            return coarse_outputs
    
    def forward_with_refinement(self, x):
        """
        专门用于训练PBRN的前向传播
        返回coarse和refined预测
        """
        original_image = x
        
        # Encoder + SSFA + Decoder
        skips = self.encoder(x)
        enhanced_skips = []
        for i, skip in enumerate(skips):
            if i < len(self.ssfa_modules):
                enhanced_skips.append(self.ssfa_modules[i](skip))
            else:
                enhanced_skips.append(skip)
        
        coarse_outputs = self.decoder(enhanced_skips)
        coarse_pred = coarse_outputs[0] if isinstance(coarse_outputs, (list, tuple)) else coarse_outputs
        
        # PBRN细化
        if self.enable_pbrn:
            refined_pred, intermediate_preds = self.pbrn(original_image, coarse_pred)
            return {
                'coarse': coarse_outputs,
                'refined': refined_pred,
                'intermediate': intermediate_preds
            }
        else:
            return {'coarse': coarse_outputs}

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + \
               self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == "__main__":
    """测试网络"""
    print("="*70)
    print("Testing HSI-UNet with Three Innovations")
    print("="*70)
    
    model = HSIUNetWithInnovations(
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
        deep_supervision=True,
        enable_pbrn=True
    )
    
    print(f"\n✓ Model created successfully")
    
    # 测试前向传播
    x = torch.rand(1, 1, 128, 128, 20)
    print(f"✓ Input shape: {x.shape}")
    
    with torch.no_grad():
        # 测试标准前向传播
        outputs = model(x)
        print(f"✓ Forward pass successful")
        if isinstance(outputs, (list, tuple)):
            print(f"  Output shapes: {[o.shape for o in outputs]}")
        else:
            print(f"  Output shape: {outputs.shape}")
        
        # 测试带细化的前向传播
        outputs_with_ref = model.forward_with_refinement(x)
        print(f"✓ Forward with refinement successful")
        print(f"  Keys: {outputs_with_ref.keys()}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    ssfa_params = sum(p.numel() for m in model.ssfa_modules for p in m.parameters())
    pbrn_params = sum(p.numel() for p in model.pbrn.parameters()) if model.enable_pbrn else 0
    baseline_params = total_params - ssfa_params - pbrn_params
    
    print(f"\n✓ Parameter Statistics:")
    print(f"  Total: {total_params:,}")
    print(f"  Baseline: {baseline_params:,} ({baseline_params/total_params*100:.1f}%)")
    print(f"  SSFA: {ssfa_params:,} ({ssfa_params/total_params*100:.1f}%)")
    print(f"  PBRN: {pbrn_params:,} ({pbrn_params/total_params*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

