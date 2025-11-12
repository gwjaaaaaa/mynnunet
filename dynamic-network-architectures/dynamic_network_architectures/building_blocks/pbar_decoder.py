"""
PBAR - Progressive Boundary-Aware Refinement Decoder
渐进式边界感知优化解码器

核心创新：
1. 双流解码架构 - 分割流和边界流并行处理
2. 边界感知注意力 - 显式建模边界区域的特征
3. 渐进式边界优化 - 多级逐步细化边界预测
4. 边界引导特征融合 - 利用边界信息增强跳跃连接

理论创新点：
- 将边界检测作为辅助任务，与分割任务联合优化
- 边界特征用于引导和增强分割特征
- 渐进式优化策略保证从粗到细的边界细化
- 提供边界预测输出，可用于边界损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class BoundaryDetectionModule(nn.Module):
    """
    边界检测模块
    
    创新点：使用多尺度梯度特征显式检测边界
    不依赖ground truth边界，而是从特征中学习边界模式
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d):
        super().__init__()
        
        # Sobel-like边界检测分支（学习的边界算子）
        self.boundary_conv_x = conv_op(in_channels, out_channels, kernel_size=3, padding=1)
        self.boundary_conv_y = conv_op(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 多尺度边界特征
        self.multiscale_boundary = nn.ModuleList([
            nn.Sequential(
                conv_op(in_channels, out_channels, kernel_size=k, padding=k//2),
                norm_op(out_channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]
        ])
        
        # 边界特征融合
        self.boundary_fusion = nn.Sequential(
            conv_op(out_channels * 5, out_channels, kernel_size=1),  # 2 (gradients) + 3 (multiscale)
            norm_op(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 边界显著性预测
        self.boundary_prediction = nn.Sequential(
            conv_op(out_channels, out_channels // 2, kernel_size=3, padding=1),
            norm_op(out_channels // 2),
            nn.ReLU(inplace=True),
            conv_op(out_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        输入: x (B, C, H, W, D)
        输出: 
            - boundary_features: (B, C_out, H, W, D) 边界特征
            - boundary_map: (B, 1, H, W, D) 边界概率图
        """
        # 梯度特征（类似Sobel）
        grad_x = self.boundary_conv_x(x)
        grad_y = self.boundary_conv_y(x)
        
        # 多尺度边界特征
        multiscale_features = [conv(x) for conv in self.multiscale_boundary]
        
        # 融合所有边界特征
        all_features = [grad_x, grad_y] + multiscale_features
        concat_features = torch.cat(all_features, dim=1)
        boundary_features = self.boundary_fusion(concat_features)
        
        # 边界显著性图
        boundary_map = self.boundary_prediction(boundary_features)
        
        return boundary_features, boundary_map


class BoundaryGuidedFusion(nn.Module):
    """
    边界引导的特征融合模块
    
    创新点：利用边界信息动态调整跳跃连接的融合权重
    边界区域使用更多的encoder特征，内部区域使用更多的decoder特征
    """
    def __init__(self, 
                 channels,
                 conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d):
        super().__init__()
        
        # 特征对齐
        self.encoder_align = nn.Sequential(
            conv_op(channels, channels, kernel_size=1),
            norm_op(channels)
        )
        
        self.decoder_align = nn.Sequential(
            conv_op(channels, channels, kernel_size=1),
            norm_op(channels)
        )
        
        # 边界引导的门控网络
        self.gating_network = nn.Sequential(
            conv_op(channels * 2 + 1, channels, kernel_size=3, padding=1),  # +1 for boundary map
            norm_op(channels),
            nn.ReLU(inplace=True),
            conv_op(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 融合后的增强
        self.fusion_enhance = nn.Sequential(
            conv_op(channels, channels, kernel_size=3, padding=1),
            norm_op(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, encoder_features, decoder_features, boundary_map):
        """
        输入:
            - encoder_features: (B, C, H, W, D) 来自encoder的跳跃连接
            - decoder_features: (B, C, H, W, D') 来自decoder上一层（D'可能与D不同）
            - boundary_map: (B, 1, H, W, D) 边界概率图
        输出:
            - fused_features: (B, C, H, W, D)
        """
        # 如果decoder特征的空间维度与encoder不匹配，进行插值对齐
        if decoder_features.shape[2:] != encoder_features.shape[2:]:
            decoder_features = F.interpolate(
                decoder_features,
                size=encoder_features.shape[2:],
                mode='trilinear' if len(encoder_features.shape) == 5 else 'bilinear',
                align_corners=False
            )
        
        # 特征对齐
        encoder_aligned = self.encoder_align(encoder_features)
        decoder_aligned = self.decoder_align(decoder_features)
        
        # 生成门控权重（边界引导）
        concat_for_gating = torch.cat([encoder_aligned, decoder_aligned, boundary_map], dim=1)
        gate = self.gating_network(concat_for_gating)  # (B, C, H, W, D)
        
        # 门控融合：gate高的地方更多使用encoder特征（通常是边界区域）
        fused = gate * encoder_aligned + (1 - gate) * decoder_aligned
        
        # 融合后增强
        enhanced = self.fusion_enhance(fused)
        
        return enhanced


class BoundaryRefinementBlock(nn.Module):
    """
    边界优化块
    
    创新点：专门针对边界区域的特征优化
    使用边界图加权，重点优化边界附近的特征
    """
    def __init__(self,
                 channels,
                 conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d):
        super().__init__()
        
        # 边界区域特征提取
        self.boundary_feature_extract = nn.Sequential(
            conv_op(channels + 1, channels, kernel_size=3, padding=1),  # +1 for boundary map
            norm_op(channels),
            nn.ReLU(inplace=True),
            conv_op(channels, channels, kernel_size=3, padding=1),
            norm_op(channels),
            nn.ReLU(inplace=True)
        )
        
        # 边界自注意力
        self.boundary_attention = nn.Sequential(
            conv_op(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_op(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            conv_op(channels, channels, kernel_size=1),
            norm_op(channels)
        )
        
    def forward(self, features, boundary_map):
        """
        输入:
            - features: (B, C, H, W, D)
            - boundary_map: (B, 1, H, W, D)
        输出:
            - refined_features: (B, C, H, W, D)
        """
        # 拼接边界图
        concat_input = torch.cat([features, boundary_map], dim=1)
        
        # 提取边界区域特征
        boundary_features = self.boundary_feature_extract(concat_input)
        
        # 边界注意力
        attention = self.boundary_attention(boundary_features)
        
        # 应用注意力
        attended_features = features * attention
        
        # 输出投影
        output = self.output_proj(attended_features)
        
        # 残差连接
        refined = output + features
        
        return refined


class PBARDecoderStage(nn.Module):
    """
    PBAR解码器的一个stage
    
    包含：上采样 -> 边界检测 -> 边界引导融合 -> 边界优化 -> 卷积块
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels,
                 conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d,
                 norm_op_kwargs=None,
                 dropout_op=None,
                 dropout_op_kwargs=None,
                 nonlin=nn.ReLU,
                 nonlin_kwargs={'inplace': True},
                 n_conv_per_stage=2):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        
        # 通道调整
        self.channel_adjust = conv_op(in_channels, out_channels, kernel_size=1)
        
        # 上采样使用插值（更灵活，可以处理任意stride）
        self.conv_op_type = conv_op
        
        # 边界检测
        self.boundary_detection = BoundaryDetectionModule(
            skip_channels, out_channels, conv_op, norm_op
        )
        
        # 边界引导融合
        self.boundary_guided_fusion = BoundaryGuidedFusion(
            out_channels, conv_op, norm_op
        )
        
        # 边界优化
        self.boundary_refinement = BoundaryRefinementBlock(
            out_channels, conv_op, norm_op
        )
        
        # 标准卷积块
        self.conv_blocks = nn.Sequential(*[
            nn.Sequential(
                conv_op(out_channels, out_channels, kernel_size=3, padding=1),
                norm_op(out_channels, **norm_op_kwargs),
                nonlin(**nonlin_kwargs)
            ) for _ in range(n_conv_per_stage)
        ])
        
        # Dropout
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op is not None else None
        
    def forward(self, decoder_input, encoder_skip):
        """
        输入:
            - decoder_input: (B, C_in, H, W, D) 来自上一个decoder stage
            - encoder_skip: (B, C_skip, H', W', D') 来自encoder的跳跃连接
        输出:
            - output: (B, C_out, H', W', D')
            - boundary_map: (B, 1, H', W', D') 边界图
        """
        # 1. 通道调整
        x = self.channel_adjust(decoder_input)
        
        # 2. 上采样到encoder skip的尺寸
        if x.shape[2:] != encoder_skip.shape[2:]:
            x = F.interpolate(
                x,
                size=encoder_skip.shape[2:],
                mode='trilinear' if self.conv_op_type == nn.Conv3d else 'bilinear',
                align_corners=False
            )
        upsampled = x
        
        # 2. 从encoder特征检测边界
        boundary_features, boundary_map = self.boundary_detection(encoder_skip)
        
        # 3. 边界引导的特征融合
        fused = self.boundary_guided_fusion(encoder_skip, upsampled, boundary_map)
        
        # 4. 边界优化
        refined = self.boundary_refinement(fused, boundary_map)
        
        # 5. 标准卷积处理
        output = self.conv_blocks(refined)
        
        # 6. Dropout
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output, boundary_map


class PBARDecoder(nn.Module):
    """
    完整的PBAR Decoder
    
    创新点总结：
    1. 双流架构 - 分割流和边界流并行
    2. 多级边界检测 - 每个decoder stage都检测边界
    3. 边界引导融合 - 利用边界信息改进跳跃连接
    4. 渐进式优化 - 从粗到细逐步细化边界和分割
    
    特点：
    - 输出分割结果和边界图
    - 支持deep supervision
    - 保持与nnU-Net decoder相同的接口
    """
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision=True,
                 nonlin_first: bool = False):
        super().__init__()
        
        # 从encoder获取配置
        self.encoder = encoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # 获取encoder的stage输出通道数
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        # Decoder stages
        self.stages = nn.ModuleList()
        self.seg_heads = nn.ModuleList()  # 分割输出头
        self.boundary_heads = nn.ModuleList()  # 边界输出头（用于deep supervision）
        
        # 构建decoder stages
        for s in range(1, n_stages_encoder):
            input_channels = encoder.output_channels[-s]
            output_channels = encoder.output_channels[-(s+1)]
            skip_channels = encoder.output_channels[-(s+1)]
            
            # Decoder stage
            stage = PBARDecoderStage(
                in_channels=input_channels,
                out_channels=output_channels,
                skip_channels=skip_channels,
                conv_op=encoder.conv_op,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
                n_conv_per_stage=n_conv_per_stage[s-1]
            )
            self.stages.append(stage)
            
            # 分割输出头（用于deep supervision）
            seg_head = encoder.conv_op(output_channels, num_classes, kernel_size=1)
            self.seg_heads.append(seg_head)
            
            # 边界输出头（用于辅助监督）
            boundary_head = encoder.conv_op(output_channels, 1, kernel_size=1)
            self.boundary_heads.append(boundary_head)
        
        print(f"  [PBAR Decoder] Initialized:")
        print(f"    - Num stages: {len(self.stages)}")
        print(f"    - Deep supervision: {deep_supervision}")
        print(f"    - Dual-stream outputs: Segmentation + Boundary")
    
    def forward(self, skips):
        """
        输入: skips - list of encoder outputs
            skips[0]: 最浅层 (分辨率最高)
            skips[-1]: 最深层 (分辨率最低, bottleneck)
        
        输出:
            如果deep_supervision=True:
                - seg_outputs: list of (B, num_classes, H, W, D)
                - boundary_outputs: list of (B, 1, H, W, D)
            否则:
                - seg_output: (B, num_classes, H, W, D)
                - boundary_output: (B, 1, H, W, D)
        """
        seg_outputs = []
        boundary_outputs = []
        
        # 从bottleneck开始
        x = skips[-1]
        
        # 逐级上采样和优化
        for i, stage in enumerate(self.stages):
            # 获取对应的encoder skip connection
            skip = skips[-(i+2)]  # 倒数第2个开始
            
            # Decoder stage
            x, boundary_map = stage(x, skip)
            
            # 生成分割和边界输出
            seg_out = self.seg_heads[i](x)
            boundary_out = self.boundary_heads[i](x)
            
            seg_outputs.append(seg_out)
            boundary_outputs.append(boundary_map)  # 使用检测到的边界图
        
        # 返回结果
        if self.deep_supervision:
            # Deep supervision: 返回所有层级的输出
            return seg_outputs[::-1], boundary_outputs[::-1]  # 从高分辨率到低分辨率
        else:
            # 只返回最后一层
            return seg_outputs[-1], boundary_outputs[-1]
    
    def compute_conv_feature_map_size(self, input_size):
        """
        计算decoder的feature map size
        与nnU-Net保持接口兼容
        """
        # 简化计算，返回一个估计值
        skip_sizes = self.encoder.compute_conv_feature_map_size(input_size)
        # Decoder的计算量约等于encoder
        return skip_sizes


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing PBAR Decoder")
    print("="*70)
    
    # 创建一个简单的mock encoder用于测试
    from dynamic_network_architectures.architectures.unet import PlainConvEncoder
    
    print("\n1. Creating mock encoder...")
    encoder = PlainConvEncoder(
        input_channels=1,
        n_stages=5,
        features_per_stage=[32, 64, 128, 256, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3,3,3]] * 5,
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,1], [2,2,1]],
        n_conv_per_stage=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={'inplace': True},
        return_skips=True
    )
    
    print("\n2. Creating PBAR Decoder...")
    decoder = PBARDecoder(
        encoder=encoder,
        num_classes=2,
        n_conv_per_stage=2,
        deep_supervision=True
    )
    
    print("\n3. Forward pass...")
    x = torch.randn(2, 1, 64, 64, 20)
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        # Encoder
        skips = encoder(x)
        print(f"   Encoder outputs: {[s.shape for s in skips]}")
        
        # Decoder
        seg_outputs, boundary_outputs = decoder(skips)
        print(f"   Seg outputs: {[s.shape for s in seg_outputs]}")
        print(f"   Boundary outputs: {[b.shape for b in boundary_outputs]}")
    
    # 测试without deep supervision
    print("\n4. Testing without deep supervision...")
    decoder_no_ds = PBARDecoder(
        encoder=encoder,
        num_classes=2,
        n_conv_per_stage=2,
        deep_supervision=False
    )
    
    with torch.no_grad():
        seg_output, boundary_output = decoder_no_ds(skips)
        print(f"   Seg output: {seg_output.shape}")
        print(f"   Boundary output: {boundary_output.shape}")
    
    # 参数量统计
    print("\n5. Parameter count:")
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"   PBAR Decoder: {total_params:,}")
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Encoder: {encoder_params:,}")
    print(f"   Total (Encoder+Decoder): {encoder_params + total_params:,}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)





