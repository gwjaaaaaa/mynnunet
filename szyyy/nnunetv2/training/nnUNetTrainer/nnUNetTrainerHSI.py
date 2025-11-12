"""
nnUNetTrainerHSI
集成SPGA、DSR和SSCA创新点的nnU-Net训练器

创新点：
1. SPGA - Spectral Prototype-Guided Adaptive Attention (编码器)
2. DSR - Dynamic Spectral Routing (编码器)
3. SSCA - Spectral-Spatial Channel Attention with Doubly Smoothed Prior (跳跃连接)
   - 超轻量SE-Net (<1K参数/stage)
   - 双重平滑密度估计先验（理论创新）
   - 先验用于初始化权重（零运行时开销）

使用方法：
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerHSI
"""

import torch
import torch.nn as nn
from typing import Union, List, Tuple
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerHSI(nnUNetTrainer):
    """
    nnUNetTrainerHSI with SPGA + DSR
    
    在标准nnU-Net基础上集成SPGA和DSR模块
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        初始化训练器
        """
        # === 配置 - SPGA + DSR + SSCA + PHSP ===
        self.use_spga = True     # ✓ 启用SPGA-Lite (编码器)
        self.use_dsr = True      # ✓ 启用DSR-Lite (编码器, 4专家版本)
        self.use_ssca = False     # ✗ 禁用SSCA (超轻量通道注意力)
        self.use_phsp = False    # ✗ 禁用PHSP (原型驱动层级光谱先验 - 禁用)
        self.use_bear = False    # ✗ 禁用BEAR (显存不足)
        self.use_pbar = False    # ✗ 禁用
        self.use_scr = False     # ✗ 禁用
        
        self.spectral_dim = 60  # 光谱维度（根据实际数据：60个波段）
        
        # SPGA轻量级配置
        self.num_spga_prototypes = 4
        self.spga_downsample_attention = True
        self.spga_apply_to_stages = [2, 3, 4]
        
        # DSR轻量级配置 (4专家版本)
        self.num_dsr_experts = 4  # 4个专家：光谱、空间、细粒度、标准3D
        self.dsr_lightweight_experts = True
        self.dsr_apply_to_stages = [1, 2, 3]
        
        # SSCA配置 (新增 - 跳跃连接)
        self.spectral_weights_path = '/data/CXY/g/szy/spectral_prior_weights/spectral_prior_weights_final.npy'
        self.ssca_reduction = 4  # 压缩率：4=标准, 8=更轻量
        self.ssca_dropout_rate = 0.15  # Dropout率（减少过拟合，稳定val loss）
        self.ssca_apply_to_stages = [0, 1, 2, 3, 4, 5, 6]  # 应用到所有跳跃连接
        
        # PHSP配置 (新增 - 原型驱动的层级光谱先验)
        self.phsp_smooth_alpha = 0.001  # 平滑性损失权重
        self.phsp_smooth_order = 1  # 平滑阶数：1=一阶差分, 2=二阶差分
        self.phsp_use_consistency = False  # 是否使用跨stage一致性损失
        self.phsp_consistency_alpha = 0.0005  # 一致性损失权重
        
        # BEAR配置 (新增 - 解码器)
        self.bear_use_lite = False  # False=标准版(30-50K参数), True=超轻量版(15-20K参数)
        self.bear_use_uncertainty = True  # 是否使用不确定性引导（来自SDAR）
        self.bear_uncertainty_threshold = 0.5  # 不确定性阈值
        
        # 调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        print("="*80)
        print("nnUNetTrainerHSI - SPGA + DSR + SSCA")
        print("="*80)
        print(f"  SPGA: {'✓ Enabled' if self.use_spga else '✗ Disabled'} (Encoder)")
        print(f"    - Prototypes: {self.num_spga_prototypes}")
        print(f"    - Apply to stages: {self.spga_apply_to_stages}")
        print(f"  DSR:  {'✓ Enabled' if self.use_dsr else '✗ Disabled'} (Encoder)")
        print(f"    - Num experts: {self.num_dsr_experts} (Spectral, Spatial, Fine-grained, Standard-3D)")
        print(f"    - Apply to stages: {self.dsr_apply_to_stages}")
        print(f"  SSCA: {'✓ Enabled' if self.use_ssca else '✗ Disabled'} (Skip Connections)")
        print(f"    - Reduction: {self.ssca_reduction}")
        print(f"    - Doubly Smoothed Prior: Yes")
        print(f"  PHSP: {'✓ Enabled' if self.use_phsp else '✗ Disabled'} (Skip Connections - Prototype-Driven)")
        print(f"  Spectral dim: {self.spectral_dim}")
        print("="*80)
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建包含SPGA和DSR的网络
        """
        from dynamic_network_architectures.architectures.unet import PlainConvEncoder, UNetDecoder
        from dynamic_network_architectures.building_blocks.spga_module import SPGAModuleEfficientLite
        from dynamic_network_architectures.building_blocks.dsr_module import DSRModuleEfficientLite4Experts
        from dynamic_network_architectures.building_blocks.dspa_module import DSPAModule, DSPAModuleLite
        
        # 处理需要导入的参数
        import importlib
        for param_name in arch_init_kwargs_req_import:
            if param_name in arch_init_kwargs and isinstance(arch_init_kwargs[param_name], str):
                module_path, class_name = arch_init_kwargs[param_name].rsplit('.', 1)
                module = importlib.import_module(module_path)
                arch_init_kwargs[param_name] = getattr(module, class_name)
        
        # 获取网络参数
        n_stages = arch_init_kwargs['n_stages']
        features_per_stage = arch_init_kwargs['features_per_stage']
        conv_op = arch_init_kwargs['conv_op']
        kernel_sizes = arch_init_kwargs['kernel_sizes']
        strides = arch_init_kwargs['strides']
        n_conv_per_stage = arch_init_kwargs['n_conv_per_stage']
        n_conv_per_stage_decoder = arch_init_kwargs.get('n_conv_per_stage_decoder', n_conv_per_stage[:-1])
        
        conv_bias = arch_init_kwargs.get('conv_bias', False)
        norm_op = arch_init_kwargs.get('norm_op', None)
        norm_op_kwargs = arch_init_kwargs.get('norm_op_kwargs', {})
        dropout_op = arch_init_kwargs.get('dropout_op', None)
        dropout_op_kwargs = arch_init_kwargs.get('dropout_op_kwargs', {})
        nonlin = arch_init_kwargs.get('nonlin', nn.ReLU)
        nonlin_kwargs = arch_init_kwargs.get('nonlin_kwargs', {'inplace': True})
        nonlin_first = arch_init_kwargs.get('nonlin_first', False)
        
        # HSI参数 - 使用类级别的默认配置
        spectral_dim = 60  # 光谱维度
        use_spga = True    # ✓ 启用SPGA (编码器)
        use_dsr = True     # ✓ 启用DSR (编码器, 4专家版本)
        use_ssca = False   # ✗ 禁用SSCA (超轻量通道注意力)
        use_phsp = False   # ✗ 禁用PHSP (跳跃连接 - 原型驱动，禁用)
        use_bear = False   # ✗ 禁用BEAR (显存不足)
        use_scr = False    # ✗ 禁用SCR
        
        # SSCA配置
        spectral_weights_path = '/data/CXY/g/szy/spectral_prior_weights/spectral_prior_weights_final.npy'
        ssca_reduction = 4  # 压缩率：4=标准, 8=更轻量
        ssca_dropout_rate = 0.15  # Dropout率（减少过拟合）
        ssca_use_doubly_smoothing = True  # 使用双重平滑
        
        # BEAR配置
        bear_use_lite = False  # False=标准版(30-50K参数), True=超轻量版(15-20K参数)
        bear_use_uncertainty = True  # 是否使用不确定性引导
        bear_uncertainty_threshold = 0.5  # 不确定性阈值
        
        print("\n" + "="*80)
        print("Building HSI-UNet")
        print("="*80)
        # 创建Encoder
        print("\n[1/3] Building Encoder...")
        encoder = PlainConvEncoder(
            num_input_channels,
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
        
        # SPGA和DSR的配置
        num_spga_prototypes = 4
        spga_downsample_attention = True
        spga_apply_to_stages = [2, 3, 4]
        
        num_dsr_experts = 4  # 4专家版本
        dsr_lightweight_experts = True
        dsr_apply_to_stages = [1, 2, 3]
        
        # 添加SPGA模块
        if use_spga:
            print("\n[2/3] Adding SPGA Modules...")
            spga_modules = nn.ModuleList()
            for i, channels in enumerate(features_per_stage):
                if i in spga_apply_to_stages:
                    spga = SPGAModuleEfficientLite(
                        channels, 
                        spectral_dim, 
                        num_prototypes=num_spga_prototypes,
                        downsample_attention=spga_downsample_attention
                    )
                else:
                    spga = nn.Identity()
                spga_modules.append(spga)
        else:
            spga_modules = nn.ModuleList([nn.Identity() for _ in range(n_stages)])
        
        # 添加DSR模块 - 使用4专家增强版
        if use_dsr:
            print("\n[3/4] Adding DSR Modules (4-Expert Version)...")
            dsr_modules = nn.ModuleList()
            for i, channels in enumerate(features_per_stage):
                if i in dsr_apply_to_stages:
                    dsr = DSRModuleEfficientLite4Experts(
                        channels,
                        spectral_dim,
                        lightweight_experts=dsr_lightweight_experts
                    )
                else:
                    dsr = nn.Identity()
                dsr_modules.append(dsr)
        else:
            dsr_modules = nn.ModuleList([nn.Identity() for _ in range(n_stages)])
        
        # 添加SSCA模块 (新增 - 应用在skip connections)
        if use_ssca:
            from dynamic_network_architectures.building_blocks.ssca_module import build_ssca_module
            print("\n[4/6] Adding SSCA Modules (Spectral-Spatial Channel Attention with Doubly Smoothed Prior)...")
            ssca_modules = nn.ModuleList()
            for i, channels in enumerate(features_per_stage):
                ssca = build_ssca_module(
                    channels=channels,
                    spectral_dim=spectral_dim,
                    spectral_prior_path=spectral_weights_path,
                    reduction=ssca_reduction,
                    use_doubly_smoothing=ssca_use_doubly_smoothing,
                    dropout_rate=ssca_dropout_rate
                )
                ssca_modules.append(ssca)
        else:
            ssca_modules = nn.ModuleList([nn.Identity() for _ in range(n_stages)])
        
        # 添加PHSP模块 (新增 - 原型驱动的层级光谱先验)
        phsp_module = None
        if use_phsp:
            from dynamic_network_architectures.building_blocks.phsp_module import build_phsp_module
            from dynamic_network_architectures.building_blocks.phsp_loss import PHSPCompositeLoss
            print("\n[5/6] Adding PHSP Module (Prototype-Driven Hierarchical Spectral Prior)...")
            
            # PHSP配置
            phsp_smooth_alpha = 0.001
            phsp_smooth_order = 1
            phsp_use_consistency = False
            phsp_consistency_alpha = 0.0005
            
            # 构建PHSP模块
            phsp_module = build_phsp_module(
                channels_per_stage=features_per_stage,
                spectral_dim=spectral_dim,
                num_prototypes=num_spga_prototypes,
                spga_stages=spga_apply_to_stages
            )
            
            print(f"  PHSP initialized with:")
            print(f"    - Smooth alpha: {phsp_smooth_alpha}")
            print(f"    - Smooth order: {phsp_smooth_order}")
            print(f"    - Consistency: {phsp_use_consistency}")
        
        # 创建Decoder
        print("\n[5/5] Building Decoder...")
        decoder = UNetDecoder(
            encoder,
            num_output_channels,
            n_conv_per_stage_decoder,
            enable_deep_supervision,
            nonlin_first
        )
        
        # 添加BEAR模块 (新增 - 应用在decoder stages)
        if use_bear:
            from dynamic_network_architectures.building_blocks.bear_module import build_bear_module
            print("\n[5/5] Adding BEAR Modules (Boundary-Enhanced Adaptive Refinement)...")
            bear_modules = nn.ModuleList()
            # BEAR应用在decoder的每个stage
            # Decoder stages的通道数是features_per_stage的反向
            decoder_features = list(reversed(features_per_stage[:-1]))  # 排除bottleneck
            for i, channels in enumerate(decoder_features):
                bear = build_bear_module(
                    channels=channels,
                    use_lite=bear_use_lite,
                    use_uncertainty=bear_use_uncertainty,
                    uncertainty_threshold=bear_uncertainty_threshold
                )
                bear_modules.append(bear)
        else:
            bear_modules = nn.ModuleList([nn.Identity() for _ in range(n_stages - 1)])
        
        # 组装完整网络
        class HSIUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.spga_modules = spga_modules
                self.dsr_modules = dsr_modules
                self.ssca_modules = ssca_modules
                self.phsp_module = phsp_module  # 新增：PHSP模块
                self.bear_modules = bear_modules
                self.decoder = decoder
                self.use_spga = use_spga
                self.use_dsr = use_dsr
                self.use_ssca = use_ssca
                self.use_phsp = use_phsp  # 新增：PHSP开关
                self.use_bear = use_bear
                self.use_scr = use_scr
                self.deep_supervision = enable_deep_supervision
                
                # PHSP辅助变量（用于保存光谱先验，供loss计算）
                self.spectral_priors = None
            
            def forward(self, x):
                # === Encoder阶段 ===
                skips = []
                for i, stage in enumerate(self.encoder.stages):
                    x = stage(x)
                    if self.use_spga:
                        x = self.spga_modules[i](x)
                    if self.use_dsr:
                        x = self.dsr_modules[i](x)
                    skips.append(x)
                
                # === SSCA阶段：光谱-空间通道注意力（双重平滑先验引导）===
                if self.use_ssca and not self.use_phsp:
                    refined_skips = []
                    for i, skip in enumerate(skips):
                        # SSCA: 超轻量SE-Net + 双重平滑先验初始化
                        refined_skip = self.ssca_modules[i](skip)
                        refined_skips.append(refined_skip)
                    skips = refined_skips
                
                # === PHSP阶段：原型驱动的层级光谱先验学习 ===
                if self.use_phsp:
                    # PHSP: 从SPGA原型提取动态光谱先验，应用到跳跃连接
                    refined_skips, spectral_priors = self.phsp_module(skips, self.spga_modules)
                    skips = refined_skips
                    # 保存光谱先验（用于loss计算）
                    self.spectral_priors = spectral_priors
                else:
                    self.spectral_priors = None
                
                # === Decoder阶段 ===
                # 自定义decoder forward以集成BEAR
                if self.use_bear:
                    seg_outputs = self._decoder_with_bear(skips)
                else:
                    seg_outputs = self.decoder(skips)
                
                return seg_outputs
            
            def _decoder_with_bear(self, skips):
                """
                自定义decoder forward，在每个stage应用BEAR
                """
                # 获取decoder的stages
                lres_input = skips[-1]
                seg_outputs = []
                
                # Decoder逐stage处理
                for s in range(len(self.decoder.stages)):
                    # 上采样并与skip connection融合
                    x = self.decoder.transpconvs[s](lres_input)
                    x = torch.cat((x, skips[-(s+2)]), 1)
                    x = self.decoder.stages[s](x)
                    
                    # 应用BEAR（边界增强）
                    if s < len(self.bear_modules) and not isinstance(self.bear_modules[s], nn.Identity):
                        # 获取对应的不确定性图（如果有）
                        uncertainty = None
                        if self.use_sdar and hasattr(self, 'uncertainty_maps') and len(self.uncertainty_maps) > 0:
                            # 使用对应stage的不确定性图
                            idx = -(s+2)  # 对应skip的索引
                            if idx >= -len(self.uncertainty_maps):
                                unc_info = self.uncertainty_maps[idx]
                                if unc_info is not None and 'uncertainty' in unc_info:
                                    uncertainty = unc_info['uncertainty']
                        
                        # 应用BEAR
                        x, bear_info = self.bear_modules[s](x, uncertainty)
                        
                        # 保存边界信息（验证/推理时）
                        if not self.training and bear_info is not None and 'boundary_map' in bear_info:
                            self.boundary_maps.append(bear_info['boundary_map'])
                    
                    # Deep supervision输出（与标准decoder一致）
                    if self.deep_supervision:
                        seg_outputs.append(self.decoder.seg_layers[s](x))
                    elif s == (len(self.decoder.stages) - 1):
                        seg_outputs.append(self.decoder.seg_layers[-1](x))
                    
                    lres_input = x
                
                # 反转输出列表（从高分辨率到低分辨率）
                seg_outputs = seg_outputs[::-1]
                
                return seg_outputs
            
            def compute_conv_feature_map_size(self, input_size):
                return (self.encoder.compute_conv_feature_map_size(input_size) +
                       self.decoder.compute_conv_feature_map_size(input_size))
        
        network = HSIUNet()
        
        print("\n" + "="*80)
        print("✅ Network Built Successfully!")
        print("="*80 + "\n")
        
        return network


# ================================================================================
# 变体Trainer：针对val_loss波动优化
# ================================================================================

class nnUNetTrainerHSI_StableLR(nnUNetTrainerHSI):
    """
    稳定学习率版本 - 解决val_loss波动问题
    
    改进：
    1. 降低初始学习率：0.0001 (原0.0003)
    2. SSCA已内置Dropout=0.15（减少过拟合）
    3. 更平滑的学习率调度
    
    适用场景：
    - val_loss波动大
    - 验证集较小
    - 需要更稳定的收敛
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 降低初始学习率（更稳定）
        self.initial_lr = 3e-4  # 从3e-4降到1e-4
        
        # 增加SSCA dropout（进一步稳定）
        self.ssca_dropout_rate = 0.2  # 从0.15增加到0.2
        
        print("\n" + "="*80)
        print("🔧 nnUNetTrainerHSI_StableLR Configuration")
        print("="*80)
        print(f"Initial LR: {self.initial_lr} (↓ from 3e-4)")
        print(f"SSCA Dropout: {self.ssca_dropout_rate} (↑ from 0.15)")
        print("Purpose: Reduce val_loss oscillation")
        print("="*80 + "\n")
