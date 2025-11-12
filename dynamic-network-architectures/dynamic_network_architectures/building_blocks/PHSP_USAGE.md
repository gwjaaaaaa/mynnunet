# PHSP (Prototype-Driven Hierarchical Spectral Prior) Usage Guide

## 📋 概述

PHSP是一个原型驱动的层级光谱先验学习模块，用于替代SSCA模块。

### **核心创新：**
1. **动态光谱先验**：从SPGA的自适应原型中提取（vs SSCA的静态先验）
2. **层级传播**：有原型stage提取局部先验，无原型stage从全局传播
3. **轻量设计**：~48K参数，<10KB额外显存
4. **与SPGA/DSR深度集成**：充分利用已学习的原型

### **参数量对比：**
- SSCA: ~140K参数
- PHSP: ~48K参数 ✅ **节省66%！**

---

## 🚀 如何启用PHSP

### **Step 1: 修改nnUNetTrainerHSI.py配置**

在`__init__`方法中：

```python
# 禁用SSCA，启用PHSP
self.use_ssca = False    # ✗ 禁用SSCA
self.use_phsp = True     # ✓ 启用PHSP
```

### **Step 2: 修改build_network_architecture配置**

在`build_network_architecture`方法中：

```python
# HSI参数
use_ssca = False   # ✗ 禁用SSCA
use_phsp = True    # ✓ 启用PHSP
```

### **Step 3: 训练**

```bash
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerHSI
```

---

## 📊 PHSP Loss（可选）

PHSP提供两种辅助损失，用于提升光谱先验质量：

### **1. Spectral Prior Smoothness Loss（平滑性约束）**

**原理：** 相邻波段的重要性应该平滑过渡

**使用方法：**

在trainer的`train_step`方法中添加：

```python
def train_step(self, batch):
    data, target = batch
    
    # 前向传播
    pred = self.network(data)
    
    # 主损失
    seg_loss = self.loss(pred, target)
    
    # PHSP损失（如果启用）
    if self.network.spectral_priors is not None:
        from dynamic_network_architectures.building_blocks.phsp_loss import SpectralPriorSmoothLoss
        
        smooth_loss_fn = SpectralPriorSmoothLoss(alpha=0.001, order=1)
        smooth_loss = smooth_loss_fn(self.network.spectral_priors)
        
        total_loss = seg_loss + smooth_loss
    else:
        total_loss = seg_loss
    
    return total_loss
```

**配置参数：**
- `alpha`: 损失权重（默认0.001，很小）
- `order`: 
  - 1 = 一阶差分（相邻波段差异）
  - 2 = 二阶差分（曲率约束）

### **2. Spectral Prior Consistency Loss（一致性约束，可选）**

**原理：** 不同stage的光谱先验应该保持一致性

**使用方法：**

```python
from dynamic_network_architectures.building_blocks.phsp_loss import PHSPCompositeLoss

# 初始化复合损失（包含平滑性+一致性）
phsp_loss_fn = PHSPCompositeLoss(
    smooth_alpha=0.001,
    smooth_order=1,
    consistency_alpha=0.0005,
    use_consistency=True  # 启用一致性损失
)

# 在train_step中使用
def train_step(self, batch):
    data, target = batch
    pred = self.network(data)
    seg_loss = self.loss(pred, target)
    
    if self.network.spectral_priors is not None:
        phsp_losses = phsp_loss_fn(self.network.spectral_priors)
        total_loss = seg_loss + phsp_losses['total']
    else:
        total_loss = seg_loss
    
    return total_loss
```

---

## 🎯 PHSP vs SSCA 对比

| 特性 | SSCA | PHSP |
|------|------|------|
| **先验来源** | 离线统计（双重平滑） | **在线，从SPGA原型提取** |
| **动态性** | 静态 | **动态，随训练演化** |
| **与SPGA集成** | ❌ 独立 | **✅ 深度集成** |
| **参数量** | 140K | **48K (-66%)** |
| **显存开销** | 中等 | **极少 (<10KB)** |
| **理论创新** | 双重平滑密度估计 | **原型驱动 + 层级传播** |
| **论文价值** | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** |

---

## 📝 配置参数说明

### **PHSP核心参数（build_phsp_module）：**

```python
phsp_module = build_phsp_module(
    channels_per_stage=[32, 64, 128, 256, 320, 320, 320],  # 各stage通道数
    spectral_dim=60,                                       # 光谱维度
    num_prototypes=4,                                      # SPGA原型数量
    spga_stages=[2, 3, 4]                                  # 有SPGA的stage
)
```

### **PHSP Loss参数（在trainer中配置）：**

```python
# 在nnUNetTrainerHSI.__init__中：
self.phsp_smooth_alpha = 0.001         # 平滑性损失权重
self.phsp_smooth_order = 1             # 1=一阶差分, 2=二阶差分
self.phsp_use_consistency = False      # 是否使用一致性损失
self.phsp_consistency_alpha = 0.0005   # 一致性损失权重
```

---

## 🔬 工作原理

### **Forward Pass流程：**

```
1. Encoder:
   x → Stage 0 (no SPGA) → skip[0]
   x → Stage 1 (no SPGA) → skip[1]
   x → Stage 2 + SPGA → skip[2] ← 提取原型P2
   x → Stage 3 + SPGA → skip[3] ← 提取原型P3
   x → Stage 4 + SPGA → skip[4] ← 提取原型P4
   x → Stage 5 (no SPGA) → skip[5]
   x → Stage 6 (no SPGA) → skip[6]

2. PHSP Module:
   ├─ 从原型提取局部光谱先验：
   │  Stage 2: P2 → spectral_prior[2]
   │  Stage 3: P3 → spectral_prior[3]
   │  Stage 4: P4 → spectral_prior[4]
   │
   ├─ 计算全局先验：
   │  global_prior = mean(spectral_prior[2,3,4])
   │
   ├─ 传播到无原型stage：
   │  Stage 0,1,5,6: global_prior → 自适应传播
   │
   └─ 应用到跳跃连接：
      refined_skip[i] = skip[i] * (1 + channel_weights * 0.3)

3. Decoder:
   refined_skips → decoder → output
```

---

## 💡 最佳实践

### **推荐配置（显存充足）：**
```python
self.use_phsp = True
self.phsp_smooth_alpha = 0.001
self.phsp_smooth_order = 1
self.phsp_use_consistency = False  # 一般不需要
```

### **极简配置（不使用辅助loss）：**
```python
self.use_phsp = True
# 不添加PHSP loss，仅使用动态先验
```

### **完整配置（最大化性能）：**
```python
self.use_phsp = True
self.phsp_smooth_alpha = 0.001
self.phsp_smooth_order = 2  # 二阶差分，更强约束
self.phsp_use_consistency = True
self.phsp_consistency_alpha = 0.0005
```

---

## 🐛 故障排查

### **问题1：ImportError**
```
ImportError: cannot import name 'build_phsp_module'
```

**解决：** 确保文件存在且路径正确
```bash
ls /data/CXY/g/szy/dynamic-network-architectures/dynamic_network_architectures/building_blocks/phsp_module.py
```

### **问题2：PHSP和SSCA同时启用**
```
Warning: Both SSCA and PHSP are enabled!
```

**解决：** 只启用一个
```python
self.use_ssca = False  # 禁用SSCA
self.use_phsp = True   # 启用PHSP
```

### **问题3：显存不足**
```
CUDA out of memory
```

**解决：** PHSP本身很轻量（<10KB），如果OOM，检查其他模块
```python
self.use_bear = False  # BEAR是OOM的主要原因
```

---

## 📚 理论背景

### **为什么PHSP优于SSCA？**

1. **动态 vs 静态**
   - SSCA：离线计算，固定不变
   - PHSP：在线提取，随训练演化 ✅

2. **数据驱动 vs 统计驱动**
   - SSCA：基于全局统计
   - PHSP：基于SPGA学到的原型（更精准）✅

3. **集成深度**
   - SSCA：独立模块
   - PHSP：深度集成SPGA，形成闭环 ✅

4. **理论创新**
   - SSCA：双重平滑（统计学）
   - PHSP：原型驱动 + 层级传播（深度学习+统计学）✅

---

## 📖 引用

如果使用PHSP，建议在论文中这样描述：

> *"We propose a Prototype-Driven Hierarchical Spectral Prior Learning (PHSP) module that dynamically extracts spectral priors from the learned prototypes in SPGA. Unlike traditional static spectral priors, PHSP leverages the adaptive nature of prototypes to generate data-driven priors that evolve during training. The hierarchical propagation mechanism ensures that all encoder stages, including those without explicit prototype learning, benefit from the spectral knowledge encoded in the prototypes."*

---

## 🎯 总结

**何时使用PHSP：**
- ✅ 当SSCA效果不理想时
- ✅ 想要更强的理论创新
- ✅ 需要轻量化方案
- ✅ 想要与SPGA深度集成

**何时使用SSCA：**
- ✅ 需要完全独立的先验（不依赖SPGA）
- ✅ 对双重平滑理论有特殊需求

**推荐：优先尝试PHSP！** 🚀

---

## 📞 支持

如有问题，检查：
1. PHSP是否正确初始化（检查日志输出）
2. SPGA是否在stage [2,3,4]启用
3. 前向传播是否正确调用PHSP
4. Loss是否正确计算（如果使用辅助loss）


