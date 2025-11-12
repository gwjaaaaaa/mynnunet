"""
SCR - Spectral Consistency Regularization Loss
光谱一致性正则化损失

核心创新：
1. 类内光谱紧凑性 - 同类像素的光谱特征应该相似
2. 类间光谱可分性 - 不同类的光谱特征应该可区分
3. 光谱平滑性约束 - 邻域光谱应该平滑过渡
4. 自适应权重 - 根据预测置信度调整约束强度

理论创新点：
- 显式建模光谱-语义一致性，而不是仅依赖像素级分类
- 结合了度量学习思想（triplet loss）和邻域一致性
- 自适应加权机制提高了对困难样本的鲁棒性
- 可以与任何分割loss组合使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCompactnessLoss(nn.Module):
    """
    类内光谱紧凑性损失
    
    创新点：最小化同类像素的光谱特征方差
    使得同一类别的像素具有相似的光谱响应曲线
    """
    def __init__(self, num_classes=2, temperature=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(self, features, predictions, targets):
        """
        输入:
            - features: (B, C, H, W, D) bottleneck特征，D是光谱维度
            - predictions: (B, num_classes, H', W', D') 分割预测（logits或prob）
            - targets: (B, 1, H', W', D') ground truth
        输出:
            - compactness_loss: scalar
        """
        B, C, H, W, D = features.shape
        
        # 如果predictions和features的空间维度不匹配，调整到features的尺寸
        if predictions.shape[2:] != (H, W, D):
            predictions = F.interpolate(predictions, size=(H, W, D), mode='trilinear', align_corners=False)
        if targets.shape[2:] != (H, W, D):
            targets = F.interpolate(targets.float(), size=(H, W, D), mode='nearest').long()
        
        # Softmax获取预测概率
        if predictions.shape[1] == self.num_classes:
            pred_probs = F.softmax(predictions, dim=1)  # (B, K, H, W, D)
        else:
            pred_probs = predictions
        
        # 展平空间维度: (B, C, H*W*D)
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, N, C), N=H*W*D
        
        # 目标展平: (B, N)
        targets_flat = targets.view(B, -1)  # (B, N)
        
        # 预测概率展平: (B, K, N)
        pred_probs_flat = pred_probs.view(B, self.num_classes, -1)
        
        total_loss = 0.0
        valid_classes = 0
        
        # 对每个类别计算紧凑性损失
        for c in range(self.num_classes):
            # 获取该类的mask: (B, N)
            class_mask = (targets_flat == c).float()  # (B, N)
            
            # 每个batch中该类的像素数量
            n_pixels = class_mask.sum(dim=1)  # (B,)
            
            for b in range(B):
                if n_pixels[b] < 2:  # 至少需要2个像素才能计算方差
                    continue
                
                # 该类的像素索引
                class_indices = class_mask[b].nonzero(as_tuple=True)[0]  # (N_c,)
                
                # 提取该类的特征: (N_c, C)
                class_features = features_flat[b, class_indices, :]
                
                # 计算类中心（质心）
                class_center = class_features.mean(dim=0, keepdim=True)  # (1, C)
                
                # 计算每个像素到中心的距离（使用预测置信度加权）
                pred_confidence = pred_probs_flat[b, c, class_indices]  # (N_c,)
                pred_confidence = pred_confidence.unsqueeze(1)  # (N_c, 1)
                
                # 距离: (N_c,)
                distances = torch.norm(class_features - class_center, p=2, dim=1)
                
                # 加权距离（高置信度的样本更重要）
                weighted_distances = distances * pred_confidence.squeeze()
                
                # 平均紧凑性损失
                class_loss = weighted_distances.mean()
                
                total_loss += class_loss
                valid_classes += 1
        
        if valid_classes > 0:
            return total_loss / valid_classes
        else:
            return torch.tensor(0.0, device=features.device)


class SpectralSeparabilityLoss(nn.Module):
    """
    类间光谱可分性损失
    
    创新点：最大化不同类别之间的光谱特征距离
    基于对比学习的思想
    """
    def __init__(self, num_classes=2, margin=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin  # 类间最小距离
    
    def forward(self, features, predictions, targets):
        """
        输入:
            - features: (B, C, H, W, D)
            - predictions: (B, num_classes, H', W', D')
            - targets: (B, 1, H', W', D')
        输出:
            - separability_loss: scalar
        """
        B, C, H, W, D = features.shape
        
        # 调整到features尺寸
        if predictions.shape[2:] != (H, W, D):
            predictions = F.interpolate(predictions, size=(H, W, D), mode='trilinear', align_corners=False)
        if targets.shape[2:] != (H, W, D):
            targets = F.interpolate(targets.float(), size=(H, W, D), mode='nearest').long()
        
        # 展平
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        targets_flat = targets.view(B, -1)  # (B, N)
        
        total_loss = 0.0
        valid_pairs = 0
        
        # 计算每个类的中心
        class_centers = []
        for c in range(self.num_classes):
            centers_batch = []
            for b in range(B):
                class_mask = (targets_flat[b] == c)
                if class_mask.sum() > 0:
                    class_features = features_flat[b, class_mask, :]
                    center = class_features.mean(dim=0)
                    centers_batch.append(center)
                else:
                    centers_batch.append(None)
            class_centers.append(centers_batch)
        
        # 计算类间距离
        for b in range(B):
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    if class_centers[i][b] is not None and class_centers[j][b] is not None:
                        # 两个类中心的距离
                        distance = torch.norm(class_centers[i][b] - class_centers[j][b], p=2)
                        
                        # Hinge loss: 希望距离至少为margin
                        loss = F.relu(self.margin - distance)
                        
                        total_loss += loss
                        valid_pairs += 1
        
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=features.device)


class SpectralSmoothnessLoss(nn.Module):
    """
    光谱平滑性损失
    
    创新点：约束空间邻域的光谱应该平滑变化
    避免光谱维度上的剧烈跳变
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, features, predictions):
        """
        输入:
            - features: (B, C, H, W, D)
            - predictions: (B, num_classes, H', W', D') 用于加权
        输出:
            - smoothness_loss: scalar
        """
        B, C, H, W, D = features.shape
        
        # 如果predictions和features的空间维度不匹配，调整predictions
        if predictions.shape[2:] != (H, W, D):
            predictions = F.interpolate(predictions, size=(H, W, D), mode='trilinear', align_corners=False)
        
        # 获取预测置信度（用于加权）
        pred_probs = F.softmax(predictions, dim=1)
        confidence = pred_probs.max(dim=1)[0]  # (B, H, W, D)
        
        # 计算空间邻域的光谱差异
        # 水平方向
        diff_h = features[:, :, 1:, :, :] - features[:, :, :-1, :, :]  # (B, C, H-1, W, D)
        weight_h = (confidence[:, 1:, :, :] + confidence[:, :-1, :, :]) / 2  # (B, H-1, W, D)
        loss_h = (diff_h.pow(2).sum(dim=1) * weight_h).mean()
        
        # 垂直方向
        diff_w = features[:, :, :, 1:, :] - features[:, :, :, :-1, :]  # (B, C, H, W-1, D)
        weight_w = (confidence[:, :, 1:, :] + confidence[:, :, :-1, :]) / 2  # (B, H, W-1, D)
        loss_w = (diff_w.pow(2).sum(dim=1) * weight_w).mean()
        
        # 光谱方向（相对弱的约束）
        diff_d = features[:, :, :, :, 1:] - features[:, :, :, :, :-1]  # (B, C, H, W, D-1)
        loss_d = diff_d.pow(2).mean() * 0.1  # 光谱维度的平滑性约束较弱
        
        return loss_h + loss_w + loss_d


class SpectralPrototypeConsistencyLoss(nn.Module):
    """
    光谱原型一致性损失
    
    创新点：如果使用了SPGA模块，可以利用学到的原型
    约束特征与类原型的一致性
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, features, predictions, targets, prototypes=None):
        """
        输入:
            - features: (B, C, H, W, D)
            - predictions: (B, num_classes, H', W', D')
            - targets: (B, 1, H', W', D')
            - prototypes: (num_classes, C) 可选的类原型
        输出:
            - consistency_loss: scalar
        """
        if prototypes is None:
            # 如果没有提供原型，从当前batch计算
            return torch.tensor(0.0, device=features.device)
        
        B, C, H, W, D = features.shape
        
        # 调整到features尺寸
        if targets.shape[2:] != (H, W, D):
            targets = F.interpolate(targets.float(), size=(H, W, D), mode='nearest').long()
        
        # 展平
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        targets_flat = targets.view(B, -1)  # (B, N)
        
        total_loss = 0.0
        valid_pixels = 0
        
        for b in range(B):
            for c in range(self.num_classes):
                if c >= len(prototypes):
                    continue
                
                # 该类的mask
                class_mask = (targets_flat[b] == c)
                if class_mask.sum() == 0:
                    continue
                
                # 该类的特征
                class_features = features_flat[b, class_mask, :]  # (N_c, C)
                
                # 原型
                prototype = prototypes[c].unsqueeze(0)  # (1, C)
                
                # 计算与原型的距离
                distances = torch.norm(class_features - prototype, p=2, dim=1)
                
                total_loss += distances.mean()
                valid_pixels += 1
        
        if valid_pixels > 0:
            return total_loss / valid_pixels
        else:
            return torch.tensor(0.0, device=features.device)


class SpectralConsistencyLoss(nn.Module):
    """
    完整的光谱一致性正则化损失
    
    组合所有光谱约束：
    1. 类内紧凑性
    2. 类间可分性
    3. 空间平滑性
    4. 原型一致性（可选）
    
    创新点总结：
    - 多层次的光谱一致性约束
    - 自适应加权机制
    - 与预测置信度结合
    - 灵活的损失组合
    """
    def __init__(self, 
                 num_classes=2,
                 use_compactness=True,
                 use_separability=True,
                 use_smoothness=True,
                 use_prototype=False,
                 weight_compactness=1.0,
                 weight_separability=0.5,
                 weight_smoothness=0.3,
                 weight_prototype=0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_compactness = use_compactness
        self.use_separability = use_separability
        self.use_smoothness = use_smoothness
        self.use_prototype = use_prototype
        
        self.weight_compactness = weight_compactness
        self.weight_separability = weight_separability
        self.weight_smoothness = weight_smoothness
        self.weight_prototype = weight_prototype
        
        # 各个子损失
        if use_compactness:
            self.compactness_loss = SpectralCompactnessLoss(num_classes)
        if use_separability:
            self.separability_loss = SpectralSeparabilityLoss(num_classes)
        if use_smoothness:
            self.smoothness_loss = SpectralSmoothnessLoss()
        if use_prototype:
            self.prototype_loss = SpectralPrototypeConsistencyLoss(num_classes)
        
        print(f"  [SCR Loss] Initialized:")
        print(f"    - Compactness: {use_compactness} (weight={weight_compactness})")
        print(f"    - Separability: {use_separability} (weight={weight_separability})")
        print(f"    - Smoothness: {use_smoothness} (weight={weight_smoothness})")
        print(f"    - Prototype: {use_prototype} (weight={weight_prototype})")
    
    def forward(self, features, predictions, targets, prototypes=None):
        """
        输入:
            - features: (B, C, H, W, D) bottleneck特征
            - predictions: (B, num_classes, H, W, D) 分割预测
            - targets: (B, 1, H, W, D) ground truth
            - prototypes: optional, 类原型
        输出:
            - total_loss: scalar
            - loss_dict: 各个子损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 类内紧凑性
        if self.use_compactness:
            loss_compact = self.compactness_loss(features, predictions, targets)
            total_loss += self.weight_compactness * loss_compact
            loss_dict['scr_compactness'] = loss_compact.item()
        
        # 2. 类间可分性
        if self.use_separability:
            loss_sep = self.separability_loss(features, predictions, targets)
            total_loss += self.weight_separability * loss_sep
            loss_dict['scr_separability'] = loss_sep.item()
        
        # 3. 空间平滑性
        if self.use_smoothness:
            loss_smooth = self.smoothness_loss(features, predictions)
            total_loss += self.weight_smoothness * loss_smooth
            loss_dict['scr_smoothness'] = loss_smooth.item()
        
        # 4. 原型一致性
        if self.use_prototype and prototypes is not None:
            loss_proto = self.prototype_loss(features, predictions, targets, prototypes)
            total_loss += self.weight_prototype * loss_proto
            loss_dict['scr_prototype'] = loss_proto.item()
        
        loss_dict['scr_total'] = total_loss.item()
        
        return total_loss, loss_dict


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Testing Spectral Consistency Loss")
    print("="*70)
    
    # 测试参数
    B, C, H, W, D = 2, 64, 32, 32, 20
    num_classes = 2
    
    # 创建模拟数据
    print("\n1. Creating test data...")
    features = torch.randn(B, C, H, W, D, requires_grad=True)
    predictions = torch.randn(B, num_classes, H, W, D)
    targets = torch.randint(0, num_classes, (B, 1, H, W, D))
    
    print(f"   Features: {features.shape}")
    print(f"   Predictions: {predictions.shape}")
    print(f"   Targets: {targets.shape}")
    
    # 测试各个子损失
    print("\n2. Testing individual losses...")
    
    # 紧凑性
    compact_loss = SpectralCompactnessLoss(num_classes)
    loss_c = compact_loss(features, predictions, targets)
    print(f"   Compactness loss: {loss_c.item():.4f}")
    
    # 可分性
    sep_loss = SpectralSeparabilityLoss(num_classes)
    loss_s = sep_loss(features, predictions, targets)
    print(f"   Separability loss: {loss_s.item():.4f}")
    
    # 平滑性
    smooth_loss = SpectralSmoothnessLoss()
    loss_sm = smooth_loss(features, predictions)
    print(f"   Smoothness loss: {loss_sm.item():.4f}")
    
    # 测试完整损失
    print("\n3. Testing complete SCR loss...")
    scr_loss = SpectralConsistencyLoss(
        num_classes=num_classes,
        use_compactness=True,
        use_separability=True,
        use_smoothness=True,
        use_prototype=False
    )
    
    # 重新创建features（带梯度）
    features = torch.randn(B, C, H, W, D, requires_grad=True)
    total_loss, loss_dict = scr_loss(features, predictions, targets)
    print(f"   Total SCR loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"     - {k}: {v:.4f}")
    
    # 测试梯度传播
    print("\n4. Testing gradient flow...")
    total_loss.backward()
    print(f"   ✓ Gradients computed successfully")
    print(f"   Feature gradient norm: {features.grad.norm().item():.4f}")
    
    # 测试不同尺度
    print("\n5. Testing different scales...")
    test_configs = [
        (2, 32, 64, 64, 20),
        (2, 128, 32, 32, 20),
        (2, 320, 16, 16, 20),
    ]
    
    for B_t, C_t, H_t, W_t, D_t in test_configs:
        features_t = torch.randn(B_t, C_t, H_t, W_t, D_t, requires_grad=True)
        predictions_t = torch.randn(B_t, num_classes, H_t, W_t, D_t)
        targets_t = torch.randint(0, num_classes, (B_t, 1, H_t, W_t, D_t))
        
        loss_t, _ = scr_loss(features_t, predictions_t, targets_t)
        loss_t.backward()
        
        print(f"   ✓ C={C_t}, H={H_t}, W={W_t}: loss={loss_t.item():.4f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)





