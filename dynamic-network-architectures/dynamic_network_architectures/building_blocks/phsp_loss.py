"""
Spectral Prior Smoothness Loss for PHSP

光谱先验平滑性约束损失

核心思想：
相邻波段的重要性权重应该平滑过渡，不应该剧烈跳变。
这是基于光谱物理特性的约束：光谱响应是连续的。

理论贡献：
- 将领域知识（光谱连续性）融入深度学习
- 提升光谱先验的物理可解释性
- 零参数开销，仅通过loss引导

Author: AI Assistant
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpectralPriorSmoothLoss(nn.Module):
    """
    光谱先验平滑性损失
    
    计算相邻波段权重的差异，鼓励平滑过渡
    """
    
    def __init__(self, alpha: float = 0.001, order: int = 1):
        """
        Args:
            alpha: 损失权重系数（默认0.001，很小，不影响主损失）
            order: 平滑阶数
                   1 = 一阶差分（相邻差异）
                   2 = 二阶差分（曲率）
        """
        super().__init__()
        self.alpha = alpha
        self.order = order
        
        print(f"\nSpectralPriorSmoothLoss initialized:")
        print(f"  Alpha: {alpha}")
        print(f"  Order: {order} ({'1st-order (adjacent diff)' if order == 1 else '2nd-order (curvature)'})")
    
    def forward(self, spectral_priors: List[torch.Tensor]) -> torch.Tensor:
        """
        计算光谱先验的平滑性损失
        
        Args:
            spectral_priors: List of (spectral_dim,), 各stage的光谱先验
        
        Returns:
            smooth_loss: scalar, 平滑性损失
        """
        if len(spectral_priors) == 0:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        smooth_loss = 0.0
        
        for prior in spectral_priors:
            if self.order == 1:
                # 一阶差分：|prior[i+1] - prior[i]|
                diff = prior[1:] - prior[:-1]  # (spectral_dim-1,)
                smooth_loss += (diff ** 2).mean()
                
            elif self.order == 2:
                # 二阶差分：|prior[i+1] - 2*prior[i] + prior[i-1]|
                # 这衡量曲率，鼓励线性变化
                if len(prior) > 2:
                    second_diff = prior[2:] - 2 * prior[1:-1] + prior[:-2]  # (spectral_dim-2,)
                    smooth_loss += (second_diff ** 2).mean()
        
        # 平均所有stage
        smooth_loss = smooth_loss / len(spectral_priors)
        
        # 应用权重系数
        smooth_loss = smooth_loss * self.alpha
        
        return smooth_loss


class SpectralPriorConsistencyLoss(nn.Module):
    """
    光谱先验一致性损失（可选）
    
    鼓励不同stage的光谱先验保持一定的一致性
    """
    
    def __init__(self, alpha: float = 0.0005):
        """
        Args:
            alpha: 损失权重系数
        """
        super().__init__()
        self.alpha = alpha
        
        print(f"\nSpectralPriorConsistencyLoss initialized:")
        print(f"  Alpha: {alpha}")
    
    def forward(self, spectral_priors: List[torch.Tensor]) -> torch.Tensor:
        """
        计算跨stage光谱先验的一致性损失
        
        Args:
            spectral_priors: List of (spectral_dim,), 各stage的光谱先验
        
        Returns:
            consistency_loss: scalar
        """
        if len(spectral_priors) <= 1:
            return torch.tensor(0.0, device=spectral_priors[0].device)
        
        # 计算全局平均先验
        global_prior = torch.stack(spectral_priors).mean(dim=0)  # (spectral_dim,)
        
        # 计算每个stage与全局的差异
        consistency_loss = 0.0
        for prior in spectral_priors:
            consistency_loss += F.mse_loss(prior, global_prior)
        
        # 平均
        consistency_loss = consistency_loss / len(spectral_priors)
        
        # 应用权重
        consistency_loss = consistency_loss * self.alpha
        
        return consistency_loss


class PHSPCompositeLoss(nn.Module):
    """
    PHSP复合损失：分割损失 + 平滑损失 + 一致性损失
    
    这是一个便捷的包装器，整合所有PHSP相关的损失
    """
    
    def __init__(self, 
                 smooth_alpha: float = 0.001,
                 smooth_order: int = 1,
                 consistency_alpha: float = 0.0005,
                 use_consistency: bool = False):
        """
        Args:
            smooth_alpha: 平滑性损失权重
            smooth_order: 平滑阶数
            consistency_alpha: 一致性损失权重
            use_consistency: 是否使用一致性损失
        """
        super().__init__()
        
        self.smooth_loss = SpectralPriorSmoothLoss(alpha=smooth_alpha, order=smooth_order)
        self.use_consistency = use_consistency
        
        if use_consistency:
            self.consistency_loss = SpectralPriorConsistencyLoss(alpha=consistency_alpha)
        else:
            self.consistency_loss = None
        
        print(f"\nPHSPCompositeLoss initialized:")
        print(f"  Smooth Loss: ✓ (alpha={smooth_alpha})")
        print(f"  Consistency Loss: {'✓' if use_consistency else '✗'}")
    
    def forward(self, spectral_priors: List[torch.Tensor]) -> dict:
        """
        计算所有PHSP相关损失
        
        Args:
            spectral_priors: List of (spectral_dim,), 各stage的光谱先验
        
        Returns:
            losses: dict包含各项损失和总损失
                - 'smooth': 平滑性损失
                - 'consistency': 一致性损失（如果启用）
                - 'total': 总辅助损失
        """
        losses = {}
        
        # 平滑性损失
        smooth = self.smooth_loss(spectral_priors)
        losses['smooth'] = smooth
        
        # 一致性损失
        if self.use_consistency and self.consistency_loss is not None:
            consistency = self.consistency_loss(spectral_priors)
            losses['consistency'] = consistency
        else:
            losses['consistency'] = torch.tensor(0.0, device=smooth.device)
        
        # 总辅助损失
        losses['total'] = losses['smooth'] + losses['consistency']
        
        return losses


# === 测试代码 ===
if __name__ == "__main__":
    print("="*80)
    print("Testing PHSP Loss Functions")
    print("="*80)
    
    # 模拟光谱先验
    spectral_priors = [
        torch.randn(60).abs(),  # Stage 0
        torch.randn(60).abs(),  # Stage 1
        torch.randn(60).abs(),  # Stage 2
        torch.randn(60).abs(),  # Stage 3
        torch.randn(60).abs(),  # Stage 4
        torch.randn(60).abs(),  # Stage 5
        torch.randn(60).abs(),  # Stage 6
    ]
    
    # 测试平滑性损失
    print("\n[1/3] Testing SpectralPriorSmoothLoss (1st order)...")
    smooth_loss_1st = SpectralPriorSmoothLoss(alpha=0.001, order=1)
    loss_1st = smooth_loss_1st(spectral_priors)
    print(f"  Loss value: {loss_1st.item():.6f}")
    
    print("\n[2/3] Testing SpectralPriorSmoothLoss (2nd order)...")
    smooth_loss_2nd = SpectralPriorSmoothLoss(alpha=0.001, order=2)
    loss_2nd = smooth_loss_2nd(spectral_priors)
    print(f"  Loss value: {loss_2nd.item():.6f}")
    
    # 测试一致性损失
    print("\n[3/3] Testing SpectralPriorConsistencyLoss...")
    consistency_loss = SpectralPriorConsistencyLoss(alpha=0.0005)
    loss_cons = consistency_loss(spectral_priors)
    print(f"  Loss value: {loss_cons.item():.6f}")
    
    # 测试复合损失
    print("\n[4/4] Testing PHSPCompositeLoss...")
    composite_loss = PHSPCompositeLoss(
        smooth_alpha=0.001,
        smooth_order=1,
        consistency_alpha=0.0005,
        use_consistency=True
    )
    losses = composite_loss(spectral_priors)
    print(f"  Smooth loss: {losses['smooth'].item():.6f}")
    print(f"  Consistency loss: {losses['consistency'].item():.6f}")
    print(f"  Total auxiliary loss: {losses['total'].item():.6f}")
    
    print("\n" + "="*80)
    print("✅ All PHSP Loss Tests Passed!")
    print("="*80)


