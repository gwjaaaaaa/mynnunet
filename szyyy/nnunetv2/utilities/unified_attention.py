import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedAttention(nn.Module):
    def __init__(self, channels, reduction=16, attention_type='full'):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.attention_type = attention_type
        
        # 根据类型选择分支
        if attention_type == 'full':
            self.global_branch = GlobalAttentionBranch(channels, reduction)
            self.depth_branch = DepthAttentionBranch(channels, reduction)
            self.spatial_branch = SpatialAttentionBranch(channels, reduction)
            self.fusion = AttentionFusion(channels, 3)
        elif attention_type == 'standard':
            self.global_branch = GlobalAttentionBranch(channels, reduction)
            self.depth_branch = DepthAttentionBranch(channels, reduction)
            self.fusion = AttentionFusion(channels, 2)
        else:  # 'light'
            self.global_branch = GlobalAttentionBranch(channels, reduction)
            self.fusion = AttentionFusion(channels, 1)
    
    def forward(self, x):
        # 并行计算各个分支
        if self.attention_type == 'full':
            global_att = self.global_branch(x)
            depth_att = self.depth_branch(x)
            spatial_att = self.spatial_branch(x)
            attention = self.fusion(global_att, depth_att, spatial_att)
        elif self.attention_type == 'standard':
            global_att = self.global_branch(x)
            depth_att = self.depth_branch(x)
            attention = self.fusion(global_att, depth_att)
        else:  # 'light'
            global_att = self.global_branch(x)
            attention = self.fusion(global_att)
        
        return x * attention

class GlobalAttentionBranch(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return y

class DepthAttentionBranch(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.depth_proj = nn.Conv3d(1, channels//reduction, 1, bias=False)
        self.feat_proj = nn.Conv3d(channels, channels//reduction, 1, bias=False)
        self.fuse = nn.Conv3d(channels//reduction, 1, 1, bias=False)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        depth = torch.mean(x, dim=1, keepdim=True)
        a = self.depth_proj(depth) + self.feat_proj(x)
        w = self.gate(self.fuse(a))
        return w

class SpatialAttentionBranch(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.spatial_proj = nn.Conv3d(channels, channels//reduction, 1, bias=False)
        self.fuse = nn.Conv3d(channels//reduction, 1, 1, bias=False)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        spatial_feat = self.spatial_proj(x)
        w = self.gate(self.fuse(spatial_feat))
        return w

class AttentionFusion(nn.Module):
    def __init__(self, channels, num_branches):
        super().__init__()
        self.num_branches = num_branches
        if num_branches > 1:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, channels//4, 1),
                nn.ReLU(),
                nn.Conv3d(channels//4, num_branches, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(self, *attentions):
        if self.num_branches == 1:
            return attentions[0]
        
        # 学习各分支的重要性权重
        gate_weights = self.gate(attentions[0])
        
        # 加权融合
        fused_attention = torch.zeros_like(attentions[0])
        for i, att in enumerate(attentions):
            fused_attention += gate_weights[:, i:i+1] * att
        
        return fused_attention
