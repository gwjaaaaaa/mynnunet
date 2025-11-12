import numpy as np
import torch
from scipy.io import loadmat

class StatisticalBandSelector:
    def __init__(self, data_file, label_file, num_bands=18):
        self.data_file = data_file
        self.label_file = label_file
        self.num_bands = num_bands
        self.statistical_weights = None
        self.selected_bands = None
        
    def compute_statistical_weights(self):
        # 加载数据
        data_dict = loadmat(self.data_file)
        label_dict = loadmat(self.label_file)
        
        data_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        label_keys = [k for k in label_dict.keys() if not k.startswith('__')]
        
        data = data_dict[data_keys[0]]  # (H, W, 60)
        label = label_dict[label_keys[0]]  # (H, W)
        
        # 重塑数据
        data_reshaped = data.reshape(-1, data.shape[-1])  # (H*W, 60)
        label_flat = label.flatten()  # (H*W,)
        
        # 计算统计差异
        normal_mask = label_flat == 0
        cancer_mask = label_flat == 1
        
        normal_avg = data_reshaped[normal_mask].mean(axis=0)  # (60,)
        cancer_avg = data_reshaped[cancer_mask].mean(axis=0)  # (60,)
        
        # 计算绝对差异
        diff_spectrum = np.abs(cancer_avg - normal_avg)
        
        # 选择前num_bands个波段
        self.selected_bands = np.argsort(diff_spectrum)[-self.num_bands:]
        self.statistical_weights = diff_spectrum
        
        return self.statistical_weights, self.selected_bands
