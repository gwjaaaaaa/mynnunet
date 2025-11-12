"""
计算光谱先验权重
基于三个统计指标：
1. 信息丰富度（Information Abundance）- 信息熵
2. 类区分度（Class Discriminability）- Fisher判别准则
3. 相关性强度（Decorrelation）- 1 - 皮尔逊相关系数

使用方法：
python compute_spectral_prior.py --dataset Dataset502_DGAGA --output spectral_prior_weights/
"""

import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_information_abundance(band_data):
    """
    计算信息丰富度（信息熵）
    
    参数:
        band_data: (N,) - 单个波段的所有像素值
    
    返回:
        entropy: float - 信息熵
    """
    # 使用直方图估计概率分布
    hist, bin_edges = np.histogram(band_data, bins=256, density=True)
    
    # 避免log(0)
    hist = hist[hist > 0]
    
    # 计算熵: H = -sum(p * log(p))
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return entropy


def compute_class_discriminability(band_data, labels):
    """
    计算类区分度（Fisher判别准则）
    
    参数:
        band_data: (N,) - 单个波段的所有像素值
        labels: (N,) - 对应的标签 (0=背景, 1=前景)
    
    返回:
        fisher_score: float - Fisher判别分数 (类间散度/类内散度)
    """
    # 分离前景和背景
    foreground = band_data[labels == 1]
    background = band_data[labels == 0]
    
    if len(foreground) == 0 or len(background) == 0:
        return 0.0
    
    # 计算均值
    mu_fg = np.mean(foreground)
    mu_bg = np.mean(background)
    mu_total = np.mean(band_data)
    
    # 类间散度 (Between-class scatter)
    n_fg = len(foreground)
    n_bg = len(background)
    n_total = len(band_data)
    
    S_b = (n_fg * (mu_fg - mu_total)**2 + n_bg * (mu_bg - mu_total)**2) / n_total
    
    # 类内散度 (Within-class scatter)
    var_fg = np.var(foreground)
    var_bg = np.var(background)
    S_w = (n_fg * var_fg + n_bg * var_bg) / n_total
    
    # Fisher判别分数
    if S_w < 1e-10:
        return 0.0
    
    fisher_score = S_b / (S_w + 1e-10)
    
    return fisher_score


def compute_decorrelation(band_data, all_bands_data):
    """
    计算去相关强度
    
    参数:
        band_data: (N,) - 单个波段的数据
        all_bands_data: (N, n_bands) - 所有波段的数据
    
    返回:
        decorrelation: float - 平均去相关度 (1 - 平均相关系数)
    """
    n_bands = all_bands_data.shape[1]
    
    # 计算当前波段与其他波段的相关系数
    correlations = []
    for i in range(n_bands):
        if not np.array_equal(band_data, all_bands_data[:, i]):
            # 计算皮尔逊相关系数
            corr = np.corrcoef(band_data, all_bands_data[:, i])[0, 1]
            correlations.append(abs(corr))  # 取绝对值
    
    if len(correlations) == 0:
        return 1.0
    
    # 去相关度 = 1 - 平均相关系数
    avg_correlation = np.mean(correlations)
    decorrelation = 1.0 - avg_correlation
    
    return decorrelation


def adaptive_clustering(info_abundance, class_disc, decorrelation, k_range=(3, 9)):
    """
    自适应选择最佳聚类数K
    
    参数:
        info_abundance: (n_bands,) - 信息丰富度
        class_disc: (n_bands,) - 类区分度
        decorrelation: (n_bands,) - 去相关强度
        k_range: tuple - K的搜索范围
    
    返回:
        best_k: int - 最佳聚类数
        labels: (n_bands,) - 每个波段的聚类标签
        kmeans: KMeans对象
        silhouette_scores: list - 每个K的轮廓系数
    """
    n_bands = len(info_abundance)
    
    # 构造特征向量: [H_i, S_i, D_i]
    features = np.stack([info_abundance, class_disc, decorrelation], axis=1)  # (n_bands, 3)
    
    # 标准化特征（重要！确保三个指标在同一尺度）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 遍历不同的K值
    print(f"\n{'='*60}")
    print("自适应聚类数选择（基于轮廓系数）")
    print(f"{'='*60}")
    
    silhouette_scores = []
    k_min, k_max = k_range
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # 计算轮廓系数（-1到1，越接近1越好）
        score = silhouette_score(features_scaled, labels)
        silhouette_scores.append((k, score))
        
        print(f"K={k}: Silhouette Score={score:.4f}")
    
    # 选择轮廓系数最高的K
    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print(f"最佳聚类数: K={best_k} (Silhouette Score={best_score:.4f})")
    print(f"{'='*60}\n")
    
    # 用最佳K进行最终聚类
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_best = kmeans_best.fit_predict(features_scaled)
    
    return best_k, labels_best, kmeans_best, silhouette_scores


def compute_clustered_weights(info_abundance, class_disc, decorrelation, labels, 
                               alpha=0.33, beta=0.33, gamma=0.34,
                               individual_weight=0.7, cluster_weight=0.3):
    """
    基于聚类计算最终权重
    
    参数:
        info_abundance: (n_bands,) - 信息丰富度
        class_disc: (n_bands,) - 类区分度
        decorrelation: (n_bands,) - 去相关强度
        labels: (n_bands,) - 聚类标签
        alpha, beta, gamma: 三个指标的组合权重
        individual_weight: 个体权重的比例
        cluster_weight: 聚类平均权重的比例
    
    返回:
        final_weights: (n_bands,) - 最终的光谱先验权重
        cluster_info: dict - 聚类信息
    """
    n_bands = len(info_abundance)
    n_clusters = len(np.unique(labels))
    
    # 1. 归一化三个指标到[0, 1]
    def normalize(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max - arr_min < 1e-10:
            return np.ones_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    H_norm = normalize(info_abundance)
    S_norm = normalize(class_disc)
    D_norm = normalize(decorrelation)
    
    # 2. 计算每个波段的个体权重
    individual_weights = alpha * H_norm + beta * S_norm + gamma * D_norm
    
    # 3. 计算每个聚类的平均权重
    cluster_avg_weights = np.zeros(n_bands)
    cluster_info = {}
    
    for cluster_id in range(n_clusters):
        # 找到该聚类的所有波段
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # 该聚类的平均权重
        cluster_avg = np.mean(individual_weights[cluster_indices])
        cluster_avg_weights[cluster_indices] = cluster_avg
        
        # 保存聚类信息
        cluster_info[f"cluster_{cluster_id}"] = {
            "band_indices": cluster_indices.tolist(),
            "num_bands": len(cluster_indices),
            "avg_weight": float(cluster_avg),
            "avg_info_abundance": float(np.mean(H_norm[cluster_indices])),
            "avg_class_disc": float(np.mean(S_norm[cluster_indices])),
            "avg_decorrelation": float(np.mean(D_norm[cluster_indices]))
        }
    
    # 4. 最终权重 = 个体权重 + 聚类平均权重
    final_weights = individual_weight * individual_weights + cluster_weight * cluster_avg_weights
    
    # 5. 再次归一化到[0, 1]
    final_weights = normalize(final_weights)
    
    return final_weights, cluster_info


def load_nnunet_data(data_folder, num_samples=None):
    """
    加载nnUNet预处理的数据
    
    参数:
        data_folder: nnUNet预处理数据文件夹
        num_samples: 使用的样本数量（None=全部）
    
    返回:
        all_data: (N, n_bands) - 所有样本的所有像素
        all_labels: (N,) - 对应的标签
        n_bands: int - 波段数
    """
    import blosc2
    
    # 找到所有图像文件（不包括_seg后缀的）
    all_files = sorted(Path(data_folder).glob("*.b2nd"))
    image_files = [f for f in all_files if not f.stem.endswith('_seg')]
    
    if num_samples is not None:
        image_files = image_files[:num_samples]
    
    print(f"找到 {len(image_files)} 个样本文件")
    
    all_data_list = []
    all_labels_list = []
    band_counts = {}  # 记录每个波段数的文件数量
    
    for img_file in tqdm(image_files, desc="加载数据"):
        # 对应的seg文件
        seg_file = img_file.parent / f"{img_file.stem}_seg.b2nd"
        
        try:
            # 加载图像数据 (使用[:] 读取完整数据)
            image_array = blosc2.open(str(img_file))
            image = image_array[:]  # 读取完整数据
            
            # 加载分割数据
            if seg_file.exists():
                seg_array = blosc2.open(str(seg_file))
                seg = seg_array[:]  # 读取完整数据
            else:
                print(f"\n警告: 找不到分割文件 {seg_file}")
                continue
            
            # 检查形状
            # nnUNet v2格式: (1, H, W, D) 或 (C, H, W, D)
            if len(image.shape) == 4:
                # 如果第一维是1，说明是单通道，实际波段在最后一维
                if image.shape[0] == 1:
                    # (1, H, W, D) -> (D, H, W) -> reshape为 (H*W, D)
                    image = image[0]  # (H, W, D)
                    H, W, D = image.shape
                    C = D  # 波段数就是D
                    image_flat = image.reshape(-1, C)  # (H*W, D)
                else:
                    # (C, H, W, D) -> (H*W*D, C)
                    C, H, W, D = image.shape
                    image_flat = image.transpose(1, 2, 3, 0).reshape(-1, C)
                
                # 处理分割标签
                # 标签格式: (1, H, W, D) 但所有深度切片的标签应该相同
                # 我们只取第一个深度切片的标签
                if seg.shape[0] == 1:
                    # (1, H, W, D) -> (H, W, D) -> 只取第一个深度切片 -> (H, W)
                    seg_2d = seg[0, :, :, 0]  # (H, W)
                    seg_flat = seg_2d.reshape(-1)  # (H*W,)
                else:
                    # (1, H, W, D) 或其他格式
                    seg_flat = seg[0].reshape(H, W, -1)[:, :, 0].reshape(-1)  # (H*W,)
                
                # 记录波段数
                if C not in band_counts:
                    band_counts[C] = []
                band_counts[C].append((img_file.stem, image_flat, seg_flat))
            else:
                print(f"\n警告: 图像形状异常 {image.shape}，跳过")
                continue
            
        except Exception as e:
            print(f"\n警告: 加载文件 {img_file.stem} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(band_counts) == 0:
        raise ValueError("没有成功加载任何数据！")
    
    # 选择最常见的波段数
    most_common_bands = max(band_counts.keys(), key=lambda k: len(band_counts[k]))
    selected_files = band_counts[most_common_bands]
    
    print(f"\n波段数分布:")
    for bands, files in sorted(band_counts.items()):
        print(f"  {bands} 波段: {len(files)} 个文件")
    
    print(f"\n选择使用: {most_common_bands} 波段的数据 ({len(selected_files)} 个文件)")
    
    if len(band_counts) > 1:
        print(f"警告: 发现波段数不一致的文件，已自动过滤")
        for bands, files in band_counts.items():
            if bands != most_common_bands:
                print(f"  跳过的文件 ({bands}波段): {[f[0] for f in files]}")
    
    # 合并选定的样本
    all_data_list = [f[1] for f in selected_files]  # image_flat
    all_labels_list = [f[2] for f in selected_files]  # seg_flat
    
    all_data = np.concatenate(all_data_list, axis=0)  # (Total_N, C)
    all_labels = np.concatenate(all_labels_list, axis=0)  # (Total_N,)
    
    n_bands = most_common_bands
    
    print(f"数据形状: {all_data.shape}")
    print(f"标签形状: {all_labels.shape}")
    print(f"波段数: {n_bands}")
    print(f"前景像素: {np.sum(all_labels > 0)} ({np.sum(all_labels > 0) / len(all_labels) * 100:.2f}%)")
    
    return all_data, all_labels, n_bands


def compute_spectral_priors(all_data, all_labels, n_bands):
    """
    计算三个光谱先验权重
    
    参数:
        all_data: (N, n_bands) - 所有样本的像素数据
        all_labels: (N,) - 标签
        n_bands: int - 波段数
    
    返回:
        info_abundance: (n_bands,) - 信息丰富度
        class_disc: (n_bands,) - 类区分度
        decorrelation: (n_bands,) - 去相关强度
    """
    info_abundance = np.zeros(n_bands)
    class_disc = np.zeros(n_bands)
    decorrelation = np.zeros(n_bands)
    
    print("\n计算光谱先验权重...")
    
    for band_idx in tqdm(range(n_bands), desc="处理波段"):
        band_data = all_data[:, band_idx]
        
        # 1. 信息丰富度（信息熵）
        info_abundance[band_idx] = compute_information_abundance(band_data)
        
        # 2. 类区分度（Fisher判别）
        class_disc[band_idx] = compute_class_discriminability(band_data, all_labels)
        
        # 3. 去相关强度
        decorrelation[band_idx] = compute_decorrelation(band_data, all_data)
    
    return info_abundance, class_disc, decorrelation


def normalize_weights(weights):
    """归一化权重到 [0, 1]"""
    weights = weights - weights.min()
    weights = weights / (weights.max() + 1e-10)
    return weights


def main():
    parser = argparse.ArgumentParser(description='计算光谱先验权重（带自适应聚类）')
    parser.add_argument('--dataset', type=str, default='Dataset502_DGAGA',
                        help='数据集名称')
    parser.add_argument('--preprocessed_folder', type=str, 
                        default='/data/CXY/g/szy/data/nnUNet_preprocessed',
                        help='nnUNet预处理数据文件夹')
    parser.add_argument('--output', type=str, default='spectral_prior_weights',
                        help='输出文件夹')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='使用的样本数量（None=全部）')
    parser.add_argument('--config', type=str, default='nnUNetPlans_3d_fullres',
                        help='nnUNet配置')
    parser.add_argument('--k_min', type=int, default=3,
                        help='聚类数K的最小值')
    parser.add_argument('--k_max', type=int, default=9,
                        help='聚类数K的最大值')
    
    args = parser.parse_args()
    
    # 路径设置
    data_folder = os.path.join(args.preprocessed_folder, args.dataset, args.config)
    output_folder = args.output
    
    print("="*80)
    print("光谱先验权重计算（自适应聚类版本）")
    print("="*80)
    print(f"数据集: {args.dataset}")
    print(f"数据文件夹: {data_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"聚类数搜索范围: K={args.k_min}~{args.k_max}")
    print("="*80)
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. 加载数据
    print("\n[1/6] 加载nnUNet预处理数据...")
    all_data, all_labels, n_bands = load_nnunet_data(data_folder, args.num_samples)
    
    # 2. 计算三个先验权重
    print("\n[2/6] 计算光谱先验权重...")
    info_abundance, class_disc, decorrelation = compute_spectral_priors(
        all_data, all_labels, n_bands
    )
    
    # 3. 归一化
    print("\n[3/6] 归一化权重...")
    info_abundance_norm = normalize_weights(info_abundance)
    class_disc_norm = normalize_weights(class_disc)
    decorrelation_norm = normalize_weights(decorrelation)
    
    # 4. 自适应聚类
    print("\n[4/6] 自适应聚类...")
    best_k, labels, kmeans, silhouette_scores = adaptive_clustering(
        info_abundance_norm, class_disc_norm, decorrelation_norm,
        k_range=(args.k_min, args.k_max)
    )
    
    # 5. 计算聚类后的最终权重
    print("\n[5/6] 计算聚类加权后的最终权重...")
    final_weights, cluster_info = compute_clustered_weights(
        info_abundance_norm, class_disc_norm, decorrelation_norm, labels
    )
    
    # 6. 保存结果
    print("\n[6/6] 保存结果...")
    
    # 保存原始三个权重
    np.save(os.path.join(output_folder, 'info_abundance.npy'), info_abundance_norm)
    np.save(os.path.join(output_folder, 'class_disc.npy'), class_disc_norm)
    np.save(os.path.join(output_folder, 'decorrelation.npy'), decorrelation_norm)
    
    # 保存聚类标签
    np.save(os.path.join(output_folder, 'cluster_labels.npy'), labels)
    
    # 保存最终权重（聚类加权后）
    np.save(os.path.join(output_folder, 'spectral_prior_weights_final.npy'), final_weights)
    
    # 保存统计信息
    stats_dict = {
        'n_bands': int(n_bands),
        'n_pixels': int(len(all_labels)),
        'n_foreground': int(np.sum(all_labels > 0)),
        'best_k': int(best_k),
        'silhouette_scores': [(int(k), float(score)) for k, score in silhouette_scores],
        'info_abundance': {
            'mean': float(np.mean(info_abundance_norm)),
            'std': float(np.std(info_abundance_norm)),
            'min': float(np.min(info_abundance_norm)),
            'max': float(np.max(info_abundance_norm)),
        },
        'class_disc': {
            'mean': float(np.mean(class_disc_norm)),
            'std': float(np.std(class_disc_norm)),
            'min': float(np.min(class_disc_norm)),
            'max': float(np.max(class_disc_norm)),
        },
        'decorrelation': {
            'mean': float(np.mean(decorrelation_norm)),
            'std': float(np.std(decorrelation_norm)),
            'min': float(np.min(decorrelation_norm)),
            'max': float(np.max(decorrelation_norm)),
        },
        'final_weights': {
            'mean': float(np.mean(final_weights)),
            'std': float(np.std(final_weights)),
            'min': float(np.min(final_weights)),
            'max': float(np.max(final_weights)),
        },
        'cluster_info': cluster_info
    }
    
    with open(os.path.join(output_folder, 'statistics.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)
    
    # 可视化
    print("\n生成可视化...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 轮廓系数曲线
    ax1 = plt.subplot(3, 2, 1)
    ks, scores = zip(*silhouette_scores)
    ax1.plot(ks, scores, 'o-', linewidth=2, markersize=8)
    ax1.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Adaptive Clustering Selection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 三个原始权重对比
    ax2 = plt.subplot(3, 2, 2)
    x = np.arange(n_bands)
    width = 0.25
    ax2.bar(x - width, info_abundance_norm, width, label='Info Abundance', alpha=0.7)
    ax2.bar(x, class_disc_norm, width, label='Class Disc', alpha=0.7)
    ax2.bar(x + width, decorrelation_norm, width, label='Decorrelation', alpha=0.7)
    ax2.set_xlabel('Band Index', fontsize=12)
    ax2.set_ylabel('Normalized Weight', fontsize=12)
    ax2.set_title('Three Spectral Priors', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类结果可视化（按聚类着色）
    ax3 = plt.subplot(3, 2, 3)
    colors = plt.cm.tab10(labels)
    ax3.bar(range(n_bands), info_abundance_norm, color=colors, alpha=0.8)
    ax3.set_xlabel('Band Index', fontsize=12)
    ax3.set_ylabel('Info Abundance', fontsize=12)
    ax3.set_title(f'Clustering Results (K={best_k})', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 每个聚类的波段分布
    ax4 = plt.subplot(3, 2, 4)
    cluster_counts = [len(cluster_info[f'cluster_{i}']['band_indices']) for i in range(best_k)]
    ax4.bar(range(best_k), cluster_counts, color=plt.cm.tab10(range(best_k)), alpha=0.8)
    ax4.set_xlabel('Cluster ID', fontsize=12)
    ax4.set_ylabel('Number of Bands', fontsize=12)
    ax4.set_title('Bands per Cluster', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 最终权重（聚类加权后）
    ax5 = plt.subplot(3, 2, 5)
    ax5.bar(range(n_bands), final_weights, color=colors, alpha=0.8)
    ax5.set_xlabel('Band Index', fontsize=12)
    ax5.set_ylabel('Final Weight', fontsize=12)
    ax5.set_title('Final Spectral Prior Weights (Clustered)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 聚类特征空间（3D散点图的2D投影）
    ax6 = plt.subplot(3, 2, 6)
    for i in range(best_k):
        cluster_mask = labels == i
        ax6.scatter(info_abundance_norm[cluster_mask], 
                   class_disc_norm[cluster_mask],
                   label=f'Cluster {i}', alpha=0.7, s=60)
    ax6.set_xlabel('Info Abundance', fontsize=12)
    ax6.set_ylabel('Class Discriminability', fontsize=12)
    ax6.set_title('Feature Space (2D Projection)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'spectral_clustering_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ 可视化已保存: {output_folder}/spectral_clustering_analysis.png")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("计算完成！")
    print("="*80)
    print(f"\n保存位置: {output_folder}/")
    print(f"  - info_abundance.npy                (信息丰富度)")
    print(f"  - class_disc.npy                    (类区分度)")
    print(f"  - decorrelation.npy                 (去相关强度)")
    print(f"  - cluster_labels.npy                (聚类标签)")
    print(f"  - spectral_prior_weights_final.npy  (最终权重)")
    print(f"  - statistics.json                   (统计信息)")
    print(f"  - spectral_clustering_analysis.png  (可视化)")
    
    print(f"\n波段数: {n_bands}")
    print(f"最佳聚类数: K={best_k}")
    
    print(f"\n聚类分布:")
    for i in range(best_k):
        info = cluster_info[f'cluster_{i}']
        print(f"  Cluster {i}: {info['num_bands']}个波段 - {info['band_indices'][:5]}{'...' if info['num_bands'] > 5 else ''}")
    
    print(f"\n最终权重统计:")
    print(f"  范围: [{stats_dict['final_weights']['min']:.4f}, {stats_dict['final_weights']['max']:.4f}]")
    print(f"  均值: {stats_dict['final_weights']['mean']:.4f} ± {stats_dict['final_weights']['std']:.4f}")
    
    print("\n" + "="*80)
    print("✓ 全部完成！")
    print("="*80)


if __name__ == '__main__':
    main()

