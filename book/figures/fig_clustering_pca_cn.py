#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图5.3/5.4 聚类与降维：K-Means++、层次聚类、PCA/SVD降维过程、t-SNE
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_classification
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

# ========== 图5.3.1 K-Means 迭代过程详图 ==========
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
np.random.seed(42)
centers = [(1,3), (5,1), (9,3)]; X, _ = make_blobs(n_samples=150, centers=centers, cluster_std=0.8)
# 打乱
idx = np.random.permutation(len(X))
initial_centers = [X[idx[0]], X[idx[1]], X[idx[2]]]
colors = ['#3498DB', '#E74C3C', '#27AE60']
# 手动K-Means迭代
def kmeans_iter(X, centers, labels):
    new_centers = []
    for c in range(len(centers)):
        pts = X[labels==c]
        new_centers.append(pts.mean(axis=0) if len(pts)>0 else centers[c])
    new_labels = np.argmin([[np.linalg.norm(x-c) for c in new_centers] for x in X], axis=1)
    return np.array(new_centers), new_labels

centers_arr = np.array(initial_centers)
labels = np.argmin([[np.linalg.norm(x-c) for c in centers_arr] for x in X], axis=1)
iter_steps = [labels]
for _ in range(7):
    centers_arr, labels = kmeans_iter(X, centers_arr, labels)
    iter_steps.append(labels)

for ax, step in zip(axes.flat, range(8)):
    if step < len(iter_steps):
        lbls = iter_steps[step]
    for c in range(3):
        pts = X[lbls==c]
        ax.scatter(pts[:,0], pts[:,1], s=25, alpha=0.7, color=colors[c],
                   label=f'簇{c+1}' if step==0 else '')
        ax.scatter(centers_arr[c,0], centers_arr[c,1], s=200, marker='X',
                   color=colors[c], edgecolors='black', linewidth=1.5, zorder=20)
    if step == 0:
        ax.set_title(f'初始化\n（随机选中心点）', fontsize=11, fontweight='bold')
    elif step == len(iter_steps)-1:
        ax.set_title(f'迭代 {step} 次\n（收敛）', fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'迭代 {step} 次', fontsize=11, fontweight='bold')
    ax.set_xlabel('特征1'); ax.set_ylabel('特征2')
    ax.legend(fontsize=8, loc='upper right')
savefig(fig, 'fig_5_3_1_kmeans_iter_detail.png')

# ========== 图5.3.2 层次聚类树状图 ==========
fig, ax = plt.subplots(figsize=(13, 6))
from scipy.cluster.hierarchy import dendrogram, linkage
np.random.seed(0)
X_demo = np.random.randn(20, 2)
Z = linkage(X_demo, method='ward')
dendrogram(Z, ax=ax, leaf_font_size=10,
           labels=[f'样本{i+1}' for i in range(20)],
           color_threshold=3)
ax.set_title('层次聚类树状图（Dendrogram）\n'
             '从上到下切割：簇数越多，粒度越细', fontsize=13, fontweight='bold')
ax.set_xlabel('样本', fontsize=11); ax.set_ylabel('距离（Ward法）', fontsize=11)
ax.axhline(3, color='#E74C3C', lw=2, ls='--', label='切割线（3个簇）')
ax.legend(fontsize=11)
savefig(fig, 'fig_5_3_2_hierarchical_dendrogram.png')

# ========== 图5.3.3 轮廓系数选K ==========
fig, ax = plt.subplots(figsize=(10, 5))
from sklearn.metrics import silhouette_score
K_range = range(2, 11)
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))
ax.plot(list(K_range), sil_scores, 'o-', color='#3498DB', lw=2.5, ms=8)
best_k = list(K_range)[np.argmax(sil_scores)]
ax.axvline(best_k, color='#E74C3C', lw=2, ls='--', label=f'最优 K={best_k}')
ax.scatter([best_k], [max(sil_scores)], s=200, color='#E74C3C', zorder=10, marker='*')
ax.set_xlabel('聚类数 K', fontsize=12); ax.set_ylabel('轮廓系数', fontsize=12)
ax.set_title('轮廓系数法选择最优 K\n'
             '轮廓系数 ∈ [-1,1]：越大表示簇内紧凑、簇间分离', fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.set_xticks(list(K_range))
savefig(fig, 'fig_5_3_3_silhouette_k.png')

# ========== 图5.4.1 PCA 降维全过程 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(42)
theta = np.pi/4; rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
X_pca = np.random.randn(300,2) @ rot.T * [4.0, 1.0] + [5, 5]
# 原始散点
ax = axes[0]
ax.scatter(X_pca[:,0], X_pca[:,1], s=25, alpha=0.6, color='#3498DB')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('特征 x1'); ax.set_ylabel('特征 x2')
ax.set_title('原始二维数据\n（高度相关，分布呈斜向椭圆）', fontsize=11, fontweight='bold')
cov = np.cov(X_pca.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]; eigvecs = eigvecs[:,order]
# 画主轴
for i, (ev, el) in enumerate(zip(eigvecs.T, eigvals)):
    color = '#E74C3C' if i==0 else '#27AE60'
    ax.annotate('', xy=(5+2*el**0.5*ev[0], 5+2*el**0.5*ev[1]),
                xytext=(5,5), arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(5+2.2*el**0.5*ev[0], 5+2.2*el**0.5*ev[1],
            f'PC{i+1} λ={el:.1f}', fontsize=9, color=color, fontweight='bold')
ax.set_aspect('equal')
# 去均值
X_centered = X_pca - X_pca.mean(axis=0)
# 旋转后的坐标系
ax = axes[1]
X_rot = X_centered @ eigvecs
ax.scatter(X_rot[:,0], X_rot[:,1], s=25, alpha=0.6, color='#3498DB')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('PC1（第一主成分）'); ax.set_ylabel('PC2（第二主成分）')
ax.set_title('旋转到主成分坐标系\n（PC1方向是最大方差方向）', fontsize=11, fontweight='bold')
ax.set_aspect('equal')
# 投影到PC1
ax = axes[2]
pc1_proj = X_rot[:,0]
for i, (x, y) in enumerate(zip(X_rot[:,0], X_rot[:,1])):
    ax.plot([x, x], [0, y], color='#3498DB', alpha=0.2, lw=0.8)
ax.scatter(pc1_proj, [0]*len(pc1_proj), s=25, alpha=0.6, color='#E74C3C')
ax.set_xlabel('PC1 坐标值（降维后）')
ax.set_title('只保留 PC1（保留最大方差）\n降维：2维→1维，保留信息量',
             fontsize=11, fontweight='bold')
ax.set_ylim(-5, 5); ax.axhline(0, color='gray', lw=1)
var_explained = eigvals / eigvals.sum()
ax.text(0.5, 0.95, f'PC1 解释方差：{var_explained[0]*100:.1f}%\nPC2 解释方差：{var_explained[1]*100:.1f}%',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
savefig(fig, 'fig_5_4_1_pca_full_process.png')

# ========== 图5.4.2 PCA + SVD 降维对比 ==========
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
# 3D → 2D
np.random.seed(0)
t = np.random.randn(200, 3) @ np.diag([3, 1, 0.5])
U, S, Vt = np.linalg.svd(t - t.mean(0), full_matrices=False)
# 原始3D
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(t[:,0], t[:,1], t[:,2], s=20, alpha=0.6, color='#3498DB')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('原始三维数据', fontsize=11, fontweight='bold')
# 投影到前两个主成分
ax = fig.add_subplot(2, 2, 2, projection='3d')
t_2d = t @ Vt[:2].T
t_proj = np.column_stack([t_2d[:,0], t_2d[:,1], np.zeros(len(t))])
ax.scatter(t_proj[:,0], t_proj[:,1], t_proj[:,2], s=20, alpha=0.6, color='#E74C3C')
for i in range(len(t)):
    ax.plot([t[i,0], t_proj[i,0]], [t[i,1], t_proj[i,1]], [t[i,2], t_proj[i,2]],
            color='gray', alpha=0.2, lw=0.5)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('')
ax.set_title('PCA投影到2D平面（投影线段=信息损失）', fontsize=11, fontweight='bold')
# t-SNE
ax = axes[1,0]
X_c, y_c = make_classification(n_samples=300, n_features=4, n_informative=3,
                                n_redundant=1, n_clusters_per_class=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_c)
scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_c, cmap='Set1', s=25, alpha=0.7)
ax.set_title('t-SNE 二维嵌入\n（非线性降维，保留局部结构）', fontsize=11, fontweight='bold')
ax.legend(*scatter.legend_elements(), title='类别', fontsize=8)
# PCA vs t-SNE vs LDA
ax = axes[1,1]
pca = PCA(n_components=2).fit_transform(X_c)
ax.scatter(pca[:,0], pca[:,1], c=y_c, cmap='Set1', s=25, alpha=0.5, label='PCA')
ax.set_title('PCA 二维投影\n（线性降维，最大化方差）', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
savefig(fig, 'fig_5_4_2_pca_svd_compare.png')

print("聚类与降维 figures done!")
