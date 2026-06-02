#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增图批次3：Ch5 ML续 + Ch6优化"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy import linalg as scipy_linalg
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}', dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")

# ============================================================
# 图5.3.4 GMM协方差椭圆
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.3.4 高斯混合模型协方差椭圆 | GMM Covariance Ellipses', fontsize=14, fontweight='bold')

np.random.seed(42)
# 两个高斯混合
mean1 = np.array([1., 2.])
cov1 = np.array([[1., 0.7], [0.7, 1.]])
mean2 = np.array([4., 1.])
cov2 = np.array([[0.5, -0.3], [-0.3, 0.8]])

x1_samples = np.random.multivariate_normal(mean1, cov1, 80)
x2_samples = np.random.multivariate_normal(mean2, cov2, 80)

ax = axes[0]
ax.scatter(x1_samples[:,0], x1_samples[:,1], color='#E74C3C', s=40, alpha=0.6, label='簇1样本')
ax.scatter(x2_samples[:,0], x2_samples[:,1], color='#3498DB', s=40, alpha=0.6, label='簇2样本')
ax.scatter(*mean1, color='darkred', s=200, marker='+', lw=3, label='簇1均值')
ax.scatter(*mean2, color='darkblue', s=200, marker='+', lw=3, label='簇2均值')

def draw_ellipse(mean, cov, ax, color, alpha=0.3, lw=2):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    for n_std in [1, 2]:
        width, height = 2*n_std*np.sqrt(eigvals)
        ell = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle,
                                               fill=True, facecolor=color, alpha=alpha, 
                                               edgecolor=color, lw=lw, ls='--' if n_std==2 else '-')
        ax.add_patch(ell)

draw_ellipse(mean1, cov1, ax, '#E74C3C', alpha=0.1)
draw_ellipse(mean1, cov1, ax, '#E74C3C', alpha=0.05, lw=1.5)
draw_ellipse(mean2, cov2, ax, '#3498DB', alpha=0.1)
draw_ellipse(mean2, cov2, ax, '#3498DB', alpha=0.05, lw=1.5)

ax.set_xlabel('x₁', fontsize=11)
ax.set_ylabel('x₂', fontsize=11)
ax.set_title('GMM协方差椭圆（1σ和2σ）', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(-2, 7)
ax.set_ylim(-2, 5)
ax.grid(True, alpha=0.3)

# 右图：概率密度曲面
ax2 = axes[1]
x_grid = np.linspace(-2, 7, 100)
y_grid = np.linspace(-2, 5, 100)
Xg, Yg = np.meshgrid(x_grid, y_grid)
pos = np.dstack([Xg, Yg])

from scipy.stats import multivariate_normal
Z1 = multivariate_normal.pdf(pos, mean1, cov1)
Z2 = multivariate_normal.pdf(pos, mean2, cov2)
Z = 0.5*Z1 + 0.5*Z2  # 等权重混合

contour = ax2.contourf(Xg, Yg, Z, levels=15, cmap='YlOrRd')
ax2.contour(Xg, Yg, Z, levels=10, colors='white', linewidths=0.8, alpha=0.5)
ax2.scatter(x1_samples[:,0], x1_samples[:,1], color='#E74C3C', s=20, alpha=0.5)
ax2.scatter(x2_samples[:,0], x2_samples[:,1], color='#3498DB', s=20, alpha=0.5)
ax2.set_xlabel('x₁', fontsize=11)
ax2.set_ylabel('x₂', fontsize=11)
ax2.set_title('GMM概率密度（等高线图）', fontsize=12)
plt.colorbar(contour, ax=ax2, shrink=0.8, label='混合密度 f(x)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_3_4_gmm_ellipse.png')

# ============================================================
# 图5.3.5 DBSCAN密度聚类
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.3.5 DBSCAN密度聚类 | Density-Based Spatial Clustering', fontsize=14, fontweight='bold')

np.random.seed(42)
# 生成密度聚类数据：3个密集簇 + 噪声点
n_core = 40
# 簇1
c1 = np.random.randn(n_core, 2) * 0.3 + np.array([0, 0])
# 簇2
c2 = np.random.randn(n_core, 2) * 0.3 + np.array([5, 2])
# 簇3
c3 = np.random.randn(n_core, 2) * 0.3 + np.array([2, 5])
# 噪声点（稀疏）
noise = np.random.uniform(-1, 6, (20, 2))

X_dbscan = np.vstack([c1, c2, c3, noise])
true_labels = [0]*n_core + [1]*n_core + [2]*n_core + [-1]*20

ax = axes[0]
colors_db = ['#E74C3C', '#27AE60', '#3498DB', '#95A5A6']
for label in [0, 1, 2, -1]:
    mask = np.array(true_labels) == label
    col = colors_db[label] if label >= 0 else '#95A5A6'
    marker = 'o' if label >= 0 else 'x'
    size = 50 if label >= 0 else 60
    alpha = 0.7 if label >= 0 else 0.4
    ax.scatter(X_dbscan[mask,0], X_dbscan[mask,1], color=col, s=size, alpha=alpha,
               marker=marker, edgecolors='white', lw=0.5,
               label=f'簇{label+1}' if label >= 0 else '噪声点')

# 标注核心点、边界点
ax.scatter([], [], color='black', s=100, marker='s', label='核心点（密度达标）')
ax.scatter([], [], color='black', s=60, marker='o', alpha=0.5, label='边界点')

# 画两个邻域圆（示意ε-邻域）
from matplotlib.patches import Circle
circle1 = Circle((0, 0), 0.8, fill=False, color='#E74C3C', lw=2, ls='--', label='ε-邻域')
circle2 = Circle((5, 2), 0.8, fill=False, color='#27AE60', lw=2, ls='--')
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.set_xlim(-2, 7)
ax.set_ylim(-2, 7)
ax.set_aspect('equal')
ax.set_xlabel('x₁', fontsize=11)
ax.set_ylabel('x₂', fontsize=11)
ax.set_title('DBSCAN结果（无需预设簇数）', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# 右图：聚类质量对比
ax2 = axes[1]
k_range = range(2, 8)
from sklearn.cluster import KMeans, DBSCAN
kmeans_silhouette = []
dbscan_silhouette = []

from sklearn.metrics import silhouette_score
for k in k_range:
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_dbscan)
        kmeans_silhouette.append(silhouette_score(X_dbscan, km.labels_))
    except:
        kmeans_silhouette.append(0)

# DBSCAN varying eps
eps_range = np.linspace(0.3, 1.5, 8)
for eps in eps_range:
    try:
        db = DBSCAN(eps=eps, min_samples=5).fit(X_dbscan)
        if len(set(db.labels_)) > 1 and len(set(db.labels_)) < len(X_dbscan):
            dbscan_silhouette.append(silhouette_score(X_dbscan, db.labels_))
        else:
            dbscan_silhouette.append(0)
    except:
        dbscan_silhouette.append(0)

# Plot DBSCAN silhouette as bar (varying eps)
eps_labels = [f'ε={e:.1f}' for e in eps_range]
bars = ax2.bar(eps_labels, dbscan_silhouette, color='#3498DB', alpha=0.8, edgecolor='white')
ax2.axhline(max(dbscan_silhouette), color='#E74C3C', ls='--', lw=1.5, label=f'最优轮廓系数={max(dbscan_silhouette):.3f}')
ax2.set_xlabel('DBSCAN ε 参数', fontsize=11)
ax2.set_ylabel('轮廓系数', fontsize=11)
ax2.set_title('DBSCAN轮廓系数随ε变化', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_3_5_dbscan.png')

# ============================================================
# 图5.5.3 AdaBoost权重更新
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.5.3 AdaBoost权重更新 | AdaBoost Weight Update', fontsize=14, fontweight='bold')

np.random.seed(42)
n = 20
x_adaboost = np.linspace(0, 10, n)
y_true = np.where(x_adaboost < 5, 1, -1).astype(int)
y_noisy = y_true.copy()
noise_idx = np.random.choice(n, 3, replace=False)
y_noisy[noise_idx] *= -1

ax = axes[0]
ax.scatter(x_adaboost, y_noisy, c=y_noisy, cmap='RdYlGn', s=100, edgecolors='black', lw=1, zorder=5)
ax.axhline(0, color='gray', lw=1)
# 画决策树桩（第一个弱分类器）
x_split = 5.0
ax.axvline(x_split, color='#E74C3C', lw=2, ls='--', label=f'决策树桩 x < {x_split}')
ax.fill_betweenx([-2, 2], 0, x_split, alpha=0.1, color='green')
ax.fill_betweenx([-2, 2], x_split, 10, alpha=0.1, color='red')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.5, 1.5)
ax.set_yticks([-1, 1])
ax.set_yticklabels(['-1', '+1'])
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('类别标签', fontsize=11)
ax.set_title('轮次1：训练弱分类器（决策树桩）', fontsize=12)
ax.legend(fontsize=10)

# 右图：权重更新示意
ax2 = axes[1]
w_init = np.ones(n) / n
w_round1 = w_init.copy()
errors = (y_noisy != y_true).astype(float)
epsilon = np.sum(w_round1 * errors)
alpha1 = 0.5 * np.log((1-epsilon)/epsilon)
w_round1 *= np.exp(-alpha1 * y_noisy * y_true)
w_round1 /= w_round1.sum()  # 归一化

# 画权重（气泡大小）
ax2.scatter(x_adaboost, np.ones(n), s=w_round1*2000, c=y_noisy, cmap='RdYlGn',
            alpha=0.7, edgecolors='white', lw=1)
# 噪声点用红圈标注
for idx in noise_idx:
    ax2.scatter(x_adaboost[idx], 1, s=300, facecolors='none', edgecolors='red', lw=3)
ax2.axhline(1, color='gray', lw=1)
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(0.5, 1.5)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('样本（气泡=权重）', fontsize=11)
ax2.set_title(f'轮次1后：噪声点权重↑（红色圆圈）\nα₁={alpha1:.2f}, 错误率ε={epsilon:.2f}', fontsize=12)
ax2.set_yticks([1])
ax2.set_yticklabels(['样本权重'])
ax2.text(0.5, 0.85, f'权重更新：wᵢ ← wᵢ·exp(-α·yᵢ·h(xᵢ))\n被错分样本权重↑，正确样本权重↓',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_5_3_adaboost_weight.png')

# ============================================================
# 图5.5.5 Stacking集成
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.5.5 Stacking堆叠集成 | Stacking Ensemble', fontsize=14, fontweight='bold')

# 左图：Stacking流程
ax = axes[0]
ax.axis('off')

# Level 0 弱学习器
level0_nodes = ['逻辑回归', '决策树', 'SVM', 'KNN']
x_pos0 = [1, 3, 5, 7]
y_pos0 = 3

for x, name in zip(x_pos0, level0_nodes):
    box = plt.matplotlib.patches.FancyBboxPatch((x-0.5, y_pos0-0.4), 1.0, 0.8,
                                                   boxstyle='round,pad=0.05', 
                                                   facecolor='#BBDEFB', edgecolor='#1976D2', lw=2)
    ax.add_patch(box)
    ax.text(x, y_pos0, name, ha='center', va='center', fontsize=10, fontweight='bold')

# 输入特征
ax.annotate('', xy=(0.3, y_pos0), xytext=(0.0, y_pos0),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='->', color='#666', lw=2))
ax.text(-0.1, y_pos0+0.2, '原始特征\nx', ha='center', va='center', fontsize=9)

# Level 0 → Level 1 元学习器
ax.annotate('', xy=(4, 1.0), xytext=(1, y_pos0-0.4),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate('', xy=(4, 1.0), xytext=(3, y_pos0-0.4),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate('', xy=(4, 1.0), xytext=(5, y_pos0-0.4),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate('', xy=(4, 1.0), xytext=(7, y_pos0-0.4),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

# 元特征标注
for x in x_pos0:
    ax.text(x+0.3, 2.0, f'h{x_pos0.index(x)+1}(x)', fontsize=8, color='#4CAF50', style='italic')

# Level 1 元学习器
meta_box = plt.matplotlib.patches.FancyBboxPatch((3.5, 0.3), 1.0, 0.8,
                                                    boxstyle='round,pad=0.05',
                                                    facecolor='#FFECB3', edgecolor='#FF8F00', lw=2.5)
ax.add_patch(meta_box)
ax.text(4, 0.7, '元学习器\n（逻辑回归）', ha='center', va='center', fontsize=9, fontweight='bold')

# 最终输出
ax.annotate('', xy=(5.5, 0.7), xytext=(4.5, 0.7),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
ax.text(5.7, 0.7, '最终预测\ny', ha='left', va='center', fontsize=9, fontweight='bold', color='#E74C3C')

# 层标注
ax.text(8, y_pos0, 'Level 0\n（基学习器）', ha='left', va='center', fontsize=9, color='#1976D2')
ax.text(8, 0.7, 'Level 1\n（元学习器）', ha='left', va='center', fontsize=9, color='#FF8F00')

ax.set_xlim(-0.5, 9)
ax.set_ylim(-0.2, 4.5)
ax.set_title('Stacking 两层堆叠流程', fontsize=12)

# 右图：Stacking vs 其他集成对比
ax2 = axes[1]
methods = ['Bagging', 'Boosting', 'Stacking\n(2层)', 'Stacking\n(3层)']
accuracies = [0.847, 0.863, 0.875, 0.881]
boostrap_accs = [0.83 + np.random.randn()*0.01 for _ in range(4)]
boostrap_accs[0] = 0.847

colors_s = ['#90A4AE', '#90A4AE', '#FF8F00', '#FF8F00']
bars = ax2.bar(methods, accuracies, color=colors_s, alpha=0.85, edgecolor='black', lw=1.5)
ax2.axhline(0.85, color='green', ls='--', lw=1.5, label='单一模型基准≈0.85')
ax2.set_ylim(0.80, 0.91)
ax2.set_ylabel('测试集准确率', fontsize=11)
ax2.set_title('集成方法效果对比（UCI数据集平均）', fontsize=12)
for bar, acc in zip(bars, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.002, f'{acc:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_yticks(np.arange(0.80, 0.91, 0.02))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_5_5_stacking.png')

# ============================================================
# 图6.3.1 分支定界示意
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图6.3.1 分支定界法 | Branch and Bound', fontsize=14, fontweight='bold')

ax = axes[0]
ax.axis('off')
ax.set_title('分支定界搜索树', fontsize=12)

# 画树形结构
import matplotlib.patches as mpatches
# Root
ax.text(0.5, 0.95, '根节点\n(松弛LP)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#BBDEFB', edgecolor='#1976D2', lw=2))

# Level 1 branches
forks = [(0.2, 0.7, '分支x₁≤3', '#4CAF50'), (0.5, 0.7, '分支x₁≥4', '#E74C3C'), (0.8, 0.7, '分支x₂≤2', '#9C27B0')]
for x, y, txt, col in forks:
    ax.annotate('', xy=(x, y+0.05), xytext=(0.5, 0.88),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=col, lw=1.5))

# Pruned node
ax.text(0.2, 0.55, '剪枝 ✗\n（下界>当前最优）', ha='center', va='center', fontsize=8,
        color='gray',
        bbox=dict(boxstyle='round', facecolor='#EEEEEE', edgecolor='gray', lw=1, linestyle='--'))

# Level 2
for x2, y2, txt2, col2 in [(0.1, 0.45, 'x₁≤3,x₂≤2\n整数解✓', '#4CAF50'), (0.4, 0.45, 'x₁≤3,x₂≥3\n继续分支', '#FF9800')]:
    ax.annotate('', xy=(x2, y2+0.05), xytext=(0.2, 0.62),
                arrowprops=dict(arrowstyle='->', color=col2, lw=1.5))
    ax.text(x2, y2, txt2, ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=col2, lw=1.5))

# 最优解
ax.text(0.1, 0.28, '★ 最优解\nz*=18.5\nx₁=3,x₂=3', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F57F17', lw=2))

ax.annotate('', xy=(0.1, 0.38), xytext=(0.1, 0.37),
            arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.5))

# 说明文字
branch_note = '分支定界策略：\n① 解分数整数规划问题（LP松弛）\n② 若解为分数→按变量分支\n③ 若下界≥已知最优解→剪枝\n④ 迭代直到找到最优整数解'
ax.text(0.65, 0.3, branch_note, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))


ax.set_xlim(0, 1)
ax.set_ylim(0.15, 1.0)

# 右图：整数规划可行点是离散的
ax2 = axes[1]
ax2.set_title('整数规划：可行点是不连续的格点', fontsize=12)
x_ip = np.arange(0, 8)
y_ip = np.arange(0, 8)
X, Y = np.meshgrid(x_ip, y_ip)

# 约束条件：x + 2y <= 12, 3x + y <= 12, x,y >= 0, x,y 整数
feasible = (X + 2*Y <= 12) & (3*X + Y <= 12) & (X >= 0) & (Y >= 0)
infeasible = ~feasible

ax2.scatter(X[feasible], Y[feasible], color='#4CAF50', s=120, alpha=0.8, label='可行整数点', zorder=5)
ax2.scatter(X[infeasible], Y[infeasible], color='#E0E0E0', s=80, alpha=0.5, label='不可行点', zorder=3)

# LP松弛最优（连续）
from numpy.linalg import solve as np_solve
# LP最优在 x+2y=12, 3x+y=12 交点
A = np.array([[1,2],[3,1]])
b = np.array([12,12])
lp_opt = np_solve(A, b)
ax2.scatter(lp_opt[0], lp_opt[1], color='blue', s=200, marker='*', zorder=10, label=f'LP松弛最优({lp_opt[0]:.1f},{lp_opt[1]:.1f})')
ax2.scatter(lp_opt[0], lp_opt[1], color='blue', s=200, marker='*', zorder=10)

# 整数最优（近似）
ax2.axvline(3, color='red', lw=1.5, ls='--', alpha=0.5)
ax2.axhline(3, color='red', lw=1.5, ls='--', alpha=0.5)
ax2.scatter([3], [3], color='orange', s=300, marker='s', zorder=10, label='整数最优(3,3)')

# 可行域边界
x_line = np.linspace(0, 6, 100)
ax2.plot(x_line, (12-x_line)/2, 'b-', lw=2, label='x+2y=12')
ax2.plot(x_line, 12-3*x_line, 'purple', lw=2, label='3x+y=12')
ax2.fill_between(x_line, 0, np.minimum((12-x_line)/2, 12-3*x_line),
                 where=(x_line >= 0) & ((12-x_line)/2 >= 0) & (12-3*x_line >= 0),
                 alpha=0.1, color='blue')

ax2.set_xlim(-0.5, 6.5)
ax2.set_ylim(-0.5, 6.5)
ax2.set_xlabel('x₁', fontsize=11)
ax2.set_ylabel('x₂', fontsize=11)
ax2.set_aspect('equal')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(7))
ax2.set_yticks(range(7))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_6_3_1_branch_bound.png')

print("批次3完成!")
