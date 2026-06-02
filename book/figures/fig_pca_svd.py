#!/usr/bin/env python3
"""
# 中文字体配置
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
PCA / SVD 几何解释图
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

plt.style.use('seaborn-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== 图1: PCA 降维演示 (2D → 1D) ==========
np.random.seed(42)

# 创建有相关性的二维数据
theta = np.pi / 4  # 45度旋转
rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
x = np.random.randn(200, 2) @ rot.T * [3.0, 1.0]

# 计算PCA
cov = np.cov(x.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 左: 原始数据 + PCA主轴
ax = axes[0]
ax.scatter(x[:, 0], x[:, 1], alpha=0.5, s=20, color='#6C8EBF', label='数据点')

# 画主成分方向
for i, (ev, col, label) in enumerate(zip(eigenvectors.T, ['#E74C3C', '#27AE60'], ['PC1 (σ₁)', 'PC2 (σ₂)'])):
    ax.annotate('', xy=ev * eigenvalues[i] * 0.5, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    ax.text(ev[0] * eigenvalues[i] * 0.6, ev[1] * eigenvalues[i] * 0.6, label,
            color=col, fontsize=11, fontweight='bold')

# 画原点十字
ax.axhline(0, color='#CCCCCC', lw=0.8)
ax.axvline(0, color='#CCCCCC', lw=0.8)
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('原始数据 + PCA 主成分方向', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-7, 7)
ax.set_ylim(-4, 4)

# 中: 投影到PC1
ax = axes[1]
ax.scatter(x[:, 0], x[:, 1], alpha=0.4, s=20, color='#6C8EBF')

# PC1方向线
pc1_dir = eigenvectors[:, 0]
x_range = np.linspace(-6, 6, 100)
ax.plot(x_range, (pc1_dir[1]/pc1_dir[0]) * x_range, color='#E74C3C', lw=2, ls='--', label='PC1 方向')

# 投影点
projected = x @ eigenvectors[:, 0:1] * eigenvectors[:, 0:1].T
ax.scatter(projected[:, 0], projected[:, 1], alpha=0.7, s=20, color='#E74C3C', label='投影点')

for i in range(0, 200, 10):
    ax.plot([x[i, 0], projected[i, 0]], [x[i, 1], projected[i, 1]],
            color='gray', lw=0.5, alpha=0.4)

ax.axhline(0, color='#CCCCCC', lw=0.8)
ax.axvline(0, color='#CCCCCC', lw=0.8)
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('投影到 PC1 (降至1维)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-7, 7)
ax.set_ylim(-4, 4)

# 右: 方差解释比例
ax = axes[2]
total_var = eigenvalues.sum()
var_ratio = eigenvalues / total_var * 100
components = ['PC1', 'PC2']
colors = ['#E74C3C', '#27AE60']
bars = ax.bar(components, var_ratio, color=colors, width=0.5, alpha=0.85)
ax.set_ylabel('解释方差比例 (%)', fontsize=12)
ax.set_title('各主成分解释的方差比例', fontsize=12)
ax.set_ylim(0, 100)
for bar, ratio in zip(bars, var_ratio):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{ratio:.1f}%', ha='center', fontsize=12, fontweight='bold')

fig.suptitle('PCA 主成分分析：找到方差最大的方向', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_pca_demonstration.png')

# ========== 图2: SVD 橡皮泥几何 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 画一个单位圆
theta_circle = np.linspace(0, 2*np.pi, 200)
unit_circle = np.stack([np.cos(theta_circle), np.sin(theta_circle)], axis=1)

# 变换矩阵 (拉伸+旋转)
A = np.array([[2.0, 0.8], [0.3, 1.5]])
transformed = unit_circle @ A.T  # A @ x^T 的转置版本

# 左: 原始单位圆
ax = axes[0]
ax.plot(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', lw=2, label='单位球面')
ax.fill(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', alpha=0.1)
ax.set_xlabel('v₁', fontsize=12)
ax.set_ylabel('v₂', fontsize=12)
ax.set_title('Step 1: 输入空间的单位球\n(V⁺, 正交基)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

# 中: 被 Σ 拉伸
ax = axes[1]
# 对角奇异值
Sigma = np.diag([2.5, 1.0])
unit_circle_scaled = unit_circle @ Sigma.T
ax.plot(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', lw=1.5, alpha=0.4, label='原单位球')
ax.plot(unit_circle_scaled[:, 0], unit_circle_scaled[:, 1], color='#E67E22', lw=2.5, label='经Σ拉伸后')
ax.set_xlabel('Σ·v₁', fontsize=12)
ax.set_ylabel('Σ·v₂', fontsize=12)
ax.set_title('Step 2: Σ 拉伸\n(奇异值 σ₁=2.5, σ₂=1.0)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

# 右: 完整SVD变换
ax = axes[2]
U, S_vals, VT = np.linalg.svd(A, full_matrices=False)
ellipse = unit_circle_scaled @ np.diag(S_vals) @ VT  # 已经做了Σ·Vᵀ

# 画正交基方向
for i, (vec, col) in enumerate(zip(VT.T, ['#3498DB', '#27AE60'])):
    ax.annotate('', xy=vec * S_vals[i] * 1.1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text(vec[0] * S_vals[i] * 1.2, vec[1] * S_vals[i] * 1.2, f'σ{i+1}·v{i+1}',
            color=col, fontsize=10, fontweight='bold')

ax.plot(ellipse[:, 0], ellipse[:, 1], color='#E74C3C', lw=2.5, label='完整变换结果')

# 画旋转后的U基方向
for i, (vec, col) in enumerate(zip(U.T, ['#9B59B6', '#F39C12'])):
    ax.annotate('', xy=vec * S_vals[i] * 1.2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2, ls='dashed'))

ax.set_xlabel("U·Σ·v'", fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.set_title('Step 3: U·Σ·Vᵀ 旋转回来\n(最终椭球)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

fig.suptitle("SVD 几何解释：橡皮泥的拉伸、压扁与旋转\n(任意矩阵 → 正交·对角·正交 分解)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_svd_geometry.png')

print("PCA / SVD 图生成完成！")
