#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCA / SVD geometric interpretation figures"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== Fig 1: PCA Demonstration ==========
np.random.seed(42)
theta = np.pi / 4
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
x = np.random.randn(200, 2) @ rot.T * [3.0, 1.0]
cov = np.cov(x.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

ax = axes[0]
ax.scatter(x[:, 0], x[:, 1], alpha=0.5, s=20, color='#6C8EBF')
for ev, col, label in zip(eigenvectors.T, ['#E74C3C', '#27AE60'], ['PC1', 'PC2']):
    ax.annotate('', xy=ev * eigenvalues[0] * 0.5, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    ax.text(ev[0] * eigenvalues[0] * 0.6, ev[1] * eigenvalues[0] * 0.6, label, color=col, fontsize=11, fontweight='bold')
ax.axhline(0, color='#CCCCCC', lw=0.8)
ax.axvline(0, color='#CCCCCC', lw=0.8)
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Original Data + PCA Axes', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-7, 7)
ax.set_ylim(-4, 4)

ax = axes[1]
ax.scatter(x[:, 0], x[:, 1], alpha=0.4, s=20, color='#6C8EBF')
pc1_dir = eigenvectors[:, 0]
x_range = np.linspace(-6, 6, 100)
ax.plot(x_range, (pc1_dir[1]/pc1_dir[0]) * x_range, color='#E74C3C', lw=2, ls='--', label='PC1 Direction')
projected = x @ eigenvectors[:, 0:1] * eigenvectors[:, 0:1].T
ax.scatter(projected[:, 0], projected[:, 1], alpha=0.7, s=20, color='#E74C3C')
for i in range(0, 200, 10):
    ax.plot([x[i, 0], projected[i, 0]], [x[i, 1], projected[i, 1]], color='gray', lw=0.5, alpha=0.4)
ax.axhline(0, color='#CCCCCC', lw=0.8)
ax.axvline(0, color='#CCCCCC', lw=0.8)
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Projection onto PC1 (Reduce to 1D)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-7, 7)
ax.set_ylim(-4, 4)

ax = axes[2]
total_var = eigenvalues.sum()
var_ratio = eigenvalues / total_var * 100
components = ['PC1', 'PC2']
colors = ['#E74C3C', '#27AE60']
bars = ax.bar(components, var_ratio, color=colors, width=0.5, alpha=0.85)
ax.set_ylabel('Variance Explained (%)', fontsize=12)
ax.set_title('Variance Explained by Each PC', fontsize=12)
ax.set_ylim(0, 100)
for bar, ratio in zip(bars, var_ratio):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{ratio:.1f}%', ha='center', fontsize=12, fontweight='bold')

fig.suptitle('PCA: Finding the Direction of Maximum Variance', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_pca_demonstration.png')

# ========== Fig 2: SVD Geometry ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
theta_circle = np.linspace(0, 2*np.pi, 200)
unit_circle = np.stack([np.cos(theta_circle), np.sin(theta_circle)], axis=1)

A = np.array([[2.0, 0.8], [0.3, 1.5]])
transformed = unit_circle @ A.T

ax = axes[0]
ax.plot(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', lw=2)
ax.fill(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', alpha=0.1)
ax.set_xlabel('v1', fontsize=12)
ax.set_ylabel('v2', fontsize=12)
ax.set_title('Step 1: Unit Circle in Input Space\n(V, Orthogonal Basis)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

ax = axes[1]
Sigma = np.diag([2.5, 1.0])
unit_circle_scaled = unit_circle @ Sigma.T
ax.plot(unit_circle[:, 0], unit_circle[:, 1], color='#3498DB', lw=1.5, alpha=0.4, label='Original circle')
ax.plot(unit_circle_scaled[:, 0], unit_circle_scaled[:, 1], color='#E67E22', lw=2.5, label='After scaling')
ax.set_xlabel('Sigma . v1', fontsize=12)
ax.set_ylabel('Sigma . v2', fontsize=12)
ax.set_title('Step 2: Sigma Stretching\n(Singular values sigma1=2.5, sigma2=1.0)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

ax = axes[2]
U, S_vals, VT = np.linalg.svd(A, full_matrices=False)
ellipse = unit_circle_scaled @ np.diag(S_vals) @ VT
for i, (vec, col) in enumerate(zip(VT.T, ['#3498DB', '#27AE60'])):
    ax.annotate('', xy=vec * S_vals[i] * 1.1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text(vec[0] * S_vals[i] * 1.2, vec[1] * S_vals[i] * 1.2, f's{i+1}.v{i+1}', color=col, fontsize=10, fontweight='bold')
ax.plot(ellipse[:, 0], ellipse[:, 1], color='#E74C3C', lw=2.5)
ax.set_xlabel("U . Sigma . v'", fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.set_title('Step 3: U . Sigma . V^T Rotation\n(Final Ellipse)', fontsize=12)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axhline(0, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#DDDDDD', lw=0.8)

fig.suptitle("SVD: Rubber Sheet Stretch, Squash, and Rotate\n(Any Matrix -> Orthogonal . Diagonal . Orthogonal)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_svd_geometry.png')

print("PCA/SVD figures done!")
