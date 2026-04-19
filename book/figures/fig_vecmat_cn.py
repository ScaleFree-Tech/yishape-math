#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图1.x 线性代数核心：向量与矩阵
生成 6 个子图：向量运算、矩阵变换、行列式几何意义、特征值分解、矩阵秩、可分离投影
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}')
    plt.close(fig)

# ========== 图1.1 向量运算 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
v1 = np.array([2, 1]); v2 = np.array([-1, 2])
for ax_obj, title, vecs in [
    (axes[0], '向量加法', [v1, v2, v1+v2]),
    (axes[1], '向量数乘', [v1, 2*v1, -1*v1]),
    (axes[2], '向量内积', [v1, v2]),
]:
    colors = ['#3498DB', '#E74C3C', '#27AE60']
    for i, vec in enumerate(vecs):
        ax_obj.arrow(0, 0, vec[0], vec[1], head_width=0.15, head_length=0.1,
                      fc=colors[i], ec=colors[i], lw=2)
    ax_obj.set_xlim(-4, 5); ax_obj.set_ylim(-2, 5)
    ax_obj.axhline(0, color='gray', lw=0.5); ax_obj.axvline(0, color='gray', lw=0.5)
    ax_obj.set_aspect('equal'); ax_obj.set_title(title, fontsize=13, fontweight='bold')
    if title == '向量加法':
        ax_obj.annotate('', xy=vecs[2], xytext=vecs[0],
                        arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5, ls='--'))
        ax_obj.text(*(vecs[2]+0.2), 'v1+v2', color='#27AE60', fontsize=11)
savefig(fig, 'fig_1_1_vector_ops.png')

# ========== 图1.2 矩阵线性变换 ==========
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
grids = [
    ('恒等变换', np.eye(2), '单位矩阵 I'),
    ('旋转 45°', np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                             [np.sin(np.pi/4),  np.cos(np.pi/4)]]), 'R(45°)'),
    ('剪切 (x轴)', np.array([[1, 0.8], [0, 1]]), 'S_x'),
    ('缩放 (2x,0.5y)', np.array([[2, 0], [0, 0.5]]), 'D(2,0.5)'),
    ('镜像 (y轴)', np.array([[-1, 0], [0, 1]]), 'M_y'),
    ('投影 (x轴)', np.array([[1, 0], [0, 0]]), 'P_x'),
]
for ax, (title, M, label) in zip(axes.flat, grids):
    # 原始单位方
    corners = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    transformed = corners @ M.T
    ax.fill(corners[:,0], corners[:,1], alpha=0.15, color='#3498DB', label='原图形')
    ax.fill(transformed[:,0], transformed[:,1], alpha=0.3, color='#E74C3C', label='变换后')
    ax.plot(corners[:,0], corners[:,1], 'o-', color='#3498DB', ms=4)
    ax.plot(transformed[:,0], transformed[:,1], 'o-', color='#E74C3C', ms=4)
    # 特征向量
    eigvals, eigvecs = np.linalg.eigh(M)
    for i, (ev, el) in enumerate(zip(eigvecs.T, eigvals)):
        if abs(el) > 0.01:
            color = '#27AE60' if i==0 else '#9B59B6'
            ax.annotate('', xy=(el*ev[0]*1.5, el*ev[1]*1.5), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.text(el*ev[0]*1.7, el*ev[1]*1.7+0.1, f'λ={el:.1f}', color=color, fontsize=9)
    ax.set_xlim(-3, 4); ax.set_ylim(-2, 3)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n{label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
savefig(fig, 'fig_1_2_matrix_transform.png')

# ========== 图1.3 行列式几何意义 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
det_examples = [
    (np.array([[2,0],[0,3]]), '面积 = 6', 'det > 0 正伸缩'),
    (np.array([[2,0],[0,-1.5]]), '面积 = -3', 'det < 0 镜像+翻转'),
    (np.array([[2,1],[1,0.5]]), '面积 ≈ 0', '行列式 ≈ 0 不可逆'),
]
for ax, (M, area_txt, note) in zip(axes, det_examples):
    corners = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    t = corners @ M.T
    ax.fill(corners[:,0], corners[:,1], alpha=0.15, color='#3498DB')
    ax.fill(t[:,0], t[:,1], alpha=0.3, color='#E74C3C')
    det = np.linalg.det(M)
    ax.text(0.3, 0.3, f'det = {det:.1f}\n{area_txt}', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlim(-1, 6); ax.set_ylim(-2, 5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal'); ax.set_title(note, fontsize=12, fontweight='bold')
savefig(fig, 'fig_1_3_determinant_geom.png')

# ========== 图1.4 特征值与特征向量 ==========
fig, ax = plt.subplots(figsize=(9, 7))
theta = np.pi/6; R = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])
eigvals, eigvecs = np.linalg.eigh(R)
# 随机椭圆点
np.random.seed(42)
angles = np.linspace(0, 2*np.pi, 200)
ellipse = np.stack([2*np.cos(angles), np.cos(angles+np.pi/4)], axis=1)
ellipse_t = ellipse @ R.T
ax.fill(ellipse[:,0], ellipse[:,1], alpha=0.1, color='#3498DB', label='原椭圆 (A)')
ax.fill(ellipse_t[:,0], ellipse_t[:,1], alpha=0.2, color='#E74C3C', label='旋转后 (RA^T)')
for i, (ev, el) in enumerate(zip(eigvecs.T, eigvals)):
    color = '#27AE60' if i==0 else '#9B59B6'
    ax.annotate('', xy=(3*ev[0], 3*ev[1]), xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(3.2*ev[0], 3.2*ev[1], f'λ={el:.2f}', color=color, fontsize=11, fontweight='bold')
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_aspect('equal')
ax.set_title('特征值分解：矩阵在特征向量方向上只缩放，不改变方向\n(旋转矩阵 R 的特征值是复数，故椭圆仅旋转而非缩放)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
savefig(fig, 'fig_1_4_eigen_geom.png')

# ========== 图1.5 矩阵的秩 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
rank_examples = [
    (np.array([[1,2],[2,4]]), '秩=1', '列共线，行也共线'),
    (np.array([[1,0,2],[0,1,3],[0,0,0]]), '秩=2', '有一个零行'),
    (np.array([[1,0,0],[0,1,0],[0,0,1]]), '秩=3', '满秩单位矩阵'),
]
for ax, (M, rank_txt, note) in zip(axes, rank_examples):
    rank = np.linalg.matrix_rank(M)
    ax.imshow(M, cmap='Blues', aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i,j]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white' if M[i,j]>1 else '#2C3E50')
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels([f'Col{c+1}' for c in range(M.shape[1])])
    ax.set_yticklabels([f'Row{r+1}' for r in range(M.shape[0])])
    ax.set_title(f'{rank_txt}：{note}', fontsize=12, fontweight='bold')
savefig(fig, 'fig_1_5_matrix_rank.png')

# ========== 图1.6 正交投影 ==========
fig, ax = plt.subplots(figsize=(9, 7))
np.random.seed(0)
# 子空间（x轴）
x = np.linspace(-1, 5, 100)
ax.plot(x, [0]*len(x), color='#3498DB', lw=3, label='子空间 U (x轴)')
# 空间中的向量
for c, vec in enumerate([[3,2], [4,1], [1.5,3]]):
    vec = np.array(vec)
    proj = np.array([vec[0], 0])
    ax.arrow(0, 0, vec[0], vec[1], head_width=0.12, head_length=0.08,
             fc='#E74C3C', ec='#E74C3C', alpha=0.8)
    ax.arrow(0, 0, proj[0], proj[1], head_width=0.12, head_length=0.08,
             fc='#27AE60', ec='#27AE60', alpha=0.8, ls='--')
    ax.plot([vec[0], proj[0]], [vec[1], proj[1]], 'k--', alpha=0.4)
    ax.text(vec[0]+0.1, vec[1], f'b_{c+1}', fontsize=11, color='#E74C3C')
    ax.text(proj[0]+0.1, proj[1], f'Proj(b_{c+1})', fontsize=9, color='#27AE60')
ax.set_xlim(-1, 6); ax.set_ylim(-1, 4)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_aspect('equal')
ax.set_title('正交投影：向量 b 投影到子空间 U 的最近点\n'
             '残差 b - Proj(b) 与子空间正交', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
savefig(fig, 'fig_1_6_orth_proj.png')

print("向量与矩阵 figures done!")
