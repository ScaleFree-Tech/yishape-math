#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图6.x 优化算法：凸函数、牛顿法、单纯形法、约束优化
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

# ========== 图6.1.1 凸函数 vs 非凸函数 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.linspace(-3, 3, 400)
# 凸函数
ax = axes[0]
y_conv = x**2 + 0.5*np.sin(2*x) + 1
y_conv_convex = x**2
ax.plot(x, y_conv, color='#3498DB', lw=2.5, label='f(x)：凸函数')
ax.plot(x, y_conv_convex, color='#E74C3C', lw=1.5, ls='--', alpha=0.6, label='上确界 = 下确界（凸）')
ax.fill_between(x, y_conv, alpha=0.1, color='#3498DB')
# 画弦
x1, x2 = -2, 2; y1, y2 = y_conv_convex[x1+3], y_conv_convex[x2+3]
ax.plot([x1, x2], [y1, y2], color='#27AE60', lw=2, ls=':', label='任意弦线在函数上方')
ax.scatter([x1, x2], [y1, y2], color='#27AE60', s=80)
ax.annotate('f(θx1+(1-θ)x2)\n≤ θf(x1)+(1-θ)f(x2)', xy=(0, 1), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.set_title('凸函数：全局最优在内部或边界\n任意两点连线在函数上方', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
# 非凸函数
ax = axes[1]
y_nc = np.sin(3*x) + 0.3*x**2
ax.plot(x, y_nc, color='#3498DB', lw=2.5)
ax.fill_between(x, y_nc, alpha=0.1, color='#3498DB')
# 局部极小
for xmin, label, color in [(-1.5, '局部极小', '#E74C3C'), (0.5, '全局极小', '#27AE60'), (2, '局部极小', '#E74C3C')]:
    idx = np.argmin(np.abs(x-xmin))
    ax.scatter([x[idx]], [y_nc[idx]], color=color, s=100, zorder=10, marker='v')
    ax.text(x[idx]+0.1, y_nc[idx], label, fontsize=9, color=color, fontweight='bold')
ax.set_title('非凸函数：多个局部极小\n梯度下降可能陷入局部最优', fontsize=12, fontweight='bold')
savefig(fig, 'fig_6_1_1_convex_vs_nonconvex.png')

# ========== 图6.1.2 梯度下降与牛顿法对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.linspace(-4, 4, 400)
y = x**4 - 5*x**3 + 2*x**2 + 5
ax = axes[0]
ax.plot(x, y, color='#34495E', lw=2)
# 梯度下降路径
def grad(x_val): return 4*x_val**3 - 15*x_val**2 + 4*x_val
w_gd = 2.5; path_gd = [w_gd]
for _ in range(20): path_gd.append(w_gd := w_gd - 0.01*grad(w_gd))
path_gd = np.array(path_gd)
ax.plot(path_gd, path_gd**4 - 5*path_gd**3 + 2*path_gd**2 + 5, 'o-', color='#E74C3C', ms=6, lw=1.5, label='梯度下降')
# 牛顿法路径
def hess(x_val): return 12*x_val**2 - 30*x_val + 2
w_newton = 2.5; path_nt = [w_newton]
for _ in range(8):
    g = grad(w_newton); h = hess(w_newton)
    if abs(h) > 1e-6: w_newton = w_newton - g/h
    path_nt.append(w_newton)
path_nt = np.array(path_nt)
ax.plot(path_nt, path_nt**4 - 5*path_nt**3 + 2*path_nt**2 + 5, 's-', color='#27AE60', ms=7, lw=1.5, label='牛顿法')
ax.scatter([path_gd[-1]], [path_gd[-1]**4-5*path_gd[-1]**3+2*path_gd[-1]**2+5], s=150, color='#E74C3C', zorder=15, marker='*')
ax.set_title('梯度下降（线性收敛）vs 牛顿法（二次收敛）\n'
             '牛顿法在极小点附近收敛极快，但需要Hessian', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('w'); ax.set_ylabel('J(w)')
# 收敛速度对比
ax = axes[1]
iters = np.arange(1, 21)
gd_loss = [path_gd[i]**4-5*path_gd[i]**3+2*path_gd[i]**2+5 for i in range(20)]
nt_loss = [path_nt[i]**4-5*path_nt[i]**3+2*path_nt[i]**2+5 for i in range(min(9,len(path_nt)))]
ax.semilogy(iters[:len(gd_loss)], gd_loss - min(gd_loss) + 1e-10,
            'o-', color='#E74C3C', lw=2, label='梯度下降')
ax.semilogy(np.arange(1, len(nt_loss)+1), nt_loss - min(nt_loss) + 1e-10,
            's-', color='#27AE60', lw=2, label='牛顿法')
ax.set_title('收敛速度对比（对数坐标）\n牛顿法：二次收敛（超线性）| 梯度下降：线性收敛',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('迭代次数'); ax.set_ylabel('loss - loss* (对数)')
savefig(fig, 'fig_6_1_2_gd_vs_newton.png')

# ========== 图6.2.1 LP 单纯形法路径 ==========
fig, ax = plt.subplots(figsize=(10, 8))
# 可行域
x1 = np.linspace(0, 8, 200)
# 约束：x1+x2<=8, x1<=5, x2<=6, x1>=0, x2>=0
y1 = np.clip(8 - x1, 0, 8)
y2 = np.clip(6 - 0*x1, 0, 8)
y3 = np.clip(5 - 0*x1, 0, 8)  # x1 <= 5
ax.fill_between(x1, 0, np.minimum(np.minimum(y1, y2), y3),
                alpha=0.15, color='#3498DB', label='可行域')
# 约束线
for yc, label, ls in [(8-x1, 'x1+x2≤8', '--'), (0*x1+6, 'x2≤6', ':'), (0*x1+5, 'x1≤5', '-.')]:
    ax.plot(x1, yc, color='#E74C3C', lw=1.5, ls=ls, label=label, alpha=0.7)
# 单纯形法路径（近似）
simplex_path = [(0,0), (5,0), (5,3), (2,6), (0,6)]
sx, sy = zip(*simplex_path)
ax.plot(sx, sy, 'o-', color='#27AE60', lw=2.5, ms=10, markeredgecolor='black',
        label='单纯形法路径')
ax.scatter([2], [6], s=200, color='#F39C12', zorder=20, marker='*', label='最优解 (2,6)')
ax.set_xlim(-0.5, 8); ax.set_ylim(-0.5, 8)
ax.axhline(0, color='gray', lw=1); ax.axvline(0, color='gray', lw=1)
ax.set_xlabel('x1', fontsize=12); ax.set_ylabel('x2', fontsize=12)
ax.set_title('单纯形法：从一个顶点出发，沿边移动到相邻顶点\n'
             '目标 max z = 3x1 + 2x2 | 最优解 z=18', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
savefig(fig, 'fig_6_2_1_simplex_path.png')

# ========== 图6.2.2 对偶理论：影子价格 ==========
fig, ax = plt.subplots(figsize=(10, 7))
# 可行域
x1 = np.linspace(0, 10, 200)
y1 = np.clip(8 - x1, 0, 8)
y2 = np.clip(0*x1, 0, 8)
ax.fill_between(x1, 0, np.minimum(y1, y2), alpha=0.15, color='#3498DB')
# 目标线系
for z_val, color, lw in [(12, '#95A5A6', 1), (16, '#3498DB', 1.5), (18, '#E74C3C', 2.5)]:
    yz = (z_val - 3*x1) / 2
    ax.plot(x1, np.clip(yz, 0, 10), color=color, lw=lw, ls='--', alpha=0.8)
ax.scatter([2], [6], s=200, color='#F39C12', zorder=20, marker='*', label='最优解 (2,6)')
ax.plot([0,5,5,0,0], [0,0,3,6,0], color='#27AE60', lw=2)
ax.set_xlim(-0.5, 8); ax.set_ylim(-0.5, 8)
ax.axhline(0, color='gray', lw=1); ax.axvline(0, color='gray', lw=1)
ax.set_xlabel('x1', fontsize=12); ax.set_ylabel('x2', fontsize=12)
ax.set_title('线性规划的几何解释\n'
             '虚线：目标函数等值线，越远越高（最大化）\n'
             '最优解：可行域顶点，约束边界交点', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_aspect('equal')
ax.text(0.5, 7.5, 'z=12', fontsize=9, color='#95A5A6')
ax.text(0.5, 5.5, 'z=16', fontsize=9, color='#3498DB')
ax.text(0.5, 3.5, 'z=18（最高可行等值线）', fontsize=10, color='#E74C3C', fontweight='bold')
savefig(fig, 'fig_6_2_2_lp_dual_shadow.png')

# ========== 图6.4.1 启发式搜索：模拟退火 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
np.random.seed(42)
# 二维 Rastrigin 函数（多局部极小）
x1 = np.linspace(-5, 5, 300); x2 = np.linspace(-5, 5, 300)
X1, X2 = np.meshgrid(x1, x2)
R = 10*2 + X1**2 + X2**2 - 10*np.cos(2*np.pi*X1) - 10*np.cos(2*np.pi*X2)
ax = axes[0]
contour = ax.contourf(X1, X2, R, levels=30, cmap='viridis', alpha=0.7)
plt.colorbar(contour, ax=ax, label='Rastrigin 函数值')
ax.set_xlabel('x1', fontsize=12); ax.set_ylabel('x2', fontsize=12)
ax.set_title('Rastrigin 函数（多局部极小的非凸函数）\n'
             '全局最优点在 (0,0)，全局最小值=0', fontsize=11, fontweight='bold')
# 模拟退火路径（示意）
np.random.seed(0)
sa_path = [(2.0, 2.0)]
for _ in range(200):
    cur = sa_path[-1]
    T = max(0.01, 5*(1 - _/200))  # 温度下降
    proposal = (cur[0] + np.random.randn()*T, cur[1] + np.random.randn()*T)
    delta = (proposal[0]**2 + proposal[1]**2 - 10*np.cos(2*np.pi*proposal[0])
             - 10*np.cos(2*np.pi*proposal[1])) - (cur[0]**2 + cur[1]**2 - 10*np.cos(2*np.pi*cur[0]) - 10*np.cos(2*np.pi*cur[1]))
    if delta < 0 or np.random.rand() < np.exp(-delta/T):
        sa_path.append(proposal)
sa_path = np.array(sa_path)
ax.plot(sa_path[:,0], sa_path[:,1], color='red', lw=0.8, alpha=0.6)
ax.scatter(sa_path[::20,0], sa_path[::20,1], color='red', s=10, alpha=0.8)
# 温度曲线
ax2 = axes[1]
temps = [max(0.01, 5*(1-i/200)) for i in range(200)]
ax2.plot(temps, color='#E74C3C', lw=2.5)
ax2.set_title('模拟退火温度衰减曲线\n'
             '高温：接受差解概率高（跳出局部最优）\n'
             '低温：几乎只接受好解（精细搜索）', fontsize=11, fontweight='bold')
ax2.set_xlabel('迭代次数', fontsize=11); ax2.set_ylabel('温度 T', fontsize=11)
ax2.fill_between(range(200), temps, alpha=0.2, color='#E74C3C')
savefig(fig, 'fig_6_4_1_simulated_annealing.png')

print("优化算法 figures done!")
