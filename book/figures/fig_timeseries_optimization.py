#!/usr/bin/env python3
"""
时间序列分解图 + LP可行域图
"""
# 中文字体配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

plt.style.use('seaborn-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== 图1: 时间序列分解 (STL) ==========
np.random.seed(42)
n = 200
dates = np.arange('2022-01-01', '2022-01-01' + n, dtype='datetime64[D]')

# 趋势 (缓慢上升)
trend = np.linspace(5, 15, n) + 2 * np.sin(np.linspace(0, 4*np.pi, n))
# 季节性 (周期=12的正弦波)
seasonal = 3 * np.sin(np.linspace(0, 2*np.pi * n/12, n))
# 残差
resid = np.random.randn(n) * 1.2
# 原始序列
y = trend + seasonal + resid

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

colors = {'trend': '#2C3E50', 'seasonal': '#27AE60', 'resid': '#E74C3C', 'observed': '#3498DB'}

axes[0].plot(dates, y, color=colors['observed'], lw=1.5, alpha=0.8)
axes[0].set_ylabel('原始序列\n(观测值)', fontsize=10)
axes[0].set_title('时间序列分解 (STL: 趋势 + 季节性 + 残差)', fontsize=13, fontweight='bold')

axes[1].plot(dates, trend, color=colors['trend'], lw=2.5)
axes[1].set_ylabel('趋势\n(Trend)', fontsize=10)

axes[2].plot(dates, seasonal, color=colors['seasonal'], lw=2)
axes[2].set_ylabel('季节性\n(Seasonal)', fontsize=10)

axes[3].plot(dates, resid, color=colors['resid'], lw=1, alpha=0.7)
axes[3].axhline(0, color='gray', lw=0.8, ls='--')
axes[3].set_ylabel('残差\n(Residual)', fontsize=10)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[3].set_xlabel('日期', fontsize=11)

# 标注
axes[0].annotate('y = Trend + Seasonal + Residual', xy=(0.02, 0.85),
                 xycoords='axes fraction', fontsize=10, color='#555555', style='italic')

plt.tight_layout()
savefig(fig, 'fig_timeseries_decomposition.png')

# ========== 图2: LP 可行域 ==========
fig, ax = plt.subplots(figsize=(8, 7))

# 可行域: x + 2y <= 8, 2x + y <= 10, x >= 0, y >= 0
# 顶点: (0,0), (0,4), (4,2), (5,0)

# 可行域多边形
feasible_x = [0, 0, 4, 5, 0]
feasible_y = [0, 4, 2, 0, 0]
ax.fill(feasible_x, feasible_y, color='#3498DB', alpha=0.15, label='可行域 (Feasible Region)')

# 约束线
x_line = np.linspace(-0.5, 6, 200)

# x + 2y = 8 => y = (8-x)/2
y1 = (8 - x_line) / 2
ax.plot(x_line, y1, color='#E74C3C', lw=2, label='x + 2y ≤ 8')

# 2x + y = 10 => y = 10 - 2x
y2 = 10 - 2 * x_line
ax.plot(x_line, y2, color='#27AE60', lw=2, label='2x + y ≤ 10')

# x = 0 和 y = 0
ax.axvline(0, color='#7F8C8D', lw=1.5, ls='--', label='x ≥ 0')
ax.axhline(0, color='#7F8C8D', lw=1.5, ls='--', label='y ≥ 0')

# 顶点标注
vertices = [(0, 0, '(0, 0)'), (0, 4, '(0, 4)'), (4, 2, '(4, 2)'), (5, 0, '(5, 0)')]
for vx, vy, label in vertices:
    ax.scatter(vx, vy, s=100, color='#E74C3C', zorder=10, marker='o')
    ax.annotate(label, xy=(vx, vy), xytext=(vx+0.15, vy+0.2),
                fontsize=11, fontweight='bold', color='#2C3E50')

# 目标函数等值线: max z = 3x + 2y, 绘制 z=0, z=6, z=12
for z_val, ls in [(6, '--'), (10, '--'), (12, ':')]:
    y_obj = (z_val - 3*x_line) / 2
    mask = (y_obj >= 0) & (y_obj <= 6)
    ax.plot(x_line[mask], y_obj[mask], color='#F39C12', lw=1.5, ls=ls, alpha=0.6)
    mid = len(x_line[mask]) // 2
    ax.text(x_line[mask][mid], y_obj[mask][mid], f'z={z_val}', fontsize=8, color='#F39C12')

# 最优点
ax.scatter(4, 2, s=200, color='#F39C12', marker='*', zorder=15, label='最优解 (4, 2)')

ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 6)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('线性规划可行域\n(max  z = 3x + 2y,  s.t.  x+2y≤8,  2x+y≤10,  x,y≥0)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')

plt.tight_layout()
savefig(fig, 'fig_lp_feasible_region.png')

# ========== 图3: ACF / PACF ==========
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

np.random.seed(42)
# AR(1) 序列: x_t = 0.7 * x_{t-1} + noise
ar1 = np.zeros(100)
for t in range(1, 100):
    ar1[t] = 0.7 * ar1[t-1] + np.random.randn()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(ar1, lags=20, ax=axes[0], color='#3498DB', vlines_kwargs={'color': '#3498DB'})
axes[0].set_title('自相关函数 (ACF)\n(AR(1) 过程: 缓慢衰减)', fontsize=12)
axes[0].set_xlabel('滞后 (lag)', fontsize=11)
axes[0].set_ylabel('ACF', fontsize=11)

plot_pacf(ar1, lags=20, ax=axes[1], color='#E74C3C', vlines_kwargs={'color': '#E74C3C'})
axes[1].set_title('偏自相关函数 (PACF)\n(AR(1): 一阶滞后显著，后续截断)', fontsize=12)
axes[1].set_xlabel('滞后 (lag)', fontsize=11)
axes[1].set_ylabel('PACF', fontsize=11)

fig.suptitle('ACF / PACF: 识别时间序列模型类型', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_acf_pacf.png')

print("时间序列+优化图生成完成！")
