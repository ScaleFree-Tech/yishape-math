#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time series decomposition and LP feasible region figures"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== Fig 1: Time Series Decomposition ==========
np.random.seed(42)
n = 200
base = np.datetime64('2022-01-01')
dates = np.array([base + np.timedelta64(i, 'D') for i in range(n)])
trend = np.linspace(5, 15, n) + 2 * np.sin(np.linspace(0, 4*np.pi, n))
seasonal = 3 * np.sin(np.linspace(0, 2*np.pi * n/12, n))
resid = np.random.randn(n) * 1.2
y = trend + seasonal + resid

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
colors = {'trend': '#2C3E50', 'seasonal': '#27AE60', 'resid': '#E74C3C', 'observed': '#3498DB'}

axes[0].plot(dates, y, color=colors['observed'], lw=1.5, alpha=0.8)
axes[0].set_ylabel('Observed\ny = T + S + R', fontsize=10)
axes[0].set_title('Time Series Decomposition (STL: Trend + Seasonal + Residual)', fontsize=13, fontweight='bold')
axes[1].plot(dates, trend, color=colors['trend'], lw=2.5)
axes[1].set_ylabel('Trend', fontsize=10)
axes[2].plot(dates, seasonal, color=colors['seasonal'], lw=2)
axes[2].set_ylabel('Seasonal', fontsize=10)
axes[3].plot(dates, resid, color=colors['resid'], lw=1, alpha=0.7)
axes[3].axhline(0, color='gray', lw=0.8, ls='--')
axes[3].set_ylabel('Residual', fontsize=10)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[3].set_xlabel('Date', fontsize=11)
plt.tight_layout()
savefig(fig, 'fig_timeseries_decomposition.png')

# ========== Fig 2: LP Feasible Region ==========
fig, ax = plt.subplots(figsize=(8, 7))
feasible_x = [0, 0, 4, 5, 0]
feasible_y = [0, 4, 2, 0, 0]
ax.fill(feasible_x, feasible_y, color='#3498DB', alpha=0.15, label='Feasible Region')

x_line = np.linspace(-0.5, 6, 200)
y1 = (8 - x_line) / 2
ax.plot(x_line, y1, color='#E74C3C', lw=2, label='x + 2y <= 8')
y2 = 10 - 2 * x_line
ax.plot(x_line, y2, color='#27AE60', lw=2, label='2x + y <= 10')
ax.axvline(0, color='#7F8C8D', lw=1.5, ls='--', label='x >= 0')
ax.axhline(0, color='#7F8C8D', lw=1.5, ls='--', label='y >= 0')

vertices = [(0, 0, '(0, 0)'), (0, 4, '(0, 4)'), (4, 2, '(4, 2)'), (5, 0, '(5, 0)')]
for vx, vy, label in vertices:
    ax.scatter(vx, vy, s=100, color='#E74C3C', zorder=10, marker='o')
    ax.annotate(label, xy=(vx, vy), xytext=(vx+0.15, vy+0.2), fontsize=11, fontweight='bold', color='#2C3E50')

for z_val in [6, 10, 12]:
    y_obj = (z_val - 3*x_line) / 2
    mask = (y_obj >= 0) & (y_obj <= 6)
    if mask.sum() > 0:
        mid = len(x_line[mask]) // 2
        ax.plot(x_line[mask], y_obj[mask], color='#F39C12', lw=1.5, ls='--', alpha=0.6)
        ax.text(x_line[mask][mid], y_obj[mask][mid], f'z={z_val}', fontsize=8, color='#F39C12')

ax.scatter(4, 2, s=200, color='#F39C12', marker='*', zorder=15, label='Optimal (4, 2)')
ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 6)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Linear Programming Feasible Region\n(max z = 3x + 2y,  s.t.  x+2y<=8,  2x+y<=10,  x,y>=0)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')
plt.tight_layout()
savefig(fig, 'fig_lp_feasible_region.png')

# ========== Fig 3: ACF/PACF ==========
np.random.seed(42)
ar1 = np.zeros(100)
for t in range(1, 100):
    ar1[t] = 0.7 * ar1[t-1] + np.random.randn()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, func, title, color in [
    (axes[0], lambda lags: np.array([(ar1[:len(ar1)-lag] * ar1[lag:]).mean() / ar1.var() for lag in lags]), 'ACF', '#3498DB'),
    (axes[1], lambda lags: np.array([np.corrcoef(ar1[:-lag], ar1[lag:])[0,1] if lag > 0 else 1.0 for lag in lags]), 'PACF', '#E74C3C'),
]:
    lags = np.arange(0, 21)
    values = func(lags)
    ax.bar(lags, values, color=color, alpha=0.7, width=0.6)
    ax.axhline(1.96/np.sqrt(len(ar1)), color='gray', lw=1, ls='--')
    ax.axhline(-1.96/np.sqrt(len(ar1)), color='gray', lw=1, ls='--')
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title}: AR(1) Process\n(Slow decay = AR structure)', fontsize=12)
    ax.set_xlim(-0.5, 21)

fig.suptitle('ACF / PACF: Identifying Time Series Model Type', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_acf_pacf.png')

print("Time series and optimization figures done!")
