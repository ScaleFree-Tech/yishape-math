#!/usr/bin/env python3
"""
生成统计分布对比图 - 小清新风格
覆盖: 正态分布、t分布、卡方分布、F分布的比较
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CH_FONT = {'fontsize': 11, 'fontfamily': 'Noto Sans CJK JP'}

# 小清新风格设置
plt.style.use('seaborn-whitegrid')
sns.set_palette("muted")
COLORS = {
    'normal': '#4C72B0',      # 蓝色
    't': '#DD8452',            # 橙色
    'chi2': '#55A868',         # 绿色
    'f': '#C44E52',            # 红色
    'hist': '#8C8C8C',         # 灰色
}

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== 图1: 正态 vs t分布 (不同自由度) ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.linspace(-5, 5, 500)

# 左图: 不同自由度的t分布 vs 正态
ax = axes[0]
ax.plot(x, stats.norm.pdf(x), color=COLORS['normal'], lw=2.5, label='正态分布 N(0,1)')
for df, color in [(2, '#E8A87C'), (5, '#C38D9E'), (10, '#85DCBA')]:
    ax.plot(x, stats.t.pdf(x, df), color=color, lw=1.8, label=f't(df={df})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('t 分布 vs 正态分布\n(尾巴越来越薄，df→∞ 时趋于正态)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)

# 右图: 不同自由度的卡方分布
ax = axes[1]
x_chi2 = np.linspace(0, 20, 500)
for df, color in [(2, '#A8D5BA'), (5, '#5CB85C'), (10, '#2E7D32'), (15, '#1B5E20')]:
    ax.plot(x_chi2, stats.chi2.pdf(x_chi2, df), color=color, lw=2, label=f'χ²(df={df})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('卡方分布\n(自由度越大，分布越接近正态', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 20)

fig.suptitle('统计分布对比 / Distribution Comparisons', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_distribution_comparison.png')

# ========== 图2: F分布 ==========
fig, ax = plt.subplots(figsize=(7, 4.5))
x_f = np.linspace(0.01, 5, 500)
for d1, d2, color in [(1, 5, '#FF6B6B'), (5, 5, '#4ECDC4'), (10, 10, '#45B7D1'), (50, 50, '#96CEB4')]:
    label = f'F(d₁={d1}, d₂={d2})'
    ax.plot(x_f, stats.f.pdf(x_f, d1, d2), color=color, lw=2, label=label)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('F 分布\n(两个卡方分布之比，ANOVA 的核心工具)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 5)
plt.tight_layout()
savefig(fig, 'fig_f_distribution.png')

# ========== 图3: 中心极限定理可视化 ==========
fig, axes = plt.subplots(3, 4, figsize=(14, 9))

# 第一行: 不同分布的原始形状
sample_sizes = [1, 2, 5, 30]
distributions = [
    ('均匀分布', lambda n: np.random.uniform(0, 1, n)),
    ('指数分布', lambda n: np.random.exponential(1, n)),
    ('两点分布', lambda n: np.random.choice([0, 1], n)),
]

for row, (dist_name, dist_fn) in enumerate(distributions):
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col]
        if row == 0:
            ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        # 采样1000次，每次取n个样本，计算均值
        means = [np.mean(dist_fn(n)) for _ in range(3000)]
        ax.hist(means, bins=40, color=COLORS['hist'], alpha=0.6, density=True)
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(labelsize=8)

fig.suptitle('中心极限定理：从任意分布到正态分布\n(样本量越大，均值分布越接近正态)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig_central_limit_theorem.png')

print("统计分布图生成完成！")
