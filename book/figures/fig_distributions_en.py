#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate statistical distribution comparison figures - clean minimalist style"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy import stats

# 小清新风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
COLORS = {
    'normal': '#4C72B0',
    't': '#DD8452',
    'chi2': '#55A868',
    'f': '#C44E52',
    'hist': '#8C8C8C',
}

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== Fig 1: Distribution Comparison ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.linspace(-5, 5, 500)
ax = axes[0]
ax.plot(x, stats.norm.pdf(x), color=COLORS['normal'], lw=2.5, label='Normal N(0,1)')
for df, color in [(2, '#E8A87C'), (5, '#C38D9E'), (10, '#85DCBA')]:
    ax.plot(x, stats.t.pdf(x, df), color=color, lw=1.8, label=f't(df={df})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('t-Distribution vs Normal\n(Tails get thinner as df increases)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)

ax = axes[1]
x_chi2 = np.linspace(0, 20, 500)
for df, color in [(2, '#A8D5BA'), (5, '#5CB85C'), (10, '#2E7D32'), (15, '#1B5E20')]:
    ax.plot(x_chi2, stats.chi2.pdf(x_chi2, df), color=color, lw=2, label=f'Chi2(df={df})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Chi-Squared Distribution\n(Becomes more symmetric as df increases)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 20)

fig.suptitle('Common Statistical Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_distribution_comparison.png')

# ========== Fig 2: F Distribution ==========
fig, ax = plt.subplots(figsize=(7, 4.5))
x_f = np.linspace(0.01, 5, 500)
for d1, d2, color in [(1, 5, '#FF6B6B'), (5, 5, '#4ECDC4'), (10, 10, '#45B7D1'), (50, 50, '#96CEB4')]:
    ax.plot(x_f, stats.f.pdf(x_f, d1, d2), color=color, lw=2, label=f'F(d1={d1}, d2={d2})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('F Distribution\n(Ratio of two Chi-Squared, core of ANOVA)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 5)
plt.tight_layout()
savefig(fig, 'fig_f_distribution.png')

# ========== Fig 3: Central Limit Theorem ==========
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
sample_sizes = [1, 2, 5, 30]
distributions = [
    ('Uniform', lambda n: np.random.uniform(0, 1, n)),
    ('Exponential', lambda n: np.random.exponential(1, n)),
    ('Bernoulli', lambda n: np.random.choice([0, 1], n)),
]

for row, (dist_name, dist_fn) in enumerate(distributions):
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col]
        ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        means = [np.mean(dist_fn(n)) for _ in range(3000)]
        ax.hist(means, bins=40, color=COLORS['hist'], alpha=0.6, density=True)
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(labelsize=8)

fig.suptitle('Central Limit Theorem: Any Distribution -> Normal Distribution\n(Sample mean distribution approaches normal as n increases)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_central_limit_theorem.png')

print("Distribution figures done!")
