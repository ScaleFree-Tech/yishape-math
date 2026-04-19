#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图4.1/4.2 统计分布全景
8张子图：6大分布对比、Q-Q图、置信区间与分布
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

x = np.linspace(-5, 5, 400)

# ========== 图4.1.1 六分布对比 ==========
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
dists = [
    ('正态分布 N(0,1)', stats.norm, '#3498DB'),
    ('t 分布 (df=5)', lambda: stats.t(5), '#E74C3C'),
    ('t 分布 (df=30)', lambda: stats.t(30), '#27AE60'),
    ('卡方分布 χ2(4)', lambda: stats.chi2(4), '#9B59B6'),
    ('F 分布 F(4,20)', lambda: stats.f(4,20), '#F39C12'),
    ('指数分布 Exp(1)', lambda: stats.expon, '#1ABC9C'),
]
for ax, (title, dist_fn, color) in zip(axes.flat, dists):
    dist = dist_fn()
    y = dist.pdf(x)
    ax.fill_between(x, y, alpha=0.25, color=color)
    ax.plot(x, y, color=color, lw=2.5, label=dist_fn.__name__ if callable(dist) else '')
    # 标注均值
    mean = dist.mean()
    ax.axvline(mean, color=color, ls='--', lw=1.5, alpha=0.8)
    ax.scatter([mean], [dist.pdf(mean)], color=color, s=50, zorder=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('概率密度')
    ax.set_xlim(-5, 5); ax.set_ylim(0, None)
    ax.text(mean+0.1, dist.pdf(mean)+0.05, f'μ={mean:.2f}', fontsize=9, color=color)
savefig(fig, 'fig_4_1_1_six_distributions.png')

# ========== 图4.1.2 CLT 详细演示 ==========
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
sample_sizes = [1, 2, 5, 30]
for ax, n in zip(axes.flat, sample_sizes):
    means = [np.mean(np.random.exponential(1, n)) for _ in range(5000)]
    ax.hist(means, bins=60, density=True, alpha=0.6, color='#3498DB', edgecolor='white', linewidth=0.3)
    # 叠加理论正态
    xh = np.linspace(min(means), max(means), 200)
    ax.plot(xh, stats.norm.pdf(xh, np.mean(means), np.std(means)),
            color='#E74C3C', lw=2, label=f'正态近似 N({np.mean(means):.3f},{np.std(means):.3f})')
    ax.set_title(f'样本量 n={n}，样本均值分布\n(总体为指数分布)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('样本均值'); ax.set_ylabel('概率密度')
savefig(fig, 'fig_4_1_2_clt_detail.png')

# ========== 图4.1.3 Q-Q 图 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
np.random.seed(42)
samples = [
    (np.random.normal(0, 1, 200), '正态分布样本', '#3498DB', 'Q-Q图接近直线，数据服从正态'),
    (np.random.exponential(1, 200), '指数分布样本', '#E74C3C', 'Q-Q图S形弯曲，数据右偏'),
    (np.random.uniform(-2, 2, 200), '均匀分布样本', '#27AE60', 'Q-Q图S形反向弯曲，数据均匀'),
]
for ax, (data, title, color, note) in zip(axes, samples):
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'{title}\n{note}', fontsize=11, fontweight='bold')
    ax.get_lines()[0].set_color(color); ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color('gray'); ax.get_lines()[1].set_linewidth(2)
    ax.set_xlabel('理论分位数'); ax.set_ylabel('样本分位数')
savefig(fig, 'fig_4_1_3_qq_plot.png')

# ========== 图4.1.4 置信区间覆盖概率 ==========
fig, ax = plt.subplots(figsize=(12, 6))
n_sim = 200; n_sample = 30; mu, sigma = 5.0, 2.0
ci_lower, ci_upper, covers = [], [], 0
for _ in range(n_sim):
    s = np.random.normal(mu, sigma, n_sample)
    se = sigma / np.sqrt(n_sample)
    lo, hi = s.mean() - 1.96*se, s.mean() + 1.96*se
    ci_lower.append(lo); ci_upper.append(hi)
    covers += 1 if lo <= mu <= hi else 0
y_idx = range(n_sim)
for i in y_idx:
    color = '#27AE60' if ci_lower[i] <= mu <= ci_upper[i] else '#E74C3C'
    ax.plot([ci_lower[i], ci_upper[i]], [i, i], color=color, lw=1.5, alpha=0.7)
ax.axvline(mu, color='#3498DB', lw=2, ls='--', label=f'真实均值 μ={mu}')
ax.set_xlabel('参数值', fontsize=12); ax.set_ylabel('模拟序号', fontsize=12)
ax.set_title(f'95% 置信区间演示（{n_sim}次模拟）\n'
             f'覆盖真实均值：{covers}/{n_sim} = {covers/n_sim*100:.1f}%（理论95%）',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
# 添加图例说明
ax.text(0.02, 0.98, f'绿色：覆盖均值\n红色：未覆盖', transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
savefig(fig, 'fig_4_1_4_confidence_interval.png')

# ========== 图4.1.5 p值与拒绝域 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# 左图：双侧检验
ax = axes[0]
xp = np.linspace(-4, 4, 400)
yp = stats.norm.pdf(xp)
ax.fill_between(xp, yp, alpha=0.15, color='#3498DB')
ax.plot(xp, yp, color='#3498DB', lw=2)
# 拒绝域
x_rej_l = xp[xp < -1.96]; x_rej_r = xp[xp > 1.96]
ax.fill_between(x_rej_l, stats.norm.pdf(x_rej_l), alpha=0.5, color='#E74C3C', label='拒绝域 (α=0.05)')
ax.fill_between(x_rej_r, stats.norm.pdf(x_rej_r), alpha=0.5, color='#E74C3C')
ax.axvline(1.96, color='#E74C3C', lw=1.5, ls='--')
ax.axvline(-1.96, color='#E74C3C', lw=1.5, ls='--')
# 假设检验的功效曲线
ax2 = ax.twinx()
pow_x = np.linspace(-3, 3, 200)
# 功效 = P(拒绝H0 | H1为真)
effect_sizes = np.linspace(0, 3, 100)
power = [1 - stats.norm.cdf(1.96 - es) + stats.norm.cdf(-1.96 - es) for es in effect_sizes]
ax2.plot(effect_sizes, power, color='#27AE60', lw=2, ls='--', label='检验功效')
ax2.set_ylabel('检验功效', color='#27AE60', fontsize=11)
ax2.set_ylim(0, 1.05); ax2.legend(fontsize=10, loc='lower right')
ax.set_title('双侧假设检验的拒绝域与功效函数', fontsize=12, fontweight='bold')
ax.set_xlabel('z 值'); ax.set_ylabel('概率密度', color='#3498DB')
ax.legend(fontsize=10, loc='upper left')
# 右图：不同p值的图示
ax = axes[1]
for pval, color, label in [(0.01, '#E74C3C', 'p=0.01 强拒绝'),
                             (0.05, '#F39C12', 'p=0.05 拒绝'),
                             (0.20, '#27AE60', 'p=0.20 不拒绝')]:
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, stats.norm.pdf(xs), color=color, lw=2, label=label)
    # 标注z值
    z_crit = stats.norm.ppf(1 - pval/2)
    ax.fill_between(xs[xs > z_crit], stats.norm.pdf(xs[xs > z_crit]),
                    alpha=0.2, color=color)
ax.axvline(0, color='gray', lw=1, ls=':')
ax.set_title('不同 p 值对应的分布尾面积', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlabel('z 值'); ax.set_ylabel('概率密度')
savefig(fig, 'fig_4_1_5_pvalue_region.png')

# ========== 图4.1.6 I类误差与II类误差 ==========
fig, ax = plt.subplots(figsize=(12, 6))
xrange = np.linspace(-4, 6, 400)
# H0 分布
ax.fill_between(xrange, stats.norm.pdf(xrange, 0, 1), alpha=0.2, color='#3498DB', label='H0: μ=0（零假设）')
ax.plot(xrange, stats.norm.pdf(xrange, 0, 1), color='#3498DB', lw=2)
# H1 分布
ax.fill_between(xrange, stats.norm.pdf(xrange, 2.5, 1), alpha=0.2, color='#E74C3C', label='H1: μ=2.5（备择假设）')
ax.plot(xrange, stats.norm.pdf(xrange, 2.5, 1), color='#E74C3C', lw=2)
# 临界值
crit = 1.645
y_crit = stats.norm.pdf(crit, 0, 1)
ax.axvline(crit, color='#9B59B6', lw=2, ls='--', label=f'临界值 c={crit}')
ax.fill_between(xrange[xrange > crit], stats.norm.pdf(xrange[xrange > crit], 0, 1),
                alpha=0.4, color='#3498DB', label=f'α = P(拒绝H0|H0为真) = {1-stats.norm.cdf(crit):.3f}')
ax.fill_between(xrange[xrange < crit], stats.norm.pdf(xrange[xrange < crit], 2.5, 1),
                alpha=0.4, color='#E74C3C', label=f'β = P(接受H0|H1为真) = {stats.norm.cdf(crit-2.5):.3f}')
power = 1 - stats.norm.cdf(crit-2.5)
ax.fill_between(xrange[xrange > crit], stats.norm.pdf(xrange[xrange > crit], 2.5, 1),
                alpha=0.6, color='#27AE60', label=f'功效 = 1-β = {power:.3f}')
ax.set_title('假设检验的两类错误\n'
             'α（I类错误）：错误地拒绝真实的零假设\n'
             'β（II类错误）：错误地接受假的零假设',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlabel('检验统计量值', fontsize=12)
savefig(fig, 'fig_4_1_6_type12_error.png')

print("统计分布 figures done!")
