#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增图批次5：收尾——Ch3可视化 + Ch4高级 + Ch5高级"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}', dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")

# ============================================================
# 图4.5.1 共轭先验示意
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('图4.5.1 共轭先验分布 | Conjugate Prior Distributions', fontsize=14, fontweight='bold')

# Beta-Binomial 共轭
ax = axes[0]
theta = np.linspace(0.001, 0.999, 300)
# 先验 Beta(2,2)
prior = scipy_stats.beta.pdf(theta, 2, 2)
# 不同似然后验
for n_success, n_total, style in [(3, 10, 'solid'), (7, 10, 'dashed'), (0, 10, 'dotted')]:
    post = scipy_stats.beta.pdf(theta, 2+n_success, 2+(n_total-n_success))
    label_post = f'后验Beta({2+n_success},{2+n_total-n_success})' if n_total > 0 else '先验Beta(2,2)'
    ax.plot(theta, prior if n_total == 10 else post, lw=2.5,
            label=label_post, linestyle=style if n_total == 10 else 'solid')
    if n_total == 10:
        prior = post  # Update for next iteration

# 重新画三条后验曲线
for n_s, n_t, col in [(3,10,'#E74C3C'), (7,10,'#27AE60'), (0,10,'#95A5A6')]:
    post_curve = scipy_stats.beta.pdf(theta, 2+n_s, 2+(n_t-n_s))
    ax.plot(theta, post_curve, lw=2.5, color=col,
            label=f'后验: n={n_t}, 成功={n_s}')

# 先验
ax.plot(theta, scipy_stats.beta.pdf(theta, 2, 2), 'k--', lw=2, label='先验 Beta(2,2)')
ax.set_xlabel('θ (硬币正面概率)', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('Beta-Binomial共轭\n先验Beta(α,β) → 后验Beta(α+s, β+f)', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 's=成功次数, f=失败次数', transform=ax.transAxes, fontsize=8,
        va='top', bbox=dict(boxstyle='round', facecolor='#FFF9C4'))

# Normal-Normal 共轭
ax2 = axes[1]
mu_range = np.linspace(-2, 6, 300)
# 先验 N(2, 1)
prior_n = scipy_stats.norm.pdf(mu_range, 2, 1)
ax2.plot(mu_range, prior_n, 'k--', lw=2, label='先验 N(μ₀=2, σ₀=1)')

# 不同数据下的后验
for data_mean, n, col in [(5, 5, '#E74C3C'), (5, 20, '#27AE60'), (3, 10, '#9B59B6')]:
    sigma2 = 1.0  # 数据方差已知
    # 后验均值
    post_var = 1 / (1/sigma2*n + 1/1**2)
    post_mean = post_var * (data_mean*n/sigma2 + 2/1**2)
    post_curve = scipy_stats.norm.pdf(mu_range, post_mean, np.sqrt(post_var))
    ax2.plot(mu_range, post_curve, lw=2.5, color=col,
             label=f'n={n}, x̄={data_mean} → N({post_mean:.1f}, {np.sqrt(post_var):.2f})')

ax2.set_xlabel('μ (总体均值)', fontsize=11)
ax2.set_ylabel('概率密度', fontsize=11)
ax2.set_title('Normal-Normal共轭\n数据越多，后验越集中在数据均值附近', fontsize=12)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)

# Dirichlet-Multinomial 共轭
ax3 = axes[2]
# 三分类的狄利克雷分布可视化（用三角形表示）
## Triangle not available, using RegularPolygon instead
ax3.axis('off')
ax3.set_title('Dirichlet-Multinomial共轭\n（三分类）', fontsize=12)

# Simplex triangle
triangle = plt.matplotlib.patches.RegularPolygon((0.5, np.sqrt(3)/6), 3, radius=0.8,
                                                  facecolor='#F5F5F5', edgecolor='#333', lw=2)
ax3.add_patch(triangle)
ax3.set_xlim(-0.2, 1.2)
ax3.set_ylim(-0.1, 1.0)
ax3.set_aspect('equal')

# 标注顶点
ax3.text(0.5, 0.93, 'p₁=1, p₂=0, p₃=0', ha='center', fontsize=9)
ax3.text(-0.08, -0.05, 'p₁=0, p₂=1, p₃=0', ha='center', fontsize=9)
ax3.text(1.08, -0.05, 'p₁=0, p₂=0, p₃=1', ha='center', fontsize=9)

# 画等高线表示 Dirichlet 分布
# Sample from Dirichlet and plot points
np.random.seed(42)
alpha = np.array([2., 3., 4.])
samples = np.random.dirichlet(alpha, 200)
# 转换到三角形坐标系
for s in samples:
    p1, p2, p3 = s
    # 在等边三角形中，坐标为
    x_tri = 0.5 * (2*p2 + p3) / (p1 + p2 + p3)
    y_tri = (np.sqrt(3)/2) * p3 / (p1 + p2 + p3)
    ax3.scatter(x_tri * 0.8 + 0.1, y_tri * 0.8 + 0.05, s=15, alpha=0.3, color='#E74C3C')

ax3.text(0.5, 0.3, f'Dirichlet(α={alpha})\n共轭于多项分布', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_4_5_1_conjugate_prior.png')

# ============================================================
# 图4.6.1 统计功效分析
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图4.6.1 统计功效与样本量 | Power Analysis', fontsize=14, fontweight='bold')

ax = axes[0]
# H0 和 H1 分布
mu0, sigma = 100., 15.
effect_sizes = [5, 10, 15]  # 效应量
colors_e = ['#27AE60', '#F39C12', '#E74C3C']

x_range = np.linspace(70, 130, 300)
# H0 分布
ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu0, sigma), 'k-', lw=2.5, label='H₀: μ=100')

# Critical value at alpha=0.05 (one-tailed)
alpha = 0.05
z_crit = scipy_stats.norm.ppf(1-alpha)
x_crit = mu0 + z_crit * sigma
ax.axvline(x_crit, color='red', lw=2, ls='--', label=f'拒绝域临界值 x*={x_crit:.1f}')
ax.fill_between(x_range[x_range >= x_crit], scipy_stats.norm.pdf(x_range[x_range >= x_crit], mu0, sigma),
                alpha=0.2, color='red', label=f'α=0.05 (I类错误)')

for eff, col in zip(effect_sizes, colors_e):
    mu1 = mu0 + eff
    power = 1 - scipy_stats.norm.cdf(x_crit, mu1, sigma)
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu1, sigma), color=col, lw=2,
            label=f'H₁: μ={mu1} (效应量d={eff/15:.1f}, 功效={power:.2f})')
    ax.fill_between(x_range[x_range >= x_crit], scipy_stats.norm.pdf(x_range[x_range >= x_crit], mu1, sigma),
                    alpha=0.1, color=col)

ax.set_xlabel('样本均值 x̄', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('假设检验功效分析（单样本t检验）', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(70, 130)
ax.grid(True, alpha=0.3)

# 右图：功效曲线（样本量 vs 功效）
ax2 = axes[1]
effect_sizes_n = np.array([5, 10, 15])
sample_sizes = np.arange(5, 200, 2)
for eff, col in zip(effect_sizes_n, colors_e):
    powers = []
    for n in sample_sizes:
        se = sigma / np.sqrt(n)
        z_crit_n = scipy_stats.norm.ppf(1-alpha)
        power_n = 1 - scipy_stats.norm.cdf(z_crit_n - eff/se)
        powers.append(power_n)
    ax2.plot(sample_sizes, powers, color=col, lw=2.5,
             label=f'效应量 d={eff/15:.1f} (Δμ={eff})')

ax2.axhline(0.8, color='gray', ls='--', lw=1.5, label='常用功效阈值 0.8')
ax2.axhline(0.9, color='gray', ls=':', lw=1.5, label='严格功效阈值 0.9')
ax2.set_xlabel('样本量 n', fontsize=11)
ax2.set_ylabel('统计功效 (1 - β)', fontsize=11)
ax2.set_title('功效曲线：n增大 → 功效增大', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 200)
ax2.set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_4_6_1_power_test.png')

# ============================================================
# 图5.2.4 朴素贝叶斯概率
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.2.4 朴素贝叶斯概率图解 | Naive Bayes Probability', fontsize=14, fontweight='bold')

np.random.seed(42)
# 两类的特征分布（条件独立假设）
n_pts = 60
class0_x1 = np.random.randn(n_pts)*1.5 + 2
class0_x2 = np.random.randn(n_pts)*1.2 + 4
class1_x1 = np.random.randn(n_pts)*1.5 + 5
class1_x2 = np.random.randn(n_pts)*1.2 + 6

ax = axes[0]
ax.scatter(class0_x1, class0_x2, color='#E74C3C', s=50, alpha=0.6, label='类别 Spam (0)', edgecolors='white')
ax.scatter(class1_x1, class1_x2, color='#3498DB', s=50, alpha=0.6, label='类别 Ham (1)', edgecolors='white')

# 特征条件分布曲线
x1_range = np.linspace(0, 8, 200)
# 估计分布
from scipy.stats import gaussian_kde
kde_c0_x1 = gaussian_kde(class0_x1)
kde_c1_x1 = gaussian_kde(class1_x1)
kde_c0_x2 = gaussian_kde(class0_x2)
kde_c1_x2 = gaussian_kde(class1_x2)

ax2_twin = ax.twinx()
ax2_twin.plot(x1_range, kde_c0_x1(x1_range), 'r-', lw=2, alpha=0.5)
ax2_twin.plot(x1_range, kde_c1_x1(x1_range), 'b-', lw=2, alpha=0.5)
ax2_twin.set_ylabel('P(x₁|类别)', fontsize=10, color='gray')
ax2_twin.tick_params(axis='y', labelcolor='gray')

# 标注一个测试点
test_point = (4.5, 5.5)
ax.scatter([test_point[0]], [test_point[1]], color='yellow', s=200, marker='*', zorder=10,
           label=f'测试点 x=({test_point[0]},{test_point[1]})', edgecolors='black', lw=1.5)

ax.set_xlabel('特征 x₁ (如：含"免费"关键词数)', fontsize=11)
ax.set_ylabel('特征 x₂ (如：链接数量)', fontsize=11)
ax.set_title('Spam/Ham 特征分布（朴素：假设特征条件独立）', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# 右图：贝叶斯公式 + 计算
ax2 = axes[1]
ax2.axis('off')
ax2.set_title('朴素贝叶斯分类计算示例', fontsize=12)

# 先验
P_spam = n_pts / (2*n_pts)
P_ham = n_pts / (2*n_pts)

# 测试点的似然
x1_test, x2_test = test_point
P_x1_spam = kde_c0_x1(x1_test)[0]
P_x1_ham = kde_c1_x1(x1_test)[0]
P_x2_spam = kde_c0_x2(x2_test)[0]
P_x2_ham = kde_c1_x2(x2_test)[0]

# 朴素贝叶斯：P(y|X) ∝ P(y)·∏P(xᵢ|y)
# 注意：这里用密度代替概率，实际中用离散计数或密度
calc_text = f"""
邮件分类计算示例
━━━━━━━━━━━━━━━━━━━━━━
先验概率：
  P(Spam) = {P_spam:.2f}
  P(Ham)  = {P_ham:.2f}

测试点特征：
  x₁ = {x1_test} (关键词数)
  x₂ = {x2_test} (链接数)

特征似然（从分布密度估算）：
  P(x₁|Spam) = {P_x1_spam:.4f}    P(x₁|Ham) = {P_x1_ham:.4f}
  P(x₂|Spam) = {P_x2_spam:.4f}    P(x₂|Ham) = {P_x2_ham:.4f}

朴素贝叶斯（特征独立）：
  P(Spam|X) ∝ P(Spam)·P(x₁|Spam)·P(x₂|Spam)
            = {P_spam:.2f} × {P_x1_spam:.4f} × {P_x2_spam:.4f}
            = {P_spam * P_x1_spam * P_x2_spam:.6f}

  P(Ham|X) ∝ P(Ham)·P(x₁|Ham)·P(x₂|Ham)
            = {P_ham:.2f} × {P_x1_ham:.4f} × {P_x2_ham:.4f}
            = {P_ham * P_x1_ham * P_x2_ham:.6f}

归一化：
  P(Spam|X) = {P_spam * P_x1_spam * P_x2_spam / (P_spam * P_x1_spam * P_x2_spam + P_ham * P_x1_ham * P_x2_ham):.4f}
  P(Ham|X)  = {P_ham * P_x1_ham * P_x2_ham / (P_spam * P_x1_spam * P_x2_spam + P_ham * P_x1_ham * P_x2_ham):.4f}

→ 分类结果：{'Spam ✗' if P_spam * P_x1_spam * P_x2_spam > P_ham * P_x1_ham * P_x2_ham else 'Ham ✓'}
"""
ax2.text(0.05, 0.95, calc_text, transform=ax2.transAxes,
         fontsize=9, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.95, edgecolor='#ccc'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_2_4_naive_bayes.png')

# ============================================================
# 图5.2.5 决策树分裂示意
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.2.5 决策树分裂 | Decision Tree Splitting', fontsize=14, fontweight='bold')

np.random.seed(42)
n_tree = 200
data_split = {'x1': np.random.uniform(0, 10, n_tree),
              'x2': np.random.uniform(0, 10, n_tree)}
# 真实决策边界
import pandas as pd
df_split = pd.DataFrame(data_split)
df_split['y'] = ((df_split['x1'] > 5) & (df_split['x2'] > 4)).astype(int)

ax = axes[0]
colors_y = ['#E74C3C' if y == 0 else '#27AE60' for y in df_split['y']]
ax.scatter(df_split['x1'], df_split['x2'], c=colors_y, s=50, alpha=0.7, edgecolors='white')

# 画分裂边界
ax.axvline(5, color='#9B59B6', lw=2.5, ls='-', label='第1次分裂: x₁ ≤ 5?')
ax.axhline(4, color='#F39C12', lw=2.5, ls='-', label='第2次分裂: x₂ ≤ 4?')
ax.fill_between([0, 5], 0, [4, 4], alpha=0.1, color='#E74C3C')
ax.fill_between([5, 10], 0, [4, 4], alpha=0.1, color='#27AE60')
ax.fill_between([0, 5], 4, [10, 10], alpha=0.1, color='#27AE60')
ax.fill_between([5, 10], 4, [10, 10], alpha=0.1, color='#E74C3C')

ax.set_xlabel('x₁', fontsize=11)
ax.set_ylabel('x₂', fontsize=11)
ax.set_title('决策树分裂：轴平行分割决策边界', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 树结构图
ax2 = axes[1]
ax2.axis('off')
ax2.set_title('决策树结构', fontsize=12)

# Draw tree nodes manually
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_node(x, y, text, color, ax, w=0.15, h=0.08):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02',
                          facecolor=color, edgecolor='black', lw=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5, fontweight='bold')

def draw_arrow(x1, y1, x2, y2, label, ax):
    ax.annotate('', xy=(x2, y2+0.04), xytext=(x1, y1-0.04),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text((x1+x2)/2, (y1+y2)/2+0.03, label, fontsize=7.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='#FFF9C4', alpha=0.8))

draw_node(0.5, 0.88, 'x₁ ≤ 5?', '#E3F2FD', ax2)
draw_arrow(0.5, 0.84, 0.25, 0.74, '是', ax2)
draw_arrow(0.5, 0.84, 0.75, 0.74, '否', ax2)

draw_node(0.25, 0.70, 'x₂ ≤ 4?', '#E3F2FD', ax2)
draw_node(0.75, 0.70, 'x₂ ≤ 4?', '#E3F2FD', ax2)
draw_arrow(0.25, 0.66, 0.12, 0.56, '是', ax2)
draw_arrow(0.25, 0.66, 0.38, 0.56, '否', ax2)
draw_arrow(0.75, 0.66, 0.62, 0.56, '是', ax2)
draw_arrow(0.75, 0.66, 0.88, 0.56, '否', ax2)

draw_node(0.12, 0.52, '类别0\n(n=50)', '#FFCDD2', ax2, w=0.14)
draw_node(0.38, 0.52, '类别1\n(n=48)', '#C8E6C9', ax2, w=0.14)
draw_node(0.62, 0.52, '类别1\n(n=52)', '#C8E6C9', ax2, w=0.14)
draw_node(0.88, 0.52, '类别0\n(n=50)', '#FFCDD2', ax2, w=0.14)

# 标注
ax2.text(0.5, 0.15,
         '决策树优点：可解释性强（IF-THEN规则）\n'
         '分裂依据：信息增益 / 基尼系数最大化\n'
         '深度过深 → 过拟合（需要剪枝）',
         ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))

ax2.set_xlim(0, 1)
ax2.set_ylim(0.1, 1.0)
ax2.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_2_5_decision_tree.png')

# ============================================================
# 图6.3.2 集合覆盖与TSP近似
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图6.3.2 集合覆盖与TSP | Set Cover & TSP Approximation', fontsize=14, fontweight='bold')

# 左图：集合覆盖
ax = axes[0]
ax.set_title('集合覆盖问题（NP难）', fontsize=12)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 7个需求点
points = [(2,3), (4,6), (5,2), (7,4), (8,8), (1,7), (6,1)]
for i, (px, py) in enumerate(points):
    circle = plt.Circle((px, py), 0.5, fill=True, color='#E74C3C', alpha=0.8)
    ax.add_patch(circle)
    ax.text(px, py, str(i+1), ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# 3个集合（广播塔位置）
sets = [(3, 3, '集合A', '#3498DB'), (6, 5, '集合B', '#27AE60'), (5, 1.5, '集合C', '#9B59B6')]
for sx, sy, name, col in sets:
    circle_cov = plt.Circle((sx, sy), 2.8, fill=True, color=col, alpha=0.15)
    ax.add_patch(circle_cov)
    ax.scatter([sx], [sy], color=col, s=150, marker='^', zorder=10, edgecolors='black', lw=1.5)
    ax.text(sx, sy+0.4, name, ha='center', fontsize=9, color=col, fontweight='bold')

ax.text(5, 0.2,
        '目标：用最少的集合覆盖所有需求点\n'
        '集合A覆盖{1,2,3}，集合B覆盖{2,4,5,7}，集合C覆盖{3,6,4}\n'
        '最优解：集合A+集合B=2个（覆盖{1,2,3,4,5,7}）',
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))

# 右图：TSP近似
ax2 = axes[1]
ax2.set_title('旅行商问题（TSP）近似算法', fontsize=12)
np.random.seed(42)
cities = [(np.random.rand()*8+1, np.random.rand()*8+1) for _ in range(10)]
cities_x = [c[0] for c in cities]
cities_y = [c[1] for c in cities]

ax2.scatter(cities_x, cities_y, color='#E74C3C', s=80, zorder=10, edgecolors='black', lw=1)

# 最近邻近似解
def nearest_neighbor_tsp(cities):
    n = len(cities)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n-1):
        last = tour[-1]
        nearest = min([(i, np.hypot(cities[last][0]-cities[i][0], cities[last][1]-cities[i][1]))
                      for i in range(n) if not visited[i]], key=lambda x: x[1])[0]
        visited[nearest] = True
        tour.append(nearest)
    tour.append(0)  # 返回起点
    return tour

tour = nearest_neighbor_tsp(cities)
tour_x = [cities[tour[i]][0] for i in range(len(tour))]
tour_y = [cities[tour[i]][1] for i in range(len(tour))]

ax2.plot(tour_x, tour_y, 'b-', lw=2, alpha=0.8, label='最近邻近似解', zorder=5)

# 城市编号
for i, (cx, cy) in enumerate(cities):
    ax2.text(cx+0.15, cy+0.15, str(i+1), fontsize=9, fontweight='bold', color='#333')

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 计算近似解路径长度
tour_length = sum(np.hypot(tour_x[i]-tour_x[i+1], tour_y[i]-tour_y[i+1]) for i in range(len(tour)-1))
ax2.text(5, 0.2,
         f'最近邻启发式路径长度 ≈ {tour_length:.1f}\n'
         f'最优解下界（贪婪）≈ {tour_length*0.75:.1f}\n'
         f'近似比 ≤ 2（最近邻保证）',
         ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_6_3_2_cover_tsp.png')

print("批次5完成!")
