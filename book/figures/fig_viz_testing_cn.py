#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图3.x 可视化类型 + 图4.3/4.4 假设检验与ANOVA
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

# ========== 图3.1.1 常见可视化类型对比 ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
np.random.seed(42)
# 散点图
ax = axes[0,0]
x = np.random.randn(100); y = 2*x + np.random.randn(100)*0.5
ax.scatter(x, y, s=30, alpha=0.6, color='#3498DB')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('散点图（Scatter Plot）\n'
             '展示两个连续变量的相关性', fontsize=11, fontweight='bold')
# 折线图
ax = axes[0,1]
months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
sales = [35,28,42,55,68,75,82,78,65,52,45,38]
ax.plot(months, sales, 'o-', color='#3498DB', lw=2, ms=6)
ax.fill_between(range(12), sales, alpha=0.15, color='#3498DB')
ax.set_title('折线图（Line Chart）\n展示数据随时间变化的趋势', fontsize=11, fontweight='bold')
ax.set_ylabel('销售额（万元）')
# 柱状图
ax = axes[0,2]
categories = ['北京','上海','深圳','广州','杭州']
revenue = [120, 95, 78, 65, 52]
bars = ax.bar(categories, revenue, color=['#3498DB','#27AE60','#9B59B6','#E74C3C','#F39C12'], alpha=0.8)
for bar, val in zip(bars, revenue):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{val}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_title('柱状图（Bar Chart）\n比较分类数据的大小', fontsize=11, fontweight='bold')
ax.set_ylabel('收入（万元）')
# 饼图
ax = axes[1,0]
sizes = [35, 25, 18, 12, 10]
labels = ['电商平台', '线下门店', '批发渠道', '经销商', '其他']
colors = ['#3498DB','#27AE60','#9B59B6','#E74C3C','#F39C12']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.75)
for at in autotexts: at.set_fontsize(10); at.set_color('white')
ax.set_title('饼图（Pie Chart）\n展示各部分占总体的比例', fontsize=11, fontweight='bold')
# 热力图
ax = axes[1,1]
data = np.random.rand(5, 5)
data = np.abs(data) * 0.5 + np.diag([1, 0.9, 0.8, 0.7, 0.6])
products = ['产品A','产品B','产品C','产品D','产品E']
im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(products, fontsize=9); ax.set_yticklabels(products, fontsize=9)
ax.set_title('热力图（Heatmap）\n展示矩阵型数据的相关性或强度', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
# 箱线图
ax = axes[1,2]
groups = ['组A','组B','组C','组D']
data_box = [np.random.randn(50)*2+5, np.random.randn(50)*2+7, np.random.randn(50)*2+6, np.random.randn(50)*2+8]
bp = ax.boxplot(data_box, labels=groups, patch_artist=True)
colors_bp = ['#3498DB','#27AE60','#9B59B6','#E74C3C']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_title('箱线图（Box Plot）\n展示数据分布：Median/Q1/Q3/异常值', fontsize=11, fontweight='bold')
ax.set_ylabel('数值')
plt.tight_layout()
savefig(fig, 'fig_3_1_1_chart_types.png')

# ========== 图4.3.1 AB测试假设检验 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
np.random.seed(42)
# 模拟AB组数据
a_conv = np.random.normal(0.12, 0.02, 1000)
b_conv = np.random.normal(0.135, 0.02, 1000)
ax = axes[0]
ax.hist(a_conv, bins=40, density=True, alpha=0.6, color='#3498DB', label=f'A组均值={a_conv.mean():.4f}')
ax.hist(b_conv, bins=40, density=True, alpha=0.6, color='#E74C3C', label=f'B组均值={b_conv.mean():.4f}')
ax.axvline(a_conv.mean(), color='#3498DB', lw=2, ls='--')
ax.axvline(b_conv.mean(), color='#E74C3C', lw=2, ls='--')
ax.set_title('AB测试：两组转化率分布模拟\nH0：A组和B组无差异 | H1：B组更好',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('转化率'); ax.set_ylabel('概率密度')
# p值图示
ax = axes[1]
# 双样本t检验
t_stat, p_value = stats.ttest_ind(b_conv, a_conv)
x_range = np.linspace(-4, 6, 400)
t_dist = stats.t(df=len(a_conv)+len(b_conv)-2)
ax.fill_between(x_range[x_range < -abs(t_stat)], t_dist.pdf(x_range[x_range < -abs(t_stat)]),
                alpha=0.4, color='#3498DB', label=f'左尾（t=-{abs(t_stat):.2f}）')
ax.fill_between(x_range[x_range > abs(t_stat)], t_dist.pdf(x_range[x_range > abs(t_stat)]),
                alpha=0.4, color='#E74C3C', label=f'右尾（t=+{abs(t_stat):.2f}）')
ax.plot(x_range, t_dist.pdf(x_range), color='#34495E', lw=2)
ax.axvline(-abs(t_stat), color='#3498DB', lw=2, ls='--')
ax.axvline(abs(t_stat), color='#E74C3C', lw=2, ls='--')
ax.text(0, 0.25, f'p值 = {p_value:.4f}\n{"显著 ↑" if p_value<0.05 else "不显著"} (α=0.05)',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('双样本 t 检验的 t 分布\n'
             f'检验统计量 t = {t_stat:.3f}，p值 = {p_value:.4f}', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('t 值'); ax.set_ylabel('概率密度')
savefig(fig, 'fig_4_3_1_ab_test.png')

# ========== 图4.4.1 ANOVA 方差分析 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
np.random.seed(42)
# 三组数据
group1 = np.random.normal(85, 8, 30)
group2 = np.random.normal(90, 8, 30)
group3 = np.random.normal(95, 8, 30)
all_data = [group1, group2, group3]
# 箱线图
ax = axes[0]
bp = ax.boxplot(all_data, labels=['广告A','广告B','广告C'], patch_artist=True,
                widths=0.5)
colors_an = ['#3498DB','#27AE60','#E74C3C']
for patch, color in zip(bp['boxes'], colors_an):
    patch.set_facecolor(color); patch.set_alpha(0.6)
overall_mean = np.mean(np.concatenate(all_data))
for gm in [overall_mean]:
    ax.axhline(gm, color='#9B59B6', lw=2, ls='--', label=f'总均值={gm:.1f}')
ax.set_title('ANOVA 前提：各组分布相似，方差齐性\n'
             '三组广告的转化率箱线图比较', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_ylabel('转化率 (%)')
# F分布
ax = axes[1]
f_vals = np.linspace(0, 5, 400)
df1, df2 = 2, 87
f_dist = stats.f(df1, df2)
ax.plot(f_vals, f_dist.pdf(f_vals), color='#3498DB', lw=2.5)
# 拒绝域
f_crit = stats.f.ppf(0.95, df1, df2)
ax.fill_between(f_vals[f_vals>f_crit], f_dist.pdf(f_vals[f_vals>f_crit]),
                alpha=0.4, color='#E74C3C', label=f'拒绝域 α=0.05\n临界值 F*={f_crit:.3f}')
ax.axvline(f_crit, color='#E74C3C', lw=2, ls='--')
# 计算F统计量
ss_between = sum(len(g)*(np.mean(g)-overall_mean)**2 for g in all_data)
ss_total = sum(np.sum((g-overall_mean)**2) for g in all_data)
ms_between = ss_between / (3-1)
ms_within = (ss_total - ss_between) / (sum(len(g) for g in all_data) - 3)
f_stat = ms_between / ms_within
ax.axvline(f_stat, color='#27AE60', lw=2.5, ls=':', label=f'观测F={f_stat:.3f}')
ax.text(f_stat+0.15, 0.1, f'F = {f_stat:.3f}', fontsize=10, color='#27AE60', fontweight='bold')
p_anova = 1 - stats.f.cdf(f_stat, df1, df2)
ax.text(0.5, f_dist.pdf(f_crit)*0.7,
        f'p值 = {p_anova:.4f}\n{"显著 ↑" if p_anova<0.05 else "不显著"}',
        fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('F分布与ANOVA F检验\n'
             'H0：三组均值相等 | H1：至少一组不同', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('F 值'); ax.set_ylabel('概率密度')
savefig(fig, 'fig_4_4_1_anova.png')

# ========== 图4.3.2 多重比较校正（Bonferroni） ==========
fig, ax = plt.subplots(figsize=(12, 5))
n_tests = 20
# 模拟20个独立检验的p值
np.random.seed(42)
p_values = np.sort(np.concatenate([np.random.uniform(0, 0.1, 5), np.random.uniform(0.1, 0.9, 15)]))
tests = [f'检验{i+1}' for i in range(n_tests)]
x_pos = np.arange(n_tests)
# 未校正
ax.scatter(x_pos, p_values, s=60, color='#E74C3C', alpha=0.7, label='各检验 p 值', zorder=5)
ax.axhline(0.05, color='#E74C3C', lw=2, ls='--', label='α=0.05（未校正）')
# Bonferroni校正：α/n = 0.05/20 = 0.0025
bonf_threshold = 0.05 / n_tests
ax.axhline(bonf_threshold, color='#27AE60', lw=2, ls='--',
           label=f'Bonferroni: α/n = {bonf_threshold:.4f}')
# Benjamini-Hochberg
sorted_p = np.sort(p_values)
bh_thresholds = (np.arange(1, n_tests+1) / n_tests) * 0.05
ax.plot(x_pos, bh_thresholds, color='#3498DB', lw=2.5, label='Benjamini-Hochberg（控制FDR）')
sig_unadj = p_values < 0.05
sig_bonf = p_values < bonf_threshold
for i, (p, x) in enumerate(zip(p_values, x_pos)):
    color = '#27AE60' if sig_bonf[i] else '#E74C3C' if sig_unadj[i] else '#95A5A6'
    ax.scatter([x], [p], s=80, color=color, zorder=10)
ax.set_xticks(x_pos); ax.set_xticklabels(tests, rotation=60, ha='right', fontsize=8)
ax.set_xlabel('检验名称', fontsize=11); ax.set_ylabel('p 值', fontsize=11)
ax.set_title('多重检验校正：20个独立检验的 p 值\n'
             '未校正：5个显著 | Bonferroni：2个显著 | BH（控制FDR）：4个显著\n'
             'Bonferroni 过度保守；Benjamini-Hochberg 在控制假阳性比例的同时功效更高',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(-0.01, 1.01)
savefig(fig, 'fig_4_3_2_multiple_testing.png')

print("可视化与假设检验 figures done!")
