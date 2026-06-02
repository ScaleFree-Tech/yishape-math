#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增图批次1：Ch1线性代数 + Ch2 DataFrame"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipy_linalg
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}', dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")

# ============================================================
# 图1.7.1 高斯消元行变换
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图1.7.1 高斯消元法 | Gaussian Elimination', fontsize=14, fontweight='bold')

# 原始矩阵
A = np.array([[2., 1., -1.],
              [4., 5., -3.],
              [6., 5., -3.]])
b = np.array([2., 7., 9.])

# Step 1: 原始增广矩阵
axes[0].set_title('步骤1：原始增广矩阵 [A|b]', fontsize=12)
axes[0].axis('off')
table_data = []
for i in range(3):
    row = [f'{A[i,j]:5.1f}' for j in range(3)] + ['|', f'{b[i]:5.1f}']
    table_data.append(row)
col_labels = ['x₁', 'x₂', 'x₃', '', 'b']
table = axes[0].table(cellText=table_data, colLabels=col_labels,
                       cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(13)
for (r, c), cell in table.get_celld().items():
    if c == 3:
        cell.set_facecolor('#EEEEEE')

# Step 2: R2 <- R2 - 2*R1
axes[1].set_title('步骤2：R₂ ← R₂ − 2R₁（消去x₁）', fontsize=12)
axes[1].axis('off')
A2 = A.copy()
A2[1] = A2[1] - 2*A2[0]
b2 = b.copy()
b2[1] = b2[1] - 2*b2[0]
table_data2 = []
for i in range(3):
    row = [f'{A2[i,j]:5.1f}' for j in range(3)] + ['|', f'{b2[i]:5.1f}']
    table_data2.append(row)
table2 = axes[1].table(cellText=table_data2, colLabels=col_labels,
                        cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
table2.auto_set_font_size(False)
table2.set_fontsize(13)
for (r, c), cell in table2.get_celld().items():
    if r == 1 and c < 4:
        cell.set_facecolor('#C8E6C9')

# Step 3: 上三角
axes[2].set_title('步骤3：继续消元得上三角矩阵', fontsize=12)
axes[2].axis('off')
A3 = np.array([[2., 1., -1.],
               [0., 3., -1.],
               [0., 0., 2.]])
b3 = np.array([2., 3., 4.])
table_data3 = []
for i in range(3):
    row = [f'{A3[i,j]:5.1f}' for j in range(3)] + ['|', f'{b3[i]:5.1f}']
    table_data3.append(row)
table3 = axes[2].table(cellText=table_data3, colLabels=col_labels,
                        cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
table3.auto_set_font_size(False)
table3.set_fontsize(13)
for (r, c), cell in table3.get_celld().items():
    if c < 4:
        cell.set_facecolor('#BBDEFB')

# 标注说明
for ax in axes:
    ax.text(0.5, -0.05, '回代求解：x₃=2, x₂=(3+2)/3=5/3, x₁=(2-5/3+2)/2=...',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='#555')

plt.tight_layout(rect=[0, 0.08, 1, 0.97])
savefig(fig, 'fig_1_7_1_gauss_elimination.png')

# ============================================================
# 图1.8.1 LDL分解 vs Cholesky
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图1.8.1 LDL分解与Cholesky分解', fontsize=14, fontweight='bold')

# 对称正定矩阵
A_sym = np.array([[4., 2., 2.],
                  [2., 5., 3.],
                  [2., 3., 6.]], dtype=float)

# LDL分解 A = L @ D @ L.T
L, D = scipy_linalg.ldl(A_sym)[:2]
L_lower = np.tril(L)
D_diag = np.diag(D)

# 可视化 L 矩阵
im0 = axes[0].imshow(L_lower, cmap='Blues', vmin=-2, vmax=2)
axes[0].set_title('L（下三角单位矩阵）', fontsize=12)
axes[0].set_xlabel('列 j', fontsize=11)
axes[0].set_ylabel('行 i', fontsize=11)
for i in range(3):
    for j in range(3):
        if j <= i:
            val = L_lower[i,j]
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=12 if i != j else 10, color='black' if abs(val) > 0.5 else '#666')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# 可视化 D 对角矩阵
D_diag_vals = np.diag(D)
colors = ['#E74C3C' if v < 0 else '#27AE60' for v in D_diag_vals]
bars = axes[1].bar(range(1,4), D_diag_vals, color=colors, alpha=0.8, edgecolor='black')
axes[1].axhline(0, color='gray', lw=1)
axes[1].set_title('D（对角矩阵，对角元素）', fontsize=12)
axes[1].set_xlabel('对角位置', fontsize=11)
axes[1].set_ylabel('D对角元素值', fontsize=11)
axes[1].set_xticks([0,1,2])
axes[1].set_xticklabels(['D₁₁', 'D₂₂', 'D₃₃'])
for bar, val in zip(bars, D_diag_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1 if val > 0 else val - 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12)

# 验证公式
axes[1].text(0.5, -0.18, f'A = LDLᵀ\n验证：A − LDLᵀ =\n{np.max(np.abs(A_sym - L_lower @ D @ L_lower.T)):.1e}',
             transform=axes[1].transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

plt.tight_layout(rect=[0, 0.1, 1, 0.96])
savefig(fig, 'fig_1_8_1_ldl_cholesky.png')

# ============================================================
# 图1.9.1 条件数与病态
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图1.9.1 矩阵条件数 | Condition Number', fontsize=14, fontweight='bold')

# 好条件矩阵
A_good = np.array([[4., 1.],
                   [2., 3.]], dtype=float)
cond_good = np.linalg.cond(A_good)

# 病态矩阵（接近奇异）
A_bad = np.array([[4., 1.001],
                  [2., 1.]], dtype=float)
cond_bad = np.linalg.cond(A_bad)

# 可视化条件数影响
conds = [cond_good, cond_bad]
labels = [f'良态矩阵\ncond={cond_good:.1f}', f'病态矩阵\ncond={cond_bad:.0f}']
colors = ['#27AE60', '#E74C3C']

for idx, (ax, cond, label, color) in enumerate(zip(axes, conds, labels, colors)):
    # 画椭圆（矩阵对区域变换的可视化）
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    
    # 矩阵变换圆
    if idx == 0:
        transformed = A_good @ circle
    else:
        transformed = A_bad @ circle
    
    # 单位圆
    ax.plot(circle[0], circle[1], 'k--', lw=1.5, label='单位圆')
    # 变换后椭圆
    ax.plot(transformed[0], transformed[1], color=color, lw=2.5, label='变换后椭圆')
    
    # 画轴
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=12, color=color if idx == 1 else 'black')
    ax.legend(fontsize=10)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    # 标注短轴/长轴
    eigvals = np.linalg.eigvalsh(A_bad if idx == 1 else A_good)
    ax.text(0.05, 0.95, f'λ₁={max(eigvals):.2f}\nλ₂={min(eigvals):.2f}\ncond=λmax/λmin={cond:.1f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.94])
savefig(fig, 'fig_1_9_1_condition_number.png')

# ============================================================
# 图2.3.1 groupBy+聚合
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图2.3.1 分组聚合操作 | GroupBy + Aggregation', fontsize=14, fontweight='bold')

# 原始数据
np.random.seed(42)
depts = ['Engineering', 'Sales', 'HR', 'Engineering', 'Sales', 'HR', 'Engineering', 'Sales']
salaries = [8500, 6200, 5500, 9200, 7100, 5800, 8800, 6500]
years = [3, 1, 2, 5, 2, 1, 4, 3]
df_data = {'dept': depts, 'salary': salaries, 'years': years}

# 可视化1: 原始DataFrame
ax = axes[0]
ax.axis('off')
ax.set_title('原始数据表（8行×3列）', fontsize=11)
table_data = [[depts[i], f'{salaries[i]:,}', years[i]] for i in range(len(depts))]
col_labels = ['部门', '月薪', '年限']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center', bbox=[0.05, 0.1, 0.9, 0.85])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold')

# 分组后聚合
from collections import defaultdict
groups = defaultdict(list)
for i in range(len(depts)):
    groups[depts[i]].append((salaries[i], years[i]))

agg_data = []
for dept, vals in groups.items():
    salaries_g = [v[0] for v in vals]
    years_g = [v[1] for v in vals]
    agg_data.append([dept, f'{sum(salaries_g)/len(salaries_g):,.0f}', len(salaries_g), f'{sum(years_g)/len(years_g):.1f}'])

ax2 = axes[1]
ax2.axis('off')
ax2.set_title('GROUP BY 部门 → 聚合结果', fontsize=11)
col_labels2 = ['部门', '平均月薪', '人数', '平均年限']
table2 = ax2.table(cellText=agg_data, colLabels=col_labels2,
                    cellLoc='center', loc='center', bbox=[0.05, 0.1, 0.9, 0.7])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
for (r, c), cell in table2.get_celld().items():
    if r == 0:
        cell.set_facecolor('#388E3C')
        cell.set_text_props(color='white', fontweight='bold')

# 柱状图对比
ax3 = axes[2]
dept_names = [a[0] for a in agg_data]
avg_salaries = [float(a[1].replace(',','')) for a in agg_data]
colors_bar = ['#1976D2', '#E74C3C', '#27AE60']
bars = ax3.bar(dept_names, avg_salaries, color=colors_bar, alpha=0.85, edgecolor='black')
ax3.set_title('各部门平均月薪对比', fontsize=11)
ax3.set_ylabel('平均月薪（元）', fontsize=11)
ax3.set_xlabel('部门', fontsize=11)
for bar, val in zip(bars, avg_salaries):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 100, f'{val:,.0f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylim(0, max(avg_salaries)*1.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_2_3_1_groupby_agg.png')

# ============================================================
# 图2.3.2 DataFrame合并/关联
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图2.3.2 数据表关联 | DataFrame Merge / Join', fontsize=14, fontweight='bold')

# Left表: 员工信息
left_data = [['E1', 'Alice', 'Sales'],
             ['E2', 'Bob', 'Engineering'],
             ['E3', 'Carol', 'HR'],
             ['E4', 'Dave', 'Engineering']]
# Right表: 工资信息
right_data = [['E1', 7200],
              ['E3', 5800],
              ['E5', 9100],
              ['E2', 8500]]

for ax, (title, data, cols, highlight_fn, note) in zip(
    axes,
    [
        ('左表：员工信息', left_data, ['ID', 'Name', 'Dept'], None, '4行'),
        ('右表：工资信息', right_data, ['ID', 'Salary'], None, '4行'),
        ('INNER JOIN on ID', None, None,
         lambda l,r: [x for x in l if x[0] in [y[0] for y in r]],
         '只保留两边ID匹配的行'),
    ]):

    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    if data:
        n_cols = len(data[0])
        col_labels = cols
        table = ax.table(cellText=data, colLabels=col_labels,
                          cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.65])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1976D2')
                cell.set_text_props(color='white', fontweight='bold')
        ax.text(0.5, 0.08, note, transform=ax.transAxes, ha='center', fontsize=10, color='#666')
    else:
        # INNER JOIN result
        joined = [['E1', 'Alice', 'Sales', '7200'],
                  ['E2', 'Bob', 'Engineering', '8500'],
                  ['E3', 'Carol', 'HR', '5800']]
        col_labels_j = ['ID', 'Name', 'Dept', 'Salary']
        table = ax.table(cellText=joined, colLabels=col_labels_j,
                          cellLoc='center', loc='center', bbox=[0.05, 0.2, 0.9, 0.65])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#388E3C')
                cell.set_text_props(color='white', fontweight='bold')
            if r > 0 and c == 3:  # salary column
                cell.set_facecolor('#C8E6C9')
        ax.text(0.5, 0.08, 'E4和E5被丢弃（另一张表无匹配）', transform=ax.transAxes,
                ha='center', fontsize=10, color='#E74C3C',
                bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

        # 箭头表示连接
        ax.annotate('', xy=(-0.08, 0.3), xytext=(0.08, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
        ax.annotate('', xy=(-0.08, 0.5), xytext=(0.08, 0.3),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_2_3_2_merge_join.png')

# ============================================================
# 图2.4.1 数据清洗流水线
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('图2.4.1 数据清洗流水线 | Data Cleaning Pipeline', fontsize=14, fontweight='bold')
fig.text(0.5, 0.01, '原始数据 → 缺失值处理 → 异常值处理 → 去重 → 类型转换 → 干净数据',
         ha='center', fontsize=11, style='italic', color='#555')

np.random.seed(0)
n = 50

# Step 0: 原始数据（带缺失和异常）
dates = [f'2024-01-{i%28+1:02d}' for i in range(n)]
values = np.random.randn(n) * 50 + 100
values[5] = np.nan
values[12] = np.nan
values[20] = 500  # 异常值（离群点）
values[33] = -100  # 异常值
df_raw = {'date': dates[:n], 'value': values}
cols_data = ['date', 'value']

def plot_table_on_ax(ax, data, cols, title, header_color='#90A4AE'):
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')
    table = ax.table(cellText=data[:8], colLabels=cols,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        if r > 0 and c == 1:
            try:
                v = float(cell.get_text().get_text())
                if np.isnan(v): cell.set_facecolor('#FFCCBC')
                elif abs(v) > 200: cell.set_facecolor('#EF9A9A')
            except: pass
        if r > 0 and r % 2 == 0:
            cell.set_facecolor('#FAFAFA')

# Step 0
plot_table_on_ax(axes[0,0], [[d, f'{v:.1f}' if not np.isnan(v) else 'NULL']
                              for d, v in zip(df_raw['date'], df_raw['value'])],
                 cols_data, '① 原始数据（含缺失+异常）', '#EF5350')

# Step 1: 缺失值处理
values_fixed = values.copy()
nan_mask = np.isnan(values_fixed)
values_fixed[nan_mask] = np.nanmean(values_fixed)
plot_table_on_ax(axes[0,1], [[d, f'{v:.1f}'] for d, v in zip(dates, values_fixed)],
                 cols_data, '② 缺失值（均值填充）', '#FFA726')

# Step 2: 异常值处理（IQR裁剪）
q1, q3 = np.percentile(values_fixed, [25, 75])
iqr = q3 - q1
values_clipped = np.clip(values_fixed, q1 - 1.5*iqr, q3 + 1.5*iqr)
outlier_mask = (values_fixed < q1 - 1.5*iqr) | (values_fixed > q3 + 1.5*iqr)
plot_table_on_ax(axes[0,2], [[d, f'{v:.1f}'] for d, v in zip(dates, values_clipped)],
                 cols_data, '③ 异常值（IQR裁剪）', '#66BB6A')

# Step 3: 去重 + 类型转换
seen = set()
values_clean = []
dates_clean = []
for d, v in zip(dates, values_clipped):
    if d not in seen:
        seen.add(d)
        dates_clean.append(d)
        values_clean.append(int(v))
plot_table_on_ax(axes[1,0], [[d, str(v)] for d, v in zip(dates_clean, values_clean)],
                 ['date', 'value(int)'], '④ 去重+类型转换', '#42A5F5')

# 可视化对比
ax_viz = axes[1,1]
ax_viz.clear()
x_idx = np.arange(len(values))
ax_viz.scatter(x_idx[~outlier_mask], values_fixed[~outlier_mask],
               color='#2196F3', alpha=0.6, s=30, label='正常值')
ax_viz.scatter(x_idx[outlier_mask], values_fixed[outlier_mask],
               color='#F44336', s=60, marker='x', lw=2, label='已移除异常值')
ax_viz.scatter(x_idx[nan_mask], values[nan_mask],
               color='#FF9800', s=60, marker='v', label='缺失值（填充后）')
ax_viz.axhline(q1 - 1.5*iqr, color='red', ls='--', lw=1, label=f'下界={q1-1.5*iqr:.0f}')
ax_viz.axhline(q3 + 1.5*iqr, color='red', ls='--', lw=1, label=f'上界={q3+1.5*iqr:.0f}')
ax_viz.set_title('⑤ 数据质量可视化', fontsize=10, fontweight='bold')
ax_viz.set_xlabel('样本索引', fontsize=9)
ax_viz.set_ylabel('value', fontsize=9)
ax_viz.legend(fontsize=7, loc='upper right')

# 最终干净数据统计
ax_stats = axes[1,2]
ax_stats.axis('off')
ax_stats.set_title('⑥ 清洗后统计', fontsize=10, fontweight='bold')
stats_text = f"""清洗后数据概况：
━━━━━━━━━━━━━━━━
样本数：{len(values_clean)}（原始{n}）
缺失值：0（原始{np.sum(nan_mask)}个）
异常值：{np.sum(outlier_mask)}个（已修正）
重复行：{n - len(dates_clean)}（已去除）

字段类型：
  date  → string → datetime
  value → float  → integer

统计摘要：
  均值：{np.mean(values_clean):.1f}
  标准差：{np.std(values_clean):.1f}
  最小值：{min(values_clean)}
  最大值：{max(values_clean)}"""
ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
              fontsize=10, va='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
savefig(fig, 'fig_2_4_1_pipeline.png')

print("Ch1/Ch2 新增图完成!")
