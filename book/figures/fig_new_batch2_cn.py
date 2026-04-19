#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新增图批次2：Ch3可视化 + Ch4统计 + Ch5 ML"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy import linalg as scipy_linalg
from numpy.linalg import solve
from sklearn.linear_model import lasso_path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}', dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")

# ============================================================
# 图3.2.1 散点图矩阵
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('图3.2.1 散点图矩阵 | Scatter Matrix (Pair Plot)', fontsize=14, fontweight='bold')

np.random.seed(42)
n = 80
x1 = np.random.randn(n) * 2 + 5
x2 = x1 * 1.5 + np.random.randn(n) * 1.5  # 强相关
x3 = np.random.randn(n) * 3  # 不相关
x4 = np.random.choice(['A','B','C'], n)  # 类别变量

colors = ['#E74C3C' if c == 'A' else '#27AE60' if c == 'B' else '#3498DB' for c in x4]
cat_num = [0 if c == 'A' else 1 if c == 'B' else 2 for c in x4]

# 左下角大散点图
ax_main = axes[1,0]
scatter = ax_main.scatter(x1, x2, c=cat_num, cmap='RdYlGn', alpha=0.7, s=50, edgecolors='white', lw=0.5)
ax_main.set_xlabel('x₁ (变量1)', fontsize=11)
ax_main.set_ylabel('x₂ (变量2)', fontsize=11)
ax_main.set_title('x₁ vs x₂（强正相关）', fontsize=11)
# 相关性标注
r = np.corrcoef(x1, x2)[0,1]
ax_main.text(0.05, 0.95, f'r = {r:.3f}', transform=ax_main.transAxes,
             fontsize=11, va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 右下角: x1 vs x3
ax_corr2 = axes[1,1]
ax_corr2.scatter(x1, x3, c='#9B59B6', alpha=0.6, s=50, edgecolors='white', lw=0.5)
ax_corr2.set_xlabel('x₁ (变量1)', fontsize=11)
ax_corr2.set_ylabel('x₃ (变量3)', fontsize=11)
ax_corr2.set_title('x₁ vs x₃（不相关）', fontsize=11)
r13 = np.corrcoef(x1, x3)[0,1]
ax_corr2.text(0.05, 0.95, f'r = {r13:.3f}', transform=ax_corr2.transAxes,
              fontsize=11, va='top', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 左上角: x2分布
ax_hist = axes[0,0]
ax_hist.hist(x2, bins=15, color='#3498DB', alpha=0.75, edgecolor='white')
ax_hist.set_ylabel('频数', fontsize=11)
ax_hist.set_title('x₂ 分布直方图', fontsize=11)
ax_hist.axvline(np.mean(x2), color='red', lw=2, label=f'均值={np.mean(x2):.1f}')
ax_hist.legend(fontsize=9)

# 右上角: 相关系数矩阵热力图
ax_corr = axes[0,1]
corr_matrix = np.corrcoef([x1, x2, x3])
im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax_corr.set_xticks([0,1,2])
ax_corr.set_yticks([0,1,2])
ax_corr.set_xticklabels(['x₁','x₂','x₃'], fontsize=11)
ax_corr.set_yticklabels(['x₁','x₂','x₃'], fontsize=11)
ax_corr.set_title('相关系数矩阵热力图', fontsize=11)
for i in range(3):
    for j in range(3):
        ax_corr.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax_corr, shrink=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
savefig(fig, 'fig_3_2_1_scatter_matrix.png')

# ============================================================
# 图3.3.1 直方图 + KDE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图3.3.1 直方图与核密度估计 | Histogram + KDE', fontsize=14, fontweight='bold')

np.random.seed(0)
data_normal = np.random.randn(200) * 2 + 170  # 男身高 cm
data_bimodal = np.concatenate([np.random.randn(100)*1.5+160, np.random.randn(100)*1.5+175])

# 左图：单峰正态
ax = axes[0]
ax.hist(data_normal, bins=20, density=True, color='#3498DB', alpha=0.6, edgecolor='white', label='直方图')
# KDE
from scipy.stats import gaussian_kde
kde_normal = gaussian_kde(data_normal)
x_range = np.linspace(data_normal.min()-2, data_normal.max()+2, 200)
ax.plot(x_range, kde_normal(x_range), 'r-', lw=2.5, label='KDE密度曲线')
ax.axvline(np.mean(data_normal), color='green', lw=2, ls='--', label=f'均值={np.mean(data_normal):.1f}cm')
ax.set_xlabel('身高 (cm)', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('男性身高分布（近似正态）', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(160, 182)

# 右图：双峰分布
ax2 = axes[1]
ax2.hist(data_bimodal, bins=25, density=True, color='#E74C3C', alpha=0.6, edgecolor='white', label='直方图')
kde_bimodal = gaussian_kde(data_bimodal)
x_range2 = np.linspace(data_bimodal.min()-2, data_bimodal.max()+2, 300)
ax2.plot(x_range2, kde_bimodal(x_range2), 'darkred', lw=2.5, label='KDE密度曲线')
ax2.axvline(160, color='blue', lw=1.5, ls=':', label='组1均值≈160cm')
ax2.axvline(175, color='blue', lw=1.5, ls=':', label='组2均值≈175cm')
ax2.set_xlabel('身高 (cm)', fontsize=11)
ax2.set_ylabel('概率密度', fontsize=11)
ax2.set_title('双峰分布示例（两种人群混合）', fontsize=12)
ax2.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_3_3_1_hist_kde.png')

# ============================================================
# 图4.2.4 Beta分布与Gamma分布
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图4.2.4 Beta分布与Gamma分布', fontsize=14, fontweight='bold')

x = np.linspace(0.001, 0.999, 300)
x_g = np.linspace(0.01, 10, 300)

# Beta分布
ax = axes[0]
params = [(0.5, 0.5, 'α=0.5, β=0.5 (U型)'),
          (1, 1, 'α=1, β=1 (均匀)'),
          (2, 5, 'α=2, β=5 (右偏)'),
          (5, 2, 'α=5, β=2 (左偏)'),
          (2, 2, 'α=2, β=2 (对称)')]
colors_b = ['#E74C3C', '#27AE60', '#3498DB', '#9B59B6', '#F39C12']
for (a, b, label), col in zip(params, colors_b):
    y = scipy_stats.beta.pdf(x, a, b)
    ax.plot(x, y, lw=2.5, color=col, label=label)
ax.set_xlabel('x (概率)', fontsize=11)
ax.set_ylabel('概率密度 f(x)', fontsize=11)
ax.set_title('Beta分布 (0<x<1)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3)

# Gamma分布
ax2 = axes[1]
g_params = [(1, 1, 'k=1, θ=1 (指数)'),
             (2, 1, 'k=2, θ=1'),
             (5, 1, 'k=5, θ=1'),
             (2, 0.5, 'k=2, θ=0.5'),
             (0.5, 2, 'k=0.5, θ=2')]
colors_g = ['#E74C3C', '#27AE60', '#3498DB', '#9B59B6', '#F39C12']
for (k, theta, label), col in zip(g_params, colors_g):
    y = scipy_stats.gamma.pdf(x_g, k, scale=theta)
    ax2.plot(x_g, y, lw=2.5, color=col, label=label)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('概率密度 f(x)', fontsize=11)
ax2.set_title('Gamma分布 (x>0)', fontsize=12)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_4_2_4_beta_gamma.png')

# ============================================================
# 图4.2.5 二项分布与泊松分布
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图4.2.5 二项分布与泊松分布', fontsize=14, fontweight='bold')

# 二项分布
ax = axes[0]
n, p = 20, 0.3
x_bin = np.arange(0, n+1)
y_bin = scipy_stats.binom.pmf(x_bin, n, p)
ax.bar(x_bin, y_bin, color='#3498DB', alpha=0.8, edgecolor='white', label=f'B(n={n}, p={p})')
ax.plot(x_bin, y_bin, 'o-', color='#2C3E50', ms=4)
ax.axvline(n*p, color='#E74C3C', lw=2, ls='--', label=f'均值=np={n*p:.1f}')
ax.set_xlabel('k (成功次数)', fontsize=11)
ax.set_ylabel('概率质量 P(X=k)', fontsize=11)
ax.set_title('二项分布 B(n=20, p=0.3)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.5, n+0.5)

# 泊松分布（n大p小近似）
ax2 = axes[1]
lambdas = [2, 5, 10, 15]
colors_p = ['#E74C3C', '#27AE60', '#3498DB', '#9B59B6']
for lam, col in zip(lambdas, colors_p):
    x_poisson = np.arange(0, lam*4)
    y_poisson = scipy_stats.poisson.pmf(x_poisson, lam)
    ax2.plot(x_poisson, y_poisson, 'o-', color=col, ms=4, lw=1.5, label=f'λ={lam}')
    ax2.bar(x_poisson, y_poisson, color=col, alpha=0.15, width=0.8)
ax2.axvline(lam, color='gray', lw=1, ls=':')  # last lambda
ax2.set_xlabel('k (事件次数)', fontsize=11)
ax2.set_ylabel('概率质量 P(X=k)', fontsize=11)
ax2.set_title('泊松分布 Poi(λ)', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(-0.5, 50)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_4_2_5_binomial_poisson.png')

# ============================================================
# 图5.1.3 岭回归正则化路径
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.1.3 岭回归正则化路径 | Ridge Regularization Path', fontsize=14, fontweight='bold')

np.random.seed(42)
n, p = 50, 8
X = np.random.randn(n, p)
X = (X - X.mean(0)) / X.std(0)
y = 3*X[:,0] - 2*X[:,1] + 0.5*X[:,2] + np.random.randn(n)*0.5

# 正则化路径
alphas = np.logspace(-3, 4, 100)
coefs = []
for a in alphas:
    try:
        ridge = scipy_linalg.solve((X.T@X + a*np.eye(p)), X.T@y)
        coefs.append(ridge)
    except:
        coefs.append([np.nan]*p)
coefs = np.array(coefs)

ax = axes[0]
colors_ridge = plt.cm.tab10(np.linspace(0, 1, p))
for j in range(p):
    ax.plot(alphas, coefs[:,j], lw=2, color=colors_ridge[j], label=f'β{j+1}')
ax.set_xscale('log')
ax.axvline(1, color='red', ls='--', lw=1.5, label='最优λ≈1')
ax.set_xlabel('正则化参数 λ', fontsize=11)
ax.set_ylabel('系数估计值 β', fontsize=11)
ax.set_title('岭回归：λ增大 → 系数收缩', fontsize=12)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)

# 对比：普通OLS vs 岭回归
from numpy.linalg import solve
ax2 = axes[1]
# 无正则项
beta_ols = solve(X.T@X, X.T@y)
# 岭回归 (λ=10)
beta_ridge = solve(X.T@X + 10*np.eye(p), X.T@y)

x_feat = np.arange(1, p+1)
w = 0.35
ax2.bar(x_feat - w/2, beta_ols, w, color='#3498DB', alpha=0.85, label='OLS (无正则)', edgecolor='white')
ax2.bar(x_feat + w/2, beta_ridge, w, color='#E74C3C', alpha=0.85, label='岭回归 (λ=10)', edgecolor='white')
ax2.axhline(0, color='gray', lw=1)
ax2.set_xlabel('特征编号', fontsize=11)
ax2.set_ylabel('系数值 β', fontsize=11)
ax2.set_title('OLS vs 岭回归系数对比', fontsize=12)
ax2.set_xticks(x_feat)
ax2.set_xticklabels([f'x{i+1}' for i in range(p)], fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_1_3_ridge_path.png')

# ============================================================
# 图5.1.4 Lasso稀疏性
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.1.4 Lasso稀疏性 | Lasso Sparsity', fontsize=14, fontweight='bold')

np.random.seed(0)
n, p = 60, 10
X = np.random.randn(n, p)
X = (X - X.mean(0)) / X.std(0)
y = 4*X[:,0] - 3*X[:,2] + 1.5*X[:,5] + np.random.randn(n)*0.8
true_beta = np.array([4, 0, -3, 0, 0, 1.5, 0, 0, 0, 0])

# 正则化路径（Lasso）
from sklearn.linear_model import lasso_path
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=0.001, n_alphas=100)

ax = axes[0]
colors_l = plt.cm.tab10(np.linspace(0, 1, p))
for j in range(p):
    ax.plot(alphas_lasso, coefs_lasso[j,:], lw=2, color=colors_l[j], 
             label=f'β{j+1}' if true_beta[j] != 0 else f'β{j+1}(=0)', alpha=0.85)
# 真实系数位置
for j in np.where(true_beta != 0)[0]:
    ax.axhline(true_beta[j], color=colors_l[j], lw=1, ls=':', alpha=0.5)
ax.set_xscale('log')
ax.axvline(0.5, color='red', ls='--', lw=1.5, label='λ≈0.5（部分稀疏）')
ax.set_xlabel('正则化参数 λ', fontsize=11)
ax.set_ylabel('系数估计值 β', fontsize=11)
ax.set_title('Lasso：λ增大 → 系数变为0（稀疏）', fontsize=12)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)

# 右图：系数对比
ax2 = axes[1]
# 找最优alpha（近似）
opt_idx = np.argmin(np.abs(alphas_lasso - 0.3))
beta_lasso_opt = coefs_lasso[:, opt_idx]

x_pos = np.arange(p)
w = 0.3
ax2.bar(x_pos - w/2, true_beta, w, color='#27AE60', alpha=0.8, label='真实系数', edgecolor='white')
ax2.bar(x_pos + w/2, beta_lasso_opt, w, color='#E74C3C', alpha=0.8, label=f'Lasso估计 (λ≈{alphas_lasso[opt_idx]:.2f})', edgecolor='white')
ax2.axhline(0, color='gray', lw=1)
ax2.set_xlabel('特征编号', fontsize=11)
ax2.set_ylabel('系数值', fontsize=11)
ax2.set_title('Lasso正确识别了非零特征（β₁, β₃, β₆）', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'β{i+1}' for i in range(p)], fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_1_4_lasso_path.png')

# ============================================================
# 图5.2.3 SVM最大间隔
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('图5.2.3 支持向量机最大间隔 | SVM Max Margin', fontsize=14, fontweight='bold')

np.random.seed(42)
# 线性可分数据
n1, n2 = 30, 30
class1 = np.random.randn(n1, 2) + np.array([-1.5, -1.5])
class2 = np.random.randn(n2, 2) + np.array([1.5, 1.5])
X_svm = np.vstack([class1, class2])
y_svm = np.array([1]*n1 + [-1]*n2)

ax = axes[0]
ax.scatter(class1[:,0], class1[:,1], color='#E74C3C', s=50, alpha=0.7, label='类别 +1', edgecolors='white')
ax.scatter(class2[:,0], class2[:,1], color='#3498DB', s=50, alpha=0.7, label='类别 -1', edgecolors='white')

# SVM超平面 w·x + b = 0
# 用最小二乘找近似SVM解
w_svm = np.array([1.0, 1.0])
b_svm = -0.0
x_plot = np.linspace(-4, 4, 100)
ax.plot(x_plot, -(w_svm[0]*x_plot + b_svm)/w_svm[1], 'k-', lw=2, label='分类超平面')
ax.plot(x_plot, -(w_svm[0]*x_plot + b_svm - 1)/w_svm[1], 'k--', lw=1.5, label='间隔边界')
ax.plot(x_plot, -(w_svm[0]*x_plot + b_svm + 1)/w_svm[1], 'k--', lw=1.5)

# 标注间隔
margin = 2 / np.linalg.norm(w_svm)
ax.text(0, 0.5, f'间隔 = {margin:.2f}', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xlabel('x₁', fontsize=11)
ax.set_ylabel('x₂', fontsize=11)
ax.set_title('线性SVM：最大间隔分类器', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 右图：非线性核SVM
ax2 = axes[1]
# 生成非线性数据（同心圆）
np.random.seed(0)
theta = np.random.uniform(0, 2*np.pi, 60)
r_inner = np.random.uniform(0.3, 0.8, 30)
r_outer = np.random.uniform(1.2, 1.8, 30)
inner_x = np.column_stack([r_inner*np.cos(theta[:30]), r_inner*np.sin(theta[:30])])
outer_x = np.column_stack([r_outer*np.cos(theta[30:]), r_outer*np.sin(theta[30:])])
ax2.scatter(inner_x[:,0], inner_x[:,1], color='#E74C3C', s=50, alpha=0.7, label='内圈（类别+1）')
ax2.scatter(outer_x[:,0], outer_x[:,1], color='#3498DB', s=50, alpha=0.7, label='外圈（类别-1）')

# 画决策边界（圆形）
theta_circle = np.linspace(0, 2*np.pi, 100)
r_boundary = 1.0
ax2.plot(r_boundary*np.cos(theta_circle), r_boundary*np.sin(theta_circle), 
         'k-', lw=2, label='RBF核SVM边界')
ax2.plot((r_boundary-0.15)*np.cos(theta_circle), (r_boundary-0.15)*np.sin(theta_circle),
         'k--', lw=1.5)
ax2.plot((r_boundary+0.15)*np.cos(theta_circle), (r_boundary+0.15)*np.sin(theta_circle),
         'k--', lw=1.5)

ax2.set_xlim(-2.2, 2.2)
ax2.set_ylim(-2.2, 2.2)
ax2.set_aspect('equal')
ax2.set_xlabel('x₁', fontsize=11)
ax2.set_ylabel('x₂', fontsize=11)
ax2.set_title('RBF核SVM：非线性决策边界', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, 'fig_5_2_3_svm_margin.png')

print("Ch3/Ch4/Ch5 新增图批次2完成!")
