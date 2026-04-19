#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图5.x 回归与分类：残差图、决策边界、分类器对比
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_regression
from scipy import stats
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

# ========== 图5.1.1 线性回归残差分析 ==========
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
np.random.seed(42)
X = np.linspace(0, 10, 80); y_true = 2*X + 1
y_obs = y_true + np.random.randn(80)*1.5
X_p = X.reshape(-1,1)
model = LinearRegression().fit(X_p, y_obs)
y_pred = model.predict(X_p)
residuals = y_obs - y_pred

# 残差 vs 拟合值
ax = axes[0,0]
ax.scatter(y_pred, residuals, alpha=0.6, color='#3498DB', s=30)
ax.axhline(0, color='#E74C3C', lw=2, ls='--')
ax.set_xlabel('拟合值 ŷ', fontsize=11)
ax.set_ylabel('残差 y - ŷ', fontsize=11)
ax.set_title('残差 vs 拟合值（检验同方差性）', fontsize=12, fontweight='bold')
ax.text(0.05, 0.95, '理想：随机散布，无明显模式', transform=ax.transAxes,
        fontsize=9, va='top', color='#555555')

# Q-Q 图（残差正态性）
ax = axes[0,1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('残差 Q-Q 图（检验正态性）', fontsize=12, fontweight='bold')
ax.get_lines()[0].set_color('#3498DB'); ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color('#E74C3C'); ax.get_lines()[1].set_linewidth(2)

# 残差直方图
ax = axes[1,0]
ax.hist(residuals, bins=20, density=True, alpha=0.6, color='#3498DB', edgecolor='white')
xh = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xh, stats.norm.pdf(xh, residuals.mean(), residuals.std()),
        color='#E74C3C', lw=2.5, label='正态分布拟合')
ax.set_xlabel('残差值', fontsize=11); ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('残差分布直方图', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Scale-Location（标准化残差开方）
ax = axes[1,1]
std_resid = np.sqrt(np.abs((residuals - residuals.mean()) / residuals.std()))
ax.scatter(y_pred, std_resid, alpha=0.6, color='#3498DB', s=30)
z = np.polyfit(y_pred, std_resid, 1); p = np.poly1d(z)
ax.plot(sorted(y_pred), p(sorted(y_pred)), color='#E74C3C', lw=2, ls='--')
ax.set_xlabel('拟合值 ŷ', fontsize=11)
ax.set_ylabel('√|标准化残差|', fontsize=11)
ax.set_title('Scale-Location 图（检验同方差性）', fontsize=12, fontweight='bold')

fig.suptitle('线性回归残差诊断四图（检验回归假设）', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
savefig(fig, 'fig_5_1_1_regression_residuals.png')

# ========== 图5.1.2 多项式回归与偏差-方差 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
np.random.seed(0)
x_train = np.sort(np.random.uniform(0, 10, 30))
y_train = np.sin(x_train) + np.random.randn(30)*0.3
x_test = np.linspace(0, 10, 200)
y_true = np.sin(x_test)
degrees = [1, 3, 15]
for ax, deg in zip(axes, degrees):
    poly = PolynomialFeatures(deg)
    X_tr = poly.fit_transform(x_train.reshape(-1,1))
    X_te = poly.transform(x_test.reshape(-1,1))
    model = LinearRegression().fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    ax.scatter(x_train, y_train, s=25, alpha=0.6, color='#3498DB', label='训练数据')
    ax.plot(x_test, y_true, color='#E74C3C', lw=2, label='真实函数 sin(x)')
    ax.plot(x_test, y_pred, color='#27AE60', lw=2.5, ls='--', label=f'degree={deg}')
    mse_train = np.mean((model.predict(X_tr) - y_train)**2)
    mse_test = np.mean((y_pred - y_true)**2)
    ax.set_title(f'degree={deg}\n训练MSE={mse_train:.3f} | 测试MSE={mse_test:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-2, 2)
fig.suptitle('多项式回归：欠拟合 → 适度 → 过拟合', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_5_1_2_poly_regression.png')

# ========== 图5.2.1 逻辑回归决策边界 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                            n_redundant=0, n_classes=2, n_clusters_per_class=1,
                            random_state=42)
# 线性边界
ax = axes[0]
lr = LogisticRegression(random_state=42).fit(X, y)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                      np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0.5], colors='#E74C3C', linewidths=2)
ax.scatter(X[y==0,0], X[y==0,1], s=20, alpha=0.7, color='#3498DB', label='类别0')
ax.scatter(X[y==1,0], X[y==1,1], s=20, alpha=0.7, color='#E74C3C', label='类别1')
ax.set_title('逻辑回归（线性决策边界）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
# 非线性（多项式）
ax = axes[1]
X_np2 = PolynomialFeatures(degree=3).fit_transform(X)
lr2 = LogisticRegression(random_state=42, max_iter=1000).fit(X_np2, y)
Z2 = lr2.predict_proba(
    PolynomialFeatures(degree=3).fit_transform(
        np.c_[xx.ravel(), yy.ravel()]))[:,1].reshape(xx.shape)
ax.contourf(xx, yy, Z2, levels=20, alpha=0.3, cmap='RdBu')
ax.contour(xx, yy, Z2, levels=[0.5], colors='#E74C3C', linewidths=2)
ax.scatter(X[y==0,0], X[y==0,1], s=20, alpha=0.7, color='#3498DB', label='类别0')
ax.scatter(X[y==1,0], X[y==1,1], s=20, alpha=0.7, color='#E74C3C', label='类别1')
ax.set_title('多项式逻辑回归（非线性决策边界）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
savefig(fig, 'fig_5_2_1_logistic_boundary.png')

# ========== 图5.2.2 KNN 决策边界随 K 变化 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
Ks = [1, 5, 21]
for ax, k in zip(axes, Ks):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Zk = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx, yy, Zk, levels=20, alpha=0.3, cmap='RdBu')
    ax.contour(xx, yy, Zk, levels=[0.5], colors='#E74C3C', linewidths=1.5)
    ax.scatter(X[y==0,0], X[y==0,1], s=15, alpha=0.6, color='#3498DB')
    ax.scatter(X[y==1,0], X[y==1,1], s=15, alpha=0.6, color='#E74C3C')
    ax.set_title(f'K={k} 近邻\n小K=过拟合，大K=平滑边界', fontsize=12, fontweight='bold')
savefig(fig, 'fig_5_2_2_knn_boundary.png')

print("回归与分类 figures done!")
