#!/usr/bin/env python3
"""
机器学习核心概念图：过拟合/欠拟合、梯度下降、KMeans、交叉验证
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
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

plt.style.use('seaborn-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

def plot_decision_boundary(ax, clf, X, y, color_map):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=color_map)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='white', linewidths=0.5, cmap=color_map)

# ========== 图1: 过拟合 vs 欠拟合 ==========
np.random.seed(42)
n_samples = 50
X_train = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
y_train = np.sin(X_train[:, 0]) + np.random.randn(n_samples) * 0.3
X_test = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
y_test = np.sin(X_test[:, 0]) + np.random.randn(30) * 0.3
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 欠拟合 (degree=1)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

for i, (degree, title, score_text) in enumerate([
    (1, '欠拟合 (d=1)', f'测试集 R² ≈ 0.1'),
    (3, '适度拟合 (d=3)', f'测试集 R² ≈ 0.92'),
    (15, '过拟合 (d=15)', f'测试集 R² ≈ 0.4'),
]):
    ax = axes[i]
    poly = PolynomialFeatures(degree=degree)
    X_tr_poly = poly.fit_transform(X_train)
    X_pl_poly = poly.transform(X_plot)
    X_te_poly = poly.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_tr_poly, y_train)
    y_plot = lr.predict(X_pl_poly)

    train_score = lr.score(X_tr_poly, y_train)
    test_score = lr.score(X_te_poly, y_test)

    ax.scatter(X_train, y_train, s=25, alpha=0.6, color='#3498DB', label='训练集', zorder=5)
    ax.scatter(X_test, y_test, s=25, alpha=0.6, color='#E74C3C', label='测试集', zorder=5)
    ax.plot(X_plot, y_plot, color='#E67E22', lw=2.5, label='拟合曲线')
    ax.set_title(f'{title}\n训练 R²={train_score:.2f}  测试 R²={test_score:.2f}', fontsize=11)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-2.5, 2.5)

fig.suptitle('欠拟合 vs 适度拟合 vs 过拟合\n(过拟合：训练集好，测试集差——模型记住了噪声而非规律)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_overfitting_underfitting.png')

# ========== 图2: K-Means 聚类过程 ==========
np.random.seed(30)
blobs = make_blobs(n_samples=[80, 60, 50], centers=[[-1, -1], [1, 1], [-1, 1]],
                   cluster_std=0.6, random_state=42)
X = blobs[0]

# K-Means 迭代过程
from sklearn.cluster import KMeans

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
step_labels = ['初始: 随机选3个圆心', '迭代1: 分配点到最近圆心', '迭代3: 圆心移动', '收敛: 最终聚类结果']
centers_all = []

for iteration, ax in zip(range(4), axes):
    if iteration == 0:
        centers = np.array([[-1.5, -0.5], [0.5, -1.0], [-0.5, 1.5]])
        labels = np.argmin(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2), axis=1)
    elif iteration < 3:
        centers = np.array([[X[labels==k].mean(axis=0) for k in range(3)]])
        centers = centers.reshape(-1, 2)
        labels = np.argmin(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2), axis=1)
    else:
        km = KMeans(n_clusters=3, init=np.array([[-1.5, -0.5], [0.5, -1.0], [-0.5, 1.5]]),
                    n_init=1, max_iter=1, random_state=42)
        km.fit(X)
        centers = km.cluster_centers_
        labels = km.labels_

    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=20, color=colors[k], alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], s=150, marker='X', color='#F39C12',
               edgecolors='white', linewidths=2, zorder=10, label='圆心')
    ax.set_title(step_labels[iteration], fontsize=11)
    ax.set_xlabel('x₁', fontsize=10)
    ax.set_ylabel('x₂', fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle('K-Means 聚类过程\n(迭代优化：分配→更新→分配→更新，直至收敛)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_kmeans_iteration.png')

# ========== 图3: 交叉验证示意图 ==========
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 3.5))
n_samples = 10
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']

X_dots = np.arange(n_samples)
width = 0.15
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_dots)):
    for sample_idx in range(n_samples):
        color = fold_colors[fold_idx] if sample_idx in val_idx else '#DDDDDD'
        marker = 's' if sample_idx in val_idx else 'o'
        size = 120 if sample_idx in val_idx else 60
        alpha = 1.0 if sample_idx in val_idx else 0.3
        ax.scatter(fold_idx + (sample_idx - 4.5) * width, 0,
                   c=color, s=size, marker=marker, alpha=alpha)

    # 标注
    ax.annotate(f'折{fold_idx+1}', xy=(fold_idx, 0.12), ha='center', fontsize=10, fontweight='bold')

ax.set_xlim(-0.6, k_folds - 0.4)
ax.set_ylim(-0.3, 0.3)
ax.set_title('5折交叉验证：每一折轮流作为验证集', fontsize=13, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 图例
legend_elements = [
    mpatches.Patch(color='#FF6B6B', label='验证集 (每折轮流)'),
    mpatches.Patch(color='#DDDDDD', label='训练集'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
ax.text(0.5, -0.18, '● 训练样本        ■ 验证样本 (每折不同)', transform=ax.transAxes,
        ha='center', fontsize=11, style='italic', color='#555555')

plt.tight_layout()
savefig(fig, 'fig_cross_validation.png')

# ========== 图4: 梯度下降收敛 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左: 梯度下降路径
x = np.linspace(-3, 3, 300)
y = x**4 - 5*x**3 + 2*x**2 + 3
loss = x**4 - 5*x**3 + 2*x**2 + 3  # 1D loss function

ax = axes[0]
ax.plot(x, loss, color='#34495E', lw=2, label='损失函数 J(w)')

# 梯度下降路径
def grad(x_val):
    return 4*x_val**3 - 15*x_val**2 + 4*x_val

w_current = 2.5
path = [w_current]
for _ in range(15):
    grad_val = grad(w_current)
    w_current = w_current - 0.15 * grad_val
    path.append(w_current)

path = np.array(path)
loss_path = path**4 - 5*path**3 + 2*path**2 + 3
ax.plot(path, loss_path, 'o-', color='#E74C3C', lw=1.8, markersize=6, label='梯度下降路径')
ax.scatter([path[-1]], [loss_path[-1]], s=100, color='#E74C3C', zorder=10, label=f'收敛点 w≈{path[-1]:.2f}')

ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('J(w)', fontsize=12)
ax.set_title('梯度下降：沿最陡方向下山', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-5, 15)

# 右: 不同学习率
ax = axes[1]
for lr, color, ls in [(0.01, '#27AE60', '-'), (0.15, '#3498DB', '-'), (0.35, '#E74C3C', '-')]:
    w_current = 2.5
    path = [w_current]
    for _ in range(20):
        w_current = w_current - lr * grad(w_current)
        path.append(w_current)
    path = np.array(path)
    loss_vals = path**4 - 5*path**3 + 2*path**2 + 3
    ax.plot(range(len(path)), loss_vals, color=color, lw=2, ls=ls, label=f'η={lr}')

ax.set_xlabel('迭代次数', fontsize=12)
ax.set_ylabel('J(w)', fontsize=12)
ax.set_title('学习率的影响\n(η太小→慢，η太大→振荡或发散)', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-5, 20)

fig.suptitle('梯度下降优化过程', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_gradient_descent.png')

print("ML概念图生成完成！")
