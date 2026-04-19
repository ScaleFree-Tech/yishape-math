#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML core concept figures: overfitting, gradient descent, KMeans, cross-validation"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

plt.style.use('seaborn-v0_8-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== Fig 1: Overfitting vs Underfitting ==========
np.random.seed(42)
n_samples = 50
X_train = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
y_train = np.sin(X_train[:, 0]) + np.random.randn(n_samples) * 0.3
X_test = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
y_test = np.sin(X_test[:, 0]) + np.random.randn(30) * 0.3
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for i, (degree, title, _) in enumerate([
    (1, 'Underfit (d=1)', 'Test R2 ~ 0.1'),
    (3, 'Good Fit (d=3)', 'Test R2 ~ 0.92'),
    (15, 'Overfit (d=15)', 'Test R2 ~ 0.4'),
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
    ax.scatter(X_train, y_train, s=25, alpha=0.6, color='#3498DB', label='Train', zorder=5)
    ax.scatter(X_test, y_test, s=25, alpha=0.6, color='#E74C3C', label='Test', zorder=5)
    ax.plot(X_plot, y_plot, color='#E67E22', lw=2.5, label='Fitted')
    ax.set_title(f'{title}\nTrain R2={train_score:.2f}  Test R2={test_score:.2f}', fontsize=11)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-2.5, 2.5)

fig.suptitle('Underfit vs Good Fit vs Overfit\n(Overfit: train good, test poor -- model memorized noise, not pattern)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_overfitting_underfitting.png')

# ========== Fig 2: K-Means Iteration ==========
np.random.seed(30)
blobs = make_blobs(n_samples=[80, 60, 50], centers=[[-1, -1], [1, 1], [-1, 1]], cluster_std=0.6, random_state=42)
X = blobs[0]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
step_labels = ['Init: 3 random centroids', 'Iter 1: assign points', 'Iter 3: centroids shift', 'Converged: final clusters']
centers_list = [
    np.array([[-1.5, -0.5], [0.5, -1.0], [-0.5, 1.5]]),
    None, None, None
]

for iteration, ax in enumerate(axes):
    if iteration == 0:
        centers = centers_list[0]
        labels = np.argmin(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2), axis=1)
    elif iteration < 3:
        km_temp = KMeans(n_clusters=3, init=centers_list[0], n_init=1, max_iter=iteration, random_state=42)
        km_temp.fit(X)
        centers = km_temp.cluster_centers_
        labels = km_temp.labels_
    else:
        km = KMeans(n_clusters=3, init=centers_list[0], n_init=1, max_iter=10, random_state=42)
        km.fit(X)
        centers = km.cluster_centers_
        labels = km.labels_

    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=20, color=colors[k], alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], s=150, marker='X', color='#F39C12', edgecolors='white', linewidths=2, zorder=10, label='Centroids')
    ax.set_title(step_labels[iteration], fontsize=11)
    ax.set_xlabel('x1', fontsize=10)
    ax.set_ylabel('x2', fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle('K-Means Clustering Process\n(Iterate: assign -> update -> assign -> update -> converge)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_kmeans_iteration.png')

# ========== Fig 3: Cross-Validation ==========
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
        ax.scatter(fold_idx + (sample_idx - 4.5) * width, 0, c=color, s=size, marker=marker, alpha=alpha)
    ax.annotate(f'Fold {fold_idx+1}', xy=(fold_idx, 0.12), ha='center', fontsize=10, fontweight='bold')

ax.set_xlim(-0.6, k_folds - 0.4)
ax.set_ylim(-0.3, 0.3)
ax.set_title('5-Fold Cross-Validation: Each Fold Takes Turns as Validation Set', fontsize=13, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
legend_elements = [
    mpatches.Patch(color='#FF6B6B', label='Validation set (rotating)'),
    mpatches.Patch(color='#DDDDDD', label='Training set'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
ax.text(0.5, -0.18, 'o = Training sample        [] = Validation sample (different each fold)', transform=ax.transAxes,
        ha='center', fontsize=11, style='italic', color='#555555')
plt.tight_layout()
savefig(fig, 'fig_cross_validation.png')

# ========== Fig 4: Gradient Descent ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.linspace(-3, 3, 300)
loss = np.sin(x) + 0.1 * x**2  # Well-behaved loss function

ax = axes[0]
ax.plot(x, loss, color='#34495E', lw=2, label='Loss J(w)')

def grad(x_val):
    return np.cos(x_val) + 0.2 * x_val

w_current = 2.5
path = [w_current]
for _ in range(20):
    grad_val = grad(w_current)
    w_current = w_current - 0.3 * grad_val
    path.append(w_current)

path = np.array(path)
loss_path = np.sin(path) + 0.1 * path**2
ax.plot(path, loss_path, 'o-', color='#E74C3C', lw=1.8, markersize=6, label='Gradient descent path')
ax.scatter([path[-1]], [loss_path[-1]], s=100, color='#E74C3C', zorder=10, label=f'Converged w~={path[-1]:.2f}')
ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('J(w)', fontsize=12)
ax.set_title('Gradient Descent: Steepest Descent Downhill', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-1.5, 3)

ax = axes[1]
for lr, color in [(0.005, '#27AE60'), (0.03, '#3498DB'), (0.08, '#E74C3C')]:
    w_current = 2.5
    path = [w_current]
    for _ in range(20):
        grad_val = np.cos(w_current) + 0.2 * w_current
        w_current = w_current - lr * grad_val
        path.append(w_current)
    path = np.array(path)
    loss_vals = np.sin(path) + 0.1 * path**2
    ax.plot(range(len(path)), loss_vals, color=color, lw=2, label=f'lr={lr}')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('J(w)', fontsize=12)
ax.set_title('Effect of Learning Rate\n(Too small=slow, too large=oscillate/diverge)', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-5, 20)

fig.suptitle('Gradient Descent Optimization', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_gradient_descent.png')

print("ML concept figures done!")
