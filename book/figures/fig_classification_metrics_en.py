#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification metrics figures: confusion matrix, ROC, PR, bias-variance"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2,
                            n_classes=2, weights=[0.4, 0.6], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

# ========== Fig 1: Confusion Matrix ==========
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted 0', 'Predicted 1'], fontsize=12)
ax.set_yticklabels(['Actual 0', 'Actual 1'], fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else '#2C3E50'
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=24, fontweight='bold', color=color)
ax.text(0, -0.25, 'TN', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(1, -0.25, 'FP', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(0, 1.05, 'FN', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(1, 1.05, 'TP', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
savefig(fig, 'fig_confusion_matrix.png')

# ========== Fig 2: ROC Curve ==========
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, ls='--', label='Random (AUC=0.5)')
ax.plot(fpr, tpr, color='#3498DB', lw=2.5, label=f'Logistic Regression (AUC={roc_auc:.3f})')
ax.fill_between(fpr, tpr, alpha=0.15, color='#3498DB')
for threshold in [0.3, 0.5, 0.7]:
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(fpr[idx], tpr[idx], s=80, zorder=10, color='#E74C3C')
    ax.annotate(f't={threshold}', xy=(fpr[idx], tpr[idx]), xytext=(fpr[idx]-0.15, tpr[idx]+0.05),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#555555'))
ax.set_xlabel('False Positive Rate  = FP/(FP+TN)', fontsize=12)
ax.set_ylabel('True Positive Rate  = TP/(TP+FN)', fontsize=12)
ax.set_title('ROC Curve\n(Closer to top-left corner = better classifier)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
plt.tight_layout()
savefig(fig, 'fig_roc_curve.png')

# ========== Fig 3: Precision-Recall Curve ==========
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
baseline = y_test.mean()
ax.axhline(baseline, color='gray', lw=1.5, ls='--', label=f'Baseline (positive rate={baseline:.2f})')
ax.plot(recall, precision, color='#E74C3C', lw=2.5, label=f'Logistic Regression (AP={avg_precision:.3f})')
ax.fill_between(recall, precision, alpha=0.15, color='#E74C3C')
ax.set_xlabel('Recall  = TP/(TP+FN)', fontsize=12)
ax.set_ylabel('Precision  = TP/(TP+FP)', fontsize=12)
ax.set_title('Precision-Recall Curve\n(Good for imbalanced data; higher = better)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1.05)
plt.tight_layout()
savefig(fig, 'fig_precision_recall.png')

# ========== Fig 4: Bias-Variance ==========
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

np.random.seed(0)
for ax, title, case in [
    (axes[0], 'Low Bias + Low Variance\n(Ideal model)', 'good'),
    (axes[1], 'High Bias + Low Variance\n(Underfit: too simple)', 'bias'),
    (axes[2], 'Low Bias + High Variance\n(Overfit: too complex)', 'variance'),
]:
    x = np.linspace(0, 1, 20)
    if case == 'good':
        for _ in range(30):
            y = np.sin(2*np.pi*x) + np.random.randn(20)*0.1
            ax.scatter(x, y, s=5, alpha=0.3, color='#6C8EBF')
        ax.plot(np.linspace(0,1,100), np.sin(2*np.pi*np.linspace(0,1,100)), color='#E74C3C', lw=2.5)
    elif case == 'bias':
        for _ in range(30):
            y = np.sin(2*np.pi*x) + np.random.randn(20)*0.1
            ax.scatter(x, y, s=5, alpha=0.3, color='#6C8EBF')
        ax.plot(np.linspace(0,1,100), np.polyval([0, 0], np.linspace(0,1,100)), color='#E74C3C', lw=2.5)
    else:
        for seed in range(30):
            np.random.seed(seed)
            y_sample = np.sin(2*np.pi*x) + np.random.randn(20)*0.2
            ax.scatter(x, y_sample, s=5, alpha=0.3, color='#6C8EBF')
            poly_c = np.polyfit(x, y_sample, 15)
            ax.plot(np.linspace(0,1,100), np.polyval(poly_c, np.linspace(0,1,100)), color='#3498DB', lw=0.8, alpha=0.4)
        ax.plot(np.linspace(0,1,100), np.sin(2*np.pi*np.linspace(0,1,100)), color='#E74C3C', lw=2.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(-1.5, 1.5)

fig.suptitle('Bias-Variance Decomposition: Two Sides of Model Complexity\n(Good model = neither too simple nor too complex)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_bias_variance.png')

print("Classification metrics figures done!")
