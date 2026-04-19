#!/usr/bin/env python3
"""
分类评估指标图：混淆矩阵、ROC曲线、Precision-Recall曲线
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
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# 生成数据
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, n_classes=2, weights=[0.4, 0.6],
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

# ========== 图1: 混淆矩阵 ==========
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5.5))

im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['负类 (0)', '正类 (1)'], fontsize=12)
ax.set_yticklabels(['负类 (0)', '正类 (1)'], fontsize=12)
ax.set_xlabel('预测标签', fontsize=13)
ax.set_ylabel('真实标签', fontsize=13)
ax.set_title('混淆矩阵 (Confusion Matrix)', fontsize=13, fontweight='bold')

# 填数字
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else '#2C3E50'
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                fontsize=24, fontweight='bold', color=color)

# 标注 TN FP FN TP
ax.text(0, -0.25, 'TN (真负)', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(1, -0.25, 'FP (假正)', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(0, 1.05, 'FN (假负)', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')
ax.text(1, 1.05, 'TP (真正)', ha='center', transform=ax.transAxes, fontsize=9, color='#555555')

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
savefig(fig, 'fig_confusion_matrix.png')

# ========== 图2: ROC 曲线 ==========
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 随机分类器的对角线
random_fpr = np.linspace(0, 1, 100)
random_tpr = random_fpr

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(random_fpr, random_tpr, color='gray', lw=1.5, ls='--', label='随机分类器 (AUC=0.5)')
ax.plot(fpr, tpr, color='#3498DB', lw=2.5, label=f'逻辑回归 (AUC={roc_auc:.3f})')
ax.fill_between(fpr, tpr, alpha=0.15, color='#3498DB')

# 标注几个关键阈值点
for threshold in [0.3, 0.5, 0.7]:
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(fpr[idx], tpr[idx], s=80, zorder=10, color='#E74C3C')
    ax.annotate(f'阈值={threshold}', xy=(fpr[idx], tpr[idx]),
                xytext=(fpr[idx]-0.15, tpr[idx]+0.05),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#555555'))

ax.set_xlabel('假正率 (FPR)  = FP/(FP+TN)', fontsize=12)
ax.set_ylabel('真正率 (TPR)  = TP/(TP+FN)', fontsize=12)
ax.set_title('ROC 曲线\n(曲线越靠近左上角，分类器越好)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

plt.tight_layout()
savefig(fig, 'fig_roc_curve.png')

# ========== 图3: Precision-Recall 曲线 ==========
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))

# 基准线（正类比例）
baseline = y_test.mean()
ax.axhline(baseline, color='gray', lw=1.5, ls='--', label=f'基准线 (正类比例={baseline:.2f})')
ax.plot(recall, precision, color='#E74C3C', lw=2.5, label=f'逻辑回归 (AP={avg_precision:.3f})')
ax.fill_between(recall, precision, alpha=0.15, color='#E74C3C')

ax.set_xlabel('召回率 (Recall)  = TP/(TP+FN)', fontsize=12)
ax.set_ylabel('精确率 (Precision)  = TP/(TP+FP)', fontsize=12)
ax.set_title('Precision-Recall 曲线\n(适合不平衡数据；曲线越高，分类器越好)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1.05)

plt.tight_layout()
savefig(fig, 'fig_precision_recall.png')

# ========== 图4: 偏差-方差分解图 ==========
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# 低偏差低方差 (好模型)
ax = axes[0]
np.random.seed(0)
for _ in range(30):
    x = np.linspace(0, 1, 20)
    y = np.sin(2*np.pi*x) + np.random.randn(20)*0.1
    ax.scatter(x, y, s=5, alpha=0.3, color='#6C8EBF')
ax.plot(np.linspace(0,1,100), np.sin(2*np.pi*np.linspace(0,1,100)), color='#E74C3C', lw=2.5, label='真实函数')
ax.set_title('低偏差 + 低方差\n(理想模型)', fontsize=12, color='#27AE60', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-1.5, 1.5)

# 高偏差 (欠拟合)
ax = axes[1]
np.random.seed(0)
for _ in range(30):
    x = np.linspace(0, 1, 20)
    y = np.sin(2*np.pi*x) + np.random.randn(20)*0.1
    ax.scatter(x, y, s=5, alpha=0.3, color='#6C8EBF')
# 欠拟合：线性拟合
x_fit = np.linspace(0, 1, 100)
ax.plot(x_fit, np.polyval([0, 0], x_fit), color='#E74C3C', lw=2.5, label='线性拟合（欠拟合）')
ax.set_title('高偏差 + 低方差\n(欠拟合：模型太简单)', fontsize=12, color='#F39C12', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-1.5, 1.5)

# 高方差 (过拟合)
ax = axes[2]
np.random.seed(0)
x_full = np.linspace(0, 1, 20)
y_full = np.sin(2*np.pi*x_full)
# 过拟合：用一个高次多项式
poly_coeffs = np.polyfit(x_full, y_full, 15)
x_fit = np.linspace(0, 1, 100)
for seed in range(30):
    np.random.seed(seed)
    y_sample = y_full + np.random.randn(20)*0.2
    poly_c = np.polyfit(x_full, y_sample, 15)
    ax.scatter(x_full, y_sample, s=5, alpha=0.3, color='#6C8EBF')
    ax.plot(x_fit, np.polyval(poly_c, x_fit), color='#3498DB', lw=0.8, alpha=0.4)
ax.plot(x_fit, np.sin(2*np.pi*x_fit), color='#E74C3C', lw=2.5, label='真实函数')
ax.set_title('低偏差 + 高方差\n(过拟合：模型太复杂)', fontsize=12, color='#E74C3C', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-1.5, 1.5)

fig.suptitle('偏差-方差分解：模型复杂度的两面\n(好模型 = 既不太简单也不太复杂)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_bias_variance.png')

print("分类评估指标图生成完成！")
