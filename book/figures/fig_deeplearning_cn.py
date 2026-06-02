#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图5.4/5.5 深度学习与集成学习：反向传播、Dropout、Bagging vs Boosting
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def savefig(fig, name):
    fig.savefig(f'/home/reremouse/work/yishape-math/book/figures/{name}',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}'); plt.close(fig)

# ========== 图5.4.1 反向传播图解 ==========
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-1, 7.5); ax.axis('off')
layer_configs = [(2,'输入层\n(x1,x2)'), (3,'隐藏层1\n(ReLU)'), (2,'隐藏层2\n(ReLU)'), (1,'输出层\n(σ)')]
layer_x = [0, 1.5, 3.0, 4.5]
colors = ['#3498DB','#27AE60','#27AE60','#E74C3C']
for li, (n, label, xpos, color) in enumerate(zip([n for n,_ in layer_configs],
                                                   [l for _,l in layer_configs],
                                                   layer_x, colors)):
    ny = np.linspace(0.5, 6, n)
    for yn in ny:
        ax.add_patch(plt.Circle((xpos, yn), 0.22, color=color, alpha=0.85, zorder=5))
    ax.text(xpos, 7.0, label, ha='center', va='bottom', fontsize=10,
            fontweight='bold', color=color)
    if li < len(layer_configs)-1:
        next_n = layer_configs[li+1][0]
        next_y = np.linspace(0.5, 6, next_n)
        for yc in ny:
            for yn in next_y:
                lw = 0.4 if li==0 else 0.7
                ax.plot([xpos+0.22, xpos+1.5-0.22], [yc, yn],
                        color='#95A5A6', lw=lw, alpha=0.4)
# 标注
ax.annotate('', xy=(1.3, 5), xytext=(0.5, 5),
            arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2))
ax.text(0.9, 5.3, '前向传播\n(Forward Pass)', fontsize=9, color='#9B59B6',
        ha='center', style='italic')
ax.annotate('', xy=(0.5, 5), xytext=(1.3, 5),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
ax.text(0.9, 4.6, '反向传播\n(Backward Pass)', fontsize=9, color='#E74C3C',
        ha='center', style='italic')
# 损失标注
ax.text(4.7, 3, 'Loss = L(y,ŷ)\n↓\n链式法则\n∂L/∂w = ∂L/∂ŷ·∂ŷ/∂a·∂a/∂z·∂z/∂w',
        fontsize=9, color='#555555', va='center',
        bbox=dict(boxstyle='round', facecolor='#F8F8F8', alpha=0.8))
ax.set_title('神经网络反向传播（Backpropagation）示意图\n'
             '前向传播：输入→隐藏层→输出，计算损失\n'
             '反向传播：从输出层向前，计算梯度并更新权重',
             fontsize=12, fontweight='bold')
savefig(fig, 'fig_5_4_1_backprop_detail.png')

# ========== 图5.4.2 Dropout 正则化 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# 全连接
ax = axes[0]
layers = [4, 6, 6, 4]
lx = [0, 1.5, 3, 4.5]
for li, (n, xpos) in enumerate(zip(layers, lx)):
    ny = np.linspace(0.5, 5.5, n)
    for yn in ny:
        ax.add_patch(plt.Circle((xpos, yn), 0.18, color='#3498DB', alpha=0.8))
        if li < len(layers)-1:
            next_n = layers[li+1]
            next_y = np.linspace(0.5, 5.5, next_n)
            for yn2 in next_y:
                ax.plot([xpos+0.18, xpos+1.5-0.18], [yn, yn2],
                        color='#BDC3C7', lw=0.3, alpha=0.4)
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 6.5); ax.axis('off')
ax.set_title('标准全连接网络\n（所有连接激活，参数多→易过拟合）',
              fontsize=12, fontweight='bold')
# Dropout
ax = axes[1]
dropout_rate = 0.5
np.random.seed(42)
for li, (n, xpos) in enumerate(zip(layers, lx)):
    ny = np.linspace(0.5, 5.5, n)
    for yn in ny:
        if li > 0 and li < len(layers)-1 and np.random.rand() > dropout_rate:
            ax.add_patch(plt.Circle((xpos, yn), 0.18, color='#E74C3C', alpha=0.8))
            ax.text(xpos+0.3, yn, '×', fontsize=10, color='#E74C3C', fontweight='bold')
        elif li == 0 or li == len(layers)-1:
            ax.add_patch(plt.Circle((xpos, yn), 0.18, color='#3498DB', alpha=0.8))
        if li < len(layers)-1:
            next_n = layers[li+1]
            next_y = np.linspace(0.5, 5.5, next_n)
            for yn2 in next_y:
                # 只画激活的连接
                active = (np.random.rand() > dropout_rate) if li > 0 else True
                if active:
                    ax.plot([xpos+0.18, xpos+1.5-0.18], [yn, yn2],
                            color='#BDC3C7', lw=0.3, alpha=0.3)
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 6.5); ax.axis('off')
ax.set_title(f'Dropout（丢弃率={dropout_rate}）\n'
             '训练时随机丢弃神经元→迫使网络不依赖单个神经元\n'
             '测试时使用全部神经元（权重按比例缩小）',
             fontsize=11, fontweight='bold')
ax.text(2.5, -0.3, '红色×：被丢弃的神经元（训练时不激活）',
        ha='center', fontsize=10, color='#E74C3C', style='italic')
savefig(fig, 'fig_5_4_2_dropout.png')

# ========== 图5.5.1 Bagging vs Boosting 详图 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
# 左上：Bagging
ax = axes[0,0]
np.random.seed(0)
x_b = np.linspace(0, 1, 100)
# 生成多个bootstrap样本
base_preds = []
for i in range(8):
    idx = np.random.choice(len(x_b), size=len(x_b), replace=True)
    x_boot = x_b[idx]
    y_boot = np.sin(2*np.pi*x_boot) + np.random.randn(len(x_boot))*0.15
    p = np.poly1d(np.polyfit(x_boot, y_boot, 1))
    base_preds.append(p(x_b))
base_preds = np.array(base_preds)
for bp in base_preds:
    ax.plot(x_b, bp, color='#BDC3C7', lw=0.8, alpha=0.4)
bagged = base_preds.mean(axis=0)
ax.plot(x_b, bagged, color='#3498DB', lw=3, label='Bagging 平均（方差降低）')
ax.plot(x_b, np.sin(2*np.pi*x_b), color='#E74C3C', lw=2, ls='--', label='真实函数')
ax.set_title('Bagging（Bootstrap Aggregating）\n'
             '并行训练多棵独立树，平均预测→降低方差', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('x'); ax.set_ylabel('y')
# 右上：Boosting
ax = axes[0,1]
np.random.seed(42)
x_b2 = np.linspace(0, 1, 80)
y_b2 = np.sin(2*np.pi*x_b2) + np.random.randn(80)*0.2
# AdaBoost-like sequential
boost_preds = []
current_residual = y_b2.copy()
for i in range(5):
    p = np.poly1d(np.polyfit(x_b2, current_residual, 1))
    pred = p(x_b2)
    boost_preds.append(pred)
    current_residual = current_residual - 0.3*pred
boost_preds = np.array(boost_preds)
cumulative = np.cumsum(boost_preds, axis=0)
colors_b = plt.cm.Reds(np.linspace(0.3, 0.9, 5))
for i, (cp, color) in enumerate(zip(cumulative, colors_b)):
    label = f'第{i+1}棵树' if i==0 else f'第{i+1}棵'
    ax.plot(x_b2, cp, color=color, lw=1.5, alpha=0.7, label=label)
ax.plot(x_b2, cumulative[-1], color='#27AE60', lw=3, label='Boosting 累加（偏差降低）')
ax.plot(x_b2, np.sin(2*np.pi*x_b2), color='#E74C3C', lw=2, ls='--', label='真实函数')
ax.set_title('Boosting（序列化提升）\n'
             '每棵树学习前一棵树的残差，累加预测→降低偏差', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2); ax.set_xlabel('x'); ax.set_ylabel('y')
# 方差对比
ax = axes[1,0]
n_trees = np.array([1, 5, 10, 20, 50, 100])
var_reduction = 1 / n_trees
bias_sq = np.ones_like(n_trees) * 0.6
ax.plot(n_trees, bias_sq, 'o-', color='#3498DB', lw=2.5, ms=8, label='偏差2（不变）')
ax.plot(n_trees, var_reduction, 's-', color='#E74C3C', lw=2.5, ms=8, label='方差（∝ 1/n）')
total_error = bias_sq + var_reduction
ax.plot(n_trees, total_error, '^-', color='#27AE60', lw=2.5, ms=8, label='总误差')
best_n = n_trees[np.argmin(total_error)]
ax.axvline(best_n, color='gray', ls='--', lw=1.5)
ax.scatter([best_n], [total_error.min()], s=200, color='#F39C12', marker='*', zorder=10)
ax.set_title('Bagging：偏差不变，方差随树数↓\n最优树数在偏差≈方差处', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('树的数量'); ax.set_ylabel('误差')
ax.set_xscale('log')
# 收敛速度
ax = axes[1,1]
boost_rounds = np.arange(1, 101)
train_err = 0.4 * np.exp(-boost_rounds/20) + 0.05
test_err = train_err + 0.15 * (1 - np.exp(-boost_rounds/20))
ax.plot(boost_rounds, train_err, color='#27AE60', lw=2.5, label='训练误差（持续↓）')
ax.plot(boost_rounds, test_err, color='#E74C3C', lw=2.5, label='测试误差（先↓后↑）')
early_stop = 30
ax.axvline(early_stop, color='#3498DB', lw=2, ls='--', label=f'早停点 r={early_stop}')
ax.fill_betweenx([0, 0.55], early_stop, 100, color='gray', alpha=0.1)
ax.text(65, 0.25, '过拟合区', fontsize=10, ha='center', color='#555555', style='italic')
ax.set_title('Boosting：测试误差先降后升（过拟合风险）\n'
             '早停（Early Stopping）是关键技巧', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlabel('提升轮数'); ax.set_ylabel('误差')
ax.set_ylim(0, 0.55)
savefig(fig, 'fig_5_5_1_bagging_boosting.png')

# ========== 图5.5.2 XGBoost 树结构与分裂 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 决策树可视化
ax = axes[0]
# 手动画树
def draw_tree(ax):
    ax.clear(); ax.set_xlim(-1, 11); ax.set_ylim(-0.5, 6); ax.axis('off')
    # 根节点
    ax.add_patch(plt.Rectangle((4, 5), 2, 0.7, facecolor='#3498DB', edgecolor='black', lw=1.5))
    ax.text(5, 5.35, 'x1 < 0.5？', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    # 左边
    ax.annotate('', xy=(2, 4), xytext=(4.5, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((1, 3.3), 2, 0.7, facecolor='#27AE60', edgecolor='black', lw=1.5))
    ax.text(2, 3.65, 'x2 < 1.3？', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # 右边
    ax.annotate('', xy=(7, 4), xytext=(5.5, 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((6, 3.3), 2, 0.7, facecolor='#27AE60', edgecolor='black', lw=1.5))
    ax.text(7, 3.65, 'x3 < 2.0？', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # 左左
    ax.annotate('', xy=(0.5, 2), xytext=(1.5, 3.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Circle((1, 1.3), 0.5, facecolor='#F39C12', edgecolor='black', lw=1.5))
    ax.text(1, 1.3, 'y=0.8', ha='center', va='center', fontsize=8, fontweight='bold')
    # 左右
    ax.annotate('', xy=(3.5, 2), xytext=(2.5, 3.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Circle((4, 1.3), 0.5, facecolor='#F39C12', edgecolor='black', lw=1.5))
    ax.text(4, 1.3, 'y=1.2', ha='center', va='center', fontsize=8, fontweight='bold')
    # 右左
    ax.annotate('', xy=(6.5, 2), xytext=(6.5, 3.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Circle((7, 1.3), 0.5, facecolor='#F39C12', edgecolor='black', lw=1.5))
    ax.text(7, 1.3, 'y=1.5', ha='center', va='center', fontsize=8, fontweight='bold')
    # 右右
    ax.annotate('', xy=(9.5, 2), xytext=(7.5, 3.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Circle((10, 1.3), 0.5, facecolor='#F39C12', edgecolor='black', lw=1.5))
    ax.text(10, 1.3, 'y=2.1', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_title('XGBoost 单棵树结构\n'
                 '每个节点是一个特征分裂，叶节点是预测值', fontsize=11, fontweight='bold')
draw_tree(ax)
# 特征重要性
ax = axes[1]
features = ['x1\n(年龄)', 'x2\n(收入)', 'x3\n(登录频率)', 'x4\n(历史购买)', 'x5\n(设备类型)']
importance = [0.35, 0.28, 0.20, 0.12, 0.05]
bars = ax.barh(features, importance, color=['#3498DB','#27AE60','#9B59B6','#E74C3C','#F39C12'])
ax.set_xlabel('重要性得分（Gain）', fontsize=11)
ax.set_title('XGBoost 特征重要性排名\n'
             '基于分裂带来的目标函数增益（Gain）', fontsize=11, fontweight='bold')
for bar, imp in zip(bars, importance):
    ax.text(imp+0.01, bar.get_y()+bar.get_height()/2,
            f'{imp:.0%}', va='center', fontsize=10)
ax.set_xlim(0, 0.45)
savefig(fig, 'fig_5_5_2_xgboost_tree.png')

print("深度学习与集成学习 figures done!")
