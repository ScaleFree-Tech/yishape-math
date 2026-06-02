#!/usr/bin/env python3
"""
树集成、神经网络、贝叶斯更新可视化
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

plt.style.use('seaborn-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== 图1: Random Forest vs 单棵决策树偏差-方差 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左: 单棵树，多棵树
ax = axes[0]
np.random.seed(42)
n_trees_list = [1, 5, 10, 20, 50, 100]
bias_sq = []
variance = []
total_error = []

base_variance = 1.0
base_bias_sq = 0.8

for n_trees in n_trees_list:
    # Bagging 降低方差，但不改变偏差
    reduction = 1.0 / n_trees  # 近似
    var = base_variance * reduction
    bias = base_bias_sq
    bias_sq.append(bias)
    variance.append(var)
    total_error.append(bias + var)

x_pos = np.arange(len(n_trees_list))
width = 0.35

bars1 = ax.bar(x_pos - width/2, bias_sq, width, label='偏差² (Bias²)', color='#3498DB', alpha=0.85)
bars2 = ax.bar(x_pos + width/2, variance, width, label='方差 (Variance)', color='#E74C3C', alpha=0.85)

ax.set_xlabel('树的数量 (n_trees)', fontsize=12)
ax.set_ylabel('误差贡献', fontsize=12)
ax.set_title('Bagging 如何降低方差\n(偏差不变，方差随 1/n 下降)', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in n_trees_list])
ax.legend(fontsize=11)

# 右: Boosting 迭代过程
ax = axes[1]
n_rounds = 50
residuals = []
train_errors = []
test_errors = []

# 模拟 Boosting 过程
np.random.seed(42)
x = np.linspace(0, 1, 100)
true_y = np.sin(2*np.pi*x)
for r in range(n_rounds):
    residual = np.random.randn(100) * np.exp(-r/15) + 0.05*np.sin(r/3)
    residuals.append(residual)
    train_errors.append(0.5 * np.exp(-r/12) + 0.05)
    test_errors.append(0.5 * np.exp(-r/12) + 0.15 + 0.05*np.sin(r/10))

rounds = np.arange(1, n_rounds+1)
ax.plot(rounds, train_errors, color='#27AE60', lw=2.5, label='训练误差 (持续下降)')
ax.plot(rounds, test_errors, color='#E74C3C', lw=2.5, label='测试误差 (先降后升→过拟合)')
ax.axvline(15, color='gray', lw=1.5, ls='--', label='早停点 (Early Stopping)')

ax.fill_betweenx([0, 0.55], 15, 50, color='gray', alpha=0.1)
ax.text(32, 0.25, '过拟合区间', fontsize=10, ha='center', color='#555555', style='italic')

ax.set_xlabel('Boosting 轮数', fontsize=12)
ax.set_ylabel('误差', fontsize=12)
ax.set_title('Boosting: 训练误差持续下降\n测试误差先降后升（过拟合风险）', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.55)

fig.suptitle('Random Forest (Bagging) vs Boosting\n(集成方法的不同策略)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_ensemble_strategies.png')

# ========== 图2: 贝叶斯更新可视化 ==========
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

from scipy.stats import beta

# 先验: Beta(2, 2)
x = np.linspace(0, 1, 300)

# 三个场景: 弱数据、强数据、非常强数据
priors = [(2, 2, '#95A5A6', '先验 Beta(2,2)')]
likelihoods = [
    (5, 3, '#3498DB', '似然 Beta(5,3)\n(弱数据: 8次实验5次成功)'),
    (15, 9, '#27AE60', '似然 Beta(15,9)\n(中等数据: 24次实验15次成功)'),
    (50, 30, '#E74C3C', '似然 Beta(50,30)\n(强数据: 80次实验50次成功)'),
]

# 固定先验 Beta(2,2)
ax = axes[0]
prior_alpha, prior_beta = 2, 2
ax.plot(x, beta.pdf(x, prior_alpha, prior_beta), color='#95A5A6', lw=2.5, label='先验')
ax.set_title('先验分布\n(实验前的信念)', fontsize=12)
ax.set_xlabel('θ (成功率)', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.legend(fontsize=10)

for idx, (ax, (la, lb, lc, label)) in enumerate(zip(axes, likelihoods)):
    prior_a, prior_b = 2, 2
    post_a = prior_a + la - 1
    post_b = prior_b + lb - 1

    ax.plot(x, beta.pdf(x, prior_a, prior_b), color='#95A5A6', lw=2, ls='--', label='先验')
    ax.plot(x, beta.pdf(x, la, lb), color=lc, lw=2, ls='--', alpha=0.6, label='似然')
    ax.plot(x, beta.pdf(x, post_a, post_b), color=lc, lw=2.5, label='后验')
    ax.set_title(f'贝叶斯更新\n{label}', fontsize=11)
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('概率密度', fontsize=11)
    ax.legend(fontsize=9)
    # 标注后验均值
    post_mean = post_a / (post_a + post_b)
    ax.axvline(post_mean, color=lc, lw=1.5, ls=':', alpha=0.8)

fig.suptitle('贝叶斯更新：先验 + 数据 → 后验\n(数据越多，后验越被似然主导)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_bayesian_update.png')

# ========== 图3: 神经网络结构示意图 ==========
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 6.5)
ax.axis('off')

layer_configs = [(2, '输入层\n(2个特征)'), (4, '隐藏层1\n(4个神经元)'), (3, '隐藏层2\n(3个神经元)'), (1, '输出层\n(1个输出)')]
layer_colors = ['#3498DB', '#27AE60', '#27AE60', '#E74C3C']
layer_x = [0, 1.5, 3.0, 4.5]

for layer_idx, (n_nodes, label, x_pos, color) in enumerate(zip(
    [n for n, _ in layer_configs],
    [l for _, l in layer_configs],
    layer_x,
    layer_colors
)):
    node_y = np.linspace(0.5, 5.5, n_nodes)
    for ny in node_y:
        circle = plt.Circle((x_pos, ny), 0.25, color=color, alpha=0.85, zorder=5)
        ax.add_patch(circle)

    ax.text(x_pos, 6.2, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    # 连接线到下一层
    if layer_idx < len(layer_configs) - 1:
        next_n = layer_configs[layer_idx+1][0]
        next_y = np.linspace(0.5, 5.5, next_n)
        for ny_current in node_y:
            for ny_next in next_y:
                ax.plot([x_pos+0.25, x_pos+1.5-0.25], [ny_current, ny_next],
                        color='#BDC3C7', lw=0.5, alpha=0.5)

ax.text(2.25, -0.15, '← 前向传播: x·W + b → ReLU / Sigmoid → ... → σ → 输出\n    反向传播: 从输出层向前，计算梯度，更新权重',
        ha='center', fontsize=10, color='#555555', style='italic')

ax.set_title('神经网络结构示意\n(全连接 / Feedforward)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig_neural_network_structure.png')

print("集成+神经网络图生成完成！")
