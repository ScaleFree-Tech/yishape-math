#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble methods and neural network structure figures"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-whitegrid')

def savefig(fig, name):
    path = f'/home/reremouse/work/yishape-math/book/figures/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

# ========== Fig 1: Ensemble Strategies ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
n_trees_list = [1, 5, 10, 20, 50, 100]
base_variance = 1.0
base_bias_sq = 0.8
x_pos = np.arange(len(n_trees_list))
width = 0.35

bias_sq = [base_bias_sq for _ in n_trees_list]
variance = [base_variance / n for n in n_trees_list]

bars1 = ax.bar(x_pos - width/2, bias_sq, width, label='Bias^2', color='#3498DB', alpha=0.85)
bars2 = ax.bar(x_pos + width/2, variance, width, label='Variance', color='#E74C3C', alpha=0.85)
ax.set_xlabel('Number of Trees', fontsize=12)
ax.set_ylabel('Error Contribution', fontsize=12)
ax.set_title('How Bagging Reduces Variance\n(Bias unchanged, variance drops as 1/n)', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in n_trees_list])
ax.legend(fontsize=11)

ax = axes[1]
n_rounds = 50
train_errors = [0.5 * np.exp(-r/12) + 0.05 for r in range(n_rounds)]
test_errors = [0.5 * np.exp(-r/12) + 0.15 + 0.05*np.sin(r/10) for r in range(n_rounds)]
rounds = np.arange(1, n_rounds+1)
ax.plot(rounds, train_errors, color='#27AE60', lw=2.5, label='Training error (keeps dropping)')
ax.plot(rounds, test_errors, color='#E74C3C', lw=2.5, label='Test error (drops then rises -> overfit)')
ax.axvline(15, color='gray', lw=1.5, ls='--', label='Early stopping point')
ax.fill_betweenx([0, 0.55], 15, 50, color='gray', alpha=0.1)
ax.text(32, 0.25, 'Overfitting zone', fontsize=10, ha='center', color='#555555', style='italic')
ax.set_xlabel('Boosting Rounds', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Boosting: Train Error Keeps Dropping\nTest Error Drops Then Rises (Overfitting Risk)', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.55)

fig.suptitle('Random Forest (Bagging) vs Boosting\n(Two different ensemble strategies)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_ensemble_strategies.png')

# ========== Fig 2: Bayesian Updating ==========
from scipy.stats import beta
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
x = np.linspace(0, 1, 300)

# Prior
prior_a, prior_b = 2, 2
ax = axes[0]
ax.plot(x, beta.pdf(x, prior_a, prior_b), color='#95A5A6', lw=2.5, label='Prior')
ax.set_title('Prior Distribution\n(Belief before data)', fontsize=12)
ax.set_xlabel('theta (success rate)', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.legend(fontsize=10)

# Weak data
post_a, post_b = prior_a + 5 - 1, prior_b + 3 - 1
ax = axes[1]
ax.plot(x, beta.pdf(x, prior_a, prior_b), color='#95A5A6', lw=2, ls='--', label='Prior')
ax.plot(x, beta.pdf(x, 5, 3), color='#3498DB', lw=2, ls='--', alpha=0.6, label='Likelihood')
ax.plot(x, beta.pdf(x, post_a, post_b), color='#3498DB', lw=2.5, label='Posterior')
ax.axvline(post_a/(post_a+post_b), color='#3498DB', lw=1.5, ls=':', alpha=0.8)
ax.set_title('Bayesian Update (Weak Data)\n(Both prior and data matter)', fontsize=12)
ax.set_xlabel('theta', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.legend(fontsize=9)

# Strong data
post_a, post_b = prior_a + 50 - 1, prior_b + 30 - 1
ax = axes[2]
ax.plot(x, beta.pdf(x, prior_a, prior_b), color='#95A5A6', lw=2, ls='--', label='Prior')
ax.plot(x, beta.pdf(x, 50, 30), color='#27AE60', lw=2, ls='--', alpha=0.6, label='Likelihood')
ax.plot(x, beta.pdf(x, post_a, post_b), color='#27AE60', lw=2.5, label='Posterior')
ax.axvline(post_a/(post_a+post_b), color='#27AE60', lw=1.5, ls=':', alpha=0.8)
ax.set_title('Bayesian Update (Strong Data)\n(Data dominates, prior overwhelmed)', fontsize=12)
ax.set_xlabel('theta', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.legend(fontsize=9)

fig.suptitle('Bayesian Updating: Prior + Data -> Posterior\n(More data -> posterior approaches likelihood)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'fig_bayesian_update.png')

# ========== Fig 3: Neural Network Structure ==========
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 6.5)
ax.axis('off')

layer_configs = [(2, 'Input Layer\n(2 features)'), (4, 'Hidden Layer 1\n(4 neurons)'), (3, 'Hidden Layer 2\n(3 neurons)'), (1, 'Output Layer\n(1 output)')]
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

    if layer_idx < len(layer_configs) - 1:
        next_n = layer_configs[layer_idx+1][0]
        next_y = np.linspace(0.5, 5.5, next_n)
        for ny_current in node_y:
            for ny_next in next_y:
                ax.plot([x_pos+0.25, x_pos+1.5-0.25], [ny_current, ny_next],
                        color='#BDC3C7', lw=0.5, alpha=0.5)

ax.text(2.25, -0.15, '-> Forward: x.W + b -> ReLU/Sigmoid -> ... -> sigma -> Output\n    Backward: gradient from output layer, update weights',
        ha='center', fontsize=10, color='#555555', style='italic')
ax.set_title('Neural Network Structure\n(Fully Connected / Feedforward)', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig_neural_network_structure.png')

print("Ensemble and neural network figures done!")
