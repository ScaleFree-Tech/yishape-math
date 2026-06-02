"""
Generate matplotlib images for 7.2 Reverse Mode AD
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)

output_dir = os.path.join(os.path.dirname(__file__), 'images')


def create_gradient_accumulation():
    """Create gradient accumulation visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Single path
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title('Single Path: y = x * x\n∂y/∂x = 2x', fontsize=13, fontweight='bold')

    # Nodes
    positions = {
        'x': (2, 4),
        'mul': (5, 4),
        'y': (8, 4),
    }

    for name, (x, y) in positions.items():
        color = 'lightblue' if name == 'x' else 'lightyellow' if name == 'mul' else 'lightgreen'
        circle = plt.Circle((x, y), 0.8, facecolor=color, edgecolor='black', linewidth=2, zorder=5)
        ax1.add_patch(circle)
        ax1.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Arrows
    ax1.annotate('', xy=(4.2, 4), xytext=(2.8, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.annotate('', xy=(7.2, 4), xytext=(5.8, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Gradient flow (backward)
    ax1.annotate('', xy=(5.8, 3.5), xytext=(7.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    ax1.annotate('', xy=(2.8, 3.5), xytext=(4.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))

    ax1.text(5, 2, '∂y/∂x = 2x', ha='center', fontsize=12, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.axis('off')

    # Multiple paths (gradient accumulation)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_title('Multiple Paths: y = x*x + x*x\n∂y/∂x = 2x + 2x = 4x (accumulated)', fontsize=13, fontweight='bold')

    # Nodes
    positions2 = {
        'x': (1, 4),
        'mul1': (4, 6),
        'mul2': (4, 2),
        'add': (7, 4),
        'y': (9, 4),
    }

    for name, (x, y) in positions2.items():
        color = 'lightblue' if name == 'x' else 'lightyellow' if 'mul' in name or name == 'add' else 'lightgreen'
        circle = plt.Circle((x, y), 0.7, facecolor=color, edgecolor='black', linewidth=2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    # Forward arrows
    ax2.annotate('', xy=(3.3, 5.7), xytext=(1.7, 4.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax2.annotate('', xy=(3.3, 2.3), xytext=(1.7, 3.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax2.annotate('', xy=(6.3, 4.3), xytext=(4.7, 5.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax2.annotate('', xy=(6.3, 3.7), xytext=(4.7, 2.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax2.annotate('', xy=(8.3, 4), xytext=(7.7, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Backward arrows (two paths to x)
    ax2.annotate('', xy=(1.7, 4.3), xytext=(3.3, 5.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    ax2.annotate('', xy=(1.7, 3.7), xytext=(3.3, 2.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))

    ax2.text(5, 1, 'Two paths → gradients ADD: 2x + 2x = 4x', ha='center', fontsize=11, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax2.axis('off')

    plt.suptitle('Gradient Accumulation in Computational Graph', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_accumulation.png'), bbox_inches='tight')
    plt.close()


def create_backward_flow():
    """Create backward pass flow visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_title('Backward Pass: Computing Gradients via Chain Rule',
                fontsize=14, fontweight='bold')

    # Forward pass (top)
    ax.text(7, 9.5, 'Forward Pass: x → y', ha='center', fontsize=13,
           fontweight='bold', color='blue')

    # Forward nodes
    forward_nodes = {
        'x₁': (2, 8), 'x₂': (2, 6), 'x₃': (2, 4),
        'f₁': (5, 7), 'f₂': (5, 5), 'f₃': (5, 3),
        'g': (8, 6), 'h': (8, 4),
        'y': (11, 5),
    }

    for name, (x, y) in forward_nodes.items():
        color = 'lightblue' if 'x' in name else 'lightyellow'
        circle = plt.Circle((x, y), 0.6, facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    # Forward arrows
    forward_edges = [
        ('x₁', 'f₁'), ('x₂', 'f₁'), ('x₂', 'f₂'), ('x₃', 'f₂'), ('x₃', 'f₃'),
        ('f₁', 'g'), ('f₂', 'g'), ('f₂', 'h'), ('f₃', 'h'),
        ('g', 'y'), ('h', 'y'),
    ]

    for start, end in forward_edges:
        x1, y1 = forward_nodes[start]
        x2, y2 = forward_nodes[end]
        ax.annotate('', xy=(x2 - 0.6, y2), xytext=(x1 + 0.6, y1),
                   arrowprops=dict(arrowstyle='->', color='lightblue', lw=1.5))

    # Backward pass (bottom)
    ax.text(7, 2.5, 'Backward Pass: y → x (Chain Rule)', ha='center', fontsize=13,
           fontweight='bold', color='red')

    # Gradient labels
    gradients = {
        '∂y/∂y': (11, 4.2),
        '∂y/∂g': (8, 5.2), '∂y/∂h': (8, 3.2),
        '∂y/∂f₁': (5, 6.2), '∂y/∂f₂': (5, 4.2), '∂y/∂f₃': (5, 2.2),
        '∂y/∂x₁': (2, 7.2), '∂y/∂x₂': (2, 5.2), '∂y/∂x₃': (2, 3.2),
    }

    for label, (x, y) in gradients.items():
        ax.text(x, y, label, ha='center', fontsize=9, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    # Chain rule formula
    ax.text(7, 1, 'Chain Rule: ∂y/∂xᵢ = Σⱼ (∂y/∂nⱼ) × (∂nⱼ/∂xᵢ)',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Input (leaf)'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Operator'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backward_flow.png'), bbox_inches='tight')
    plt.close()


def create_topological_sort():
    """Create topological sort visualization for backward pass"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before topological sort
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title('Computational Graph (Forward Order)', fontsize=12, fontweight='bold')

    nodes = {
        'x': (2, 6), 'W': (2, 3),
        'matmul': (5, 5), 'bias': (5, 2),
        'add': (8, 4), 'relu': (8, 2),
        'loss': (8, 0.5),
    }

    for name, (x, y) in nodes.items():
        color = 'lightblue' if name in ['x', 'W'] else 'lightyellow' if name != 'loss' else 'lightgreen'
        circle = plt.Circle((x, y), 0.6, facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax1.add_patch(circle)
        ax1.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)

    edges = [('x', 'matmul'), ('W', 'matmul'), ('matmul', 'add'),
             ('bias', 'add'), ('add', 'relu'), ('relu', 'loss')]

    for start, end in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax1.annotate('', xy=(x2 - 0.6, y2), xytext=(x1 + 0.6, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.text(5, 7.5, 'Forward: x → matmul → add → relu → loss',
            ha='center', fontsize=10, color='blue')
    ax1.axis('off')

    # After topological sort (backward)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_title('Topological Order for Backward Pass', fontsize=12, fontweight='bold')

    # Same nodes, but show backward order
    for name, (x, y) in nodes.items():
        color = 'lightblue' if name in ['x', 'W'] else 'lightyellow' if name != 'loss' else 'lightgreen'
        circle = plt.Circle((x, y), 0.6, facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)

    # Backward arrows (reversed)
    backward_edges = [('loss', 'relu'), ('relu', 'add'), ('add', 'matmul'),
                     ('matmul', 'x'), ('matmul', 'W'), ('add', 'bias')]

    for i, (start, end) in enumerate(backward_edges):
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax2.annotate('', xy=(x2 + 0.6, y2), xytext=(x1 - 0.6, y1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
        ax2.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.3, f'{i+1}', ha='center', fontsize=9,
                color='red', fontweight='bold')

    ax2.text(5, 7.5, 'Backward: loss → relu → add → matmul → x, W',
            ha='center', fontsize=10, color='red')
    ax2.axis('off')

    plt.suptitle('Topological Sort Ensures Correct Gradient Order', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topological_sort.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Generating images for 7.2 Reverse Mode AD...")
    create_gradient_accumulation()
    print("  [OK] gradient_accumulation.png")
    create_backward_flow()
    print("  [OK] backward_flow.png")
    create_topological_sort()
    print("  [OK] topological_sort.png")
    print(f"\nAll images saved to: {output_dir}")
