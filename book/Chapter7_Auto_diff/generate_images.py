"""
Generate matplotlib images for Chapter 7: Automatic Differentiation
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
os.makedirs(output_dir, exist_ok=True)


def create_forward_reverse_comparison():
    """Create comparison chart for forward vs reverse mode AD"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Forward mode
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.set_title('Forward Mode AD\n(One input direction at a time)', fontsize=14, fontweight='bold')

    # Draw nodes
    inputs = ['x₁', 'x₂', 'x₃']
    outputs = ['y₁', 'y₂']
    hiddens = ['h₁', 'h₂']

    # Input layer
    for i, inp in enumerate(inputs):
        ax1.scatter(2, 8 - i*2, s=800, c='lightblue', edgecolors='black', zorder=5)
        ax1.text(2, 8 - i*2, inp, ha='center', va='center', fontsize=12, fontweight='bold')

    # Hidden layer
    for i, h in enumerate(hiddens):
        ax1.scatter(5, 7 - i*2, s=800, c='lightyellow', edgecolors='black', zorder=5)
        ax1.text(5, 7 - i*2, h, ha='center', va='center', fontsize=12, fontweight='bold')

    # Output layer
    for i, out in enumerate(outputs):
        ax1.scatter(8, 7 - i*2, s=800, c='lightgreen', edgecolors='black', zorder=5)
        ax1.text(8, 7 - i*2, out, ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows
    for i in range(3):
        for j in range(2):
            ax1.annotate('', xy=(5, 7 - j*2), xytext=(2, 8 - i*2),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    for i in range(2):
        for j in range(2):
            ax1.annotate('', xy=(8, 7 - j*2), xytext=(5, 7 - i*2),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Highlight one path
    ax1.annotate('', xy=(5, 7), xytext=(2, 8),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.annotate('', xy=(8, 7), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax1.text(5, 0.5, 'Compute ∂y/∂x₁ only\n(one column of J)', ha='center',
            fontsize=11, style='italic', color='red')
    ax1.axis('off')

    # Reverse mode
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Reverse Mode AD\n(All inputs at once)', fontsize=14, fontweight='bold')

    # Draw nodes (same layout)
    for i, inp in enumerate(inputs):
        ax2.scatter(2, 8 - i*2, s=800, c='lightblue', edgecolors='black', zorder=5)
        ax2.text(2, 8 - i*2, inp, ha='center', va='center', fontsize=12, fontweight='bold')

    for i, h in enumerate(hiddens):
        ax2.scatter(5, 7 - i*2, s=800, c='lightyellow', edgecolors='black', zorder=5)
        ax2.text(5, 7 - i*2, h, ha='center', va='center', fontsize=12, fontweight='bold')

    for i, out in enumerate(outputs):
        ax2.scatter(8, 7 - i*2, s=800, c='lightgreen', edgecolors='black', zorder=5)
        ax2.text(8, 7 - i*2, out, ha='center', va='center', fontsize=12, fontweight='bold')

    # Forward arrows (gray)
    for i in range(3):
        for j in range(2):
            ax2.annotate('', xy=(5, 7 - j*2), xytext=(2, 8 - i*2),
                        arrowprops=dict(arrowstyle='->', color='lightgray', lw=1))
    for i in range(2):
        for j in range(2):
            ax2.annotate('', xy=(8, 7 - j*2), xytext=(5, 7 - i*2),
                        arrowprops=dict(arrowstyle='->', color='lightgray', lw=1))

    # Backward arrows (red, all paths)
    for i in range(2):
        for j in range(3):
            ax2.annotate('', xy=(5, 7 - i*2), xytext=(8, 7 - j*2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    for i in range(2):
        for j in range(3):
            ax2.annotate('', xy=(2, 8 - j*2), xytext=(5, 7 - i*2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))

    ax2.text(5, 0.5, 'Compute ∂y/∂x₁, ∂y/∂x₂, ∂y/∂x₃\n(one row of J)', ha='center',
            fontsize=11, style='italic', color='red')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forward_vs_reverse.png'), bbox_inches='tight')
    plt.close()


def create_jvp_geometry():
    """Create geometric interpretation of JVP"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a simple 2D function surface (contour)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple quadratic

    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=10)

    # Point of interest
    point = np.array([1.0, 0.5])
    ax.plot(point[0], point[1], 'ro', markersize=12, zorder=5, label='Point x₀')

    # Direction vector
    v = np.array([0.5, 0.8])
    v_norm = v / np.linalg.norm(v)

    # Draw direction vector
    ax.annotate('', xy=point + v_norm, xytext=point,
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(point[0] + v_norm[0]/2 + 0.1, point[1] + v_norm[1]/2,
           'v (tangent)', color='red', fontsize=12)

    # Tangent line
    tangent_start = point - 0.5 * v_norm
    tangent_end = point + 0.5 * v_norm
    ax.plot([tangent_start[0], tangent_end[0]],
           [tangent_start[1], tangent_end[1]],
           'r--', lw=2, label='Tangent line')

    # Gradient (perpendicular to contour)
    grad = np.array([2*point[0], 2*point[1]])
    grad_norm = grad / np.linalg.norm(grad) * 0.5
    ax.annotate('', xy=point + grad_norm, xytext=point,
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(point[0] + grad_norm[0], point[1] + grad_norm[1] + 0.15,
           '∇f (gradient)', color='green', fontsize=12)

    # JVP annotation
    jvp_value = np.dot(grad, v_norm)
    ax.text(0.02, 0.98, f'JVP = ∇f · v = {jvp_value:.3f}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Geometric Interpretation of JVP\nf(x) = x₁² + x₂² at x₀ = (1, 0.5)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jvp_geometry.png'), bbox_inches='tight')
    plt.close()


def create_hessian_visualization():
    """Create Hessian matrix visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Function 1: f = x² + y² (positive definite)
    ax1 = axes[0]
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z1 = X**2 + Y**2
    contour1 = ax1.contourf(X, Y, Z1, levels=20, cmap='RdYlGn_r')
    ax1.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax1.set_title('f = x² + y²\nH = [[2,0],[0,2]] (Positive Definite)\nBowl shape → local min',
                 fontsize=11)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    plt.colorbar(contour1, ax=ax1)

    # Function 2: f = x² - y² (indefinite - saddle point)
    ax2 = axes[1]
    Z2 = X**2 - Y**2
    contour2 = ax2.contourf(X, Y, Z2, levels=20, cmap='RdBu_r')
    ax2.plot(0, 0, 'g*', markersize=15, label='Saddle point')
    ax2.set_title('f = x² - y²\nH = [[2,0],[0,-2]] (Indefinite)\nSaddle shape → not extremum',
                 fontsize=11)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    plt.colorbar(contour2, ax=ax2)

    # Function 3: f = -x² - y² (negative definite)
    ax3 = axes[2]
    Z3 = -X**2 - Y**2
    contour3 = ax3.contourf(X, Y, Z3, levels=20, cmap='RdYlGn')
    ax3.plot(0, 0, 'r*', markersize=15, label='Maximum')
    ax3.set_title('f = -x² - y²\nH = [[-2,0],[0,-2]] (Negative Definite)\nHill shape → local max',
                 fontsize=11)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    plt.colorbar(contour3, ax=ax3)

    plt.suptitle('Hessian Matrix and Surface Curvature', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hessian_visualization.png'), bbox_inches='tight')
    plt.close()


def create_memory_checkpoint():
    """Create memory usage comparison for gradient checkpointing"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Without checkpointing
    ax1 = axes[0]
    layers = list(range(1, 9))
    memory_no_ckpt = layers  # O(n) memory

    ax1.bar(range(len(layers)), memory_no_ckpt, color='salmon', edgecolor='black')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Memory (activations stored)', fontsize=12)
    ax1.set_title('Without Checkpointing\nO(n) memory - store ALL activations', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f'Layer {i}' for i in layers], rotation=45)
    ax1.text(0.5, 0.85, 'Total: 8 units', transform=ax1.transAxes,
            fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='white'))

    # With checkpointing
    ax2 = axes[1]
    # Checkpoints at layers 2, 4, 6
    checkpoint_layers = [2, 4, 6]
    colors = ['lightgreen' if i in checkpoint_layers else 'lightgray' for i in layers]
    memory_ckpt = [1 if i not in checkpoint_layers else 2 for i in layers]

    bars = ax2.bar(range(len(layers)), memory_ckpt, color=colors, edgecolor='black')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Memory (activations stored)', fontsize=12)
    ax2.set_title('With Checkpointing\nO(√n) memory - store only checkpoints', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f'Layer {i}' for i in layers], rotation=45)

    # Legend
    legend_elements = [mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Checkpoint'),
                      mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Recompute on demand')]
    ax2.legend(handles=legend_elements, loc='upper right')
    ax2.text(0.5, 0.85, 'Total: ~3√8 ≈ 8.5 units\n(saves ~60% for deep nets)',
            transform=ax2.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='white'))

    plt.suptitle('Gradient Checkpointing: Time-Space Tradeoff', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_checkpoint.png'), bbox_inches='tight')
    plt.close()


def create_vmap_batch():
    """Create vmap batch processing visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Without vmap (loop)
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 6)
    ax1.set_title('Without vmap: Manual Loop', fontsize=14, fontweight='bold')

    # Input samples
    samples = ['Sample 1', 'Sample 2', 'Sample 3']
    for i, s in enumerate(samples):
        ax1.add_patch(FancyBboxPatch((0.5, 4 - i*1.5), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black'))
        ax1.text(1.5, 4.5 - i*1.5, s, ha='center', va='center', fontsize=10)

    # Function
    ax1.add_patch(FancyBboxPatch((4, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', edgecolor='black'))
    ax1.text(5.25, 3, 'fn(x)', ha='center', va='center', fontsize=11, fontweight='bold')

    # Output
    for i in range(3):
        ax1.add_patch(FancyBboxPatch((8, 4 - i*1.5), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black'))
        ax1.text(9, 4.5 - i*1.5, f'Result {i+1}', ha='center', va='center', fontsize=10)

    # Arrows (sequential)
    for i in range(3):
        ax1.annotate('', xy=(4, 3), xytext=(2.5, 4.5 - i*1.5),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        ax1.annotate('', xy=(8, 4.5 - i*1.5), xytext=(6.5, 3),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax1.text(3.5, 4.8 - i*1.5, f'Step {i+1}', fontsize=9, color='blue')

    ax1.text(6, 0.5, 'Sequential: 3 forward passes, 3 backward passes',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.axis('off')

    # With vmap
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6)
    ax2.set_title('With vmap: Vectorized Batch', fontsize=14, fontweight='bold')

    # Batch input
    ax2.add_patch(FancyBboxPatch((0.5, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightblue', edgecolor='black'))
    ax2.text(1.75, 3, 'Batch\n[Sample1,\nSample2,\nSample3]', ha='center', va='center', fontsize=9)

    # Vectorized function
    ax2.add_patch(FancyBboxPatch((4.5, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', edgecolor='black'))
    ax2.text(5.75, 3, 'vmap(fn)', ha='center', va='center', fontsize=11, fontweight='bold')

    # Batch output
    ax2.add_patch(FancyBboxPatch((8.5, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black'))
    ax2.text(9.75, 3, 'Batch\n[Result1,\nResult2,\nResult3]', ha='center', va='center', fontsize=9)

    # Arrows (parallel)
    ax2.annotate('', xy=(4.5, 3), xytext=(3, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(8.5, 3), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax2.text(6, 0.5, 'Parallel: 1 vectorized pass, automatic batching!',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vmap_batch.png'), bbox_inches='tight')
    plt.close()


def create_computational_graph_example():
    """Create detailed computational graph example"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Computational Graph: y = (x₁ + x₂) × sin(x₃)',
                fontsize=14, fontweight='bold')

    # Nodes
    nodes = {
        'x₁': (1, 8, 'lightblue'),
        'x₂': (1, 6, 'lightblue'),
        'x₃': (1, 4, 'lightblue'),
        'add': (4, 7, 'lightyellow'),
        'sin': (4, 4, 'lightyellow'),
        'mul': (8, 5.5, 'lightyellow'),
        'y': (12, 5.5, 'lightgreen'),
    }

    for name, (x, y, color) in nodes.items():
        circle = plt.Circle((x, y), 0.8, facecolor=color, edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Edges
    edges = [
        ('x₁', 'add'),
        ('x₂', 'add'),
        ('x₃', 'sin'),
        ('add', 'mul'),
        ('sin', 'mul'),
        ('mul', 'y'),
    ]

    for start, end in edges:
        x1, y1 = nodes[start][:2]
        x2, y2 = nodes[end][:2]
        ax.annotate('', xy=(x2 - 0.8, y2), xytext=(x1 + 0.8, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Add value annotations
    values = {
        'x₁': '1.0',
        'x₂': '2.0',
        'x₃': 'π/2',
        'add': '3.0',
        'sin': '1.0',
        'mul': '3.0',
        'y': '3.0',
    }

    for name, (x, y, _) in nodes.items():
        ax.text(x, y - 1.2, f'val={values[name]}', ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Add gradient annotations (backward pass)
    gradients = {
        'y': '∂L/∂y=1',
        'mul': '∂L/∂a=1\n∂L/∂b=3',
        'add': '∂L/∂x₁=1',
        'sin': '∂L/∂x₃=3',
    }

    for name, (x, y, _) in nodes.items():
        if name in gradients:
            ax.text(x, y + 1.2, gradients[name], ha='center', fontsize=8,
                   color='red', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Input (leaf)'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Operator'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.text(7, 9, 'Forward: x₁ + x₂ = 3, sin(π/2) = 1, 3 × 1 = 3',
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    ax.text(7, 0.5, 'Backward: ∂y/∂x₁ = 1, ∂y/∂x₂ = 1, ∂y/∂x₃ = 3cos(π/2) = 0',
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computational_graph.png'), bbox_inches='tight')
    plt.close()


def create_ad_history_timeline():
    """Create timeline of automatic differentiation history"""
    fig, ax = plt.subplots(figsize=(14, 5))

    events = [
        (1964, 'Seinnhausen\nFirst AD concepts', 'blue'),
        (1970, 'Wengert\nSimple forward AD', 'green'),
        (1980, 'Speelpenning\nFast Jacobian evaluation', 'orange'),
        (1986, 'Rumelhart et al.\nBackpropagation popularized', 'red'),
        (2008, 'Baydin et al.\nModern AD survey', 'purple'),
        (2015, 'PyTorch/TensorFlow\nAD in deep learning', 'brown'),
        (2018, 'JAX\nComposable AD transforms', 'pink'),
    ]

    years = [e[0] for e in events]
    ax.set_xlim(1960, 2025)
    ax.set_ylim(-2, 3)

    # Draw timeline
    ax.axhline(y=0, color='black', linewidth=2)

    for year, label, color in events:
        ax.plot(year, 0, 'o', markersize=12, color=color, zorder=5)
        ax.text(year, 0.3, str(year), ha='center', fontsize=9, fontweight='bold')
        ax.text(year, -0.5, label, ha='center', fontsize=8, va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

    ax.set_title('History of Automatic Differentiation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ad_history.png'), bbox_inches='tight')
    plt.close()


def create_loss_landscape():
    """Create loss landscape visualization for optimization"""
    fig = plt.figure(figsize=(12, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) * np.exp(-0.1*(X**2 + Y**2))

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Landscape\n(gradient points downhill)', fontsize=11)

    # Gradient descent path
    theta = np.array([2.5, 2.5])
    path = [theta.copy()]
    lr = 0.3
    for _ in range(20):
        # Simple gradient (numerical)
        grad = np.array([
            np.cos(theta[0]) * np.cos(theta[1]) * np.exp(-0.1*theta[0]**2),
            -np.sin(theta[0]) * np.sin(theta[1]) * np.exp(-0.1*theta[1]**2)
        ])
        theta = theta - lr * grad
        path.append(theta.copy())

    path = np.array(path)
    ax1.plot(path[:, 0], path[:, 1], [np.sin(p[0])*np.cos(p[1])*np.exp(-0.1*sum(p**2)) for p in path],
            'r-o', markersize=3, linewidth=2, label='Gradient descent')

    # 2D contour with path
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    ax2.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=2, label='Gradient descent')
    ax2.plot(path[0, 0], path[0, 1], 'ro', markersize=10, label='Start')
    ax2.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('Loss Contour + Gradient Path', fontsize=11)
    ax2.legend()
    plt.colorbar(contour, ax=ax2)

    plt.suptitle('Optimization: Following the Gradient', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_landscape.png'), bbox_inches='tight')
    plt.close()


def create_chain_rule_visualization():
    """Create chain rule visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_title('Chain Rule: ∂f/∂x = ∂f/∂g × ∂g/∂x', fontsize=14, fontweight='bold')

    # Nodes
    positions = {
        'x': (2, 4),
        'g(x)': (7, 4),
        'f(g)': (12, 4),
    }

    for name, (x, y) in positions.items():
        circle = plt.Circle((x, y), 1, facecolor='lightblue', edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # Forward arrow
    ax.annotate('', xy=(6, 4), xytext=(3, 4),
               arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.text(4.5, 5, '∂g/∂x = 2x', ha='center', fontsize=12, color='blue',
           bbox=dict(boxstyle='round', facecolor='lightcyan'))

    ax.annotate('', xy=(11, 4), xytext=(8, 4),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(9.5, 5, '∂f/∂g = cos(g)', ha='center', fontsize=12, color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Backward arrow (chain rule)
    ax.annotate('', xy=(3, 3), xytext=(11, 3),
               arrowprops=dict(arrowstyle='->', color='red', lw=3, linestyle='--'))
    ax.text(7, 2, '∂f/∂x = ∂f/∂g × ∂g/∂x = cos(x²) × 2x',
           ha='center', fontsize=13, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Example
    ax.text(7, 7, 'Example: f(g(x)) = sin(x²)\nChain Rule: d/dx[sin(x²)] = cos(x²) · 2x',
           ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chain_rule.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Generating images for Chapter 7: Automatic Differentiation...")
    create_forward_reverse_comparison()
    print("  [OK] forward_vs_reverse.png")
    create_jvp_geometry()
    print("  [OK] jvp_geometry.png")
    create_hessian_visualization()
    print("  [OK] hessian_visualization.png")
    create_memory_checkpoint()
    print("  [OK] memory_checkpoint.png")
    create_vmap_batch()
    print("  [OK] vmap_batch.png")
    create_computational_graph_example()
    print("  [OK] computational_graph.png")
    create_ad_history_timeline()
    print("  [OK] ad_history.png")
    create_loss_landscape()
    print("  [OK] loss_landscape.png")
    create_chain_rule_visualization()
    print("  [OK] chain_rule.png")
    print(f"\nAll images saved to: {output_dir}")
