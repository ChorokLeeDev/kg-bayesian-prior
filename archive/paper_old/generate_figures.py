"""Generate figures for GP-KGE paper."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (5, 3.5),
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# Figure 1: Relation Threshold Analysis
# =============================================================================
def plot_relation_threshold():
    """Plot AUROC vs number of relations."""

    # Data
    datasets = ['WN18RR', 'YAGO3-10', 'FB15k-237']
    relations = [11, 37, 237]
    gpkge_auroc = [0.629, 0.830, 0.854]
    distmult_auroc = [0.860, 0.619, 0.550]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, gpkge_auroc, width, label='GP-KGE (Ours)',
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, distmult_auroc, width, label='DistMult',
                   color='#3498db', edgecolor='black', linewidth=0.5)

    # Add threshold line
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(0.65, 0.92, 'Threshold\n(~30 relations)', fontsize=9, color='red',
            transform=ax.transAxes, ha='left')

    # Annotations
    for i, (g, d) in enumerate(zip(gpkge_auroc, distmult_auroc)):
        diff = ((g - d) / d) * 100
        if diff > 0:
            ax.annotate(f'+{diff:.0f}%', xy=(i - width/2, g + 0.02),
                       ha='center', fontsize=9, fontweight='bold', color='#27ae60')
        else:
            ax.annotate(f'{diff:.0f}%', xy=(i - width/2, g + 0.02),
                       ha='center', fontsize=9, color='#c0392b')

    ax.set_ylabel('AUROC (OOD Detection)')
    ax.set_xlabel('Dataset (# Relations)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n({r})' for d, r in zip(datasets, relations)])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig('figures/relation_threshold.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/relation_threshold.png', bbox_inches='tight', dpi=300)
    print("Saved: figures/relation_threshold.pdf")


# =============================================================================
# Figure 2: Model Architecture (Conceptual)
# =============================================================================
def plot_architecture():
    """Plot GP-KGE model architecture."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    c_kg = '#3498db'
    c_kernel = '#e74c3c'
    c_gp = '#2ecc71'
    c_output = '#9b59b6'

    # Knowledge Graph box
    kg_box = mpatches.FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                      facecolor=c_kg, alpha=0.3, edgecolor=c_kg, linewidth=2)
    ax.add_patch(kg_box)
    ax.text(1.5, 3, 'Knowledge\nGraph\n$\\mathcal{G}$', ha='center', va='center', fontsize=10)

    # Per-relation Laplacians
    for i, r in enumerate(['$L_{r_1}$', '$L_{r_2}$', '$\\cdots$', '$L_{r_M}$']):
        y = 4.5 - i * 0.8
        ax.annotate('', xy=(3.5, y), xytext=(2.5, 3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.text(3.7, y, r, fontsize=9, va='center')

    # Relation-aware Kernel box
    kernel_box = mpatches.FancyBboxPatch((4.5, 1.5), 2, 3, boxstyle="round,pad=0.1",
                                          facecolor=c_kernel, alpha=0.3, edgecolor=c_kernel, linewidth=2)
    ax.add_patch(kernel_box)
    ax.text(5.5, 3, 'Relation-Aware\nKernel\n$K = \\sum_r \\sigma_r^2 e^{-L_r/\\ell_r^2}$',
            ha='center', va='center', fontsize=9)

    # Arrow to GP
    ax.annotate('', xy=(7.5, 3), xytext=(6.5, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # GP Prior box
    gp_box = mpatches.FancyBboxPatch((7.5, 2), 2, 2, boxstyle="round,pad=0.1",
                                      facecolor=c_gp, alpha=0.3, edgecolor=c_gp, linewidth=2)
    ax.add_patch(gp_box)
    ax.text(8.5, 3, 'GP Prior\n$f \\sim \\mathcal{GP}(0, K)$',
            ha='center', va='center', fontsize=10)

    # Output arrows
    ax.annotate('', xy=(8.5, 1.5), xytext=(8.5, 2),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Outputs
    ax.text(8.5, 0.8, 'Entity Embeddings $\\mathbf{F}$\n+ Uncertainty $\\mathbf{S}$',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=c_output, alpha=0.3, edgecolor=c_output))

    # Title
    ax.text(5, 5.5, 'GP-KGE Architecture', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/architecture.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/architecture.png', bbox_inches='tight', dpi=300)
    print("Saved: figures/architecture.pdf")


# =============================================================================
# Figure 3: AUROC Comparison Bar Chart (Simple)
# =============================================================================
def plot_auroc_comparison():
    """Simple AUROC comparison on FB15k-237."""

    fig, ax = plt.subplots(figsize=(4, 3))

    models = ['GGPN', 'DistMult', 'GP-KGE\n(Ours)']
    aurocs = [0.221, 0.550, 0.854]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(models, aurocs, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.0)
    ax.set_title('OOD Detection on FB15k-237', fontsize=11)

    # Highlight GP-KGE
    bars[2].set_edgecolor('#27ae60')
    bars[2].set_linewidth(2)

    plt.tight_layout()
    plt.savefig('figures/auroc_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/auroc_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved: figures/auroc_comparison.pdf")


if __name__ == '__main__':
    plot_relation_threshold()
    plot_auroc_comparison()
    plot_architecture()
    print("\nAll figures generated!")
