#!/usr/bin/env python3
"""
Generate publication-quality figure: "Most Confident Where It Knows Least"

Key message: Energy-based methods are systematically overconfident on
queries with ZERO training evidence (novel contexts).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Data
baseline_zero_coverage = 32  # % of all test triples with zero coverage
top100_zero_coverage = 84    # % of Energy's top-100 with zero coverage

# Error rates within each group
zero_cov_error = 84.5
zero_cov_hits = 15.5
nonzero_cov_error = 100.0  # Note: 100% error for non-zero coverage in top-100
nonzero_cov_hits = 0.0

# Colors - using a color-blind friendly palette
COLOR_ZERO_COV = '#E64B35'      # Red - danger/warning for zero coverage
COLOR_NONZERO_COV = '#4DBBD5'   # Blue - for non-zero coverage
COLOR_BASELINE = '#8C8C8C'      # Gray for baseline
COLOR_ERROR = '#E64B35'         # Red for errors
COLOR_CORRECT = '#00A087'       # Green for correct


def create_publication_figure():
    """
    Clean 2-panel figure optimized for NeurIPS 2-column format.
    """
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.4)

    # =========================================================================
    # LEFT: Stacked bar comparison (cleaner than pie)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])

    bar_width = 0.5
    positions = [0, 1]

    baseline_nonzero = 100 - baseline_zero_coverage
    top100_nonzero = 100 - top100_zero_coverage

    # Stacked bars
    ax1.bar(positions[0], baseline_nonzero, bar_width,
            color=COLOR_NONZERO_COV, edgecolor='white', linewidth=0.5)
    ax1.bar(positions[1], top100_nonzero, bar_width,
            color=COLOR_NONZERO_COV, edgecolor='white', linewidth=0.5)

    ax1.bar(positions[0], baseline_zero_coverage, bar_width,
            bottom=baseline_nonzero, color=COLOR_ZERO_COV,
            edgecolor='white', linewidth=0.5)
    ax1.bar(positions[1], top100_zero_coverage, bar_width,
            bottom=top100_nonzero, color=COLOR_ZERO_COV,
            edgecolor='white', linewidth=0.5)

    # Percentage labels
    ax1.text(positions[0], baseline_nonzero/2, f'{baseline_nonzero}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax1.text(positions[0], baseline_nonzero + baseline_zero_coverage/2, f'{baseline_zero_coverage}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax1.text(positions[1], top100_nonzero/2, f'{top100_nonzero}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax1.text(positions[1], top100_nonzero + top100_zero_coverage/2, f'{top100_zero_coverage}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Arrow showing the 2.6x increase
    ax1.annotate('', xy=(1, 92), xytext=(0, 40),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2,
                               connectionstyle='arc3,rad=0.25'))
    ax1.text(0.5, 68, '2.6x', ha='center', va='center', fontsize=11,
             fontweight='bold', color='#333333',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['All Test\nTriples', "Energy's\nTop-100"], fontsize=10)
    ax1.set_ylabel('Percentage', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(-0.5, 1.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Coverage Composition', fontsize=11, fontweight='bold', pad=8)

    # Legend
    handles = [
        mpatches.Patch(color=COLOR_ZERO_COV, label='Zero evidence'),
        mpatches.Patch(color=COLOR_NONZERO_COV, label='Has evidence'),
    ]
    ax1.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)

    # =========================================================================
    # RIGHT: Error rate comparison
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])

    x = np.array([0, 1])
    width = 0.35

    errors = [84.5, 100]
    hits = [15.5, 0]

    bars1 = ax2.bar(x - width/2, errors, width, label='Error@1', color=COLOR_ERROR,
                    edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, hits, width, label='Hits@1', color=COLOR_CORRECT,
                    edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, hits):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Zero Coverage\n(84 predictions)',
                         'Has Coverage\n(16 predictions)'], fontsize=9)
    ax2.set_ylabel('Percentage', fontsize=10)
    ax2.set_ylim(0, 120)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    ax2.set_title('(b) Prediction Accuracy in Top-100', fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    return fig


def create_waffle_figure():
    """
    Waffle chart: visceral 10x10 grid showing 84 red / 16 blue squares.
    Single-column width for supplementary or inline use.
    """
    fig, ax = plt.subplots(figsize=(3.2, 3.4))

    grid_size = 10
    n_zero_cov = 84

    # Create grid
    grid = np.zeros((grid_size, grid_size))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_zero_cov:
                grid[i, j] = 1
            count += 1

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLOR_NONZERO_COV, COLOR_ZERO_COV])

    ax.imshow(grid, cmap=cmap, aspect='equal')

    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Energy's 100 Most Confident\nPredictions (FB15k-237)",
                 fontsize=11, fontweight='bold', pad=12)

    # Legend below
    fig.text(0.5, 0.08, '84% Zero Coverage', ha='center', fontsize=11,
             fontweight='bold', color=COLOR_ZERO_COV)
    fig.text(0.5, 0.02, '16% Has Evidence  (vs. 32% baseline)', ha='center',
             fontsize=9, color=COLOR_NONZERO_COV)

    plt.subplots_adjust(bottom=0.18)
    return fig


def create_combined_figure():
    """
    Ultimate publication figure: waffle + bar chart side by side.
    This is the most visually striking version.
    """
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = GridSpec(1, 2, width_ratios=[0.9, 1.1], wspace=0.3)

    # =========================================================================
    # LEFT: Waffle chart (visceral)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])

    grid_size = 10
    n_zero_cov = 84

    grid = np.zeros((grid_size, grid_size))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_zero_cov:
                grid[i, j] = 1
            count += 1

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLOR_NONZERO_COV, COLOR_ZERO_COV])

    ax1.imshow(grid, cmap=cmap, aspect='equal')
    ax1.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax1.grid(which='minor', color='white', linewidth=2)
    ax1.tick_params(which='minor', size=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax1.set_title("(a) Energy's Top-100\nMost Confident", fontsize=11, fontweight='bold', pad=8)

    # Add text annotation below
    ax1.text(4.5, 11.5, '84% Zero Evidence', ha='center', va='top',
             fontsize=10, fontweight='bold', color=COLOR_ZERO_COV)
    ax1.text(4.5, 12.8, '(vs. 32% baseline)', ha='center', va='top',
             fontsize=9, color='gray', style='italic')

    # =========================================================================
    # RIGHT: Grouped bar chart for error rates
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])

    x = np.array([0, 1.1])
    width = 0.38

    errors = [84.5, 100]
    hits = [15.5, 0]

    bars1 = ax2.bar(x - width/2, errors, width, label='Error@1', color=COLOR_ERROR,
                    edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, hits, width, label='Hits@1', color=COLOR_CORRECT,
                    edgecolor='white', linewidth=0.5)

    # Value labels inside bars for cleaner look
    ax2.text(bars1[0].get_x() + bars1[0].get_width()/2, bars1[0].get_height() - 8,
             f'{errors[0]:.0f}%', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    ax2.text(bars1[1].get_x() + bars1[1].get_width()/2, bars1[1].get_height() - 8,
             f'{errors[1]:.0f}%', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    ax2.text(bars2[0].get_x() + bars2[0].get_width()/2, bars2[0].get_height() + 2,
             f'{hits[0]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(bars2[1].get_x() + bars2[1].get_width()/2, 3,
             f'{hits[1]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Zero Coverage\n(84 queries)', 'Has Coverage\n(16 queries)'], fontsize=10)
    ax2.set_ylabel('Percentage', fontsize=10)
    ax2.set_ylim(0, 112)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper left', frameon=False, fontsize=9, bbox_to_anchor=(0.02, 0.98))
    ax2.set_title('(b) Accuracy in Top-100 Confident', fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os

    output_dir = '/Users/i767700/Github/kg-bayesian-prior/paper/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate all figure variants
    print("Generating publication figures...")

    # Option 1: Clean stacked bar + grouped bar
    fig1 = create_publication_figure()
    fig1.savefig(f'{output_dir}/confident_wrong_bars.pdf', format='pdf')
    fig1.savefig(f'{output_dir}/confident_wrong_bars.png', format='png', dpi=300)
    print(f"  [1] Stacked bars: {output_dir}/confident_wrong_bars.pdf")
    plt.close(fig1)

    # Option 2: Waffle chart only (for supplementary)
    fig2 = create_waffle_figure()
    fig2.savefig(f'{output_dir}/confident_wrong_waffle.pdf', format='pdf')
    fig2.savefig(f'{output_dir}/confident_wrong_waffle.png', format='png', dpi=300)
    print(f"  [2] Waffle chart: {output_dir}/confident_wrong_waffle.pdf")
    plt.close(fig2)

    # Option 3: Combined waffle + bar (RECOMMENDED)
    fig3 = create_combined_figure()
    fig3.savefig(f'{output_dir}/confident_wrong_combined.pdf', format='pdf')
    fig3.savefig(f'{output_dir}/confident_wrong_combined.png', format='png', dpi=300)
    print(f"  [3] Combined (recommended): {output_dir}/confident_wrong_combined.pdf")
    plt.close(fig3)

    print("\n" + "="*70)
    print("RECOMMENDED: confident_wrong_combined.pdf")
    print("="*70)
    print("""
SUGGESTED CAPTION:

\\textbf{Energy-based uncertainty is overconfident on novel contexts.}
(a) Among Energy's 100 most confident predictions on FB15k-237, 84\\%
have zero training evidence for the queried (entity, relation) pair---2.6$\\times$
higher than the 32\\% baseline. (b) These zero-coverage predictions achieve
only 15.5\\% Hits@1, while the remaining 16 predictions with coverage achieve
0\\% Hits@1. The model is systematically most confident where it has the
least basis for confidence.
""")
    print("="*70)
