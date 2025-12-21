#!/usr/bin/env python3
"""
Generate publication-quality figures for NeurIPS submission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create output directory
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA
# =============================================================================

RESULTS = {
    'WN18RR': {'GP': 0.647, 'Coverage': 0.657, 'CAGP': 0.871, 'synergy': 0.32},
    'FB15k-237': {'GP': 0.749, 'Coverage': 0.821, 'CAGP': 0.960, 'synergy': 0.17},
    'YAGO3-10': {'GP': 0.824, 'Coverage': 0.760, 'CAGP': 0.942, 'synergy': 0.14},
}

THEOREM_VALIDATION = {
    'WN18RR': {'predicted': 0.6808, 'observed': 0.6570, 'p_h': 0.636, 'p_t': 0.885, 's_r': 0.834},
    'FB15k-237': {'predicted': 0.8147, 'observed': 0.8210, 'p_h': 0.763, 'p_t': 0.905, 's_r': 0.960},
}

# Colors
COLORS = {
    'GP': '#6B7FD7',        # Muted blue
    'Coverage': '#F2A154',  # Muted orange
    'CAGP': '#5AAA6D',      # Muted green
    'synergy': '#D75A5A',   # Muted red
}


# =============================================================================
# FIGURE 1: Main Results Bar Chart with Synergy
# =============================================================================

def create_main_results_figure():
    """Create the main results bar chart showing synergy."""

    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = list(RESULTS.keys())
    x = np.arange(len(datasets))
    width = 0.25

    # Extract data
    gp_scores = [RESULTS[d]['GP'] for d in datasets]
    cov_scores = [RESULTS[d]['Coverage'] for d in datasets]
    cagp_scores = [RESULTS[d]['CAGP'] for d in datasets]

    # Create bars
    bars_gp = ax.bar(x - width, gp_scores, width, label='GP Variance',
                     color=COLORS['GP'], edgecolor='white', linewidth=0.5)
    bars_cov = ax.bar(x, cov_scores, width, label='Coverage',
                      color=COLORS['Coverage'], edgecolor='white', linewidth=0.5)
    bars_cagp = ax.bar(x + width, cagp_scores, width, label='CAGP (Ours)',
                       color=COLORS['CAGP'], edgecolor='white', linewidth=0.5)

    # Add synergy annotations
    for i, d in enumerate(datasets):
        best_single = max(RESULTS[d]['GP'], RESULTS[d]['Coverage'])
        synergy_pct = int(RESULTS[d]['synergy'] * 100)

        # Draw synergy arrow
        ax.annotate('',
                    xy=(i + width, RESULTS[d]['CAGP']),
                    xytext=(i + width, best_single),
                    arrowprops=dict(arrowstyle='->', color=COLORS['synergy'], lw=2))

        # Synergy label
        mid_y = (RESULTS[d]['CAGP'] + best_single) / 2
        ax.text(i + width + 0.12, mid_y, f'+{synergy_pct}%',
                color=COLORS['synergy'], fontweight='bold', fontsize=11, va='center')

    # Formatting
    ax.set_ylabel('AUROC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')

    # Add horizontal line at 0.5 (random)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.4, 0.51, 'random', color='gray', fontsize=9, style='italic')

    # Title
    ax.set_title('OOD Detection: CAGP Synergy Across Datasets', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_main_results.pdf')
    plt.savefig(FIGURES_DIR / 'fig1_main_results.png')
    plt.close()
    print(f"Saved: fig1_main_results.pdf/png")


# =============================================================================
# FIGURE 2: Theorem Validation
# =============================================================================

def create_theorem_validation_figure():
    """Create theorem validation scatter plot."""

    fig, ax = plt.subplots(figsize=(6, 5))

    # Data points
    datasets = list(THEOREM_VALIDATION.keys())
    predicted = [THEOREM_VALIDATION[d]['predicted'] for d in datasets]
    observed = [THEOREM_VALIDATION[d]['observed'] for d in datasets]

    # Perfect prediction line
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction')

    # Scatter points
    colors_list = [COLORS['GP'], COLORS['Coverage']]
    for i, d in enumerate(datasets):
        ax.scatter(predicted[i], observed[i], s=200, c=colors_list[i],
                   edgecolor='black', linewidth=1.5, zorder=5)

        # Error annotation
        error = abs(predicted[i] - observed[i]) / observed[i] * 100
        offset = (0.02, 0.02) if i == 0 else (-0.08, 0.02)
        ax.annotate(f'{d}\n({error:.1f}% error)',
                    (predicted[i], observed[i]),
                    xytext=(predicted[i] + offset[0], observed[i] + offset[1]),
                    fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel('Predicted AUROC (Theorem)', fontweight='bold')
    ax.set_ylabel('Observed AUROC (Empirical)', fontweight='bold')
    ax.set_xlim(0.6, 0.9)
    ax.set_ylim(0.6, 0.9)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray')

    # Title
    ax.set_title('Coverage AUROC Theorem Validation', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_theorem_validation.pdf')
    plt.savefig(FIGURES_DIR / 'fig2_theorem_validation.png')
    plt.close()
    print(f"Saved: fig2_theorem_validation.pdf/png")


# =============================================================================
# FIGURE 3: Decomposition Conceptual Diagram
# =============================================================================

def create_decomposition_figure():
    """Create conceptual diagram showing semantic vs structural uncertainty."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: The limitation of each signal
    ax1 = axes[0]

    # 2x2 grid showing where each method succeeds/fails
    scenarios = [
        ['Known entity\nKnown relation', 'Known entity\nUnknown relation'],
        ['Unknown entity\nKnown relation', 'Unknown entity\nUnknown relation']
    ]

    # Create a table-like visualization
    table_data = [
        ['GP: Low ✓\nCov: Low ✓', 'GP: Low ✗\nCov: High ✓'],
        ['GP: High ✓\nCov: High ⚠', 'GP: High ✓\nCov: High ✓']
    ]

    colors_grid = [
        ['#C8E6C9', '#FFF9C4'],  # light green, light yellow
        ['#FFF9C4', '#C8E6C9']   # light yellow, light green
    ]

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1,
                                  facecolor=colors_grid[i][j],
                                  edgecolor='black', linewidth=2)
            ax1.add_patch(rect)

            # Scenario label (top)
            ax1.text(j + 0.5, 1.5 - i + 0.35, scenarios[i][j],
                     ha='center', va='center', fontsize=9, fontweight='bold')

            # Method results (bottom)
            ax1.text(j + 0.5, 1.5 - i - 0.1, table_data[i][j],
                     ha='center', va='center', fontsize=8)

    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('When Each Signal Succeeds/Fails', fontweight='bold', pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='black', label='Both correct'),
        mpatches.Patch(facecolor='#FFF9C4', edgecolor='black', label='One correct'),
    ]
    ax1.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    # Right: The synergy equation
    ax2 = axes[1]
    ax2.axis('off')

    # Main equation
    eq_text = r'$U_{\mathrm{CAGP}} = \alpha \cdot U_{\mathrm{GP}} + (1-\alpha) \cdot U_{\mathrm{Cov}}$'
    ax2.text(0.5, 0.7, eq_text, ha='center', va='center', fontsize=18,
             transform=ax2.transAxes)

    # Component explanations
    explanations = [
        (r'$U_{\mathrm{GP}} = \frac{1}{2}(\sigma^2_h + \sigma^2_t)$',
         'Semantic uncertainty\n(embedding quality)', COLORS['GP']),
        (r'$U_{\mathrm{Cov}} = 2 - c(h,r) - c(t,r)$',
         'Structural uncertainty\n(relation-specific observation)', COLORS['Coverage']),
    ]

    for i, (eq, desc, color) in enumerate(explanations):
        y_pos = 0.4 - i * 0.25

        # Colored box
        rect = mpatches.FancyBboxPatch((0.1, y_pos - 0.08), 0.35, 0.16,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=2,
                                        transform=ax2.transAxes)
        ax2.add_patch(rect)

        ax2.text(0.275, y_pos, eq, ha='center', va='center', fontsize=12,
                 transform=ax2.transAxes)

        ax2.text(0.65, y_pos, desc, ha='left', va='center', fontsize=10,
                 transform=ax2.transAxes)

    # Alpha annotation
    ax2.text(0.5, 0.05, r'Learned $\alpha \approx 0.5$ (equal contribution)',
             ha='center', va='center', fontsize=11, style='italic',
             transform=ax2.transAxes, color='gray')

    ax2.set_title('CAGP: Combining Both Signals', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_decomposition.pdf')
    plt.savefig(FIGURES_DIR / 'fig3_decomposition.png')
    plt.close()
    print(f"Saved: fig3_decomposition.pdf/png")


# =============================================================================
# FIGURE 4: Synergy Breakdown
# =============================================================================

def create_synergy_breakdown_figure():
    """Create figure showing synergy breakdown."""

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = list(RESULTS.keys())
    x = np.arange(len(datasets))

    # Calculate components
    gp_contrib = []
    cov_contrib = []
    synergy_contrib = []

    for d in datasets:
        best_single = max(RESULTS[d]['GP'], RESULTS[d]['Coverage'])
        worst_single = min(RESULTS[d]['GP'], RESULTS[d]['Coverage'])

        # Base (random = 0.5)
        base = 0.5

        # Attribution (simplified)
        gp_contrib.append(RESULTS[d]['GP'] - base)
        cov_contrib.append(max(0, RESULTS[d]['Coverage'] - RESULTS[d]['GP']))
        synergy_contrib.append(RESULTS[d]['CAGP'] - best_single)

    # Stacked bar
    width = 0.5

    ax.bar(x, [0.5]*3, width, label='Random baseline', color='lightgray', edgecolor='white')
    ax.bar(x, gp_contrib, width, bottom=0.5, label='GP contribution',
           color=COLORS['GP'], edgecolor='white')

    bottom2 = [0.5 + g for g in gp_contrib]
    ax.bar(x, cov_contrib, width, bottom=bottom2, label='Coverage contribution',
           color=COLORS['Coverage'], edgecolor='white')

    bottom3 = [b + c for b, c in zip(bottom2, cov_contrib)]
    ax.bar(x, synergy_contrib, width, bottom=bottom3, label='Synergy bonus',
           color=COLORS['synergy'], edgecolor='white', hatch='///')

    # Add CAGP total markers
    for i, d in enumerate(datasets):
        ax.plot(i, RESULTS[d]['CAGP'], 'k*', markersize=15, zorder=5)
        ax.text(i, RESULTS[d]['CAGP'] + 0.02, f"{RESULTS[d]['CAGP']:.3f}",
                ha='center', fontweight='bold', fontsize=10)

    # Formatting
    ax.set_ylabel('AUROC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')

    ax.set_title('AUROC Decomposition: Where Does Performance Come From?',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_synergy_breakdown.pdf')
    plt.savefig(FIGURES_DIR / 'fig4_synergy_breakdown.png')
    plt.close()
    print(f"Saved: fig4_synergy_breakdown.pdf/png")


# =============================================================================
# FIGURE 5: GP Limitation Illustration
# =============================================================================

def create_gp_limitation_figure():
    """Illustrate why GP variance is relation-agnostic."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Entity with different relation coverage
    ax1 = axes[0]

    # Simulated entity
    entity_name = "Entity E"
    relations = ['works_at', 'lives_in', 'born_in', 'friend_of']
    coverage = [1, 1, 0, 0]  # Seen with first two, not last two

    y_pos = np.arange(len(relations))
    colors_bars = [COLORS['CAGP'] if c else '#DDDDDD' for c in coverage]

    ax1.barh(y_pos, [1]*4, color=colors_bars, edgecolor='black', height=0.6)

    # Labels
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(relations, fontsize=11)
    ax1.set_xlim(0, 1.5)
    ax1.set_xticks([])

    # Coverage labels
    for i, c in enumerate(coverage):
        label = 'Seen ✓' if c else 'Unseen ✗'
        color = 'darkgreen' if c else 'darkred'
        ax1.text(1.1, i, label, va='center', fontsize=10, color=color, fontweight='bold')

    ax1.set_title(f'Coverage: Relation-Specific\n(for entity "{entity_name}")',
                  fontweight='bold', pad=10)
    ax1.set_xlabel('Observed in training?', fontsize=10)

    # Right: GP variance is constant
    ax2 = axes[1]

    gp_var = 0.35  # Single variance value

    ax2.barh(y_pos, [gp_var]*4, color=COLORS['GP'], edgecolor='black', height=0.6, alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(relations, fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['0', '0.5', '1.0'])

    # Same variance annotation
    ax2.axvline(x=gp_var, color='red', linestyle='--', linewidth=2)
    ax2.text(gp_var + 0.05, 3.3, f'σ² = {gp_var}\n(same for all!)',
             fontsize=10, color='red', fontweight='bold')

    ax2.set_title(f'GP Variance: Entity-Level Only\n(ignores relation)',
                  fontweight='bold', pad=10)
    ax2.set_xlabel('Learned variance σ²', fontsize=10)

    # Overall title
    fig.suptitle('Why GP Variance Misses Structural Uncertainty',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_gp_limitation.pdf')
    plt.savefig(FIGURES_DIR / 'fig5_gp_limitation.png')
    plt.close()
    print(f"Saved: fig5_gp_limitation.pdf/png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Generating NeurIPS figures...")
    print(f"Output directory: {FIGURES_DIR}")
    print("-" * 50)

    create_main_results_figure()
    create_theorem_validation_figure()
    create_decomposition_figure()
    create_synergy_breakdown_figure()
    create_gp_limitation_figure()

    print("-" * 50)
    print("Done! All figures saved to figures/")


if __name__ == '__main__':
    main()
