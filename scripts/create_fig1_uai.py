"""
Create Figure 1 for UAI paper: Main Results emphasizing RelCondVar > CAGP > baselines
Updated to reflect revised paper structure (December 2025)
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# Data from paper results (ICEWS14 temporal OOD)
# Organized by method category to emphasize progression

methods = [
    # Probabilistic Baselines (fail on temporal)
    'UKGE',
    'Energy',
    'Deep\nEnsemble',
    'SNGP',
    '',  # Spacer
    # Single Signals
    'Freq-only\n(U_sem)',
    'Coverage-only\n(U_str)',
    '',  # Spacer
    # Simple Combination
    'Simple Avg\n(α=0.5)',
    '',  # Spacer
    # Learned (Ours) - emphasize these
    'CAGP\n(learned α)',
    'RelCondVar\n(learned σ²(e,r))',
]

auroc = [
    # Probabilistic baselines
    0.523,  # UKGE
    0.541,  # Energy
    0.578,  # Deep Ensemble
    0.614,  # SNGP
    np.nan,  # Spacer
    # Single signals
    0.687,  # Frequency-only
    0.824,  # Coverage-only
    np.nan,  # Spacer
    # Simple combination
    0.868,  # Simple average
    np.nan,  # Spacer
    # Learned (ours)
    0.891,  # CAGP
    0.912,  # RelCondVar
]

# Colors to emphasize progression
colors = [
    # Probabilistic baselines (gray - they fail)
    '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',
    'white',  # Spacer
    # Single signals (light colors)
    '#B8D4E3',  # Light blue for semantic
    '#C9E4CA',  # Light green for structural
    'white',  # Spacer
    # Simple combination (medium)
    '#7FA8C9',  # Medium blue
    'white',  # Spacer
    # Learned combinations (dark, emphasis)
    '#3A7CA5',  # Dark blue for CAGP
    '#2C5F7F',  # Darkest blue for RelCondVar (best)
]

# Create figure
fig, ax = plt.subplots(figsize=(8, 3.5))

# Create bars
x = np.arange(len(methods))
bars = ax.bar(x, auroc, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)

# Emphasize RelCondVar with thicker border
bars[-1].set_linewidth(2.0)
bars[-1].set_edgecolor('#2C5F7F')

# Add horizontal line at random performance
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (0.5)')

# Add value labels on bars (skip spacers)
for i, (method, val) in enumerate(zip(methods, auroc)):
    if method and not np.isnan(val):
        # Emphasize RelCondVar and CAGP values
        fontweight = 'bold' if i >= len(methods) - 2 else 'normal'
        fontsize = 9 if i >= len(methods) - 2 else 8
        ax.text(i, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=fontsize, fontweight=fontweight)

# Styling
ax.set_ylabel('AUROC on Temporal OOD (ICEWS14)', fontweight='bold')
ax.set_xlabel('Method Category', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0, ha='center')
ax.set_ylim(0.45, 0.95)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)

# Add category labels with background boxes
def add_category_label(ax, x_start, x_end, label, y_pos=-0.15, color='lightgray'):
    """Add a category label spanning multiple bars"""
    width = x_end - x_start
    rect = mpatches.FancyBboxPatch(
        (x_start - 0.4, y_pos), width + 0.8, 0.05,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='black', linewidth=0.5,
        transform=ax.get_xaxis_transform(), zorder=0, alpha=0.3
    )
    ax.add_patch(rect)
    ax.text((x_start + x_end) / 2, y_pos + 0.025, label,
            ha='center', va='center', fontsize=7, fontweight='bold',
            transform=ax.get_xaxis_transform())

# Add category labels
add_category_label(ax, 0, 3, 'Probabilistic Baselines\n(Relation-Agnostic)', color='#FFE6E6')
add_category_label(ax, 5, 6, 'Single Signals', color='#E6F2FF')
add_category_label(ax, 8, 8, 'Simple\nDecomp', color='#FFF4E6')
add_category_label(ax, 10, 11, 'Learned (Ours)', color='#E6FFE6')

# Add annotations for key insights
# Annotation 1: Baselines fail
ax.annotate('Relation-agnostic\nmethods fail\n(Theorem 1)',
            xy=(1.5, 0.56), xytext=(1.5, 0.35),
            fontsize=7, ha='center', style='italic',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

# Annotation 2: Coverage dominates
ax.annotate('Coverage is\ndominant signal',
            xy=(6, 0.824), xytext=(7.5, 0.78),
            fontsize=7, ha='center', style='italic',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))

# Annotation 3: Decomposition works
ax.annotate('Decomposition\nframework\neffective',
            xy=(8, 0.868), xytext=(9.2, 0.82),
            fontsize=7, ha='center', style='italic',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))

# Annotation 4: Best result
ax.annotate('Best:\nLearned\nrelation-specific\nvariance',
            xy=(11, 0.912), xytext=(11, 0.95),
            fontsize=7, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2C5F7F',
                     edgecolor='black', alpha=0.2))

# Legend
legend_elements = [
    mpatches.Patch(color='#CCCCCC', label='Probabilistic baselines (fail)'),
    mpatches.Patch(color='#B8D4E3', label='Frequency-only (semantic)'),
    mpatches.Patch(color='#C9E4CA', label='Coverage-only (structural)'),
    mpatches.Patch(color='#7FA8C9', label='Simple decomposition'),
    mpatches.Patch(color='#3A7CA5', label='CAGP (learned α)'),
    mpatches.Patch(color='#2C5F7F', edgecolor='#2C5F7F', linewidth=2,
                   label='RelCondVar (learned σ²(e,r)) - Best'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
         edgecolor='black', ncol=2)

# Title
fig.suptitle('Temporal OOD Detection on ICEWS14: Progression from Baselines to Learned Solutions',
             fontsize=11, fontweight='bold', y=0.98)

plt.tight_layout()

# Save figure
output_path = '/Users/i767700/Github/kg-bayesian-prior/paper/figures/fig1_main_results.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"✅ Figure saved to {output_path}")

# Also save PNG for easy viewing
png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
print(f"✅ PNG preview saved to {png_path}")

plt.show()
