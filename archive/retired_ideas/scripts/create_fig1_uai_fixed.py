"""
Create Figure 1 for UAI paper: Main Results (CAGP-only version)
Updated after Plan B: Focus on CAGP, remove RelCondVar
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Data from paper results (FB15k-237 temporal OOD - Table 1)
methods = [
    # Probabilistic Baselines
    'UKGE',
    'MC\nDropout',
    'Deep\nEnsemble',
    'SNGP',
    '',  # Spacer
    # Single Signals
    'Semantic\n(freq)',
    'Structural\n(coverage)',
    '',  # Spacer
    # Simple Combination
    'Simple avg\n(α=0.5)',
    '',  # Spacer
    # Learned (Ours)
    'CAGP\n(learned α)',
]

auroc = [
    # Probabilistic baselines (from Table 1)
    0.52,   # UKGE
    0.56,   # MC Dropout
    0.58,   # Deep Ensemble
    0.61,   # SNGP
    np.nan,  # Spacer
    # Single signals (from Table 1)
    0.542,  # Semantic (U_sem)
    0.935,  # Structural (U_str)
    np.nan,  # Spacer
    # Simple combination (from Table 1)
    0.951,  # Simple average
    np.nan,  # Spacer
    # Learned (ours) - from Table 1
    0.986,  # CAGP
]

# Colors to emphasize progression
colors = [
    # Probabilistic baselines (gray - they fail)
    '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',
    'white',  # Spacer
    # Single signals (contrasting colors)
    '#B8D4E3',  # Light blue for semantic
    '#7BC47D',  # Green for structural (dominant)
    'white',  # Spacer
    # Simple combination (medium blue)
    '#5A9BD5',
    'white',  # Spacer
    # Learned CAGP (dark blue, emphasis)
    '#2C5F7F',  # Darkest for best
]

# Create figure with more space
fig, ax = plt.subplots(figsize=(9, 4))

# Create bars
x = np.arange(len(methods))
bars = ax.bar(x, auroc, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)

# Emphasize CAGP with thicker border
bars[-1].set_linewidth(2.5)
bars[-1].set_edgecolor('#2C5F7F')

# Add horizontal line at random performance
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random')

# Add value labels on bars (skip spacers) - reduced font size to avoid overlap
for i, (method, val) in enumerate(zip(methods, auroc)):
    if method and not np.isnan(val):
        # Emphasize CAGP value
        fontweight = 'bold' if i == len(methods) - 1 else 'normal'
        fontsize = 10 if i == len(methods) - 1 else 9
        y_offset = 0.03 if val < 0.95 else -0.05  # Put label below if too high
        va = 'bottom' if val < 0.95 else 'top'
        ax.text(i, val + y_offset, f'{val:.3f}',
                ha='center', va=va, fontsize=fontsize, fontweight=fontweight)

# Styling
ax.set_ylabel('AUROC on Temporal OOD', fontweight='bold', fontsize=12)
ax.set_xlabel('Method', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0, ha='center')
ax.set_ylim(0.45, 1.02)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)

# Add category labels - adjusted positions to avoid overlap
def add_category_label(ax, x_start, x_end, label, y_pos=-0.12, color='lightgray'):
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
            ha='center', va='center', fontsize=8, fontweight='bold',
            transform=ax.get_xaxis_transform())

# Add category labels with reduced text
add_category_label(ax, 0, 3, 'Probabilistic Baselines', color='#FFE6E6')
add_category_label(ax, 5, 6, 'Single Signals', color='#E6F2FF')
add_category_label(ax, 8, 8, 'Simple\nCombination', color='#FFF4E6')
add_category_label(ax, 10, 10, 'Learned\n(Ours)', color='#E6FFE6')

# Add simplified annotations - fewer to avoid overlap
# Annotation 1: Baselines fail
ax.annotate('Near-random\n(Theorem 1)',
            xy=(2, 0.58), xytext=(2, 0.35),
            fontsize=8, ha='center', style='italic',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.7))

# Annotation 2: Coverage is key
ax.annotate('Structural signal\ndominates',
            xy=(6, 0.935), xytext=(7.2, 0.85),
            fontsize=8, ha='center', style='italic',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2, alpha=0.7))

# Annotation 3: CAGP best
ax.annotate('Best: 0.986\n(60-80% improvement)',
            xy=(10, 0.986), xytext=(8.5, 0.99),
            fontsize=9, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2C5F7F', lw=1.5))

# Legend - simplified
legend_elements = [
    mpatches.Patch(color='#CCCCCC', label='Probabilistic baselines (0.52-0.61)'),
    mpatches.Patch(color='#B8D4E3', label='Semantic only (0.542)'),
    mpatches.Patch(color='#7BC47D', label='Structural only (0.935)'),
    mpatches.Patch(color='#5A9BD5', label='Simple avg α=0.5 (0.951)'),
    mpatches.Patch(color='#2C5F7F', edgecolor='#2C5F7F', linewidth=2.5,
                   label='CAGP learned α (0.986)'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
         edgecolor='black', ncol=1, fontsize=9)

# Title - more concise
fig.suptitle('Temporal OOD Detection on FB15k-237: CAGP achieves 0.986 AUROC',
             fontsize=12, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = '/Users/i767700/Github/kg-bayesian-prior/paper/figures/fig1_main_results.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
print(f"✅ Figure saved to {output_path}")

# Also save PNG for easy viewing
png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.15)
print(f"✅ PNG preview saved to {png_path}")

# plt.show()  # Comment out to avoid blocking
