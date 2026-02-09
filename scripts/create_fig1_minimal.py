"""
Create Figure 1 for UAI paper: Minimal clean version (FB15k-237)
NO annotations, NO overlapping text - just clean bars

Data sources:
  - UKGE, Energy, GPOnly, CoverageOnly, CAGP: outputs/canonical_temporal_results_v2.json
  - MC Dropout, Deep Ensemble, SNGP: outputs/fb15k237_missing_baselines.json
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Data from Table 1 (FB15k-237, canonical 3-seed means)
methods = [
    'UKGE',
    'Energy',
    'MC\nDropout',
    'Deep\nEnsemble',
    'SNGP',
    '',  # Spacer
    'Semantic\n($U_{sem}$)',
    'Structural\n($U_{str}$)',
    '',  # Spacer
    'CAGP\n(ours)',
]

auroc = [
    0.42, 0.51, 0.63, 0.61, 0.46,  # Baselines
    np.nan,
    0.59, 0.94,  # Single signals
    np.nan,
    0.94,  # CAGP
]

# Simple colors
colors = [
    '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',  # Gray
    'white',
    '#87CEEB', '#90EE90',  # Blue, green
    'white',
    '#2C5F7F',  # Dark blue
]

# Create figure
fig, ax = plt.subplots(figsize=(10, 4))

# Create bars
x = np.arange(len(methods))
bars = ax.bar(x, auroc, color=colors, edgecolor='black', linewidth=0.8, width=0.7)

# Emphasize CAGP
bars[-1].set_linewidth(2.5)

# Random baseline
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.5)

# Value labels on bars ONLY
for i, (method, val) in enumerate(zip(methods, auroc)):
    if method and not np.isnan(val):
        fontweight = 'bold' if i == len(methods) - 1 else 'normal'
        fontsize = 11 if i == len(methods) - 1 else 9

        # Smart positioning
        if val > 0.96:
            y_pos = val - 0.03
            va = 'top'
            color = 'white'
        else:
            y_pos = val + 0.02
            va = 'bottom'
            color = 'black'

        ax.text(i, y_pos, f'{val:.2f}', ha='center', va=va,
                fontsize=fontsize, fontweight=fontweight, color=color)

# Clean axes
ax.set_ylabel('AUROC', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0.3, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Simple title
ax.set_title('Temporal OOD Detection on FB15k-237',
             fontsize=13, fontweight='bold', pad=15)

# Legend only
from matplotlib.patches import Patch
legend = [
    Patch(facecolor='#CCCCCC', label='Probabilistic baselines'),
    Patch(facecolor='#87CEEB', label='Semantic (entity variance)'),
    Patch(facecolor='#90EE90', label='Structural (coverage)'),
    Patch(facecolor='#2C5F7F', linewidth=2.5, label='CAGP (ours)'),
]
ax.legend(handles=legend, loc='upper left', framealpha=0.95)

plt.tight_layout()

# Save
output = '/Users/i767700/Github/kg-bayesian-prior/paper/figures/fig1_main_results.pdf'
plt.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.15)
print(f"Saved: {output}")

plt.savefig(output.replace('.pdf', '.png'), dpi=150, bbox_inches='tight', pad_inches=0.15)
print(f"Saved PNG version")
