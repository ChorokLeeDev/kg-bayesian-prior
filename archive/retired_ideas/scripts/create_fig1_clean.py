"""
Create Figure 1 for UAI paper: Clean version with NO text overlaps
Simplified design focusing on key message
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Simplified data - just the key methods
methods = [
    'UKGE',
    'MC\nDropout',
    'Deep\nEnsemble',
    'SNGP',
    '',  # Spacer
    'Semantic\nonly',
    'Structural\nonly',
    '',  # Spacer
    'Simple\naverage',
    '',  # Spacer
    'CAGP',
]

auroc = [
    0.52, 0.56, 0.58, 0.61,  # Baselines
    np.nan,  # Spacer
    0.542, 0.935,  # Single signals
    np.nan,  # Spacer
    0.951,  # Simple
    np.nan,  # Spacer
    0.986,  # CAGP
]

# Simple color scheme
colors = [
    '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',  # Gray for baselines
    'white',
    '#87CEEB', '#90EE90',  # Light blue, light green
    'white',
    '#4682B4',  # Steel blue
    'white',
    '#2C5F7F',  # Dark blue for CAGP
]

# Create larger figure for more space
fig, ax = plt.subplots(figsize=(10, 4.5))

# Create bars with more space
x = np.arange(len(methods))
bars = ax.bar(x, auroc, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9, width=0.7)

# Emphasize CAGP
bars[-1].set_linewidth(2.5)
bars[-1].set_edgecolor('#2C5F7F')

# Random baseline line
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(5.5, 0.505, 'Random (0.5)', fontsize=9, color='red', va='bottom')

# Add value labels on bars - careful positioning
for i, (method, val) in enumerate(zip(methods, auroc)):
    if method and not np.isnan(val):
        fontweight = 'bold' if i == len(methods) - 1 else 'normal'
        fontsize = 11 if i == len(methods) - 1 else 10

        # Position label to avoid overlap
        if val > 0.95:  # Too high, put inside bar
            y_pos = val - 0.04
            va = 'top'
            color = 'white'
        else:
            y_pos = val + 0.025
            va = 'bottom'
            color = 'black'

        ax.text(i, y_pos, f'{val:.3f}',
                ha='center', va=va, fontsize=fontsize,
                fontweight=fontweight, color=color)

# Clean styling
ax.set_ylabel('AUROC', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0.4, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Simplified category labels - positioned BELOW x-axis
ax.text(1.5, -0.08, 'Probabilistic\nBaselines', ha='center', va='top',
        fontsize=9, style='italic', transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6',
                 edgecolor='gray', alpha=0.5, linewidth=0.5))

ax.text(5.5, -0.08, 'Single\nSignals', ha='center', va='top',
        fontsize=9, style='italic', transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6F2FF',
                 edgecolor='gray', alpha=0.5, linewidth=0.5))

ax.text(8, -0.08, 'Combined', ha='center', va='top',
        fontsize=9, style='italic', transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6',
                 edgecolor='gray', alpha=0.5, linewidth=0.5))

ax.text(10, -0.08, 'Ours', ha='center', va='top',
        fontsize=9, fontweight='bold', transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6',
                 edgecolor='darkgreen', alpha=0.7, linewidth=1.5))

# Clean title
ax.set_title('Temporal OOD Detection (FB15k-237): CAGP achieves 0.986 AUROC',
             fontsize=13, fontweight='bold', pad=15)

# Simple legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#CCCCCC', edgecolor='black', label='Baselines (0.52-0.61)'),
    Patch(facecolor='#87CEEB', edgecolor='black', label='Semantic signal (0.542)'),
    Patch(facecolor='#90EE90', edgecolor='black', label='Structural signal (0.935)'),
    Patch(facecolor='#4682B4', edgecolor='black', label='Simple avg (0.951)'),
    Patch(facecolor='#2C5F7F', edgecolor='#2C5F7F', linewidth=2.5, label='CAGP - Best (0.986)'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=10)

plt.tight_layout()

# Save
output_path = '/Users/i767700/Github/kg-bayesian-prior/paper/figures/fig1_main_results.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"✅ Clean figure saved to {output_path}")

png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
print(f"✅ PNG preview saved to {png_path}")
