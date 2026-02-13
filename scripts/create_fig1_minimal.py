"""
Create Figure 1 for UAI paper: FB15k-237 temporal OOD with error bars

Data sources:
  - UKGE, Energy, GPOnly, CoverageOnly, CAGP: outputs/canonical_temporal_results_v2.json
  - MC Dropout, Deep Ensemble, SNGP: outputs/fb15k237_missing_baselines.json
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Load canonical data
root = Path(__file__).parent.parent
with open(root / 'outputs' / 'canonical_temporal_results_v2.json') as f:
    canonical = json.load(f)
with open(root / 'outputs' / 'fb15k237_missing_baselines.json') as f:
    baselines = json.load(f)

fb = canonical['fb15k237']['summary']

# Compute missing baseline stats
def baseline_stats(method):
    vals = [baselines[f'seed_{s}'][method]['overall_auroc']
            for s in [42, 123, 456]]
    return np.mean(vals), np.std(vals)

mc_mean, mc_std = baseline_stats('MCDropout')
de_mean, de_std = baseline_stats('DeepEnsemble')
sngp_mean, sngp_std = baseline_stats('SNGP')

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
    fb['UKGE']['temporal_auroc_mean'],
    fb['Energy']['temporal_auroc_mean'],
    mc_mean,
    de_mean,
    sngp_mean,
    np.nan,
    fb['GPOnly']['temporal_auroc_mean'],
    fb['CoverageOnly']['temporal_auroc_mean'],
    np.nan,
    fb['CAGP']['temporal_auroc_mean'],
]

yerr = [
    fb['UKGE']['temporal_auroc_std'],
    fb['Energy']['temporal_auroc_std'],
    mc_std,
    de_std,
    sngp_std,
    0,
    fb['GPOnly']['temporal_auroc_std'],
    fb['CoverageOnly']['temporal_auroc_std'],
    0,
    fb['CAGP']['temporal_auroc_std'],
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

# Create bars with error bars
x = np.arange(len(methods))
bars = ax.bar(x, auroc, color=colors, edgecolor='black', linewidth=0.8, width=0.7,
              yerr=yerr, capsize=3, error_kw={'linewidth': 1.2, 'color': '#444444'})

# Emphasize CAGP
bars[-1].set_linewidth(2.5)

# Random baseline
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.5)

# Value labels on bars
for i, (method, val, err) in enumerate(zip(methods, auroc, yerr)):
    if method and not np.isnan(val):
        fontweight = 'bold' if i == len(methods) - 1 else 'normal'
        fontsize = 11 if i == len(methods) - 1 else 9

        # Smart positioning
        if val > 0.90:
            y_pos = val - 0.03
            va = 'top'
            color = 'white'
        else:
            y_pos = val + err + 0.02
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

# Legend
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
output = str(root / 'paper' / 'figures' / 'fig1_main_results.pdf')
plt.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.15)
print(f"Saved: {output}")

plt.savefig(output.replace('.pdf', '.png'), dpi=150, bbox_inches='tight', pad_inches=0.15)
print(f"Saved PNG version")
