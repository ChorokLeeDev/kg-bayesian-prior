#!/usr/bin/env python3
"""Generate alpha sensitivity plot for the paper (Figure 2).

Reads alpha_sensitivity_sweep.csv and produces a PDF figure showing
AUROC vs alpha for each dataset, with shaded std bands.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import csv
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

def main():
    csv_path = project_root / 'outputs' / 'alpha_sensitivity_sweep.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run alpha_sensitivity_sweep.py first.")
        sys.exit(1)

    # Parse CSV
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            alpha = float(row['alpha'])
            data[ds][alpha].append(float(row['overall_auroc']))

    # Compute mean/std
    datasets = ['WN18RR', 'FB15k-237', 'YAGO3-10']
    colors = {'WN18RR': '#2196F3', 'FB15k-237': '#FF5722', 'YAGO3-10': '#4CAF50'}
    markers = {'WN18RR': 'o', 'FB15k-237': 's', 'YAGO3-10': '^'}

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for ds in datasets:
        if ds not in data:
            continue
        alphas = sorted(data[ds].keys())
        means = [np.mean(data[ds][a]) for a in alphas]
        stds = [np.std(data[ds][a]) for a in alphas]

        ax.plot(alphas, means, '-' + markers[ds], color=colors[ds],
                label=ds, markersize=5, linewidth=1.5)
        ax.fill_between(alphas,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=colors[ds])

    # Mark alpha=0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.annotate(r'$\alpha{=}0.5$', xy=(0.52, 0.88), fontsize=8, color='gray')

    ax.set_xlabel(r'Mixing weight $\alpha$', fontsize=10)
    ax.set_ylabel('AUROC (temporal OOD)', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8, loc='lower center')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Annotations for extremes
    ax.text(0.02, 0.42, r'$U_{\mathrm{str}}$ only', fontsize=7, color='gray', ha='left')
    ax.text(0.98, 0.42, r'$U_{\mathrm{sem}}$ only', fontsize=7, color='gray', ha='right')

    plt.tight_layout()
    out_path = project_root / 'paper' / 'figures' / 'fig2_alpha_sensitivity.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Also save PNG for quick preview
    png_path = out_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
