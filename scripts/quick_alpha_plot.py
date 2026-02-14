#!/usr/bin/env python3
"""Alpha sensitivity plot from pre-computed CSV sweep data.
Reads alpha_sensitivity_sweep.csv and plots all 3 datasets (WN18RR, FB15k-237, YAGO3-10).
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required")
    sys.exit(1)


def main():
    csv_path = project_root / 'outputs' / 'alpha_sensitivity_sweep.csv'
    df = pd.read_csv(csv_path)

    # Compute mean and std across seeds for each (dataset, alpha)
    grouped = df.groupby(['dataset', 'alpha'])['overall_auroc'].agg(['mean', 'std']).reset_index()

    # Plot settings
    colors = {'WN18RR': '#2196F3', 'FB15k-237': '#FF5722', 'YAGO3-10': '#4CAF50'}
    markers = {'WN18RR': 'o', 'FB15k-237': 's', 'YAGO3-10': '^'}
    plot_order = ['WN18RR', 'FB15k-237', 'YAGO3-10']

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for ds_name in plot_order:
        ds = grouped[grouped['dataset'] == ds_name].sort_values('alpha')
        alphas = ds['alpha'].values
        means = ds['mean'].values
        stds = ds['std'].values
        ax.plot(alphas, means, '-' + markers[ds_name], color=colors[ds_name],
                label=ds_name, markersize=4, linewidth=1.5)
        ax.fill_between(alphas,
                        means - stds,
                        means + stds,
                        alpha=0.15, color=colors[ds_name])

    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.annotate(r'$\alpha{=}0.5$', xy=(0.52, 0.88), fontsize=8, color='gray')

    ax.set_xlabel(r'Mixing weight $\alpha$', fontsize=10)
    ax.set_ylabel('AUROC (temporal OOD)', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9, loc='lower center')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    ax.text(0.02, 0.42, r'$U_{\mathrm{str}}$ only', fontsize=7, color='gray', ha='left')
    ax.text(0.98, 0.42, r'$U_{\mathrm{sem}}$ only', fontsize=7, color='gray', ha='right')

    plt.tight_layout()
    out_path = project_root / 'paper' / 'figures' / 'fig3_alpha_sensitivity.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
