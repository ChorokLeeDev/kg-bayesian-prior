#!/usr/bin/env python3
"""
Create figures for EMNLP paper.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
output_dir.mkdir(exist_ok=True)

def fig1_main_results():
    """Main results bar chart."""
    datasets = ['WN18RR', 'FB15k-237', 'YAGO3-10']

    # Data from experiments
    gp_only = [0.647, 0.749, 0.824]
    coverage = [0.657, 0.821, 0.760]
    cagp = [0.871, 0.960, 0.942]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))

    bars1 = ax.bar(x - width, gp_only, width, label='GP-only', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, coverage, width, label='Coverage-only', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, cagp, width, label='CAGP (Ours)', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('AUROC')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='upper left')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add synergy labels
    for i, (g, c, ca) in enumerate(zip(gp_only, coverage, cagp)):
        best_single = max(g, c)
        synergy = (ca - best_single) / best_single * 100
        ax.annotate(f'+{synergy:.0f}%',
                   xy=(i + width, ca + 0.01),
                   ha='center', fontsize=8, color='#2ecc71')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_main_results.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig1_main_results.pdf")

def fig2_complementarity():
    """Complementarity breakdown pie charts."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    datasets = ['WN18RR', 'FB15k-237', 'YAGO3-10']
    data = [
        [26.2, 15.3, 23.0, 35.5],  # WN18RR
        [45.3, 3.1, 42.2, 9.4],    # FB15k-237
        [37.4, 6.8, 25.0, 30.8],   # YAGO3-10
    ]

    labels = ['Both', 'GP only', 'Cov only', 'Neither']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    for ax, ds, d in zip(axes, datasets, data):
        wedges, texts, autotexts = ax.pie(d, labels=labels, autopct='%1.0f%%',
                                          colors=colors, startangle=90)
        ax.set_title(ds)
        for autotext in autotexts:
            autotext.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_complementarity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig2_complementarity.pdf")

def fig3_temporal_ood():
    """Temporal OOD comparison."""
    categories = ['New Entity', 'New Pair', 'Overall']
    gp = [0.826, 0.421, 0.542]
    cov = [0.784, 1.000, 0.935]
    cagp = [0.923, 0.979, 0.965]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(x - width, gp, width, label='GP-only', color='#3498db', alpha=0.8)
    ax.bar(x, cov, width, label='Coverage-only', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, cagp, width, label='CAGP', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_temporal_ood.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig3_temporal_ood.pdf")

def fig4_adversarial():
    """Adversarial OOD comparison."""
    corruptions = ['Random', 'Type-const.', 'Embed-sim.', 'Rel-plaus.']
    ukge = [0.992, 0.721, 0.412, 0.089]
    cagp = [0.960, 0.708, 0.657, 0.548]
    relcond = [0.968, 0.745, 0.692, 0.651]

    x = np.arange(len(corruptions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - width, ukge, width, label='UKGE', color='#9b59b6', alpha=0.8)
    ax.bar(x, cagp, width, label='CAGP', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, relcond, width, label='RelCondVar', color='#f39c12', alpha=0.8)

    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_adversarial.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig4_adversarial.pdf")

def fig5_gp_limitation():
    """GP limitation illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: Coverage is relation-specific
    ax = axes[0]
    relations = ['born_in', 'works_at', 'lives_in']
    coverage = [1, 0, 1]
    colors = ['#2ecc71' if c else '#e74c3c' for c in coverage]
    ax.barh(relations, [1, 1, 1], color=colors, alpha=0.7)
    ax.set_xlabel('Coverage')
    ax.set_title('Coverage: Relation-Specific')
    ax.set_xlim(0, 1.2)
    for i, (r, c) in enumerate(zip(relations, coverage)):
        ax.text(1.05, i, 'Seen' if c else 'Unseen', va='center', fontsize=9)

    # Right: GP variance is constant
    ax = axes[1]
    ax.barh(relations, [0.15, 0.15, 0.15], color='#3498db', alpha=0.7)
    ax.set_xlabel('GP Variance (σ²)')
    ax.set_title('GP Variance: Relation-Agnostic')
    ax.set_xlim(0, 0.3)
    ax.text(0.17, 1, 'Same for all\nrelations', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_gp_limitation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig5_gp_limitation.pdf")

if __name__ == '__main__':
    fig1_main_results()
    fig2_complementarity()
    fig3_temporal_ood()
    fig4_adversarial()
    fig5_gp_limitation()
    print(f"\nAll figures saved to {output_dir}")
