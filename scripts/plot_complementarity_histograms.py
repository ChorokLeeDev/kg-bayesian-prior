#!/usr/bin/env python3
"""Generate Figure 2: Uncertainty distribution histograms by OOD type.

Shows why semantic uncertainty fails on novel contexts and structural
uncertainty fails on emerging entities, visualizing the complementarity.
Generates a 2x3 panel figure (2 signals × 3 OOD categories).
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

from src.data.loaders import load_fb15k237
from scripts.run_wn18rr_temporal import (
    CAGP, train_model, setup_device, _is_emerging
)


def compute_uncertainties(model, triples, device):
    """Compute semantic and structural uncertainty for each triple."""
    model.eval()
    with torch.no_grad():
        h = torch.tensor(triples[:, 0]).to(device)
        r = torch.tensor(triples[:, 1]).to(device)
        t = torch.tensor(triples[:, 2]).to(device)

        h_var = torch.exp(model.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(model.entity_logvar[t]).mean(dim=-1)
        u_sem = ((h_var + t_var) / 2).cpu().numpy()

        u_str = (2.0 - model.coverage[h, r] - model.coverage[t, r]).cpu().numpy()

    return u_sem, u_str


def main():
    device = setup_device()
    print(f"Device: {device}")

    # Use FB15k-237 (clearest complementarity pattern)
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Train model
    torch.manual_seed(42)
    np.random.seed(42)
    model = CAGP(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=30)

    # Categorize test triples
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()

    emerging_idx, novel_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, 'leq'):
            emerging_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            novel_idx.append(i)
        else:
            id_idx.append(i)

    print(f"Emerging: {len(emerging_idx)}, Novel: {len(novel_idx)}, ID: {len(id_idx)}")

    # Compute uncertainties
    u_sem, u_str = compute_uncertainties(model, test, device)

    # Normalize semantic for comparable scale
    sem_mean = u_sem.mean()
    str_mean = u_str.mean()
    u_sem_norm = u_sem / (sem_mean + 1e-8) * (str_mean + 1e-8)

    # Create figure: 2 rows (sem, str) × 3 columns (emerging, novel, all OOD)
    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharey='row')

    categories = [
        ('Emerging', emerging_idx, '#E74C3C'),
        ('Novel Ctx', novel_idx, '#3498DB'),
        ('All OOD', emerging_idx + novel_idx, '#9B59B6'),
    ]

    for col, (cat_name, cat_idx, color) in enumerate(categories):
        # Semantic uncertainty (top row)
        ax = axes[0, col]
        ax.hist(u_sem_norm[id_idx], bins=40, alpha=0.5, color='#95A5A6',
                label='ID', density=True, edgecolor='none')
        ax.hist(u_sem_norm[cat_idx], bins=40, alpha=0.6, color=color,
                label=cat_name, density=True, edgecolor='none')
        if col == 0:
            ax.set_ylabel(r'$U_{\mathrm{sem}}$', fontsize=10, fontweight='bold')
        ax.set_title(cat_name, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max(u_sem_norm[id_idx + cat_idx].max() * 1.1, 0.1))

        # Structural uncertainty (bottom row)
        ax = axes[1, col]
        # Structural is discrete {0, 1, 2} - use bar chart
        for val, label, c, alpha in [(0, 'ID', '#95A5A6', 0.5),
                                      (1, cat_name, color, 0.6)]:
            if val == 0:
                vals = u_str[id_idx]
                counts = np.array([np.sum(vals == v) for v in [0, 1, 2]])
                counts = counts / counts.sum()
                ax.bar([0 - 0.15, 1 - 0.15, 2 - 0.15], counts, width=0.3,
                       alpha=alpha, color=c, label='ID')
            else:
                vals = u_str[cat_idx]
                counts = np.array([np.sum(vals == v) for v in [0, 1, 2]])
                counts = counts / counts.sum()
                ax.bar([0 + 0.15, 1 + 0.15, 2 + 0.15], counts, width=0.3,
                       alpha=alpha, color=color, label=cat_name)
        if col == 0:
            ax.set_ylabel(r'$U_{\mathrm{str}}$', fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel('Uncertainty', fontsize=8)
        ax.tick_params(labelsize=7)

    # Add legends
    axes[0, 2].legend(fontsize=7, loc='upper right')
    axes[1, 2].legend(fontsize=7, loc='upper left')

    # Add AUROC annotations
    from sklearn.metrics import roc_auc_score
    for col, (cat_name, cat_idx, _) in enumerate(categories):
        labels = np.concatenate([np.zeros(len(id_idx)), np.ones(len(cat_idx))])
        # Semantic
        scores = np.concatenate([u_sem_norm[id_idx], u_sem_norm[cat_idx]])
        auroc_sem = roc_auc_score(labels, scores)
        axes[0, col].text(0.95, 0.95, f'AUC={auroc_sem:.2f}',
                         transform=axes[0, col].transAxes, fontsize=7,
                         ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        # Structural
        scores = np.concatenate([u_str[id_idx], u_str[cat_idx]])
        auroc_str = roc_auc_score(labels, scores)
        axes[1, col].text(0.95, 0.95, f'AUC={auroc_str:.2f}',
                         transform=axes[1, col].transAxes, fontsize=7,
                         ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout()

    out_path = project_root / 'paper' / 'figures' / 'fig2_complementarity.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path.with_suffix('.png')}")


if __name__ == '__main__':
    main()
