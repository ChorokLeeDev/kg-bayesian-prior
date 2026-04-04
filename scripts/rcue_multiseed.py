#!/usr/bin/env python3
"""
Multi-seed experiment for RCUE robustness.
3 seeds × 4 datasets = 12 runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_fb15k237, load_wn18rr, load_yago310, load_icews14
from src.models.relation_conditioned import RCUE, train_rcue


def run_single(name, load_fn, seed, epochs=20):
    """Run single experiment with given seed."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, valid_ds, test_ds = load_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Coverage from train
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])

    # Energy baseline
    from scripts.rcue_experiment import EnergyBaseline, train_baseline
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_baseline(energy, train, device, epochs=epochs)

    energy.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)
    with torch.no_grad():
        unc_energy = energy.get_uncertainty(h, r, t).cpu().numpy()
    auroc_energy = roc_auc_score(ood_mask, unc_energy)

    # RCUE
    torch.manual_seed(seed)
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=epochs, verbose=False)

    rcue.eval()
    with torch.no_grad():
        unc_rcue = rcue.get_uncertainty(h, r, t).cpu().numpy()
    auroc_rcue = roc_auc_score(ood_mask, unc_rcue)

    return auroc_energy, auroc_rcue


def main():
    seeds = [42, 123, 456]
    datasets = [
        ('FB15k-237', load_fb15k237, 20),
        ('WN18RR', load_wn18rr, 30),
        ('YAGO3-10', load_yago310, 15),
        ('ICEWS14', load_icews14, 30),
    ]

    results = {}

    for ds_name, load_fn, epochs in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        energy_scores = []
        rcue_scores = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            e, r = run_single(ds_name, load_fn, seed, epochs)
            energy_scores.append(e)
            rcue_scores.append(r)
            print(f"Energy={e:.4f}, RCUE={r:.4f}")

        results[ds_name] = {
            'energy_mean': np.mean(energy_scores),
            'energy_std': np.std(energy_scores),
            'rcue_mean': np.mean(rcue_scores),
            'rcue_std': np.std(rcue_scores),
        }

    # Summary table
    print("\n" + "="*70)
    print("MULTI-SEED RESULTS (mean ± std)")
    print("="*70)
    print(f"{'Dataset':<15} {'Energy':<20} {'RCUE':<20} {'Δ':<10}")
    print("-"*70)

    for ds_name, r in results.items():
        e_str = f"{r['energy_mean']:.3f}±{r['energy_std']:.3f}"
        r_str = f"{r['rcue_mean']:.3f}±{r['rcue_std']:.3f}"
        delta = r['rcue_mean'] - r['energy_mean']
        print(f"{ds_name:<15} {e_str:<20} {r_str:<20} +{delta:.3f}")


if __name__ == "__main__":
    main()
