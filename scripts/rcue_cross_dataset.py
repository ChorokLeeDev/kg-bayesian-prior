#!/usr/bin/env python3
"""
RCUE on YAGO3-10 and ICEWS14 for cross-dataset validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_yago310, load_icews14
from src.models.relation_conditioned import RCUE, train_rcue


def test_dataset(name, load_fn, epochs=30):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # Load
    train_ds, _, test_ds = load_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent:,}, Relations: {n_rel}")
    print(f"Train: {len(train):,}, Test: {len(test):,}")

    # Coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])
    print(f"OOD fraction: {ood_mask.mean()*100:.1f}%")

    results = {}

    # Energy baseline
    print("\n--- Energy Baseline ---")
    torch.manual_seed(42)
    from scripts.rcue_experiment import EnergyBaseline, train_baseline
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_baseline(energy, train, device, epochs=epochs)

    energy.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)
    with torch.no_grad():
        unc = energy.get_uncertainty(h, r, t).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    results['Energy'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # RCUE
    print("\n--- RCUE ---")
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=epochs, verbose=True)

    rcue.eval()
    with torch.no_grad():
        unc = rcue.get_uncertainty(h, r, t).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    results['RCUE'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # Stats
    print(f"\nMean unc (ID): {unc[~ood_mask].mean():.4f}")
    print(f"Mean unc (OOD): {unc[ood_mask].mean():.4f}")
    print(f"Ratio: {unc[ood_mask].mean() / unc[~ood_mask].mean():.2f}x")

    return results


def main():
    all_results = {}

    # YAGO3-10 (larger, 123K entities)
    all_results['YAGO3-10'] = test_dataset('YAGO3-10', load_yago310, epochs=20)

    # ICEWS14 (temporal, 7K entities)
    all_results['ICEWS14'] = test_dataset('ICEWS14', load_icews14, epochs=30)

    # Summary
    print("\n" + "="*60)
    print("CROSS-DATASET SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Energy':<10} {'RCUE':<10} {'Δ':<10}")
    print("-"*45)
    for ds, res in all_results.items():
        delta = res['RCUE'] - res['Energy']
        print(f"{ds:<15} {res['Energy']:.4f}     {res['RCUE']:.4f}     +{delta:.4f}")


if __name__ == "__main__":
    main()
