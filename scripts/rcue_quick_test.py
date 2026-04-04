#!/usr/bin/env python3
"""
Quick RCUE test on small subset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data.loaders import load_fb15k237, load_wn18rr
from src.models.relation_conditioned import RCUE, train_rcue

def test_dataset(name, load_fn):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")

    # Load data
    train_ds, _, test_ds = load_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Train: {len(train)}, Test: {len(test)}")

    # Build coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    # OOD mask
    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])
    print(f"OOD fraction: {ood_mask.mean()*100:.1f}%")

    # Train RCUE
    torch.manual_seed(42)
    model = RCUE(n_ent, n_rel, use_coverage=True)
    model = train_rcue(model, train, device, epochs=30, verbose=True)

    # Evaluate
    model.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(ood_mask, unc)

    print(f"\nRCUE OOD AUROC: {auroc:.4f}")
    print(f"Mean uncertainty (ID): {unc[~ood_mask].mean():.4f}")
    print(f"Mean uncertainty (OOD): {unc[ood_mask].mean():.4f}")

    return auroc

def main():
    results = {}
    results['FB15k-237'] = test_dataset('FB15k-237', load_fb15k237)
    results['WN18RR'] = test_dataset('WN18RR', load_wn18rr)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, auroc in results.items():
        print(f"{name}: {auroc:.4f}")

if __name__ == "__main__":
    main()
