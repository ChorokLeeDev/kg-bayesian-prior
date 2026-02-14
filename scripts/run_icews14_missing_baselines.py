#!/usr/bin/env python3
"""
Run MC Dropout, Deep Ensemble, and SNGP baselines on ICEWS14 temporal OOD.

ICEWS14 uses ground-truth timestamps (not simulated coverage splits).
Reuses model classes from run_wn18rr_missing_baselines.py.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import time
from collections import defaultdict

from src.data.loaders import load_icews14
from scripts.run_wn18rr_missing_baselines import (
    MCDropoutKGE, DeepEnsembleKGE, SNGPBaseline,
    setup_device, train_model, train_ensemble, train_sngp
)
from scripts.run_wn18rr_temporal import evaluate_temporal


def main():
    device = setup_device()
    print(f"Device: {device}")

    train_ds, _, test_ds = load_icews14()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"ICEWS14: {n_ent} entities, {n_rel} relations, {len(train)} train, {len(test)} test")

    seeds = [42, 123, 456]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_results = {}

        # --- MC Dropout ---
        print("\n  MC Dropout:")
        t0 = time.time()
        model = MCDropoutKGE(n_ent, n_rel, dropout_rate=0.1, num_samples=20)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        if 'emerging_auroc' in temporal:
            print(f"    Emerging AUROC: {temporal['emerging_auroc']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['MCDropout'] = temporal

        # --- Deep Ensemble ---
        print("\n  Deep Ensemble (5 models):")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepEnsembleKGE(n_ent, n_rel, num_models=5)
        model.precompute_coverage(train)
        model = train_ensemble(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        if 'emerging_auroc' in temporal:
            print(f"    Emerging AUROC: {temporal['emerging_auroc']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['DeepEnsemble'] = temporal

        # --- SNGP ---
        print("\n  SNGP:")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SNGPBaseline(n_ent, n_rel, num_rff=512)
        model.precompute_coverage(train)
        model = train_sngp(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        if 'emerging_auroc' in temporal:
            print(f"    Emerging AUROC: {temporal['emerging_auroc']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['SNGP'] = temporal

        all_results[f'seed_{seed}'] = seed_results

    # Summary
    print("\n" + "=" * 60)
    print("ICEWS14 SUMMARY (mean ± std over 3 seeds)")
    print("=" * 60)

    for method in ['MCDropout', 'DeepEnsemble', 'SNGP']:
        aurocs = []
        emerging_aurocs = []
        for seed in seeds:
            r = all_results[f'seed_{seed}'][method]
            if 'overall_auroc' in r:
                aurocs.append(r['overall_auroc'])
            if 'emerging_auroc' in r:
                emerging_aurocs.append(r['emerging_auroc'])
        if aurocs:
            print(f"  {method}:")
            print(f"    Overall: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
            if emerging_aurocs:
                print(f"    Emerging: {np.mean(emerging_aurocs):.3f} ± {np.std(emerging_aurocs):.3f}")

    # Save
    out = project_root / 'outputs' / 'icews14_missing_baselines.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
