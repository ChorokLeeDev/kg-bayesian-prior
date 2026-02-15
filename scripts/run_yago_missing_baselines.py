#!/usr/bin/env python3
"""
Run MC Dropout, Deep Ensemble, and SNGP baselines on YAGO3-10 temporal OOD.

Reuses baseline model classes and training functions from run_wn18rr_missing_baselines.py.
Saves results incrementally after each seed/method so partial results survive crashes.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json
import time

from src.data.loaders import load_yago310
from scripts.run_wn18rr_missing_baselines import (
    MCDropoutKGE,
    DeepEnsembleKGE,
    SNGPBaseline,
    train_model,
    train_ensemble,
    train_sngp,
    evaluate_temporal,
    setup_device,
)

OUTPUT_PATH = project_root / 'outputs' / 'yago_missing_baselines.json'


def save_results(results):
    """Save results incrementally so partial results survive crashes."""
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(f"{OUTPUT_PATH.suffix}.tmp")
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    tmp_path.replace(OUTPUT_PATH)
    print(f"  [saved to {OUTPUT_PATH}]")


def main():
    device = setup_device()
    print(f"Device: {device}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nLoading YAGO3-10...")
    t0 = time.time()
    train_ds, _, test_ds = load_yago310()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"YAGO3-10: {n_ent} entities, {n_rel} relations, {len(train)} train, {len(test)} test")
    print(f"Loaded in {time.time() - t0:.1f}s")

    seeds = [42, 123, 456]

    # Load existing partial results if any
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing partial results from {OUTPUT_PATH}")
    else:
        all_results = {}

    for seed in seeds:
        seed_key = f'seed_{seed}'
        if seed_key not in all_results:
            all_results[seed_key] = {}

        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # --- MC Dropout ---
        if 'MCDropout' not in all_results[seed_key]:
            print("\n  MC Dropout:")
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            model = MCDropoutKGE(n_ent, n_rel, dim=100, dropout_rate=0.1, num_samples=20)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30, lr=1e-3)
            temporal = evaluate_temporal(model, train, test, n_ent, device)
            elapsed = time.time() - t0
            print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
            print(f"    Emerging AUROC: {temporal.get('emerging_auroc', 'N/A')}")
            print(f"    Time: {elapsed:.1f}s")
            all_results[seed_key]['MCDropout'] = temporal
            save_results(all_results)
        else:
            print(f"\n  MC Dropout: already done (AUROC={all_results[seed_key]['MCDropout'].get('overall_auroc', '?')})")

        # --- Deep Ensemble (5 members) ---
        if 'DeepEnsemble' not in all_results[seed_key]:
            print("\n  Deep Ensemble (5 models):")
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            model = DeepEnsembleKGE(n_ent, n_rel, dim=100, num_models=5)
            model.precompute_coverage(train)
            model = train_ensemble(model, train, device, epochs=30, lr=1e-3)
            temporal = evaluate_temporal(model, train, test, n_ent, device)
            elapsed = time.time() - t0
            print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
            print(f"    Emerging AUROC: {temporal.get('emerging_auroc', 'N/A')}")
            print(f"    Time: {elapsed:.1f}s")
            all_results[seed_key]['DeepEnsemble'] = temporal
            save_results(all_results)
        else:
            print(f"\n  Deep Ensemble: already done (AUROC={all_results[seed_key]['DeepEnsemble'].get('overall_auroc', '?')})")

        # --- SNGP ---
        if 'SNGP' not in all_results[seed_key]:
            print("\n  SNGP:")
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            model = SNGPBaseline(n_ent, n_rel, dim=100, num_rff=512)
            model.precompute_coverage(train)
            model = train_sngp(model, train, device, epochs=30, lr=1e-3)
            temporal = evaluate_temporal(model, train, test, n_ent, device)
            elapsed = time.time() - t0
            print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
            print(f"    Emerging AUROC: {temporal.get('emerging_auroc', 'N/A')}")
            print(f"    Time: {elapsed:.1f}s")
            all_results[seed_key]['SNGP'] = temporal
            save_results(all_results)
        else:
            print(f"\n  SNGP: already done (AUROC={all_results[seed_key]['SNGP'].get('overall_auroc', '?')})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY -- YAGO3-10 (mean +/- std over 3 seeds)")
    print(f"{'='*60}")

    for method in ['MCDropout', 'DeepEnsemble', 'SNGP']:
        aurocs = []
        emerging = []
        for seed in seeds:
            r = all_results.get(f'seed_{seed}', {}).get(method, {})
            if 'overall_auroc' in r:
                aurocs.append(r['overall_auroc'])
            if 'emerging_auroc' in r:
                emerging.append(r['emerging_auroc'])
        if aurocs:
            em_str = f"  emerging={np.mean(emerging):.3f}+/-{np.std(emerging):.3f}" if emerging else ""
            print(f"  {method:15s}  overall={np.mean(aurocs):.3f}+/-{np.std(aurocs):.3f}{em_str}")

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
