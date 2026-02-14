#!/usr/bin/env python3
"""
Alpha Sensitivity Sweep for CAGP.

Since alpha only affects get_uncertainty() (not training), we train CAGP once
per seed/dataset, then sweep alpha at eval time. This is nearly free compute.

Outputs: CSV with columns [dataset, alpha, overall_auroc, emerging_auroc, novel_ctx_auroc]
         + JSON with full details
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import csv
import time
from collections import defaultdict

from src.data.loaders import load_wn18rr, load_fb15k237, load_yago310
from scripts.run_wn18rr_temporal import (
    CAGP, train_model, setup_device, _is_emerging
)


def evaluate_temporal_with_alpha(model, train, test, n_ent, device, alpha_val):
    """Evaluate temporal OOD with a specific alpha override."""
    model.eval()

    # Compute normalization stats (same for all alpha values)
    if not hasattr(model, '_cached_norm') or model._cached_norm is None:
        with torch.no_grad():
            h = torch.tensor(train[:, 0]).to(device)
            r = torch.tensor(train[:, 1]).to(device)
            t = torch.tensor(train[:, 2]).to(device)
            h_var = torch.exp(model.entity_logvar[h]).mean(dim=-1)
            t_var = torch.exp(model.entity_logvar[t]).mean(dim=-1)
            gp_var = (h_var + t_var) / 2
            cov_unc = 2.0 - model.coverage[h, r] - model.coverage[t, r]
            model._cached_norm = {
                'gp_mean': gp_var.mean().item(),
                'cov_mean': cov_unc.mean().item(),
            }

    # Entity frequencies
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()

    # Categorize
    new_entity_idx, new_pair_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, 'leq'):
            new_entity_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) < 50 or len(id_idx) < 50:
        return None

    # Compute uncertainty with custom alpha
    gp_mean = model._cached_norm['gp_mean']
    cov_mean = model._cached_norm['cov_mean']

    def get_unc(indices):
        triples = test[indices]
        with torch.no_grad():
            h = torch.tensor(triples[:, 0]).to(device)
            r = torch.tensor(triples[:, 1]).to(device)
            t = torch.tensor(triples[:, 2]).to(device)
            h_var = torch.exp(model.entity_logvar[h]).mean(dim=-1)
            t_var = torch.exp(model.entity_logvar[t]).mean(dim=-1)
            gp_var = (h_var + t_var) / 2
            cov_unc = 2.0 - model.coverage[h, r] - model.coverage[t, r]
            gp_norm = gp_var / (gp_mean + 1e-8) * (cov_mean + 1e-8)
            return (alpha_val * gp_norm + (1 - alpha_val) * cov_unc).cpu().numpy()

    ood_unc = get_unc(ood_idx)
    id_unc = get_unc(id_idx)
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])
    overall_auroc = float(roc_auc_score(labels, scores))

    # Emerging vs ID
    e_unc = get_unc(new_entity_idx) if len(new_entity_idx) > 50 else None
    emerging_auroc = None
    if e_unc is not None and len(id_unc) > 50:
        l = np.concatenate([np.zeros(len(id_unc)), np.ones(len(e_unc))])
        s = np.concatenate([id_unc, e_unc])
        emerging_auroc = float(roc_auc_score(l, s))

    # Novel ctx vs ID
    n_unc = get_unc(new_pair_idx) if len(new_pair_idx) > 50 else None
    novel_auroc = None
    if n_unc is not None and len(id_unc) > 50:
        l = np.concatenate([np.zeros(len(id_unc)), np.ones(len(n_unc))])
        s = np.concatenate([id_unc, n_unc])
        novel_auroc = float(roc_auc_score(l, s))

    return {
        'overall_auroc': overall_auroc,
        'emerging_auroc': emerging_auroc,
        'novel_ctx_auroc': novel_auroc,
    }


def main():
    device = setup_device()
    print(f"Device: {device}")

    datasets = {
        'WN18RR': load_wn18rr,
        'FB15k-237': load_fb15k237,
        'YAGO3-10': load_yago310,
    }
    seeds = [42, 123, 456]
    alpha_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_results = {}
    csv_rows = []

    for ds_name, loader in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        train_ds, _, test_ds = loader()
        train = train_ds.triples
        test = test_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations
        print(f"Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}, Test: {len(test)}")

        ds_results = {}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Train ONCE
            t0 = time.time()
            model = CAGP(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30)
            model._cached_norm = None  # will be computed on first eval
            elapsed = time.time() - t0
            print(f"  Training: {elapsed:.1f}s")

            # Sweep alpha at eval time
            for alpha_val in alpha_values:
                result = evaluate_temporal_with_alpha(model, train, test, n_ent, device, alpha_val)
                if result:
                    key = f"seed_{seed}_alpha_{alpha_val}"
                    ds_results[key] = result
                    csv_rows.append([
                        ds_name, alpha_val,
                        result['overall_auroc'],
                        result.get('emerging_auroc', ''),
                        result.get('novel_ctx_auroc', ''),
                        seed
                    ])
                    if alpha_val in [0.0, 0.5, 1.0]:
                        print(f"  alpha={alpha_val:.1f}: overall={result['overall_auroc']:.4f}, "
                              f"emerging={result.get('emerging_auroc', 'N/A')}, "
                              f"novel={result.get('novel_ctx_auroc', 'N/A')}")

        all_results[ds_name] = ds_results

    # Compute mean±std across seeds for each alpha
    print("\n" + "=" * 60)
    print("SUMMARY: Mean ± Std over 3 seeds")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\n{ds_name}:")
        print(f"  {'alpha':>6}  {'Overall':>12}  {'Emerging':>12}  {'Novel Ctx':>12}")
        for alpha_val in alpha_values:
            overall_vals = []
            emerging_vals = []
            for seed in seeds:
                key = f"seed_{seed}_alpha_{alpha_val}"
                r = all_results.get(ds_name, {}).get(key)
                if r:
                    overall_vals.append(r['overall_auroc'])
                    if r.get('emerging_auroc') is not None:
                        emerging_vals.append(r['emerging_auroc'])
            if overall_vals:
                o_mean, o_std = np.mean(overall_vals), np.std(overall_vals)
                e_str = f"{np.mean(emerging_vals):.3f}±{np.std(emerging_vals):.3f}" if emerging_vals else "N/A"
                print(f"  {alpha_val:>6.2f}  {o_mean:.3f}±{o_std:.3f}  {e_str:>12}")

    # Save
    out_dir = project_root / 'outputs'
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / 'alpha_sensitivity_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    with open(out_dir / 'alpha_sensitivity_sweep.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'alpha', 'overall_auroc', 'emerging_auroc', 'novel_ctx_auroc', 'seed'])
        writer.writerows(csv_rows)

    print(f"\nResults saved to {out_dir}/alpha_sensitivity_sweep.{{json,csv}}")


if __name__ == '__main__':
    main()
