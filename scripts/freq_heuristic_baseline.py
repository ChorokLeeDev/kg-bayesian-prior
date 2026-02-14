#!/usr/bin/env python3
"""
Frequency heuristic baseline for temporal OOD detection.

Shows that:
- Binary freq heuristic (freq(e,r) > 0?) = U_str exactly
- Continuous log-freq heuristic performs near-random (~0.56)

This validates the paper's claim: binary presence/absence is the key signal,
not continuous frequency information.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import json

from scripts.run_wn18rr_temporal import (
    CoverageOnly, train_model, setup_device, _is_emerging
)
from src.data.loaders import load_fb15k237, load_wn18rr


def compute_freq_heuristic_uncertainties(train, test, n_ent, n_rel):
    """Compute binary and continuous frequency heuristics."""
    # Build coverage and frequency matrices
    coverage = np.zeros((n_ent, n_rel))
    freq_matrix = np.zeros((n_ent, n_rel))

    for i in range(len(train)):
        h, r, t = train[i]
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0
        freq_matrix[h, r] += 1.0
        freq_matrix[t, r] += 1.0

    # Binary heuristic: 2 - I(freq(h,r)>0) - I(freq(t,r)>0)
    binary_unc = []
    # Continuous log-freq: 2 - log(1+freq(h,r))/Z - log(1+freq(t,r))/Z
    log_freq = np.log1p(freq_matrix)
    Z = log_freq.max() + 1e-8
    continuous_unc = []

    for i in range(len(test)):
        h, r, t = test[i]
        # Binary
        b = 2.0 - coverage[h, r] - coverage[t, r]
        binary_unc.append(b)
        # Continuous
        c = 2.0 - log_freq[h, r] / Z - log_freq[t, r] / Z
        continuous_unc.append(c)

    return np.array(binary_unc), np.array(continuous_unc)


def evaluate_temporal_heuristic(binary_unc, continuous_unc, train, test, n_ent, n_rel):
    """Evaluate both heuristics on temporal OOD."""
    # Build coverage for categorization
    coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train)):
        h, r, t = train[i]
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0

    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)

    emerging_idx, novel_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, 'leq'):
            emerging_idx.append(i)
        elif coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_idx.append(i)
        else:
            id_idx.append(i)

    ood_idx = emerging_idx + novel_idx
    results = {}

    for name, unc in [('binary', binary_unc), ('log_freq', continuous_unc)]:
        if len(ood_idx) > 50 and len(id_idx) > 50:
            labels = np.concatenate([np.zeros(len(id_idx)), np.ones(len(ood_idx))])
            scores = np.concatenate([unc[id_idx], unc[ood_idx]])
            results[f'{name}_overall'] = float(roc_auc_score(labels, scores))

        if len(emerging_idx) > 50 and len(id_idx) > 50:
            labels = np.concatenate([np.zeros(len(id_idx)), np.ones(len(emerging_idx))])
            scores = np.concatenate([unc[id_idx], unc[emerging_idx]])
            results[f'{name}_emerging'] = float(roc_auc_score(labels, scores))

        if len(novel_idx) > 50 and len(id_idx) > 50:
            labels = np.concatenate([np.zeros(len(id_idx)), np.ones(len(novel_idx))])
            scores = np.concatenate([unc[id_idx], unc[novel_idx]])
            results[f'{name}_novel'] = float(roc_auc_score(labels, scores))

    results['n_emerging'] = len(emerging_idx)
    results['n_novel'] = len(novel_idx)
    results['n_id'] = len(id_idx)

    return results


def main():
    all_results = {}

    for name, load_fn in [('WN18RR', load_wn18rr), ('FB15k-237', load_fb15k237)]:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}")

        train_ds, _, test_ds = load_fn()
        train = train_ds.triples
        test = test_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations

        results_per_seed = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            binary_unc, continuous_unc = compute_freq_heuristic_uncertainties(
                train, test, n_ent, n_rel
            )
            results = evaluate_temporal_heuristic(
                binary_unc, continuous_unc, train, test, n_ent, n_rel
            )
            results_per_seed.append(results)
            print(f"\nSeed {seed}:")
            print(f"  Binary (= U_str):  overall={results.get('binary_overall', 'N/A'):.4f}"
                  f"  emerging={results.get('binary_emerging', 'N/A'):.4f}"
                  f"  novel={results.get('binary_novel', 'N/A'):.4f}")
            print(f"  Log-freq (cont.):  overall={results.get('log_freq_overall', 'N/A'):.4f}"
                  f"  emerging={results.get('log_freq_emerging', 'N/A'):.4f}"
                  f"  novel={results.get('log_freq_novel', 'N/A'):.4f}")

        # Average over seeds (binary is deterministic, but log_freq may vary slightly)
        avg = {}
        for key in results_per_seed[0]:
            if isinstance(results_per_seed[0][key], float):
                vals = [r[key] for r in results_per_seed]
                avg[f'{key}_mean'] = float(np.mean(vals))
                avg[f'{key}_std'] = float(np.std(vals))
            else:
                avg[key] = results_per_seed[0][key]

        all_results[name] = avg

        print(f"\n{name} Summary (3-seed mean):")
        print(f"  Binary overall:   {avg.get('binary_overall_mean', 0):.4f} ± {avg.get('binary_overall_std', 0):.4f}")
        print(f"  Log-freq overall: {avg.get('log_freq_overall_mean', 0):.4f} ± {avg.get('log_freq_overall_std', 0):.4f}")

    outfile = project_root / 'outputs' / 'freq_heuristic_baseline.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
