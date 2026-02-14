#!/usr/bin/env python3
"""
Validate Theorem 2(iv): at mixed OOD proportions, the oracle combination
of U_sem and U_str outperforms either signal alone.

Approach:
- Train GPOnly and CoverageOnly models
- Get per-triple uncertainties for emerging, novel-context, and ID triples
- At varying emerging/(emerging+novel) ratios, compute AUROC for:
    U_sem alone, U_str alone, oracle combination (grid-search alpha)
- Output: CSV + summary showing the crossing-curves pattern
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
import csv

from scripts.run_wn18rr_temporal import (
    GPOnly, CoverageOnly, train_model, setup_device, _is_emerging
)
from src.data.loaders import load_fb15k237, load_wn18rr


def get_per_triple_uncertainties(model, triples, device):
    """Get uncertainty for each triple."""
    model.eval()
    with torch.no_grad():
        h = torch.tensor(triples[:, 0]).to(device)
        r = torch.tensor(triples[:, 1]).to(device)
        t = torch.tensor(triples[:, 2]).to(device)
        # Process in batches to avoid OOM
        batch_size = 4096
        uncs = []
        for i in range(0, len(h), batch_size):
            unc = model.get_uncertainty(
                h[i:i+batch_size], r[i:i+batch_size], t[i:i+batch_size]
            ).cpu().numpy()
            uncs.append(unc)
    return np.concatenate(uncs)


def categorize_triples(train, test, coverage):
    """Split test triples into emerging, novel-context, ID."""
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)
    cov = coverage.cpu().numpy()

    emerging_idx, novel_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, 'leq'):
            emerging_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            novel_idx.append(i)
        else:
            id_idx.append(i)

    return emerging_idx, novel_idx, id_idx


def compute_mixture_auroc(sem_emerging, sem_novel, sem_id,
                          str_emerging, str_novel, str_id,
                          frac_emerging, n_ood=1000, n_id=1000, n_trials=20):
    """
    Compute AUROC at a given emerging/(emerging+novel) ratio.
    Samples n_ood OOD triples at the given ratio, n_id ID triples.
    Returns mean AUROC over n_trials for U_sem, U_str, and oracle combination.
    """
    n_emerging = int(frac_emerging * n_ood)
    n_novel = n_ood - n_emerging

    # Skip if we don't have enough samples
    if (n_emerging > 0 and len(sem_emerging) < 10) or \
       (n_novel > 0 and len(sem_novel) < 10) or len(sem_id) < 10:
        return None, None, None, None

    sem_results, str_results, oracle_results = [], [], []
    best_alphas = []

    for _ in range(n_trials):
        # Sample OOD
        ood_sem, ood_str = [], []
        if n_emerging > 0:
            idx = np.random.choice(len(sem_emerging), n_emerging, replace=True)
            ood_sem.append(sem_emerging[idx])
            ood_str.append(str_emerging[idx])
        if n_novel > 0:
            idx = np.random.choice(len(sem_novel), n_novel, replace=True)
            ood_sem.append(sem_novel[idx])
            ood_str.append(str_novel[idx])

        ood_sem = np.concatenate(ood_sem)
        ood_str = np.concatenate(ood_str)

        # Sample ID
        idx = np.random.choice(len(sem_id), min(n_id, len(sem_id)), replace=True)
        id_sem = sem_id[idx]
        id_str = str_id[idx]

        # Labels
        labels = np.concatenate([np.zeros(len(id_sem)), np.ones(len(ood_sem))])
        all_sem = np.concatenate([id_sem, ood_sem])
        all_str = np.concatenate([id_str, ood_str])

        # AUROC for each signal
        try:
            sem_auc = roc_auc_score(labels, all_sem)
            str_auc = roc_auc_score(labels, all_str)
        except ValueError:
            continue

        # Oracle combination: grid search alpha
        best_auc = max(sem_auc, str_auc)
        best_alpha = 1.0 if sem_auc > str_auc else 0.0
        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * all_sem + (1 - alpha) * all_str
            try:
                auc = roc_auc_score(labels, combined)
                if auc > best_auc:
                    best_auc = auc
                    best_alpha = alpha
            except ValueError:
                pass

        sem_results.append(sem_auc)
        str_results.append(str_auc)
        oracle_results.append(best_auc)
        best_alphas.append(best_alpha)

    if not sem_results:
        return None, None, None, None

    return (np.mean(sem_results), np.mean(str_results),
            np.mean(oracle_results), np.mean(best_alphas))


def run_dataset(name, load_fn, seed=42):
    """Run the mixture experiment on one dataset."""
    device = setup_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    train_ds, _, test_ds = load_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}")

    # Train GPOnly
    print("\nTraining GPOnly...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gp = GPOnly(n_ent, n_rel)
    gp.precompute_coverage(train)
    gp = train_model(gp, train, device, epochs=30)
    if hasattr(gp, 'calibrate_normalization'):
        gp.calibrate_normalization(train, device)

    # Train CoverageOnly
    print("Training CoverageOnly...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cov = CoverageOnly(n_ent, n_rel)
    cov.precompute_coverage(train)
    cov = train_model(cov, train, device, epochs=30)

    # Categorize test triples
    emerging_idx, novel_idx, id_idx = categorize_triples(train, test, cov.coverage)
    print(f"Split: emerging={len(emerging_idx)}, novel={len(novel_idx)}, ID={len(id_idx)}")

    # Get per-triple uncertainties
    print("Computing uncertainties...")
    sem_all = get_per_triple_uncertainties(gp, test, device)
    str_all = get_per_triple_uncertainties(cov, test, device)

    sem_emerging = sem_all[emerging_idx]
    sem_novel = sem_all[novel_idx]
    sem_id = sem_all[id_idx]
    str_emerging = str_all[emerging_idx]
    str_novel = str_all[novel_idx]
    str_id = str_all[id_idx]

    # Mixture experiment
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print(f"\n{'Frac_Emerg':>10} {'U_sem':>8} {'U_str':>8} {'Oracle':>8} {'Alpha*':>8}")
    print("-" * 46)

    for frac in fractions:
        sem_auc, str_auc, oracle_auc, alpha = compute_mixture_auroc(
            sem_emerging, sem_novel, sem_id,
            str_emerging, str_novel, str_id,
            frac_emerging=frac
        )
        if sem_auc is not None:
            print(f"{frac:>10.1f} {sem_auc:>8.3f} {str_auc:>8.3f} {oracle_auc:>8.3f} {alpha:>8.2f}")
            results.append({
                'dataset': name,
                'frac_emerging': frac,
                'auroc_sem': sem_auc,
                'auroc_str': str_auc,
                'auroc_oracle': oracle_auc,
                'best_alpha': alpha,
            })

    return results


def main():
    all_results = []

    for name, load_fn in [('WN18RR', load_wn18rr), ('FB15k-237', load_fb15k237)]:
        results = run_dataset(name, load_fn)
        all_results.extend(results)

    # Save results
    outfile = project_root / 'outputs' / 'theorem2iv_mixture.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Also save CSV for easy plotting
    csvfile = project_root / 'outputs' / 'theorem2iv_mixture.csv'
    with open(csvfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'frac_emerging',
                                                'auroc_sem', 'auroc_str',
                                                'auroc_oracle', 'best_alpha'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved to {csvfile}")

    # Summary
    print("\n" + "="*60)
    print("THEOREM 2(iv) VALIDATION SUMMARY")
    print("="*60)
    for name in ['WN18RR', 'FB15k-237']:
        ds_results = [r for r in all_results if r['dataset'] == name]
        if not ds_results:
            continue
        print(f"\n{name}:")
        # Find crossing point
        for r in ds_results:
            gain = r['auroc_oracle'] - max(r['auroc_sem'], r['auroc_str'])
            marker = " <-- oracle helps" if gain > 0.005 else ""
            print(f"  frac={r['frac_emerging']:.1f}: sem={r['auroc_sem']:.3f} "
                  f"str={r['auroc_str']:.3f} oracle={r['auroc_oracle']:.3f} "
                  f"(gain={gain:+.3f}){marker}")


if __name__ == "__main__":
    main()
