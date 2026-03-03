#!/usr/bin/env python3
"""
Margin Loss Ablation - 30 epochs, 3 seeds, WN18RR + FB15k-237.
Wrapper that patches the original script's defaults.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import gc
import time

from src.data.loaders import load_wn18rr, load_fb15k237

# Import model classes from original script
sys.path.insert(0, str(Path(__file__).parent))
from margin_loss_ablation import CAGP, CoverageOnly, GPOnly

EPOCHS = 30
SEEDS = [42, 123, 456]

def train_model(model, triples, device, epochs=30, unc_weight=0.1, batch_size=512):
    """Train with configurable epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    h_all, r_all, t_all = triples[:, 0], triples[:, 1], triples[:, 2]
    dataset = TensorDataset(h_all, r_all, t_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss

            # KL if variational
            if hasattr(model, 'entity_logvar'):
                kl = -0.5 * torch.mean(1 + model.entity_logvar - model.entity_mean.pow(2) - model.entity_logvar.exp())
                loss += 0.001 * kl

            # Margin loss
            if unc_weight > 0 and hasattr(model, 'get_uncertainty'):
                unc_pos = model.get_uncertainty(h, r, t)
                unc_neg = model.get_uncertainty(h, r, neg_t)
                margin_loss = F.relu(0.3 - (unc_neg - unc_pos)).mean()
                loss += unc_weight * margin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


def evaluate_ood(model, train_triples, test_triples, device):
    """Evaluate OOD detection."""
    model.eval()

    # Precompute coverage
    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(train_triples)

    # Get entity frequencies
    freq = torch.zeros(model.num_entities, dtype=torch.long)
    for col in [0, 2]:
        for e in train_triples[:, col]:
            freq[e.item()] += 1

    tau = int(np.percentile(freq[freq > 0].numpy(), 25))

    h_test = test_triples[:, 0]
    r_test = test_triples[:, 1]
    t_test = test_triples[:, 2]

    # OOD labels
    min_freq = torch.minimum(freq[h_test], freq[t_test])
    is_emerging = min_freq <= tau

    cov_h = model.coverage[h_test, r_test] if hasattr(model, 'coverage') else torch.ones(len(h_test))
    cov_t = model.coverage[t_test, r_test] if hasattr(model, 'coverage') else torch.ones(len(h_test))
    is_novel = (~is_emerging) & ((cov_h == 0) | (cov_t == 0))

    is_ood = is_emerging | is_novel

    if is_ood.sum() == 0 or (~is_ood).sum() == 0:
        return {'overall': float('nan'), 'emerging': float('nan'), 'novel': float('nan')}

    with torch.no_grad():
        uncertainties = model.get_uncertainty(
            h_test.to(device), r_test.to(device), t_test.to(device)
        ).cpu().numpy()

    labels = is_ood.numpy().astype(int)

    results = {}
    results['overall'] = float(roc_auc_score(labels, uncertainties))

    if is_emerging.sum() > 0:
        mask = is_emerging | (~is_ood)
        if mask.sum() > 0 and is_emerging[mask].sum() > 0 and (~is_ood)[mask].sum() > 0:
            results['emerging'] = float(roc_auc_score(is_emerging[mask].numpy(), uncertainties[mask]))

    if is_novel.sum() > 0:
        mask = is_novel | (~is_ood)
        if mask.sum() > 0 and is_novel[mask].sum() > 0 and (~is_ood)[mask].sum() > 0:
            results['novel'] = float(roc_auc_score(is_novel[mask].numpy(), uncertainties[mask]))

    return results


def run_single(dataset_name, train_ds, test_ds, seed, device):
    """Run one seed for one dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_triples = torch.tensor(train_ds.triples, dtype=torch.long)
    test_triples = torch.tensor(test_ds.triples, dtype=torch.long)
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    results = {}

    # CAGP with margin loss (w_unc=0.1)
    print(f"  [seed={seed}] CAGP w_unc=0.1 ...")
    model = CAGP(num_entities, num_relations).to(device)
    model = train_model(model, train_triples, device, epochs=EPOCHS, unc_weight=0.1)
    model.precompute_coverage(train_triples)
    r = evaluate_ood(model, train_triples, test_triples, device)
    results['cagp_with_margin'] = r
    del model; gc.collect()

    # CAGP without margin loss (w_unc=0.0)
    print(f"  [seed={seed}] CAGP w_unc=0.0 ...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CAGP(num_entities, num_relations).to(device)
    model = train_model(model, train_triples, device, epochs=EPOCHS, unc_weight=0.0)
    model.precompute_coverage(train_triples)
    r = evaluate_ood(model, train_triples, test_triples, device)
    results['cagp_without_margin'] = r
    del model; gc.collect()

    # CoverageOnly baseline
    print(f"  [seed={seed}] CoverageOnly ...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CoverageOnly(num_entities, num_relations).to(device)
    model = train_model(model, train_triples, device, epochs=EPOCHS, unc_weight=0.0)
    model.precompute_coverage(train_triples)
    r = evaluate_ood(model, train_triples, test_triples, device)
    results['coverage_only'] = r
    del model; gc.collect()

    # GPOnly baseline
    print(f"  [seed={seed}] GPOnly ...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GPOnly(num_entities, num_relations).to(device)
    model = train_model(model, train_triples, device, epochs=EPOCHS, unc_weight=0.0)
    r = evaluate_ood(model, train_triples, test_triples, device)
    results['gp_only'] = r
    del model; gc.collect()

    return results


def main():
    device = torch.device('cpu')
    all_results = {}
    start_time = time.time()

    datasets = {}
    print("Loading WN18RR...")
    train_wn, _, test_wn = load_wn18rr()
    datasets['WN18RR'] = (train_wn, test_wn)
    print("Loading FB15k-237...")
    try:
        train_fb, _, test_fb = load_fb15k237()
        datasets['FB15k-237'] = (train_fb, test_fb)
    except Exception as e:
        print(f"Skipping FB15k-237: {e}")

    for ds_name, (train_ds, test_ds) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            r = run_single(ds_name, train_ds, test_ds, seed, device)
            seed_results.append(r)

        # Aggregate
        summary = {}
        for method in seed_results[0].keys():
            for metric in ['overall', 'emerging', 'novel']:
                vals = [s[method].get(metric, float('nan')) for s in seed_results]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    key = f"{method}_{metric}"
                    summary[key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'values': vals,
                    }

        all_results[ds_name] = {
            'per_seed': seed_results,
            'summary': summary,
        }

    elapsed = time.time() - start_time
    all_results['metadata'] = {
        'epochs': EPOCHS,
        'seeds': SEEDS,
        'elapsed_seconds': elapsed,
    }

    out_path = Path(__file__).parent.parent / 'outputs' / 'margin_loss_ablation_30epoch_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nDone! Elapsed: {elapsed:.0f}s")
    print(f"Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY (30 epochs, {len(SEEDS)} seeds)")
    print(f"{'='*80}")
    for ds_name in datasets:
        print(f"\n{ds_name}:")
        s = all_results[ds_name]['summary']
        for key, val in sorted(s.items()):
            print(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")


if __name__ == '__main__':
    main()
