#!/usr/bin/env python3
"""
Matched-Coverage Analysis for UAI 2026 Rebuttal

Defense against "coverage is trivial" critique:
  Among triples that SHARE the same coverage value, does the GP semantic
  component still provide separation between ID and OOD?

If yes → coverage alone is not doing all the work; the semantic signal
  adds genuine value even when coverage is controlled for.

Protocol:
  1. Split test triples into temporal OOD categories (emerging, novel_ctx, ID)
  2. Bin triples by coverage value (0, 1, or 2 endpoints covered)
  3. Within each bin, compute AUROC using GP-only uncertainty
  4. If GP-only AUROC > 0.5 within a bin, the semantic signal provides
     separation beyond what coverage alone explains.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from collections import defaultdict

from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class GPOnly(nn.Module):
    """GP variance only — same as in run_wn18rr_temporal.py."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def _kl_entity_gaussian(model):
    if not (hasattr(model, 'entity_mean') and hasattr(model, 'entity_logvar')):
        return None
    mean = model.entity_mean
    logvar = model.entity_logvar
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()


def train_model(model, triples, device, epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)
            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl
            # Uncertainty margin: OOD (neg) should have higher uncertainty
            if hasattr(model, 'entity_logvar') or hasattr(model, 'var_net'):
                pos_unc = model.get_uncertainty(h, r, t)
                neg_unc = model.get_uncertainty(h, r, neg_t)
                unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
                loss = loss + unc_weight * unc_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")
    return model


def run_matched_coverage(ds_name, loader, device, seeds=(42, 123, 456), epochs=30):
    """Run matched-coverage analysis on a dataset."""
    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Build coverage matrix
    coverage = np.zeros((n_ent, n_rel), dtype=np.float32)
    for i in range(len(train)):
        coverage[train[i, 0], train[i, 1]] = 1.0
        coverage[train[i, 2], train[i, 1]] = 1.0

    # Entity frequencies
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    thresh = np.percentile(list(freq.values()), 25)

    # Categorize test triples
    emerging_idx, novel_ctx_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if freq.get(h, 0) <= thresh or freq.get(t, 0) <= thresh:
            emerging_idx.append(i)
        elif coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_ctx_idx.append(i)
        else:
            id_idx.append(i)

    # Compute coverage score for each test triple
    cov_scores = np.array([
        coverage[test[i, 0], test[i, 1]] + coverage[test[i, 2], test[i, 1]]
        for i in range(len(test))
    ])

    print(f"\n{'='*60}")
    print(f"  Matched-Coverage Analysis: {ds_name}")
    print(f"{'='*60}")
    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Split: emerging={len(emerging_idx)}, novel_ctx={len(novel_ctx_idx)}, id={len(id_idx)}")
    print(f"Threshold (25th pctl): {thresh}")

    all_seed_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GPOnly(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=epochs)
        model.eval()

        # Get GP uncertainty for all test triples
        with torch.no_grad():
            h = torch.tensor(test[:, 0]).to(device)
            r = torch.tensor(test[:, 1]).to(device)
            t = torch.tensor(test[:, 2]).to(device)
            gp_unc = model.get_uncertainty(h, r, t).cpu().numpy()

        seed_result = {}

        # === Analysis 1: Among COVERED triples (cov_score == 2), ===
        # === can GP separate emerging vs ID?                     ===
        covered_emerging = [i for i in emerging_idx if cov_scores[i] == 2]
        covered_id = [i for i in id_idx if cov_scores[i] == 2]

        if len(covered_emerging) > 20 and len(covered_id) > 20:
            labels = np.concatenate([
                np.zeros(len(covered_id)), np.ones(len(covered_emerging))
            ])
            scores = np.concatenate([
                gp_unc[covered_id], gp_unc[covered_emerging]
            ])
            auroc = roc_auc_score(labels, scores)
            seed_result['covered_emerging_vs_id'] = {
                'auroc': float(auroc),
                'n_emerging': len(covered_emerging),
                'n_id': len(covered_id),
            }
            print(f"  Covered (cov=2): emerging vs ID AUROC = {auroc:.4f} "
                  f"(n={len(covered_emerging)} vs {len(covered_id)})")
        else:
            print(f"  Covered (cov=2): insufficient samples "
                  f"(emerging={len(covered_emerging)}, id={len(covered_id)})")

        # === Analysis 2: Among PARTIALLY covered (cov_score == 1) ===
        partial_emerging = [i for i in emerging_idx if cov_scores[i] == 1]
        partial_id = [i for i in id_idx if cov_scores[i] == 1]

        if len(partial_emerging) > 20 and len(partial_id) > 20:
            labels = np.concatenate([
                np.zeros(len(partial_id)), np.ones(len(partial_emerging))
            ])
            scores = np.concatenate([
                gp_unc[partial_id], gp_unc[partial_emerging]
            ])
            auroc = roc_auc_score(labels, scores)
            seed_result['partial_emerging_vs_id'] = {
                'auroc': float(auroc),
                'n_emerging': len(partial_emerging),
                'n_id': len(partial_id),
            }
            print(f"  Partial (cov=1): emerging vs ID AUROC = {auroc:.4f} "
                  f"(n={len(partial_emerging)} vs {len(partial_id)})")

        # === Analysis 3: Among UNCOVERED (cov_score == 0) ===
        uncov_emerging = [i for i in emerging_idx if cov_scores[i] == 0]
        uncov_novel = [i for i in novel_ctx_idx if cov_scores[i] == 0]

        if len(uncov_emerging) > 20 and len(uncov_novel) > 20:
            # Here both are OOD but different type — can GP distinguish?
            labels = np.concatenate([
                np.zeros(len(uncov_novel)), np.ones(len(uncov_emerging))
            ])
            scores = np.concatenate([
                gp_unc[uncov_novel], gp_unc[uncov_emerging]
            ])
            auroc = roc_auc_score(labels, scores)
            seed_result['uncov_emerging_vs_novel'] = {
                'auroc': float(auroc),
                'n_emerging': len(uncov_emerging),
                'n_novel': len(uncov_novel),
            }
            print(f"  Uncovered (cov=0): emerging vs novel AUROC = {auroc:.4f}")

        # === Analysis 4: Overall GP-only on emerging (controls for novel_ctx) ===
        if len(emerging_idx) > 50 and len(id_idx) > 50:
            labels = np.concatenate([
                np.zeros(len(id_idx)), np.ones(len(emerging_idx))
            ])
            scores = np.concatenate([
                gp_unc[id_idx], gp_unc[emerging_idx]
            ])
            auroc = roc_auc_score(labels, scores)
            seed_result['gp_emerging_vs_id_overall'] = {
                'auroc': float(auroc),
                'n_emerging': len(emerging_idx),
                'n_id': len(id_idx),
            }
            print(f"  GP-only emerging vs ID (overall): AUROC = {auroc:.4f}")

        all_seed_results.append(seed_result)

    # Aggregate across seeds
    summary = {}
    for key in all_seed_results[0]:
        aurocs = [r[key]['auroc'] for r in all_seed_results if key in r]
        if aurocs:
            summary[key] = {
                'auroc_mean': float(np.mean(aurocs)),
                'auroc_std': float(np.std(aurocs)),
                'n_seeds': len(aurocs),
            }

    return {
        'dataset': ds_name,
        'n_entities': n_ent,
        'n_relations': n_rel,
        'threshold': float(thresh),
        'split': {
            'emerging': len(emerging_idx),
            'novel_ctx': len(novel_ctx_idx),
            'id': len(id_idx),
        },
        'per_seed': all_seed_results,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Matched-coverage analysis")
    parser.add_argument('--datasets', type=str, default='wn18rr,fb15k237')
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output', type=str,
                        default=str(project_root / 'outputs' / 'matched_coverage_results.json'))
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    seeds = tuple(int(s.strip()) for s in args.seeds.split(','))
    datasets = [d.strip().lower() for d in args.datasets.split(',')]

    loaders = {
        'wn18rr': ('WN18RR', load_wn18rr),
        'fb15k237': ('FB15k-237', load_fb15k237),
    }

    results = {}
    for ds in datasets:
        name, loader = loaders[ds]
        results[ds] = run_matched_coverage(name, loader, device, seeds=seeds, epochs=args.epochs)

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out}")

    # Print summary
    print("\n" + "=" * 70)
    print("MATCHED-COVERAGE SUMMARY")
    print("=" * 70)
    for ds in results:
        print(f"\n{results[ds]['dataset']}:")
        for key, vals in results[ds]['summary'].items():
            print(f"  {key}: {vals['auroc_mean']:.3f} ± {vals['auroc_std']:.3f}")


if __name__ == "__main__":
    main()
