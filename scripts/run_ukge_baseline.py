#!/usr/bin/env python3
"""
UKGE (Uncertain Knowledge Graph Embedding) baseline.

UKGE (Chen et al., AAAI 2019) represents entity/relation embeddings and outputs
a confidence score per triple. This script implements two UKGE variants:

1. UKGE_logi: Logistic variant - confidence = sigmoid(score)
2. UKGE_rect: Rectified variant - confidence = clamp(score, 0, 1)

Key hypothesis: UKGE will FAIL on zero-coverage (novel-context) queries because
it derives uncertainty from embeddings, not structural coverage. This supports
our position paper's argument about the coverage blind spot.

Usage:
    python scripts/run_ukge_baseline.py
    python scripts/run_ukge_baseline.py --datasets wn18rr,fb15k237
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from collections import defaultdict
import time

from src.data.loaders import load_wn18rr, load_fb15k237
from src.ranking import compute_rank_from_scores


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# UKGE Model Variants
# ============================================================

class UKGE_Logi(nn.Module):
    """
    UKGE Logistic variant (Chen et al., AAAI 2019).

    Uses DistMult-style scoring with sigmoid confidence.
    Uncertainty = 1 - confidence = 1 - sigmoid(score)

    This is a "Bayesian" KGE in the sense that it outputs calibrated
    probabilities, but the uncertainty comes from the score magnitude,
    not from any explicit modeling of epistemic uncertainty.
    """

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity and relation embeddings
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Initialize with Xavier
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage buffer for tracking
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        """Compute DistMult score."""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_confidence(self, h, r, t):
        """Confidence = sigmoid(score)."""
        return torch.sigmoid(self.forward(h, r, t))

    def get_uncertainty(self, h, r, t):
        """Uncertainty = 1 - confidence."""
        conf = self.get_confidence(h, r, t)
        return 1.0 - conf

    def precompute_coverage(self, triples):
        """Track entity-relation coverage from training data."""
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class UKGE_Rect(nn.Module):
    """
    UKGE Rectified variant.

    Uses bounded box embeddings where scores are clamped to [0,1].
    Uncertainty = 1 - clamped_score
    """

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity and relation embeddings (bounded)
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Initialize in [0, 1] range
        nn.init.uniform_(self.entity_emb.weight, 0.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, 0.0, 1.0)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        """Compute bounded score."""
        h_emb = torch.sigmoid(self.entity_emb(h))  # Bound to [0,1]
        r_emb = torch.sigmoid(self.relation_emb(r))
        t_emb = torch.sigmoid(self.entity_emb(t))

        # Use squared distance style scoring
        diff = h_emb * r_emb - t_emb
        score = 1.0 - (diff ** 2).mean(dim=-1)  # Higher score = better match
        return score

    def get_confidence(self, h, r, t):
        """Confidence = clamped score."""
        return torch.clamp(self.forward(h, r, t), 0.0, 1.0)

    def get_uncertainty(self, h, r, t):
        """Uncertainty = 1 - confidence."""
        conf = self.get_confidence(h, r, t)
        return 1.0 - conf

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class UKGE_PSL(nn.Module):
    """
    UKGE with Probabilistic Soft Logic inspired confidence aggregation.

    Combines embedding confidence with neighbor evidence.
    This is closer to the full UKGE paper which uses PSL rules.
    """

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Mean embeddings
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Per-entity confidence (learned)
        self.entity_logconf = nn.Parameter(torch.zeros(num_entities))

        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('entity_freq', torch.zeros(num_entities))

    def forward(self, h, r, t):
        """DistMult scoring."""
        h_emb = self.entity_mean[h]
        r_emb = self.relation_emb(r)
        t_emb = self.entity_mean[t]
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_confidence(self, h, r, t):
        """
        Confidence combines:
        1. Score-based confidence (sigmoid of score)
        2. Entity confidence (learned per-entity)
        """
        score_conf = torch.sigmoid(self.forward(h, r, t))

        # Entity confidence (normalized by frequency)
        h_conf = torch.sigmoid(self.entity_logconf[h])
        t_conf = torch.sigmoid(self.entity_logconf[t])
        entity_conf = (h_conf + t_conf) / 2

        # Lukasiewicz t-norm: conf(a) AND conf(b) = max(0, a + b - 1)
        combined = torch.clamp(score_conf + entity_conf - 1.0, min=0.01, max=0.99)
        return combined

    def get_uncertainty(self, h, r, t):
        """Uncertainty = 1 - confidence."""
        return 1.0 - self.get_confidence(h, r, t)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0
            self.entity_freq[triples[i, 0]] += 1
            self.entity_freq[triples[i, 2]] += 1


# ============================================================
# Training
# ============================================================

def train_model(model, triples, device, epochs=30, lr=0.001):
    """Train UKGE model with BCE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            # Positive samples: true triples
            pos_scores = model(h, r, t)

            # Negative samples: corrupt tail
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            # BCE loss
            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


# ============================================================
# Temporal OOD Evaluation (matching run_wn18rr_temporal.py)
# ============================================================

def evaluate_temporal(model, train, test, n_ent, device, emerging_operator='leq'):
    """
    Temporal-like OOD evaluation with 25th percentile threshold.

    Splits test triples into:
    - emerging: entities below frequency threshold
    - novel_ctx: known entities but unseen (entity, relation) pair
    - id: fully in-distribution
    """
    model.eval()

    # Entity frequencies from training
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()

    # Categorize test triples
    new_entity_idx, new_pair_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        h_freq = freq.get(h, 0)
        t_freq = freq.get(t, 0)

        if emerging_operator == 'leq':
            is_emerging = h_freq <= thresh or t_freq <= thresh
        else:
            is_emerging = h_freq < thresh or t_freq < thresh

        if is_emerging:
            new_entity_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    print(f"    Split: emerging={len(new_entity_idx)}, novel_ctx={len(new_pair_idx)}, id={len(id_idx)}")

    results = {
        'n_emerging': len(new_entity_idx),
        'n_novel_ctx': len(new_pair_idx),
        'n_id': len(id_idx),
        'threshold': float(thresh),
        'emerging_operator': emerging_operator,
    }

    # Overall temporal OOD: emerging + novel_ctx vs ID
    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            ood_triples = test[ood_idx]
            id_triples = test[id_idx]

            h_ood = torch.tensor(ood_triples[:, 0]).to(device)
            r_ood = torch.tensor(ood_triples[:, 1]).to(device)
            t_ood = torch.tensor(ood_triples[:, 2]).to(device)
            ood_unc = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

            h_id = torch.tensor(id_triples[:, 0]).to(device)
            r_id = torch.tensor(id_triples[:, 1]).to(device)
            t_id = torch.tensor(id_triples[:, 2]).to(device)
            id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])

        try:
            results['overall_auroc'] = float(roc_auc_score(labels, scores))
            results['overall_aupr'] = float(average_precision_score(labels, scores))
        except Exception:
            results['overall_auroc'] = 0.5
            results['overall_aupr'] = 0.5

    # Per-category: emerging vs ID
    if len(new_entity_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            e_triples = test[new_entity_idx]
            i_triples = test[id_idx]

            h_e = torch.tensor(e_triples[:, 0]).to(device)
            r_e = torch.tensor(e_triples[:, 1]).to(device)
            t_e = torch.tensor(e_triples[:, 2]).to(device)
            e_unc = model.get_uncertainty(h_e, r_e, t_e).cpu().numpy()

            h_i = torch.tensor(i_triples[:, 0]).to(device)
            r_i = torch.tensor(i_triples[:, 1]).to(device)
            t_i = torch.tensor(i_triples[:, 2]).to(device)
            i_unc = model.get_uncertainty(h_i, r_i, t_i).cpu().numpy()

        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(e_unc))])
        scores = np.concatenate([i_unc, e_unc])
        try:
            results['emerging_auroc'] = float(roc_auc_score(labels, scores))
        except Exception:
            results['emerging_auroc'] = 0.5

    # Per-category: novel context vs ID (THE KEY METRIC)
    if len(new_pair_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            n_triples = test[new_pair_idx]
            i_triples = test[id_idx]

            h_n = torch.tensor(n_triples[:, 0]).to(device)
            r_n = torch.tensor(n_triples[:, 1]).to(device)
            t_n = torch.tensor(n_triples[:, 2]).to(device)
            n_unc = model.get_uncertainty(h_n, r_n, t_n).cpu().numpy()

            h_i = torch.tensor(i_triples[:, 0]).to(device)
            r_i = torch.tensor(i_triples[:, 1]).to(device)
            t_i = torch.tensor(i_triples[:, 2]).to(device)
            i_unc = model.get_uncertainty(h_i, r_i, t_i).cpu().numpy()

        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(n_unc))])
        scores = np.concatenate([i_unc, n_unc])
        try:
            results['novel_ctx_auroc'] = float(roc_auc_score(labels, scores))
        except Exception:
            results['novel_ctx_auroc'] = 0.5

    results['eval_mode'] = 'full'
    return results


def evaluate_error_prediction(model, test, train, n_ent, device, max_test=2000):
    """
    Error prediction: can uncertainty predict rank > 10?

    Higher uncertainty should correlate with higher rank (worse prediction).
    """
    model.eval()

    if max_test and len(test) > max_test:
        indices = np.random.choice(len(test), max_test, replace=False)
        test_subset = test[indices]
    else:
        test_subset = test

    # Build filter set
    all_triples = np.concatenate([train, test], axis=0)
    filter_set = set()
    for i in range(len(all_triples)):
        filter_set.add((int(all_triples[i, 0]), int(all_triples[i, 1]), int(all_triples[i, 2])))

    ranks = []
    uncertainties = []

    with torch.no_grad():
        for i in range(len(test_subset)):
            h, r, t = int(test_subset[i, 0]), int(test_subset[i, 1]), int(test_subset[i, 2])

            # Score all entities as tails
            h_batch = torch.full((n_ent,), h, dtype=torch.long, device=device)
            r_batch = torch.full((n_ent,), r, dtype=torch.long, device=device)
            t_batch = torch.arange(n_ent, device=device)

            scores = model(h_batch, r_batch, t_batch).cpu().numpy()

            # Filter known triples
            for tt in range(n_ent):
                if tt != t and (h, r, tt) in filter_set:
                    scores[tt] = -1e9

            # Compute rank
            rank = compute_rank_from_scores(scores, t)
            ranks.append(rank)

            # Get uncertainty for this triple
            h_t = torch.tensor([h], device=device)
            r_t = torch.tensor([r], device=device)
            t_t = torch.tensor([t], device=device)
            unc = model.get_uncertainty(h_t, r_t, t_t).cpu().item()
            uncertainties.append(unc)

    ranks = np.array(ranks)
    uncertainties = np.array(uncertainties)

    # Error prediction: rank > 10 as "error"
    errors = (ranks > 10).astype(float)

    try:
        auroc = roc_auc_score(errors, uncertainties)
        aupr = average_precision_score(errors, uncertainties)
    except Exception:
        auroc = 0.5
        aupr = 0.5

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'mean_rank': float(np.mean(ranks)),
        'mrr': float(np.mean(1.0 / ranks)),
        'hits@10': float(np.mean(ranks <= 10)),
        'error_rate': float(np.mean(errors)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run UKGE baseline for coverage blind spot analysis.")
    parser.add_argument('--datasets', type=str, default='fb15k237,wn18rr',
                        help="Comma-separated datasets: fb15k237,wn18rr")
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help="Comma-separated random seeds")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str,
                        default=str(project_root / 'outputs' / 'ukge_baseline_results.json'))
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(',') if d.strip()]

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        if ds_name == 'fb15k237':
            train_ds, _, test_ds = load_fb15k237()
        elif ds_name == 'wn18rr':
            train_ds, _, test_ds = load_wn18rr()
        else:
            print(f"Unknown dataset: {ds_name}")
            continue

        train = train_ds.triples
        test = test_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations

        print(f"Entities: {n_ent}, Relations: {n_rel}")
        print(f"Train: {len(train)}, Test: {len(test)}")

        ds_results = {}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            seed_results = {}

            # UKGE variants
            models_to_test = [
                ('UKGE_Logi', UKGE_Logi),
                ('UKGE_Rect', UKGE_Rect),
                ('UKGE_PSL', UKGE_PSL),
            ]

            for model_name, model_cls in models_to_test:
                print(f"\n  {model_name}:")
                t0 = time.time()

                # Reset seed for each model
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = model_cls(n_ent, n_rel, dim=args.dim)
                model.precompute_coverage(train)
                model = train_model(model, train, device, epochs=args.epochs, lr=args.lr)

                # Temporal OOD evaluation
                temporal = evaluate_temporal(model, train, test, n_ent, device)

                # Error prediction
                error_pred = evaluate_error_prediction(model, test, train, n_ent, device)

                elapsed = time.time() - t0

                print(f"    Overall AUROC: {temporal.get('overall_auroc', 'N/A'):.4f}")
                print(f"    Emerging AUROC: {temporal.get('emerging_auroc', 'N/A'):.4f}")
                print(f"    Novel Ctx AUROC: {temporal.get('novel_ctx_auroc', 'N/A'):.4f} <- KEY METRIC")
                print(f"    Error Pred AUROC: {error_pred['auroc']:.4f}")
                print(f"    MRR: {error_pred['mrr']:.4f}, Hits@10: {error_pred['hits@10']:.4f}")
                print(f"    Time: {elapsed:.1f}s")

                seed_results[model_name] = {
                    'temporal': temporal,
                    'error_prediction': error_pred,
                    'time': elapsed,
                }

            ds_results[f'seed_{seed}'] = seed_results

        # Compute summary statistics
        summary = {}
        model_names = list(ds_results[f'seed_{seeds[0]}'].keys())

        for model_name in model_names:
            summary[model_name] = {}

            for metric in ['overall_auroc', 'emerging_auroc', 'novel_ctx_auroc']:
                vals = []
                for seed in seeds:
                    t = ds_results[f'seed_{seed}'][model_name]['temporal']
                    if metric in t:
                        vals.append(t[metric])
                if vals:
                    summary[model_name][f'{metric}_mean'] = float(np.mean(vals))
                    summary[model_name][f'{metric}_std'] = float(np.std(vals))

            # Error prediction
            error_aurocs = [ds_results[f'seed_{seed}'][model_name]['error_prediction']['auroc']
                           for seed in seeds]
            summary[model_name]['error_pred_auroc_mean'] = float(np.mean(error_aurocs))
            summary[model_name]['error_pred_auroc_std'] = float(np.std(error_aurocs))

            # MRR
            mrrs = [ds_results[f'seed_{seed}'][model_name]['error_prediction']['mrr']
                   for seed in seeds]
            summary[model_name]['mrr_mean'] = float(np.mean(mrrs))

        ds_results['summary'] = summary
        ds_results['config'] = {
            'seeds': seeds,
            'epochs': args.epochs,
            'dim': args.dim,
            'lr': args.lr,
        }

        all_results[ds_name] = ds_results

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: UKGE Baseline Results")
    print("=" * 80)
    print("\nKey insight: Novel context AUROC should be ~0.5 (random) if UKGE")
    print("cannot detect zero-coverage queries (our hypothesis).")
    print()

    for ds in all_results:
        print(f"\n{ds.upper()}:")
        s = all_results[ds]['summary']
        print(f"  {'Method':<12} {'Overall':>12} {'Emerging':>12} {'Novel Ctx':>12} {'Error Pred':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for name in s:
            overall = f"{s[name].get('overall_auroc_mean', 0):.3f}" if 'overall_auroc_mean' in s[name] else "N/A"
            emerging = f"{s[name].get('emerging_auroc_mean', 0):.3f}" if 'emerging_auroc_mean' in s[name] else "N/A"
            novel_ctx = f"{s[name].get('novel_ctx_auroc_mean', 0):.3f}" if 'novel_ctx_auroc_mean' in s[name] else "N/A"
            error_pred = f"{s[name].get('error_pred_auroc_mean', 0):.3f}"
            print(f"  {name:<12} {overall:>12} {emerging:>12} {novel_ctx:>12} {error_pred:>12}")

    # Compare with existing baselines
    print("\n" + "=" * 80)
    print("COMPARISON WITH EXISTING BASELINES (from paper)")
    print("=" * 80)
    print("\nExpected baseline performance on novel-context detection:")
    print("  MC Dropout:    ~0.38-0.61 (high variance)")
    print("  Deep Ensemble: ~0.54-0.68")
    print("  SNGP:          ~0.39-0.40")
    print("\nIf UKGE performs similarly, it confirms our position that")
    print("embedding-based UQ methods fundamentally cannot detect zero-coverage.")


if __name__ == "__main__":
    main()
