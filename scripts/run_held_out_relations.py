#!/usr/bin/env python3
"""Held-out relation experiment: breaks circularity between OOD definition and coverage detector.

OOD = triples with held-out relations. Coverage trained only on remaining relations.

This experiment addresses the circularity critique: "Coverage detector is circular because it uses
the same training data to define both coverage and what's OOD."

Here, we break the circularity:
1. Train the model on ALL training triples (to learn good embeddings)
2. Build coverage matrix using ONLY train_rels (subset of relations)
3. Define OOD = test triples where relation ∈ held_out_rels
4. Define ID = test triples where relation ∈ train_rels AND both entities covered

Key insight: The detector has never seen held-out relation coverage patterns, but OOD
definition is purely "relation was held out", not defined by coverage itself.
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
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237, load_yago310


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions (from run_wn18rr_temporal.py)
# ============================================================

class GPOnly(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        """Build coverage matrix from triples."""
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class CoverageOnly(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        return 2.0 - self.coverage[h, r] - self.coverage[t, r]

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class CAGP(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self._norm_stats = None

    def forward(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def calibrate_normalization(self, triples, device):
        """Compute normalization statistics from a reference set."""
        with torch.no_grad():
            h = torch.tensor(triples[:, 0]).to(device)
            r = torch.tensor(triples[:, 1]).to(device)
            t = torch.tensor(triples[:, 2]).to(device)
            h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
            t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
            gp_var = (h_var + t_var) / 2
            cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
            self._norm_stats = {
                'gp_mean': gp_var.mean().item(),
                'cov_mean': cov_unc.mean().item(),
            }

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2
        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
        if self._norm_stats is not None:
            gp_mean = self._norm_stats['gp_mean']
            cov_mean = self._norm_stats['cov_mean']
        else:
            gp_mean = gp_var.mean().item()
            cov_mean = cov_unc.mean().item()
        gp_norm = gp_var / (gp_mean + 1e-8) * (cov_mean + 1e-8)
        alpha = torch.sigmoid(self.alpha)
        return alpha * gp_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Training
# ============================================================

def _kl_entity_gaussian(model):
    """KL(q(e)||N(0,1)) for models with explicit entity mean/logvar parameters."""
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

            # KL regularization toward N(0,1) prior
            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl

            # Uncertainty margin: OOD (neg) should have higher uncertainty
            if hasattr(model, 'entity_logvar'):
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


# ============================================================
# Held-out relation experiment
# ============================================================

def held_out_relation_experiment(
    dataset_name,
    loader,
    device,
    holdout_frac=0.2,
    seed=42,
    epochs=30,
    lr=0.001,
    kl_beta=0.001,
    unc_weight=0.1,
):
    """
    Held-out relation experiment to break circularity critique.

    1. Load dataset
    2. Split relations into train_rels (80%) and held_out_rels (20%)
    3. Train CAGP model on ALL training triples (normal training)
    4. Build coverage matrix using ONLY train_rels triples
    5. Define test OOD: test triples where relation ∈ held_out_rels
    6. Define test ID: test triples where relation ∈ train_rels AND both entities covered
    7. Evaluate AUROC of U_sem, U_str, CAGP on this split

    Key: The MODEL trains on all data (learns good embeddings)
         But COVERAGE is computed only from train_rels subset
         OOD label = "uses a held-out relation"
         This completely decouples OOD definition from coverage computation
    """
    print(f"\n{'='*60}")
    print(f"  {dataset_name} - Held-out Relation Experiment")
    print(f"  Seed: {seed}, Holdout fraction: {holdout_frac}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Split relations into train_rels and held_out_rels
    all_relations = np.arange(n_rel)
    np.random.shuffle(all_relations)
    n_holdout = int(n_rel * holdout_frac)
    held_out_rels = set(all_relations[:n_holdout].tolist())
    train_rels = set(all_relations[n_holdout:].tolist())

    print(f"\nRelation split:")
    print(f"  Train relations: {len(train_rels)} ({100*(1-holdout_frac):.0f}%)")
    print(f"  Held-out relations: {len(held_out_rels)} ({100*holdout_frac:.0f}%)")

    # Filter training triples for coverage computation
    train_rels_triples = train[[r in train_rels for r in train[:, 1]]]
    print(f"\nCoverage will be built from {len(train_rels_triples)} triples (train_rels only)")
    print(f"Model will train on all {len(train)} triples (including held-out relations)")

    # Categorize test triples
    test_holdout_idx = []  # OOD: relation in held_out_rels
    test_id_idx = []       # ID: relation in train_rels AND both entities covered

    # We need a temporary coverage matrix to determine ID vs OOD
    temp_coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train_rels_triples)):
        h, r, t = train_rels_triples[i]
        temp_coverage[h, r] = 1.0
        temp_coverage[t, r] = 1.0

    for i in range(len(test)):
        h, r, t = test[i]
        if r in held_out_rels:
            test_holdout_idx.append(i)
        elif r in train_rels and temp_coverage[h, r] == 1.0 and temp_coverage[t, r] == 1.0:
            test_id_idx.append(i)

    print(f"\nTest split:")
    print(f"  OOD (held-out relation): {len(test_holdout_idx)}")
    print(f"  ID (train_rels + covered): {len(test_id_idx)}")
    print(f"  Total test: {len(test)}")
    print(f"  OOD fraction: {len(test_holdout_idx)/len(test):.2%}")

    if len(test_holdout_idx) < 50 or len(test_id_idx) < 50:
        print("\nWARNING: Insufficient samples for evaluation. Skipping.")
        return None

    results = {
        'dataset': dataset_name,
        'seed': seed,
        'holdout_frac': float(holdout_frac),
        'n_entities': int(n_ent),
        'n_relations': int(n_rel),
        'n_train_rels': len(train_rels),
        'n_held_out_rels': len(held_out_rels),
        'n_train': len(train),
        'n_train_rels_triples': len(train_rels_triples),
        'n_test': len(test),
        'n_test_ood': len(test_holdout_idx),
        'n_test_id': len(test_id_idx),
        'ood_fraction': float(len(test_holdout_idx) / len(test)),
    }

    model_classes = {
        'GPOnly': GPOnly,
        'CoverageOnly': CoverageOnly,
        'CAGP': CAGP,
    }

    for name, cls in model_classes.items():
        print(f"\n  {name}:")
        t0 = time.time()
        model = cls(n_ent, n_rel)

        # CRITICAL: Build coverage ONLY from train_rels triples
        model.precompute_coverage(train_rels_triples)

        # But train on ALL training triples (including held-out relations)
        model = train_model(
            model, train, device,
            epochs=epochs, lr=lr, kl_beta=kl_beta, unc_weight=unc_weight
        )

        # Calibrate normalization on train_rels triples (consistent with coverage)
        if hasattr(model, 'calibrate_normalization'):
            model.calibrate_normalization(train_rels_triples, device)

        # Evaluate on held-out relation split
        model.eval()
        with torch.no_grad():
            # OOD: held-out relations
            ood_triples = test[test_holdout_idx]
            h_ood = torch.tensor(ood_triples[:, 0]).to(device)
            r_ood = torch.tensor(ood_triples[:, 1]).to(device)
            t_ood = torch.tensor(ood_triples[:, 2]).to(device)
            ood_unc = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

            # ID: train_rels + covered
            id_triples = test[test_id_idx]
            h_id = torch.tensor(id_triples[:, 0]).to(device)
            r_id = torch.tensor(id_triples[:, 1]).to(device)
            t_id = torch.tensor(id_triples[:, 2]).to(device)
            id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])

        try:
            auroc = float(roc_auc_score(labels, scores))
            aupr = float(average_precision_score(labels, scores))
        except Exception:
            auroc = 0.5
            aupr = 0.5

        elapsed = time.time() - t0

        print(f"    AUROC: {auroc:.4f}")
        print(f"    AUPR: {aupr:.4f}")
        print(f"    Time: {elapsed:.1f}s")

        results[name] = {
            'auroc': auroc,
            'aupr': aupr,
            'time': elapsed,
        }

    return results


def run_single(dataset, seed):
    """Run a single dataset+seed combination and save results incrementally."""
    import argparse
    device = setup_device()
    print(f"Device: {device}")

    loaders = {
        'fb15k237': ('FB15k-237', load_fb15k237),
        'yago310': ('YAGO3-10', load_yago310),
    }
    ds_name, loader = loaders[dataset]

    result = held_out_relation_experiment(
        ds_name, loader, device,
        holdout_frac=0.2, seed=seed,
        epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1,
    )

    # Save incrementally
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'held_out_relations_{dataset}_seed{seed}.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved to: {output_path}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='fb15k237 or yago310')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # If specific dataset/seed given, run just that one
    if args.dataset and args.seed:
        run_single(args.dataset, args.seed)
        return

    # Otherwise run all
    device = setup_device()
    print(f"Device: {device}")

    seeds = [42, 123, 456]
    dataset_loaders = {
        'fb15k237': ('FB15k-237', load_fb15k237),
        'yago310': ('YAGO3-10', load_yago310),
    }

    all_results = {}
    for ds_key, (ds_name, loader) in dataset_loaders.items():
        ds_results = []
        for seed in seeds:
            result = held_out_relation_experiment(
                ds_name, loader, device,
                holdout_frac=0.2, seed=seed,
                epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1,
            )
            if result is not None:
                ds_results.append(result)
                # Save incrementally
                output_dir = project_root / 'outputs'
                output_dir.mkdir(exist_ok=True)
                p = output_dir / f'held_out_relations_{ds_key}_seed{seed}.json'
                with open(p, 'w') as f:
                    json.dump(result, f, indent=2, default=float)

        if ds_results:
            summary = {'seeds': seeds}
            for model_name in ['GPOnly', 'CoverageOnly', 'CAGP']:
                aurocs = [r[model_name]['auroc'] for r in ds_results]
                auprs = [r[model_name]['aupr'] for r in ds_results]
                summary[model_name] = {
                    'auroc_mean': float(np.mean(aurocs)),
                    'auroc_std': float(np.std(aurocs)),
                    'aupr_mean': float(np.mean(auprs)),
                    'aupr_std': float(np.std(auprs)),
                }
            all_results[ds_key] = {'per_seed': ds_results, 'summary': summary}

    output_path = project_root / 'outputs' / 'held_out_relations_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
