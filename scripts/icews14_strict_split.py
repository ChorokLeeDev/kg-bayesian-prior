#!/usr/bin/env python3
"""
ICEWS14 Strict Split Experiment for UAI 2026

Defense against "transductive artifact" criticism:
ICEWS14 has ~53% inverse-relation overlap and ~28% exact triple repetition
between train and test. This script removes both, creating a strict test set,
then re-evaluates all 6 models to show CAGP results hold.

Protocol:
1. Load ICEWS14 train/test (original chronological split)
2. Remove from test:
   - Exact duplicates: test triples (h,r,t) that appear in train
   - Inverse overlaps: test triples (h,r,t) where (t,r',h) exists in train for ANY r'
3. Train models on FULL original training set (unchanged)
4. Evaluate on both original and strict test sets
5. Report comparison table

Reports mean +/- std over 3 seeds.
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

from src.data.loaders import load_icews14


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions (same as run_icews14_temporal.py)
# ============================================================

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
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

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
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def calibrate_normalization(self, triples, device):
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


class RelCondVar(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self._norm_stats = None
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        nn.init.zeros_(self.var_net[-1].weight)
        nn.init.constant_(self.var_net[-1].bias, -1.0)
        self.entity_base_logvar = nn.Parameter(torch.zeros(num_entities) - 1.0)

    def forward(self, h, r, t):
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_entity_relation_var(self, e, r):
        e_emb = self.entity_mean[e]
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        raw = self.var_net(combined).squeeze(-1)
        base_var = torch.exp(self.entity_base_logvar[e])
        return F.softplus(raw) + base_var * 0.1 + 1e-4

    def calibrate_normalization(self, triples, device):
        with torch.no_grad():
            h = torch.tensor(triples[:, 0]).to(device)
            r = torch.tensor(triples[:, 1]).to(device)
            t = torch.tensor(triples[:, 2]).to(device)
            h_var = self.get_entity_relation_var(h, r)
            t_var = self.get_entity_relation_var(t, r)
            semantic = (h_var + t_var) / 2
            cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
            self._norm_stats = {
                'sem_mean': semantic.mean().item(),
                'cov_mean': cov_unc.mean().item(),
            }

    def get_uncertainty(self, h, r, t):
        h_var = self.get_entity_relation_var(h, r)
        t_var = self.get_entity_relation_var(t, r)
        semantic = (h_var + t_var) / 2
        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
        if self._norm_stats is not None:
            sem_mean = self._norm_stats['sem_mean']
            cov_mean = self._norm_stats['cov_mean']
        else:
            sem_mean = semantic.mean().item()
            cov_mean = cov_unc.mean().item()
        semantic_norm = semantic / (sem_mean + 1e-8) * (cov_mean + 1e-8)
        return 0.5 * semantic_norm + 0.5 * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class EnergyBased(nn.Module):
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
        return -self.forward(h, r, t)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class UKGE(nn.Module):
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
        scores = self.forward(h, r, t)
        probs = torch.sigmoid(scores)
        confidence = torch.abs(probs - 0.5) * 2
        return 1 - confidence

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Strict split construction
# ============================================================

def build_strict_test(train, test):
    """
    Remove from test set:
    1. Exact duplicates: (h,r,t) in test that also appear in train
    2. Inverse overlaps: (h,r,t) in test where (t,r',h) exists in train for ANY r'

    Returns: strict_test (numpy array), stats dict
    """
    # Build set of train triples for exact match
    train_triple_set = set()
    for i in range(len(train)):
        h, r, t = int(train[i, 0]), int(train[i, 1]), int(train[i, 2])
        train_triple_set.add((h, r, t))

    # Build set of (h, t) pairs in train for inverse match
    # If (t, r', h) exists in train, then train has the pair (t, h)
    # So for a test triple (h, r, t), we check if (t, h) exists as (head, tail) in train
    train_ht_pairs = set()
    for i in range(len(train)):
        h, r, t = int(train[i, 0]), int(train[i, 1]), int(train[i, 2])
        train_ht_pairs.add((h, t))

    n_exact = 0
    n_inverse = 0
    n_both = 0  # counted as exact
    keep_mask = np.ones(len(test), dtype=bool)

    for i in range(len(test)):
        h, r, t = int(test[i, 0]), int(test[i, 1]), int(test[i, 2])

        is_exact = (h, r, t) in train_triple_set
        # Inverse: does (t, ?, h) exist in train? Check if (t, h) is in train_ht_pairs
        is_inverse = (t, h) in train_ht_pairs

        if is_exact and is_inverse:
            n_both += 1
            keep_mask[i] = False
            n_exact += 1  # count under exact
        elif is_exact:
            n_exact += 1
            keep_mask[i] = False
        elif is_inverse:
            n_inverse += 1
            keep_mask[i] = False

    strict_test = test[keep_mask]

    stats = {
        'original_test_size': len(test),
        'removed_exact_only': n_exact - n_both,
        'removed_inverse_only': n_inverse,
        'removed_both': n_both,
        'removed_total': int((~keep_mask).sum()),
        'strict_test_size': len(strict_test),
        'pct_removed': 100.0 * (~keep_mask).sum() / len(test),
    }

    return strict_test, stats


# ============================================================
# Training and evaluation
# ============================================================

def train_model(model, triples, device, epochs=30, lr=0.001):
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

            if hasattr(model, 'entity_logvar') or hasattr(model, 'var_net'):
                pos_unc = model.get_uncertainty(h, r, t)
                neg_unc = model.get_uncertainty(h, r, neg_t)
                unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
                loss = loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


def evaluate_temporal_real(model, train, test, n_ent, device, label=""):
    """
    Temporal OOD evaluation using REAL temporal split.
    Same protocol as run_icews14_temporal.py.
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
        if freq.get(h, 0) <= thresh or freq.get(t, 0) <= thresh:
            new_entity_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}Split: emerging={len(new_entity_idx)}, novel_ctx={len(new_pair_idx)}, id={len(id_idx)}")

    results = {
        'n_emerging': len(new_entity_idx),
        'n_novel_ctx': len(new_pair_idx),
        'n_id': len(id_idx),
        'threshold': float(thresh),
    }

    # Overall temporal OOD: emerging + novel_ctx vs ID
    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            ood_sample = ood_idx[:min(len(ood_idx), 5000)]
            id_sample = id_idx[:min(len(id_idx), 5000)]

            ood_triples = test[ood_sample]
            id_triples = test[id_sample]

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
    else:
        print(f"{prefix}WARNING: Not enough samples (ood={len(ood_idx)}, id={len(id_idx)})")
        results['overall_auroc'] = float('nan')
        results['overall_aupr'] = float('nan')

    # Per-category: emerging vs ID
    if len(new_entity_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            emerge_sample = new_entity_idx[:min(len(new_entity_idx), 3000)]
            id_sample2 = id_idx[:min(len(id_idx), 3000)]

            e_triples = test[emerge_sample]
            i_triples = test[id_sample2]

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

    # Per-category: novel context vs ID
    if len(new_pair_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            novel_sample = new_pair_idx[:min(len(new_pair_idx), 3000)]
            id_sample3 = id_idx[:min(len(id_idx), 3000)]

            n_triples = test[novel_sample]
            i_triples = test[id_sample3]

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

    return results


def main():
    device = setup_device()
    print(f"Device: {device}")
    print(f"\n{'='*70}")
    print(f"  ICEWS14 Strict Split - Transductive Artifact Defense")
    print(f"{'='*70}")

    # Load data
    train_ds, _, test_ds = load_icews14()
    train = train_ds.triples
    test_orig = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test (original): {len(test_orig)}")

    # Build strict test set
    print(f"\n--- Building Strict Test Set ---")
    test_strict, split_stats = build_strict_test(train, test_orig)

    print(f"Original test size:    {split_stats['original_test_size']}")
    print(f"Removed exact only:    {split_stats['removed_exact_only']}")
    print(f"Removed inverse only:  {split_stats['removed_inverse_only']}")
    print(f"Removed both:          {split_stats['removed_both']}")
    print(f"Removed total:         {split_stats['removed_total']} ({split_stats['pct_removed']:.1f}%)")
    print(f"Strict test size:      {split_stats['strict_test_size']}")

    # Run experiments
    seeds = [42, 123, 456]
    all_results = {'original': {}, 'strict': {}}

    model_classes = {
        'UKGE': UKGE,
        'Energy': EnergyBased,
        'GPOnly': GPOnly,
        'CoverageOnly': CoverageOnly,
        'CAGP': CAGP,
        'RelCondVar': RelCondVar,
    }

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  Seed {seed}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, cls in model_classes.items():
            print(f"\n  {name}:")
            t0 = time.time()

            # Create and train model (on FULL original training set)
            model = cls(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30)

            # Calibrate on training set (not test) to avoid leakage
            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)

            # Evaluate on ORIGINAL test set
            print(f"    Evaluating on original test set...")
            res_orig = evaluate_temporal_real(model, train, test_orig, n_ent, device, label="orig")

            # Evaluate on STRICT test set (same trained model)
            print(f"    Evaluating on strict test set...")
            res_strict = evaluate_temporal_real(model, train, test_strict, n_ent, device, label="strict")

            elapsed = time.time() - t0

            if 'overall_auroc' in res_orig:
                orig_auroc = res_orig['overall_auroc']
                strict_auroc = res_strict.get('overall_auroc', float('nan'))
                print(f"    Original AUROC: {orig_auroc:.4f}")
                print(f"    Strict AUROC:   {strict_auroc:.4f}")
                if not np.isnan(orig_auroc) and not np.isnan(strict_auroc):
                    delta = strict_auroc - orig_auroc
                    print(f"    Delta:          {delta:+.4f}")
            print(f"    Time: {elapsed:.1f}s")

            # Store results
            seed_key = f'seed_{seed}'
            if seed_key not in all_results['original']:
                all_results['original'][seed_key] = {}
                all_results['strict'][seed_key] = {}
            all_results['original'][seed_key][name] = res_orig
            all_results['strict'][seed_key][name] = res_strict

    # Compute summary statistics
    print(f"\n\n{'='*80}")
    print("COMPUTING SUMMARY STATISTICS")
    print(f"{'='*80}")

    summary = {'original': {}, 'strict': {}}
    for split_name in ['original', 'strict']:
        for name in model_classes:
            summary[split_name][name] = {}
            for key in ['overall_auroc', 'overall_aupr', 'emerging_auroc', 'novel_ctx_auroc']:
                vals = []
                for s in seeds:
                    seed_key = f'seed_{s}'
                    if seed_key in all_results[split_name]:
                        res = all_results[split_name][seed_key].get(name, {})
                        if key in res and not np.isnan(res[key]):
                            vals.append(res[key])
                if vals:
                    summary[split_name][name][f'{key}_mean'] = float(np.mean(vals))
                    summary[split_name][name][f'{key}_std'] = float(np.std(vals))

    # Print comparison table
    print(f"\n{'='*90}")
    print("ICEWS14: ORIGINAL vs STRICT SPLIT COMPARISON")
    print(f"{'='*90}")
    print(f"  Strict split removes: exact duplicates + inverse-relation overlaps from test")
    print(f"  Removed {split_stats['removed_total']}/{split_stats['original_test_size']} "
          f"({split_stats['pct_removed']:.1f}%) test triples")
    print()
    print(f"  {'Method':<15} {'Original AUROC':>18} {'Strict AUROC':>18} {'Delta':>10}")
    print(f"  {'-'*15} {'-'*18} {'-'*18} {'-'*10}")

    for name in model_classes:
        s_orig = summary['original'].get(name, {})
        s_strict = summary['strict'].get(name, {})

        orig_str = "N/A"
        strict_str = "N/A"
        delta_str = "N/A"

        if 'overall_auroc_mean' in s_orig:
            orig_str = f"{s_orig['overall_auroc_mean']:.3f}+/-{s_orig['overall_auroc_std']:.3f}"
        if 'overall_auroc_mean' in s_strict:
            strict_str = f"{s_strict['overall_auroc_mean']:.3f}+/-{s_strict['overall_auroc_std']:.3f}"

        if 'overall_auroc_mean' in s_orig and 'overall_auroc_mean' in s_strict:
            delta = s_strict['overall_auroc_mean'] - s_orig['overall_auroc_mean']
            delta_str = f"{delta:+.3f}"

        print(f"  {name:<15} {orig_str:>18} {strict_str:>18} {delta_str:>10}")

    # Print per-category breakdown for strict split
    print(f"\n  Per-category breakdown (strict split):")
    print(f"  {'Method':<15} {'Emerging':>12} {'Novel Ctx':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for name in model_classes:
        s = summary['strict'].get(name, {})
        emerge = f"{s.get('emerging_auroc_mean', 0):.3f}" if 'emerging_auroc_mean' in s else "N/A"
        novel = f"{s.get('novel_ctx_auroc_mean', 0):.3f}" if 'novel_ctx_auroc_mean' in s else "N/A"
        print(f"  {name:<15} {emerge:>12} {novel:>12}")

    # Save results
    output = {
        'split_stats': split_stats,
        'summary': summary,
        'all_results': all_results,
        'dataset_info': {
            'name': 'ICEWS14',
            'num_entities': int(n_ent),
            'num_relations': int(n_rel),
            'train_triples': int(len(train)),
            'test_triples_original': int(len(test_orig)),
            'test_triples_strict': int(len(test_strict)),
        },
    }

    out_path = project_root / 'outputs' / 'icews14_strict_split_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
