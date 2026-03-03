#!/usr/bin/env python3
"""
ICEWS14 Strict Split Experiment (5 Seeds) for NeurIPS Submission
Optimized for rapid convergence: 5 epochs, 2000 sample eval

Usage: python scripts/icews14_strict_split_5seed.py
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
import warnings
warnings.filterwarnings('ignore')

from src.data.loaders import load_icews14

torch.set_num_threads(1)


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
            h = torch.tensor(triples[:, 0], device=device)
            r = torch.tensor(triples[:, 1], device=device)
            t = torch.tensor(triples[:, 2], device=device)
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
        self.var_net = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, 1))
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
            h = torch.tensor(triples[:, 0], device=device)
            r = torch.tensor(triples[:, 1], device=device)
            t = torch.tensor(triples[:, 2], device=device)
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


def build_strict_test(train, test):
    """Remove exact duplicates and inverse overlaps from test set."""
    train_triple_set = set()
    for i in range(len(train)):
        h, r, t = int(train[i, 0]), int(train[i, 1]), int(train[i, 2])
        train_triple_set.add((h, r, t))

    train_ht_pairs = set()
    for i in range(len(train)):
        h, r, t = int(train[i, 0]), int(train[i, 1]), int(train[i, 2])
        train_ht_pairs.add((h, t))

    n_exact, n_inverse, n_both = 0, 0, 0
    keep_mask = np.ones(len(test), dtype=bool)

    for i in range(len(test)):
        h, r, t = int(test[i, 0]), int(test[i, 1]), int(test[i, 2])
        is_exact = (h, r, t) in train_triple_set
        is_inverse = (t, h) in train_ht_pairs

        if is_exact and is_inverse:
            n_both += 1
            keep_mask[i] = False
            n_exact += 1
        elif is_exact:
            n_exact += 1
            keep_mask[i] = False
        elif is_inverse:
            n_inverse += 1
            keep_mask[i] = False

    strict_test = test[keep_mask]
    return strict_test, {
        'original_test_size': len(test),
        'removed_exact_only': n_exact - n_both,
        'removed_inverse_only': n_inverse,
        'removed_both': n_both,
        'removed_total': int((~keep_mask).sum()),
        'strict_test_size': len(strict_test),
        'pct_removed': 100.0 * (~keep_mask).sum() / len(test),
    }


def train_model(model, triples, device, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    heads, rels, tails = torch.tensor(triples[:, 0]), torch.tensor(triples[:, 1]), torch.tensor(triples[:, 2])
    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)
            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            if hasattr(model, 'entity_logvar') or hasattr(model, 'var_net'):
                pos_unc = model.get_uncertainty(h, r, t)
                neg_unc = model.get_uncertainty(h, r, neg_t)
                unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
                loss = loss + 0.1 * unc_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def evaluate_temporal_real(model, train, test, n_ent, device, max_samples=1500):
    """Temporal OOD evaluation."""
    model.eval()
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()
    
    new_entity_idx, new_pair_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if freq.get(h, 0) <= thresh or freq.get(t, 0) <= thresh:
            new_entity_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    results = {'n_emerging': len(new_entity_idx), 'n_novel_ctx': len(new_pair_idx), 'n_id': len(id_idx)}

    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            ood_sample = np.random.choice(len(ood_idx), min(len(ood_idx), max_samples), replace=False)
            id_sample = np.random.choice(len(id_idx), min(len(id_idx), max_samples), replace=False)
            ood_triples = test[[ood_idx[i] for i in ood_sample]]
            id_triples = test[[id_idx[i] for i in id_sample]]
            
            h_ood = torch.tensor(ood_triples[:, 0], device=device)
            r_ood = torch.tensor(ood_triples[:, 1], device=device)
            t_ood = torch.tensor(ood_triples[:, 2], device=device)
            ood_unc = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()
            
            h_id = torch.tensor(id_triples[:, 0], device=device)
            r_id = torch.tensor(id_triples[:, 1], device=device)
            t_id = torch.tensor(id_triples[:, 2], device=device)
            id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])
        try:
            results['overall_auroc'] = float(roc_auc_score(labels, scores))
            results['overall_aupr'] = float(average_precision_score(labels, scores))
        except:
            results['overall_auroc'], results['overall_aupr'] = 0.5, 0.5
    else:
        results['overall_auroc'], results['overall_aupr'] = float('nan'), float('nan')

    if len(new_entity_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            e_idx = new_entity_idx[:min(len(new_entity_idx), max_samples)]
            i_idx = id_idx[:min(len(id_idx), max_samples)]
            e_triples, i_triples = test[e_idx], test[i_idx]
            h_e = torch.tensor(e_triples[:, 0], device=device)
            r_e = torch.tensor(e_triples[:, 1], device=device)
            t_e = torch.tensor(e_triples[:, 2], device=device)
            e_unc = model.get_uncertainty(h_e, r_e, t_e).cpu().numpy()
            h_i = torch.tensor(i_triples[:, 0], device=device)
            r_i = torch.tensor(i_triples[:, 1], device=device)
            t_i = torch.tensor(i_triples[:, 2], device=device)
            i_unc = model.get_uncertainty(h_i, r_i, t_i).cpu().numpy()
        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(e_unc))])
        scores = np.concatenate([i_unc, e_unc])
        try:
            results['emerging_auroc'] = float(roc_auc_score(labels, scores))
        except:
            results['emerging_auroc'] = 0.5

    if len(new_pair_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            n_idx = new_pair_idx[:min(len(new_pair_idx), max_samples)]
            i_idx = id_idx[:min(len(id_idx), max_samples)]
            n_triples, i_triples = test[n_idx], test[i_idx]
            h_n = torch.tensor(n_triples[:, 0], device=device)
            r_n = torch.tensor(n_triples[:, 1], device=device)
            t_n = torch.tensor(n_triples[:, 2], device=device)
            n_unc = model.get_uncertainty(h_n, r_n, t_n).cpu().numpy()
            h_i = torch.tensor(i_triples[:, 0], device=device)
            r_i = torch.tensor(i_triples[:, 1], device=device)
            t_i = torch.tensor(i_triples[:, 2], device=device)
            i_unc = model.get_uncertainty(h_i, r_i, t_i).cpu().numpy()
        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(n_unc))])
        scores = np.concatenate([i_unc, n_unc])
        try:
            results['novel_ctx_auroc'] = float(roc_auc_score(labels, scores))
        except:
            results['novel_ctx_auroc'] = 0.5

    return results


def main():
    device = torch.device('cpu')
    print("\nICEWS14 Strict Split (5 Seeds) - NeurIPS Submission")
    print(f"{'='*60}")

    train_ds, _, test_ds = load_icews14()
    train, test_orig = train_ds.triples, test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}, Test: {len(test_orig)}")

    test_strict, split_stats = build_strict_test(train, test_orig)
    print(f"Removed {split_stats['removed_total']}/{split_stats['original_test_size']} "
          f"({split_stats['pct_removed']:.1f}%) test triples")

    seeds = [42, 123, 456, 789, 1024]
    all_results = {'original': {}, 'strict': {}}
    models = {'UKGE': UKGE, 'Energy': EnergyBased, 'GPOnly': GPOnly, 'CoverageOnly': CoverageOnly, 'CAGP': CAGP, 'RelCondVar': RelCondVar}

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/5)")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, cls in models.items():
            t0 = time.time()
            model = cls(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=5)
            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)
            res_orig = evaluate_temporal_real(model, train, test_orig, n_ent, device)
            res_strict = evaluate_temporal_real(model, train, test_strict, n_ent, device)
            
            seed_key = f'seed_{seed}'
            if seed_key not in all_results['original']:
                all_results['original'][seed_key] = {}
                all_results['strict'][seed_key] = {}
            all_results['original'][seed_key][name] = res_orig
            all_results['strict'][seed_key][name] = res_strict
            
            elapsed = time.time() - t0
            auroc_orig = res_orig.get('overall_auroc', float('nan'))
            auroc_strict = res_strict.get('overall_auroc', float('nan'))
            print(f"  {name:15s} orig={auroc_orig:.3f} strict={auroc_strict:.3f} ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    summary = {'original': {}, 'strict': {}}
    for split_name in ['original', 'strict']:
        for name in models:
            summary[split_name][name] = {}
            for key in ['overall_auroc', 'overall_aupr', 'emerging_auroc', 'novel_ctx_auroc']:
                vals = [all_results[split_name].get(f'seed_{s}', {}).get(name, {}).get(key) for s in seeds]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if vals:
                    summary[split_name][name][f'{key}_mean'] = float(np.mean(vals))
                    summary[split_name][name][f'{key}_std'] = float(np.std(vals))

    print("ICEWS14: ORIGINAL vs STRICT SPLIT COMPARISON (5 SEEDS)")
    print(f"{'Method':<15} {'Original AUROC':>20} {'Strict AUROC':>20} {'Delta':>10}")
    print(f"{'-'*15} {'-'*20} {'-'*20} {'-'*10}")
    for name in models:
        s_orig, s_strict = summary['original'].get(name, {}), summary['strict'].get(name, {})
        orig_str = f"{s_orig['overall_auroc_mean']:.3f}+/-{s_orig['overall_auroc_std']:.3f}" if 'overall_auroc_mean' in s_orig else "N/A"
        strict_str = f"{s_strict['overall_auroc_mean']:.3f}+/-{s_strict['overall_auroc_std']:.3f}" if 'overall_auroc_mean' in s_strict else "N/A"
        delta_str = f"{s_strict['overall_auroc_mean'] - s_orig['overall_auroc_mean']:+.3f}" if ('overall_auroc_mean' in s_orig and 'overall_auroc_mean' in s_strict) else "N/A"
        print(f"{name:<15} {orig_str:>20} {strict_str:>20} {delta_str:>10}")

    print("\nPer-category (strict split):")
    print(f"{'Method':<15} {'Emerging':>18} {'Novel Ctx':>18}")
    for name in models:
        s = summary['strict'].get(name, {})
        emerge = f"{s['emerging_auroc_mean']:.3f}+/-{s['emerging_auroc_std']:.3f}" if 'emerging_auroc_mean' in s else "N/A"
        novel = f"{s['novel_ctx_auroc_mean']:.3f}+/-{s['novel_ctx_auroc_std']:.3f}" if 'novel_ctx_auroc_mean' in s else "N/A"
        print(f"{name:<15} {emerge:>18} {novel:>18}")

    output = {
        'split_stats': split_stats,
        'summary': summary,
        'all_results': all_results,
        'seeds': seeds,
        'dataset_info': {
            'name': 'ICEWS14',
            'num_entities': int(n_ent),
            'num_relations': int(n_rel),
            'train_triples': int(len(train)),
            'test_triples_original': int(len(test_orig)),
            'test_triples_strict': int(len(test_strict)),
        },
    }

    out_path = project_root / 'outputs' / 'icews14_strict_split_5seed_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
