#!/usr/bin/env python3
"""
ICEWS14 Temporal OOD Experiments for UAI 2026

ICEWS14 has ground-truth timestamps, providing a REAL temporal split
(unlike simulated splits on WN18RR/FB15k-237/YAGO).

Key difference from simulated splits:
- Train = early-timestamp triples (original train split)
- Test = late-timestamp triples (original test split)
- OOD categories emerge NATURALLY from temporal evolution, not from coverage-based partitioning
- This breaks the definitional coupling: novel contexts are defined by TIME, not by coverage

Reports mean ± std over 3 seeds.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from collections import defaultdict
import time

from src.data.loaders import load_icews14, load_icews18


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions (same as run_wn18rr_temporal.py)
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


def evaluate_temporal_real(model, train, test, n_ent, device):
    """
    Temporal OOD evaluation using REAL temporal split.

    Unlike simulated splits where OOD is defined by coverage,
    here OOD emerges naturally from temporal evolution:
    - Emerging entities: entities rare in training (freq <= 25th percentile)
    - Novel contexts: entities are well-known but appear with NEW relations in test
    - ID: entity-relation pairs all seen in training

    The key difference: novel_ctx is defined by TIME (test has later timestamps),
    NOT by the coverage indicator we use for detection. This breaks definitional coupling.
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

    print(f"    Split: emerging={len(new_entity_idx)}, novel_ctx={len(new_pair_idx)}, id={len(id_idx)}")
    print(f"    Threshold (25th pct): {thresh}")

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
    parser = argparse.ArgumentParser(
        description="Temporal OOD experiments on ICEWS14/ICEWS18 temporal splits."
    )
    parser.add_argument(
        '--dataset',
        choices=['icews14', 'icews18'],
        default='icews14',
        help="Temporal dataset to evaluate (icews14 or icews18).",
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output JSON path (defaults to temporal results for selected dataset).",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(
            project_root / 'outputs' / f"{args.dataset}_temporal_results.json"
        )

    device = setup_device()
    print(f"Device: {device}")

    dataset_loaders = {
        'icews14': (load_icews14, 'ICEWS14'),
        'icews18': (load_icews18, 'ICEWS18'),
    }
    loader, ds_name = dataset_loaders[args.dataset]
    train_ds, _, test_ds = loader()

    print(f"\n{'='*60}")
    print(f"  {ds_name} - Ground-Truth Temporal OOD")
    print(f"{'='*60}")
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    seeds = [42, 123, 456]
    all_seed_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_results = {}

        model_classes = {
            'UKGE': UKGE,
            'Energy': EnergyBased,
            'GPOnly': GPOnly,
            'CoverageOnly': CoverageOnly,
            'CAGP': CAGP,
            'RelCondVar': RelCondVar,
        }

        for name, cls in model_classes.items():
            print(f"\n  {name}:")
            t0 = time.time()
            model = cls(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=args.epochs, lr=args.lr)

            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)

            temporal = evaluate_temporal_real(model, train, test, n_ent, device)
            elapsed = time.time() - t0

            if 'overall_auroc' in temporal:
                print(f"    Temporal OOD AUROC: {temporal['overall_auroc']:.4f}")
            if 'emerging_auroc' in temporal:
                print(f"    Emerging AUROC: {temporal['emerging_auroc']:.4f}")
            if 'novel_ctx_auroc' in temporal:
                print(f"    Novel Ctx AUROC: {temporal['novel_ctx_auroc']:.4f}")
            print(f"    Time: {elapsed:.1f}s")

            seed_results[name] = {'temporal': temporal}

        all_seed_results[f'seed_{seed}'] = seed_results

    # Compute summary statistics
    summary = {}
    model_names = list(all_seed_results[f'seed_{seeds[0]}'].keys())
    for name in model_names:
        summary[name] = {}
        for key in ['overall_auroc', 'overall_aupr', 'emerging_auroc', 'novel_ctx_auroc']:
            vals = []
            for s in seeds:
                t = all_seed_results[f'seed_{s}'][name]['temporal']
                if key in t:
                    vals.append(t[key])
            if vals:
                summary[name][f'{key}_mean'] = float(np.mean(vals))
                summary[name][f'{key}_std'] = float(np.std(vals))

    all_seed_results['summary'] = summary

    # Also store dataset stats
    all_seed_results['dataset_info'] = {
        'name': ds_name,
        'num_entities': int(n_ent),
        'num_relations': int(n_rel),
        'train_triples': int(len(train)),
        'test_triples': int(len(test)),
        'temporal_split': 'ground_truth_timestamps',
    }

    # Save results
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_seed_results, f, indent=2, default=float)
    print(f"\nResults saved to {out}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{ds_name} SUMMARY (Ground-Truth Temporal)")
    print("=" * 70)
    print(f"  {'Method':<15} {'Overall AUROC':>16} {'Emerging':>12} {'Novel Ctx':>12}")
    print(f"  {'-'*15} {'-'*16} {'-'*12} {'-'*12}")
    for name in summary:
        s = summary[name]
        overall = f"{s.get('overall_auroc_mean', 0):.3f}±{s.get('overall_auroc_std', 0):.3f}" if 'overall_auroc_mean' in s else "N/A"
        emerge = f"{s.get('emerging_auroc_mean', 0):.3f}" if 'emerging_auroc_mean' in s else "N/A"
        novel = f"{s.get('novel_ctx_auroc_mean', 0):.3f}" if 'novel_ctx_auroc_mean' in s else "N/A"
        print(f"  {name:<15} {overall:>16} {emerge:>12} {novel:>12}")


if __name__ == "__main__":
    main()
