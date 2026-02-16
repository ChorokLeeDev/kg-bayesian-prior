#!/usr/bin/env python3
"""
WN18RR Temporal OOD Experiments for UAI 2026

Runs all models (UKGE, Energy, GPOnly, CoverageOnly, CAGP, RelCondVar)
on WN18RR with temporal-like OOD evaluation (25th percentile threshold).
Reports mean ± std over 3 seeds.

Also runs FB15k-237 for RelCondVar (missing from current tables).
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

from src.data.loaders import (
    load_fb15k237,
    load_wn18rr,
    load_icews18,
    load_yago310,
)


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions (from run_focused_experiments.py)
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
        self._norm_stats = None  # cached normalization statistics

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
        """Compute normalization statistics from a reference set (e.g., training set).
        Must be called before get_uncertainty() during evaluation."""
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
        # Use cached normalization stats if available, else fall back to batch stats
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
    def __init__(self, num_entities, num_relations, dim=100, use_reparam=False):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.use_reparam = use_reparam
        self._norm_stats = None  # cached normalization statistics

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
        h_emb = self.entity_mean[h]
        t_emb = self.entity_mean[t]

        if self.training and self.use_reparam:
            h_scale = torch.sqrt(self.get_entity_relation_var(h, r))
            t_scale = torch.sqrt(self.get_entity_relation_var(t, r))
            h_noise = torch.randn_like(h_emb)
            t_noise = torch.randn_like(t_emb)
            h_emb = h_emb + h_scale.unsqueeze(-1) * h_noise
            t_emb = t_emb + t_scale.unsqueeze(-1) * t_noise

        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def get_entity_relation_var(self, e, r):
        e_emb = self.entity_mean[e]
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        raw = self.var_net(combined).squeeze(-1)
        base_var = torch.exp(self.entity_base_logvar[e])
        return F.softplus(raw) + base_var * 0.1 + 1e-4

    def calibrate_normalization(self, triples, device):
        """Compute normalization statistics from a reference set."""
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


def evaluate_ood(model, test, n_ent, device):
    """Evaluate standard OOD (random corruption)."""
    model.eval()
    test = test[:min(len(test), 5000)]

    ood_t = np.random.randint(0, n_ent, len(test))

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t_id = torch.tensor(test[:, 2]).to(device)
        t_ood = torch.tensor(ood_t).to(device)

        id_unc = model.get_uncertainty(h, r, t_id).cpu().numpy()
        ood_unc = model.get_uncertainty(h, r, t_ood).cpu().numpy()

    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])

    return roc_auc_score(labels, scores)


def _is_emerging(h_freq, t_freq, thresh, emerging_operator):
    if emerging_operator == 'lt':
        return h_freq < thresh or t_freq < thresh
    if emerging_operator == 'leq':
        return h_freq <= thresh or t_freq <= thresh
    raise ValueError(f"Unsupported emerging_operator: {emerging_operator}")


def evaluate_temporal(model, train, test, n_ent, device, emerging_operator='leq'):
    """Temporal-like OOD evaluation with 25th percentile threshold."""
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
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, emerging_operator):
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
            # Use full evaluation set to avoid class-ratio distortions.
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

    # Per-category: novel context vs ID
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


def run_dataset(
    ds_name,
    loader,
    device,
    models=None,
    seeds=None,
    epochs=30,
    lr=0.001,
    kl_beta=0.001,
    unc_weight=0.1,
    emerging_operator='leq',
    relcondvar_reparam=False,
):
    """Run all models on a dataset across multiple seeds."""
    if seeds is None:
        seeds = [42, 123, 456]

    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    if models is None:
        models = ['UKGE', 'Energy', 'GPOnly', 'CoverageOnly', 'CAGP', 'RelCondVar']
    else:
        models = [m for m in models if m]

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

        for name in models:
            cls = model_classes[name]
            print(f"\n  {name}:")
            t0 = time.time()
            if name == 'RelCondVar':
                model = cls(n_ent, n_rel, use_reparam=relcondvar_reparam)
            else:
                model = cls(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=epochs, lr=lr, kl_beta=kl_beta, unc_weight=unc_weight)

            # Calibrate normalization on training set (consistent with what the
            # model saw during training, avoids train/eval mismatch for learned alpha)
            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)

            # Standard OOD
            random_auroc = evaluate_ood(model, test, n_ent, device)
            print(f"    Random OOD AUROC: {random_auroc:.4f}")

            # Temporal OOD
            temporal = evaluate_temporal(
                model, train, test, n_ent, device, emerging_operator=emerging_operator
            )
            elapsed = time.time() - t0

            if 'overall_auroc' in temporal:
                print(f"    Temporal OOD AUROC: {temporal['overall_auroc']:.4f}")
            if 'emerging_auroc' in temporal:
                print(f"    Emerging AUROC: {temporal['emerging_auroc']:.4f}")
            if 'novel_ctx_auroc' in temporal:
                print(f"    Novel Ctx AUROC: {temporal['novel_ctx_auroc']:.4f}")
            print(f"    Time: {elapsed:.1f}s")

            seed_results[name] = {
                'random_auroc': float(random_auroc),
                'temporal': temporal,
            }

        all_seed_results[f'seed_{seed}'] = seed_results

    # Compute summary statistics
    summary = {}
    model_names = list(all_seed_results[f'seed_{seeds[0]}'].keys())
    for name in model_names:
        random_aurocs = [all_seed_results[f'seed_{s}'][name]['random_auroc'] for s in seeds]
        summary[name] = {
            'random_auroc_mean': float(np.mean(random_aurocs)),
            'random_auroc_std': float(np.std(random_aurocs)),
        }

        # Temporal overall
        temporal_aurocs = []
        temporal_auprs = []
        for s in seeds:
            t = all_seed_results[f'seed_{s}'][name]['temporal']
            if 'overall_auroc' in t:
                temporal_aurocs.append(t['overall_auroc'])
            if 'overall_aupr' in t:
                temporal_auprs.append(t['overall_aupr'])

        if temporal_aurocs:
            summary[name]['temporal_auroc_mean'] = float(np.mean(temporal_aurocs))
            summary[name]['temporal_auroc_std'] = float(np.std(temporal_aurocs))
        if temporal_auprs:
            summary[name]['temporal_aupr_mean'] = float(np.mean(temporal_auprs))
            summary[name]['temporal_aupr_std'] = float(np.std(temporal_auprs))

        # Per-category
        for key in ['emerging_auroc', 'novel_ctx_auroc']:
            vals = []
            for s in seeds:
                t = all_seed_results[f'seed_{s}'][name]['temporal']
                if key in t:
                    vals.append(t[key])
            if vals:
                summary[name][f'{key}_mean'] = float(np.mean(vals))
                summary[name][f'{key}_std'] = float(np.std(vals))

    all_seed_results['summary'] = summary
    all_seed_results['config'] = {
        'seeds': list(seeds),
        'epochs': int(epochs),
        'lr': float(lr),
        'kl_beta': float(kl_beta),
        'unc_weight': float(unc_weight),
        'emerging_operator': emerging_operator,
    }

    return all_seed_results


def main():
    parser = argparse.ArgumentParser(description="Temporal OOD experiments on WN18RR/FB15k-237/ICEWS18.")
    parser.add_argument(
        '--datasets',
        type=str,
        default='wn18rr,fb15k237',
        help="Comma-separated datasets: wn18rr,fb15k237,icews18.",
    )
    parser.add_argument(
        '--models',
        type=str,
        default='UKGE,Energy,GPOnly,CoverageOnly,CAGP,RelCondVar',
        help="Comma-separated models to evaluate.",
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='42,123,456',
        help="Comma-separated integer seeds.",
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kl-beta', type=float, default=1e-3)
    parser.add_argument('--unc-weight', type=float, default=0.1,
                        help="Weight for uncertainty margin loss (pos_unc < neg_unc).")
    parser.add_argument(
        '--relcondvar-reparam',
        action='store_true',
        help="Enable reparameterization sampling in RelCondVar scoring during training.",
    )
    parser.add_argument(
        '--emerging-operator',
        choices=['leq', 'lt'],
        default='leq',
        help="Threshold rule for emerging entities: leq uses <= tau, lt uses < tau.",
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(project_root / 'outputs' / 'wn18rr_temporal_results.json'),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    results = {}
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    requested = [d.strip().lower() for d in args.datasets.split(',') if d.strip()]
    requested_models = [m.strip() for m in args.models.split(',') if m.strip()]

    dataset_loaders = {
        'wn18rr': ('WN18RR', load_wn18rr),
        'fb15k237': ('FB15k-237', load_fb15k237),
        'icews18': ('ICEWS18', load_icews18),
        'yago310': ('YAGO3-10', load_yago310),
    }

    for ds in requested:
        if ds not in dataset_loaders:
            raise ValueError(f"Unknown dataset '{ds}'. Valid options: {', '.join(dataset_loaders)}")
        pretty_name, loader = dataset_loaders[ds]
        results[ds] = run_dataset(
            pretty_name,
            loader,
            device,
            models=requested_models,
            seeds=seeds,
            epochs=args.epochs,
            lr=args.lr,
            kl_beta=args.kl_beta,
            unc_weight=args.unc_weight,
            emerging_operator=args.emerging_operator,
            relcondvar_reparam=args.relcondvar_reparam,
        )

    # Save results
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {out}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for ds in results:
        print(f"\n{ds}:")
        s = results[ds]['summary']
        print(f"  {'Method':<15} {'Random AUROC':>14} {'Temporal AUROC':>16} {'Emerging':>12} {'Novel Ctx':>12}")
        print(f"  {'-'*15} {'-'*14} {'-'*16} {'-'*12} {'-'*12}")
        for name in s:
            random_str = f"{s[name]['random_auroc_mean']:.3f}±{s[name]['random_auroc_std']:.3f}"
            temp_str = f"{s[name].get('temporal_auroc_mean', 0):.3f}±{s[name].get('temporal_auroc_std', 0):.3f}" if 'temporal_auroc_mean' in s[name] else "N/A"
            emerge_str = f"{s[name].get('emerging_auroc_mean', 0):.3f}" if 'emerging_auroc_mean' in s[name] else "N/A"
            novel_str = f"{s[name].get('novel_ctx_auroc_mean', 0):.3f}" if 'novel_ctx_auroc_mean' in s[name] else "N/A"
            print(f"  {name:<15} {random_str:>14} {temp_str:>16} {emerge_str:>12} {novel_str:>12}")


if __name__ == "__main__":
    main()
