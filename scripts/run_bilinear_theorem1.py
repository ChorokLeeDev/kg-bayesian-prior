#!/usr/bin/env python3
"""
Bilinear Scoring Experiment for Theorem 1 Validation (UAI 2026)

Tests whether relation-conditioned bilinear scoring (h^T W_r t with basis
decomposition) improves OOD detection when uncertainty remains entity-level.

Key insight: relation-aware SCORING ≠ relation-aware UNCERTAINTY.
BilinearScoring uses per-relation W_r but entity-level variance → same AUROC
as GPOnly on novel contexts. Adding coverage (BilinearScoring_CAGP) recovers
to match standard CAGP.

Models: GPOnly, BilinearScoring, CAGP, BilinearScoring_CAGP
Datasets: WN18RR, FB15k-237, YAGO3-10
Seeds: 42, 123, 456
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

from src.data.loaders import load_fb15k237, load_wn18rr, load_yago310


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions
# ============================================================

class GPOnly(nn.Module):
    """Entity-level GP uncertainty with DistMult scoring."""
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
    """Coverage-Augmented GP-KGE with DistMult scoring."""
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


class BilinearScoring(nn.Module):
    """Relation-conditioned bilinear scorer with basis decomposition.

    Scoring: h^T W_r t  where W_r = sum_b coeff[r,b] * basis[b]
    Uncertainty: entity-level (SAME as GPOnly) — r is UNUSED.
    This is the key test: relation-aware scoring ≠ relation-aware uncertainty.
    """
    def __init__(self, num_entities, num_relations, dim=100, num_bases=10):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.num_bases = num_bases

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Basis decomposition for per-relation bilinear matrices
        self.bases = nn.Parameter(torch.randn(num_bases, dim * dim) * 0.01)
        self.coefficients = nn.Parameter(torch.randn(num_relations, num_bases) * 0.1)

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

        # W_r = coefficients[r] @ bases → (batch, dim*dim) → (batch, dim, dim)
        coeff = self.coefficients[r]  # (batch, num_bases)
        W_r = torch.mm(coeff, self.bases)  # (batch, dim*dim)
        W_r = W_r.view(-1, self.dim, self.dim)  # (batch, dim, dim)

        # h^T W_r t → bilinear scoring
        scores = torch.bmm(
            h_emb.unsqueeze(1),  # (batch, 1, dim)
            torch.bmm(W_r, t_emb.unsqueeze(2))  # (batch, dim, 1)
        ).squeeze(-1).squeeze(-1)  # (batch,)

        return scores

    def get_uncertainty(self, h, r, t):
        """Entity-level uncertainty — IDENTICAL to GPOnly.
        r is accepted but UNUSED. THIS IS THE POINT."""
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class BilinearScoring_CAGP(nn.Module):
    """Bilinear scoring + coverage augmentation.

    Same bilinear scorer as BilinearScoring, but uncertainty combines
    entity-level GP variance with coverage (like CAGP).
    """
    def __init__(self, num_entities, num_relations, dim=100, num_bases=10):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.num_bases = num_bases

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Basis decomposition for per-relation bilinear matrices
        self.bases = nn.Parameter(torch.randn(num_bases, dim * dim) * 0.01)
        self.coefficients = nn.Parameter(torch.randn(num_relations, num_bases) * 0.1)

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

        coeff = self.coefficients[r]
        W_r = torch.mm(coeff, self.bases)
        W_r = W_r.view(-1, self.dim, self.dim)

        scores = torch.bmm(
            h_emb.unsqueeze(1),
            torch.bmm(W_r, t_emb.unsqueeze(2))
        ).squeeze(-1).squeeze(-1)

        return scores

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


# ============================================================
# Training and evaluation (from run_wn18rr_temporal.py)
# ============================================================

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

    # Logvar spread diagnostic
    if hasattr(model, 'entity_logvar'):
        logvar_std = model.entity_logvar.data.std().item()
        print(f"    entity_logvar std: {logvar_std:.4f}")

    return model


def evaluate_ood(model, test, n_ent, device):
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


def _is_emerging(h_freq, t_freq, thresh):
    return h_freq <= thresh or t_freq <= thresh


def evaluate_temporal(model, train, test, n_ent, device):
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
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh):
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

    # Emerging vs ID
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

    # Novel context vs ID
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

    return results


# ============================================================
# Main experiment runner
# ============================================================

def run_dataset(ds_name, loader, device, seeds=None, epochs=30):
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

    all_seed_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_results = {}

        model_classes = {
            'GPOnly': GPOnly,
            'BilinearScoring': BilinearScoring,
            'CAGP': CAGP,
            'BilinearScoring_CAGP': BilinearScoring_CAGP,
        }

        for name, cls in model_classes.items():
            print(f"\n  {name}:")
            t0 = time.time()
            model = cls(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=epochs)

            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)

            # Standard OOD
            random_auroc = evaluate_ood(model, test, n_ent, device)
            print(f"    Random OOD AUROC: {random_auroc:.4f}")

            # Temporal OOD
            temporal = evaluate_temporal(model, train, test, n_ent, device)
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

        for key in ['overall_auroc', 'overall_aupr']:
            vals = [all_seed_results[f'seed_{s}'][name]['temporal'].get(key)
                    for s in seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                tkey = key.replace('overall_', 'temporal_') if 'overall' in key else key
                summary[name][f'{tkey}_mean'] = float(np.mean(vals))
                summary[name][f'{tkey}_std'] = float(np.std(vals))

        for key in ['emerging_auroc', 'novel_ctx_auroc']:
            vals = [all_seed_results[f'seed_{s}'][name]['temporal'].get(key)
                    for s in seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[name][f'{key}_mean'] = float(np.mean(vals))
                summary[name][f'{key}_std'] = float(np.std(vals))

    all_seed_results['summary'] = summary
    all_seed_results['config'] = {
        'seeds': list(seeds),
        'epochs': int(epochs),
    }

    return all_seed_results


def validate_results(results):
    """Run quality gates and theorem validation checks."""
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    all_pass = True

    for ds in results:
        if ds in ('config',):
            continue
        s = results[ds].get('summary', {})
        if not s:
            continue

        print(f"\n{ds}:")

        # Q1: Standard OOD AUROC > 0.55 for BilinearScoring
        if 'BilinearScoring' in s:
            val = s['BilinearScoring'].get('random_auroc_mean', 0)
            ok = val > 0.55
            print(f"  Q1 Standard OOD > 0.55: {val:.3f} {'PASS' if ok else 'FAIL'}")
            all_pass &= ok

        # T1: |novel_ctx(BilinearScoring) - novel_ctx(GPOnly)| <= 0.05
        gp_nc = s.get('GPOnly', {}).get('novel_ctx_auroc_mean')
        bl_nc = s.get('BilinearScoring', {}).get('novel_ctx_auroc_mean')
        if gp_nc is not None and bl_nc is not None:
            diff = abs(bl_nc - gp_nc)
            ok = diff <= 0.05
            print(f"  T1 |BilinScoring - GPOnly| novel_ctx: {diff:.3f} {'PASS' if ok else 'FAIL'}")
            all_pass &= ok

        # T2: Both GPOnly and BilinearScoring novel_ctx_auroc_mean <= 0.55
        for name in ['GPOnly', 'BilinearScoring']:
            nc = s.get(name, {}).get('novel_ctx_auroc_mean')
            if nc is not None:
                ok = nc <= 0.55
                print(f"  T2 {name} novel_ctx <= 0.55: {nc:.3f} {'PASS' if ok else 'FAIL'}")
                all_pass &= ok

        # C1: |temporal(BilinScoring_CAGP) - temporal(CAGP)| <= 0.03
        cagp_t = s.get('CAGP', {}).get('temporal_auroc_mean')
        bcagp_t = s.get('BilinearScoring_CAGP', {}).get('temporal_auroc_mean')
        if cagp_t is not None and bcagp_t is not None:
            diff = abs(bcagp_t - cagp_t)
            ok = diff <= 0.03
            print(f"  C1 |BilinScoring_CAGP - CAGP| temporal: {diff:.3f} {'PASS' if ok else 'FAIL'}")
            all_pass &= ok

        # C2: BilinScoring_CAGP novel_ctx >= 0.95
        bcagp_nc = s.get('BilinearScoring_CAGP', {}).get('novel_ctx_auroc_mean')
        if bcagp_nc is not None:
            ok = bcagp_nc >= 0.95
            print(f"  C2 BilinScoring_CAGP novel_ctx >= 0.95: {bcagp_nc:.3f} {'PASS' if ok else 'FAIL'}")
            all_pass &= ok

    # T3: Per-seed check (all individual novel_ctx_auroc <= 0.60)
    print(f"\n  T3 Per-seed novel_ctx checks:")
    for ds in results:
        if ds in ('config',):
            continue
        for seed_key in results[ds]:
            if not seed_key.startswith('seed_'):
                continue
            for name in ['GPOnly', 'BilinearScoring']:
                nc = results[ds][seed_key].get(name, {}).get('temporal', {}).get('novel_ctx_auroc')
                if nc is not None:
                    ok = nc <= 0.60
                    if not ok:
                        print(f"    {ds}/{seed_key}/{name} novel_ctx = {nc:.3f} FAIL")
                        all_pass = False

    status = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"\n  Overall: {status}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Bilinear scoring experiment for Theorem 1 validation."
    )
    parser.add_argument(
        '--datasets', type=str, default='wn18rr,fb15k237,yago310',
        help="Comma-separated datasets: wn18rr,fb15k237,yago310",
    )
    parser.add_argument(
        '--seeds', type=str, default='42,123,456',
        help="Comma-separated integer seeds.",
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument(
        '--output', type=str,
        default=str(project_root / 'outputs' / 'bilinear_theorem1_results.json'),
    )
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    results = {}
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    requested = [d.strip().lower() for d in args.datasets.split(',') if d.strip()]

    dataset_loaders = {
        'wn18rr': ('WN18RR', load_wn18rr),
        'fb15k237': ('FB15k-237', load_fb15k237),
        'yago310': ('YAGO3-10', load_yago310),
    }

    for ds in requested:
        if ds not in dataset_loaders:
            raise ValueError(f"Unknown dataset '{ds}'. Valid: {', '.join(dataset_loaders)}")
        pretty_name, loader = dataset_loaders[ds]
        results[ds] = run_dataset(pretty_name, loader, device, seeds=seeds, epochs=args.epochs)

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
        s = results[ds].get('summary', {})
        if not s:
            continue
        print(f"\n{ds}:")
        print(f"  {'Method':<22} {'Random':>8} {'Temporal':>10} {'Emerging':>10} {'Novel Ctx':>10}")
        print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for name in s:
            r_str = f"{s[name]['random_auroc_mean']:.3f}"
            t_str = f"{s[name].get('temporal_auroc_mean', 0):.3f}" if 'temporal_auroc_mean' in s[name] else "N/A"
            e_str = f"{s[name].get('emerging_auroc_mean', 0):.3f}" if 'emerging_auroc_mean' in s[name] else "N/A"
            n_str = f"{s[name].get('novel_ctx_auroc_mean', 0):.3f}" if 'novel_ctx_auroc_mean' in s[name] else "N/A"
            print(f"  {name:<22} {r_str:>8} {t_str:>10} {e_str:>10} {n_str:>10}")

    # Run validation checks
    validate_results(results)


if __name__ == "__main__":
    main()
