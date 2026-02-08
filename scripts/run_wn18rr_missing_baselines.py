#!/usr/bin/env python3
"""
Run MC Dropout, Deep Ensemble, and SNGP baselines on WN18RR temporal OOD.

These are the three '--' entries in Table 1 that need filling.
Uses the same temporal OOD evaluation as run_wn18rr_temporal.py.
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

from src.data.loaders import load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# MC Dropout baseline
# ============================================================

class MCDropoutKGE(nn.Module):
    """MC Dropout uncertainty via DistMult with dropout at inference."""
    def __init__(self, num_entities, num_relations, dim=100, dropout_rate=0.1, num_samples=20):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_samples = num_samples
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        h_emb = self.dropout(self.entity_emb(h))
        r_emb = self.relation_emb(r)
        t_emb = self.dropout(self.entity_emb(t))
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        # Enable dropout at inference time
        self.dropout.train()
        scores = []
        for _ in range(self.num_samples):
            s = self.forward(h, r, t)
            scores.append(s)
        self.dropout.eval()
        scores = torch.stack(scores, dim=0)  # (num_samples, batch)
        # Uncertainty = variance across MC samples
        return scores.var(dim=0)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Deep Ensemble baseline
# ============================================================

class DeepEnsembleKGE(nn.Module):
    """Deep Ensemble: train N independent DistMult models, use disagreement."""
    def __init__(self, num_entities, num_relations, dim=100, num_models=5):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_models = num_models

        # Create independent entity/relation embeddings per model
        self.entity_embs = nn.ModuleList([nn.Embedding(num_entities, dim) for _ in range(num_models)])
        self.relation_embs = nn.ModuleList([nn.Embedding(num_relations, dim) for _ in range(num_models)])
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward_single(self, h, r, t, idx):
        return (self.entity_embs[idx](h) * self.relation_embs[idx](r) * self.entity_embs[idx](t)).sum(-1)

    def forward(self, h, r, t):
        # Average score across ensemble
        scores = torch.stack([self.forward_single(h, r, t, i) for i in range(self.num_models)], dim=0)
        return scores.mean(dim=0)

    def get_uncertainty(self, h, r, t):
        scores = torch.stack([self.forward_single(h, r, t, i) for i in range(self.num_models)], dim=0)
        return scores.var(dim=0)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# SNGP baseline (simplified, compatible interface)
# ============================================================

class SNGPBaseline(nn.Module):
    """SNGP: spectral-normalized feature extractor + GP output layer."""
    def __init__(self, num_entities, num_relations, dim=100, num_rff=512):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Feature extractor with spectral normalization
        hidden = dim * 2
        self.feat_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(dim * 3, hidden)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )

        # Random Fourier Features
        self.register_buffer('rff_w', torch.randn(hidden, num_rff))
        self.register_buffer('rff_b', torch.rand(num_rff) * 2 * np.pi)
        self.rff_dim = num_rff * 2  # cos + sin

        # Precision matrix for GP
        self.register_buffer('precision', torch.eye(self.rff_dim))
        self.ridge = 1.0

        # Entity frequency
        self.register_buffer('entity_freq', torch.zeros(num_entities))
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def _rff(self, x):
        proj = x @ self.rff_w + self.rff_b
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * np.sqrt(2.0 / self.rff_w.shape[1])

    def _features(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        triple = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        hidden = self.feat_net(triple)
        return self._rff(hidden)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        features = self._features(h, r, t)
        # GP variance: phi^T Lambda^{-1} phi
        precision = self.precision + self.ridge * torch.eye(self.rff_dim, device=features.device)
        try:
            L = torch.linalg.cholesky(precision)
            solved = torch.cholesky_solve(features.unsqueeze(-1), L).squeeze(-1)
            gp_var = (features * solved).sum(dim=-1)
        except Exception:
            gp_var = features.norm(dim=-1)

        # Entity frequency component
        h_freq = self.entity_freq[h]
        t_freq = self.entity_freq[t]
        max_freq = self.entity_freq.max() + 1
        freq_unc = 2.0 - (h_freq / max_freq) - (t_freq / max_freq)

        gp_norm = gp_var / (gp_var.mean() + 1e-8)
        return gp_norm + 0.1 * freq_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0
            self.entity_freq[triples[i, 0]] += 1
            self.entity_freq[triples[i, 2]] += 1

    def fit_precision(self, triples, device, max_batches=50):
        """Fit precision matrix on training data after training."""
        self.eval()
        heads = torch.tensor(triples[:, 0])
        rels = torch.tensor(triples[:, 1])
        tails = torch.tensor(triples[:, 2])
        loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

        self.precision = torch.eye(self.rff_dim, device=device) * self.ridge
        with torch.no_grad():
            for i, (h, r, t) in enumerate(loader):
                if i >= max_batches:
                    break
                h, r, t = h.to(device), r.to(device), t.to(device)
                feat = self._features(h, r, t)
                batch_prec = feat.T @ feat / feat.shape[0]
                if i == 0:
                    self.precision = batch_prec
                else:
                    self.precision = (i / (i + 1)) * self.precision + (1 / (i + 1)) * batch_prec

            self.precision += self.ridge * torch.eye(self.rff_dim, device=device)


# ============================================================
# Training
# ============================================================

def train_model(model, triples, device, epochs=30, lr=0.001):
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

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

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


def train_ensemble(model, triples, device, epochs=30, lr=0.001):
    """Train each ensemble member independently with different data order."""
    model = model.to(device)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    for m_idx in range(model.num_models):
        print(f"    Training ensemble member {m_idx+1}/{model.num_models}")
        # Only optimize this member's parameters
        params = list(model.entity_embs[m_idx].parameters()) + list(model.relation_embs[m_idx].parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for h, r, t in loader:
                h, r, t = h.to(device), r.to(device), t.to(device)

                pos_scores = model.forward_single(h, r, t, m_idx)
                neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
                neg_scores = model.forward_single(h, r, neg_t, m_idx)

                loss = F.binary_cross_entropy_with_logits(
                    pos_scores, torch.ones_like(pos_scores)
                ) + F.binary_cross_entropy_with_logits(
                    neg_scores, torch.zeros_like(neg_scores)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                total_loss += loss.item()

    return model


def train_sngp(model, triples, device, epochs=30, lr=0.001):
    """Train SNGP with precision matrix updates."""
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

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

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

    # Fit precision matrix after training
    print("    Fitting precision matrix...")
    model.fit_precision(triples, device)
    return model


# ============================================================
# Temporal OOD evaluation (same as run_wn18rr_temporal.py)
# ============================================================

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
        if freq.get(h, 0) <= thresh or freq.get(t, 0) <= thresh:
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

    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            ood_sample = ood_idx[:min(len(ood_idx), 3000)]
            id_sample = id_idx[:min(len(id_idx), 3000)]

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
        except Exception:
            results['overall_auroc'] = 0.5

    return results


# ============================================================
# Also compute MRR/Hits@10 for the base model (for paper footnote)
# ============================================================

def evaluate_link_prediction(model, test, train, n_ent, device, max_test=1000):
    """Compute MRR and Hits@10 for base model link prediction."""
    model.eval()
    test_subset = test[:min(len(test), max_test)]

    # Build filter set (all known triples)
    all_triples = np.concatenate([train, test], axis=0)
    filter_set = set()
    for i in range(len(all_triples)):
        filter_set.add((int(all_triples[i, 0]), int(all_triples[i, 1]), int(all_triples[i, 2])))

    ranks = []
    with torch.no_grad():
        for i in range(len(test_subset)):
            h, r, t = int(test_subset[i, 0]), int(test_subset[i, 1]), int(test_subset[i, 2])

            # Score all entities as tails
            h_batch = torch.full((n_ent,), h, dtype=torch.long, device=device)
            r_batch = torch.full((n_ent,), r, dtype=torch.long, device=device)
            t_batch = torch.arange(n_ent, device=device)

            scores = model(h_batch, r_batch, t_batch).cpu().numpy()

            # Filter: set scores of known triples (except the test triple) to -inf
            for tt in range(n_ent):
                if tt != t and (h, r, tt) in filter_set:
                    scores[tt] = -1e9

            # Rank of correct tail
            rank = (scores >= scores[t]).sum()
            ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    mrr = float(np.mean(1.0 / ranks))
    hits10 = float(np.mean(ranks <= 10))
    hits1 = float(np.mean(ranks <= 1))

    return {'mrr': mrr, 'hits@10': hits10, 'hits@1': hits1}


# ============================================================
# Main
# ============================================================

def main():
    device = setup_device()
    print(f"Device: {device}")

    train_ds, _, test_ds = load_wn18rr()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"WN18RR: {n_ent} entities, {n_rel} relations, {len(train)} train, {len(test)} test")

    seeds = [42, 123, 456]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_results = {}

        # --- MC Dropout ---
        print("\n  MC Dropout:")
        t0 = time.time()
        model = MCDropoutKGE(n_ent, n_rel, dropout_rate=0.1, num_samples=20)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['MCDropout'] = temporal

        # --- Deep Ensemble ---
        print("\n  Deep Ensemble (5 models):")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepEnsembleKGE(n_ent, n_rel, num_models=5)
        model.precompute_coverage(train)
        model = train_ensemble(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['DeepEnsemble'] = temporal

        # --- SNGP ---
        print("\n  SNGP:")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SNGPBaseline(n_ent, n_rel, num_rff=512)
        model.precompute_coverage(train)
        model = train_sngp(model, train, device, epochs=30)
        temporal = evaluate_temporal(model, train, test, n_ent, device)
        elapsed = time.time() - t0
        print(f"    Temporal AUROC: {temporal.get('overall_auroc', 'N/A')}")
        print(f"    Time: {elapsed:.1f}s")
        seed_results['SNGP'] = temporal

        all_results[f'seed_{seed}'] = seed_results

    # Also get link prediction metrics (once, seed=42)
    print("\n\nLink Prediction (base DistMult, seed=42):")
    torch.manual_seed(42)
    np.random.seed(42)
    from scripts.run_wn18rr_temporal import UKGE as BaseModel
    model = BaseModel(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=30)
    lp_results = evaluate_link_prediction(model, test, train, n_ent, device, max_test=500)
    print(f"  MRR: {lp_results['mrr']:.3f}, Hits@10: {lp_results['hits@10']:.3f}, Hits@1: {lp_results['hits@1']:.3f}")
    all_results['link_prediction'] = lp_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std over 3 seeds)")
    print("=" * 60)

    for method in ['MCDropout', 'DeepEnsemble', 'SNGP']:
        aurocs = []
        for seed in seeds:
            r = all_results[f'seed_{seed}'][method]
            if 'overall_auroc' in r:
                aurocs.append(r['overall_auroc'])
        if aurocs:
            print(f"  {method}: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")

    # Save
    out = project_root / 'outputs' / 'wn18rr_missing_baselines.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
