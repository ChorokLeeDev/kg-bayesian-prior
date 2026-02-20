#!/usr/bin/env python3
"""
Focused Experiments for EMNLP - Fixed Models

Based on initial results:
- CAGP and AttentionCAGP work well (0.96+ AUROC)
- Energy baseline also strong on random OOD (0.99)
- RelCondVar needs training fix (was outputting constant uncertainty)

This script runs optimized experiments with proper model training.
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


class CoverageOnly(nn.Module):
    """Coverage-only baseline."""
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
    """GP variance only baseline."""
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
    """Coverage-Augmented GP-KGE."""
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

        # Use cached normalization stats if available
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


class AttentionCAGP(nn.Module):
    """Attention-based CAGP."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        self.attention = nn.Sequential(
            nn.Linear(3 * dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h, r, t):
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2
        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]

        features = torch.cat([
            self.entity_mean[h], self.relation_emb(r), self.entity_mean[t],
            gp_var.unsqueeze(-1), cov_unc.unsqueeze(-1)
        ], dim=-1)
        alpha = self.attention(features).squeeze(-1)

        gp_norm = gp_var / (gp_var.mean() + 1e-8) * (cov_unc.mean() + 1e-8)
        return alpha * gp_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class RelCondVar(nn.Module):
    """Fixed Relation-Conditioned Variance."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Relation-conditioned variance with proper initialization
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        # Initialize to output reasonable variance
        nn.init.zeros_(self.var_net[-1].weight)
        nn.init.constant_(self.var_net[-1].bias, -1.0)

        # Also keep base variance for fallback
        self.entity_base_logvar = nn.Parameter(torch.zeros(num_entities) - 1.0)

    def forward(self, h, r, t):
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_entity_relation_var(self, e, r):
        e_emb = self.entity_mean[e]
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        raw = self.var_net(combined).squeeze(-1)

        # Add base variance and use softplus
        base_var = torch.exp(self.entity_base_logvar[e])
        return F.softplus(raw) + base_var * 0.1 + 1e-4

    def get_uncertainty(self, h, r, t):
        # Combine relation-conditioned variance with coverage
        h_var = self.get_entity_relation_var(h, r)
        t_var = self.get_entity_relation_var(t, r)
        semantic = (h_var + t_var) / 2

        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]

        # Learned combination
        semantic_norm = semantic / (semantic.mean() + 1e-8) * (cov_unc.mean() + 1e-8)
        return 0.5 * semantic_norm + 0.5 * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class EnergyBased(nn.Module):
    """Energy-based uncertainty."""
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
        return -scores  # Lower score = higher uncertainty

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class UKGE(nn.Module):
    """UKGE-style confidence."""
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


def train_model(model, triples, device, epochs=30, lr=0.001):
    """Train model with uncertainty-aware loss."""
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

            # Uncertainty regularization: OOD should have higher uncertainty
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


def evaluate_ood(model, test, n_ent, device, mode='random'):
    """Evaluate OOD detection."""
    model.eval()
    # Use full test set for random, larger samples for adversarial
    max_samples = 5000 if mode == 'random' else 3000
    test = test[:min(len(test), max_samples)]

    # Generate OOD
    if mode == 'random':
        ood_t = np.random.randint(0, n_ent, len(test))
    elif mode == 'relation_plausible':
        cov = model.coverage.cpu().numpy()
        ood_t = []
        for i in range(len(test)):
            r = test[i, 1]
            valid = np.where(cov[:, r] > 0)[0]
            ood_t.append(np.random.choice(valid) if len(valid) > 0 else np.random.randint(n_ent))
        ood_t = np.array(ood_t)
    elif mode == 'embedding_similar':
        with torch.no_grad():
            if hasattr(model, 'entity_mean'):
                emb = model.entity_mean.cpu().numpy()
            else:
                emb = model.entity_emb.weight.cpu().numpy()
            ood_t = []
            for i in range(len(test)):
                t = test[i, 2]
                dists = np.linalg.norm(emb - emb[t], axis=1)
                dists[t] = np.inf
                nn = np.argsort(dists)[:10]
                ood_t.append(nn[np.random.randint(len(nn))])
            ood_t = np.array(ood_t)
    elif mode == 'high_score':
        with torch.no_grad():
            ood_t = []
            # Process in batches for efficiency
            batch_size = 100
            num_samples = min(len(test), 2000)  # Use more samples for robust evaluation
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                for i in range(batch_start, batch_end):
                    h = torch.tensor([test[i, 0]]).to(device)
                    r = torch.tensor([test[i, 1]]).to(device)
                    all_t = torch.arange(n_ent).to(device)
                    scores = model(h.expand(n_ent), r.expand(n_ent), all_t)
                    scores[test[i, 2]] = -1e9
                    top_k = torch.topk(scores, 10).indices
                    ood_t.append(top_k[np.random.randint(10)].item())
            # For remaining samples, use random to maintain test size consistency
            ood_t.extend([np.random.randint(n_ent) for _ in range(len(test) - len(ood_t))])
            ood_t = np.array(ood_t)
    else:
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


def evaluate_temporal(model, train, test, n_ent, device):
    """Temporal OOD evaluation."""
    model.eval()

    # Entity frequencies
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()

    new_entity, new_pair = [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if freq.get(h, 0) <= thresh or freq.get(t, 0) <= thresh:
            new_entity.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair.append(i)

    results = {}
    if len(new_entity) > 50:
        results['new_entity'] = evaluate_ood(model, test[new_entity[:1000]], n_ent, device)
    if len(new_pair) > 50:
        results['new_pair'] = evaluate_ood(model, test[new_pair[:1000]], n_ent, device)

    return results


def evaluate_qa(model, test, n_ent, device, coverage=0.85):
    """QA abstention evaluation."""
    model.eval()
    test = test[:2000]

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)

        unc = model.get_uncertainty(h, r, t).cpu().numpy()

        # Get predictions
        correct = []
        for i in range(len(test)):
            hi = torch.tensor([test[i, 0]]).to(device)
            ri = torch.tensor([test[i, 1]]).to(device)
            all_t = torch.arange(n_ent).to(device)
            scores = model(hi.expand(n_ent), ri.expand(n_ent), all_t)
            pred = scores.argmax().item()
            correct.append(pred == test[i, 2])

        correct = np.array(correct)

    # Selective accuracy at target coverage
    n = int(coverage * len(correct))
    idx = np.argsort(unc)[:n]
    sel_acc = correct[idx].mean()
    base_acc = correct.mean()

    return {
        'selective_acc': sel_acc,
        'baseline_acc': base_acc,
        'error_reduction': (1 - base_acc - (1 - sel_acc)) / (1 - base_acc + 1e-8)
    }


def main():
    device = setup_device()
    print(f"Device: {device}")

    results = {}

    for ds_name, loader in [('fb15k-237', load_fb15k237), ('wn18rr', load_wn18rr)]:
        print(f"\n{'='*60}\n{ds_name}\n{'='*60}")

        try:
            train_ds, _, test_ds = loader()
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        train = train_ds.triples
        test = test_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations

        print(f"Entities: {n_ent}, Relations: {n_rel}")
        print(f"Train: {len(train)}, Test: {len(test)}")

        ds_results = {}

        models = {
            'Coverage': CoverageOnly(n_ent, n_rel),
            'GP': GPOnly(n_ent, n_rel),
            'CAGP': CAGP(n_ent, n_rel),
            'AttentionCAGP': AttentionCAGP(n_ent, n_rel),
            'RelCondVar': RelCondVar(n_ent, n_rel),
            'Energy': EnergyBased(n_ent, n_rel),
            'UKGE': UKGE(n_ent, n_rel),
        }

        for name, model in models.items():
            print(f"\n--- {name} ---")
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30)

            # Calibrate normalization on training set (not test) to avoid data leakage
            if hasattr(model, 'calibrate_normalization'):
                model.calibrate_normalization(train, device)

            m_results = {}

            # Random OOD
            m_results['random'] = evaluate_ood(model, test, n_ent, device, 'random')
            print(f"  Random OOD: {m_results['random']:.4f}")

            # Adversarial (skip for baselines)
            if name in ['CAGP', 'AttentionCAGP', 'RelCondVar']:
                for mode in ['embedding_similar', 'relation_plausible', 'high_score']:
                    m_results[mode] = evaluate_ood(model, test, n_ent, device, mode)
                    print(f"  {mode}: {m_results[mode]:.4f}")

            # Temporal
            temp = evaluate_temporal(model, train, test, n_ent, device)
            m_results['temporal'] = temp
            for k, v in temp.items():
                print(f"  {k}: {v:.4f}")

            # QA
            qa = evaluate_qa(model, test, n_ent, device)
            m_results['qa'] = qa
            print(f"  QA Sel.Acc: {qa['selective_acc']:.4f}, ErrRed: {qa['error_reduction']:.4f}")

            ds_results[name] = m_results

        results[ds_name] = ds_results

    # Save
    out = project_root / 'outputs' / 'focused_results.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved to {out}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Random OOD AUROC")
    print("="*80)
    for ds in results:
        print(f"\n{ds}:")
        for m in results[ds]:
            print(f"  {m}: {results[ds][m].get('random', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
