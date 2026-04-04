#!/usr/bin/env python3
"""
RCUE Experiment: Compare relation-conditioned uncertainty vs baselines.

Compares:
1. Energy (relation-agnostic)
2. UKGE (relation-agnostic variance)
3. RCUE (relation-conditioned variance via MLP)
4. RCUE-Attention (relation-conditioned variance via attention)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import load_fb15k237, load_wn18rr
from src.models.relation_conditioned import RCUE, RCUEWithAttention, train_rcue


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Baselines

class EnergyBaseline(nn.Module):
    """Energy scoring baseline (relation-agnostic uncertainty)."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        # Energy-based: uncertainty = negative score
        return -self.forward(h, r, t)

    def score_tails(self, h, r):
        hr = self.entity_emb(h) * self.relation_emb(r)
        return hr @ self.entity_emb.weight.T


class UKGEBaseline(nn.Module):
    """UKGE-style baseline (entity-level variance, relation-agnostic)."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.entity_mean = nn.Embedding(num_entities, dim)
        self.entity_logvar = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_logvar.weight, -1.0)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0

    def forward(self, h, r, t):
        h_emb = self.entity_mean(h)
        t_emb = self.entity_mean(t)
        r_emb = self.relation_emb(r)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        # Entity-level variance only (relation-agnostic!)
        h_var = torch.exp(self.entity_logvar(h)).mean(-1)
        t_var = torch.exp(self.entity_logvar(t)).mean(-1)
        return h_var + t_var

    def score_tails(self, h, r):
        hr = self.entity_mean(h) * self.relation_emb(r)
        return hr @ self.entity_mean.weight.T


def train_baseline(model, train_triples, device, epochs=50, lr=1e-3):
    """Train baseline model."""
    model = model.to(device)
    model.precompute_coverage(train_triples)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h_all = torch.tensor(train_triples[:, 0])
    r_all = torch.tensor(train_triples[:, 1])
    t_all = torch.tensor(train_triples[:, 2])

    loader = DataLoader(TensorDataset(h_all, r_all, t_all), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)

            pos_scores = model(h, r, t)
            neg_scores = model(h, r, neg_t)

            loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=1.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def create_ood_split(train_triples, test_triples, ood_fraction=0.2):
    """
    Create OOD split based on novel contexts.
    OOD = test triples where (h,r) or (t,r) not seen in training.
    """
    # Build coverage from training
    coverage = set()
    for h, r, t in train_triples:
        coverage.add((h, r))
        coverage.add((t, r))

    # Mark OOD
    ood_mask = []
    for h, r, t in test_triples:
        is_ood = (h, r) not in coverage or (t, r) not in coverage
        ood_mask.append(is_ood)

    return np.array(ood_mask)


def evaluate_ood(model, test_triples, ood_mask, device):
    """Evaluate OOD detection AUROC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    h = torch.tensor(test_triples[:, 0], device=device)
    r = torch.tensor(test_triples[:, 1], device=device)
    t = torch.tensor(test_triples[:, 2], device=device)

    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()

    return roc_auc_score(ood_mask, unc)


def run_experiment(dataset_name, load_fn, seed=42, epochs=50):
    """Run full experiment on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    device = setup_device()
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_ds, _, test_ds = load_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Create OOD split
    ood_mask = create_ood_split(train, test)
    print(f"OOD fraction: {ood_mask.mean()*100:.1f}%")

    results = {}

    # 1. Energy baseline
    print("\n--- Energy Baseline ---")
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_baseline(energy, train, device, epochs=epochs)
    auroc = evaluate_ood(energy, test, ood_mask, device)
    results['Energy'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # 2. UKGE baseline
    print("\n--- UKGE Baseline ---")
    ukge = UKGEBaseline(n_ent, n_rel)
    ukge = train_baseline(ukge, train, device, epochs=epochs)
    auroc = evaluate_ood(ukge, test, ood_mask, device)
    results['UKGE'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # 3. RCUE (MLP)
    print("\n--- RCUE (MLP) ---")
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=epochs, verbose=True)
    auroc = evaluate_ood(rcue, test, ood_mask, device)
    results['RCUE'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # 4. RCUE without coverage (ablation)
    print("\n--- RCUE (no coverage) ---")
    rcue_nocov = RCUE(n_ent, n_rel, use_coverage=False)
    rcue_nocov = train_rcue(rcue_nocov, train, device, epochs=epochs, verbose=True)
    auroc = evaluate_ood(rcue_nocov, test, ood_mask, device)
    results['RCUE-noCov'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    for name, auroc in results.items():
        print(f"{name:<15}: {auroc:.4f}")

    return results


def main():
    print("RCUE Experiment: Relation-Conditioned Uncertainty")
    print(f"Date: {datetime.now().isoformat()}")
    print("="*60)

    all_results = {}

    # FB15k-237
    all_results['FB15k-237'] = run_experiment('FB15k-237', load_fb15k237, seed=42, epochs=50)

    # WN18RR
    all_results['WN18RR'] = run_experiment('WN18RR', load_wn18rr, seed=42, epochs=50)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Method':<15} {'FB15k-237':<12} {'WN18RR':<12}")
    print("-"*40)

    methods = ['Energy', 'UKGE', 'RCUE', 'RCUE-noCov']
    for method in methods:
        fb = all_results['FB15k-237'].get(method, 0)
        wn = all_results['WN18RR'].get(method, 0)
        print(f"{method:<15} {fb:.4f}       {wn:.4f}")


if __name__ == "__main__":
    main()
