#!/usr/bin/env python3
"""
Deep Ensemble baseline for KG OOD detection.
Ensemble of DistMult models, use variance as uncertainty.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_fb15k237


class DistMultModel(nn.Module):
    """Simple DistMult for ensemble."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def score_tails(self, h, r):
        hr = self.entity_emb(h) * self.relation_emb(r)
        return hr @ self.entity_emb.weight.T


class DeepEnsemble:
    """Ensemble of DistMult models."""
    def __init__(self, num_entities, num_relations, n_models=5, dim=100):
        self.models = [DistMultModel(num_entities, num_relations, dim) for _ in range(n_models)]
        self.num_entities = num_entities
        self.n_models = n_models

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    def train_all(self, train_triples, device, epochs=30, lr=1e-3):
        h_all = torch.tensor(train_triples[:, 0], dtype=torch.long)
        r_all = torch.tensor(train_triples[:, 1], dtype=torch.long)
        t_all = torch.tensor(train_triples[:, 2], dtype=torch.long)
        loader = DataLoader(TensorDataset(h_all, r_all, t_all), batch_size=1024, shuffle=True)

        for i, model in enumerate(self.models):
            print(f"  Training model {i+1}/{self.n_models}")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                for h, r, t in loader:
                    h, r, t = h.to(device), r.to(device), t.to(device)
                    neg_t = torch.randint(0, self.num_entities, t.shape, device=device)

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

    def get_uncertainty(self, h, r, t):
        """Uncertainty = variance of ensemble predictions."""
        scores = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                s = model(h, r, t)
                scores.append(s)

        scores = torch.stack(scores, dim=0)  # [n_models, batch]
        # Uncertainty = variance across ensemble
        variance = scores.var(dim=0)
        return variance


class MCDropoutModel(nn.Module):
    """DistMult with dropout for MC Dropout uncertainty."""
    def __init__(self, num_entities, num_relations, dim=100, dropout=0.2):
        super().__init__()
        self.num_entities = num_entities
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t, apply_dropout=True):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        if apply_dropout:
            h_emb = self.dropout(h_emb)
            t_emb = self.dropout(t_emb)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t, n_samples=10):
        """MC Dropout uncertainty."""
        self.train()  # Keep dropout active
        scores = []
        with torch.no_grad():
            for _ in range(n_samples):
                s = self.forward(h, r, t, apply_dropout=True)
                scores.append(s)
        scores = torch.stack(scores, dim=0)
        return scores.var(dim=0)


def train_mc_dropout(model, train_triples, device, epochs=30, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h_all = torch.tensor(train_triples[:, 0], dtype=torch.long)
    r_all = torch.tensor(train_triples[:, 1], dtype=torch.long)
    t_all = torch.tensor(train_triples[:, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(h_all, r_all, t_all), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        model.train()
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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")

    return model


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # OOD mask
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])

    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"OOD fraction: {ood_mask.mean()*100:.1f}%")

    h_test = torch.tensor(test[:, 0], device=device)
    r_test = torch.tensor(test[:, 1], device=device)
    t_test = torch.tensor(test[:, 2], device=device)

    results = {}

    # 1. Deep Ensemble
    print("\n--- Deep Ensemble (5 models) ---")
    torch.manual_seed(42)
    ensemble = DeepEnsemble(n_ent, n_rel, n_models=5)
    ensemble.to(device)
    ensemble.train_all(train, device, epochs=30)

    unc = ensemble.get_uncertainty(h_test, r_test, t_test).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    results['DeepEnsemble'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # 2. MC Dropout
    print("\n--- MC Dropout ---")
    torch.manual_seed(42)
    mc_model = MCDropoutModel(n_ent, n_rel, dropout=0.3)
    mc_model = train_mc_dropout(mc_model, train, device, epochs=30)

    unc = mc_model.get_uncertainty(h_test, r_test, t_test, n_samples=20).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    results['MCDropout'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # 3. RCUE for comparison
    print("\n--- RCUE ---")
    torch.manual_seed(42)
    from src.models.relation_conditioned import RCUE, train_rcue
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=30, verbose=True)

    rcue.eval()
    with torch.no_grad():
        unc = rcue.get_uncertainty(h_test, r_test, t_test).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    results['RCUE'] = auroc
    print(f"  OOD AUROC: {auroc:.4f}")

    # Summary
    print("\n" + "="*50)
    print("OOD DETECTION SUMMARY")
    print("="*50)
    for name, auroc in results.items():
        print(f"{name:<20}: {auroc:.4f}")


if __name__ == "__main__":
    main()
