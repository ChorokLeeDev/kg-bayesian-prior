#!/usr/bin/env python3
"""
Test learnable boost factor vs fixed heuristic.
Question: Is 3× the right value? Can we learn it?
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


class RCUELearnableBoost(nn.Module):
    """RCUE with learnable boost factor."""

    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.uncertainty_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # LEARNABLE boost factor (initialized to log(2) so exp gives ~2)
        # boost = 1 + exp(boost_logit) * (1 - coverage)
        # When coverage=0: boost = 1 + exp(boost_logit)
        self.boost_logit = nn.Parameter(torch.tensor(0.7))  # exp(0.7) ≈ 2

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0], triples[i, 1], triples[i, 2]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def get_entity_variance(self, entity_ids, relation_ids):
        e_emb = self.entity_emb(entity_ids)
        r_emb = self.relation_emb(relation_ids)
        unc_input = torch.cat([e_emb, r_emb], dim=-1)
        base_variance = self.uncertainty_net(unc_input).squeeze(-1)

        cov = self.coverage[entity_ids, relation_ids]
        # Learnable boost: 1 + exp(logit) * (1-cov)
        boost = 1.0 + torch.exp(self.boost_logit) * (1.0 - cov)
        return base_variance * boost

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_uncertainty(self, h, r, t):
        return self.get_entity_variance(h, r) + self.get_entity_variance(t, r)

    def get_boost_factor(self):
        """Return current boost factor for unseen pairs."""
        return 1.0 + torch.exp(self.boost_logit).item()


def train_model(model, train_triples, device, epochs=30, lr=1e-3):
    model = model.to(device)
    model.precompute_coverage(train_triples)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h_all = torch.tensor(train_triples[:, 0], dtype=torch.long)
    r_all = torch.tensor(train_triples[:, 1], dtype=torch.long)
    t_all = torch.tensor(train_triples[:, 2], dtype=torch.long)

    loader = DataLoader(TensorDataset(h_all, r_all, t_all), batch_size=1024, shuffle=True)

    boost_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)

            pos_scores = model(h, r, t)
            neg_scores = model(h, r, neg_t)

            score_loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=1.0
            )

            pos_unc = model.get_uncertainty(h, r, t)
            neg_unc = model.get_uncertainty(h, r, neg_t)
            unc_loss = F.relu(pos_unc - neg_unc + 0.1).mean()

            loss = score_loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        boost = model.get_boost_factor()
        boost_history.append(boost)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Boost={boost:.3f}")

    return model, boost_history


def evaluate(model, test_triples, ood_mask, device):
    model.eval()
    h = torch.tensor(test_triples[:, 0], device=device)
    r = torch.tensor(test_triples[:, 1], device=device)
    t = torch.tensor(test_triples[:, 2], device=device)

    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()

    return roc_auc_score(ood_mask, unc)


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

    # Test different fixed boost values
    print("\n" + "="*50)
    print("FIXED BOOST VALUES")
    print("="*50)

    for boost_val in [1.5, 2.0, 3.0, 5.0, 10.0]:
        torch.manual_seed(42)

        # Use original RCUE but modify boost
        from src.models.relation_conditioned import RCUE
        model = RCUE(n_ent, n_rel, use_coverage=True)

        # Monkey-patch the boost factor
        original_get_var = model.get_entity_variance
        def new_get_var(entity_ids, relation_ids, bv=boost_val):
            e_emb = model.entity_emb(entity_ids)
            r_emb = model.relation_emb(relation_ids)
            unc_input = torch.cat([e_emb, r_emb], dim=-1)
            base_variance = model.uncertainty_net(unc_input).squeeze(-1)
            cov = model.coverage[entity_ids, relation_ids]
            boost = 1.0 + (bv - 1.0) * (1.0 - cov)
            return base_variance * boost
        model.get_entity_variance = new_get_var

        from src.models.relation_conditioned import train_rcue
        model = train_rcue(model, train, device, epochs=30, verbose=False)
        auroc = evaluate(model, test, ood_mask, device)
        print(f"Boost={boost_val:.1f}x: AUROC={auroc:.4f}")

    # Test learnable boost
    print("\n" + "="*50)
    print("LEARNABLE BOOST")
    print("="*50)

    torch.manual_seed(42)
    model = RCUELearnableBoost(n_ent, n_rel)
    model, boost_history = train_model(model, train, device, epochs=30)
    auroc = evaluate(model, test, ood_mask, device)

    print(f"\nFinal boost: {model.get_boost_factor():.3f}x")
    print(f"AUROC: {auroc:.4f}")
    print(f"\nBoost evolution: {boost_history[0]:.2f} -> {boost_history[-1]:.2f}")


if __name__ == "__main__":
    main()
