#!/usr/bin/env python3
"""
Active Learning Simulation Experiment

Simulate: If we could query labels for uncertain samples,
does RCUE uncertainty lead to faster learning?
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237
from src.models.relation_conditioned import RCUE


class SimpleKGE(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def train_model(model, train_data, n_ent, epochs=5, batch_size=512):
    """Quick training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = model(h, r, t)
            neg = model(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()

    return model


def evaluate_hits10(model, test_data, n_ent, max_samples=500):
    """Compute Hits@10."""
    model.eval()
    hits = 0

    with torch.no_grad():
        for idx, (h, r, t) in enumerate(test_data[:max_samples]):
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)

            scores = model(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = (scores > true_score).sum() + 1

            if rank <= 10:
                hits += 1

    return hits / max_samples


def main():
    print("="*70)
    print("ACTIVE LEARNING SIMULATION")
    print("Goal: Show RCUE-guided sampling leads to faster learning")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    full_train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Full train: {len(full_train)}, Test: {len(test)}")

    # Split train into initial (10%) and pool (90%)
    np.random.seed(42)
    np.random.shuffle(full_train)
    initial_size = len(full_train) // 10
    initial_train = full_train[:initial_size]
    pool = full_train[initial_size:]

    print(f"Initial train: {len(initial_train)}, Pool: {len(pool)}")

    # ========================================
    # Train initial RCUE for uncertainty estimation
    # ========================================
    print("\n--- Training initial RCUE for uncertainty ---")
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, embedding_dim=100, hidden_dim=64, use_coverage=True)
    rcue.precompute_coverage(initial_train)

    optimizer = torch.optim.Adam(rcue.parameters(), lr=1e-3)
    for epoch in range(10):
        np.random.shuffle(initial_train)
        for i in range(0, len(initial_train), 512):
            batch = initial_train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = rcue(h, r, t)
            neg = rcue(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()

    # ========================================
    # Active learning simulation
    # ========================================
    budget_steps = [0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of pool to add

    results_random = []
    results_rcue = []
    results_coverage = []

    # Build coverage set
    coverage_set = set()
    for h, r, t in initial_train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    for budget_frac in budget_steps:
        n_to_add = int(len(pool) * budget_frac)
        print(f"\n--- Budget: {budget_frac:.0%} of pool ({n_to_add} samples) ---")

        # Strategy 1: Random sampling
        torch.manual_seed(42)
        random_indices = np.random.choice(len(pool), n_to_add, replace=False)
        random_train = np.vstack([initial_train, pool[random_indices]])

        model_random = SimpleKGE(n_ent, n_rel)
        model_random = train_model(model_random, random_train, n_ent, epochs=5)
        hits10_random = evaluate_hits10(model_random, test, n_ent)
        results_random.append(hits10_random)
        print(f"  Random:   Hits@10 = {hits10_random:.4f}")

        # Strategy 2: RCUE uncertainty sampling (high uncertainty first)
        rcue.eval()
        with torch.no_grad():
            h_pool = torch.tensor(pool[:, 0])
            r_pool = torch.tensor(pool[:, 1])
            t_pool = torch.tensor(pool[:, 2])
            pool_unc = rcue.get_uncertainty(h_pool, r_pool, t_pool).numpy()

        rcue_indices = np.argsort(pool_unc)[-n_to_add:]  # Highest uncertainty
        rcue_train = np.vstack([initial_train, pool[rcue_indices]])

        torch.manual_seed(42)
        model_rcue = SimpleKGE(n_ent, n_rel)
        model_rcue = train_model(model_rcue, rcue_train, n_ent, epochs=5)
        hits10_rcue = evaluate_hits10(model_rcue, test, n_ent)
        results_rcue.append(hits10_rcue)
        print(f"  RCUE:     Hits@10 = {hits10_rcue:.4f}")

        # Strategy 3: Coverage-based sampling (OOD first)
        pool_cov = np.array([
            0.0 if ((int(h), int(r)) not in coverage_set or (int(t), int(r)) not in coverage_set) else 1.0
            for h, r, t in pool
        ])
        # OOD samples have coverage=0, so we want those first
        cov_indices = np.argsort(pool_cov)[:n_to_add]  # Lowest coverage (OOD)
        cov_train = np.vstack([initial_train, pool[cov_indices]])

        torch.manual_seed(42)
        model_cov = SimpleKGE(n_ent, n_rel)
        model_cov = train_model(model_cov, cov_train, n_ent, epochs=5)
        hits10_cov = evaluate_hits10(model_cov, test, n_ent)
        results_coverage.append(hits10_cov)
        print(f"  Coverage: Hits@10 = {hits10_cov:.4f}")

    # ========================================
    # Results
    # ========================================
    print("\n" + "="*70)
    print("ACTIVE LEARNING RESULTS")
    print("="*70)

    print(f"\n{'Budget':<10} {'Random':<12} {'Coverage':<12} {'RCUE':<12}")
    print("-"*46)
    for i, budget in enumerate(budget_steps):
        print(f"{budget:<10.0%} {results_random[i]:<12.4f} {results_coverage[i]:<12.4f} {results_rcue[i]:<12.4f}")

    # Compute area under learning curve
    auc_random = np.trapz(results_random, budget_steps)
    auc_coverage = np.trapz(results_coverage, budget_steps)
    auc_rcue = np.trapz(results_rcue, budget_steps)

    print(f"\nArea Under Learning Curve:")
    print(f"  Random:   {auc_random:.4f}")
    print(f"  Coverage: {auc_coverage:.4f}")
    print(f"  RCUE:     {auc_rcue:.4f}")

    improvement = (auc_rcue - auc_random) / auc_random * 100
    print(f"\nRCUE improvement over Random: {improvement:.1f}%")


if __name__ == "__main__":
    main()
