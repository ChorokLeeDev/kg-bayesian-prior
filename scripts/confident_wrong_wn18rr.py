#!/usr/bin/env python3
"""
Confident-Wrong Analysis on WN18RR.
Validates the 78% finding on a different dataset.
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

from src.data.loaders import load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def analyze_confident_wrong(model, test, device, k_values=[100, 500, 1000]):
    model.eval()
    cov = model.coverage.cpu().numpy()

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)

        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -uncertainties

    zero_evidence = []
    for i in range(len(test)):
        h_cov = cov[test[i, 0], test[i, 1]]
        t_cov = cov[test[i, 2], test[i, 1]]
        zero_evidence.append(h_cov == 0 or t_cov == 0)
    zero_evidence = np.array(zero_evidence)

    sorted_indices = np.argsort(confidence)[::-1]

    results = {}
    for k in k_values:
        actual_k = min(k, len(sorted_indices))
        top_k_indices = sorted_indices[:actual_k]
        top_k_zero_evidence = zero_evidence[top_k_indices]
        fraction = top_k_zero_evidence.mean()
        results[k] = fraction

    return results


def main():
    device = setup_device()
    print(f"WN18RR Confident-Wrong Analysis")
    print(f"Device: {device}")
    print(f"Date: {datetime.now().isoformat()}")

    # Load WN18RR
    train_ds, _, test_ds = load_wn18rr()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"\nWN18RR: {n_ent} entities, {n_rel} relations")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Compute baseline
    coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train)):
        coverage[train[i, 0], train[i, 1]] = 1.0
        coverage[train[i, 2], train[i, 1]] = 1.0

    novel_ctx_count = 0
    for i in range(len(test)):
        h, r, t = test[i]
        if coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_ctx_count += 1

    baseline_pct = 100.0 * novel_ctx_count / len(test)
    print(f"Baseline (random): {baseline_pct:.1f}% zero-evidence")

    # Run 3 seeds
    seeds = [42, 123, 456]
    all_results = {100: [], 500: [], 1000: []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = EnergyBased(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)

        results = analyze_confident_wrong(model, test, device)
        for k in [100, 500, 1000]:
            pct = results[k] * 100
            all_results[k].append(pct)
            print(f"  Top-{k}: {pct:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("WN18RR SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_pct:.1f}%")
    for k in [100, 500, 1000]:
        mean = np.mean(all_results[k])
        std = np.std(all_results[k])
        elevation = mean / baseline_pct
        print(f"Top-{k}: {mean:.1f}% ± {std:.1f}% (elevation: {elevation:.2f}x)")


if __name__ == "__main__":
    main()
