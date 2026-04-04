#!/usr/bin/env python3
"""
Verify 83% confident-wrong is actual errors, not just zero-coverage.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def main():
    print("="*70)
    print("83% CONFIDENT-WRONG VERIFICATION")
    print("Check: Are zero-coverage predictions actually WRONG?")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Test: {len(test)} triples")

    # Coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    # Train Energy
    print("\n--- Training Energy ---")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(15):
        np.random.shuffle(train)
        total_loss = 0
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = energy(h, r, t)
            neg = energy(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15, Loss: {total_loss:.2f}")

    # Compute confidence and ranks for test set
    print("\n--- Computing confidence and ranks ---")
    energy.eval()

    results = []

    with torch.no_grad():
        for idx, (h, r, t) in enumerate(test):
            if idx % 2000 == 0:
                print(f"  Processing {idx}/{len(test)}...")

            # Confidence = score for this triple
            h_t = torch.tensor([h])
            r_t = torch.tensor([r])
            t_t = torch.tensor([t])
            confidence = energy(h_t, r_t, t_t).item()

            # Coverage
            is_zero_cov = (int(h), int(r)) not in coverage_set or (int(t), int(r)) not in coverage_set

            # Rank (is prediction correct?)
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)

            scores = energy(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = (scores > true_score).sum() + 1

            results.append({
                'confidence': confidence,
                'zero_coverage': is_zero_cov,
                'rank': rank,
                'hits10': rank <= 10,
                'hits1': rank == 1
            })

    # Analyze top-K most confident
    print("\n" + "="*70)
    print("RESULTS: Top-K Most Confident Predictions")
    print("="*70)

    # Sort by confidence (higher = more confident)
    results.sort(key=lambda x: x['confidence'], reverse=True)

    for k in [100, 500, 1000, 5000]:
        top_k = results[:k]

        zero_cov_rate = sum(r['zero_coverage'] for r in top_k) / k
        hits10_rate = sum(r['hits10'] for r in top_k) / k
        hits1_rate = sum(r['hits1'] for r in top_k) / k
        avg_rank = np.mean([r['rank'] for r in top_k])

        # Zero-coverage AND wrong
        zero_cov_wrong = sum(1 for r in top_k if r['zero_coverage'] and not r['hits10']) / k

        print(f"\nTop-{k} most confident:")
        print(f"  Zero-coverage rate: {zero_cov_rate:.1%}")
        print(f"  Hits@10:            {hits10_rate:.1%}")
        print(f"  Hits@1:             {hits1_rate:.1%}")
        print(f"  Avg rank:           {avg_rank:.1f}")
        print(f"  Zero-cov AND wrong: {zero_cov_wrong:.1%}")

    # Baseline (all test)
    print("\n" + "="*70)
    print("BASELINE: All Test Triples")
    print("="*70)

    all_zero_cov = sum(r['zero_coverage'] for r in results) / len(results)
    all_hits10 = sum(r['hits10'] for r in results) / len(results)
    all_hits1 = sum(r['hits1'] for r in results) / len(results)
    all_avg_rank = np.mean([r['rank'] for r in results])
    all_zero_cov_wrong = sum(1 for r in results if r['zero_coverage'] and not r['hits10']) / len(results)

    print(f"Zero-coverage rate: {all_zero_cov:.1%}")
    print(f"Hits@10:            {all_hits10:.1%}")
    print(f"Hits@1:             {all_hits1:.1%}")
    print(f"Avg rank:           {all_avg_rank:.1f}")
    print(f"Zero-cov AND wrong: {all_zero_cov_wrong:.1%}")

    # Key metric: Among zero-coverage, what's the error rate?
    print("\n" + "="*70)
    print("KEY ANALYSIS: Error Rate by Coverage Status")
    print("="*70)

    zero_cov_samples = [r for r in results if r['zero_coverage']]
    has_cov_samples = [r for r in results if not r['zero_coverage']]

    if zero_cov_samples:
        zc_hits10 = sum(r['hits10'] for r in zero_cov_samples) / len(zero_cov_samples)
        print(f"\nZero-coverage ({len(zero_cov_samples)} samples):")
        print(f"  Hits@10: {zc_hits10:.1%}")
        print(f"  Error rate (not in top-10): {1-zc_hits10:.1%}")

    if has_cov_samples:
        hc_hits10 = sum(r['hits10'] for r in has_cov_samples) / len(has_cov_samples)
        print(f"\nHas-coverage ({len(has_cov_samples)} samples):")
        print(f"  Hits@10: {hc_hits10:.1%}")
        print(f"  Error rate (not in top-10): {1-hc_hits10:.1%}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("If zero-coverage has higher error rate than has-coverage,")
    print("then 'confident on zero-coverage' = 'confidently WRONG'")


if __name__ == "__main__":
    main()
