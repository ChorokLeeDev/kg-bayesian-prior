#!/usr/bin/env python3
"""
Investigate: Why does OOD have HIGHER MRR?

Surprising finding from cascading_v2:
- Covered (ID) MRR: 0.1479
- Uncovered (OOD) MRR: 0.4713

This seems to contradict our understanding. Let's investigate.

Hypotheses:
1. Test set distribution: OOD triples might have "easier" relations
2. Metric artifact: MRR favors high-confidence predictions
3. Energy correlation: OOD triples might have higher energy (easier to score)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237


class DistMultBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, dim)
        self.relation_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.n_ent = n_ent

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def score_tails(self, h, r):
        hr = self.entity_emb(h) * self.relation_emb(r)
        return hr @ self.entity_emb.weight.T


def main():
    print("=" * 70)
    print("INVESTIGATING: Why does OOD have higher MRR?")
    print("=" * 70)

    # Load data
    print("\nLoading FB15k-237...")
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Build coverage
    coverage = np.zeros((n_ent, n_rel), dtype=bool)
    entity_freq = defaultdict(int)
    rel_freq = defaultdict(int)

    for h, r, t in train:
        coverage[int(h), int(r)] = True
        coverage[int(t), int(r)] = True
        entity_freq[int(h)] += 1
        entity_freq[int(t)] += 1
        rel_freq[int(r)] += 1

    # Get coverage status
    is_covered = np.zeros(len(test), dtype=bool)
    h_covered = np.zeros(len(test), dtype=bool)
    t_covered = np.zeros(len(test), dtype=bool)

    for idx, (h, r, t) in enumerate(test):
        h_cov = coverage[int(h), int(r)]
        t_cov = coverage[int(t), int(r)]
        h_covered[idx] = h_cov
        t_covered[idx] = t_cov
        if h_cov and t_cov:
            is_covered[idx] = True

    print(f"\nTest distribution:")
    print(f"  Full covered: {is_covered.sum()} ({is_covered.mean():.1%})")
    print(f"  H covered only: {(h_covered & ~t_covered).sum()} ({(h_covered & ~t_covered).mean():.1%})")
    print(f"  T covered only: {(~h_covered & t_covered).sum()} ({(~h_covered & t_covered).mean():.1%})")
    print(f"  Full zero: {(~h_covered & ~t_covered).sum()} ({(~h_covered & ~t_covered).mean():.1%})")

    # Train model
    model = DistMultBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining DistMult (15 epochs)...")
    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 1024):
            batch = train[i:i+1024]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            loss = torch.relu(1.0 - model(h, r, t) + model(h, r, t_neg)).mean()
            loss.backward()
            optimizer.step()

    model.eval()

    # Compute ranks and scores
    h_t = torch.tensor(test[:, 0])
    r_t = torch.tensor(test[:, 1])
    t_t = torch.tensor(test[:, 2])

    with torch.no_grad():
        energy_scores = model(h_t, r_t, t_t).numpy()

    # Compute ranks
    ranks = []
    with torch.no_grad():
        for i in range(0, len(test), 500):
            batch_h = h_t[i:i+500]
            batch_r = r_t[i:i+500]
            batch_t = t_t[i:i+500]
            scores = model.score_tails(batch_h, batch_r)
            true_scores = scores[torch.arange(len(batch_h)), batch_t]
            batch_ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(batch_ranks.numpy())
    ranks = np.array(ranks)

    # ============================================================
    # Analysis 1: Basic statistics by coverage type
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Basic Statistics")
    print("=" * 60)

    for name, mask in [
        ("Full covered", is_covered),
        ("H only covered", h_covered & ~t_covered),
        ("T only covered", ~h_covered & t_covered),
        ("Full zero", ~h_covered & ~t_covered)
    ]:
        if mask.sum() == 0:
            continue

        subset_ranks = ranks[mask]
        subset_scores = energy_scores[mask]

        mrr = (1.0 / subset_ranks).mean()
        hits10 = (subset_ranks <= 10).mean()
        avg_rank = subset_ranks.mean()
        avg_score = subset_scores.mean()

        print(f"\n{name}: n={mask.sum()}")
        print(f"  MRR: {mrr:.4f}, Hits@10: {hits10:.1%}")
        print(f"  Avg Rank: {avg_rank:.1f}")
        print(f"  Avg Energy Score: {avg_score:.4f}")

    # ============================================================
    # Analysis 2: Relation distribution
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Relation Distribution")
    print("=" * 60)

    covered_rels = defaultdict(int)
    uncovered_rels = defaultdict(int)

    for idx, (h, r, t) in enumerate(test):
        if is_covered[idx]:
            covered_rels[int(r)] += 1
        else:
            uncovered_rels[int(r)] += 1

    # Top relations for each
    print("\nTop 5 relations for COVERED triples:")
    for r, count in sorted(covered_rels.items(), key=lambda x: -x[1])[:5]:
        print(f"  Relation {r}: {count} ({count/is_covered.sum()*100:.1f}%)")

    print("\nTop 5 relations for UNCOVERED triples:")
    for r, count in sorted(uncovered_rels.items(), key=lambda x: -x[1])[:5]:
        print(f"  Relation {r}: {count} ({count/(~is_covered).sum()*100:.1f}%)")

    # ============================================================
    # Analysis 3: Entity frequency
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Entity Frequency")
    print("=" * 60)

    covered_h_freq = [entity_freq[int(test[idx, 0])] for idx in range(len(test)) if is_covered[idx]]
    covered_t_freq = [entity_freq[int(test[idx, 2])] for idx in range(len(test)) if is_covered[idx]]

    uncovered_h_freq = [entity_freq[int(test[idx, 0])] for idx in range(len(test)) if not is_covered[idx]]
    uncovered_t_freq = [entity_freq[int(test[idx, 2])] for idx in range(len(test)) if not is_covered[idx]]

    print("\nCOVERED triples:")
    print(f"  Avg head freq: {np.mean(covered_h_freq):.1f}")
    print(f"  Avg tail freq: {np.mean(covered_t_freq):.1f}")

    print("\nUNCOVERED triples:")
    print(f"  Avg head freq: {np.mean(uncovered_h_freq):.1f}")
    print(f"  Avg tail freq: {np.mean(uncovered_t_freq):.1f}")

    # ============================================================
    # Analysis 4: Score distribution
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Score Distribution")
    print("=" * 60)

    print("\nCOVERED triples:")
    covered_scores = energy_scores[is_covered]
    print(f"  Mean: {covered_scores.mean():.4f}")
    print(f"  Std: {covered_scores.std():.4f}")
    print(f"  Median: {np.median(covered_scores):.4f}")
    print(f"  Min: {covered_scores.min():.4f}, Max: {covered_scores.max():.4f}")

    print("\nUNCOVERED triples:")
    uncovered_scores = energy_scores[~is_covered]
    print(f"  Mean: {uncovered_scores.mean():.4f}")
    print(f"  Std: {uncovered_scores.std():.4f}")
    print(f"  Median: {np.median(uncovered_scores):.4f}")
    print(f"  Min: {uncovered_scores.min():.4f}, Max: {uncovered_scores.max():.4f}")

    # ============================================================
    # Analysis 5: Look at specific examples
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Example Triples")
    print("=" * 60)

    # Best OOD triples (rank 1)
    ood_rank1_idx = np.where((~is_covered) & (ranks == 1))[0]
    print(f"\nUNCOVERED triples with rank=1: {len(ood_rank1_idx)}")

    # Worst covered triples
    covered_worst_idx = np.where(is_covered)[0][np.argsort(-ranks[is_covered])[:5]]
    print(f"\nWorst COVERED triples:")
    for idx in covered_worst_idx:
        h, r, t = test[idx]
        print(f"  ({h}, {r}, {t}): rank={ranks[idx]}, score={energy_scores[idx]:.4f}")
        print(f"    h_freq={entity_freq[int(h)]}, t_freq={entity_freq[int(t)]}, r_freq={rel_freq[int(r)]}")

    # ============================================================
    # Key Insight
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    print("""
The paradox (OOD MRR > ID MRR) likely comes from:

1. COVERAGE DEFINITION:
   - "Covered" means entity has been seen WITH THIS SPECIFIC RELATION
   - An entity can be frequent overall but "uncovered" for a specific relation
   - Uncovered entities might still have well-learned embeddings

2. RELATION SPECIFICITY:
   - Covered triples are in "saturated" relation spaces
   - More competition -> lower ranks
   - Uncovered triples might be in sparser relation contexts

3. ENERGY ≠ CORRECTNESS:
   - High energy (score) doesn't mean correct
   - Model can be confidently wrong on covered triples
   - Model can guess correctly on uncovered (if entities are frequent)

REFRAMING THE PROBLEM:
- OOD detection is not about "will model be wrong"
- It's about "does model have evidence"
- Coverage = evidence existence
- Energy = model confidence (can be miscalibrated)

PRACTICAL IMPLICATION:
- Flag zero-coverage not because model will be wrong
- But because there's NO EVIDENCE to judge correctness
- This is the "blind spot" - we can't even calibrate confidence
""")


if __name__ == "__main__":
    main()
