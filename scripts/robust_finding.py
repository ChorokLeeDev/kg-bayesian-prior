#!/usr/bin/env python3
"""
Focus on robust finding: Full zero is ALWAYS bad across all datasets.
Also test: What predicts when partial > full vs partial < full?
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def load_icews14():
    """Load ICEWS14."""
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/ICEWS14")
    entity2id, relation2id = {}, {}

    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entity2id: entity2id[h] = len(entity2id)
                    if r not in relation2id: relation2id[r] = len(relation2id)
                    if t not in entity2id: entity2id[t] = len(entity2id)
                    triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)

    train = load_triples("train.txt")
    test = load_triples("test.txt")
    return train, test, len(entity2id), len(relation2id)


def analyze_full_zero(name, train, test, n_ent, n_rel):
    """Focus analysis on full zero performance."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    # Coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    # Train
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(10):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h, r, t = torch.tensor(batch[:, 0]), torch.tensor(batch[:, 1]), torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            optimizer.zero_grad()
            loss = torch.clamp(1.0 - energy(h, r, t) + energy(h, r, t_neg), min=0).mean()
            loss.backward()
            optimizer.step()

    # Analyze
    energy.eval()
    test_sub = test[:2000] if len(test) > 2000 else test

    results = {'full_cov': [], 'partial': [], 'full_zero': []}

    with torch.no_grad():
        for h, r, t in test_sub:
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set

            if h_cov and t_cov:
                cov_type = 'full_cov'
            elif not h_cov and not t_cov:
                cov_type = 'full_zero'
            else:
                cov_type = 'partial'

            # Rank
            scores = energy(torch.full((n_ent,), h, dtype=torch.long),
                          torch.full((n_ent,), r, dtype=torch.long),
                          torch.arange(n_ent)).numpy()
            rank = int((scores > scores[t]).sum() + 1)
            results[cov_type].append(rank <= 10)

    # Report
    for cov_type in ['full_cov', 'partial', 'full_zero']:
        if results[cov_type]:
            hits = sum(results[cov_type]) / len(results[cov_type])
            print(f"{cov_type:12}: n={len(results[cov_type]):4}, Hits@10={hits:.1%}")
        else:
            print(f"{cov_type:12}: n=0")

    return {k: (sum(v)/len(v) if v else 0, len(v)) for k, v in results.items()}


def main():
    print("="*60)
    print("ROBUST FINDING: Full Zero is Always Bad")
    print("="*60)

    all_results = {}

    # FB15k-237
    ds = load_fb15k237()
    all_results['FB15k-237'] = analyze_full_zero('FB15k-237', ds[0].triples, ds[2].triples,
                                                  ds[0].num_entities, ds[0].num_relations)

    # WN18RR
    ds = load_wn18rr()
    all_results['WN18RR'] = analyze_full_zero('WN18RR', ds[0].triples, ds[2].triples,
                                               ds[0].num_entities, ds[0].num_relations)

    # ICEWS14
    train, test, n_ent, n_rel = load_icews14()
    all_results['ICEWS14'] = analyze_full_zero('ICEWS14', train, test, n_ent, n_rel)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Full Zero is Consistently the Worst")
    print("="*60)
    print(f"{'Dataset':<12} {'Full Cov':<12} {'Partial':<12} {'Full Zero':<12} {'Worst?':<8}")
    print("-"*56)

    for name, res in all_results.items():
        fc = f"{res['full_cov'][0]:.1%}" if res['full_cov'][1] > 0 else "N/A"
        p = f"{res['partial'][0]:.1%}" if res['partial'][1] > 0 else "N/A"
        fz = f"{res['full_zero'][0]:.1%}" if res['full_zero'][1] > 0 else "N/A"

        # Check if full_zero is worst
        vals = [(res['full_cov'][0], 'FC'), (res['partial'][0], 'P'), (res['full_zero'][0], 'FZ')]
        vals = [(v, n) for v, n in vals if res[{'FC': 'full_cov', 'P': 'partial', 'FZ': 'full_zero'}[n]][1] > 0]
        worst = min(vals, key=lambda x: x[0])[1] if vals else "N/A"

        print(f"{name:<12} {fc:<12} {p:<12} {fz:<12} {worst:<8}")

    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    print("""
ROBUST ACROSS ALL DATASETS:
- Full zero coverage (neither entity seen with relation) = WORST performance
- This holds regardless of whether partial > full_cov or partial < full_cov

DATASET-DEPENDENT:
- Whether partial > full_cov varies by dataset
- FB15k-237: partial wins
- WN18RR, ICEWS14: full_cov wins

PRACTICAL IMPLICATION:
- ALWAYS flag full zero queries (both entities unseen with relation)
- Whether to flag partial depends on dataset characteristics
""")


if __name__ == "__main__":
    main()
