#!/usr/bin/env python3
"""
Explore: What dataset characteristics predict coverage effect direction?
- Relation density (triples per relation)
- Graph structure (avg degree)
- Relation type (hierarchical vs flat)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr


def load_icews14():
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


def compute_dataset_stats(name, train, n_ent, n_rel):
    """Compute various dataset statistics."""
    print(f"\n{name}:")

    # Basic stats
    print(f"  Entities: {n_ent}")
    print(f"  Relations: {n_rel}")
    print(f"  Train triples: {len(train)}")

    # Relation density
    rel_density = len(train) / n_rel
    print(f"  Triples per relation: {rel_density:.1f}")

    # Entity density
    ent_density = len(train) / n_ent
    print(f"  Triples per entity: {ent_density:.1f}")

    # Coverage density
    coverage = set()
    for h, r, t in train:
        coverage.add((int(h), int(r)))
        coverage.add((int(t), int(r)))
    possible_pairs = n_ent * n_rel
    cov_density = len(coverage) / possible_pairs
    print(f"  Coverage density: {cov_density:.4f} ({len(coverage)} / {possible_pairs})")

    # Relation distribution (Gini coefficient)
    rel_counts = {}
    for h, r, t in train:
        rel_counts[int(r)] = rel_counts.get(int(r), 0) + 1

    counts = sorted(rel_counts.values())
    n = len(counts)
    if n > 1:
        cumsum = np.cumsum(counts)
        gini = (2 * sum((i+1) * c for i, c in enumerate(counts)) / (n * sum(counts))) - (n+1)/n
        print(f"  Relation Gini (inequality): {gini:.3f}")

    # Average entity degree
    entity_degree = {}
    for h, r, t in train:
        entity_degree[int(h)] = entity_degree.get(int(h), 0) + 1
        entity_degree[int(t)] = entity_degree.get(int(t), 0) + 1
    avg_degree = np.mean(list(entity_degree.values()))
    print(f"  Avg entity degree: {avg_degree:.1f}")

    return {
        'n_ent': n_ent,
        'n_rel': n_rel,
        'n_train': len(train),
        'rel_density': rel_density,
        'ent_density': ent_density,
        'cov_density': cov_density,
        'avg_degree': avg_degree
    }


def main():
    print("="*60)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("="*60)

    stats = {}

    # FB15k-237
    ds = load_fb15k237()
    stats['FB15k-237'] = compute_dataset_stats('FB15k-237', ds[0].triples,
                                                ds[0].num_entities, ds[0].num_relations)

    # WN18RR
    ds = load_wn18rr()
    stats['WN18RR'] = compute_dataset_stats('WN18RR', ds[0].triples,
                                             ds[0].num_entities, ds[0].num_relations)

    # ICEWS14
    train, test, n_ent, n_rel = load_icews14()
    stats['ICEWS14'] = compute_dataset_stats('ICEWS14', train, n_ent, n_rel)

    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON: Which factors predict paradox?")
    print("="*60)

    print(f"\n{'Dataset':<12} {'Paradox?':<10} {'Rels':<8} {'Triples/Rel':<12} {'Cov Density':<12}")
    print("-"*54)

    paradox_map = {'FB15k-237': 'YES', 'WN18RR': 'NO', 'ICEWS14': 'NO'}

    for name in ['FB15k-237', 'WN18RR', 'ICEWS14']:
        s = stats[name]
        print(f"{name:<12} {paradox_map[name]:<10} {s['n_rel']:<8} {s['rel_density']:<12.1f} {s['cov_density']:<12.4f}")

    print("\n" + "="*60)
    print("HYPOTHESIS ANALYSIS")
    print("="*60)
    print("""
Looking for patterns that distinguish FB15k-237 (paradox) from others (no paradox):

1. RELATION COUNT: FB15k=237, ICEWS=226, WN18RR=11
   - ICEWS has similar count but no paradox
   - NOT the determining factor

2. TRIPLES PER RELATION: FB15k=1148, ICEWS=282, WN18RR=7894
   - FB15k is middle, not extreme
   - NOT clearly the factor

3. COVERAGE DENSITY: Check if FB15k has lower coverage density
   - Lower density → more novel contexts → more compositional generalization?

4. DOMAIN: FB15k=general facts, ICEWS=events, WN18RR=lexical
   - FB15k may have more compositional structure
   - Events/lexical may be more idiosyncratic
""")


if __name__ == "__main__":
    main()
