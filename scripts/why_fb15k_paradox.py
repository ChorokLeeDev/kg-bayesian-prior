#!/usr/bin/env python3
"""
1. WHY FB15k-237 only shows paradox?

Hypothesis: FB15k-237 has more "compositional" structure
- Relations that can be inferred from entity types
- Regular patterns that transfer to unseen pairs

Test: Analyze relation compositionality
"""

import torch
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
    return train, len(entity2id), len(relation2id)


def analyze_compositionality(name, train, n_ent, n_rel):
    """
    Measure how "compositional" a KG is.

    Compositionality = can you predict (h, r, ?) from knowing r and type(h)?

    Proxy metrics:
    1. Tail entropy per relation: Low = predictable, High = idiosyncratic
    2. Head-tail correlation: High = compositional patterns
    3. Relation regularity: Does same head pattern → same tail pattern?
    """
    print(f"\n{'='*60}")
    print(f"{name}: Compositionality Analysis")
    print(f"{'='*60}")

    # Per-relation tail distribution
    rel_tails = {}  # r -> {t -> count}
    rel_heads = {}  # r -> {h -> count}

    for h, r, t in train:
        r, h, t = int(r), int(h), int(t)
        if r not in rel_tails:
            rel_tails[r] = {}
            rel_heads[r] = {}
        rel_tails[r][t] = rel_tails[r].get(t, 0) + 1
        rel_heads[r][h] = rel_heads[r].get(h, 0) + 1

    # 1. Average tail entropy per relation
    def entropy(counts):
        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total
        return -np.sum(probs * np.log(probs + 1e-10))

    tail_entropies = [entropy(tails) for tails in rel_tails.values() if len(tails) > 1]
    avg_tail_entropy = np.mean(tail_entropies)
    print(f"  Avg tail entropy per relation: {avg_tail_entropy:.2f}")
    print(f"    (Lower = more predictable tails)")

    # 2. Tail concentration: What fraction of tails appear in top-10?
    tail_concentrations = []
    for r, tails in rel_tails.items():
        if len(tails) > 10:
            sorted_counts = sorted(tails.values(), reverse=True)
            top10_frac = sum(sorted_counts[:10]) / sum(sorted_counts)
            tail_concentrations.append(top10_frac)

    avg_concentration = np.mean(tail_concentrations) if tail_concentrations else 0
    print(f"  Avg top-10 tail concentration: {avg_concentration:.1%}")
    print(f"    (Higher = fewer unique tails dominate)")

    # 3. Relation "regularity": Do entities that share one relation tend to share others?
    # Compute entity-pair co-occurrence across relations
    entity_relations = {}  # e -> set of relations
    for h, r, t in train:
        h, r, t = int(h), int(r), int(t)
        if h not in entity_relations:
            entity_relations[h] = set()
        if t not in entity_relations:
            entity_relations[t] = set()
        entity_relations[h].add(r)
        entity_relations[t].add(r)

    # Average relations per entity
    avg_rels_per_entity = np.mean([len(rels) for rels in entity_relations.values()])
    print(f"  Avg relations per entity: {avg_rels_per_entity:.2f}")
    print(f"    (Higher = entities appear in more contexts)")

    # 4. "Type regularity": Do entities with similar relation profiles have similar tails?
    # Simplified: correlation between head relation set and tail relation set

    # 5. Inverse relation ratio
    # Check if (h, r, t) often implies (t, r', h) for some r'
    pair_relations = {}  # (h, t) -> set of relations
    for h, r, t in train:
        h, r, t = int(h), int(r), int(t)
        if (h, t) not in pair_relations:
            pair_relations[(h, t)] = set()
        if (t, h) not in pair_relations:
            pair_relations[(t, h)] = set()
        pair_relations[(h, t)].add(r)

    # Count bidirectional pairs
    bidirectional = sum(1 for (h, t) in pair_relations if (t, h) in pair_relations and pair_relations[(t,h)])
    total_pairs = len(pair_relations)
    bidirectional_ratio = bidirectional / total_pairs if total_pairs > 0 else 0
    print(f"  Bidirectional pair ratio: {bidirectional_ratio:.1%}")
    print(f"    (Higher = more symmetric/inverse relations)")

    return {
        'tail_entropy': avg_tail_entropy,
        'tail_concentration': avg_concentration,
        'rels_per_entity': avg_rels_per_entity,
        'bidirectional_ratio': bidirectional_ratio
    }


def main():
    print("="*60)
    print("WHY DOES FB15k-237 SHOW PARADOX?")
    print("="*60)

    results = {}

    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = analyze_compositionality('FB15k-237', ds[0].triples,
                                                     ds[0].num_entities, ds[0].num_relations)

    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = analyze_compositionality('WN18RR', ds[0].triples,
                                                  ds[0].num_entities, ds[0].num_relations)

    # ICEWS14
    train, n_ent, n_rel = load_icews14()
    results['ICEWS14'] = analyze_compositionality('ICEWS14', train, n_ent, n_rel)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What makes FB15k-237 different?")
    print("="*60)

    print(f"\n{'Dataset':<12} {'Paradox':<8} {'Tail Ent':<10} {'Conc':<8} {'Rels/Ent':<10} {'Bidir':<8}")
    print("-"*56)

    paradox = {'FB15k-237': 'YES', 'WN18RR': 'NO', 'ICEWS14': 'NO'}

    for name in ['FB15k-237', 'WN18RR', 'ICEWS14']:
        r = results[name]
        print(f"{name:<12} {paradox[name]:<8} {r['tail_entropy']:<10.2f} {r['tail_concentration']:<8.1%} {r['rels_per_entity']:<10.2f} {r['bidirectional_ratio']:<8.1%}")

    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    print("""
If FB15k-237 has:
- LOWER tail entropy → more predictable patterns
- HIGHER tail concentration → common tails dominate
- HIGHER rels per entity → entities are more "typed"

Then partial coverage works because:
- One anchor entity constrains predictions via type
- Compositional patterns transfer to unseen pairs
""")


if __name__ == "__main__":
    main()
