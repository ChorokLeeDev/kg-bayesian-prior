#!/usr/bin/env python3
"""Analyze WHY paradox occurs in multi-relational KGs but not in temporal/hierarchical."""
import os
import numpy as np
from collections import defaultdict
import functools
print = functools.partial(print, flush=True)

def load_triples(path):
    """Load triples from standard format."""
    triples = []
    for fname in ['train.txt', 'train.tsv']:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        triples.append((parts[0], parts[1], parts[2]))
            break
    return triples

def analyze_kg_structure(name, triples):
    """Compute structural metrics for a KG."""
    entities = set()
    relations = set()
    entity_relations = defaultdict(set)  # entity -> set of relations
    relation_counts = defaultdict(int)

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
        entity_relations[h].add(r)
        entity_relations[t].add(r)
        relation_counts[r] += 1

    # Metrics
    n_ent = len(entities)
    n_rel = len(relations)
    n_triples = len(triples)

    # Avg relations per entity
    rels_per_entity = [len(rels) for rels in entity_relations.values()]
    avg_rels = np.mean(rels_per_entity)
    max_rels = max(rels_per_entity)

    # Relation entropy (uniformity of relation distribution)
    rel_probs = np.array(list(relation_counts.values())) / n_triples
    entropy = -np.sum(rel_probs * np.log(rel_probs + 1e-10))
    max_entropy = np.log(n_rel)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Density
    density = n_triples / (n_ent * n_rel) if n_ent * n_rel > 0 else 0

    return {
        'entities': n_ent,
        'relations': n_rel,
        'triples': n_triples,
        'avg_rels_per_entity': avg_rels,
        'max_rels_per_entity': max_rels,
        'relation_entropy': entropy,
        'norm_entropy': norm_entropy,
        'density': density
    }

def main():
    base = '/Users/i767700/Github/kg-bayesian-prior/data/raw'

    # Datasets with known paradox status
    datasets = {
        # Paradox = True
        'fb15k-237': ('Encyclopedic', True),
        'fb15k': ('Encyclopedic', True),
        'codex-s': ('Encyclopedic', True),
        'codex-m': ('Encyclopedic', True),
        'codex-l': ('Encyclopedic', True),
        'yago3-10': ('Encyclopedic', True),
        # Paradox = False
        'wn18rr': ('Hierarchical', False),
        'wn18': ('Hierarchical', False),
        'icews14': ('Temporal', False),
        'icews18': ('Temporal', False),
        'gdelt': ('Temporal', False),
    }

    # Also check temporal from RE-Net
    temporal_paths = {
        'WIKI': ('temporal/renet_data/data/WIKI', 'Temporal', False),
        'YAGO-temp': ('temporal/renet_data/data/YAGO', 'Temporal', False),
    }

    results = []

    print("="*80)
    print("STRUCTURAL ANALYSIS: Why does paradox occur?")
    print("="*80)

    # Standard datasets
    for name, (kg_type, paradox) in datasets.items():
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue

        triples = load_triples(path)
        if not triples:
            continue

        metrics = analyze_kg_structure(name, triples)
        metrics['name'] = name
        metrics['type'] = kg_type
        metrics['paradox'] = paradox
        results.append(metrics)

    # Temporal datasets
    for name, (rel_path, kg_type, paradox) in temporal_paths.items():
        path = os.path.join(base, rel_path)
        if not os.path.exists(path):
            continue

        triples = load_triples(path)
        if not triples:
            continue

        metrics = analyze_kg_structure(name, triples)
        metrics['name'] = name
        metrics['type'] = kg_type
        metrics['paradox'] = paradox
        results.append(metrics)

    # Print results
    print(f"\n{'Dataset':<12} {'Type':<12} {'Paradox':<8} {'Rels':<6} {'AvgR/E':<8} {'Entropy':<8} {'Density':<10}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x['paradox'], reverse=True):
        print(f"{r['name']:<12} {r['type']:<12} {'YES' if r['paradox'] else 'NO':<8} "
              f"{r['relations']:<6} {r['avg_rels_per_entity']:<8.2f} {r['norm_entropy']:<8.2f} {r['density']:<10.6f}")

    # Statistical comparison
    paradox_group = [r for r in results if r['paradox']]
    normal_group = [r for r in results if not r['paradox']]

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    if paradox_group and normal_group:
        metrics_to_compare = ['relations', 'avg_rels_per_entity', 'norm_entropy', 'density']

        for metric in metrics_to_compare:
            p_vals = [r[metric] for r in paradox_group]
            n_vals = [r[metric] for r in normal_group]

            p_mean = np.mean(p_vals)
            n_mean = np.mean(n_vals)

            print(f"\n{metric}:")
            print(f"  Paradox group:    {p_mean:.4f} (n={len(p_vals)})")
            print(f"  Non-paradox group: {n_mean:.4f} (n={len(n_vals)})")
            print(f"  Ratio: {p_mean/n_mean:.2f}x" if n_mean > 0 else "  Ratio: N/A")

    # Key finding
    print("\n" + "="*80)
    print("KEY STRUCTURAL DIFFERENCES")
    print("="*80)

    if paradox_group and normal_group:
        p_rels = np.mean([r['relations'] for r in paradox_group])
        n_rels = np.mean([r['relations'] for r in normal_group])
        p_avg = np.mean([r['avg_rels_per_entity'] for r in paradox_group])
        n_avg = np.mean([r['avg_rels_per_entity'] for r in normal_group])

        print(f"\n1. RELATION COUNT: Paradox KGs have {p_rels/n_rels:.1f}x more relations")
        print(f"   - Paradox: avg {p_rels:.0f} relations")
        print(f"   - Normal:  avg {n_rels:.0f} relations")

        print(f"\n2. RELATIONS PER ENTITY: Paradox KGs have {p_avg/n_avg:.1f}x more relations per entity")
        print(f"   - Paradox: avg {p_avg:.2f} rels/entity")
        print(f"   - Normal:  avg {n_avg:.2f} rels/entity")

        print("\n3. INTERPRETATION:")
        print("   - Multi-relational KGs: Entities participate in MANY relation types")
        print("   - Full coverage = seen in ALL relations = model overfits to patterns")
        print("   - Partial coverage = novel combination = model must generalize")
        print("   - Temporal/Hierarchical: Few relations, coverage is meaningful signal")

if __name__ == '__main__':
    main()
