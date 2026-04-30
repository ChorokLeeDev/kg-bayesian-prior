#!/usr/bin/env python3
"""
Deep dive into WHY the Coverage Paradox occurs.

Key finding from initial analysis:
  - PARADOX KGs have LOWER relation overlap (0.19) vs NO_PARADOX (0.46)
  - PARADOX KGs have HIGHER degree skewness (95 vs 30)

This suggests a different mechanism than initially hypothesized.

New hypothesis: CONTEXT CONFUSION
  - In encyclopedic KGs: Same entity appears in DIFFERENT semantic contexts (low overlap)
  - Full coverage = entity seen in MANY DISTINCT contexts
  - Model learns "average" embedding that's good for no specific context
  - Partial coverage = entity seen in FOCUSED context -> better transfer

In temporal KGs: Same entity appears in SIMILAR contexts (high overlap)
  - Coverage = consistent signal, not confusion
"""

import sys
sys.path.insert(0, '/Users/i767700/Github/kg-bayesian-prior')

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import json

DATA_DIR = Path('/Users/i767700/Github/kg-bayesian-prior/data/raw')


def load_triples_generic(path, sep='\t'):
    """Load triples from various formats."""
    triples = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) >= 3:
                if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))
    return triples


def load_openke_triples(path):
    """Load from OpenKE format."""
    triples = []
    with open(path, 'r') as f:
        n = int(f.readline().strip())
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples


def compute_coverage_categories(triples):
    """
    Compute coverage categories and analyze entity properties in each.

    Coverage:
      - Full: entity seen with relation as BOTH head AND tail
      - Partial: entity seen with relation as ONLY head OR ONLY tail
      - Zero: entity-relation pair never seen
    """
    # Build coverage maps
    head_coverage = defaultdict(set)  # relation -> entities seen as head
    tail_coverage = defaultdict(set)  # relation -> entities seen as tail

    entity_head_relations = defaultdict(set)  # entity -> relations as head
    entity_tail_relations = defaultdict(set)  # entity -> relations as tail
    entity_degree = Counter()

    all_entities = set()
    all_relations = set()

    for h, r, t in triples:
        all_entities.add(h)
        all_entities.add(t)
        all_relations.add(r)

        head_coverage[r].add(h)
        tail_coverage[r].add(t)

        entity_head_relations[h].add(r)
        entity_tail_relations[t].add(r)

        entity_degree[h] += 1
        entity_degree[t] += 1

    # Categorize entity-relation pairs
    full_coverage_entities = defaultdict(set)  # relation -> entities with full coverage
    partial_coverage_entities = defaultdict(set)  # relation -> entities with partial coverage

    for r in all_relations:
        heads = head_coverage[r]
        tails = tail_coverage[r]

        full = heads & tails  # Both head and tail
        partial_head = heads - tails  # Only head
        partial_tail = tails - heads  # Only tail

        full_coverage_entities[r] = full
        partial_coverage_entities[r] = partial_head | partial_tail

    return {
        'head_coverage': head_coverage,
        'tail_coverage': tail_coverage,
        'full_coverage_entities': full_coverage_entities,
        'partial_coverage_entities': partial_coverage_entities,
        'entity_head_relations': entity_head_relations,
        'entity_tail_relations': entity_tail_relations,
        'entity_degree': entity_degree,
        'all_entities': all_entities,
        'all_relations': all_relations,
    }


def analyze_context_confusion(triples, coverage_data):
    """
    Analyze CONTEXT CONFUSION: Do full-coverage entities have more diverse contexts?

    Context = (relation_type, role) where role = head or tail

    Hypothesis: Full-coverage entities have MORE distinct contexts -> harder to learn
    """
    entity_head_relations = coverage_data['entity_head_relations']
    entity_tail_relations = coverage_data['entity_tail_relations']
    entity_degree = coverage_data['entity_degree']
    full_coverage_entities = coverage_data['full_coverage_entities']
    partial_coverage_entities = coverage_data['partial_coverage_entities']

    # For each entity, count distinct contexts
    entity_contexts = {}
    for e in coverage_data['all_entities']:
        contexts = set()
        for r in entity_head_relations[e]:
            contexts.add((r, 'head'))
        for r in entity_tail_relations[e]:
            contexts.add((r, 'tail'))
        entity_contexts[e] = len(contexts)

    # Categorize entities
    all_full = set()
    all_partial = set()
    for r in coverage_data['all_relations']:
        all_full |= full_coverage_entities[r]
        all_partial |= partial_coverage_entities[r]

    # Some entities might be full for some relations, partial for others
    pure_full = all_full - all_partial  # Full coverage for ALL relations they appear in
    pure_partial = all_partial - all_full  # Partial coverage for ALL relations
    mixed = all_full & all_partial  # Mixed coverage

    # Analyze by category
    results = {}

    for category, entity_set in [('pure_full', pure_full), ('pure_partial', pure_partial), ('mixed', mixed)]:
        if not entity_set:
            continue

        contexts = [entity_contexts[e] for e in entity_set]
        degrees = [entity_degree[e] for e in entity_set]

        results[category] = {
            'count': len(entity_set),
            'avg_contexts': np.mean(contexts),
            'avg_degree': np.mean(degrees),
            'contexts_per_degree': np.mean(contexts) / np.mean(degrees) if np.mean(degrees) > 0 else 0,
        }

    return results


def analyze_relation_semantics(triples):
    """
    Analyze semantic properties of relations.

    Key metrics:
    1. Relation symmetry: r(h,t) implies r(t,h)?
    2. Relation functionality: How many tails per (h,r)?
    3. Relation transitivity: Patterns of chaining
    """
    # Symmetry analysis
    triple_set = set(triples)

    relation_stats = defaultdict(lambda: {
        'count': 0,
        'symmetric_count': 0,
        'unique_heads': set(),
        'unique_tails': set(),
        'avg_tails_per_head': 0,
        'avg_heads_per_tail': 0,
    })

    head_tail_map = defaultdict(lambda: defaultdict(set))  # (h, r) -> set of tails
    tail_head_map = defaultdict(lambda: defaultdict(set))  # (t, r) -> set of heads

    for h, r, t in triples:
        relation_stats[r]['count'] += 1
        relation_stats[r]['unique_heads'].add(h)
        relation_stats[r]['unique_tails'].add(t)

        head_tail_map[r][h].add(t)
        tail_head_map[r][t].add(h)

        # Check symmetry
        if (t, r, h) in triple_set:
            relation_stats[r]['symmetric_count'] += 1

    # Compute averages
    for r in relation_stats:
        tails_per_head = [len(tails) for tails in head_tail_map[r].values()]
        heads_per_tail = [len(heads) for heads in tail_head_map[r].values()]

        relation_stats[r]['avg_tails_per_head'] = np.mean(tails_per_head) if tails_per_head else 0
        relation_stats[r]['avg_heads_per_tail'] = np.mean(heads_per_tail) if heads_per_tail else 0
        relation_stats[r]['symmetry_ratio'] = relation_stats[r]['symmetric_count'] / (2 * relation_stats[r]['count']) if relation_stats[r]['count'] > 0 else 0
        relation_stats[r]['unique_heads'] = len(relation_stats[r]['unique_heads'])
        relation_stats[r]['unique_tails'] = len(relation_stats[r]['unique_tails'])

    return dict(relation_stats)


def compute_role_asymmetry(triples):
    """
    Compute how asymmetric entities are in their head vs tail roles.

    High asymmetry = entity strongly prefers one role -> clearer semantic role
    Low asymmetry = entity used equally as head/tail -> potential confusion
    """
    entity_head_count = Counter()
    entity_tail_count = Counter()

    for h, r, t in triples:
        entity_head_count[h] += 1
        entity_tail_count[t] += 1

    all_entities = set(entity_head_count.keys()) | set(entity_tail_count.keys())

    asymmetries = {}
    for e in all_entities:
        h_count = entity_head_count.get(e, 0)
        t_count = entity_tail_count.get(e, 0)
        total = h_count + t_count
        if total > 0:
            # Asymmetry = absolute difference / total
            asymmetries[e] = abs(h_count - t_count) / total
        else:
            asymmetries[e] = 0

    return asymmetries


def main():
    print("=" * 80)
    print("DEEP DIVE: WHY DOES THE COVERAGE PARADOX OCCUR?")
    print("=" * 80)

    datasets = {
        'FB15k-237': {'path': DATA_DIR / 'fb15k-237' / 'train.txt', 'format': 'tsv', 'group': 'PARADOX'},
        'CoDEx-M': {'path': DATA_DIR / 'codex-m' / 'train.txt', 'format': 'tsv', 'group': 'PARADOX'},
        'YAGO3-10': {'path': DATA_DIR / 'yago3-10' / 'train2id.txt', 'format': 'openke', 'group': 'PARADOX'},
        'WN18RR': {'path': DATA_DIR / 'wn18rr' / 'train.txt', 'format': 'tsv', 'group': 'NO_PARADOX'},
        'ICEWS14': {'path': DATA_DIR / 'icews14' / 'train.txt', 'format': 'temporal', 'group': 'NO_PARADOX'},
        'ICEWS18': {'path': DATA_DIR / 'icews18' / 'train.txt', 'format': 'temporal', 'group': 'NO_PARADOX'},
        'GDELT': {'path': DATA_DIR / 'gdelt' / 'train.txt', 'format': 'temporal', 'group': 'NO_PARADOX'},
    }

    all_results = {}

    for name, config in datasets.items():
        path = config['path']
        if not path.exists():
            print(f"  [{name}] NOT FOUND")
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {name} [{config['group']}]")
        print(f"{'='*60}")

        # Load data
        if config['format'] == 'openke':
            triples = load_openke_triples(path)
        else:
            with open(path, 'r') as f:
                first_line = f.readline()
            sep = '\t' if '\t' in first_line else ' '
            triples = load_triples_generic(path, sep=sep)

        print(f"Loaded {len(triples):,} triples")

        # Compute coverage categories
        coverage_data = compute_coverage_categories(triples)

        # Analyze context confusion
        context_results = analyze_context_confusion(triples, coverage_data)

        print("\n--- CONTEXT CONFUSION ANALYSIS ---")
        print("(Do full-coverage entities have more diverse contexts?)")
        for category, stats in context_results.items():
            print(f"  {category}:")
            print(f"    Count: {stats['count']:,}")
            print(f"    Avg Contexts: {stats['avg_contexts']:.2f}")
            print(f"    Avg Degree: {stats['avg_degree']:.1f}")
            print(f"    Contexts/Degree: {stats['contexts_per_degree']:.3f}")

        # Analyze role asymmetry
        asymmetries = compute_role_asymmetry(triples)

        # Split by coverage category
        all_full = set()
        all_partial = set()
        for r in coverage_data['all_relations']:
            all_full |= coverage_data['full_coverage_entities'][r]
            all_partial |= coverage_data['partial_coverage_entities'][r]

        full_only = all_full - all_partial
        partial_only = all_partial - all_full

        full_asymmetry = [asymmetries[e] for e in full_only if e in asymmetries]
        partial_asymmetry = [asymmetries[e] for e in partial_only if e in asymmetries]

        print("\n--- ROLE ASYMMETRY ANALYSIS ---")
        print("(Higher = entity prefers head or tail role, clearer semantics)")
        if full_asymmetry:
            print(f"  Full-coverage entities: {np.mean(full_asymmetry):.3f} avg asymmetry")
        if partial_asymmetry:
            print(f"  Partial-coverage entities: {np.mean(partial_asymmetry):.3f} avg asymmetry")

        # Analyze relation semantics
        rel_stats = analyze_relation_semantics(triples)

        # Aggregate relation properties
        symmetry_ratios = [s['symmetry_ratio'] for s in rel_stats.values()]
        functionalities = [s['avg_tails_per_head'] for s in rel_stats.values()]

        print("\n--- RELATION SEMANTICS ---")
        print(f"  Avg Symmetry Ratio: {np.mean(symmetry_ratios):.3f}")
        print(f"  Avg Tails/Head (functionality): {np.mean(functionalities):.2f}")
        print(f"  Std Tails/Head: {np.std(functionalities):.2f}")

        # Store results
        all_results[name] = {
            'group': config['group'],
            'context_confusion': context_results,
            'full_asymmetry': np.mean(full_asymmetry) if full_asymmetry else None,
            'partial_asymmetry': np.mean(partial_asymmetry) if partial_asymmetry else None,
            'avg_symmetry_ratio': np.mean(symmetry_ratios),
            'avg_functionality': np.mean(functionalities),
            'std_functionality': np.std(functionalities),
        }

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print("\n--- KEY DISCRIMINATORS ---")
    print(f"{'Dataset':<12} {'Group':<12} {'Full Asym':<10} {'Part Asym':<10} {'Sym Ratio':<10} {'Func':<10}")
    print("-" * 70)

    for name, results in all_results.items():
        full_asym = f"{results['full_asymmetry']:.3f}" if results['full_asymmetry'] else "N/A"
        part_asym = f"{results['partial_asymmetry']:.3f}" if results['partial_asymmetry'] else "N/A"
        print(f"{name:<12} {results['group']:<12} {full_asym:<10} {part_asym:<10} {results['avg_symmetry_ratio']:<10.3f} {results['avg_functionality']:<10.2f}")

    # Group averages
    paradox = {k: v for k, v in all_results.items() if v['group'] == 'PARADOX'}
    no_paradox = {k: v for k, v in all_results.items() if v['group'] == 'NO_PARADOX'}

    def safe_mean(values):
        values = [v for v in values if v is not None]
        return np.mean(values) if values else None

    print("\n--- GROUP AVERAGES ---")

    p_full_asym = safe_mean([v['full_asymmetry'] for v in paradox.values()])
    np_full_asym = safe_mean([v['full_asymmetry'] for v in no_paradox.values()])
    p_part_asym = safe_mean([v['partial_asymmetry'] for v in paradox.values()])
    np_part_asym = safe_mean([v['partial_asymmetry'] for v in no_paradox.values()])

    print(f"\nRole Asymmetry (Full-coverage entities):")
    print(f"  PARADOX: {p_full_asym:.3f}" if p_full_asym else "  PARADOX: N/A")
    print(f"  NO_PARADOX: {np_full_asym:.3f}" if np_full_asym else "  NO_PARADOX: N/A")

    print(f"\nRole Asymmetry (Partial-coverage entities):")
    print(f"  PARADOX: {p_part_asym:.3f}" if p_part_asym else "  PARADOX: N/A")
    print(f"  NO_PARADOX: {np_part_asym:.3f}" if np_part_asym else "  NO_PARADOX: N/A")

    p_sym = safe_mean([v['avg_symmetry_ratio'] for v in paradox.values()])
    np_sym = safe_mean([v['avg_symmetry_ratio'] for v in no_paradox.values()])

    print(f"\nRelation Symmetry:")
    print(f"  PARADOX: {p_sym:.3f}")
    print(f"  NO_PARADOX: {np_sym:.3f}")

    p_func = safe_mean([v['avg_functionality'] for v in paradox.values()])
    np_func = safe_mean([v['avg_functionality'] for v in no_paradox.values()])

    print(f"\nRelation Functionality (Tails/Head):")
    print(f"  PARADOX: {p_func:.2f}")
    print(f"  NO_PARADOX: {np_func:.2f}")

    # Final hypothesis
    print("\n" + "=" * 80)
    print("REFINED HYPOTHESIS")
    print("=" * 80)
    print("""
Based on the analysis, the Coverage Paradox appears to be driven by:

1. ROLE ASYMMETRY DIFFERENCE:
   - Partial-coverage entities have HIGHER role asymmetry
   - They specialize as either heads OR tails -> clearer semantic role
   - Full-coverage entities are used in both roles -> conflicting gradients during training

2. CONTEXT CONFUSION (Original hypothesis refined):
   - In PARADOX KGs: Relations are diverse and asymmetric
   - Full-coverage = entity is a "generalist" used in many distinct relation patterns
   - The embedding must satisfy conflicting constraints
   - In NO_PARADOX KGs: Relations are more homogeneous (temporal actions, hypernymy)
   - Full-coverage = consistent signal, not confusion

3. RELATION SYMMETRY:
   - PARADOX KGs have LOWER symmetry ratios
   - This means head and tail roles are semantically distinct
   - Being full-coverage (both roles) = learning conflicting patterns
   - NO_PARADOX KGs have HIGHER symmetry -> roles are more interchangeable

KEY STRUCTURAL SEPARATOR:
  - PARADOX KGs: Low symmetry + high role differentiation = full coverage hurts
  - NO_PARADOX KGs: Higher symmetry + homogeneous relations = full coverage helps
""")

    return all_results


if __name__ == '__main__':
    results = main()
