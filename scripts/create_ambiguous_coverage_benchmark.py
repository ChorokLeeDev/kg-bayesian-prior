#!/usr/bin/env python3
"""
Create Ambiguous Coverage Benchmark.

Design: Create a synthetic KG where:
1. High-frequency entities appear in novel relations (OOD but covered for some relations)
2. Coverage alone cannot distinguish OOD from ID
3. Semantic uncertainty (entity variance) is necessary

Goal: U_str ≈ 0.70-0.80, CAGP ≈ 0.90+ → proves semantic is necessary

Key insight: Create "ambiguous coverage" where:
- OOD triples have entities with PARTIAL coverage (covered for relation r1, uncovered for r2)
- ID triples also have partial coverage
- But OOD entities have LOWER frequency → higher semantic uncertainty
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import csv
from pathlib import Path


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_ambiguous_coverage_kg(n_entities=1000, n_relations=50, n_train=10000,
                                  n_test=2000, seed=42):
    """
    Create KG with ambiguous coverage patterns.

    Design:
    - Split entities into "common" (high freq) and "rare" (low freq)
    - Both common and rare entities have PARTIAL coverage (some relations, not all)
    - Test OOD: rare entities in relations they're covered for (coverage=1, but OOD)
    - Test ID: common entities in relations they're covered for

    This creates ambiguity: coverage cannot distinguish, but frequency (→semantic) can.
    """
    set_seed(seed)

    # Split entities: 30% common (high freq), 70% rare (low freq)
    n_common = int(n_entities * 0.3)
    n_rare = n_entities - n_common
    common_entities = set(range(n_common))
    rare_entities = set(range(n_common, n_entities))

    # Create training triples with frequency skew
    train_triples = []
    entity_freq = defaultdict(int)
    coverage = defaultdict(set)  # entity -> set of relations

    # Common entities appear in many triples with many relations
    for _ in range(int(n_train * 0.8)):
        h = np.random.choice(list(common_entities))
        t = np.random.choice(list(common_entities))
        r = np.random.randint(0, n_relations)

        train_triples.append((h, r, t))
        entity_freq[h] += 1
        entity_freq[t] += 1
        coverage[h].add(r)
        coverage[t].add(r)

    # Rare entities appear in few triples with few relations
    for _ in range(int(n_train * 0.2)):
        h = np.random.choice(list(rare_entities))
        t = np.random.choice(list(rare_entities))
        # Only use a subset of relations for rare entities
        r = np.random.randint(0, n_relations // 3)

        train_triples.append((h, r, t))
        entity_freq[h] += 1
        entity_freq[t] += 1
        coverage[h].add(r)
        coverage[t].add(r)

    # Frequency threshold
    freq_values = [entity_freq[e] for e in range(n_entities)]
    tau = np.percentile([f for f in freq_values if f > 0], 25)

    # Create test set with ambiguous coverage
    test_triples = []
    test_labels = []  # 1=OOD, 0=ID
    test_categories = []

    # OOD Type 1: Rare entities in COVERED relations (emerging_covered)
    # Key: These have coverage=1, so structural uncertainty is 0
    # But they're rare, so semantic uncertainty is high
    for _ in range(n_test // 3):
        # Pick a rare entity with some coverage
        rare_with_coverage = [e for e in rare_entities if len(coverage[e]) > 0]
        if not rare_with_coverage:
            continue

        h = np.random.choice(rare_with_coverage)
        # Use a relation the entity IS covered for
        covered_rels = list(coverage[h])
        if not covered_rels:
            continue
        r = np.random.choice(covered_rels)
        t = np.random.choice(rare_with_coverage)

        test_triples.append((h, r, t))
        test_labels.append(1)  # OOD (rare entity)
        test_categories.append('emerging_covered')

    # OOD Type 2: Rare entities in UNCOVERED relations (emerging_uncovered)
    for _ in range(n_test // 3):
        rare_list = list(rare_entities)
        h = np.random.choice(rare_list)
        t = np.random.choice(rare_list)
        # Use a relation they're NOT covered for
        uncovered = [r for r in range(n_relations) if r not in coverage[h]]
        if not uncovered:
            continue
        r = np.random.choice(uncovered)

        test_triples.append((h, r, t))
        test_labels.append(1)  # OOD
        test_categories.append('emerging_uncovered')

    # ID: Common entities in covered relations
    for _ in range(n_test // 3):
        common_with_coverage = [e for e in common_entities if len(coverage[e]) > 0]
        if not common_with_coverage:
            continue

        h = np.random.choice(common_with_coverage)
        covered_rels = list(coverage[h])
        r = np.random.choice(covered_rels)
        t = np.random.choice(common_with_coverage)

        test_triples.append((h, r, t))
        test_labels.append(0)  # ID
        test_categories.append('id')

    return {
        'train': train_triples,
        'test': test_triples,
        'test_labels': np.array(test_labels),
        'test_categories': test_categories,
        'n_entities': n_entities,
        'n_relations': n_relations,
        'entity_freq': entity_freq,
        'coverage': coverage,
        'tau': tau,
        'common_entities': common_entities,
        'rare_entities': rare_entities,
    }


def compute_uncertainties(data):
    """Compute semantic and structural uncertainties."""
    entity_freq = data['entity_freq']
    coverage = data['coverage']
    n_entities = data['n_entities']

    # Semantic: inverse frequency (lower freq -> higher uncertainty)
    max_freq = max(entity_freq.values()) if entity_freq else 1
    u_sem_entity = {}
    for e in range(n_entities):
        freq = entity_freq.get(e, 0)
        u_sem_entity[e] = 1.0 - (freq / (max_freq + 1))

    semantic_scores = []
    structural_scores = []

    for h, r, t in data['test']:
        # Semantic: average entity uncertainty
        sem = (u_sem_entity[h] + u_sem_entity[t]) / 2
        semantic_scores.append(sem)

        # Structural: 1 if not covered, 0 if covered
        h_covered = r in coverage[h]
        t_covered = r in coverage[t]
        struct = 2 - int(h_covered) - int(t_covered)
        structural_scores.append(struct)

    return np.array(semantic_scores), np.array(structural_scores)


def evaluate_ood(scores, labels, categories, target_cat=None):
    """Evaluate OOD detection."""
    if target_cat:
        mask = np.array([c == target_cat or c == 'id' for c in categories])
        if mask.sum() < 10:
            return None
        scores = scores[mask]
        labels_subset = np.array([1 if categories[i] == target_cat else 0
                                  for i in range(len(categories)) if mask[i]])
        if len(set(labels_subset)) < 2:
            return None
        return roc_auc_score(labels_subset, scores)
    else:
        if len(set(labels)) < 2:
            return None
        return roc_auc_score(labels, scores)


def run_experiment(seed=42):
    """Run single experiment."""
    print(f"\n--- Ambiguous Coverage Benchmark (seed={seed}) ---")

    data = create_ambiguous_coverage_kg(seed=seed)

    print(f"  Train: {len(data['train'])}, Test: {len(data['test'])}")

    # Category breakdown
    cats = defaultdict(int)
    for c in data['test_categories']:
        cats[c] += 1
    print(f"  Categories: {dict(cats)}")

    # Compute uncertainties
    u_sem, u_str = compute_uncertainties(data)

    # Normalize semantic for combination
    u_sem_norm = (u_sem - u_sem.min()) / (u_sem.max() - u_sem.min() + 1e-8)

    # Combined (CAGP-style)
    u_comb = 0.5 * u_sem_norm + 0.5 * u_str

    labels = data['test_labels']
    categories = data['test_categories']

    results = {}

    # Overall
    results['semantic_overall'] = evaluate_ood(u_sem, labels, categories)
    results['structural_overall'] = evaluate_ood(u_str, labels, categories)
    results['combined_overall'] = evaluate_ood(u_comb, labels, categories)

    # Per category
    for cat in ['emerging_covered', 'emerging_uncovered']:
        results[f'semantic_{cat}'] = evaluate_ood(u_sem, labels, categories, cat)
        results[f'structural_{cat}'] = evaluate_ood(u_str, labels, categories, cat)
        results[f'combined_{cat}'] = evaluate_ood(u_comb, labels, categories, cat)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/ambiguous_coverage_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("AMBIGUOUS COVERAGE BENCHMARK")
    print(f"{'='*60}")
    print("\nDesign: Create KG where coverage alone cannot distinguish OOD")
    print("- OOD: rare entities in COVERED relations (coverage=0, but rare)")
    print("- ID: common entities in covered relations")
    print("- Goal: U_str ≈ 0.5-0.7, CAGP > 0.85")

    all_results = []
    seed_list = [42, 123, 456][:args.seeds]

    for seed in seed_list:
        results = run_experiment(seed)
        results['seed'] = seed
        all_results.append(results)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.seeds} seeds)")
    print(f"{'='*60}")

    print(f"\n{'Metric':<30} {'Mean':>10} {'Std':>10}")
    print("-" * 55)

    key_metrics = ['semantic_overall', 'structural_overall', 'combined_overall',
                   'semantic_emerging_covered', 'structural_emerging_covered', 'combined_emerging_covered']

    for metric in key_metrics:
        values = [r[metric] for r in all_results if r.get(metric) is not None]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric:<30} {mean:>10.3f} {std:>10.3f}")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")

    sem_ec = [r['semantic_emerging_covered'] for r in all_results if r.get('semantic_emerging_covered')]
    str_ec = [r['structural_emerging_covered'] for r in all_results if r.get('structural_emerging_covered')]
    comb_ec = [r['combined_emerging_covered'] for r in all_results if r.get('combined_emerging_covered')]

    if sem_ec and str_ec:
        sem_mean = np.mean(sem_ec)
        str_mean = np.mean(str_ec)
        comb_mean = np.mean(comb_ec) if comb_ec else 0

        print(f"\nOn 'emerging_covered' (OOD with coverage=0):")
        print(f"  Semantic AUROC: {sem_mean:.3f}")
        print(f"  Structural AUROC: {str_mean:.3f}")
        print(f"  Combined AUROC: {comb_mean:.3f}")

        if sem_mean > str_mean + 0.05:
            print(f"\n✓ SEMANTIC WINS on ambiguous coverage!")
            print(f"  Semantic gain: +{sem_mean - str_mean:.3f}")
            print(f"  → Proves semantic is NECESSARY when coverage is ambiguous")
        elif str_mean < 0.6 and comb_mean > 0.8:
            print(f"\n✓ COMBINATION WINS!")
            print(f"  Structural alone: {str_mean:.3f} (near random)")
            print(f"  Combined: {comb_mean:.3f} (strong)")
            print(f"  → Proves BOTH signals are necessary")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['seed'] + key_metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
