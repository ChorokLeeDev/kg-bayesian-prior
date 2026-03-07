#!/usr/bin/env python3
"""
Deep dive: Why U_sem shows 83% zero-evidence on FB15k-237 but only 7% on ICEWS14.

The freq-variance correlations are identical (-0.85 for both), so that's not the answer.
Let's investigate the coverage structure.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from scipy import stats

from src.data.loaders import load_fb15k237, load_icews14


def analyze_coverage_structure(dataset_name, loader):
    """Analyze the coverage structure of a dataset."""
    print(f"\n{'='*60}")
    print(f"  COVERAGE STRUCTURE: {dataset_name}")
    print(f"{'='*60}")

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Build coverage matrix
    coverage = np.zeros((n_ent, n_rel), dtype=np.float32)
    for h, r, t in train:
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0

    # Entity frequency
    entity_freq = np.zeros(n_ent)
    for h, r, t in train:
        entity_freq[h] += 1
        entity_freq[t] += 1

    # Coverage density per entity
    entity_coverage_count = coverage.sum(axis=1)  # How many relations each entity is covered for

    # Test set statistics
    zero_ev_test = []
    triple_freq_test = []
    for h, r, t in test:
        is_zero_ev = (coverage[h, r] == 0) or (coverage[t, r] == 0)
        zero_ev_test.append(is_zero_ev)
        triple_freq_test.append((entity_freq[h] + entity_freq[t]) / 2)
    zero_ev_test = np.array(zero_ev_test)
    triple_freq_test = np.array(triple_freq_test)

    # Key insight: What's the relationship between frequency and coverage?
    # High frequency entities should have HIGH coverage (many relations covered)
    freq_coverage_corr, _ = stats.spearmanr(entity_freq[entity_freq > 0],
                                             entity_coverage_count[entity_freq > 0])

    print(f"\n  Basic stats:")
    print(f"    Entities: {n_ent}, Relations: {n_rel}")
    print(f"    Train triples: {len(train)}")
    print(f"    Avg coverage per entity: {entity_coverage_count.mean():.1f} relations")
    print(f"    Max coverage per entity: {entity_coverage_count.max():.0f} relations")

    print(f"\n  Frequency-Coverage relationship:")
    print(f"    Spearman(freq, coverage_count): {freq_coverage_corr:.3f}")

    # Crucial: Among high-frequency entities, what % of test queries are zero-evidence?
    q75_freq = np.percentile(entity_freq[entity_freq > 0], 75)
    high_freq_entities = set(np.where(entity_freq >= q75_freq)[0])

    high_freq_test_mask = np.array([
        (test[i, 0] in high_freq_entities) and (test[i, 2] in high_freq_entities)
        for i in range(len(test))
    ])

    print(f"\n  High-frequency entity test queries:")
    print(f"    N test triples (both entities high-freq): {high_freq_test_mask.sum()}")
    if high_freq_test_mask.sum() > 0:
        print(f"    Zero-evidence rate: {100*zero_ev_test[high_freq_test_mask].mean():.1f}%")

    # The KEY difference: ratio of relations to entities
    # FB15k-237 has 237 relations for 14K entities (sparse coverage)
    # ICEWS14 has 230 relations for 7K entities (denser coverage)
    rel_to_ent_ratio = n_rel / n_ent
    print(f"\n  Relation/entity ratio: {rel_to_ent_ratio:.4f}")
    print(f"    (Higher = denser coverage possible)")

    # Coverage sparsity
    coverage_density = coverage.sum() / (n_ent * n_rel)
    print(f"    Coverage matrix density: {100*coverage_density:.2f}%")

    # Critical insight: Among the high-frequency entities,
    # what's the coverage rate (fraction of relations covered)?
    high_freq_ents = np.where(entity_freq >= q75_freq)[0]
    high_freq_coverage_rate = entity_coverage_count[high_freq_ents] / n_rel
    print(f"\n  High-freq entities coverage rate:")
    print(f"    Mean: {100*high_freq_coverage_rate.mean():.1f}% of relations covered")
    print(f"    Min: {100*high_freq_coverage_rate.min():.1f}%")
    print(f"    Max: {100*high_freq_coverage_rate.max():.1f}%")

    return {
        'n_ent': n_ent,
        'n_rel': n_rel,
        'rel_to_ent_ratio': rel_to_ent_ratio,
        'coverage_density': coverage_density,
        'freq_coverage_corr': freq_coverage_corr,
        'high_freq_coverage_mean': high_freq_coverage_rate.mean(),
        'high_freq_zero_ev_rate': zero_ev_test[high_freq_test_mask].mean() if high_freq_test_mask.sum() > 0 else None,
    }


def main():
    print("\n" + "="*80)
    print("DEEP DIVE: U_sem ANOMALY INVESTIGATION")
    print("="*80)
    print("\nWhy does U_sem show 83% zero-evidence on FB15k-237 but only 7% on ICEWS14?")
    print("The freq-variance correlation is the same (-0.85), so that's not the answer.")
    print("\nHypothesis: The coverage structure differs. U_sem's top-confident are")
    print("high-frequency entities. If high-freq entities have DIFFERENT coverage")
    print("patterns between datasets, that explains the difference.")

    fb_stats = analyze_coverage_structure("FB15k-237", load_fb15k237)
    icews_stats = analyze_coverage_structure("ICEWS14", load_icews14)

    print("\n" + "="*80)
    print("EXPLANATION")
    print("="*80)

    print(f"\n  KEY FINDING:")
    print(f"    FB15k-237 high-freq entities cover {100*fb_stats['high_freq_coverage_mean']:.1f}% of relations on average")
    print(f"    ICEWS14 high-freq entities cover {100*icews_stats['high_freq_coverage_mean']:.1f}% of relations on average")

    if fb_stats['high_freq_coverage_mean'] < icews_stats['high_freq_coverage_mean']:
        print(f"\n  ICEWS14's high-frequency entities have MUCH BETTER coverage.")
        print(f"  This means U_sem's top-confident (high-freq) predictions on ICEWS14")
        print(f"  are more likely to have training evidence.")

    if fb_stats['high_freq_zero_ev_rate'] is not None and icews_stats['high_freq_zero_ev_rate'] is not None:
        print(f"\n  Zero-evidence rate among high-freq entity test queries:")
        print(f"    FB15k-237: {100*fb_stats['high_freq_zero_ev_rate']:.1f}%")
        print(f"    ICEWS14:   {100*icews_stats['high_freq_zero_ev_rate']:.1f}%")


if __name__ == "__main__":
    main()
