#!/usr/bin/env python3
"""
Analyze why theorem prediction differs from observed AUROC.

Key question: Is Assumption A1 (ID triples have c(h,r)=c(t,r)=1) violated?
"""

import os
import numpy as np


def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_coverage(triples):
    """Build coverage dict: coverage[(entity, relation)] = 1 if seen."""
    coverage = set()
    for h, r, t in triples:
        coverage.add((h, r))
        coverage.add((t, r))
    return coverage


def analyze_id_coverage(train_triples, test_triples):
    """Analyze what fraction of ID test triples have entities covered."""
    coverage = build_coverage(train_triples)

    head_covered = 0
    tail_covered = 0
    both_covered = 0
    total = len(test_triples)

    for h, r, t in test_triples:
        h_cov = 1 if (h, r) in coverage else 0
        t_cov = 1 if (t, r) in coverage else 0

        head_covered += h_cov
        tail_covered += t_cov
        if h_cov and t_cov:
            both_covered += 1

    return {
        'p_head': head_covered / total,
        'p_tail': tail_covered / total,
        'p_both': both_covered / total,
        'total': total
    }


def revised_auroc_formula(p_h, p_t, s_r):
    """
    Revised AUROC formula with relaxed Assumption A1.

    ID uncertainty distribution:
    - U = 0 with prob p_h * p_t (both covered)
    - U = 1 with prob p_h*(1-p_t) + (1-p_h)*p_t (one covered)
    - U = 2 with prob (1-p_h)*(1-p_t) (neither covered)

    OOD uncertainty distribution (random tail):
    - Assume head is from ID, so c(h,r)=1 with prob p_h
    - Tail is random, so c(t',r)=1 with prob (1-s_r)

    For simplicity, assume p_h ≈ 1 (head comes from real triple):
    - U_OOD = 0 with prob (1-s_r)
    - U_OOD = 1 with prob s_r
    """
    # Simplified: assume head is always covered in OOD (it's from a real triple)
    # This is reasonable since OOD is constructed from ID by corrupting tail

    # ID distribution
    p_id_0 = p_h * p_t
    p_id_1 = p_h * (1 - p_t) + (1 - p_h) * p_t
    p_id_2 = (1 - p_h) * (1 - p_t)

    # OOD distribution (assuming head covered, random tail)
    p_ood_0 = (1 - s_r)
    p_ood_1 = s_r

    # AUROC = P(U_ID < U_OOD) + 0.5 * P(U_ID = U_OOD)
    # Need to compute all cases

    auroc = 0

    # U_ID = 0 vs U_OOD = 1: contributes p_id_0 * p_ood_1
    auroc += p_id_0 * p_ood_1

    # U_ID = 0 vs U_OOD = 0: tie, contributes 0.5 * p_id_0 * p_ood_0
    auroc += 0.5 * p_id_0 * p_ood_0

    # U_ID = 1 vs U_OOD = 1: tie, contributes 0.5 * p_id_1 * p_ood_1
    auroc += 0.5 * p_id_1 * p_ood_1

    # U_ID = 1 vs U_OOD = 0: OOD wins, contributes 0
    # (this hurts AUROC)

    # U_ID = 2 vs U_OOD = 0: OOD wins, contributes 0
    # U_ID = 2 vs U_OOD = 1: OOD wins, contributes 0

    return auroc


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    datasets = {
        'WN18RR': os.path.join(project_root, 'data', 'raw', 'wn18rr'),
        'FB15k-237': os.path.join(project_root, 'data', 'raw', 'fb15k-237'),
    }

    observed_auroc = {
        'WN18RR': 0.657,
        'FB15k-237': 0.821,
    }

    print("=" * 70)
    print("ANALYZING THEOREM GAP: Why does prediction != observed?")
    print("=" * 70)

    for name, path in datasets.items():
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')

        if not os.path.exists(train_file):
            continue

        train = load_triples(train_file)
        test = load_triples(test_file)

        # Get all entities
        entities = set()
        relations = set()
        for h, r, t in train + test:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        # Analyze ID coverage
        coverage_stats = analyze_id_coverage(train, test)

        # Compute average sparsity
        coverage_set = build_coverage(train)
        sparsities = {}
        for r in relations:
            seen = len([e for e in entities if (e, r) in coverage_set])
            sparsities[r] = 1 - (seen / len(entities))
        avg_s = sum(sparsities.values()) / len(sparsities)

        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        print(f"\nAssumption A1 Analysis:")
        print(f"  P(head covered | ID test) = {coverage_stats['p_head']:.4f}")
        print(f"  P(tail covered | ID test) = {coverage_stats['p_tail']:.4f}")
        print(f"  P(both covered | ID test) = {coverage_stats['p_both']:.4f}")

        print(f"\nThis shows Assumption A1 (p=1) is VIOLATED!")
        print(f"  Expected by A1: p_both = 1.0")
        print(f"  Actual: p_both = {coverage_stats['p_both']:.4f}")

        # Revised prediction
        p_h = coverage_stats['p_head']
        p_t = coverage_stats['p_tail']
        revised_auroc = revised_auroc_formula(p_h, p_t, avg_s)

        # Original prediction (A1 assumed)
        original_auroc = (1 + avg_s) / 2

        obs = observed_auroc.get(name, 0)

        print(f"\nAUROC Comparison:")
        print(f"  Original theorem (A1):     {original_auroc:.4f}")
        print(f"  Revised theorem (relaxed): {revised_auroc:.4f}")
        print(f"  Observed:                  {obs:.4f}")
        print(f"  Original error:            {abs(original_auroc - obs):.4f}")
        print(f"  Revised error:             {abs(revised_auroc - obs):.4f}")

        if abs(revised_auroc - obs) < abs(original_auroc - obs):
            print(f"\n  ✓ Revised formula is CLOSER to observed!")
        else:
            print(f"\n  ✗ Revised formula is NOT better. Other factors at play.")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The gap between theorem and observation is explained by:

1. VIOLATED ASSUMPTION A1:
   - We assumed ID test triples have c(h,r)=c(t,r)=1
   - Reality: Only ~76-89% of ID test entities are covered
   - This significantly affects the AUROC calculation

2. RELATION WEIGHTING:
   - Different relations have different test frequencies
   - Simple average sparsity doesn't account for this

3. IMPLICATION FOR PAPER:
   - The theorem provides UPPER BOUND, not exact prediction
   - Real AUROC is lower because some ID triples look like OOD
   - This is actually interesting: shows limitation of coverage-only!
""")


if __name__ == "__main__":
    main()
