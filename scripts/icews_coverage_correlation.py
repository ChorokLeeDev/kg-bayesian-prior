#!/usr/bin/env python3
"""
Compute correlation between temporal OOD labels and coverage on ICEWS14.

Skeptical reviewer question: Is the temporal OOD detection task circular
with coverage-based detection?

If Spearman correlation > 0.9: "non-circular" claim is weakened
If Spearman correlation < 0.7: experiments genuinely test something different
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct loading (no external dependencies)


def compute_coverage_matrix(train_triples, num_entities, num_relations):
    """Build binary coverage matrix: coverage[e, r] = 1 if (e, r, *) or (*, r, e) seen."""
    coverage = np.zeros((num_entities, num_relations), dtype=np.int8)

    for h, r, t in train_triples:
        coverage[h, r] = 1
        coverage[t, r] = 1

    return coverage


def load_icews14_direct(data_dir):
    """Load ICEWS14 without stat.txt file."""

    def load_temporal_triples(path):
        """Load triples with timestamp IDs."""
        triples = []
        timestamps = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    s, r, o, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    triples.append([s, r, o])
                    timestamps.append(ts)
        return np.array(triples), np.array(timestamps)

    train_triples, train_ts = load_temporal_triples(data_dir / "train.txt")
    valid_triples, valid_ts = load_temporal_triples(data_dir / "valid.txt")
    test_triples, test_ts = load_temporal_triples(data_dir / "test.txt")

    # Compute entity/relation counts from data
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    num_entities = max(all_triples[:, 0].max(), all_triples[:, 2].max()) + 1
    num_relations = all_triples[:, 1].max() + 1

    class SimpleDataset:
        def __init__(self, triples, num_ent, num_rel, timestamps):
            self.triples = triples
            self.num_entities = num_ent
            self.num_relations = num_rel
            self.timestamps = timestamps

    return (
        SimpleDataset(train_triples, num_entities, num_relations, train_ts),
        SimpleDataset(valid_triples, num_entities, num_relations, valid_ts),
        SimpleDataset(test_triples, num_entities, num_relations, test_ts),
    )


def main():
    print("=" * 60)
    print("ICEWS14: Temporal OOD vs Coverage Correlation Analysis")
    print("=" * 60)

    # Load ICEWS14 directly (without stat.txt)
    data_dir = Path(__file__).parent.parent / "data" / "ICEWS14"
    train_ds, valid_ds, test_ds = load_icews14_direct(data_dir)

    print(f"\nDataset statistics:")
    print(f"  Train triples: {len(train_ds.triples)}")
    print(f"  Valid triples: {len(valid_ds.triples)}")
    print(f"  Test triples:  {len(test_ds.triples)}")
    print(f"  Entities:      {train_ds.num_entities}")
    print(f"  Relations:     {train_ds.num_relations}")

    # Build coverage matrix from training data
    coverage = compute_coverage_matrix(
        train_ds.triples,
        train_ds.num_entities,
        train_ds.num_relations
    )

    print(f"\nCoverage matrix: {coverage.shape}")
    print(f"  Non-zero entries: {coverage.sum()} / {coverage.size} ({100*coverage.sum()/coverage.size:.2f}%)")

    # For each test triple, compute:
    # 1. is_temporal_ood: True for all test triples (by definition, they occur later)
    # 2. has_zero_coverage: True if head OR tail has zero coverage for this relation

    test_triples = test_ds.triples

    # Compute coverage-based labels
    has_zero_coverage_head = []
    has_zero_coverage_tail = []
    has_zero_coverage_either = []

    for h, r, t in test_triples:
        head_covered = coverage[h, r] == 1
        tail_covered = coverage[t, r] == 1

        has_zero_coverage_head.append(not head_covered)
        has_zero_coverage_tail.append(not tail_covered)
        has_zero_coverage_either.append(not head_covered or not tail_covered)

    has_zero_coverage_head = np.array(has_zero_coverage_head)
    has_zero_coverage_tail = np.array(has_zero_coverage_tail)
    has_zero_coverage_either = np.array(has_zero_coverage_either)

    # In temporal OOD setting, all test triples are OOD (label = 1)
    # All train triples are in-distribution (label = 0)
    # We need to compute correlation on a mixed set

    print("\n" + "=" * 60)
    print("Analysis 1: Test triples only (all temporal OOD by definition)")
    print("=" * 60)

    n_test = len(test_triples)
    n_zero_head = has_zero_coverage_head.sum()
    n_zero_tail = has_zero_coverage_tail.sum()
    n_zero_either = has_zero_coverage_either.sum()

    print(f"\nTest triples breakdown:")
    print(f"  Zero coverage (head):   {n_zero_head:5d} / {n_test} ({100*n_zero_head/n_test:.1f}%)")
    print(f"  Zero coverage (tail):   {n_zero_tail:5d} / {n_test} ({100*n_zero_tail/n_test:.1f}%)")
    print(f"  Zero coverage (either): {n_zero_either:5d} / {n_test} ({100*n_zero_either/n_test:.1f}%)")
    print(f"  Full coverage (both):   {n_test - n_zero_either:5d} / {n_test} ({100*(n_test - n_zero_either)/n_test:.1f}%)")

    # Now compute correlation on combined train + test
    print("\n" + "=" * 60)
    print("Analysis 2: Train vs Test correlation")
    print("=" * 60)

    # Train triples: is_ood = 0, coverage = 1 (by definition)
    # Test triples: is_ood = 1, coverage varies

    train_triples = train_ds.triples

    # For train, compute zero coverage (should be very low since coverage is built from train)
    train_zero_coverage = []
    for h, r, t in train_triples:
        head_covered = coverage[h, r] == 1
        tail_covered = coverage[t, r] == 1
        train_zero_coverage.append(not head_covered or not tail_covered)
    train_zero_coverage = np.array(train_zero_coverage)

    print(f"\nTrain zero-coverage (sanity check): {train_zero_coverage.sum()} / {len(train_triples)} ({100*train_zero_coverage.mean():.3f}%)")

    # Combined arrays
    is_temporal_ood = np.concatenate([
        np.zeros(len(train_triples)),  # Train = in-distribution
        np.ones(len(test_triples))     # Test = OOD
    ])

    has_zero_coverage_combined = np.concatenate([
        train_zero_coverage,
        has_zero_coverage_either
    ])

    # Spearman correlation
    rho, pvalue = stats.spearmanr(is_temporal_ood, has_zero_coverage_combined)

    print(f"\nSpearman correlation (temporal_ood vs zero_coverage):")
    print(f"  rho = {rho:.4f}")
    print(f"  p-value = {pvalue:.2e}")

    # Point-biserial correlation (appropriate for binary variables)
    pb_corr, pb_pvalue = stats.pointbiserialr(is_temporal_ood.astype(int), has_zero_coverage_combined.astype(int))

    print(f"\nPoint-biserial correlation:")
    print(f"  r = {pb_corr:.4f}")
    print(f"  p-value = {pb_pvalue:.2e}")

    # Phi coefficient (for 2x2 contingency table)
    # Rows: temporal_ood (0/1), Cols: zero_coverage (0/1)
    a = ((is_temporal_ood == 0) & (has_zero_coverage_combined == 0)).sum()  # TN
    b = ((is_temporal_ood == 0) & (has_zero_coverage_combined == 1)).sum()  # FP
    c = ((is_temporal_ood == 1) & (has_zero_coverage_combined == 0)).sum()  # FN
    d = ((is_temporal_ood == 1) & (has_zero_coverage_combined == 1)).sum()  # TP

    print(f"\nContingency table:")
    print(f"                     Coverage=1  Coverage=0")
    print(f"  Temporal_OOD=0     {a:8d}    {b:8d}")
    print(f"  Temporal_OOD=1     {c:8d}    {d:8d}")

    phi = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    print(f"\nPhi coefficient: {phi:.4f}")

    # Jaccard similarity between temporal OOD and zero coverage
    jaccard = d / (b + c + d) if (b + c + d) > 0 else 0
    print(f"Jaccard similarity (OOD=1 ∩ zero_coverage=1): {jaccard:.4f}")

    # Key insight: what fraction of temporal OOD is NOT captured by coverage?
    ood_but_covered = c  # Temporal OOD but has coverage
    ood_and_zero_coverage = d  # Temporal OOD and no coverage

    print("\n" + "=" * 60)
    print("KEY FINDING: Temporal OOD decomposition")
    print("=" * 60)
    print(f"\nTemporal OOD test triples: {c + d}")
    print(f"  - Has coverage (not detected by coverage): {c} ({100*c/(c+d):.1f}%)")
    print(f"  - Zero coverage (detected by coverage):    {d} ({100*d/(c+d):.1f}%)")

    if c / (c + d) > 0.3:
        print(f"\n>>> {100*c/(c+d):.1f}% of temporal OOD has full coverage")
        print(">>> This validates that temporal OOD tests BEYOND the coverage blind spot")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if rho > 0.9:
        interpretation = "HIGH correlation (>0.9): The 'non-circular' claim is WEAKENED.\n" \
                         "Temporal OOD and zero-coverage are nearly equivalent."
    elif rho > 0.7:
        interpretation = "MODERATE correlation (0.7-0.9): Partial overlap between tasks.\n" \
                         "Some temporal OOD is captured by coverage, but not all."
    else:
        interpretation = "LOW correlation (<0.7): Temporal OOD tests genuinely different scenarios.\n" \
                         "Coverage and temporal OOD measure distinct aspects of distributional shift."

    print(f"\n{interpretation}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "icews_coverage_correlation.txt"

    with open(output_file, "w") as f:
        f.write("ICEWS14: Temporal OOD vs Coverage Correlation Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset:\n")
        f.write(f"  Train: {len(train_triples)} triples\n")
        f.write(f"  Test:  {len(test_triples)} triples\n")
        f.write(f"  Entities: {train_ds.num_entities}\n")
        f.write(f"  Relations: {train_ds.num_relations}\n\n")

        f.write("Test triple zero-coverage rates:\n")
        f.write(f"  Head:   {100*n_zero_head/n_test:.1f}%\n")
        f.write(f"  Tail:   {100*n_zero_tail/n_test:.1f}%\n")
        f.write(f"  Either: {100*n_zero_either/n_test:.1f}%\n\n")

        f.write("Correlation metrics (is_temporal_ood vs has_zero_coverage):\n")
        f.write(f"  Spearman rho:       {rho:.4f} (p={pvalue:.2e})\n")
        f.write(f"  Point-biserial r:   {pb_corr:.4f} (p={pb_pvalue:.2e})\n")
        f.write(f"  Phi coefficient:    {phi:.4f}\n")
        f.write(f"  Jaccard similarity: {jaccard:.4f}\n\n")

        f.write("Contingency table:\n")
        f.write("                     Coverage=1  Coverage=0\n")
        f.write(f"  Temporal_OOD=0     {a:8d}    {b:8d}\n")
        f.write(f"  Temporal_OOD=1     {c:8d}    {d:8d}\n\n")

        f.write("Key finding - Temporal OOD decomposition:\n")
        f.write(f"  - Has coverage (not detected by coverage): {c} ({100*c/(c+d):.1f}%)\n")
        f.write(f"  - Zero coverage (detected by coverage):    {d} ({100*d/(c+d):.1f}%)\n\n")

        f.write("INTERPRETATION:\n")
        f.write(f"{interpretation}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
