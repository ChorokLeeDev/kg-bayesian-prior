#!/usr/bin/env python3
"""
Comprehensive leakage audit for ICEWS14 dataset.
Checks temporal integrity, inverse relations, duplicates, entity-relation overlap,
coverage statistics, and reciprocal relation detection.

UAI 2026 paper audit script.
"""

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "icews14"

# =============================================================================
# Data Loading
# =============================================================================

def load_quads(filepath):
    """Load ICEWS14 file: subject_id  relation_id  object_id  timestamp_id  -1"""
    quads = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            quads.append((s, r, o, t))
    return quads


def main():
    print("=" * 80)
    print("ICEWS14 LEAKAGE AUDIT")
    print("=" * 80)

    train_quads = load_quads(DATA_DIR / "train.txt")
    valid_quads = load_quads(DATA_DIR / "valid.txt")
    test_quads = load_quads(DATA_DIR / "test.txt")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_quads):,} quads")
    print(f"  Valid: {len(valid_quads):,} quads")
    print(f"  Test:  {len(test_quads):,} quads")
    print(f"  Total: {len(train_quads) + len(valid_quads) + len(test_quads):,} quads")

    # Extract components
    train_s = [q[0] for q in train_quads]
    train_r = [q[1] for q in train_quads]
    train_o = [q[2] for q in train_quads]
    train_t = [q[3] for q in train_quads]

    valid_s = [q[0] for q in valid_quads]
    valid_r = [q[1] for q in valid_quads]
    valid_o = [q[2] for q in valid_quads]
    valid_t = [q[3] for q in valid_quads]

    test_s = [q[0] for q in test_quads]
    test_r = [q[1] for q in test_quads]
    test_o = [q[2] for q in test_quads]
    test_t = [q[3] for q in test_quads]

    all_entities = set(train_s + train_o + valid_s + valid_o + test_s + test_o)
    all_relations = set(train_r + valid_r + test_r)
    print(f"\n  Unique entities: {len(all_entities):,}")
    print(f"  Unique relations: {len(all_relations):,}")

    # ==========================================================================
    # 1. TEMPORAL SPLIT INTEGRITY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. TEMPORAL SPLIT INTEGRITY")
    print("=" * 80)

    train_ts = np.array(train_t)
    valid_ts = np.array(valid_t)
    test_ts = np.array(test_t)

    print(f"\n  Train timestamps: min={train_ts.min()}, max={train_ts.max()}, "
          f"unique={len(np.unique(train_ts))}")
    print(f"  Valid timestamps: min={valid_ts.min()}, max={valid_ts.max()}, "
          f"unique={len(np.unique(valid_ts))}")
    print(f"  Test  timestamps: min={test_ts.min()}, max={test_ts.max()}, "
          f"unique={len(np.unique(test_ts))}")

    # Check strict temporal ordering
    train_max = train_ts.max()
    valid_min = valid_ts.min()
    valid_max = valid_ts.max()
    test_min = test_ts.min()

    if train_max < valid_min and valid_max < test_min:
        print(f"\n  [PASS] Strict temporal ordering: train({train_max}) < valid({valid_min}) < test({test_min})")
    else:
        # Check overlaps
        train_valid_overlap = np.sum(train_ts >= valid_min)
        valid_test_overlap = np.sum(valid_ts >= test_min)
        train_test_overlap = np.sum(train_ts >= test_min)
        print(f"\n  [WARN] Temporal ordering is NOT strict:")
        print(f"    Train timestamps >= valid min ({valid_min}): {train_valid_overlap:,} "
              f"({100*train_valid_overlap/len(train_ts):.1f}%)")
        print(f"    Valid timestamps >= test min ({test_min}):  {valid_test_overlap:,} "
              f"({100*valid_test_overlap/len(valid_ts):.1f}%)")
        print(f"    Train timestamps >= test min ({test_min}):  {train_test_overlap:,} "
              f"({100*train_test_overlap/len(train_ts):.1f}%)")

    # Check if timestamps overlap between splits
    train_ts_set = set(train_t)
    valid_ts_set = set(valid_t)
    test_ts_set = set(test_t)

    tv_overlap = train_ts_set & valid_ts_set
    vt_overlap = valid_ts_set & test_ts_set
    tt_overlap = train_ts_set & test_ts_set

    print(f"\n  Timestamp ID overlap:")
    print(f"    Train & Valid share {len(tv_overlap)} timestamp IDs")
    print(f"    Valid & Test  share {len(vt_overlap)} timestamp IDs")
    print(f"    Train & Test  share {len(tt_overlap)} timestamp IDs")

    # Distribution of timestamps
    print(f"\n  Timestamp distribution (percentiles):")
    for name, ts in [("Train", train_ts), ("Valid", valid_ts), ("Test", test_ts)]:
        pcts = np.percentile(ts, [0, 25, 50, 75, 100])
        print(f"    {name}: p0={pcts[0]:.0f}, p25={pcts[1]:.0f}, p50={pcts[2]:.0f}, "
              f"p75={pcts[3]:.0f}, p100={pcts[4]:.0f}")

    # ==========================================================================
    # 2. INVERSE RELATION LEAKAGE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. INVERSE RELATION LEAKAGE")
    print("=" * 80)

    # Build index: for each (tail, head) pair -> set of relations in train
    train_reverse_index = defaultdict(set)
    for s, r, o, t in train_quads:
        train_reverse_index[(o, s)].add(r)  # reverse: (tail, head)

    # For each test triple (h, r, t), check if (t, r', h) exists in train for any r'
    inverse_leak_count = 0
    inverse_leak_same_r = 0
    inverse_leak_diff_r = 0
    for s, r, o, t in test_quads:
        if (s, o) in train_reverse_index:
            # Train has (o, r', s) for some r' -- meaning (o as head, s as tail)
            # We stored train_reverse_index[(o, s)] = set of r' for train triples (s', r', o') where o'=s, s'=o
            # Wait, let me re-check the indexing...
            pass

    # Re-do more carefully
    # Train triple: (h_train, r_train, t_train)
    # Test triple: (h_test, r_test, t_test)
    # Inverse leakage: train has (t_test, r', h_test) for some r'
    # i.e., h_train = t_test, t_train = h_test
    train_ht_to_rels = defaultdict(set)
    for s, r, o, t in train_quads:
        train_ht_to_rels[(s, o)].add(r)

    inverse_leak_count = 0
    inverse_leak_same_r = 0
    inverse_leak_diff_r = 0
    inverse_leak_details = Counter()

    for s, r, o, t in test_quads:
        # Check if (o, ?, s) exists in train (i.e., reverse of test triple)
        if (o, s) in train_ht_to_rels:
            inverse_leak_count += 1
            rels_in_train = train_ht_to_rels[(o, s)]
            if r in rels_in_train:
                inverse_leak_same_r += 1
            else:
                inverse_leak_diff_r += 1
            for r_train in rels_in_train:
                inverse_leak_details[(r, r_train)] += 1

    print(f"\n  Test triples with inverse in train: {inverse_leak_count:,} / {len(test_quads):,} "
          f"({100*inverse_leak_count/len(test_quads):.1f}%)")
    print(f"    Same relation (h,r,t) <-> (t,r,h): {inverse_leak_same_r:,}")
    print(f"    Different relation (h,r,t) <-> (t,r',h): {inverse_leak_diff_r:,}")

    if inverse_leak_details:
        print(f"\n  Top 20 inverse relation pairs (test_r, train_r) -> count:")
        for (r_test, r_train), cnt in inverse_leak_details.most_common(20):
            print(f"    r_test={r_test}, r_train={r_train}: {cnt}")

    # ==========================================================================
    # 3. NEAR-DUPLICATE / EVENT OVERLAP
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. NEAR-DUPLICATE / EVENT OVERLAP (same h,r,t ignoring timestamp)")
    print("=" * 80)

    train_triples_set = set()
    for s, r, o, t in train_quads:
        train_triples_set.add((s, r, o))

    valid_triples_set = set()
    for s, r, o, t in valid_quads:
        valid_triples_set.add((s, r, o))

    test_triples_set = set()
    for s, r, o, t in test_quads:
        test_triples_set.add((s, r, o))

    test_in_train = 0
    for s, r, o, t in test_quads:
        if (s, r, o) in train_triples_set:
            test_in_train += 1

    test_triples_in_train = test_triples_set & train_triples_set
    valid_triples_in_train = valid_triples_set & train_triples_set
    test_triples_in_valid = test_triples_set & valid_triples_set

    print(f"\n  Unique triples (h,r,t) per split:")
    print(f"    Train: {len(train_triples_set):,}")
    print(f"    Valid: {len(valid_triples_set):,}")
    print(f"    Test:  {len(test_triples_set):,}")

    print(f"\n  Exact triple overlap (ignoring timestamp):")
    print(f"    Test triples also in Train:  {len(test_triples_in_train):,} unique "
          f"({100*len(test_triples_in_train)/len(test_triples_set):.1f}% of unique test)")
    print(f"    Test quads with triple in Train: {test_in_train:,} / {len(test_quads):,} "
          f"({100*test_in_train/len(test_quads):.1f}%)")
    print(f"    Valid triples also in Train: {len(valid_triples_in_train):,} unique "
          f"({100*len(valid_triples_in_train)/len(valid_triples_set):.1f}% of unique valid)")
    print(f"    Test triples also in Valid:  {len(test_triples_in_valid):,} unique "
          f"({100*len(test_triples_in_valid)/len(test_triples_set):.1f}% of unique test)")

    if test_triples_in_train:
        print(f"\n  [WARN] {len(test_triples_in_train):,} test triples are exact repeats of train triples!")
        print(f"         These are recurring events at different timestamps.")
        # Show a few examples
        examples = list(test_triples_in_train)[:5]
        print(f"         Examples: {examples}")

    # ==========================================================================
    # 4. ENTITY-RELATION OVERLAP ANALYSIS (OOD categories)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("4. ENTITY-RELATION OVERLAP ANALYSIS (OOD categories)")
    print("=" * 80)

    # Compute entity frequency from training data
    entity_freq = Counter()
    for s, r, o, t in train_quads:
        entity_freq[s] += 1
        entity_freq[o] += 1

    freqs = np.array(list(entity_freq.values()))
    threshold = np.percentile(freqs, 25)
    print(f"\n  Entity frequency stats (from train):")
    print(f"    Num entities in train: {len(entity_freq):,}")
    print(f"    Min freq: {freqs.min()}, Max freq: {freqs.max()}, Mean: {freqs.mean():.1f}")
    print(f"    25th percentile (threshold): {threshold}")
    print(f"    Entities at or below threshold: {np.sum(freqs <= threshold):,} "
          f"({100*np.sum(freqs <= threshold)/len(freqs):.1f}%)")

    # Build entity-relation pair set from training
    train_er_pairs = set()
    for s, r, o, t in train_quads:
        train_er_pairs.add((s, r))   # entity as head with relation r
        train_er_pairs.add((o, r))   # entity as tail with relation r

    # Categorize test triples
    emerging = []
    novel_ctx = []
    in_distribution = []

    for s, r, o, t in test_quads:
        s_freq = entity_freq.get(s, 0)
        o_freq = entity_freq.get(o, 0)

        # Emerging: at least one entity is low-frequency or unseen
        if s_freq <= threshold or o_freq <= threshold:
            emerging.append((s, r, o, t))
        else:
            # Check if entity-relation pairs are seen in training
            s_r_seen = (s, r) in train_er_pairs
            o_r_seen = (o, r) in train_er_pairs
            if not s_r_seen or not o_r_seen:
                novel_ctx.append((s, r, o, t))
            else:
                in_distribution.append((s, r, o, t))

    print(f"\n  OOD categorization of test triples:")
    print(f"    Emerging (entity freq <= {threshold}): {len(emerging):,} "
          f"({100*len(emerging)/len(test_quads):.1f}%)")
    print(f"    Novel context (known entities, unseen e-r pair): {len(novel_ctx):,} "
          f"({100*len(novel_ctx)/len(test_quads):.1f}%)")
    print(f"    In-distribution: {len(in_distribution):,} "
          f"({100*len(in_distribution)/len(test_quads):.1f}%)")
    print(f"    Total: {len(emerging) + len(novel_ctx) + len(in_distribution):,}")

    # For NOVEL CONTEXT triples: verify what fraction have truly unseen e-r pairs
    print(f"\n  Novel context deep-dive:")
    if novel_ctx:
        nc_head_unseen = sum(1 for s, r, o, t in novel_ctx if (s, r) not in train_er_pairs)
        nc_tail_unseen = sum(1 for s, r, o, t in novel_ctx if (o, r) not in train_er_pairs)
        nc_both_unseen = sum(1 for s, r, o, t in novel_ctx
                            if (s, r) not in train_er_pairs and (o, r) not in train_er_pairs)

        print(f"    Head e-r pair unseen in train: {nc_head_unseen:,} / {len(novel_ctx):,} "
              f"({100*nc_head_unseen/len(novel_ctx):.1f}%)")
        print(f"    Tail e-r pair unseen in train: {nc_tail_unseen:,} / {len(novel_ctx):,} "
              f"({100*nc_tail_unseen/len(novel_ctx):.1f}%)")
        print(f"    Both e-r pairs unseen:        {nc_both_unseen:,} / {len(novel_ctx):,} "
              f"({100*nc_both_unseen/len(novel_ctx):.1f}%)")

        # How many novel_ctx triples are also exact duplicates of training triples?
        nc_also_dup = sum(1 for s, r, o, t in novel_ctx if (s, r, o) in train_triples_set)
        print(f"    Novel_ctx also exact duplicate of train triple: {nc_also_dup:,} / {len(novel_ctx):,} "
              f"({100*nc_also_dup/len(novel_ctx):.1f}%)")

        # Entity frequency distribution for novel_ctx entities
        nc_entity_freqs = []
        for s, r, o, t in novel_ctx:
            nc_entity_freqs.append(entity_freq.get(s, 0))
            nc_entity_freqs.append(entity_freq.get(o, 0))
        nc_entity_freqs = np.array(nc_entity_freqs)
        print(f"    Entity freq in novel_ctx: min={nc_entity_freqs.min()}, "
              f"max={nc_entity_freqs.max()}, mean={nc_entity_freqs.mean():.1f}")
    else:
        print(f"    No novel context triples found.")

    # For EMERGING triples: check how many entities are completely unseen
    print(f"\n  Emerging deep-dive:")
    if emerging:
        em_unseen_entities = sum(1 for s, r, o, t in emerging
                                if entity_freq.get(s, 0) == 0 or entity_freq.get(o, 0) == 0)
        em_head_unseen = sum(1 for s, r, o, t in emerging if entity_freq.get(s, 0) == 0)
        em_tail_unseen = sum(1 for s, r, o, t in emerging if entity_freq.get(o, 0) == 0)
        print(f"    At least one entity completely unseen in train: {em_unseen_entities:,} "
              f"({100*em_unseen_entities/len(emerging):.1f}%)")
        print(f"    Head entity unseen: {em_head_unseen:,}")
        print(f"    Tail entity unseen: {em_tail_unseen:,}")

        # Freq distribution of the "rare" entities
        em_rare_freqs = []
        for s, r, o, t in emerging:
            sf = entity_freq.get(s, 0)
            of_ = entity_freq.get(o, 0)
            if sf <= threshold:
                em_rare_freqs.append(sf)
            if of_ <= threshold:
                em_rare_freqs.append(of_)
        em_rare_freqs = np.array(em_rare_freqs)
        print(f"    Rare entity freq distribution: min={em_rare_freqs.min()}, "
              f"max={em_rare_freqs.max()}, mean={em_rare_freqs.mean():.1f}")

    # ==========================================================================
    # 5. COVERAGE STATISTICS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("5. COVERAGE STATISTICS")
    print("=" * 80)

    # Test entity-relation pairs vs train
    test_er_pairs = set()
    for s, r, o, t in test_quads:
        test_er_pairs.add((s, r))
        test_er_pairs.add((o, r))

    test_er_unseen = test_er_pairs - train_er_pairs
    test_er_seen = test_er_pairs & train_er_pairs

    print(f"\n  Entity-relation pair coverage:")
    print(f"    Unique e-r pairs in train: {len(train_er_pairs):,}")
    print(f"    Unique e-r pairs in test:  {len(test_er_pairs):,}")
    print(f"    Test e-r pairs seen in train:   {len(test_er_seen):,} "
          f"({100*len(test_er_seen)/len(test_er_pairs):.1f}%)")
    print(f"    Test e-r pairs unseen in train: {len(test_er_unseen):,} "
          f"({100*len(test_er_unseen)/len(test_er_pairs):.1f}%)")

    # Per-triple coverage: for each test triple, are both (h,r) and (t,r) seen?
    both_seen = sum(1 for s, r, o, t in test_quads
                    if (s, r) in train_er_pairs and (o, r) in train_er_pairs)
    at_least_one_unseen = len(test_quads) - both_seen

    print(f"\n  Per-triple coverage:")
    print(f"    Both (h,r) and (t,r) seen in train: {both_seen:,} / {len(test_quads):,} "
          f"({100*both_seen/len(test_quads):.1f}%)")
    print(f"    At least one e-r pair unseen:       {at_least_one_unseen:,} / {len(test_quads):,} "
          f"({100*at_least_one_unseen/len(test_quads):.1f}%)")

    # Random baseline: what fraction of all possible e-r pairs are covered by train?
    num_entities_train = len(set(train_s + train_o))
    num_relations_train = len(set(train_r))
    total_possible_er = num_entities_train * num_relations_train
    train_coverage_density = len(train_er_pairs) / total_possible_er

    print(f"\n  Random baseline comparison:")
    print(f"    Total possible e-r pairs (entities x relations): {total_possible_er:,}")
    print(f"    Train e-r coverage density: {train_coverage_density:.4f} "
          f"({100*train_coverage_density:.2f}%)")
    print(f"    Expected fraction of random test e-r pair being unseen: "
          f"{100*(1-train_coverage_density):.2f}%")
    print(f"    Actual fraction of test e-r pairs unseen: "
          f"{100*len(test_er_unseen)/len(test_er_pairs):.2f}%")

    # Entity coverage
    train_entities = set(train_s + train_o)
    test_entities = set(test_s + test_o)
    test_only_entities = test_entities - train_entities
    print(f"\n  Entity coverage:")
    print(f"    Entities in train: {len(train_entities):,}")
    print(f"    Entities in test:  {len(test_entities):,}")
    print(f"    Test-only entities (never in train): {len(test_only_entities):,} "
          f"({100*len(test_only_entities)/len(test_entities):.1f}%)")

    # Relation coverage
    train_relations = set(train_r)
    test_relations = set(test_r)
    test_only_relations = test_relations - train_relations
    print(f"\n  Relation coverage:")
    print(f"    Relations in train: {len(train_relations):,}")
    print(f"    Relations in test:  {len(test_relations):,}")
    print(f"    Test-only relations (never in train): {len(test_only_relations):,}")

    # ==========================================================================
    # 6. RECIPROCAL RELATION DETECTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("6. RECIPROCAL RELATION DETECTION")
    print("=" * 80)

    # For each pair of relations (r1, r2), count how often (h, r1, t) and (t, r2, h) co-occur
    # We use train data for detection
    # Build forward index: (h, t) -> set of relations
    forward_rels = defaultdict(set)
    for s, r, o, t in train_quads:
        forward_rels[(s, o)].add(r)

    # For each triple (h, r, t), check if (t, ?, h) exists
    reciprocal_counts = Counter()  # (r1, r2) -> count
    reciprocal_total = Counter()   # r1 -> total count

    for s, r, o, t in train_quads:
        reciprocal_total[r] += 1
        if (o, s) in forward_rels:
            for r2 in forward_rels[(o, s)]:
                reciprocal_counts[(r, r2)] += 1

    # Find strong reciprocal pairs
    print(f"\n  Checking all relation pairs for reciprocal patterns...")
    print(f"  (A reciprocal pair means (h,r1,t) frequently co-occurs with (t,r2,h))\n")

    reciprocal_pairs = []
    for (r1, r2), count in reciprocal_counts.most_common():
        total_r1 = reciprocal_total[r1]
        ratio = count / total_r1 if total_r1 > 0 else 0
        if ratio > 0.3 and count >= 50:  # At least 30% reciprocal and 50 instances
            reciprocal_pairs.append((r1, r2, count, total_r1, ratio))

    if reciprocal_pairs:
        print(f"  Found {len(reciprocal_pairs)} strong reciprocal relation patterns (>30%, >=50 instances):")
        print(f"  {'r1':>4} {'r2':>4} {'count':>8} {'total_r1':>10} {'ratio':>8}")
        print(f"  {'-'*4} {'-'*4} {'-'*8} {'-'*10} {'-'*8}")
        for r1, r2, count, total_r1, ratio in sorted(reciprocal_pairs, key=lambda x: -x[4]):
            marker = " <-- SELF-RECIPROCAL" if r1 == r2 else ""
            print(f"  {r1:>4} {r2:>4} {count:>8,} {total_r1:>10,} {ratio:>8.1%}{marker}")
    else:
        print(f"  No strong reciprocal patterns found.")

    # Check if dataset has dedicated reciprocal relation IDs (r and r + num_relations/2)
    num_rel = len(all_relations)
    print(f"\n  Checking for systematic reciprocal ID scheme (r and r+{num_rel//2}):")
    systematic_count = 0
    for r in sorted(train_relations):
        r_recip = r + num_rel // 2
        if r_recip in train_relations:
            # Check if they actually co-occur as reciprocals
            if (r, r_recip) in reciprocal_counts and reciprocal_counts[(r, r_recip)] > 10:
                systematic_count += 1
                if systematic_count <= 10:
                    print(f"    r={r}, r'={r_recip}: "
                          f"{reciprocal_counts[(r, r_recip)]} reciprocal co-occurrences")

    if systematic_count == 0:
        print(f"    No systematic reciprocal ID scheme detected.")
    else:
        print(f"    ... {systematic_count} systematic reciprocal pairs total")

    # ==========================================================================
    # 7. SUMMARY / KEY CONCERNS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("7. SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    issues = []

    # Temporal
    if train_max >= test_min:
        issues.append(f"TEMPORAL LEAKAGE: Train timestamps overlap with test timestamps")
    if train_max >= valid_min:
        issues.append(f"TEMPORAL OVERLAP: Train timestamps overlap with validation timestamps")

    # Exact duplicates
    dup_pct = 100 * len(test_triples_in_train) / len(test_triples_set)
    if dup_pct > 0:
        issues.append(f"EXACT DUPLICATES: {len(test_triples_in_train):,} test triples ({dup_pct:.1f}%) "
                       f"are exact repeats of train triples (recurring events)")

    # Inverse leakage
    inv_pct = 100 * inverse_leak_count / len(test_quads)
    if inv_pct > 5:
        issues.append(f"INVERSE LEAKAGE: {inverse_leak_count:,} test triples ({inv_pct:.1f}%) "
                       f"have an inverse counterpart in train")

    # Coverage
    unseen_pct = 100 * len(test_er_unseen) / len(test_er_pairs)
    issues.append(f"COVERAGE: {unseen_pct:.1f}% of test entity-relation pairs are unseen in training")

    # Novel context
    if novel_ctx:
        issues.append(f"NOVEL CONTEXT: {len(novel_ctx):,} test triples classified as novel context "
                       f"({100*len(novel_ctx)/len(test_quads):.1f}%)")

    print()
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No significant issues found.")

    # Final metrics for paper
    print(f"\n  --- Metrics for paper reference ---")
    print(f"  Total test triples: {len(test_quads):,}")
    print(f"  Emerging: {len(emerging):,} ({100*len(emerging)/len(test_quads):.1f}%)")
    print(f"  Novel context: {len(novel_ctx):,} ({100*len(novel_ctx)/len(test_quads):.1f}%)")
    print(f"  In-distribution: {len(in_distribution):,} ({100*len(in_distribution)/len(test_quads):.1f}%)")
    print(f"  Exact train-test duplicates (recurring events): {len(test_triples_in_train):,} "
          f"({dup_pct:.1f}%)")
    print(f"  Inverse counterparts in train: {inverse_leak_count:,} ({inv_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
