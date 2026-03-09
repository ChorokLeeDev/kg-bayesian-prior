#!/usr/bin/env python3
"""
Deep dive: Why does WN18RR have 39% novel-context despite similar coverage density to BioKG?
"""
import numpy as np
from collections import defaultdict, Counter
import sys
sys.path.insert(0, '/Users/i767700/Github/kg-bayesian-prior')
from src.data.loaders import load_fb15k237, load_wn18rr

print("="*70)
print("WN18RR vs FB15k-237 vs BioKG: Coverage Structure Deep Dive")
print("="*70)

# Load WN18RR
print("\n[WN18RR Analysis]")
wn_train, wn_valid, wn_test = load_wn18rr()

# Build coverage
wn_coverage = defaultdict(set)
for h, r, t in wn_train.triples:
    wn_coverage[h].add(r)
    wn_coverage[t].add(r)

# Analyze test set in detail
wn_test_rel_counts = Counter(int(t[1]) for t in wn_test.triples)
wn_train_rel_counts = Counter(int(t[1]) for t in wn_train.triples)

print(f"\nRelation distribution in train vs test:")
print(f"{'Rel':<5} {'Train Edges':<12} {'Test Edges':<12} {'Train %':<10} {'Test %':<10}")
print("-"*50)
for r in range(wn_train.num_relations):
    train_count = wn_train_rel_counts.get(r, 0)
    test_count = wn_test_rel_counts.get(r, 0)
    train_pct = train_count / len(wn_train.triples) * 100
    test_pct = test_count / len(wn_test.triples) * 100 if test_count > 0 else 0
    print(f"{r:<5} {train_count:<12,} {test_count:<12,} {train_pct:<10.2f} {test_pct:<10.2f}")

# Key insight: check if test edges have different relation distribution
print(f"\nCorrelation analysis:")
train_dist = np.array([wn_train_rel_counts.get(r, 0) for r in range(wn_train.num_relations)])
test_dist = np.array([wn_test_rel_counts.get(r, 0) for r in range(wn_train.num_relations)])
train_dist_norm = train_dist / train_dist.sum()
test_dist_norm = test_dist / test_dist.sum() if test_dist.sum() > 0 else test_dist

print(f"Train relation distribution: {train_dist_norm}")
print(f"Test relation distribution:  {test_dist_norm}")
corr = np.corrcoef(train_dist_norm, test_dist_norm)[0,1]
print(f"Correlation: {corr:.4f}")

# Check: for each test edge, what's the coverage?
novel_by_rel = defaultdict(int)
total_by_rel = defaultdict(int)
for h, r, t in wn_test.triples:
    h_has_r = r in wn_coverage[h]
    t_has_r = r in wn_coverage[t]
    total_by_rel[r] += 1
    if not h_has_r or not t_has_r:
        novel_by_rel[r] += 1

print(f"\nNovel-context rate by relation:")
print(f"{'Rel':<5} {'Total':<10} {'Novel':<10} {'Rate':<10}")
print("-"*40)
for r in sorted(total_by_rel.keys()):
    total = total_by_rel[r]
    novel = novel_by_rel[r]
    rate = novel / total * 100 if total > 0 else 0
    print(f"{r:<5} {total:<10,} {novel:<10,} {rate:<10.1f}%")

# Now check FB15k-237
print("\n" + "="*70)
print("[FB15k-237 Analysis]")
fb_train, fb_valid, fb_test = load_fb15k237()

fb_coverage = defaultdict(set)
for h, r, t in fb_train.triples:
    fb_coverage[h].add(r)
    fb_coverage[t].add(r)

fb_relations_per_entity = np.array([len(rels) for rels in fb_coverage.values()])
wn_relations_per_entity = np.array([len(rels) for rels in wn_coverage.values()])

print(f"\nCoverage distribution comparison:")
print(f"{'Metric':<25} {'WN18RR':<15} {'FB15k-237':<15}")
print("-"*55)
print(f"{'Total entities':<25} {len(wn_coverage):<15,} {len(fb_coverage):<15,}")
print(f"{'Total relations':<25} {wn_train.num_relations:<15} {fb_train.num_relations:<15}")
print(f"{'Mean rel/entity':<25} {np.mean(wn_relations_per_entity):<15.2f} {np.mean(fb_relations_per_entity):<15.2f}")
print(f"{'Median rel/entity':<25} {np.median(wn_relations_per_entity):<15.1f} {np.median(fb_relations_per_entity):<15.1f}")
print(f"{'Max rel/entity':<25} {np.max(wn_relations_per_entity):<15} {np.max(fb_relations_per_entity):<15}")

# Coverage as % of possible relations
print(f"{'Coverage ratio (mean)':<25} {np.mean(wn_relations_per_entity)/wn_train.num_relations*100:<14.1f}% {np.mean(fb_relations_per_entity)/fb_train.num_relations*100:<14.1f}%")

# Key insight: what % of entities have been seen with ALL relations in test?
print("\n\nKEY INSIGHT:")
print("="*70)

# For each entity in test set, how many of its test relations are covered?
wn_test_entities = set()
wn_entity_test_rels = defaultdict(set)
for h, r, t in wn_test.triples:
    wn_test_entities.add(h)
    wn_test_entities.add(t)
    wn_entity_test_rels[h].add(r)
    wn_entity_test_rels[t].add(r)

wn_coverage_gaps = []
for e in wn_test_entities:
    test_rels = wn_entity_test_rels[e]
    train_rels = wn_coverage.get(e, set())
    gap = len(test_rels - train_rels)  # relations in test but not train
    wn_coverage_gaps.append(gap)

print(f"\nWN18RR: Coverage gaps for test entities")
print(f"  Mean gap (test rels not in train): {np.mean(wn_coverage_gaps):.2f}")
print(f"  % entities with gap > 0: {np.sum(np.array(wn_coverage_gaps) > 0) / len(wn_coverage_gaps) * 100:.1f}%")

# Same for FB15k-237
fb_test_entities = set()
fb_entity_test_rels = defaultdict(set)
for h, r, t in fb_test.triples:
    fb_test_entities.add(h)
    fb_test_entities.add(t)
    fb_entity_test_rels[h].add(r)
    fb_entity_test_rels[t].add(r)

fb_coverage_gaps = []
for e in fb_test_entities:
    test_rels = fb_entity_test_rels[e]
    train_rels = fb_coverage.get(e, set())
    gap = len(test_rels - train_rels)
    fb_coverage_gaps.append(gap)

print(f"\nFB15k-237: Coverage gaps for test entities")
print(f"  Mean gap (test rels not in train): {np.mean(fb_coverage_gaps):.2f}")
print(f"  % entities with gap > 0: {np.sum(np.array(fb_coverage_gaps) > 0) / len(fb_coverage_gaps) * 100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The key difference between WN18RR/FB15k-237 and BioKG is NOT just
mean coverage per entity - it's how the TEST SET was constructed.

WN18RR and FB15k-237 test sets were DESIGNED to include edges where:
- The entity IS in training
- But the specific (entity, relation) pair is NOT

BioKG's random split means test edges are statistically indistinguishable
from training edges - if an entity-relation pair exists at all, 93.6%
of those edges are in training.

This is the structural vs random split distinction.
""")
