#!/usr/bin/env python3
"""
Test GNN boundary condition on YAGO3-10.

Current data points:
- WN18RR: |R|=11, avg_neighbors≈2.2, ratio≈5 → GNNSafe works (0.79)
- FB15k-237: |R|=237, avg_neighbors≈19, ratio≈12 → GNNSafe fails (0.43)

YAGO3-10: |R|=37, will compute avg_neighbors and test GNNSafe.
This should help narrow the 5-12 ratio range.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import defaultdict

from src.data.loaders import load_yago310

print("="*60)
print("GNN Boundary Condition Test: YAGO3-10")
print("="*60)

# Load data
print("\nLoading YAGO3-10...")
train_ds, valid_ds, test_ds = load_yago310()
train_triples = train_ds.triples
num_entities = train_ds.num_entities
num_relations = train_ds.num_relations

print(f"Entities: {num_entities:,}")
print(f"Relations: {num_relations}")
print(f"Train triples: {len(train_triples):,}")

# Compute average neighborhood size
print("\nComputing neighborhood statistics...")
neighbors = defaultdict(set)
for h, r, t in train_triples:
    neighbors[h].add(t)
    neighbors[t].add(h)

neighbor_counts = [len(neighbors[e]) for e in range(num_entities) if e in neighbors]
avg_neighbors = np.mean(neighbor_counts)
median_neighbors = np.median(neighbor_counts)

print(f"Entities with neighbors: {len(neighbor_counts):,}")
print(f"Average neighbors: {avg_neighbors:.1f}")
print(f"Median neighbors: {median_neighbors:.1f}")

# Compute ratio
ratio = num_relations / avg_neighbors
print(f"\n|R| / avg_neighbors = {num_relations} / {avg_neighbors:.1f} = {ratio:.1f}")

# Coverage statistics
print("\nCoverage statistics...")
coverage = defaultdict(set)
for h, r, t in train_triples:
    coverage[h].add(r)
    coverage[t].add(r)

relations_per_entity = [len(coverage[e]) for e in range(num_entities) if e in coverage]
avg_relations = np.mean(relations_per_entity)

print(f"Average relations per entity: {avg_relations:.1f}")
print(f"Relations coverage: {avg_relations/num_relations*100:.1f}%")

print("\n" + "="*60)
print("SUMMARY: GNN Boundary Condition")
print("="*60)

print(f"""
Dataset        |R|    Avg N(e)   Ratio    GNNSafe Nov. AUROC
--------------------------------------------------------------
WN18RR         11     2.2        5.0      0.79 (works)
YAGO3-10       {num_relations}     {avg_neighbors:.1f}       {ratio:.1f}      ??? (need experiment)
FB15k-237      237    19.0       12.5     0.43 (fails)
""")

print(f"YAGO ratio ({ratio:.1f}) is between WN18RR (5.0) and FB15k-237 (12.5)")
print("Prediction: GNNSafe should show intermediate performance on YAGO")

# Predict GNNSafe behavior
if ratio < 7:
    print("\n→ Ratio < 7: Neighbors likely CAN proxy for coverage")
    print("  Expected: GNNSafe AUROC > 0.6 on novel contexts")
elif ratio > 10:
    print("\n→ Ratio > 10: Neighbors likely CANNOT proxy for coverage")
    print("  Expected: GNNSafe AUROC ≈ 0.5 on novel contexts")
else:
    print("\n→ Ratio in transition zone (7-10): Unclear prediction")
    print("  Need empirical GNNSafe test to determine")
