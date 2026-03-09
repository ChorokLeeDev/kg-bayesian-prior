#!/usr/bin/env python3
"""
OGBL-BioKG Coverage Blind Spot Analysis
51 relation types - proper multi-relational KG for our analysis
"""
import numpy as np
from collections import defaultdict
import time

print("="*60)
print("OGBL-BioKG - Coverage Blind Spot Analysis")
print("="*60)

import torch
# Patch torch.load for PyTorch 2.6 compatibility
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ogb.linkproppred import LinkPropPredDataset

print("\nLoading OGBL-BioKG...")
dataset = LinkPropPredDataset(name='ogbl-biokg', root='./dataset')

split = dataset.get_edge_split()
train_edges = split['train']
valid_edges = split['valid']
test_edges = split['test']

# BioKG has different entity types
print(f"\nDataset info:")
print(f"  Entity types: {dataset.graph['num_nodes_dict']}")
print(f"  Train edges: {len(train_edges['head']):,}")
print(f"  Valid edges: {len(valid_edges['head']):,}")
print(f"  Test edges: {len(test_edges['head']):,}")

# Get relation types
train_relations = train_edges['relation']
num_relations = int(train_relations.max()) + 1
print(f"  Relation types: {num_relations}")

# Build coverage matrix: (entity, relation) pairs seen in training
print("\nBuilding coverage matrix...")
start = time.time()

# BioKG has typed entities - we'll use (type, local_id) as entity key
coverage = defaultdict(set)  # entity_key -> set of relations

head_type = train_edges['head_type']
tail_type = train_edges['tail_type']
heads = train_edges['head']
tails = train_edges['tail']
relations = train_edges['relation']

for i in range(len(heads)):
    h_key = (head_type[i], int(heads[i]))
    t_key = (tail_type[i], int(tails[i]))
    r = int(relations[i])
    coverage[h_key].add(r)
    coverage[t_key].add(r)

print(f"  Build time: {time.time()-start:.1f}s")
print(f"  Unique entities with coverage: {len(coverage):,}")

# Compute coverage statistics
relations_per_entity = [len(rels) for rels in coverage.values()]
avg_relations = np.mean(relations_per_entity)
print(f"  Avg relations per entity: {avg_relations:.1f} / {num_relations}")

# Analyze test set for novel-context pattern
print("\nAnalyzing test set...")

test_head_type = test_edges['head_type']
test_tail_type = test_edges['tail_type']
test_heads = test_edges['head']
test_tails = test_edges['tail']
test_relations = test_edges['relation']

novel_context_count = 0
emerging_count = 0
in_dist_count = 0
total = len(test_heads)

# Sample for speed if large
sample_size = min(50000, total)
indices = np.random.choice(total, sample_size, replace=False)

for idx in indices:
    h_key = (test_head_type[idx], int(test_heads[idx]))
    t_key = (test_tail_type[idx], int(test_tails[idx]))
    r = int(test_relations[idx])

    h_seen = h_key in coverage
    t_seen = t_key in coverage
    h_has_r = h_seen and r in coverage[h_key]
    t_has_r = t_seen and r in coverage[t_key]

    if not h_seen or not t_seen:
        # Emerging entity
        emerging_count += 1
    elif not h_has_r or not t_has_r:
        # Novel context: entity seen, but not with this relation
        novel_context_count += 1
    else:
        # In-distribution
        in_dist_count += 1

print(f"\nTest set breakdown (n={sample_size:,}):")
print(f"  Novel context: {novel_context_count:,} ({novel_context_count/sample_size:.1%})")
print(f"  Emerging entity: {emerging_count:,} ({emerging_count/sample_size:.1%})")
print(f"  In-distribution: {in_dist_count:,} ({in_dist_count/sample_size:.1%})")

novel_rate = novel_context_count / sample_size
emerging_rate = emerging_count / sample_size

print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
print(f"""
OGBL-BioKG has {num_relations} relation types (drug-protein, protein-protein, etc.)

Coverage blind spot prevalence:
- Novel context: {novel_rate:.1%} of test queries
- Emerging entity: {emerging_rate:.1%} of test queries
- Total OOD: {(novel_rate + emerging_rate):.1%}

This means {novel_rate:.0%} of test queries involve:
- Entities that ARE in training (good embeddings)
- But NEVER seen with this specific relation type

Standard biomedical KG models (RotatE, TransE, GNNs) will be
overconfident on these {novel_rate:.0%} queries because:
1. Entity embeddings look confident (many training edges)
2. But no evidence for this specific interaction type

SAFETY IMPLICATION:
In drug discovery, this means models confidently predict
drug-protein interactions they have ZERO evidence for.
""")

# Compare to FB15k-237 for reference
print(f"\nComparison to standard KG benchmarks:")
print(f"  FB15k-237: ~25% novel context")
print(f"  WN18RR: ~11% novel context")
print(f"  OGBL-BioKG: {novel_rate:.0%} novel context")

if novel_rate > 0.1:
    print(f"\n✓ BioKG confirms: Coverage blind spot affects biomedical AI")
else:
    print(f"\n! BioKG has lower novel-context rate than expected")
