#!/usr/bin/env python3
"""Bloom filter test on OGB-WikiKG2 (real data)."""
import os
os.environ['OGB_DOWNLOAD_DIR'] = './dataset'

import numpy as np
import time
import math
import hashlib
import torch

# Fix PyTorch 2.6 compatibility
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

class BloomFilter:
    def __init__(self, expected_elements, fpr=0.01):
        self.size = int(-expected_elements * math.log(fpr) / (math.log(2) ** 2))
        self.num_hashes = max(1, int((self.size / expected_elements) * math.log(2)))
        self.bit_array = np.zeros(self.size, dtype=np.uint8)
        
    def _hashes(self, item):
        hashes = []
        for i in range(self.num_hashes):
            h = hashlib.md5(f"{item}_{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes
    
    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1
    
    def __contains__(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))
    
    def memory_mb(self):
        return self.bit_array.nbytes / (1024 * 1024)

print("="*60)
print("WikiKG2 Bloom Filter Scalability Test")
print("="*60)

# Auto-accept download
import sys
from unittest.mock import patch

def mock_input(*args, **kwargs):
    return 'y'

with patch('builtins.input', mock_input):
    from ogb.linkproppred import LinkPropPredDataset
    print("\nLoading OGB-WikiKG2 (2.5M entities)...")
    dataset = LinkPropPredDataset(name='ogbl-wikikg2')

split = dataset.get_edge_split()
train = split['train']
valid = split['valid']

train_triples = np.column_stack([train['head'], train['relation'], train['tail']])
test_triples = np.column_stack([valid['head'], valid['relation'], valid['tail']])

num_entities = dataset.graph['num_nodes']
num_relations = int(train_triples[:, 1].max()) + 1

print(f"\nDataset stats:")
print(f"  Entities: {num_entities:,}")
print(f"  Relations: {num_relations}")
print(f"  Train triples: {len(train_triples):,}")
print(f"  Test triples: {len(test_triples):,}")

# Build exact coverage
print("\nBuilding exact coverage...")
start = time.time()
exact_coverage = set()
for h, r, t in train_triples:
    exact_coverage.add((int(h), int(r)))
    exact_coverage.add((int(t), int(r)))
print(f"  Covered pairs: {len(exact_coverage):,}")
print(f"  Time: {time.time()-start:.1f}s")
exact_mb = len(exact_coverage) * 50 / 1e6

# Test Bloom filters
print("\nTesting Bloom filters...")
results = []
for fpr in [0.001, 0.01, 0.05]:
    print(f"\n  FPR={fpr*100:.1f}%...")
    bloom = BloomFilter(len(exact_coverage), fpr)
    for h, r, t in train_triples:
        bloom.add((int(h), int(r)))
        bloom.add((int(t), int(r)))
    
    # Evaluate on test
    tp, fn = 0, 0
    for h, r, t in test_triples[:10000]:  # Sample for speed
        is_novel = (int(h), int(r)) not in exact_coverage or (int(t), int(r)) not in exact_coverage
        pred_novel = (int(h), int(r)) not in bloom or (int(t), int(r)) not in bloom
        if is_novel and pred_novel: tp += 1
        if is_novel and not pred_novel: fn += 1
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    results.append({'fpr': fpr, 'mb': bloom.memory_mb(), 'recall': recall, 'fn': fn})
    print(f"    Storage: {bloom.memory_mb():.1f} MB, Recall: {recall:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Method':<20} {'Storage':<12} {'Reduction':<10} {'Recall Drop'}")
print("-"*55)
print(f"{'Exact hash':<20} {exact_mb:>8.1f} MB {'1x':>8} {'---':>10}")
for r in results:
    red = exact_mb / r['mb']
    drop = 1 - r['recall']
    print(f"Bloom ({r['fpr']*100:.1f}% FPR){'':<6} {r['mb']:>8.1f} MB {red:>7.0f}x {drop:>10.4f}")

print("\n✓ WikiKG2 validation complete")
