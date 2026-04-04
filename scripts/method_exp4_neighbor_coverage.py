#!/usr/bin/env python3
"""
Method 4: Neighbor-Aware Coverage
Idea: Even if (h,r) unseen, if h's neighbors are seen with r, that helps
"""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def main():
    print("="*60)
    print("METHOD 4: Neighbor-Aware Coverage")
    print("="*60)
    
    ds = load_fb15k237()
    train, test = ds[0].triples, ds[2].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Build neighbor graph
    neighbors = defaultdict(set)
    for h, r, t in train:
        neighbors[int(h)].add(int(t))
        neighbors[int(t)].add(int(h))
    
    def neighbor_coverage(entity, rel):
        """What fraction of entity's neighbors have been seen with rel?"""
        neighs = neighbors.get(entity, set())
        if not neighs:
            return 0.0
        covered = sum(1 for n in neighs if (n, rel) in coverage_set)
        return covered / len(neighs)
    
    # Evaluate
    test_sub = test[:2000]
    
    results = []
    for h, r, t in test_sub:
        h, r, t = int(h), int(r), int(t)
        
        h_cov = (h, r) in coverage_set
        t_cov = (t, r) in coverage_set
        
        h_neigh_cov = neighbor_coverage(h, r)
        t_neigh_cov = neighbor_coverage(t, r)
        
        is_ood = not (h_cov and t_cov)
        
        results.append({
            'binary_cov': int(h_cov) + int(t_cov),
            'h_neigh': h_neigh_cov,
            't_neigh': t_neigh_cov,
            'ood': is_ood
        })
    
    labels = [r['ood'] for r in results]
    
    # Baseline
    binary_unc = [2 - r['binary_cov'] for r in results]
    auroc_binary = roc_auc_score(labels, binary_unc)
    print(f"Binary coverage AUROC: {auroc_binary:.4f}")
    
    # Neighbor-augmented
    # If direct coverage is 0, use neighbor coverage as soft signal
    def augmented_cov(r):
        direct = r['binary_cov']
        if direct == 2:
            return 2.0
        elif direct == 1:
            # One missing, use neighbor coverage for that one
            return 1.0 + max(r['h_neigh'], r['t_neigh']) * 0.5
        else:
            # Both missing, use both neighbor coverages
            return (r['h_neigh'] + r['t_neigh']) * 0.5
    
    aug_unc = [2 - augmented_cov(r) for r in results]
    auroc_aug = roc_auc_score(labels, aug_unc)
    print(f"Neighbor-augmented AUROC: {auroc_aug:.4f}")
    
    # Alternative: multiply neighbor coverage
    def mult_cov(r):
        base = r['binary_cov']
        neigh_bonus = (r['h_neigh'] + r['t_neigh']) / 2
        return base + neigh_bonus * 0.3
    
    mult_unc = [2 - mult_cov(r) for r in results]
    auroc_mult = roc_auc_score(labels, mult_unc)
    print(f"Multiplicative neighbor AUROC: {auroc_mult:.4f}")
    
    print(f"\nImprovement over binary: {max(auroc_aug, auroc_mult) - auroc_binary:+.4f}")

if __name__ == "__main__":
    main()
