#!/usr/bin/env python3
"""
Method 5: Entity-level Uncertainty
Idea: Some entities are inherently harder (low degree, rare)
"""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237
from sklearn.metrics import roc_auc_score

def main():
    print("="*60)
    print("METHOD 5: Entity-Level Uncertainty")
    print("="*60)
    
    ds = load_fb15k237()
    train, test = ds[0].triples, ds[2].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Entity statistics
    entity_degree = np.zeros(n_ent)
    entity_rel_count = [set() for _ in range(n_ent)]
    
    for h, r, t in train:
        entity_degree[int(h)] += 1
        entity_degree[int(t)] += 1
        entity_rel_count[int(h)].add(int(r))
        entity_rel_count[int(t)].add(int(r))
    
    entity_rel_diversity = np.array([len(s) for s in entity_rel_count])
    
    # Evaluate
    test_sub = test[:2000]
    
    results = []
    for h, r, t in test_sub:
        h, r, t = int(h), int(r), int(t)
        
        h_cov = (h, r) in coverage_set
        t_cov = (t, r) in coverage_set
        
        is_ood = not (h_cov and t_cov)
        
        results.append({
            'binary_cov': int(h_cov) + int(t_cov),
            'h_degree': entity_degree[h],
            't_degree': entity_degree[t],
            'h_div': entity_rel_diversity[h],
            't_div': entity_rel_diversity[t],
            'ood': is_ood
        })
    
    labels = [r['ood'] for r in results]
    
    # Baseline
    binary_unc = [2 - r['binary_cov'] for r in results]
    auroc_binary = roc_auc_score(labels, binary_unc)
    print(f"Binary coverage AUROC: {auroc_binary:.4f}")
    
    # Degree-weighted coverage
    def degree_weighted(r):
        base = r['binary_cov']
        # Penalize low-degree entities even if covered
        min_deg = min(r['h_degree'], r['t_degree'])
        deg_penalty = 1.0 / (1 + np.log1p(min_deg))
        return base - deg_penalty * 0.3
    
    deg_unc = [2 - degree_weighted(r) for r in results]
    auroc_deg = roc_auc_score(labels, deg_unc)
    print(f"Degree-weighted AUROC: {auroc_deg:.4f}")
    
    # Diversity-weighted
    def div_weighted(r):
        base = r['binary_cov']
        min_div = min(r['h_div'], r['t_div'])
        div_penalty = 1.0 / (1 + min_div)
        return base - div_penalty * 0.3
    
    div_unc = [2 - div_weighted(r) for r in results]
    auroc_div = roc_auc_score(labels, div_unc)
    print(f"Diversity-weighted AUROC: {auroc_div:.4f}")
    
    # Combined
    def combined(r):
        base = r['binary_cov']
        min_deg = min(r['h_degree'], r['t_degree'])
        min_div = min(r['h_div'], r['t_div'])
        penalty = (1.0 / (1 + np.log1p(min_deg))) * (1.0 / (1 + min_div))
        return base - penalty * 0.5
    
    comb_unc = [2 - combined(r) for r in results]
    auroc_comb = roc_auc_score(labels, comb_unc)
    print(f"Combined AUROC: {auroc_comb:.4f}")
    
    print(f"\nBest improvement: {max(auroc_deg, auroc_div, auroc_comb) - auroc_binary:+.4f}")

if __name__ == "__main__":
    main()
