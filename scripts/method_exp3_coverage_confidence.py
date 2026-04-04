#!/usr/bin/env python3
"""
Method 3: Coverage-Calibrated Confidence
Idea: Use coverage to calibrate model confidence, not just as OOD signal
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237
from sklearn.metrics import roc_auc_score

def main():
    print("="*60)
    print("METHOD 3: Coverage-Calibrated Confidence")
    print("="*60)
    
    ds = load_fb15k237()
    train, test = ds[0].triples, ds[2].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Build coverage with counts
    coverage_count = {}
    for h, r, t in train:
        key_h = (int(h), int(r))
        key_t = (int(t), int(r))
        coverage_count[key_h] = coverage_count.get(key_h, 0) + 1
        coverage_count[key_t] = coverage_count.get(key_t, 0) + 1
    
    # Evaluate different uncertainty formulations
    test_sub = test[:2000]
    
    results = []
    for h, r, t in test_sub:
        h_count = coverage_count.get((int(h), int(r)), 0)
        t_count = coverage_count.get((int(t), int(r)), 0)
        
        is_ood = h_count == 0 or t_count == 0
        
        results.append({
            'h_count': h_count,
            't_count': t_count,
            'min_count': min(h_count, t_count),
            'sum_count': h_count + t_count,
            'binary_cov': int(h_count > 0) + int(t_count > 0),
            'ood': is_ood
        })
    
    labels = [r['ood'] for r in results]
    
    print("Different coverage formulations:")
    
    # Binary coverage (baseline)
    binary_unc = [2 - r['binary_cov'] for r in results]
    auroc_binary = roc_auc_score(labels, binary_unc)
    print(f"  Binary (0/1/2): AUROC={auroc_binary:.4f}")
    
    # Min count (bottleneck)
    min_unc = [-r['min_count'] for r in results]  # Negative = higher unc for lower count
    auroc_min = roc_auc_score(labels, min_unc)
    print(f"  Min count: AUROC={auroc_min:.4f}")
    
    # Sum count
    sum_unc = [-r['sum_count'] for r in results]
    auroc_sum = roc_auc_score(labels, sum_unc)
    print(f"  Sum count: AUROC={auroc_sum:.4f}")
    
    # Log count (diminishing returns)
    log_unc = [-(np.log1p(r['h_count']) + np.log1p(r['t_count'])) for r in results]
    auroc_log = roc_auc_score(labels, log_unc)
    print(f"  Log count: AUROC={auroc_log:.4f}")
    
    # Harmonic mean (balanced)
    def harmonic(a, b):
        if a == 0 or b == 0:
            return 0
        return 2 * a * b / (a + b)
    harm_unc = [-harmonic(r['h_count'], r['t_count']) for r in results]
    auroc_harm = roc_auc_score(labels, harm_unc)
    print(f"  Harmonic mean: AUROC={auroc_harm:.4f}")
    
    # Geometric mean
    geo_unc = [-np.sqrt(r['h_count'] * r['t_count']) for r in results]
    auroc_geo = roc_auc_score(labels, geo_unc)
    print(f"  Geometric mean: AUROC={auroc_geo:.4f}")
    
    best = max([
        ('Binary', auroc_binary),
        ('Min', auroc_min),
        ('Sum', auroc_sum),
        ('Log', auroc_log),
        ('Harmonic', auroc_harm),
        ('Geometric', auroc_geo)
    ], key=lambda x: x[1])
    
    print(f"\nBest: {best[0]} with AUROC={best[1]:.4f}")

if __name__ == "__main__":
    main()
