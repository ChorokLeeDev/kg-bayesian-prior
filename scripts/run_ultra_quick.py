#!/usr/bin/env python3
"""Quick ULTRA blind spot test - CPU version with smaller sample."""
import sys
import os
sys.path.insert(0, '/Users/i767700/Github/ultra_test')
os.chdir('/Users/i767700/Github/ultra_test')

import torch
import numpy as np
from collections import defaultdict

print("="*60)
print("ULTRA Foundation Model - Coverage Blind Spot Test")
print("="*60)

# Check if ULTRA is available
try:
    from ultra import tasks, util
    from ultra.models import Ultra
    print("✓ ULTRA imported successfully")
except ImportError as e:
    print(f"✗ ULTRA not available: {e}")
    print("Creating mock test with random scores...")
    
    # Fallback: demonstrate the analysis framework
    np.random.seed(42)
    n_test = 1000
    
    # Simulate: novel context entities have similar scores to ID
    id_scores = np.random.normal(0.5, 0.15, n_test // 2)
    novel_scores = np.random.normal(0.52, 0.15, n_test // 2)  # Nearly identical
    emerging_scores = np.random.normal(0.7, 0.2, n_test // 2)  # Higher uncertainty
    
    from sklearn.metrics import roc_auc_score
    
    # Novel context detection
    labels_novel = np.concatenate([np.zeros(n_test//2), np.ones(n_test//2)])
    scores_novel = np.concatenate([id_scores, novel_scores])
    auroc_novel = roc_auc_score(labels_novel, scores_novel)
    
    # Emerging entity detection
    labels_emerge = np.concatenate([np.zeros(n_test//2), np.ones(n_test//2)])
    scores_emerge = np.concatenate([id_scores, emerging_scores])
    auroc_emerge = roc_auc_score(labels_emerge, scores_emerge)
    
    print(f"\n[SIMULATED - ULTRA not installed]")
    print(f"Novel Context AUROC: {auroc_novel:.3f} (expected ~0.5)")
    print(f"Emerging Entity AUROC: {auroc_emerge:.3f} (expected >0.6)")
    print(f"\nTo run real ULTRA test, use Colab notebook.")
    sys.exit(0)

# If ULTRA is available, run actual test
print("\nLoading FB15k-237...")
# ... (full implementation would go here)
