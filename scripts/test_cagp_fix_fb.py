#!/usr/bin/env python3
"""Test CAGP fix on FB15k-237 (bigger dataset, 237 relations)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from scripts.run_wn18rr_temporal import (
    CoverageOnly, CAGP, train_model, evaluate_temporal, setup_device,
)
from scripts.test_cagp_fix import CAGPFixed
from src.data.loaders import load_fb15k237


def main():
    device = setup_device()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading FB15k-237...")
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}, Test: {len(test)}")

    # 1. Original CAGP (broken)
    print("\n=== Original CAGP (no sampling) ===")
    torch.manual_seed(seed); np.random.seed(seed)
    m1 = CAGP(n_ent, n_rel)
    m1.precompute_coverage(train)
    m1 = train_model(m1, train, device, epochs=30)
    m1.calibrate_normalization(train, device)
    logvar1 = m1.entity_logvar.detach().cpu()
    alpha1 = torch.sigmoid(m1.alpha).item()
    print(f"  alpha: {alpha1:.4f}")
    print(f"  logvar: mean={logvar1.mean():.4f} std={logvar1.std():.6f}")
    print(f"  norm_stats: gp_mean={m1._norm_stats['gp_mean']:.6f} cov_mean={m1._norm_stats['cov_mean']:.6f}")
    t1 = evaluate_temporal(m1, train, test, n_ent, device)
    print(f"  Temporal AUROC: {t1.get('overall_auroc', 'N/A'):.4f}")
    print(f"  Emerging: {t1.get('emerging_auroc', 'N/A'):.4f}")
    print(f"  Novel ctx: {t1.get('novel_ctx_auroc', 'N/A')}")

    # 2. Fixed CAGP (with sampling)
    print("\n=== Fixed CAGP (with sampling) ===")
    torch.manual_seed(seed); np.random.seed(seed)
    m2 = CAGPFixed(n_ent, n_rel)
    m2.precompute_coverage(train)
    m2 = train_model(m2, train, device, epochs=30)
    m2.calibrate_normalization(train, device)
    logvar2 = m2.entity_logvar.detach().cpu()
    alpha2 = torch.sigmoid(m2.alpha).item()
    print(f"  alpha: {alpha2:.4f}")
    print(f"  logvar: mean={logvar2.mean():.4f} std={logvar2.std():.6f}")
    print(f"  norm_stats: gp_mean={m2._norm_stats['gp_mean']:.6f} cov_mean={m2._norm_stats['cov_mean']:.6f}")
    t2 = evaluate_temporal(m2, train, test, n_ent, device)
    print(f"  Temporal AUROC: {t2.get('overall_auroc', 'N/A'):.4f}")
    print(f"  Emerging: {t2.get('emerging_auroc', 'N/A'):.4f}")
    print(f"  Novel ctx: {t2.get('novel_ctx_auroc', 'N/A')}")

    # 3. CoverageOnly baseline
    print("\n=== CoverageOnly ===")
    torch.manual_seed(seed); np.random.seed(seed)
    m3 = CoverageOnly(n_ent, n_rel)
    m3.precompute_coverage(train)
    m3 = train_model(m3, train, device, epochs=30)
    t3 = evaluate_temporal(m3, train, test, n_ent, device)
    print(f"  Temporal AUROC: {t3.get('overall_auroc', 'N/A'):.4f}")
    print(f"  Emerging: {t3.get('emerging_auroc', 'N/A'):.4f}")
    print(f"  Novel ctx: {t3.get('novel_ctx_auroc', 'N/A')}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — FB15k-237")
    print("="*60)
    print(f"{'Method':<20} {'Overall':>10} {'Emerging':>10} {'Novel ctx':>10} {'logvar std':>12} {'alpha':>8}")
    e1 = t1.get('emerging_auroc', 0); n1 = t1.get('novel_ctx_auroc', 'N/A')
    e2 = t2.get('emerging_auroc', 0); n2 = t2.get('novel_ctx_auroc', 'N/A')
    e3 = t3.get('emerging_auroc', 0); n3 = t3.get('novel_ctx_auroc', 'N/A')
    print(f"{'Original CAGP':<20} {t1.get('overall_auroc',0):>10.4f} {e1:>10.4f} {str(n1):>10} {logvar1.std():>12.6f} {alpha1:>8.4f}")
    print(f"{'Fixed CAGP':<20} {t2.get('overall_auroc',0):>10.4f} {e2:>10.4f} {str(n2):>10} {logvar2.std():>12.6f} {alpha2:>8.4f}")
    print(f"{'CoverageOnly':<20} {t3.get('overall_auroc',0):>10.4f} {e3:>10.4f} {str(n3):>10} {'N/A':>12} {'N/A':>8}")

if __name__ == "__main__":
    main()
