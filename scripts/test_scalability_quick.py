#!/usr/bin/env python3
"""
Quick scalability test: How does RCUE scale with entity count?
Test on synthetic KGs of increasing size.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score

from src.models.relation_conditioned import RCUE, train_rcue


def generate_synthetic_kg(n_entities, n_relations=50, n_triples_per_entity=10):
    """Generate synthetic KG with controlled size."""
    triples = []
    for e in range(n_entities):
        # Each entity participates in ~n_triples_per_entity triples
        for _ in range(n_triples_per_entity):
            r = np.random.randint(0, n_relations)
            t = np.random.randint(0, n_entities)
            triples.append([e, r, t])
    return np.array(triples)


def test_scalability(n_entities, n_relations=50, epochs=10):
    """Test RCUE on synthetic KG of given size."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Generate data
    np.random.seed(42)
    train = generate_synthetic_kg(n_entities, n_relations)
    test = generate_synthetic_kg(n_entities // 10, n_relations)  # 10% for test

    # Build coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])

    # Train
    torch.manual_seed(42)
    model = RCUE(n_entities, n_relations, use_coverage=True)

    start = time.time()
    model = train_rcue(model, train, device, epochs=epochs, verbose=False)
    train_time = time.time() - start

    # Evaluate
    model.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    start = time.time()
    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()
    infer_time = time.time() - start

    auroc = roc_auc_score(ood_mask, unc)

    # Memory
    param_count = sum(p.numel() for p in model.parameters())
    coverage_size = model.coverage.numel()

    return {
        'n_entities': n_entities,
        'n_triples': len(train),
        'train_time': train_time,
        'infer_time': infer_time,
        'auroc': auroc,
        'params': param_count,
        'coverage_size': coverage_size,
        'ood_frac': ood_mask.mean()
    }


def main():
    print("RCUE Scalability Test")
    print("="*70)

    results = []
    for n_ent in [1000, 5000, 10000, 50000]:
        print(f"\nTesting n_entities={n_ent}...")
        r = test_scalability(n_ent, epochs=5)
        results.append(r)
        print(f"  Triples: {r['n_triples']:,}")
        print(f"  Train time: {r['train_time']:.1f}s")
        print(f"  Infer time: {r['infer_time']:.3f}s")
        print(f"  AUROC: {r['auroc']:.4f}")
        print(f"  OOD fraction: {r['ood_frac']*100:.1f}%")
        print(f"  Parameters: {r['params']:,}")
        print(f"  Coverage matrix: {r['coverage_size']:,} ({r['coverage_size']*4/1024/1024:.1f} MB)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Entities':<12} {'Triples':<12} {'Train(s)':<10} {'Infer(s)':<10} {'AUROC':<8} {'Cov(MB)':<10}")
    print("-"*70)
    for r in results:
        cov_mb = r['coverage_size'] * 4 / 1024 / 1024
        print(f"{r['n_entities']:<12,} {r['n_triples']:<12,} {r['train_time']:<10.1f} {r['infer_time']:<10.3f} {r['auroc']:<8.4f} {cov_mb:<10.1f}")

    # Extrapolate to WikiKG2 scale (500K entities)
    print("\n" + "="*70)
    print("EXTRAPOLATION TO WIKIKG2 SCALE (500K entities)")
    print("="*70)

    # Linear extrapolation from 50K
    if len(results) >= 4:
        r50k = results[-1]
        scale = 500000 / 50000  # 10x

        # Training scales ~linearly with triples
        est_train = r50k['train_time'] * scale
        # Inference scales ~linearly with test size
        est_infer = r50k['infer_time'] * scale
        # Coverage matrix scales with entities * relations
        est_cov_mb = r50k['coverage_size'] * scale * 4 / 1024 / 1024

        print(f"Estimated train time: {est_train/60:.1f} min")
        print(f"Estimated inference: {est_infer:.1f}s")
        print(f"Estimated coverage matrix: {est_cov_mb:.0f} MB")
        print(f"\nConclusion: Coverage matrix is the bottleneck ({est_cov_mb:.0f} MB)")
        print("For 500K entities × 500 relations = 250M entries = 1GB (float32)")
        print("Bloom filter alternative: ~100MB with 1% FPR")


if __name__ == "__main__":
    main()
