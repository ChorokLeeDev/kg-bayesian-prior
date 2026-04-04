#!/usr/bin/env python3
"""
Evaluate link prediction metrics (MRR, Hits@k) for RCUE.
Question: Does RCUE maintain competitive link prediction while improving OOD?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data.loaders import load_fb15k237
from src.models.relation_conditioned import RCUE, train_rcue, evaluate_link_prediction


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    results = {}

    # 1. Energy baseline
    print("\n--- Energy Baseline ---")
    torch.manual_seed(42)

    from scripts.rcue_experiment import EnergyBaseline, train_baseline
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_baseline(energy, train, device, epochs=30)

    lp_metrics = evaluate_link_prediction(energy, test, device)
    results['Energy'] = lp_metrics
    print(f"  MRR: {lp_metrics['mrr']:.4f}")
    print(f"  Hits@1: {lp_metrics['hits@1']:.4f}")
    print(f"  Hits@10: {lp_metrics['hits@10']:.4f}")

    # 2. RCUE
    print("\n--- RCUE ---")
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=30, verbose=True)

    lp_metrics = evaluate_link_prediction(rcue, test, device)
    results['RCUE'] = lp_metrics
    print(f"  MRR: {lp_metrics['mrr']:.4f}")
    print(f"  Hits@1: {lp_metrics['hits@1']:.4f}")
    print(f"  Hits@10: {lp_metrics['hits@10']:.4f}")

    # 3. RCUE without coverage (ablation)
    print("\n--- RCUE (no coverage) ---")
    torch.manual_seed(42)
    rcue_nocov = RCUE(n_ent, n_rel, use_coverage=False)
    rcue_nocov = train_rcue(rcue_nocov, train, device, epochs=30, verbose=True)

    lp_metrics = evaluate_link_prediction(rcue_nocov, test, device)
    results['RCUE-noCov'] = lp_metrics
    print(f"  MRR: {lp_metrics['mrr']:.4f}")
    print(f"  Hits@1: {lp_metrics['hits@1']:.4f}")
    print(f"  Hits@10: {lp_metrics['hits@10']:.4f}")

    # Summary
    print("\n" + "="*50)
    print("LINK PREDICTION SUMMARY")
    print("="*50)
    print(f"{'Method':<15} {'MRR':<10} {'H@1':<10} {'H@10':<10}")
    print("-"*45)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['mrr']:.4f}     {metrics['hits@1']:.4f}     {metrics['hits@10']:.4f}")


if __name__ == "__main__":
    main()
