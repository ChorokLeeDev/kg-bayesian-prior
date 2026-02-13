#!/usr/bin/env python3
"""Run a single model on YAGO3-10 temporal OOD (for parallel execution)."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import json
import time

# Import the model classes and evaluation functions from the temporal script
from scripts.run_wn18rr_temporal import (
    UKGE, EnergyBased, GPOnly, CoverageOnly, CAGP, RelCondVar,
    train_model, evaluate_ood, evaluate_temporal, setup_device,
)
from src.data.loaders import load_yago310

MODEL_MAP = {
    'UKGE': UKGE,
    'Energy': EnergyBased,
    'GPOnly': GPOnly,
    'CoverageOnly': CoverageOnly,
    'CAGP': CAGP,
    'RelCondVar': RelCondVar,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    device = setup_device()
    print(f"[{args.model}] Device: {device}, Seed: {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[{args.model}] Loading YAGO3-10...")
    train_ds, _, test_ds = load_yago310()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"[{args.model}] Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}")

    t0 = time.time()
    model = MODEL_MAP[args.model](n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=args.epochs)

    if hasattr(model, 'calibrate_normalization'):
        model.calibrate_normalization(train, device)

    random_auroc = evaluate_ood(model, test, n_ent, device)
    print(f"[{args.model}] Random OOD AUROC: {random_auroc:.4f}")

    temporal = evaluate_temporal(model, train, test, n_ent, device, emerging_operator='leq')
    elapsed = time.time() - t0

    if 'overall_auroc' in temporal:
        print(f"[{args.model}] Temporal OOD AUROC: {temporal['overall_auroc']:.4f}")
    if 'emerging_auroc' in temporal:
        print(f"[{args.model}] Emerging AUROC: {temporal['emerging_auroc']:.4f}")
    if 'novel_ctx_auroc' in temporal:
        print(f"[{args.model}] Novel Ctx AUROC: {temporal['novel_ctx_auroc']:.4f}")
    print(f"[{args.model}] Time: {elapsed:.1f}s")

    result = {
        'model': args.model,
        'seed': args.seed,
        'random_auroc': float(random_auroc),
        'temporal': temporal,
        'elapsed': elapsed,
    }

    out = project_root / 'outputs' / f'yago_temporal_{args.model.lower()}_seed{args.seed}.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[{args.model}] Saved to {out}")

if __name__ == "__main__":
    main()
