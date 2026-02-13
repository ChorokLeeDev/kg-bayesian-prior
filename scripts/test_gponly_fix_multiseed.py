#!/usr/bin/env python3
"""Test whether fixing GPOnly sampling changes U_sem numbers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.run_wn18rr_temporal import GPOnly, train_model, evaluate_temporal, setup_device
from src.data.loaders import load_wn18rr, load_fb15k237, load_yago310

SEEDS = [42, 123, 456]

def run_dataset(name, loader, device):
    print(f"\n{'='*60}")
    print(f"  {name} — GPOnly (U_sem) — 3 seeds")
    print(f"{'='*60}")
    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}")

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed); np.random.seed(seed)
        m = GPOnly(n_ent, n_rel)
        m.precompute_coverage(train)
        m = train_model(m, train, device, epochs=30)
        logvar = m.entity_logvar.detach().cpu()
        t = evaluate_temporal(m, train, test, n_ent, device)
        r = {
            'seed': seed,
            'overall': t.get('overall_auroc', 0),
            'emerging': t.get('emerging_auroc', 0),
            'novel_ctx': t.get('novel_ctx_auroc', 'N/A'),
            'logvar_std': logvar.std().item(),
        }
        results.append(r)
        print(f"  GPOnly: overall={r['overall']:.4f} emerging={r['emerging']:.4f} novel_ctx={r['novel_ctx']} logvar_std={r['logvar_std']:.6f}")

    ovs = [r['overall'] for r in results]
    ems = [r['emerging'] for r in results]
    lvs = [r['logvar_std'] for r in results]
    print(f"\n  SUMMARY: overall={np.mean(ovs):.4f}+/-{np.std(ovs):.4f}  emerging={np.mean(ems):.4f}+/-{np.std(ems):.4f}  logvar_std={np.mean(lvs):.6f}")

    # Compare with old canonical numbers
    old = {'WN18RR': 0.658, 'FB15k-237': 0.587, 'YAGO3-10': 0.538}
    if name in old:
        print(f"  OLD U_sem: {old[name]:.3f}  NEW U_sem: {np.mean(ovs):.3f}  DELTA: {np.mean(ovs)-old[name]:+.3f}")

def main():
    device = setup_device()
    for name, loader in [("WN18RR", load_wn18rr), ("FB15k-237", load_fb15k237), ("YAGO3-10", load_yago310)]:
        run_dataset(name, loader, device)

if __name__ == "__main__":
    main()
