#!/usr/bin/env python3
"""Multi-seed (3 seeds) test of the CAGP fix across all 3 datasets."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.run_wn18rr_temporal import (
    CoverageOnly, train_model, evaluate_temporal, setup_device,
)
from scripts.test_cagp_fix import CAGPFixed
from src.data.loaders import load_wn18rr, load_fb15k237, load_yago310

SEEDS = [42, 123, 456]

def run_dataset(name, loader, device):
    print(f"\n{'='*60}")
    print(f"  {name} — 3 seeds")
    print(f"{'='*60}")

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}, Test: {len(test)}")

    results = {'fixed_cagp': [], 'coverage_only': []}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Fixed CAGP
        torch.manual_seed(seed); np.random.seed(seed)
        m = CAGPFixed(n_ent, n_rel)
        m.precompute_coverage(train)
        m = train_model(m, train, device, epochs=30)
        m.calibrate_normalization(train, device)
        logvar = m.entity_logvar.detach().cpu()
        alpha = torch.sigmoid(m.alpha).item()
        t = evaluate_temporal(m, train, test, n_ent, device)
        r = {
            'seed': seed,
            'overall_auroc': t.get('overall_auroc', 0),
            'emerging_auroc': t.get('emerging_auroc', 0),
            'novel_ctx_auroc': t.get('novel_ctx_auroc', 'N/A'),
            'logvar_std': logvar.std().item(),
            'logvar_mean': logvar.mean().item(),
            'alpha': alpha,
            'gp_mean': m._norm_stats['gp_mean'],
        }
        results['fixed_cagp'].append(r)
        print(f"  Fixed CAGP: overall={r['overall_auroc']:.4f} emerging={r['emerging_auroc']:.4f} logvar_std={r['logvar_std']:.6f} alpha={alpha:.4f}")

        # CoverageOnly
        torch.manual_seed(seed); np.random.seed(seed)
        m3 = CoverageOnly(n_ent, n_rel)
        m3.precompute_coverage(train)
        m3 = train_model(m3, train, device, epochs=30)
        t3 = evaluate_temporal(m3, train, test, n_ent, device)
        r3 = {
            'seed': seed,
            'overall_auroc': t3.get('overall_auroc', 0),
            'emerging_auroc': t3.get('emerging_auroc', 0),
            'novel_ctx_auroc': t3.get('novel_ctx_auroc', 'N/A'),
        }
        results['coverage_only'].append(r3)
        print(f"  CoverageOnly: overall={r3['overall_auroc']:.4f} emerging={r3['emerging_auroc']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {name} — SUMMARY (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    for method in ['fixed_cagp', 'coverage_only']:
        ovs = [r['overall_auroc'] for r in results[method]]
        ems = [r['emerging_auroc'] for r in results[method]]
        om, os = np.mean(ovs), np.std(ovs)
        em, es = np.mean(ems), np.std(ems)
        print(f"  {method:<15}: overall={om:.4f}±{os:.4f}  emerging={em:.4f}±{es:.4f}")
        if method == 'fixed_cagp':
            lvs = [r['logvar_std'] for r in results[method]]
            print(f"  {'':15}  logvar_std={np.mean(lvs):.6f}±{np.std(lvs):.6f}")

    # Save JSON
    outpath = Path(f"outputs/{name.lower().replace('-','')}_fixed_cagp_multiseed.json")
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {outpath}")

    return results


def main():
    device = setup_device()
    all_results = {}

    for name, loader in [
        ("WN18RR", load_wn18rr),
        ("FB15k-237", load_fb15k237),
        ("YAGO3-10", load_yago310),
    ]:
        all_results[name] = run_dataset(name, loader, device)

    # Grand summary
    print(f"\n{'='*60}")
    print(f"  GRAND SUMMARY — Fixed CAGP (mean ± std)")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'CAGP Overall':>16} {'CAGP Emerging':>16} {'Cov Overall':>16} {'Cov Emerging':>16}")
    for name in ["WN18RR", "FB15k-237", "YAGO3-10"]:
        fc = all_results[name]['fixed_cagp']
        co = all_results[name]['coverage_only']
        fco = np.mean([r['overall_auroc'] for r in fc])
        fcs = np.std([r['overall_auroc'] for r in fc])
        fce = np.mean([r['emerging_auroc'] for r in fc])
        fces = np.std([r['emerging_auroc'] for r in fc])
        coo = np.mean([r['overall_auroc'] for r in co])
        cos = np.std([r['overall_auroc'] for r in co])
        coe = np.mean([r['emerging_auroc'] for r in co])
        coes = np.std([r['emerging_auroc'] for r in co])
        print(f"{name:<12} {fco:.3f}±{fcs:.3f}{'':>4} {fce:.3f}±{fces:.3f}{'':>4} {coo:.3f}±{cos:.3f}{'':>4} {coe:.3f}±{coes:.3f}")


if __name__ == "__main__":
    main()
