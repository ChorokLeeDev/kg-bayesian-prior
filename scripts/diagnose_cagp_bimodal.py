#!/usr/bin/env python3
"""Diagnose CAGP bimodality on YAGO: why does seed 42 differ from 123/456?

Key question: Is the GP normalization term truly negligible (cov_mean=0 on train)?
If so, how can the GP affect AUROC at all?
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from scripts.run_wn18rr_temporal import (
    CAGP, CoverageOnly, GPOnly, train_model, setup_device, _is_emerging,
)
from src.data.loaders import load_yago310

def main():
    device = setup_device()
    print(f"Device: {device}")

    print("Loading YAGO3-10...")
    train_ds, _, test_ds = load_yago310()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}, Train: {len(train)}, Test: {len(test)}")

    # Step 1: Verify coverage stats on training set (seed-independent)
    print("\n=== Step 1: Coverage verification ===")
    model = CAGP(n_ent, n_rel)
    model.precompute_coverage(train)

    with torch.no_grad():
        h = torch.tensor(train[:, 0])
        r = torch.tensor(train[:, 1])
        t = torch.tensor(train[:, 2])
        cov_h = model.coverage[h, r]
        cov_t = model.coverage[t, r]
        cov_unc = 2.0 - cov_h - cov_t
        print(f"Coverage[h,r] min={cov_h.min():.6f} max={cov_h.max():.6f} mean={cov_h.mean():.6f}")
        print(f"Coverage[t,r] min={cov_t.min():.6f} max={cov_t.max():.6f} mean={cov_t.mean():.6f}")
        print(f"cov_unc on train: min={cov_unc.min():.10f} max={cov_unc.max():.10f} mean={cov_unc.mean():.10f}")
        print(f"cov_unc == 0 count: {(cov_unc == 0).sum()} / {len(cov_unc)}")

    # Step 2: Check GP stats with untrained model
    print("\n=== Step 2: Untrained model normalization ===")
    with torch.no_grad():
        h_var = torch.exp(model.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(model.entity_logvar[t]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2
        print(f"Untrained gp_var: mean={gp_var.mean():.6f} std={gp_var.std():.6f}")
        print(f"Untrained entity_logvar: mean={model.entity_logvar.mean():.4f} std={model.entity_logvar.std():.4f}")

        # What calibration would give
        gp_mean = gp_var.mean().item()
        cov_mean = cov_unc.mean().item()
        print(f"calibrate_normalization would give: gp_mean={gp_mean:.8f}, cov_mean={cov_mean:.8f}")
        print(f"GP scaling factor: cov_mean / (gp_mean + 1e-8) = {(cov_mean + 1e-8) / (gp_mean + 1e-8):.2e}")

    # Step 3: Categorize test triples
    print("\n=== Step 3: Test set categorization ===")
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    thresh = np.percentile(list(freq.values()), 25)
    cov_np = model.coverage.cpu().numpy()

    emerging_idx, novel_idx, id_idx = [], [], []
    for i in range(len(test)):
        h_i, r_i, t_i = test[i]
        if _is_emerging(freq.get(h_i, 0), freq.get(t_i, 0), thresh, 'leq'):
            emerging_idx.append(i)
        elif cov_np[h_i, r_i] == 0 or cov_np[t_i, r_i] == 0:
            novel_idx.append(i)
        else:
            id_idx.append(i)

    print(f"Split: emerging={len(emerging_idx)}, novel_ctx={len(novel_idx)}, id={len(id_idx)}")

    # Check coverage of emerging vs ID test triples
    with torch.no_grad():
        e_triples = test[emerging_idx]
        i_triples = test[id_idx]
        e_cov_unc = 2.0 - model.coverage[torch.tensor(e_triples[:, 0]), torch.tensor(e_triples[:, 1])] - \
                    model.coverage[torch.tensor(e_triples[:, 2]), torch.tensor(e_triples[:, 1])]
        i_cov_unc = 2.0 - model.coverage[torch.tensor(i_triples[:, 0]), torch.tensor(i_triples[:, 1])] - \
                    model.coverage[torch.tensor(i_triples[:, 2]), torch.tensor(i_triples[:, 1])]

        print(f"\nEmerging cov_unc: mean={e_cov_unc.mean():.4f} std={e_cov_unc.std():.4f}")
        print(f"  values: {dict(zip(*np.unique(e_cov_unc.numpy(), return_counts=True)))}")
        print(f"ID cov_unc: mean={i_cov_unc.mean():.4f} std={i_cov_unc.std():.4f}")
        print(f"  values: {dict(zip(*np.unique(i_cov_unc.numpy(), return_counts=True)))}")

    # Step 4: Train CAGP for each seed and extract diagnostics
    for seed in [42, 123, 456]:
        print(f"\n{'='*60}")
        print(f"=== Step 4: Training CAGP seed {seed} (30 epochs) ===")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CAGP(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)

        # Extract diagnostics
        model.eval()
        with torch.no_grad():
            alpha = torch.sigmoid(model.alpha).item()
            logvar = model.entity_logvar.cpu()
            print(f"\nalpha (sigmoid): {alpha:.6f}")
            print(f"alpha (logit): {model.alpha.item():.6f}")
            print(f"entity_logvar: mean={logvar.mean():.4f} std={logvar.std():.4f} "
                  f"min={logvar.min():.4f} max={logvar.max():.4f}")

            # GP variance on training set
            h_t = torch.tensor(train[:, 0]).to(device)
            r_t = torch.tensor(train[:, 1]).to(device)
            t_t = torch.tensor(train[:, 2]).to(device)
            h_var_train = torch.exp(model.entity_logvar[h_t]).mean(dim=-1)
            t_var_train = torch.exp(model.entity_logvar[t_t]).mean(dim=-1)
            gp_var_train = (h_var_train + t_var_train) / 2
            cov_unc_train = 2.0 - model.coverage[h_t, r_t] - model.coverage[t_t, r_t]

            gp_mean_train = gp_var_train.mean().item()
            cov_mean_train = cov_unc_train.mean().item()

            print(f"\nCalibration stats (train set):")
            print(f"  gp_mean={gp_mean_train:.8e}")
            print(f"  cov_mean={cov_mean_train:.8e}")
            print(f"  GP scaling: (cov_mean+1e-8)/(gp_mean+1e-8) = {(cov_mean_train+1e-8)/(gp_mean_train+1e-8):.2e}")

        # Calibrate normalization
        model.calibrate_normalization(train, device)
        print(f"  Cached norm_stats: {model._norm_stats}")

        # Evaluate emerging vs ID with full breakdown
        with torch.no_grad():
            e_triples = test[emerging_idx]
            i_triples = test[id_idx]

            # Emerging uncertainties
            h_e = torch.tensor(e_triples[:, 0]).to(device)
            r_e = torch.tensor(e_triples[:, 1]).to(device)
            t_e = torch.tensor(e_triples[:, 2]).to(device)

            gp_e = model.get_gp_variance(h_e, t_e) if hasattr(model, 'get_gp_variance') else None
            # Manual decomposition
            h_var_e = torch.exp(model.entity_logvar[h_e]).mean(dim=-1)
            t_var_e = torch.exp(model.entity_logvar[t_e]).mean(dim=-1)
            gp_var_e = (h_var_e + t_var_e) / 2
            cov_unc_e = 2.0 - model.coverage[h_e, r_e] - model.coverage[t_e, r_e]
            gp_norm_e = gp_var_e / (model._norm_stats['gp_mean'] + 1e-8) * (model._norm_stats['cov_mean'] + 1e-8)
            combined_e = alpha * gp_norm_e + (1 - alpha) * cov_unc_e

            # ID uncertainties
            h_i = torch.tensor(i_triples[:, 0]).to(device)
            r_i = torch.tensor(i_triples[:, 1]).to(device)
            t_i = torch.tensor(i_triples[:, 2]).to(device)

            h_var_i = torch.exp(model.entity_logvar[h_i]).mean(dim=-1)
            t_var_i = torch.exp(model.entity_logvar[t_i]).mean(dim=-1)
            gp_var_i = (h_var_i + t_var_i) / 2
            cov_unc_i = 2.0 - model.coverage[h_i, r_i] - model.coverage[t_i, r_i]
            gp_norm_i = gp_var_i / (model._norm_stats['gp_mean'] + 1e-8) * (model._norm_stats['cov_mean'] + 1e-8)
            combined_i = alpha * gp_norm_i + (1 - alpha) * cov_unc_i

            print(f"\n  Emerging (n={len(e_triples)}):")
            print(f"    gp_var: mean={gp_var_e.mean():.6f} std={gp_var_e.std():.6f}")
            print(f"    gp_norm: mean={gp_norm_e.mean():.2e} std={gp_norm_e.std():.2e} "
                  f"min={gp_norm_e.min():.2e} max={gp_norm_e.max():.2e}")
            print(f"    cov_unc: mean={cov_unc_e.mean():.4f}")
            print(f"    combined: mean={combined_e.mean():.6e} std={combined_e.std():.6e}")

            print(f"  ID (n={len(i_triples)}):")
            print(f"    gp_var: mean={gp_var_i.mean():.6f} std={gp_var_i.std():.6f}")
            print(f"    gp_norm: mean={gp_norm_i.mean():.2e} std={gp_norm_i.std():.2e} "
                  f"min={gp_norm_i.min():.2e} max={gp_norm_i.max():.2e}")
            print(f"    cov_unc: mean={cov_unc_i.mean():.4f}")
            print(f"    combined: mean={combined_i.mean():.6e} std={combined_i.std():.6e}")

            # Compute AUROCs
            labels = np.concatenate([np.zeros(len(i_triples)), np.ones(len(e_triples))])

            gp_scores = np.concatenate([gp_var_i.cpu().numpy(), gp_var_e.cpu().numpy()])
            cov_scores = np.concatenate([cov_unc_i.cpu().numpy(), cov_unc_e.cpu().numpy()])
            gp_norm_scores = np.concatenate([gp_norm_i.cpu().numpy(), gp_norm_e.cpu().numpy()])
            combined_scores = np.concatenate([combined_i.cpu().numpy(), combined_e.cpu().numpy()])

            print(f"\n  Emerging AUROC breakdown (seed {seed}):")
            print(f"    GP variance (raw):  {roc_auc_score(labels, gp_scores):.4f}")
            print(f"    GP variance (norm): {roc_auc_score(labels, gp_norm_scores):.4f}")
            print(f"    Coverage:           {roc_auc_score(labels, cov_scores):.4f}")
            print(f"    Combined (CAGP):    {roc_auc_score(labels, combined_scores):.4f}")

            # Also check via model.get_uncertainty
            unc_e = model.get_uncertainty(h_e, r_e, t_e).cpu().numpy()
            unc_i = model.get_uncertainty(h_i, r_i, t_i).cpu().numpy()
            model_labels = np.concatenate([np.zeros(len(unc_i)), np.ones(len(unc_e))])
            model_scores = np.concatenate([unc_i, unc_e])
            print(f"    model.get_uncertainty: {roc_auc_score(model_labels, model_scores):.4f}")

if __name__ == "__main__":
    main()
