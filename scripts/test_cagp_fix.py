#!/usr/bin/env python3
"""Quick test: does adding sampling to CAGP.forward() fix the training collapse?
Run on WN18RR (small, fast) with 1 seed. Check if entity_logvar learns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from scripts.run_wn18rr_temporal import (
    CoverageOnly, CAGP, train_model, evaluate_temporal, setup_device,
)
from src.data.loaders import load_wn18rr


class CAGPFixed(nn.Module):
    """CAGP with reparameterization sampling in forward()."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self._norm_stats = None

    def forward(self, h, r, t):
        # THE FIX: use reparameterization sampling
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def calibrate_normalization(self, triples, device):
        with torch.no_grad():
            h = torch.tensor(triples[:, 0]).to(device)
            r = torch.tensor(triples[:, 1]).to(device)
            t = torch.tensor(triples[:, 2]).to(device)
            h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
            t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
            gp_var = (h_var + t_var) / 2
            cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
            self._norm_stats = {
                'gp_mean': gp_var.mean().item(),
                'cov_mean': cov_unc.mean().item(),
            }

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2
        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
        if self._norm_stats is not None:
            gp_mean = self._norm_stats['gp_mean']
            cov_mean = self._norm_stats['cov_mean']
        else:
            gp_mean = gp_var.mean().item()
            cov_mean = cov_unc.mean().item()
        gp_norm = gp_var / (gp_mean + 1e-8) * (cov_mean + 1e-8)
        alpha = torch.sigmoid(self.alpha)
        return alpha * gp_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def main():
    device = setup_device()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading WN18RR...")
    train_ds, _, test_ds = load_wn18rr()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    print(f"Entities: {n_ent}, Relations: {n_rel}")

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

    # 3. CoverageOnly baseline
    print("\n=== CoverageOnly ===")
    torch.manual_seed(seed); np.random.seed(seed)
    m3 = CoverageOnly(n_ent, n_rel)
    m3.precompute_coverage(train)
    m3 = train_model(m3, train, device, epochs=30)
    t3 = evaluate_temporal(m3, train, test, n_ent, device)
    print(f"  Temporal AUROC: {t3.get('overall_auroc', 'N/A'):.4f}")
    print(f"  Emerging: {t3.get('emerging_auroc', 'N/A'):.4f}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"{'Method':<20} {'Overall':>10} {'Emerging':>10} {'logvar std':>12} {'alpha':>8}")
    print(f"{'Original CAGP':<20} {t1.get('overall_auroc',0):.4f}{'':>4} {t1.get('emerging_auroc',0):.4f}{'':>4} {logvar1.std():.6f} {alpha1:.4f}")
    print(f"{'Fixed CAGP':<20} {t2.get('overall_auroc',0):.4f}{'':>4} {t2.get('emerging_auroc',0):.4f}{'':>4} {logvar2.std():.6f} {alpha2:.4f}")
    print(f"{'CoverageOnly':<20} {t3.get('overall_auroc',0):.4f}{'':>4} {t3.get('emerging_auroc',0):.4f}{'':>4} {'N/A':>12} {'N/A':>8}")

if __name__ == "__main__":
    main()
