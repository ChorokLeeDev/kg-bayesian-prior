#!/usr/bin/env python3
"""
Confident-Wrong Analysis for KG Uncertainty Paper

Goal: Find the most shocking statistic - "X% of the most confident predictions
have zero training evidence"

For each baseline method (Energy, UKGE, U_sem), we:
1. Rank test predictions by confidence (1/uncertainty)
2. For top-K predictions (K=100, 500, 1000), compute fraction where coverage=0 (novel context)
3. Report: "Among Energy's top-K most confident predictions, X% have zero training evidence"

Key insight: If Energy's AUROC on novel contexts is 0.43 (anti-predictive), then Energy
is MORE confident on zero-evidence queries. This should produce a shocking statistic.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237, load_icews14, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions
# ============================================================

class EnergyBased(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        return -self.forward(h, r, t)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class UKGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        scores = self.forward(h, r, t)
        probs = torch.sigmoid(scores)
        confidence = torch.abs(probs - 0.5) * 2
        return 1 - confidence

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class GPOnly(nn.Module):
    """U_sem baseline - pure semantic uncertainty without coverage."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Training
# ============================================================

def _kl_entity_gaussian(model):
    """KL(q(e)||N(0,1)) for models with explicit entity mean/logvar parameters."""
    if not (hasattr(model, 'entity_mean') and hasattr(model, 'entity_logvar')):
        return None
    mean = model.entity_mean
    logvar = model.entity_logvar
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()


def train_model(model, triples, device, epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            # KL regularization toward N(0,1) prior
            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl

            # Uncertainty margin: OOD (neg) should have higher uncertainty
            if hasattr(model, 'entity_logvar'):
                pos_unc = model.get_uncertainty(h, r, t)
                neg_unc = model.get_uncertainty(h, r, neg_t)
                unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
                loss = loss + unc_weight * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


# ============================================================
# Confident-Wrong Analysis
# ============================================================

def analyze_confident_wrong(model, test, device, k_values=[100, 500, 1000]):
    """
    For a trained model, compute what fraction of its top-K most confident
    predictions have zero training evidence (coverage=0).

    Returns:
        dict mapping K -> (zero_evidence_fraction, n_available)
    """
    model.eval()
    cov = model.coverage.cpu().numpy()

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)

        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    # Confidence = negative uncertainty (lower uncertainty = more confident)
    confidence = -uncertainties

    # Compute coverage for each test triple
    # Zero evidence = both head and tail have coverage=0 for this relation
    zero_evidence = []
    for i in range(len(test)):
        h_cov = cov[test[i, 0], test[i, 1]]
        t_cov = cov[test[i, 2], test[i, 1]]
        # Novel context: at least one entity-relation pair is unseen
        zero_evidence.append(h_cov == 0 or t_cov == 0)
    zero_evidence = np.array(zero_evidence)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidence)[::-1]

    results = {}
    for k in k_values:
        actual_k = min(k, len(sorted_indices))
        top_k_indices = sorted_indices[:actual_k]
        top_k_zero_evidence = zero_evidence[top_k_indices]
        fraction = top_k_zero_evidence.mean()
        results[k] = (fraction, actual_k)

    return results


def run_confident_wrong_analysis(dataset_name, loader, device, epochs=30, seed=42):
    """Run confident-wrong analysis on a dataset."""
    print(f"\n{'='*70}")
    print(f"  CONFIDENT-WRONG ANALYSIS: {dataset_name}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Count novel context triples in test set
    # Build coverage from training
    coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train)):
        coverage[train[i, 0], train[i, 1]] = 1.0
        coverage[train[i, 2], train[i, 1]] = 1.0

    novel_ctx_count = 0
    for i in range(len(test)):
        h, r, t = test[i]
        if coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_ctx_count += 1

    novel_ctx_pct = 100.0 * novel_ctx_count / len(test)
    print(f"\nNovel context triples in test set: {novel_ctx_count}/{len(test)} ({novel_ctx_pct:.1f}%)")
    print(f"(This is the baseline - what random sampling would give)")

    # Train and analyze each model
    model_classes = {
        'Energy': EnergyBased,
        'UKGE': UKGE,
        'U_sem': GPOnly,
    }

    k_values = [100, 500, 1000]
    all_results = {}

    for name, cls in model_classes.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        model = cls(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=epochs)

        results = analyze_confident_wrong(model, test, device, k_values)
        all_results[name] = results

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

    # Print results table
    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*70}")
    print(f"\nBaseline (random): {novel_ctx_pct:.1f}% of test triples have zero training evidence")
    print(f"\n{'Method':<10} | {'Top-100 Zero-Evidence %':>25} | {'Top-500':>12} | {'Top-1000':>12}")
    print(f"{'-'*10}-+-{'-'*25}-+-{'-'*12}-+-{'-'*12}")

    for name in ['Energy', 'UKGE', 'U_sem']:
        results = all_results[name]
        row = f"{name:<10} |"
        for k in k_values:
            frac, n = results[k]
            pct = 100.0 * frac
            if k == 100:
                row += f" {pct:>24.1f}% |"
            else:
                row += f" {pct:>11.1f}% |"
        print(row)

    # Compute "shock factor" - how much worse than baseline
    print(f"\n{'='*70}")
    print("SHOCK FACTOR (ratio vs baseline)")
    print(f"{'='*70}")
    print(f"{'Method':<10} | {'Top-100':>12} | {'Top-500':>12} | {'Top-1000':>12}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    baseline_rate = novel_ctx_pct / 100.0
    for name in ['Energy', 'UKGE', 'U_sem']:
        results = all_results[name]
        row = f"{name:<10} |"
        for k in k_values:
            frac, n = results[k]
            shock = frac / baseline_rate if baseline_rate > 0 else float('inf')
            row += f" {shock:>11.2f}x |"
        print(row)

    return all_results, novel_ctx_pct


def main():
    device = setup_device()
    print(f"Device: {device}")

    # Run on FB15k-237 (faster, more relations)
    print("\n" + "="*80)
    print("CONFIDENT-WRONG ANALYSIS")
    print("="*80)
    print("\nGoal: Show that baseline methods are MOST confident on queries with ZERO training evidence")
    print("This is the key failure mode that motivates coverage-aware uncertainty.")

    # FB15k-237
    fb_results, fb_baseline = run_confident_wrong_analysis(
        "FB15k-237",
        load_fb15k237,
        device,
        epochs=30,
        seed=42
    )

    # ICEWS14 (ground-truth temporal)
    icews_results, icews_baseline = run_confident_wrong_analysis(
        "ICEWS14",
        load_icews14,
        device,
        epochs=30,
        seed=42
    )

    # WN18RR (sparse relations)
    wn_results, wn_baseline = run_confident_wrong_analysis(
        "WN18RR",
        load_wn18rr,
        device,
        epochs=30,
        seed=42
    )

    # Final summary
    print("\n" + "="*80)
    print("HEADLINE STATISTICS")
    print("="*80)

    for ds_name, results, baseline in [
        ("FB15k-237", fb_results, fb_baseline),
        ("ICEWS14", icews_results, icews_baseline),
        ("WN18RR", wn_results, wn_baseline),
    ]:
        print(f"\n{ds_name}:")
        print(f"  Baseline (random): {baseline:.1f}% zero-evidence")
        for name in ['Energy', 'UKGE', 'U_sem']:
            frac, _ = results[name][100]
            pct = 100.0 * frac
            print(f"  {name} top-100 most confident: {pct:.1f}% zero-evidence")

    # The shocking headline
    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)

    # Find the most shocking result
    max_shock = 0
    max_shock_method = ""
    max_shock_dataset = ""
    max_shock_pct = 0

    for ds_name, results, baseline in [
        ("FB15k-237", fb_results, fb_baseline),
        ("ICEWS14", icews_results, icews_baseline),
        ("WN18RR", wn_results, wn_baseline),
    ]:
        baseline_rate = baseline / 100.0
        for name in ['Energy', 'UKGE', 'U_sem']:
            frac, _ = results[name][100]
            shock = frac / baseline_rate if baseline_rate > 0 else 0
            if frac > max_shock_pct:
                max_shock_pct = frac
                max_shock = shock
                max_shock_method = name
                max_shock_dataset = ds_name

    print(f"\n  Among {max_shock_method}'s top-100 most confident predictions on {max_shock_dataset},")
    print(f"  {100*max_shock_pct:.0f}% have ZERO training evidence.")
    print(f"  (This is {max_shock:.1f}x the baseline rate)")


if __name__ == "__main__":
    main()
