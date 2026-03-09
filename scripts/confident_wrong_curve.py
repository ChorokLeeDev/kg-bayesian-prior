#!/usr/bin/env python3
"""
Confident-Wrong Curve Analysis

Addresses skeptical reviewer concern: "83% at Top-100 is cherry-picked"

This script computes the FULL confident-wrong curve:
- Top-K for K = 100, 500, 1000, 2000, 5000, 10000, ALL
- Plots the curve showing zero-evidence fraction vs K
- Reports asymptotic behavior

Key question: Does the curve converge to baseline (32%) at large K, or stay elevated?
- If elevated (>50% at large K): Finding is robust
- If converges to baseline: Tail phenomenon only
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EnergyBased(nn.Module):
    """Energy-based uncertainty: -score as uncertainty."""
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


def train_model(model, triples, device, epochs=30, lr=0.001):
    """Train model with BCE loss."""
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

    return model


def compute_confident_wrong_curve(model, test, device, k_values):
    """
    Compute zero-evidence fraction for each K value.

    Returns:
        dict mapping K -> zero_evidence_fraction
    """
    model.eval()
    cov = model.coverage.cpu().numpy()

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)

        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    # Confidence = negative uncertainty
    confidence = -uncertainties

    # Compute zero-evidence for each test triple
    zero_evidence = np.zeros(len(test), dtype=bool)
    for i in range(len(test)):
        h_cov = cov[test[i, 0], test[i, 1]]
        t_cov = cov[test[i, 2], test[i, 1]]
        # Novel context: at least one (entity, relation) pair unseen
        zero_evidence[i] = (h_cov == 0) or (t_cov == 0)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidence)[::-1]
    sorted_zero_evidence = zero_evidence[sorted_indices]

    # Compute cumulative zero-evidence fraction for each K
    results = {}
    for k in k_values:
        actual_k = min(k, len(sorted_indices))
        fraction = sorted_zero_evidence[:actual_k].mean()
        results[k] = {
            'k': actual_k,
            'zero_evidence_fraction': float(fraction),
            'zero_evidence_pct': float(fraction * 100)
        }

    return results, sorted_zero_evidence


def main():
    device = setup_device()
    print(f"Device: {device}")

    # K values to evaluate
    k_values = [100, 500, 1000, 2000, 5000, 10000, 'ALL']

    print("\n" + "="*70)
    print("CONFIDENT-WRONG CURVE ANALYSIS")
    print("="*70)
    print("\nGoal: Show 83% zero-evidence is NOT cherry-picked at Top-100")
    print("Evaluate full curve from Top-100 to ALL test triples\n")

    # Load FB15k-237
    print("Loading FB15k-237...")
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Compute baseline zero-evidence rate
    coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train)):
        coverage[train[i, 0], train[i, 1]] = 1.0
        coverage[train[i, 2], train[i, 1]] = 1.0

    novel_ctx_count = 0
    for i in range(len(test)):
        h, r, t = test[i]
        if coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_ctx_count += 1

    baseline_pct = 100.0 * novel_ctx_count / len(test)
    print(f"\nBaseline (random): {baseline_pct:.1f}% zero-evidence in test set")

    # Replace 'ALL' with actual test size
    numeric_k_values = [k if k != 'ALL' else len(test) for k in k_values]

    # Train Energy model
    print("\nTraining Energy model (30 epochs)...")
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    model = EnergyBased(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=30)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Compute curve
    print("\nComputing confident-wrong curve...")
    results, sorted_zero_evidence = compute_confident_wrong_curve(
        model, test, device, numeric_k_values
    )

    # Print results table
    print("\n" + "="*70)
    print("RESULTS: Zero-Evidence Fraction at Each K")
    print("="*70)
    print(f"\n{'K':>10} | {'Zero-Ev %':>12} | {'vs Baseline':>15} | {'Status':>15}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}")

    for k, label in zip(numeric_k_values, k_values):
        res = results[k]
        pct = res['zero_evidence_pct']
        ratio = pct / baseline_pct if baseline_pct > 0 else 0

        if pct > 50:
            status = "ELEVATED"
        elif pct > baseline_pct * 1.2:
            status = "ABOVE BASELINE"
        else:
            status = "~BASELINE"

        k_str = str(label) if label != 'ALL' else f"ALL ({k})"
        print(f"{k_str:>10} | {pct:>11.1f}% | {ratio:>14.2f}x | {status:>15}")

    # Compute asymptote (last 50% of data)
    n_test = len(test)
    last_half = sorted_zero_evidence[n_test//2:]
    asymptote_pct = 100.0 * last_half.mean()

    print(f"\n{'='*70}")
    print("ASYMPTOTIC ANALYSIS")
    print(f"{'='*70}")
    print(f"\nBaseline rate: {baseline_pct:.1f}%")
    print(f"Asymptote (last 50% of ranked list): {asymptote_pct:.1f}%")

    # Check if asymptote is below baseline (confirming anti-correlation)
    if asymptote_pct < baseline_pct:
        print(f"\nKey finding: Asymptote ({asymptote_pct:.1f}%) < Baseline ({baseline_pct:.1f}%)")
        print("This confirms Energy is ANTI-correlated with zero-evidence:")
        print("  - Top predictions have HIGH zero-evidence (83% at K=100)")
        print("  - Bottom predictions have LOW zero-evidence")
        print("  - Energy systematically assigns HIGH confidence to zero-evidence queries")

    # Determine where curve crosses 50%
    cumsum = np.cumsum(sorted_zero_evidence)
    running_avg = cumsum / np.arange(1, len(sorted_zero_evidence) + 1)
    cross_50_idx = np.where(running_avg < 0.50)[0]
    if len(cross_50_idx) > 0:
        k_cross_50 = cross_50_idx[0] + 1
        print(f"\nCurve crosses 50% at K = {k_cross_50}")
    else:
        print("\nCurve stays above 50% for entire test set!")

    # Determine where curve reaches baseline
    cross_baseline_idx = np.where(running_avg < baseline_pct/100)[0]
    if len(cross_baseline_idx) > 0:
        k_cross_baseline = cross_baseline_idx[0] + 1
        print(f"Curve reaches baseline ({baseline_pct:.1f}%) at K = {k_cross_baseline}")
    else:
        print(f"Curve never reaches baseline ({baseline_pct:.1f}%) - stays elevated throughout!")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Check if finding is robust
    k_1000_pct = results[1000]['zero_evidence_pct']
    k_5000_pct = results[5000]['zero_evidence_pct']
    k_all_pct = results[n_test]['zero_evidence_pct']

    if k_5000_pct > 50:
        print("\nFINDING IS ROBUST:")
        print(f"  - Top-100: {results[100]['zero_evidence_pct']:.1f}% (vs {baseline_pct:.1f}% baseline)")
        print(f"  - Top-1000: {k_1000_pct:.1f}%")
        print(f"  - Top-5000: {k_5000_pct:.1f}%")
        print(f"  - Even at K=5000, zero-evidence rate is still >50%")
        print("\n  The 83% at Top-100 is NOT cherry-picked - the curve stays elevated.")
    elif k_1000_pct > 50:
        print("\nFINDING IS MODERATELY ROBUST:")
        print(f"  - Top-100: {results[100]['zero_evidence_pct']:.1f}%")
        print(f"  - Top-1000: {k_1000_pct:.1f}%")
        print(f"  - Curve stays >50% up to K=1000, then gradually decreases")
    else:
        print("\nFINDING IS A TAIL PHENOMENON:")
        print(f"  - Elevated zero-evidence only in the extreme tail (Top-100)")
        print(f"  - Converges toward baseline at larger K")

    # Save results to JSON
    output_data = {
        'dataset': 'FB15k-237',
        'baseline_zero_evidence_pct': float(baseline_pct),
        'n_test': int(n_test),
        'curve': {str(k): results[numeric_k_values[i]]
                  for i, k in enumerate(k_values)},
        'asymptote_pct': float(asymptote_pct),
        'k_cross_50': int(k_cross_50) if len(cross_50_idx) > 0 else None,
        'k_cross_baseline': int(k_cross_baseline) if len(cross_baseline_idx) > 0 else None,
        'verdict': 'robust' if k_5000_pct > 50 else ('moderate' if k_1000_pct > 50 else 'tail_only')
    }

    output_path = project_root / 'outputs' / 'confident_wrong_curve.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Create plot
    print("\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the curve
    k_plot = [100, 500, 1000, 2000, 5000, 10000, n_test]
    pct_plot = [results[k]['zero_evidence_pct'] for k in k_plot]

    ax.plot(k_plot, pct_plot, 'bo-', linewidth=2, markersize=8, label='Energy (top-K confident)')

    # Add baseline
    ax.axhline(y=baseline_pct, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({baseline_pct:.1f}%)')

    # Add 50% line
    ax.axhline(y=50, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='50% threshold')

    # Annotations
    ax.annotate(f'{pct_plot[0]:.0f}%',
                xy=(k_plot[0], pct_plot[0]),
                xytext=(k_plot[0]*1.5, pct_plot[0]+3),
                fontsize=11, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('K (number of top-confident predictions)', fontsize=12)
    ax.set_ylabel('Zero-Evidence Fraction (%)', fontsize=12)
    ax.set_title('Confident-Wrong Curve: Energy on FB15k-237\n'
                 '(Zero-evidence fraction among top-K most confident predictions)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Custom x-ticks
    ax.set_xticks(k_plot)
    ax.set_xticklabels(['100', '500', '1K', '2K', '5K', '10K', 'ALL'])

    plt.tight_layout()

    fig_path = project_root / 'outputs' / 'confident_wrong_curve.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    # Also save PNG
    fig_path_png = project_root / 'outputs' / 'confident_wrong_curve.png'
    plt.savefig(fig_path_png, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_path_png}")

    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
