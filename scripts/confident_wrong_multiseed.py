#!/usr/bin/env python3
"""
Multi-seed confident-wrong analysis for FB15k-237.

Addresses reviewer concern: "83% confident-wrong has no error bars, uses single seed"

Runs confident-wrong analysis with 5 seeds and reports mean ± std.
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
from datetime import datetime

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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


def train_model(model, triples, device, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
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

    return model


def analyze_confident_wrong(model, test, device, k_values=[100, 500, 1000]):
    model.eval()
    cov = model.coverage.cpu().numpy()

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)

        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -uncertainties

    zero_evidence = []
    for i in range(len(test)):
        h_cov = cov[test[i, 0], test[i, 1]]
        t_cov = cov[test[i, 2], test[i, 1]]
        zero_evidence.append(h_cov == 0 or t_cov == 0)
    zero_evidence = np.array(zero_evidence)

    sorted_indices = np.argsort(confidence)[::-1]

    results = {}
    for k in k_values:
        actual_k = min(k, len(sorted_indices))
        top_k_indices = sorted_indices[:actual_k]
        top_k_zero_evidence = zero_evidence[top_k_indices]
        fraction = top_k_zero_evidence.mean()
        results[k] = fraction

    return results


def run_single_seed(seed, device, train, test, n_ent, n_rel, epochs=30):
    """Run confident-wrong analysis for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {}

    for name, cls in [('Energy', EnergyBased), ('UKGE', UKGE)]:
        model = cls(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=epochs)
        results[name] = analyze_confident_wrong(model, test, device)

    return results


def main():
    device = setup_device()
    print(f"Device: {device}")
    print(f"Running at: {datetime.now().isoformat()}")

    # Load data once
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"\nFB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Compute baseline
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
    print(f"Baseline (random): {baseline_pct:.1f}% zero-evidence")

    # Run 5 seeds
    seeds = [42, 123, 456, 789, 1024]
    all_results = {
        'Energy': {100: [], 500: [], 1000: []},
        'UKGE': {100: [], 500: [], 1000: []},
    }

    print(f"\nRunning 5 seeds: {seeds}")
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        seed_results = run_single_seed(seed, device, train, test, n_ent, n_rel, epochs=30)
        for method in ['Energy', 'UKGE']:
            for k in [100, 500, 1000]:
                all_results[method][k].append(seed_results[method][k] * 100)

    # Compute stats
    print("\n" + "="*70)
    print("MULTI-SEED CONFIDENT-WRONG ANALYSIS: FB15k-237")
    print("="*70)

    output_lines = []
    output_lines.append("Multi-seed Confident-Wrong Analysis: FB15k-237")
    output_lines.append(f"Date: {datetime.now().isoformat()}")
    output_lines.append(f"Seeds: {seeds}")
    output_lines.append(f"Baseline: {baseline_pct:.1f}%")
    output_lines.append("")
    output_lines.append("="*60)

    print(f"\nBaseline: {baseline_pct:.1f}%")
    print(f"\n{'Method':<10} | {'Top-100':>20} | {'Top-500':>20} | {'Top-1000':>20}")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    output_lines.append(f"{'Method':<10} | {'Top-100':>20} | {'Top-500':>20} | {'Top-1000':>20}")
    output_lines.append(f"{'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    robust = True
    for method in ['Energy', 'UKGE']:
        row = f"{method:<10} |"
        out_row = f"{method:<10} |"
        for k in [100, 500, 1000]:
            vals = all_results[method][k]
            mean = np.mean(vals)
            std = np.std(vals)
            row += f" {mean:>6.1f} +/- {std:>4.1f}% |"
            out_row += f" {mean:>6.1f} +/- {std:>4.1f}% |"
            if k == 100 and std >= 5:
                robust = False
        print(row)
        output_lines.append(out_row)

    # Individual seed results
    print("\n" + "="*70)
    print("Individual seed results (Top-100):")
    output_lines.append("")
    output_lines.append("="*60)
    output_lines.append("Individual seed results (Top-100):")

    for method in ['Energy', 'UKGE']:
        vals = all_results[method][100]
        line = f"  {method}: {vals}"
        print(line)
        output_lines.append(line)

    # Summary
    energy_mean = np.mean(all_results['Energy'][100])
    energy_std = np.std(all_results['Energy'][100])

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nEnergy Top-100: {energy_mean:.1f}% +/- {energy_std:.1f}%")
    print(f"UKGE Top-100:   {np.mean(all_results['UKGE'][100]):.1f}% +/- {np.std(all_results['UKGE'][100]):.1f}%")

    output_lines.append("")
    output_lines.append("="*60)
    output_lines.append("SUMMARY")
    output_lines.append("="*60)
    output_lines.append(f"Energy Top-100: {energy_mean:.1f}% +/- {energy_std:.1f}%")
    output_lines.append(f"UKGE Top-100:   {np.mean(all_results['UKGE'][100]):.1f}% +/- {np.std(all_results['UKGE'][100]):.1f}%")

    if energy_std < 5:
        print(f"\nRobustness check PASSED: std={energy_std:.1f}% < 5%")
        output_lines.append(f"\nRobustness check PASSED: std={energy_std:.1f}% < 5%")
    else:
        print(f"\nRobustness check FAILED: std={energy_std:.1f}% >= 5%")
        output_lines.append(f"\nRobustness check FAILED: std={energy_std:.1f}% >= 5%")

    # Save output
    output_path = project_root / "outputs" / "confident_wrong_multiseed.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nResults saved to: {output_path}")

    return energy_mean, energy_std


if __name__ == "__main__":
    main()
