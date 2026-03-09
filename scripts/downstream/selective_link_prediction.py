#!/usr/bin/env python3
"""
Selective Link Prediction: Coverage-Based Abstention vs Confidence-Based

This experiment demonstrates that coverage tracking improves downstream task
performance, not just OOD detection AUROC.

Key insight from paper: 83% of top-confident predictions have zero training evidence.
Implication: Confidence-based abstention abstains on the WRONG queries.

Experiment:
1. Train KG model (Energy, UKGE, or any baseline)
2. Evaluate link prediction with different abstention strategies:
   - Confidence-based: Abstain when model confidence < threshold
   - Coverage-based: Abstain when coverage(e, r) = 0
   - Combined: Abstain when either condition holds
3. Measure: Risk (error rate) vs Coverage (fraction answered)

Expected result: Coverage-based achieves lower risk at all coverage levels.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

from src.data.loaders import load_fb15k237, load_icews14, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model (simplified DistMult)
# ============================================================

class DistMultWithUncertainty(nn.Module):
    """DistMult with uncertainty estimation."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage matrix (precomputed from training)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        """Score a triple."""
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def score_tails(self, h, r):
        """Score all possible tails for (h, r, ?)."""
        h_emb = self.entity_emb(h)  # (batch, dim)
        r_emb = self.relation_emb(r)  # (batch, dim)
        all_t = self.entity_emb.weight  # (num_entities, dim)
        # (batch, dim) * (dim, num_entities) -> (batch, num_entities)
        return (h_emb * r_emb) @ all_t.T

    def get_energy_uncertainty(self, h, r, t):
        """Energy-based uncertainty (negative score)."""
        return -self.forward(h, r, t)

    def get_coverage_score(self, h, r, t):
        """Coverage-based score (higher = more evidence)."""
        h_cov = self.coverage[h, r]
        t_cov = self.coverage[t, r]
        return h_cov + t_cov  # 2 = both covered, 1 = one covered, 0 = novel context

    def precompute_coverage(self, triples):
        """Build coverage matrix from training triples."""
        self.coverage.zero_()
        for i in range(len(triples)):
            h, r, t = triples[i, 0], triples[i, 1], triples[i, 2]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


def train_model(model, triples, device, epochs=30, lr=0.001):
    """Train the model with BCE loss."""
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
            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return model


# ============================================================
# Selective Prediction Evaluation
# ============================================================

def evaluate_link_prediction(model, test_triples, device, k=10):
    """
    Evaluate Hits@K on test triples.

    Returns:
        np.array: Binary array of whether each test triple is in top-K
    """
    model.eval()
    model = model.to(device)

    hits = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            h = torch.tensor(batch[:, 0]).to(device)
            r = torch.tensor(batch[:, 1]).to(device)
            t = torch.tensor(batch[:, 2]).to(device)

            # Score all tails
            scores = model.score_tails(h, r)  # (batch, num_entities)

            # Get ranks
            for j in range(len(h)):
                true_tail = t[j].item()
                true_score = scores[j, true_tail].item()
                rank = (scores[j] > true_score).sum().item() + 1
                hits.append(1 if rank <= k else 0)

    return np.array(hits)


def compute_abstention_scores(model, test_triples, device):
    """
    Compute different abstention scores for each test triple.

    Returns:
        dict: {strategy_name: scores (higher = more confident/should answer)}
    """
    model.eval()
    model = model.to(device)

    scores = {
        'confidence': [],  # Higher = more confident (should answer)
        'coverage': [],    # Higher = more evidence (should answer)
        'energy': [],      # Higher = more confident (should answer)
    }

    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            h = torch.tensor(batch[:, 0]).to(device)
            r = torch.tensor(batch[:, 1]).to(device)
            t = torch.tensor(batch[:, 2]).to(device)

            # Confidence = model score (sigmoid)
            raw_scores = model(h, r, t)
            confidence = torch.sigmoid(raw_scores)
            scores['confidence'].extend(confidence.cpu().numpy())

            # Energy (negative uncertainty)
            energy = raw_scores  # Higher score = lower energy = more confident
            scores['energy'].extend(energy.cpu().numpy())

            # Coverage (from precomputed matrix)
            cov = model.get_coverage_score(h, r, t)
            scores['coverage'].extend(cov.cpu().numpy())

    return {k: np.array(v) for k, v in scores.items()}


def compute_risk_coverage_curve(hits, abstention_scores, num_points=100):
    """
    Compute risk-coverage curve for an abstention strategy.

    Args:
        hits: Binary array (1 = correct prediction)
        abstention_scores: Array of scores (higher = more confident, should answer)
        num_points: Number of coverage levels to compute

    Returns:
        coverages: Array of coverage fractions
        risks: Array of error rates at each coverage level
    """
    n = len(hits)

    # Sort by abstention score (descending - most confident first)
    sorted_idx = np.argsort(abstention_scores)[::-1]
    sorted_hits = hits[sorted_idx]

    coverages = []
    risks = []

    for cov_frac in np.linspace(0.01, 1.0, num_points):
        num_answer = int(cov_frac * n)
        if num_answer == 0:
            continue

        # Answer the top num_answer most confident queries
        answered_hits = sorted_hits[:num_answer]
        risk = 1 - answered_hits.mean()  # Error rate

        coverages.append(cov_frac)
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def compute_aurc(coverages, risks):
    """Area Under Risk-Coverage Curve (lower is better)."""
    return np.trapz(risks, coverages)


def run_selective_prediction_experiment(dataset_name, loader, device, epochs=30, seed=42):
    """Run full selective prediction experiment on a dataset."""
    print(f"\n{'='*70}")
    print(f"SELECTIVE LINK PREDICTION: {dataset_name}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Train model
    print("\nTraining model...")
    model = DistMultWithUncertainty(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=epochs)

    # Evaluate base link prediction
    print("\nEvaluating link prediction...")
    hits = evaluate_link_prediction(model, test, device, k=10)
    base_hits_at_10 = hits.mean()
    print(f"Base Hits@10: {base_hits_at_10:.3f}")

    # Compute abstention scores
    print("\nComputing abstention scores...")
    abstention_scores = compute_abstention_scores(model, test, device)

    # Add combined strategy: min of coverage and confidence
    # Normalize to [0, 1] first
    conf_norm = (abstention_scores['confidence'] - abstention_scores['confidence'].min())
    conf_norm = conf_norm / (conf_norm.max() + 1e-8)
    cov_norm = abstention_scores['coverage'] / 2.0  # Already in [0, 2], normalize to [0, 1]

    abstention_scores['combined'] = np.minimum(conf_norm, cov_norm)

    # Compute risk-coverage curves
    print("\nComputing risk-coverage curves...")
    results = {}
    for strategy_name, scores in abstention_scores.items():
        coverages, risks = compute_risk_coverage_curve(hits, scores)
        aurc = compute_aurc(coverages, risks)
        results[strategy_name] = {
            'coverages': coverages,
            'risks': risks,
            'aurc': aurc
        }
        print(f"  {strategy_name}: AURC = {aurc:.4f}")

    # Compute statistics at specific coverage levels
    print("\n" + "-"*50)
    print("Risk (error rate) at different coverage levels:")
    print("-"*50)
    print(f"{'Strategy':<15} | {'90% cov':>10} | {'80% cov':>10} | {'70% cov':>10} | {'AURC':>10}")
    print("-"*15 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10)

    for strategy_name in ['confidence', 'energy', 'coverage', 'combined']:
        covs = results[strategy_name]['coverages']
        risks = results[strategy_name]['risks']

        row = f"{strategy_name:<15} |"
        for target_cov in [0.9, 0.8, 0.7]:
            idx = np.argmin(np.abs(covs - target_cov))
            risk = risks[idx]
            row += f" {risk:>9.3f} |"
        row += f" {results[strategy_name]['aurc']:>9.4f} |"
        print(row)

    # Compute improvement over baseline
    print("\n" + "-"*50)
    print("Accuracy improvement (Hits@K) at different coverage levels:")
    print("-"*50)
    print(f"{'Strategy':<15} | {'90% cov':>10} | {'80% cov':>10} | {'70% cov':>10}")
    print("-"*15 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10)

    for strategy_name in ['confidence', 'energy', 'coverage', 'combined']:
        covs = results[strategy_name]['coverages']
        risks = results[strategy_name]['risks']

        row = f"{strategy_name:<15} |"
        for target_cov in [0.9, 0.8, 0.7]:
            idx = np.argmin(np.abs(covs - target_cov))
            acc = 1 - risks[idx]  # Convert risk to accuracy
            row += f" {acc:>9.1%} |"
        print(row)

    print(f"\nBase accuracy (100% coverage): {base_hits_at_10:.1%}")

    return results, hits, abstention_scores, base_hits_at_10


def plot_risk_coverage_curves(all_results, output_path):
    """Plot risk-coverage curves for all datasets and strategies."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 4))

    if len(all_results) == 1:
        axes = [axes]

    colors = {
        'confidence': '#1f77b4',  # Blue
        'energy': '#ff7f0e',      # Orange
        'coverage': '#2ca02c',    # Green
        'combined': '#d62728',    # Red
    }

    labels = {
        'confidence': 'Confidence-based',
        'energy': 'Energy-based',
        'coverage': 'Coverage-based (ours)',
        'combined': 'Combined',
    }

    for ax, (dataset_name, results) in zip(axes, all_results.items()):
        for strategy_name in ['confidence', 'energy', 'coverage', 'combined']:
            covs = results[strategy_name]['coverages']
            risks = results[strategy_name]['risks']
            aurc = results[strategy_name]['aurc']
            ax.plot(covs, risks, color=colors[strategy_name],
                    label=f"{labels[strategy_name]} (AURC={aurc:.3f})")

        ax.set_xlabel('Coverage (fraction answered)')
        ax.set_ylabel('Risk (error rate)')
        ax.set_title(f'{dataset_name}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['fb15k237'],
                        choices=['fb15k237', 'icews14', 'wn18rr'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/selective_link_prediction.pdf')
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    loaders = {
        'fb15k237': load_fb15k237,
        'icews14': load_icews14,
        'wn18rr': load_wn18rr,
    }

    all_results = {}

    for dataset_name in args.datasets:
        results, hits, scores, base_acc = run_selective_prediction_experiment(
            dataset_name,
            loaders[dataset_name],
            device,
            epochs=args.epochs,
            seed=args.seed
        )
        all_results[dataset_name] = results

    # Plot results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_risk_coverage_curves(all_results, output_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: COVERAGE-BASED ABSTENTION IMPROVEMENT")
    print("="*70)

    for dataset_name, results in all_results.items():
        conf_aurc = results['confidence']['aurc']
        cov_aurc = results['coverage']['aurc']
        improvement = (conf_aurc - cov_aurc) / conf_aurc * 100

        print(f"\n{dataset_name}:")
        print(f"  Confidence AURC: {conf_aurc:.4f}")
        print(f"  Coverage AURC:   {cov_aurc:.4f}")
        print(f"  Improvement:     {improvement:.1f}% lower risk")


if __name__ == "__main__":
    main()
