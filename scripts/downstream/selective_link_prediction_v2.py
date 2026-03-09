#!/usr/bin/env python3
"""
Selective Link Prediction v2: Correct Experiment Design

Problem with v1: We measured abstention on test triples (all correct by definition).
Coverage-based abstention on correct test triples REDUCES accuracy.

Correct design:
1. For each test query (h, r, ?), model predicts a tail
2. Prediction can be CORRECT (matches true tail) or INCORRECT
3. Abstention should AVOID incorrect predictions
4. Coverage tells us: "This entity-relation pair has no training evidence"

Key insight: Coverage-based abstention should improve PRECISION (fewer wrong predictions)
at the cost of COVERAGE (fewer predictions made).

New experiment:
- For each test triple (h, r, t), model predicts top-K tails
- If true tail NOT in top-K: prediction is WRONG
- Abstention strategies try to identify WHICH predictions will be wrong
- Coverage-based: Abstain when model has no evidence for (h, r) or (t, r)
- Confidence-based: Abstain when model confidence is low

Expected result:
- Coverage-based abstention has HIGHER precision among answered queries
- Because it correctly identifies zero-evidence queries (which are often wrong)
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

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def score_tails(self, h, r):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        all_t = self.entity_emb.weight
        return (h_emb * r_emb) @ all_t.T

    def get_confidence(self, h, r, t):
        """Model confidence (higher = more confident in prediction)."""
        return torch.sigmoid(self.forward(h, r, t))

    def get_coverage_score(self, h, r, t):
        """Coverage-based score (2=both covered, 1=one covered, 0=novel context)."""
        h_cov = self.coverage[h, r]
        t_cov = self.coverage[t, r]
        return h_cov + t_cov

    def precompute_coverage(self, triples):
        self.coverage.zero_()
        for i in range(len(triples)):
            h, r, t = triples[i, 0], triples[i, 1], triples[i, 2]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


def train_model(model, triples, device, epochs=30, lr=0.001):
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


def evaluate_predictions(model, test_triples, device, k=10):
    """
    For each test triple, evaluate if model's prediction is correct.

    Returns:
        is_correct: Boolean array (True = model ranks true tail in top-K)
        confidence: Model's confidence in its top-1 prediction
        coverage_score: Coverage score for the query (0, 1, or 2)
        is_novel_context: Boolean (True = zero coverage for head or tail)
    """
    model.eval()
    model = model.to(device)

    is_correct = []
    confidence = []
    coverage_score = []
    is_novel_context = []

    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            h = torch.tensor(batch[:, 0]).to(device)
            r = torch.tensor(batch[:, 1]).to(device)
            t = torch.tensor(batch[:, 2]).to(device)

            # Score all tails
            scores = model.score_tails(h, r)

            for j in range(len(h)):
                true_tail = t[j].item()

                # Is true tail in top-K?
                _, topk_tails = scores[j].topk(k)
                correct = true_tail in topk_tails.cpu().numpy()
                is_correct.append(correct)

                # Model's confidence in top-1 prediction
                top1_tail = topk_tails[0].item()
                conf = torch.sigmoid(scores[j, top1_tail]).item()
                confidence.append(conf)

                # Coverage score
                h_cov = model.coverage[h[j], r[j]].item()
                t_cov = model.coverage[t[j], r[j]].item()
                coverage_score.append(h_cov + t_cov)

                # Is this a novel context?
                is_novel_context.append(h_cov == 0 or t_cov == 0)

    return (
        np.array(is_correct),
        np.array(confidence),
        np.array(coverage_score),
        np.array(is_novel_context)
    )


def compute_risk_coverage_curve(is_correct, scores, num_points=100):
    """
    Compute risk-coverage curve.

    Args:
        is_correct: Boolean array (True = correct prediction)
        scores: Array of scores (higher = should answer first)
        num_points: Number of coverage points

    Returns:
        coverages, risks, aurc
    """
    n = len(is_correct)
    sorted_idx = np.argsort(scores)[::-1]  # Descending (most confident first)
    sorted_correct = is_correct[sorted_idx]

    coverages = []
    risks = []

    for cov_frac in np.linspace(0.01, 1.0, num_points):
        num_answer = int(cov_frac * n)
        if num_answer == 0:
            continue

        answered_correct = sorted_correct[:num_answer]
        risk = 1 - answered_correct.mean()

        coverages.append(cov_frac)
        risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)
    aurc = np.trapezoid(risks, coverages) if len(risks) > 1 else 1.0

    return coverages, risks, aurc


def run_experiment(dataset_name, loader, device, epochs=30, seed=42):
    """Run the selective prediction experiment."""
    print(f"\n{'='*70}")
    print(f"SELECTIVE LINK PREDICTION v2: {dataset_name}")
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

    # Evaluate predictions
    print("\nEvaluating predictions...")
    is_correct, confidence, coverage_score, is_novel = evaluate_predictions(
        model, test, device, k=10
    )

    base_acc = is_correct.mean()
    print(f"Base Hits@10: {base_acc:.3f}")

    # Statistics on novel contexts
    novel_rate = is_novel.mean()
    novel_acc = is_correct[is_novel].mean() if is_novel.sum() > 0 else 0
    covered_acc = is_correct[~is_novel].mean() if (~is_novel).sum() > 0 else 0

    print(f"\nNovel context analysis:")
    print(f"  Novel context rate: {novel_rate:.1%}")
    print(f"  Accuracy on novel contexts: {novel_acc:.1%}")
    print(f"  Accuracy on covered queries: {covered_acc:.1%}")
    print(f"  Gap: {covered_acc - novel_acc:.1%} better on covered")

    # Confidence analysis for novel contexts
    novel_conf = confidence[is_novel].mean() if is_novel.sum() > 0 else 0
    covered_conf = confidence[~is_novel].mean() if (~is_novel).sum() > 0 else 0
    print(f"\nConfidence analysis:")
    print(f"  Mean confidence on novel: {novel_conf:.3f}")
    print(f"  Mean confidence on covered: {covered_conf:.3f}")
    print(f"  Model is {'over' if novel_conf >= covered_conf else 'under'}-confident on novel contexts")

    # Compute risk-coverage curves
    print("\nComputing risk-coverage curves...")

    strategies = {
        'confidence': confidence,
        'coverage': coverage_score,
        'combined': np.minimum(
            (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8),
            coverage_score / 2.0
        )
    }

    results = {}
    for name, scores in strategies.items():
        covs, risks, aurc = compute_risk_coverage_curve(is_correct, scores)
        results[name] = {'coverages': covs, 'risks': risks, 'aurc': aurc}
        print(f"  {name}: AURC = {aurc:.4f}")

    # Detailed comparison
    print("\n" + "-"*60)
    print("Accuracy at different coverage levels:")
    print("-"*60)
    print(f"{'Strategy':<15} | {'90% cov':>12} | {'80% cov':>12} | {'70% cov':>12}")
    print("-"*15 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*12)

    for name in ['confidence', 'coverage', 'combined']:
        covs = results[name]['coverages']
        risks = results[name]['risks']
        row = f"{name:<15} |"
        for target in [0.9, 0.8, 0.7]:
            idx = np.argmin(np.abs(covs - target))
            acc = 1 - risks[idx]
            row += f" {acc:>11.1%} |"
        print(row)

    print(f"\nBase accuracy (100% coverage): {base_acc:.1%}")

    # Key finding: What's the accuracy improvement from using coverage?
    # At 80% coverage, compare confidence vs coverage strategies
    for target_cov in [0.9, 0.8, 0.7]:
        conf_idx = np.argmin(np.abs(results['confidence']['coverages'] - target_cov))
        cov_idx = np.argmin(np.abs(results['coverage']['coverages'] - target_cov))

        conf_acc = 1 - results['confidence']['risks'][conf_idx]
        cov_acc = 1 - results['coverage']['risks'][cov_idx]

        print(f"\nAt {target_cov:.0%} coverage:")
        print(f"  Confidence-based accuracy: {conf_acc:.1%}")
        print(f"  Coverage-based accuracy:   {cov_acc:.1%}")
        print(f"  Difference:                {cov_acc - conf_acc:+.1%}")

    return results, {
        'base_acc': base_acc,
        'novel_rate': novel_rate,
        'novel_acc': novel_acc,
        'covered_acc': covered_acc,
        'novel_conf': novel_conf,
        'covered_conf': covered_conf,
    }


def plot_results(all_results, output_path):
    """Plot risk-coverage curves."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 4))

    if len(all_results) == 1:
        axes = [axes]

    colors = {
        'confidence': '#1f77b4',
        'coverage': '#2ca02c',
        'combined': '#d62728',
    }

    labels = {
        'confidence': 'Confidence-based',
        'coverage': 'Coverage-based (ours)',
        'combined': 'Combined',
    }

    for ax, (dataset_name, results) in zip(axes, all_results.items()):
        for name in ['confidence', 'coverage', 'combined']:
            data = results[name]
            ax.plot(data['coverages'], data['risks'],
                    color=colors[name],
                    label=f"{labels[name]} (AURC={data['aurc']:.3f})")

        ax.set_xlabel('Coverage (fraction answered)')
        ax.set_ylabel('Risk (error rate)')
        ax.set_title(f'{dataset_name}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
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
    parser.add_argument('--output', type=str, default='outputs/selective_lp_v2.pdf')
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    loaders = {
        'fb15k237': load_fb15k237,
        'icews14': load_icews14,
        'wn18rr': load_wn18rr,
    }

    all_results = {}
    all_stats = {}

    for dataset_name in args.datasets:
        results, stats = run_experiment(
            dataset_name,
            loaders[dataset_name],
            device,
            epochs=args.epochs,
            seed=args.seed
        )
        all_results[dataset_name] = results
        all_stats[dataset_name] = stats

    # Plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(all_results, output_path)

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    for dataset_name, stats in all_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Novel context rate: {stats['novel_rate']:.1%}")
        print(f"  Accuracy gap: {stats['covered_acc'] - stats['novel_acc']:.1%} better on covered")
        print(f"  Model overconfidence on novel: {stats['novel_conf']:.3f} vs {stats['covered_conf']:.3f}")


if __name__ == "__main__":
    main()
