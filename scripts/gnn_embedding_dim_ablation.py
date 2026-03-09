#!/usr/bin/env python3
"""
GNN Embedding Dimension Ablation: Test |R| <= O(embedding_dim) Hypothesis.

Hypothesis:
- GNNs can detect novel-context OOD when |R| <= O(embedding_dim)
- WN18RR (11 relations) works with dim=100 because 11 << 100
- If we reduce dim to ~10, WN18RR should fail (11 relations ~= 10 dims)
- If we increase dim for YAGO3-10, it might improve (37 relations)

Experiments:
1. WN18RR with dims [10, 25, 50, 100, 200] - expect failure at dim=10
2. YAGO3-10 with dims [100, 200, 400] - expect improvement with higher dims

Usage:
    python scripts/gnn_embedding_dim_ablation.py --dataset wn18rr --device mps
    python scripts/gnn_embedding_dim_ablation.py --dataset yago3-10 --device mps
    python scripts/gnn_embedding_dim_ablation.py --all --device mps
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.loaders import load_wn18rr, load_yago310


def compute_coverage_matrix(triples, num_entities, num_relations):
    """Compute binary coverage matrix."""
    coverage = torch.zeros(num_entities, num_relations, dtype=torch.bool)
    for h, r, t in tqdm(triples, desc="Building coverage", leave=False):
        coverage[h, r] = True
        coverage[t, r] = True
    return coverage


def compute_entity_frequency(triples, num_entities):
    """Compute entity frequency."""
    freq = torch.zeros(num_entities)
    for h, r, t in triples:
        freq[h] += 1
        freq[t] += 1
    return freq


def compute_gamma_ratio(triples, num_entities, num_relations):
    """Compute gamma = |R| / avg|N(e)|."""
    neighbors = defaultdict(set)
    for h, r, t in triples:
        neighbors[h].add(t)
        neighbors[t].add(h)

    neighbor_counts = [len(neighbors[e]) for e in range(num_entities) if e in neighbors]
    avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 1.0
    gamma = num_relations / avg_neighbors

    return gamma, avg_neighbors


def categorize_test_triples(test_triples, coverage, entity_freq, tau_percentile=25):
    """
    Categorize test triples into:
    - Emerging: At least one entity has low frequency
    - Novel-context: Both entities seen, but at least one hasn't seen this relation
    - In-distribution: Both entities covered for this relation
    """
    tau = np.percentile(entity_freq.numpy(), tau_percentile)

    emerging = []
    novel_context = []
    in_distribution = []

    for h, r, t in test_triples:
        min_freq = min(entity_freq[h].item(), entity_freq[t].item())
        h_covered = coverage[h, r].item()
        t_covered = coverage[t, r].item()

        if min_freq <= tau:
            emerging.append((h, r, t))
        elif not h_covered or not t_covered:
            novel_context.append((h, r, t))
        else:
            in_distribution.append((h, r, t))

    return emerging, novel_context, in_distribution


class SimpleGNN(nn.Module):
    """
    Simple 2-layer MLP with energy-based OOD scoring.

    Key insight: This model learns entity embeddings of size `embedding_dim`.
    The hypothesis is that if |R| > embedding_dim, the model cannot encode
    enough relation-specific information to detect novel (e,r) contexts.
    """

    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2

        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        # 2-layer MLP for scoring
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits.squeeze(-1)

    def energy_score(self, h, r, t):
        """GNNSafe-style energy score: -logit (higher = more OOD)."""
        logits = self.forward(h, r, t)
        return -logits


def train_model(model, train_triples, num_entities, epochs=50, batch_size=1024,
                lr=1e-3, device='cpu', num_negatives=5, verbose=True):
    """Train model with BCE loss and negative sampling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_h = torch.tensor([t[0] for t in train_triples], dtype=torch.long)
    train_r = torch.tensor([t[1] for t in train_triples], dtype=torch.long)
    train_t = torch.tensor([t[2] for t in train_triples], dtype=torch.long)

    n_batches = (len(train_triples) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Shuffle
        perm = torch.randperm(len(train_triples))
        train_h = train_h[perm]
        train_r = train_r[perm]
        train_t = train_t[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(train_triples))

            batch_h = train_h[start:end].to(device)
            batch_r = train_r[start:end].to(device)
            batch_t = train_t[start:end].to(device)

            # Positive scores
            pos_scores = model(batch_h, batch_r, batch_t)

            # Negative samples (corrupt tails)
            neg_scores_list = []
            for _ in range(num_negatives):
                neg_t = torch.randint(0, num_entities, (end - start,), device=device)
                neg_scores_list.append(model(batch_h, batch_r, neg_t))

            # BCE loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores))
            neg_loss = sum(
                F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
                for ns in neg_scores_list
            ) / num_negatives

            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_ood_stratified(model, test_triples_by_category, train_triples,
                            num_entities, device='cpu', sample_size=5000):
    """Evaluate OOD detection using energy scores, stratified by category."""
    model.eval()
    results = {}

    # Sample ID triples from training
    id_sample_size = min(sample_size, len(train_triples))
    id_indices = np.random.choice(len(train_triples), id_sample_size, replace=False)
    id_triples = [train_triples[i] for i in id_indices]

    with torch.no_grad():
        id_h = torch.tensor([t[0] for t in id_triples], device=device)
        id_r = torch.tensor([t[1] for t in id_triples], device=device)
        id_t = torch.tensor([t[2] for t in id_triples], device=device)
        id_energy = model.energy_score(id_h, id_r, id_t).cpu().numpy()

    for category, triples in test_triples_by_category.items():
        if len(triples) < 20:
            continue

        # Sample OOD triples
        ood_sample_size = min(sample_size, len(triples))
        ood_indices = np.random.choice(len(triples), ood_sample_size, replace=False)
        ood_triples = [triples[i] for i in ood_indices]

        with torch.no_grad():
            ood_h = torch.tensor([t[0] for t in ood_triples], device=device)
            ood_r = torch.tensor([t[1] for t in ood_triples], device=device)
            ood_t = torch.tensor([t[2] for t in ood_triples], device=device)
            ood_energy = model.energy_score(ood_h, ood_r, ood_t).cpu().numpy()

        # Balance samples for AUROC
        n_samples = min(len(ood_energy), len(id_energy))
        ood_energy_balanced = ood_energy[:n_samples]
        id_energy_balanced = id_energy[:n_samples]

        # AUROC: OOD should have higher energy
        labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
        scores = np.concatenate([ood_energy_balanced, id_energy_balanced])

        try:
            auroc = roc_auc_score(labels, scores)
            results[category] = auroc
        except Exception:
            pass

    return results


def run_single_experiment(dataset_name, embedding_dim, train_triples, test_triples,
                          coverage, entity_freq, num_entities, num_relations,
                          epochs=50, device='cpu', seed=42, verbose=True):
    """Run a single experiment with given embedding dimension."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Categorize test triples
    emerging, novel_ctx, id_triples = categorize_test_triples(
        test_triples, coverage, entity_freq
    )

    if verbose:
        print(f"\n  Embedding dim: {embedding_dim}")
        print(f"    |R|/dim ratio: {num_relations/embedding_dim:.2f}")

    # Create and train model
    model = SimpleGNN(num_entities, num_relations, embedding_dim)
    model = train_model(
        model, train_triples, num_entities,
        epochs=epochs, device=device, verbose=verbose
    )

    # Evaluate
    test_by_category = {
        'emerging': emerging,
        'novel_context': novel_ctx,
        'in_distribution': id_triples,
    }

    results = evaluate_ood_stratified(
        model, test_by_category, train_triples,
        num_entities, device
    )

    if verbose:
        for cat, auroc in results.items():
            print(f"    {cat} AUROC: {auroc:.3f}")

    return results


def run_wn18rr_ablation(dims, epochs, device, seed):
    """Run WN18RR ablation with varying embedding dimensions."""
    print("\n" + "="*70)
    print("WN18RR Embedding Dimension Ablation")
    print("="*70)
    print(f"Hypothesis: dim=10 should fail (|R|=11), dim>=25 should work")
    print(f"Dims to test: {dims}")

    # Load data
    print("\nLoading WN18RR...")
    train_ds, _, test_ds = load_wn18rr()

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    train_triples = train_ds.triples.tolist()
    test_triples = test_ds.triples.tolist()

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples):,}")
    print(f"  Test triples: {len(test_triples):,}")

    # Compute statistics
    gamma, avg_neighbors = compute_gamma_ratio(train_triples, num_entities, num_relations)
    print(f"  Gamma (|R|/avg|N(e)|): {gamma:.2f}")

    # Build coverage matrix
    print("\nBuilding coverage matrix...")
    coverage = compute_coverage_matrix(train_triples, num_entities, num_relations)
    entity_freq = compute_entity_frequency(train_triples, num_entities)

    # Run experiments
    all_results = {}
    for dim in dims:
        results = run_single_experiment(
            'wn18rr', dim, train_triples, test_triples,
            coverage, entity_freq, num_entities, num_relations,
            epochs=epochs, device=device, seed=seed
        )
        all_results[dim] = results

    return {
        'dataset': 'WN18RR',
        'num_relations': num_relations,
        'gamma': gamma,
        'results_by_dim': all_results
    }


def run_yago310_ablation(dims, epochs, device, seed, max_train=100000):
    """Run YAGO3-10 ablation with varying embedding dimensions."""
    print("\n" + "="*70)
    print("YAGO3-10 Embedding Dimension Ablation")
    print("="*70)
    print(f"Hypothesis: Higher dims may help (|R|=37)")
    print(f"Dims to test: {dims}")

    # Load data
    print("\nLoading YAGO3-10...")
    train_ds, _, test_ds = load_yago310()

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    train_triples = train_ds.triples.tolist()
    test_triples = test_ds.triples.tolist()

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples):,}")
    print(f"  Test triples: {len(test_triples):,}")

    # Subsample training if too large
    if len(train_triples) > max_train:
        print(f"\nSubsampling training: {len(train_triples):,} -> {max_train:,}")
        indices = np.random.choice(len(train_triples), max_train, replace=False)
        train_triples_sample = [train_triples[i] for i in indices]
    else:
        train_triples_sample = train_triples

    # Compute statistics using FULL training data
    gamma, avg_neighbors = compute_gamma_ratio(train_triples, num_entities, num_relations)
    print(f"  Gamma (|R|/avg|N(e)|): {gamma:.2f}")

    # Build coverage matrix using FULL training data
    print("\nBuilding coverage matrix...")
    coverage = compute_coverage_matrix(train_triples, num_entities, num_relations)
    entity_freq = compute_entity_frequency(train_triples, num_entities)

    # Run experiments
    all_results = {}
    for dim in dims:
        results = run_single_experiment(
            'yago3-10', dim, train_triples_sample, test_triples,
            coverage, entity_freq, num_entities, num_relations,
            epochs=epochs, device=device, seed=seed
        )
        all_results[dim] = results

    return {
        'dataset': 'YAGO3-10',
        'num_relations': num_relations,
        'gamma': gamma,
        'results_by_dim': all_results
    }


def print_summary(wn18rr_results, yago_results):
    """Print summary table of results."""
    print("\n" + "="*70)
    print("SUMMARY: Embedding Dimension vs Novel-Context AUROC")
    print("="*70)

    print("\nWN18RR (|R|=11):")
    print("-" * 50)
    print(f"{'Dim':<10} {'|R|/dim':<10} {'Novel-Ctx AUROC':<15} {'Prediction':<15}")
    print("-" * 50)

    for dim, results in wn18rr_results['results_by_dim'].items():
        ratio = 11 / dim
        auroc = results.get('novel_context', float('nan'))
        prediction = "FAIL" if dim <= 11 else "WORK"
        actual = "FAIL" if auroc < 0.55 else "WORK" if auroc >= 0.65 else "?"
        match = "OK" if prediction == actual else "UNEXPECTED"
        print(f"{dim:<10} {ratio:<10.2f} {auroc:<15.3f} {prediction} -> {actual} [{match}]")

    if yago_results:
        print("\nYAGO3-10 (|R|=37):")
        print("-" * 50)
        print(f"{'Dim':<10} {'|R|/dim':<10} {'Novel-Ctx AUROC':<15} {'Expected':<15}")
        print("-" * 50)

        for dim, results in yago_results['results_by_dim'].items():
            ratio = 37 / dim
            auroc = results.get('novel_context', float('nan'))
            expected = "Higher dim -> better?" if dim > 100 else "baseline"
            print(f"{dim:<10} {ratio:<10.2f} {auroc:<15.3f} {expected}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)

    # Check WN18RR hypothesis
    wn_dims = sorted(wn18rr_results['results_by_dim'].keys())
    wn_novel = [wn18rr_results['results_by_dim'][d].get('novel_context', 0) for d in wn_dims]

    # Find dim where AUROC transitions
    transition_dim = None
    for i, (dim, auroc) in enumerate(zip(wn_dims, wn_novel)):
        if auroc >= 0.60:
            transition_dim = dim
            break

    if transition_dim:
        print(f"\nWN18RR transition point: dim >= {transition_dim} for AUROC >= 0.60")
        print(f"  |R|=11, so hypothesis predicts transition around dim ~11")
        if 10 <= transition_dim <= 50:
            print(f"  HYPOTHESIS SUPPORTED: transition at {transition_dim} (within 5x of |R|)")
        else:
            print(f"  HYPOTHESIS UNCLEAR: transition at {transition_dim} (far from |R|=11)")
    else:
        print("\nWN18RR: No clear transition found")
        print("  Check if all dims fail or all succeed")


def main():
    parser = argparse.ArgumentParser(description="GNN Embedding Dimension Ablation")
    parser.add_argument("--dataset", type=str, default=None, choices=["wn18rr", "yago3-10"])
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=100000,
                       help="Max training triples for YAGO3-10")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print("GNN Embedding Dimension Ablation")
    print(f"{'#'*70}")
    print(f"\nHypothesis: |R| <= O(embedding_dim) determines GNN success on novel-context")
    print(f"Config: epochs={args.epochs}, device={args.device}, seed={args.seed}")

    wn18rr_results = None
    yago_results = None

    # WN18RR experiments
    if args.all or args.dataset == "wn18rr" or args.dataset is None:
        wn18rr_dims = [10, 25, 50, 100, 200]
        wn18rr_results = run_wn18rr_ablation(
            wn18rr_dims, args.epochs, args.device, args.seed
        )

    # YAGO3-10 experiments
    if args.all or args.dataset == "yago3-10":
        yago_dims = [100, 200, 400]
        yago_results = run_yago310_ablation(
            yago_dims, args.epochs, args.device, args.seed, args.max_train
        )

    # Print summary
    if wn18rr_results:
        print_summary(wn18rr_results, yago_results)

    # Save results
    output = {
        'hypothesis': '|R| <= O(embedding_dim) determines GNN success on novel-context',
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
    }

    if wn18rr_results:
        output['wn18rr'] = wn18rr_results
    if yago_results:
        output['yago3-10'] = yago_results

    output_path = Path(__file__).parent.parent / "outputs" / "gnn_embedding_dim_ablation.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
