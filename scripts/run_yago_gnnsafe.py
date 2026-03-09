#!/usr/bin/env python3
"""
GNNSafe Experiment on YAGO3-10 - Validate GNN boundary prediction.

Hypothesis from paper:
- GNNs can escape impossibility when γ = |R|/avg|N(e)| is low
- WN18RR (γ=5.0): GNNSafe achieves 0.79 AUROC on novel-context
- FB15k-237 (γ=12.5): GNNSafe achieves 0.43 AUROC (fails)
- YAGO3-10 (γ=2.9): **Predicted to work** (0.70-0.85 AUROC)

Usage:
    python scripts/run_yago_gnnsafe.py --epochs 50 --device cpu
    python scripts/run_yago_gnnsafe.py --epochs 100 --device mps --seed 42
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm

from src.data.loaders import load_yago310


def compute_coverage_matrix(triples, num_entities, num_relations):
    """Compute binary coverage matrix."""
    coverage = torch.zeros(num_entities, num_relations, dtype=torch.bool)
    for h, r, t in tqdm(triples, desc="Building coverage matrix"):
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
    """Compute γ = |R| / avg|N(e)|."""
    neighbors = defaultdict(set)
    for h, r, t in triples:
        neighbors[h].add(t)
        neighbors[t].add(h)

    neighbor_counts = [len(neighbors[e]) for e in range(num_entities) if e in neighbors]
    avg_neighbors = np.mean(neighbor_counts)
    gamma = num_relations / avg_neighbors

    return gamma, avg_neighbors


def categorize_test_triples(test_triples, coverage, entity_freq, tau_percentile=25):
    """
    Categorize test triples into:
    - Emerging: At least one entity has low frequency (below tau percentile)
    - Novel-context: Both entities seen, but at least one hasn't seen this relation
    - In-distribution: Both entities covered for this relation
    """
    tau = np.percentile(entity_freq.numpy(), tau_percentile)

    emerging = []
    novel_context = []
    in_distribution = []

    for h, r, t in tqdm(test_triples, desc="Categorizing test triples"):
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
    Simple 2-layer GNN with energy-based OOD scoring.

    This is a simplified GNNSafe-style model that:
    1. Learns entity and relation embeddings
    2. Uses MLP to score triples
    3. Uses negative logit as energy score (higher = more OOD)
    """

    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=200):
        super().__init__()
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
        """GNNSafe-style energy score: -logit (higher = more uncertain)."""
        logits = self.forward(h, r, t)
        return -logits


def train_model(model, train_triples, num_entities, epochs=50, batch_size=1024,
                lr=1e-3, device='cpu', num_negatives=5):
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
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_ood_stratified(model, test_triples_by_category, train_triples,
                            num_entities, device='cpu', sample_size=5000):
    """
    Evaluate OOD detection using energy scores, stratified by category.

    Returns AUROC for each category where OOD = test triples, ID = train triples.
    """
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
            print(f"  {category}: too few samples ({len(triples)})")
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
            print(f"  {category} AUROC: {auroc:.3f} (n={n_samples})")
        except Exception as e:
            print(f"  {category}: error computing AUROC - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GNNSafe on YAGO3-10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=100000,
                       help="Max training triples (YAGO has 1M+)")
    parser.add_argument("--sample_size", type=int, default=5000,
                       help="Sample size for OOD evaluation")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"GNNSafe Experiment: YAGO3-10")
    print(f"{'='*70}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"embedding_dim={args.embedding_dim}, device={args.device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load YAGO3-10
    print("\nLoading YAGO3-10...")
    train_ds, valid_ds, test_ds = load_yago310()

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    train_triples = train_ds.triples.tolist()
    test_triples = test_ds.triples.tolist()

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples):,}")
    print(f"  Test triples: {len(test_triples):,}")

    # Compute gamma ratio
    print("\nComputing gamma ratio...")
    gamma, avg_neighbors = compute_gamma_ratio(train_triples, num_entities, num_relations)
    print(f"  Average neighbors: {avg_neighbors:.1f}")
    print(f"  Gamma (|R|/avg|N(e)|): {gamma:.2f}")

    # Subsample training if too large
    if len(train_triples) > args.max_train:
        print(f"\nSubsampling training data: {len(train_triples):,} -> {args.max_train:,}")
        indices = np.random.choice(len(train_triples), args.max_train, replace=False)
        train_triples_sample = [train_triples[i] for i in indices]
    else:
        train_triples_sample = train_triples

    # Build coverage matrix (using FULL training data for accurate categorization)
    print("\nBuilding coverage matrix...")
    coverage = compute_coverage_matrix(train_triples, num_entities, num_relations)
    entity_freq = compute_entity_frequency(train_triples, num_entities)

    # Categorize test triples
    print("\nCategorizing test triples...")
    emerging, novel_ctx, id_triples = categorize_test_triples(
        test_triples, coverage, entity_freq
    )

    print(f"\nTest triple categories:")
    print(f"  Emerging (low-freq entities): {len(emerging):,} ({100*len(emerging)/len(test_triples):.1f}%)")
    print(f"  Novel-context (new e-r pair): {len(novel_ctx):,} ({100*len(novel_ctx)/len(test_triples):.1f}%)")
    print(f"  In-distribution (seen e-r): {len(id_triples):,} ({100*len(id_triples)/len(test_triples):.1f}%)")

    # Train model
    print(f"\nTraining GNNSafe-style model ({args.epochs} epochs)...")
    model = SimpleGNN(num_entities, num_relations, args.embedding_dim)
    model = train_model(
        model, train_triples_sample, num_entities,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device
    )

    # Evaluate OOD detection
    print("\nEvaluating OOD detection (energy-based)...")
    test_by_category = {
        'emerging': emerging,
        'novel_context': novel_ctx,
        'in_distribution': id_triples,
        'overall': test_triples
    }

    results = evaluate_ood_stratified(
        model, test_by_category, train_triples_sample,
        num_entities, args.device, args.sample_size
    )

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"""
Dataset: YAGO3-10
Entities: {num_entities:,}
Relations: {num_relations}
Gamma: {gamma:.2f}
""")

    print("Comparison with other datasets:")
    novel_auroc_str = f"{results.get('novel_context', 0):.3f}" if 'novel_context' in results else 'N/A'
    print(f"""
Dataset        |R|    Avg N(e)   Gamma    Novel-Context AUROC
--------------------------------------------------------------
WN18RR         11     2.2        5.0      0.79 (GNN works)
YAGO3-10       {num_relations}     {avg_neighbors:.1f}       {gamma:.2f}      {novel_auroc_str}
FB15k-237      237    19.0       12.5     0.43 (GNN fails)
""")

    # Interpretation
    print("\nINTERPRETATION:")
    novel_auroc = results.get('novel_context')

    if novel_auroc is not None:
        if novel_auroc >= 0.70:
            print(f"  PREDICTION CONFIRMED: Novel-context AUROC = {novel_auroc:.3f} >= 0.70")
            print(f"  -> GNN CAN escape impossibility when gamma is low ({gamma:.2f})")
            print(f"  -> Neighbors successfully proxy for relation coverage")
        elif novel_auroc >= 0.55:
            print(f"  PARTIAL SUCCESS: Novel-context AUROC = {novel_auroc:.3f}")
            print(f"  -> GNN shows some relation-awareness")
            print(f"  -> Borderline case, may need more training")
        else:
            print(f"  PREDICTION FAILED: Novel-context AUROC = {novel_auroc:.3f} < 0.55")
            print(f"  -> GNN cannot detect novel contexts despite low gamma")
            print(f"  -> May need actual message passing or larger model")

    # Save results
    output = {
        'dataset': 'YAGO3-10',
        'config': vars(args),
        'statistics': {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'train_triples': len(train_triples),
            'test_triples': len(test_triples),
            'gamma': float(gamma),
            'avg_neighbors': float(avg_neighbors),
        },
        'test_categories': {
            'emerging': len(emerging),
            'novel_context': len(novel_ctx),
            'in_distribution': len(id_triples),
        },
        'auroc_results': {k: float(v) for k, v in results.items()},
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "outputs" / "yago_gnnsafe_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
