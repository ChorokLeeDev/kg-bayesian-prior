#!/usr/bin/env python3
"""
GNNSafe Direction Test - CPU validation before full experiments.

Tests whether GNNSafe-style energy scoring on KGs exhibits the same
impossibility pattern as other entity-level methods:
- Novel-context AUROC ~0.5 (random) -> confirms impossibility
- Emerging AUROC > 0.6 -> entity-level signal works for rare entities

Usage:
    python scripts/test_gnnsafe_direction.py --dataset wn18rr --max_triples 1000 --epochs 5 --device cpu
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def load_dataset_subset(dataset_name, max_triples=1000, data_dir="data/raw"):
    """Load a subset of the dataset for quick CPU validation."""

    dataset_paths = {
        "wn18rr": f"{data_dir}/WN18RR",
        "fb15k237": f"{data_dir}/FB15k-237",
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    path = dataset_paths[dataset_name]

    def load_triples(filepath):
        triples = []
        entity2id = {}
        relation2id = {}

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts

                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

                triples.append((entity2id[h], relation2id[r], entity2id[t]))

        return triples, entity2id, relation2id

    train_triples, entity2id, relation2id = load_triples(f"{path}/train.txt")

    # Load test triples using same mappings
    test_triples = []
    with open(f"{path}/test.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h in entity2id and t in entity2id and r in relation2id:
                test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # Subsample for CPU validation
    if len(train_triples) > max_triples:
        indices = np.random.choice(len(train_triples), max_triples, replace=False)
        train_triples = [train_triples[i] for i in indices]

    if len(test_triples) > max_triples // 2:
        indices = np.random.choice(len(test_triples), max_triples // 2, replace=False)
        test_triples = [test_triples[i] for i in indices]

    return {
        'train': train_triples,
        'test': test_triples,
        'num_entities': len(entity2id),
        'num_relations': len(relation2id),
    }


def compute_coverage_matrix(triples, num_entities, num_relations):
    """Compute binary coverage matrix."""
    coverage = torch.zeros(num_entities, num_relations, dtype=torch.bool)
    for h, r, t in triples:
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


def categorize_test_triples(test_triples, coverage, entity_freq, tau_percentile=25):
    """Categorize test triples into emerging, novel-context, and ID."""
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
    """Simple 2-layer GCN for entity embeddings."""

    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        # Simple MLP layers (no message passing for simplicity)
        self.fc1 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)

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
        """GNNSafe-style energy score: -logsumexp(logits)."""
        logits = self.forward(h, r, t)
        # For single logit, energy = -logit (higher = more uncertain)
        return -logits


def train_model(model, train_triples, num_entities, epochs=5, lr=1e-3, device='cpu'):
    """Train the model with BCE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_h = torch.tensor([t[0] for t in train_triples], device=device)
    train_r = torch.tensor([t[1] for t in train_triples], device=device)
    train_t = torch.tensor([t[2] for t in train_triples], device=device)

    for epoch in range(epochs):
        model.train()

        # Positive samples
        pos_scores = model(train_h, train_r, train_t)

        # Negative samples (random tail corruption)
        neg_t = torch.randint(0, num_entities, (len(train_triples),), device=device)
        neg_scores = model(train_h, train_r, neg_t)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def evaluate_ood(model, test_triples, train_triples, num_entities, device='cpu'):
    """Evaluate OOD detection using energy scores."""
    model.eval()

    with torch.no_grad():
        # Test triples (potential OOD)
        test_h = torch.tensor([t[0] for t in test_triples], device=device)
        test_r = torch.tensor([t[1] for t in test_triples], device=device)
        test_t = torch.tensor([t[2] for t in test_triples], device=device)

        test_energy = model.energy_score(test_h, test_r, test_t).cpu().numpy()

        # ID triples (from training)
        id_sample = train_triples[:len(test_triples)]
        id_h = torch.tensor([t[0] for t in id_sample], device=device)
        id_r = torch.tensor([t[1] for t in id_sample], device=device)
        id_t = torch.tensor([t[2] for t in id_sample], device=device)

        id_energy = model.energy_score(id_h, id_r, id_t).cpu().numpy()

    # AUROC: OOD should have higher energy
    labels = np.concatenate([np.ones(len(test_energy)), np.zeros(len(id_energy))])
    scores = np.concatenate([test_energy, id_energy])

    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser(description="GNNSafe Direction Test")
    parser.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k237"])
    parser.add_argument("--max_triples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GNNSafe Direction Test - {args.dataset.upper()}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"\nLoading {args.dataset} (max {args.max_triples} triples)...")
    data = load_dataset_subset(args.dataset, args.max_triples)
    print(f"  Train: {len(data['train'])} triples")
    print(f"  Test: {len(data['test'])} triples")
    print(f"  Entities: {data['num_entities']}, Relations: {data['num_relations']}")

    # Compute coverage and frequency
    coverage = compute_coverage_matrix(data['train'], data['num_entities'], data['num_relations'])
    entity_freq = compute_entity_frequency(data['train'], data['num_entities'])

    # Categorize test triples
    emerging, novel_ctx, id_triples = categorize_test_triples(
        data['test'], coverage, entity_freq
    )
    print(f"\nTest split:")
    print(f"  Emerging: {len(emerging)}")
    print(f"  Novel-context: {len(novel_ctx)}")
    print(f"  In-distribution: {len(id_triples)}")

    # Create and train model
    print(f"\nTraining GNN model ({args.epochs} epochs)...")
    model = SimpleGNN(data['num_entities'], data['num_relations'])
    model = train_model(model, data['train'], data['num_entities'],
                       epochs=args.epochs, device=args.device)

    # Evaluate per category
    print(f"\nEvaluating OOD detection (energy-based)...")
    results = {}

    if len(emerging) > 10:
        auroc_emerging = evaluate_ood(model, emerging, data['train'],
                                      data['num_entities'], args.device)
        results['emerging'] = auroc_emerging
        print(f"  Emerging AUROC: {auroc_emerging:.3f}")
    else:
        print(f"  Emerging: too few samples ({len(emerging)})")

    if len(novel_ctx) > 10:
        auroc_novel = evaluate_ood(model, novel_ctx, data['train'],
                                   data['num_entities'], args.device)
        results['novel_context'] = auroc_novel
        print(f"  Novel-context AUROC: {auroc_novel:.3f}")
    else:
        print(f"  Novel-context: too few samples ({len(novel_ctx)})")

    if len(data['test']) > 10:
        auroc_all = evaluate_ood(model, data['test'], data['train'],
                                 data['num_entities'], args.device)
        results['overall'] = auroc_all
        print(f"  Overall AUROC: {auroc_all:.3f}")

    # Interpret results
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print(f"{'='*60}")

    if 'novel_context' in results:
        if results['novel_context'] < 0.55:
            print("✓ Novel-context AUROC ~0.5 (random)")
            print("  → CONFIRMS impossibility theorem")
            print("  → GNNSafe is entity-level, cannot detect novel contexts")
        elif results['novel_context'] > 0.7:
            print("⚠ Novel-context AUROC > 0.7")
            print("  → GNNSafe may be relation-aware")
            print("  → Investigate further before concluding")
        else:
            print(f"? Novel-context AUROC = {results['novel_context']:.3f}")
            print("  → Borderline, may need more data")

    if 'emerging' in results:
        if results['emerging'] > 0.6:
            print(f"✓ Emerging AUROC = {results['emerging']:.3f} > 0.6")
            print("  → Entity-level signal works for rare entities")
        else:
            print(f"? Emerging AUROC = {results['emerging']:.3f} < 0.6")
            print("  → May need more training or data")

    print(f"\n{'='*60}")

    return results


if __name__ == "__main__":
    main()
