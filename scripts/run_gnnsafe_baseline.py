#!/usr/bin/env python3
"""
GNNSafe Baseline for KG OOD Detection.

Implements GNNSafe-style energy scoring for knowledge graphs:
- Energy score: E(x) = -logsumexp(f(x)) where f is model logits
- Uses a simple MLP (not full GNN) for fair comparison with other baselines

Usage:
    python scripts/run_gnnsafe_baseline.py --dataset fb15k237 --seeds 3
    python scripts/run_gnnsafe_baseline.py --dataset wn18rr --seeds 3
    python scripts/run_gnnsafe_baseline.py --dataset icews14 --seeds 3
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
import csv
from pathlib import Path


def load_dataset(dataset_name, data_dir="data/raw"):
    """Load full dataset."""
    dataset_paths = {
        "wn18rr": f"{data_dir}/WN18RR",
        "fb15k237": f"{data_dir}/FB15k-237",
        "icews14": f"{data_dir}/ICEWS14",
        "icews18": f"{data_dir}/icews18",
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    path = dataset_paths[dataset_name]

    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    entity2id = {}
    relation2id = {}

    def load_triples(filepath, update_vocab=True):
        triples = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Handle both 3-column (WN18RR, FB15k-237) and 5-column (ICEWS) formats
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]

                if update_vocab:
                    if h not in entity2id:
                        entity2id[h] = len(entity2id)
                    if t not in entity2id:
                        entity2id[t] = len(entity2id)
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)

                if h in entity2id and t in entity2id and r in relation2id:
                    triples.append((entity2id[h], relation2id[r], entity2id[t]))

        return triples

    train_triples = load_triples(f"{path}/train.txt", update_vocab=True)
    test_triples = load_triples(f"{path}/test.txt", update_vocab=False)

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


class GNNSafeModel(nn.Module):
    """GNNSafe-style model for KG triple scoring."""

    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        # 2-layer MLP (GNNSafe uses GNN, but we use MLP for fair comparison)
        self.fc1 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, 1)

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits.squeeze(-1)

    def energy_score(self, h, r, t):
        """GNNSafe energy score: -logit (higher = more uncertain)."""
        logits = self.forward(h, r, t)
        return -logits


def train_model(model, train_triples, num_entities, epochs=30, lr=1e-3,
                batch_size=1024, device='cpu'):
    """Train the model with BCE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_h = torch.tensor([t[0] for t in train_triples])
    train_r = torch.tensor([t[1] for t in train_triples])
    train_t = torch.tensor([t[2] for t in train_triples])

    n_batches = (len(train_triples) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0

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

            # Positive samples
            pos_scores = model(batch_h, batch_r, batch_t)

            # Negative samples (random tail corruption)
            neg_t = torch.randint(0, num_entities, (end - start,), device=device)
            neg_scores = model(batch_h, batch_r, neg_t)

            # BCE loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model


def evaluate_ood(model, ood_triples, id_triples, device='cpu'):
    """Evaluate OOD detection using energy scores."""
    model.eval()

    if len(ood_triples) < 5 or len(id_triples) < 5:
        return None

    with torch.no_grad():
        # OOD triples
        ood_h = torch.tensor([t[0] for t in ood_triples], device=device)
        ood_r = torch.tensor([t[1] for t in ood_triples], device=device)
        ood_t = torch.tensor([t[2] for t in ood_triples], device=device)
        ood_energy = model.energy_score(ood_h, ood_r, ood_t).cpu().numpy()

        # ID triples (sample same size as OOD)
        n_id = min(len(id_triples), len(ood_triples) * 2)
        id_sample = id_triples[:n_id]
        id_h = torch.tensor([t[0] for t in id_sample], device=device)
        id_r = torch.tensor([t[1] for t in id_sample], device=device)
        id_t = torch.tensor([t[2] for t in id_sample], device=device)
        id_energy = model.energy_score(id_h, id_r, id_t).cpu().numpy()

    # AUROC: OOD should have higher energy
    labels = np.concatenate([np.ones(len(ood_energy)), np.zeros(len(id_energy))])
    scores = np.concatenate([ood_energy, id_energy])

    return roc_auc_score(labels, scores)


def run_experiment(dataset_name, seed, device='cpu', epochs=30):
    """Run single experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- {dataset_name.upper()} (seed={seed}) ---")

    # Load data
    data = load_dataset(dataset_name)
    print(f"  Train: {len(data['train'])}, Test: {len(data['test'])}")
    print(f"  Entities: {data['num_entities']}, Relations: {data['num_relations']}")

    # Compute coverage and frequency
    coverage = compute_coverage_matrix(data['train'], data['num_entities'], data['num_relations'])
    entity_freq = compute_entity_frequency(data['train'], data['num_entities'])

    # Categorize test triples
    emerging, novel_ctx, id_triples = categorize_test_triples(
        data['test'], coverage, entity_freq
    )
    print(f"  Emerging: {len(emerging)}, Novel-ctx: {len(novel_ctx)}, ID: {len(id_triples)}")

    # Train model
    print(f"  Training ({epochs} epochs)...")
    model = GNNSafeModel(data['num_entities'], data['num_relations'])
    model = train_model(model, data['train'], data['num_entities'],
                       epochs=epochs, device=device)

    # Evaluate
    print(f"  Evaluating...")
    results = {
        'dataset': dataset_name,
        'seed': seed,
    }

    # Use training triples as ID reference
    auroc_emerging = evaluate_ood(model, emerging, data['train'], device)
    auroc_novel = evaluate_ood(model, novel_ctx, data['train'], device)
    auroc_all = evaluate_ood(model, emerging + novel_ctx, data['train'], device)

    results['emerging'] = auroc_emerging
    results['novel_context'] = auroc_novel
    results['overall'] = auroc_all

    em_str = f"{auroc_emerging:.3f}" if auroc_emerging else "N/A"
    nov_str = f"{auroc_novel:.3f}" if auroc_novel else "N/A"
    all_str = f"{auroc_all:.3f}" if auroc_all else "N/A"
    print(f"  Results: Em={em_str}, Nov={nov_str}, All={all_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GNNSafe Baseline")
    parser.add_argument("--dataset", type=str, default="fb15k237",
                       choices=["wn18rr", "fb15k237", "icews14", "icews18"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/gnnsafe_results.csv")
    args = parser.parse_args()

    seed_list = [42, 123, 456, 789, 1000, 1234, 2024, 2025, 2026, 314][:args.seeds]

    all_results = []
    for seed in seed_list:
        result = run_experiment(args.dataset, seed, args.device, args.epochs)
        all_results.append(result)

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS - {args.dataset.upper()} ({args.seeds} seeds)")
    print(f"{'='*60}")

    metrics = ['emerging', 'novel_context', 'overall']
    for metric in metrics:
        values = [r[metric] for r in all_results if r[metric] is not None]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.3f} ± {std:.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'seed', 'emerging', 'novel_context', 'overall'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
