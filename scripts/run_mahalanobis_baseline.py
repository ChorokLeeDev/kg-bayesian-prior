#!/usr/bin/env python3
"""
Mahalanobis Distance Baseline for OOD Detection on FB15k-237.

Hypothesis: Mahalanobis distance should also fail (~0.5 AUROC) on novel-context
OOD because it's relation-agnostic -- it only measures distance from the
class-conditional Gaussian of all training embeddings, regardless of which
relations the entity has been observed with.

This validates our core thesis: relation-agnostic methods cannot detect
novel (entity, relation) contexts.
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
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class DistMultEmbedding(nn.Module):
    """Standard DistMult for embedding learning."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def train_embeddings(model, triples, device, epochs=30, lr=0.001):
    """Train embedding model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h = torch.tensor(triples[:, 0])
    r = torch.tensor(triples[:, 1])
    t = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(h, r, t), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for hb, rb, tb in loader:
            hb, rb, tb = hb.to(device), rb.to(device), tb.to(device)

            pos_scores = model(hb, rb, tb)
            neg_t = torch.randint(0, model.entity_emb.num_embeddings, tb.shape, device=device)
            neg_scores = model(hb, rb, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return model


def compute_mahalanobis_params(embeddings):
    """
    Compute mean and precision matrix (inverse covariance) for Mahalanobis distance.

    Uses shrinkage to handle potentially singular covariance.
    """
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean

    # Empirical covariance
    cov = centered.T @ centered / (embeddings.size(0) - 1)

    # Shrinkage regularization for numerical stability
    shrinkage = 0.1
    cov = (1 - shrinkage) * cov + shrinkage * torch.eye(cov.size(0), device=cov.device)

    # Precision matrix
    precision = torch.linalg.inv(cov)

    return mean, precision


def mahalanobis_distance(embeddings, mean, precision):
    """
    Compute Mahalanobis distance for each embedding.

    D(x) = sqrt((x - mu)^T @ Sigma^{-1} @ (x - mu))
    """
    centered = embeddings - mean
    # (N, D) @ (D, D) -> (N, D)
    left = centered @ precision
    # Element-wise multiply and sum -> (N,)
    distances = (left * centered).sum(dim=-1).sqrt()
    return distances


def evaluate_novel_context_ood(model, train, test, device, mean, precision):
    """
    Evaluate Mahalanobis on novel-context OOD (same partition as main paper).
    """
    model.eval()

    # Build coverage matrix from training
    n_ent = model.entity_emb.num_embeddings
    n_rel = model.relation_emb.num_embeddings
    coverage = np.zeros((n_ent, n_rel))
    for i in range(len(train)):
        coverage[train[i, 0], train[i, 1]] = 1
        coverage[train[i, 2], train[i, 1]] = 1

    # Entity frequencies
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)

    # Categorize test triples
    novel_ctx_idx = []
    id_idx = []

    for i in range(len(test)):
        h, r, t = test[i]
        h_freq = freq.get(h, 0)
        t_freq = freq.get(t, 0)

        # Skip emerging entities (low freq) -- focus on novel context
        if h_freq <= thresh or t_freq <= thresh:
            continue

        # Novel context: established entity, new relation
        if coverage[h, r] == 0 or coverage[t, r] == 0:
            novel_ctx_idx.append(i)
        else:
            id_idx.append(i)

    print(f"  Novel context: {len(novel_ctx_idx)}, ID: {len(id_idx)}")

    if len(novel_ctx_idx) < 50 or len(id_idx) < 50:
        print("  Insufficient samples")
        return None

    with torch.no_grad():
        # Novel context triples
        nc_triples = test[novel_ctx_idx]
        h_nc = torch.tensor(nc_triples[:, 0], device=device)
        t_nc = torch.tensor(nc_triples[:, 2], device=device)

        h_emb_nc = model.entity_emb(h_nc)
        t_emb_nc = model.entity_emb(t_nc)

        # Average head/tail Mahalanobis distance as triple uncertainty
        h_dist_nc = mahalanobis_distance(h_emb_nc, mean, precision)
        t_dist_nc = mahalanobis_distance(t_emb_nc, mean, precision)
        nc_unc = ((h_dist_nc + t_dist_nc) / 2).cpu().numpy()

        # ID triples
        id_triples = test[id_idx]
        h_id = torch.tensor(id_triples[:, 0], device=device)
        t_id = torch.tensor(id_triples[:, 2], device=device)

        h_emb_id = model.entity_emb(h_id)
        t_emb_id = model.entity_emb(t_id)

        h_dist_id = mahalanobis_distance(h_emb_id, mean, precision)
        t_dist_id = mahalanobis_distance(t_emb_id, mean, precision)
        id_unc = ((h_dist_id + t_dist_id) / 2).cpu().numpy()

    # AUROC: OOD (novel context) should have higher uncertainty
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(nc_unc))])
    scores = np.concatenate([id_unc, nc_unc])

    auroc = roc_auc_score(labels, scores)
    return auroc


def main():
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    log_path = output_dir / "mahalanobis_baseline.log"

    device = setup_device()

    results = []
    results.append("=" * 60)
    results.append("Mahalanobis Distance Baseline for Novel-Context OOD")
    results.append("=" * 60)
    results.append(f"Device: {device}")
    results.append("")

    print(results[-4])
    print(f"Device: {device}")

    # Load FB15k-237
    print("\nLoading FB15k-237...")
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    results.append(f"Dataset: FB15k-237")
    results.append(f"Entities: {n_ent}, Relations: {n_rel}")
    results.append(f"Train: {len(train)}, Test: {len(test)}")
    results.append("")

    print(f"Entities: {n_ent}, Relations: {n_rel}")

    # Run 3 seeds for stability
    seeds = [42, 123, 456]
    aurocs = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results.append(f"--- Seed {seed} ---")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train embeddings
        model = DistMultEmbedding(n_ent, n_rel, dim=100)
        t0 = time.time()
        model = train_embeddings(model, train, device, epochs=30)
        train_time = time.time() - t0

        # Compute Mahalanobis parameters from all entity embeddings
        with torch.no_grad():
            all_emb = model.entity_emb.weight.to(device)
            mean, precision = compute_mahalanobis_params(all_emb)

        # Evaluate
        auroc = evaluate_novel_context_ood(model, train, test, device, mean, precision)

        if auroc is not None:
            aurocs.append(auroc)
            msg = f"  Novel-Context AUROC: {auroc:.4f}  (time: {train_time:.1f}s)"
        else:
            msg = f"  Novel-Context AUROC: N/A"

        print(msg)
        results.append(msg)

    # Summary
    results.append("")
    results.append("=" * 60)
    results.append("SUMMARY")
    results.append("=" * 60)

    if aurocs:
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        summary = f"Mahalanobis Novel-Context AUROC: {mean_auroc:.3f} +/- {std_auroc:.3f}"
        results.append(summary)
        results.append("")
        results.append("Interpretation:")
        if mean_auroc < 0.55:
            results.append("  AUROC ~ 0.5 confirms hypothesis: Mahalanobis is relation-agnostic")
            results.append("  and cannot detect novel (entity, relation) contexts.")
        else:
            results.append(f"  AUROC = {mean_auroc:.3f} > 0.55: unexpected result.")

        print(f"\n{summary}")

    # Write log
    with open(log_path, 'w') as f:
        f.write("\n".join(results))

    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
