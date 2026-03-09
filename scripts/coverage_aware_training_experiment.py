#!/usr/bin/env python3
"""
Coverage-Aware Training Experiment

Key question: Can we train a model to recognize "absence" (unseen e,r pairs)?

Design:
1. Split unseen (e,r) pairs into:
   - Training unseen: used for coverage regularization
   - Held-out unseen: only for evaluation (never seen during training)

2. Train with coverage regularization: encourage high uncertainty on training unseen

3. Evaluate:
   - Training unseen → high uncertainty (trivial, we trained on these)
   - Held-out unseen → high uncertainty? (KEY: does it generalize?)

If held-out unseen also has high uncertainty, the model learned a generalizable
pattern of "I don't know this combination" - a major finding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237


class CoverageAwareKGE(nn.Module):
    """KGE with learnable uncertainty that can be regularized on unseen pairs."""

    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_mean = nn.Embedding(num_entities, embedding_dim)
        self.entity_logvar = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_logvar.weight, -3.0)  # Low initial variance
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.embedding_dim = embedding_dim

    def forward(self, h, r, t, sample=True):
        """DistMult scoring with optional reparameterization sampling."""
        h_mean = self.entity_mean(h)
        t_mean = self.entity_mean(t)
        r_emb = self.relation_emb(r)

        if sample and self.training:
            h_std = torch.exp(0.5 * self.entity_logvar(h))
            t_std = torch.exp(0.5 * self.entity_logvar(t))
            h_emb = h_mean + h_std * torch.randn_like(h_mean)
            t_emb = t_mean + t_std * torch.randn_like(t_mean)
        else:
            h_emb = h_mean
            t_emb = t_mean

        score = (h_emb * r_emb * t_emb).sum(dim=-1)
        return score

    def get_uncertainty(self, h, r, t):
        """Get entity-level uncertainty (mean variance of h and t)."""
        h_var = torch.exp(self.entity_logvar(h)).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar(t)).mean(dim=-1)
        return 0.5 * (h_var + t_var)

    def get_entity_relation_uncertainty(self, e, r):
        """Get uncertainty for (entity, relation) pair."""
        e_var = torch.exp(self.entity_logvar(e)).mean(dim=-1)
        return e_var


def build_coverage_matrix(triples, num_entities, num_relations):
    """Build coverage matrix: coverage[e, r] = 1 if entity e seen with relation r."""
    coverage = np.zeros((num_entities, num_relations), dtype=np.float32)
    for h, r, t in triples:
        coverage[h, r] = 1
        coverage[t, r] = 1
    return coverage


def sample_unseen_pairs(coverage, num_samples, exclude_set=None):
    """Sample (entity, relation) pairs that have coverage=0."""
    num_entities, num_relations = coverage.shape
    unseen_pairs = []

    # Find all unseen pairs
    for e in range(num_entities):
        for r in range(num_relations):
            if coverage[e, r] == 0:
                if exclude_set is None or (e, r) not in exclude_set:
                    unseen_pairs.append((e, r))

    if len(unseen_pairs) <= num_samples:
        return unseen_pairs

    return random.sample(unseen_pairs, num_samples)


def run_experiment(dataset_name="FB15k-237", embedding_dim=100, epochs=50,
                   coverage_weight=0.1, batch_size=256, lr=0.001):
    """
    Main experiment: Compare baseline vs coverage-aware training.
    """
    print(f"\n{'='*60}")
    print("Coverage-Aware Training Experiment")
    print(f"{'='*60}\n")

    # Load data
    print("Loading dataset...")
    train_ds, valid_ds, test_ds = load_fb15k237()
    train_triples = train_ds.triples
    valid_triples = valid_ds.triples
    test_triples = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")

    # Build coverage matrix
    coverage = build_coverage_matrix(train_triples, num_entities, num_relations)
    total_pairs = num_entities * num_relations
    covered_pairs = coverage.sum()
    unseen_total = total_pairs - covered_pairs
    coverage_rate = covered_pairs / total_pairs

    print(f"\nCoverage matrix: {num_entities} x {num_relations} = {total_pairs:,} pairs")
    print(f"Covered: {covered_pairs:,.0f} ({coverage_rate:.2%})")
    print(f"Unseen: {unseen_total:,.0f} ({1-coverage_rate:.2%})")

    # Split unseen pairs into training and held-out
    print("\nSampling unseen pairs...")
    num_unseen_train = 50000  # Use 50K for training regularization
    num_unseen_heldout = 10000  # Hold out 10K for evaluation

    all_unseen = sample_unseen_pairs(coverage, num_unseen_train + num_unseen_heldout)
    random.shuffle(all_unseen)

    unseen_train = set(all_unseen[:num_unseen_train])
    unseen_heldout = set(all_unseen[num_unseen_train:num_unseen_train + num_unseen_heldout])

    print(f"Training unseen: {len(unseen_train):,}")
    print(f"Held-out unseen: {len(unseen_heldout):,}")

    # Also sample "seen" pairs for comparison
    seen_pairs = []
    for e in range(num_entities):
        for r in range(num_relations):
            if coverage[e, r] == 1:
                seen_pairs.append((e, r))
    seen_sample = random.sample(seen_pairs, min(10000, len(seen_pairs)))
    print(f"Seen sample for comparison: {len(seen_sample):,}")

    # Prepare training data
    train_h = torch.tensor([t[0] for t in train_triples], dtype=torch.long)
    train_r = torch.tensor([t[1] for t in train_triples], dtype=torch.long)
    train_t = torch.tensor([t[2] for t in train_triples], dtype=torch.long)

    unseen_train_list = list(unseen_train)
    unseen_e_train = torch.tensor([p[0] for p in unseen_train_list], dtype=torch.long)
    unseen_r_train = torch.tensor([p[1] for p in unseen_train_list], dtype=torch.long)

    # ====== BASELINE MODEL ======
    print(f"\n{'='*60}")
    print("Training BASELINE model (no coverage regularization)")
    print(f"{'='*60}")

    baseline_model = CoverageAwareKGE(num_entities, num_relations, embedding_dim)
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=lr)

    for epoch in range(epochs):
        baseline_model.train()
        epoch_loss = 0

        # Shuffle and batch
        perm = torch.randperm(len(train_triples))
        for i in range(0, len(train_triples), batch_size):
            idx = perm[i:i+batch_size]
            h, r, t = train_h[idx], train_r[idx], train_t[idx]

            # Positive scores
            pos_scores = baseline_model(h, r, t)

            # Negative sampling (corrupt tail)
            neg_t = torch.randint(0, num_entities, (len(h),))
            neg_scores = baseline_model(h, r, neg_t)

            # BCE loss
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels) + \
                   F.binary_cross_entropy_with_logits(neg_scores, neg_labels)

            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # ====== COVERAGE-AWARE MODEL ======
    print(f"\n{'='*60}")
    print(f"Training COVERAGE-AWARE model (weight={coverage_weight})")
    print(f"{'='*60}")

    coverage_model = CoverageAwareKGE(num_entities, num_relations, embedding_dim)
    optimizer_cov = torch.optim.Adam(coverage_model.parameters(), lr=lr)

    for epoch in range(epochs):
        coverage_model.train()
        epoch_loss = 0
        epoch_cov_loss = 0

        # Shuffle and batch
        perm = torch.randperm(len(train_triples))
        for i in range(0, len(train_triples), batch_size):
            idx = perm[i:i+batch_size]
            h, r, t = train_h[idx], train_r[idx], train_t[idx]

            # Positive scores
            pos_scores = coverage_model(h, r, t)

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(h),))
            neg_scores = coverage_model(h, r, neg_t)

            # BCE loss
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            bce_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels) + \
                       F.binary_cross_entropy_with_logits(neg_scores, neg_labels)

            # Coverage regularization: sample unseen pairs, encourage HIGH uncertainty
            unseen_idx = torch.randint(0, len(unseen_train_list), (batch_size // 4,))
            unseen_e_batch = unseen_e_train[unseen_idx]
            unseen_r_batch = unseen_r_train[unseen_idx]

            # Get uncertainty for unseen pairs
            unseen_unc = coverage_model.get_entity_relation_uncertainty(unseen_e_batch, unseen_r_batch)

            # We want HIGH uncertainty on unseen → minimize -log(uncertainty)
            # Or equivalently, maximize uncertainty → minimize -uncertainty
            # Using log for stability: minimize -log(uncertainty + eps)
            coverage_loss = -torch.log(unseen_unc + 1e-6).mean()

            total_loss = bce_loss + coverage_weight * coverage_loss

            optimizer_cov.zero_grad()
            total_loss.backward()
            optimizer_cov.step()

            epoch_loss += bce_loss.item()
            epoch_cov_loss += coverage_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, BCE: {epoch_loss:.4f}, Coverage: {epoch_cov_loss:.4f}")

    # ====== EVALUATION ======
    print(f"\n{'='*60}")
    print("EVALUATION: Does coverage-awareness generalize?")
    print(f"{'='*60}")

    baseline_model.eval()
    coverage_model.eval()

    # Convert evaluation sets to tensors
    unseen_heldout_list = list(unseen_heldout)
    heldout_e = torch.tensor([p[0] for p in unseen_heldout_list], dtype=torch.long)
    heldout_r = torch.tensor([p[1] for p in unseen_heldout_list], dtype=torch.long)

    seen_e = torch.tensor([p[0] for p in seen_sample], dtype=torch.long)
    seen_r = torch.tensor([p[1] for p in seen_sample], dtype=torch.long)

    with torch.no_grad():
        # Baseline uncertainties
        base_unc_train = baseline_model.get_entity_relation_uncertainty(unseen_e_train, unseen_r_train)
        base_unc_heldout = baseline_model.get_entity_relation_uncertainty(heldout_e, heldout_r)
        base_unc_seen = baseline_model.get_entity_relation_uncertainty(seen_e, seen_r)

        # Coverage-aware uncertainties
        cov_unc_train = coverage_model.get_entity_relation_uncertainty(unseen_e_train, unseen_r_train)
        cov_unc_heldout = coverage_model.get_entity_relation_uncertainty(heldout_e, heldout_r)
        cov_unc_seen = coverage_model.get_entity_relation_uncertainty(seen_e, seen_r)

    print("\n--- BASELINE MODEL ---")
    print(f"Unseen (train):   {base_unc_train.mean():.4f} ± {base_unc_train.std():.4f}")
    print(f"Unseen (heldout): {base_unc_heldout.mean():.4f} ± {base_unc_heldout.std():.4f}")
    print(f"Seen:             {base_unc_seen.mean():.4f} ± {base_unc_seen.std():.4f}")
    print(f"Ratio (heldout/seen): {base_unc_heldout.mean() / base_unc_seen.mean():.2f}x")

    print("\n--- COVERAGE-AWARE MODEL ---")
    print(f"Unseen (train):   {cov_unc_train.mean():.4f} ± {cov_unc_train.std():.4f}")
    print(f"Unseen (heldout): {cov_unc_heldout.mean():.4f} ± {cov_unc_heldout.std():.4f}")
    print(f"Seen:             {cov_unc_seen.mean():.4f} ± {cov_unc_seen.std():.4f}")
    print(f"Ratio (heldout/seen): {cov_unc_heldout.mean() / cov_unc_seen.mean():.2f}x")

    # KEY METRIC: Does coverage regularization generalize?
    print(f"\n{'='*60}")
    print("KEY RESULTS")
    print(f"{'='*60}")

    baseline_ratio = base_unc_heldout.mean() / base_unc_seen.mean()
    coverage_ratio = cov_unc_heldout.mean() / cov_unc_seen.mean()
    improvement = coverage_ratio / baseline_ratio

    print(f"\nBaseline: Held-out unseen / Seen uncertainty ratio = {baseline_ratio:.2f}x")
    print(f"Coverage-aware: Held-out unseen / Seen ratio = {coverage_ratio:.2f}x")
    print(f"Improvement: {improvement:.2f}x")

    # AUROC-like metric: Can we distinguish held-out unseen from seen?
    from sklearn.metrics import roc_auc_score

    # Labels: 1 = unseen (OOD), 0 = seen (ID)
    labels = np.concatenate([np.ones(len(unseen_heldout_list)), np.zeros(len(seen_sample))])

    base_scores = torch.cat([base_unc_heldout, base_unc_seen]).numpy()
    cov_scores = torch.cat([cov_unc_heldout, cov_unc_seen]).numpy()

    base_auroc = roc_auc_score(labels, base_scores)
    cov_auroc = roc_auc_score(labels, cov_scores)

    print(f"\nAUROC (Held-out Unseen vs Seen):")
    print(f"  Baseline:       {base_auroc:.4f}")
    print(f"  Coverage-aware: {cov_auroc:.4f}")
    print(f"  Δ AUROC:        {cov_auroc - base_auroc:+.4f}")

    # Final verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if cov_auroc > base_auroc + 0.05:
        print("✓ Coverage-aware training GENERALIZES!")
        print("  The model learned to recognize unseen (e,r) pairs.")
        print("  → Major finding: Absence CAN be used as training signal.")
    elif cov_auroc > base_auroc:
        print("~ Marginal improvement.")
        print("  Coverage regularization helps but doesn't fully generalize.")
    else:
        print("✗ Coverage-aware training does NOT generalize.")
        print("  The model only memorized the training unseen pairs.")
        print("  → Confirms Theorem 2: Embeddings cannot encode coverage.")

    return {
        'baseline_auroc': base_auroc,
        'coverage_auroc': cov_auroc,
        'baseline_ratio': baseline_ratio,
        'coverage_ratio': coverage_ratio,
    }


if __name__ == "__main__":
    results = run_experiment(
        embedding_dim=100,
        epochs=50,
        coverage_weight=0.1,
        batch_size=256,
        lr=0.001
    )
