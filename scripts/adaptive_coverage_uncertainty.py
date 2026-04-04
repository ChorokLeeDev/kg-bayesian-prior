#!/usr/bin/env python3
"""
2. ADAPTIVE COVERAGE-AWARE UNCERTAINTY

Method: Learn when to trust partial coverage vs not

Key idea:
- Coverage effect direction is predictable from local graph structure
- Train a meta-predictor: given (h, r, t) features, predict if partial is reliable

Features:
- Relation frequency
- Entity degrees
- Local coverage density
- Relation type (hierarchical vs flat)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score


class AdaptiveCoverageUncertainty(nn.Module):
    """
    Learns to predict uncertainty that adapts to coverage patterns.

    U(h, r, t) = base_uncertainty(h, r, t) * coverage_weight(features)

    where coverage_weight is learned to be:
    - High when partial coverage is unreliable (like WN18RR)
    - Low when partial coverage is reliable (like FB15k-237)
    """
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)

        # Feature dimension: emb_dim*2 (h+t) + structural features
        feature_dim = emb_dim * 2 + 5  # 5 structural features

        # Uncertainty predictor
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Coverage weight predictor (learns when coverage matters)
        self.coverage_weight_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        """Score function (DistMult)."""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_uncertainty(self, h, r, t, structural_features, coverage):
        """
        Adaptive uncertainty.

        structural_features: [batch, 5] - rel_freq, h_degree, t_degree, local_cov_density, rel_entropy
        coverage: [batch] - 0 (full_zero), 1 (partial), 2 (full_cov)
        """
        h_emb = self.entity_emb(h)
        t_emb = self.entity_emb(t)

        # Combine embeddings with structural features
        features = torch.cat([h_emb, t_emb, structural_features], dim=-1)

        # Base uncertainty
        base_unc = self.uncertainty_net(features).squeeze(-1)

        # Coverage weight (how much to penalize low coverage)
        cov_weight = self.coverage_weight_net(features).squeeze(-1)

        # Adaptive uncertainty
        # Higher weight = coverage matters more = penalize low coverage
        # Lower weight = coverage matters less = trust partial coverage
        coverage_penalty = (2 - coverage) * cov_weight  # 0 for full_cov, up to 2*weight for full_zero

        return base_unc * (1 + coverage_penalty)


def compute_structural_features(train, n_ent, n_rel):
    """Precompute structural features."""
    # Relation frequency
    rel_freq = np.zeros(n_rel)
    for h, r, t in train:
        rel_freq[int(r)] += 1
    rel_freq = rel_freq / rel_freq.max()  # Normalize

    # Entity degree
    ent_degree = np.zeros(n_ent)
    for h, r, t in train:
        ent_degree[int(h)] += 1
        ent_degree[int(t)] += 1
    ent_degree = np.log1p(ent_degree) / np.log1p(ent_degree.max())  # Log-normalize

    # Per-relation tail entropy
    rel_tails = {}
    for h, r, t in train:
        r = int(r)
        if r not in rel_tails:
            rel_tails[r] = {}
        rel_tails[r][int(t)] = rel_tails[r].get(int(t), 0) + 1

    rel_entropy = np.zeros(n_rel)
    for r, tails in rel_tails.items():
        total = sum(tails.values())
        probs = np.array(list(tails.values())) / total
        rel_entropy[r] = -np.sum(probs * np.log(probs + 1e-10))
    rel_entropy = rel_entropy / (rel_entropy.max() + 1e-10)

    # Coverage set
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    return rel_freq, ent_degree, rel_entropy, coverage_set


def get_features_and_coverage(triples, rel_freq, ent_degree, rel_entropy, coverage_set):
    """Get features and coverage for a batch of triples."""
    features = []
    coverages = []

    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)

        # Structural features
        f = [
            rel_freq[r],
            ent_degree[h],
            ent_degree[t],
            rel_entropy[r],
            (ent_degree[h] + ent_degree[t]) / 2  # avg degree
        ]
        features.append(f)

        # Coverage
        h_cov = (h, r) in coverage_set
        t_cov = (t, r) in coverage_set
        if h_cov and t_cov:
            cov = 2
        elif h_cov or t_cov:
            cov = 1
        else:
            cov = 0
        coverages.append(cov)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(coverages, dtype=torch.float32)


def train_and_evaluate(name, train_triples, test_triples, n_ent, n_rel):
    """Train adaptive model and evaluate."""
    print(f"\n{'='*60}")
    print(f"{name}: Adaptive Coverage-Aware Uncertainty")
    print(f"{'='*60}")

    # Compute features
    rel_freq, ent_degree, rel_entropy, coverage_set = compute_structural_features(train_triples, n_ent, n_rel)

    # Train model
    model = AdaptiveCoverageUncertainty(n_ent, n_rel)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training...")
    for epoch in range(10):
        np.random.shuffle(train_triples)
        total_loss = 0

        for i in range(0, len(train_triples), 512):
            batch = train_triples[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            features, coverage = get_features_and_coverage(batch, rel_freq, ent_degree, rel_entropy, coverage_set)

            optimizer.zero_grad()

            # Scoring loss
            pos_scores = model(h, r, t)
            neg_scores = model(h, r, t_neg)
            score_loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0).mean()

            # Uncertainty loss: high uncertainty on negatives
            pos_unc = model.get_uncertainty(h, r, t, features, coverage)
            neg_features, neg_cov = get_features_and_coverage(
                np.column_stack([batch[:, 0], batch[:, 1], t_neg.numpy()]),
                rel_freq, ent_degree, rel_entropy, coverage_set
            )
            neg_unc = model.get_uncertainty(h, r, t_neg, neg_features, neg_cov)

            unc_loss = torch.clamp(1.0 - neg_unc + pos_unc, min=0).mean()

            loss = score_loss + 0.1 * unc_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.2f}")

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    test_sub = test_triples[:2000] if len(test_triples) > 2000 else test_triples

    results = []
    with torch.no_grad():
        for h, r, t in test_sub:
            h_t, r_t, t_t = torch.tensor([h]), torch.tensor([r]), torch.tensor([t])
            features, coverage = get_features_and_coverage([[h, r, t]], rel_freq, ent_degree, rel_entropy, coverage_set)

            # Uncertainty
            unc = model.get_uncertainty(h_t, r_t, t_t, features, coverage).item()

            # Coverage type
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            is_ood = not (h_cov and t_cov)

            # Rank
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            scores = model(h_exp, r_exp, torch.arange(n_ent)).numpy()
            rank = int((scores > scores[t]).sum() + 1)

            results.append({
                'unc': unc,
                'is_ood': is_ood,
                'hits10': rank <= 10
            })

    # OOD detection AUROC
    labels = [r['is_ood'] for r in results]
    unc_scores = [r['unc'] for r in results]
    auroc = roc_auc_score(labels, unc_scores)
    print(f"OOD Detection AUROC: {auroc:.4f}")

    # Compare to baseline Energy
    energy_unc = [-r['unc'] for r in results]  # Negative for comparison
    # (In reality we'd compute Energy separately, but this is a proxy)

    return auroc


def main():
    print("="*60)
    print("ADAPTIVE COVERAGE-AWARE UNCERTAINTY")
    print("="*60)

    results = {}

    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = train_and_evaluate('FB15k-237', ds[0].triples, ds[2].triples,
                                               ds[0].num_entities, ds[0].num_relations)

    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = train_and_evaluate('WN18RR', ds[0].triples, ds[2].triples,
                                            ds[0].num_entities, ds[0].num_relations)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Adaptive AUROC':<15}")
    print("-"*30)
    for name, auroc in results.items():
        print(f"{name:<15} {auroc:<15.4f}")


if __name__ == "__main__":
    main()
