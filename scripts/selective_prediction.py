#!/usr/bin/env python3
"""
Selective Prediction Experiment

Key insight: If uncertainty is meaningful, abstaining on high-uncertainty samples
should improve accuracy on the remaining samples.

This demonstrates PRACTICAL value of gradation (not just detection).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237
from src.models.relation_conditioned import RCUE


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)


def compute_mrr_at_coverage(scores, uncertainties, coverage_levels):
    """
    Compute MRR at different coverage levels.

    Coverage = fraction of samples we make predictions on (1 - abstention rate)
    Lower uncertainty → predict, higher uncertainty → abstain
    """
    n = len(scores)
    sorted_indices = np.argsort(uncertainties)  # Low uncertainty first

    results = {}
    for coverage in coverage_levels:
        n_predict = int(n * coverage)
        if n_predict == 0:
            continue

        predict_indices = sorted_indices[:n_predict]
        selected_scores = scores[predict_indices]

        # For link prediction: higher score = better rank
        # MRR approximation: use score directly as proxy
        # Actually compute ranking-based metric
        mrr = selected_scores.mean()  # Simplified - real MRR needs ranking
        results[coverage] = mrr

    return results


def evaluate_link_prediction(model, test_triples, n_ent, batch_size=1000):
    """Compute MRR with proper ranking."""
    model.eval()

    all_ranks = []

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])

            # Score all possible tails
            all_t = torch.arange(n_ent)

            for j in range(len(batch)):
                h_j = h[j].expand(n_ent)
                r_j = r[j].expand(n_ent)

                scores = model(h_j, r_j, all_t).numpy()

                true_score = scores[t[j].item()]
                rank = (scores > true_score).sum() + 1
                all_ranks.append(rank)

    ranks = np.array(all_ranks)
    mrr = (1.0 / ranks).mean()
    hits1 = (ranks <= 1).mean()
    hits10 = (ranks <= 10).mean()

    return mrr, hits1, hits10, ranks


def main():
    print("="*70)
    print("SELECTIVE PREDICTION EXPERIMENT")
    print("Goal: Show RCUE uncertainty enables better abstention than baselines")
    print("="*70)

    device = torch.device('cpu')

    # Load data
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples[:2000]  # Subsample for speed
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Test subset: {len(test)} triples")

    # Coverage set
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    # ========================================
    # Train models
    # ========================================
    print("\n--- Training RCUE ---")
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, embedding_dim=100, hidden_dim=64, use_coverage=True)
    rcue.precompute_coverage(train)

    optimizer = torch.optim.Adam(rcue.parameters(), lr=1e-3)
    batch_size = 512

    for epoch in range(15):
        np.random.shuffle(train)
        total_loss = 0
        for i in range(0, len(train), batch_size):
            batch = train[i:i+batch_size]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = rcue(h, r, t)
            neg = rcue(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15, Loss: {total_loss:.2f}")

    print("\n--- Training Energy ---")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel, emb_dim=100)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(15):
        np.random.shuffle(train)
        total_loss = 0
        for i in range(0, len(train), batch_size):
            batch = train[i:i+batch_size]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = energy(h, r, t)
            neg = energy(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15, Loss: {total_loss:.2f}")

    # ========================================
    # Compute uncertainties
    # ========================================
    print("\n--- Computing uncertainties ---")

    h_t = torch.tensor(test[:, 0])
    r_t = torch.tensor(test[:, 1])
    t_t = torch.tensor(test[:, 2])

    rcue.eval()
    energy.eval()

    with torch.no_grad():
        rcue_unc = rcue.get_uncertainty(h_t, r_t, t_t).numpy()
        energy_unc = -energy(h_t, r_t, t_t).numpy()  # Negative score as uncertainty

        # Coverage-only uncertainty
        cov_unc = np.array([
            1.0 if ((int(h), int(r)) not in coverage_set or (int(t), int(r)) not in coverage_set) else 0.0
            for h, r, t in test
        ])

    # ========================================
    # Evaluate at different coverage levels
    # ========================================
    print("\n--- Computing MRR at coverage levels ---")

    # For each triple, get the rank
    def get_ranks(model, test_triples, n_ent):
        """Get rank of correct tail for each test triple."""
        model.eval()
        ranks = []

        with torch.no_grad():
            for h, r, t in test_triples:
                h_exp = torch.tensor([h] * n_ent)
                r_exp = torch.tensor([r] * n_ent)
                all_t = torch.arange(n_ent)

                scores = model(h_exp, r_exp, all_t).numpy()
                true_score = scores[t]
                rank = (scores > true_score).sum() + 1
                ranks.append(rank)

        return np.array(ranks)

    print("  Computing ranks for RCUE...")
    rcue_ranks = get_ranks(rcue, test, n_ent)
    print("  Computing ranks for Energy...")
    energy_ranks = get_ranks(energy, test, n_ent)

    # Selective prediction: keep low-uncertainty samples
    coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    print("\n" + "="*70)
    print("SELECTIVE PREDICTION RESULTS")
    print("Coverage = fraction of samples we predict on (rest are abstained)")
    print("MRR = Mean Reciprocal Rank (higher is better)")
    print("="*70)

    print(f"\n{'Coverage':<10} {'Energy MRR':<12} {'Cov-Only MRR':<14} {'RCUE MRR':<12}")
    print("-"*50)

    for cov in coverage_levels:
        n_keep = int(len(test) * cov)

        # Energy: keep lowest uncertainty (highest score)
        energy_keep = np.argsort(energy_unc)[:n_keep]
        energy_mrr = (1.0 / energy_ranks[energy_keep]).mean()

        # Coverage-only: keep ID samples first (uncertainty=0), then random
        cov_keep_id = np.where(cov_unc == 0)[0]
        if len(cov_keep_id) >= n_keep:
            cov_keep = cov_keep_id[:n_keep]
        else:
            # Need to include some OOD
            cov_keep_ood = np.where(cov_unc == 1)[0]
            n_ood_needed = n_keep - len(cov_keep_id)
            cov_keep = np.concatenate([cov_keep_id, cov_keep_ood[:n_ood_needed]])
        # Use same ranks as RCUE (same underlying model assumption)
        cov_mrr = (1.0 / rcue_ranks[cov_keep]).mean()

        # RCUE: keep lowest uncertainty
        rcue_keep = np.argsort(rcue_unc)[:n_keep]
        rcue_mrr = (1.0 / rcue_ranks[rcue_keep]).mean()

        print(f"{cov:<10.0%} {energy_mrr:<12.4f} {cov_mrr:<14.4f} {rcue_mrr:<12.4f}")

    # ========================================
    # Key metric: Risk-Coverage AUC
    # ========================================
    print("\n" + "="*70)
    print("RISK-COVERAGE AUC (lower is better)")
    print("="*70)

    def risk_coverage_auc(uncertainties, errors, n_points=100):
        """Compute area under risk-coverage curve."""
        sorted_idx = np.argsort(uncertainties)
        sorted_errors = errors[sorted_idx]

        coverages = np.linspace(0.1, 1.0, n_points)
        risks = []

        for cov in coverages:
            n_keep = int(len(sorted_errors) * cov)
            risk = sorted_errors[:n_keep].mean()
            risks.append(risk)

        # AUC via trapezoidal rule
        auc = np.trapz(risks, coverages)
        return auc

    # Error = 1 if rank > 10, else 0 (Hits@10 error)
    errors = (rcue_ranks > 10).astype(float)

    rc_auc_energy = risk_coverage_auc(energy_unc, errors)
    rc_auc_cov = risk_coverage_auc(cov_unc, errors)
    rc_auc_rcue = risk_coverage_auc(rcue_unc, errors)

    print(f"Energy:        {rc_auc_energy:.4f}")
    print(f"Coverage-Only: {rc_auc_cov:.4f}")
    print(f"RCUE:          {rc_auc_rcue:.4f}")

    improvement_over_energy = (rc_auc_energy - rc_auc_rcue) / rc_auc_energy * 100
    improvement_over_cov = (rc_auc_cov - rc_auc_rcue) / rc_auc_cov * 100

    print(f"\nRCUE improvement over Energy: {improvement_over_energy:.1f}%")
    print(f"RCUE improvement over Coverage-Only: {improvement_over_cov:.1f}%")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("If RCUE improves Risk-Coverage AUC, it means:")
    print("  - RCUE's gradation helps identify 'easy' vs 'hard' samples")
    print("  - Abstaining on high-RCUE-uncertainty improves reliability")
    print("  - This is PRACTICAL value beyond binary OOD detection")


if __name__ == "__main__":
    main()
