#!/usr/bin/env python3
"""
Confidence-Error Correlation Experiment

Measure Spearman correlation between uncertainty and prediction error.
Strong correlation = uncertainty is meaningful for prediction quality.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
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
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def get_ranks(model, test_triples, n_ent):
    """Get rank of correct tail for each test triple."""
    model.eval()
    ranks = []

    with torch.no_grad():
        for idx, (h, r, t) in enumerate(test_triples):
            if idx % 500 == 0:
                print(f"    Ranking {idx}/{len(test_triples)}...")

            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)

            scores = model(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = (scores > true_score).sum() + 1
            ranks.append(rank)

    return np.array(ranks)


def main():
    print("="*70)
    print("CONFIDENCE-ERROR CORRELATION EXPERIMENT")
    print("Goal: Show RCUE uncertainty correlates with prediction error")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples[:1500]  # Smaller subset for speed
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Test subset: {len(test)} triples")

    # Coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))

    # ========================================
    # Train RCUE
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

    # ========================================
    # Train Energy
    # ========================================
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
    # Get uncertainties and ranks
    # ========================================
    print("\n--- Computing uncertainties ---")
    rcue.eval()
    energy.eval()

    h_t = torch.tensor(test[:, 0])
    r_t = torch.tensor(test[:, 1])
    t_t = torch.tensor(test[:, 2])

    with torch.no_grad():
        rcue_unc = rcue.get_uncertainty(h_t, r_t, t_t).numpy()
        energy_unc = -energy(h_t, r_t, t_t).numpy()

        cov_unc = np.array([
            1.0 if ((int(h), int(r)) not in coverage_set or (int(t), int(r)) not in coverage_set) else 0.0
            for h, r, t in test
        ])

    print("\n--- Computing ranks (this takes a while) ---")
    print("  RCUE ranks...")
    rcue_ranks = get_ranks(rcue, test, n_ent)
    print("  Energy ranks...")
    energy_ranks = get_ranks(energy, test, n_ent)

    # Error = log(rank) to handle heavy tail
    rcue_error = np.log(rcue_ranks)
    energy_error = np.log(energy_ranks)

    # ========================================
    # Compute correlations
    # ========================================
    print("\n" + "="*70)
    print("SPEARMAN CORRELATION: Uncertainty vs log(Rank)")
    print("Higher correlation = uncertainty predicts prediction difficulty")
    print("="*70)

    # Energy
    corr_energy, p_energy = spearmanr(energy_unc, energy_error)
    print(f"\nEnergy:        ρ = {corr_energy:.4f} (p = {p_energy:.2e})")

    # Coverage-only
    corr_cov, p_cov = spearmanr(cov_unc, rcue_error)
    print(f"Coverage-Only: ρ = {corr_cov:.4f} (p = {p_cov:.2e})")

    # RCUE
    corr_rcue, p_rcue = spearmanr(rcue_unc, rcue_error)
    print(f"RCUE:          ρ = {corr_rcue:.4f} (p = {p_rcue:.2e})")

    # ========================================
    # Stratified analysis
    # ========================================
    print("\n" + "="*70)
    print("STRATIFIED CORRELATION (ID vs OOD)")
    print("="*70)

    ood_mask = cov_unc == 1.0
    id_mask = ~ood_mask

    if id_mask.sum() > 50:
        corr_rcue_id, _ = spearmanr(rcue_unc[id_mask], rcue_error[id_mask])
        corr_energy_id, _ = spearmanr(energy_unc[id_mask], energy_error[id_mask])
        print(f"\nID only ({id_mask.sum()} samples):")
        print(f"  Energy: ρ = {corr_energy_id:.4f}")
        print(f"  RCUE:   ρ = {corr_rcue_id:.4f}")

    if ood_mask.sum() > 50:
        corr_rcue_ood, _ = spearmanr(rcue_unc[ood_mask], rcue_error[ood_mask])
        corr_energy_ood, _ = spearmanr(energy_unc[ood_mask], energy_error[ood_mask])
        print(f"\nOOD only ({ood_mask.sum()} samples):")
        print(f"  Energy: ρ = {corr_energy_ood:.4f}")
        print(f"  RCUE:   ρ = {corr_rcue_ood:.4f}")

    # ========================================
    # Conclusion
    # ========================================
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if corr_rcue > corr_energy + 0.1:
        print(f"RCUE correlation ({corr_rcue:.3f}) >> Energy ({corr_energy:.3f})")
        print("RCUE uncertainty is MUCH more predictive of errors!")
    elif corr_rcue > corr_energy:
        print(f"RCUE correlation ({corr_rcue:.3f}) > Energy ({corr_energy:.3f})")
        print("RCUE uncertainty is more predictive of errors.")
    else:
        print(f"RCUE correlation ({corr_rcue:.3f}) ~ Energy ({corr_energy:.3f})")
        print("Similar predictive power.")


if __name__ == "__main__":
    main()
