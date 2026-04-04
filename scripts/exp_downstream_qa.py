#!/usr/bin/env python3
"""
Downstream Task: Knowledge Graph Question Answering Simulation

Simulate: When answering questions, flag high-uncertainty answers.
Does RCUE flagging improve answer reliability?
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
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def predict_tail(self, h, r, n_ent):
        """Predict the most likely tail entity."""
        h_exp = torch.full((n_ent,), h, dtype=torch.long)
        r_exp = torch.full((n_ent,), r, dtype=torch.long)
        all_t = torch.arange(n_ent)
        scores = self(h_exp, r_exp, all_t)
        return scores.argmax().item(), scores.max().item()


def main():
    print("="*70)
    print("DOWNSTREAM TASK: KG-QA RELIABILITY SIMULATION")
    print("Goal: Show RCUE flagging improves answer reliability")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples[:2000]
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Test (simulated questions): {len(test)}")

    # Coverage
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
    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
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

    print("\n--- Training Energy ---")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
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

    # ========================================
    # Simulate QA
    # ========================================
    print("\n--- Simulating Question Answering ---")
    rcue.eval()
    energy.eval()

    # For each test triple (h, r, t), simulate question "What is (h, r, ?)?"
    # Answer = predicted t, correct if matches ground truth

    results = []

    with torch.no_grad():
        for idx, (h, r, t_true) in enumerate(test):
            if idx % 500 == 0:
                print(f"  Processing {idx}/{len(test)}...")

            h_tensor = torch.tensor([h])
            r_tensor = torch.tensor([r])

            # Predict answer
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)

            scores = rcue(h_exp, r_exp, all_t)
            pred_t = scores.argmax().item()
            correct = (pred_t == t_true)

            # Get uncertainties for the predicted answer
            t_pred_tensor = torch.tensor([pred_t])

            rcue_unc = rcue.get_uncertainty(h_tensor, r_tensor, t_pred_tensor).item()
            energy_score = energy(h_tensor, r_tensor, t_pred_tensor).item()
            energy_unc = -energy_score

            # Coverage-based uncertainty
            cov_unc = 0.0 if ((int(h), int(r)) in coverage_set and (int(pred_t), int(r)) in coverage_set) else 1.0

            results.append({
                'correct': correct,
                'rcue_unc': rcue_unc,
                'energy_unc': energy_unc,
                'cov_unc': cov_unc
            })

    # ========================================
    # Analyze: Flagging high-uncertainty improves precision
    # ========================================
    print("\n" + "="*70)
    print("QA RELIABILITY ANALYSIS")
    print("="*70)

    correct = np.array([r['correct'] for r in results])
    rcue_unc = np.array([r['rcue_unc'] for r in results])
    energy_unc = np.array([r['energy_unc'] for r in results])
    cov_unc = np.array([r['cov_unc'] for r in results])

    baseline_accuracy = correct.mean()
    print(f"\nBaseline accuracy (no filtering): {baseline_accuracy:.4f}")

    # Test different flagging thresholds
    print(f"\n{'Coverage':<10} {'Method':<15} {'Accuracy':<12} {'Flagged':<10}")
    print("-"*50)

    for keep_frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = int(len(results) * keep_frac)

        # Energy: keep lowest uncertainty
        energy_keep = np.argsort(energy_unc)[:n_keep]
        energy_acc = correct[energy_keep].mean()

        # Coverage: keep ID first
        cov_order = np.argsort(cov_unc)
        cov_keep = cov_order[:n_keep]
        cov_acc = correct[cov_keep].mean()

        # RCUE: keep lowest uncertainty
        rcue_keep = np.argsort(rcue_unc)[:n_keep]
        rcue_acc = correct[rcue_keep].mean()

        flagged = 1 - keep_frac

        if keep_frac == 1.0:
            print(f"{keep_frac:<10.0%} {'All':<15} {baseline_accuracy:<12.4f} {flagged:<10.0%}")
        else:
            print(f"{keep_frac:<10.0%} {'Energy':<15} {energy_acc:<12.4f} {flagged:<10.0%}")
            print(f"{'':<10} {'Coverage-Only':<15} {cov_acc:<12.4f} {'':<10}")
            print(f"{'':<10} {'RCUE':<15} {rcue_acc:<12.4f} {'':<10}")

    # ========================================
    # Key metric: Accuracy at 80% coverage
    # ========================================
    print("\n" + "="*70)
    print("KEY RESULT: Accuracy when answering 80% of questions")
    print("(20% flagged as uncertain)")
    print("="*70)

    n_keep = int(len(results) * 0.8)

    energy_keep = np.argsort(energy_unc)[:n_keep]
    cov_keep = np.argsort(cov_unc)[:n_keep]
    rcue_keep = np.argsort(rcue_unc)[:n_keep]

    print(f"\nBaseline (all):   {baseline_accuracy:.4f}")
    print(f"Energy flagging:  {correct[energy_keep].mean():.4f}")
    print(f"Coverage flagging:{correct[cov_keep].mean():.4f}")
    print(f"RCUE flagging:    {correct[rcue_keep].mean():.4f}")

    improvement = correct[rcue_keep].mean() - baseline_accuracy
    print(f"\nRCUE improvement: +{improvement:.4f} ({improvement/baseline_accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
