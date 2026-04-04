#!/usr/bin/env python3
"""
Cross-Dataset Transfer Experiment

Train on FB15k-237, test on WN18RR (different domain).
Shows if RCUE uncertainty generalizes.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score


class TransferableRCUE(nn.Module):
    """RCUE that can transfer embeddings across datasets."""
    def __init__(self, emb_dim=100, hidden_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        # MLP takes concatenated embeddings, not indices
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        self.boost_logit = nn.Parameter(torch.tensor(0.7))

    def get_uncertainty(self, h_emb, r_emb, t_emb, coverage):
        """Compute uncertainty from embeddings (not indices)."""
        # Head uncertainty
        inp_h = torch.cat([h_emb, r_emb], dim=-1)
        var_h = self.uncertainty_mlp(inp_h).squeeze(-1)

        # Tail uncertainty
        inp_t = torch.cat([t_emb, r_emb], dim=-1)
        var_t = self.uncertainty_mlp(inp_t).squeeze(-1)

        # Boost
        k = torch.exp(self.boost_logit)
        boost = 1.0 + k * (1.0 - coverage)

        return (var_h + var_t) * boost


def main():
    print("="*70)
    print("CROSS-DATASET TRANSFER EXPERIMENT")
    print("Train uncertainty MLP on FB15k-237, test on WN18RR")
    print("="*70)

    # Load FB15k-237
    print("\n--- Loading FB15k-237 ---")
    fb_train_ds, _, fb_test_ds = load_fb15k237()
    fb_train = fb_train_ds.triples
    fb_test = fb_test_ds.triples[:3000]
    fb_n_ent = fb_train_ds.num_entities
    fb_n_rel = fb_train_ds.num_relations
    print(f"FB15k-237: {fb_n_ent} entities, {fb_n_rel} relations")

    # Load WN18RR
    print("\n--- Loading WN18RR ---")
    wn_train_ds, _, wn_test_ds = load_wn18rr()
    wn_train = wn_train_ds.triples
    wn_test = wn_test_ds.triples[:2000]
    wn_n_ent = wn_train_ds.num_entities
    wn_n_rel = wn_train_ds.num_relations
    print(f"WN18RR: {wn_n_ent} entities, {wn_n_rel} relations")

    # Train embeddings on each dataset separately
    emb_dim = 100

    print("\n--- Training FB15k-237 embeddings ---")
    torch.manual_seed(42)
    fb_ent_emb = nn.Embedding(fb_n_ent, emb_dim)
    fb_rel_emb = nn.Embedding(fb_n_rel, emb_dim)
    nn.init.xavier_uniform_(fb_ent_emb.weight)
    nn.init.xavier_uniform_(fb_rel_emb.weight)

    optimizer = torch.optim.Adam(list(fb_ent_emb.parameters()) + list(fb_rel_emb.parameters()), lr=1e-3)

    for epoch in range(10):
        np.random.shuffle(fb_train)
        total_loss = 0
        for i in range(0, len(fb_train), 512):
            batch = fb_train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, fb_n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = (fb_ent_emb(h) * fb_rel_emb(r) * fb_ent_emb(t)).sum(-1)
            neg = (fb_ent_emb(h) * fb_rel_emb(r) * fb_ent_emb(t_neg)).sum(-1)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.2f}")

    print("\n--- Training WN18RR embeddings ---")
    torch.manual_seed(42)
    wn_ent_emb = nn.Embedding(wn_n_ent, emb_dim)
    wn_rel_emb = nn.Embedding(wn_n_rel, emb_dim)
    nn.init.xavier_uniform_(wn_ent_emb.weight)
    nn.init.xavier_uniform_(wn_rel_emb.weight)

    optimizer = torch.optim.Adam(list(wn_ent_emb.parameters()) + list(wn_rel_emb.parameters()), lr=1e-3)

    for epoch in range(10):
        np.random.shuffle(wn_train)
        total_loss = 0
        for i in range(0, len(wn_train), 512):
            batch = wn_train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, wn_n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = (wn_ent_emb(h) * wn_rel_emb(r) * wn_ent_emb(t)).sum(-1)
            neg = (wn_ent_emb(h) * wn_rel_emb(r) * wn_ent_emb(t_neg)).sum(-1)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.2f}")

    # Train uncertainty MLP on FB15k-237
    print("\n--- Training Uncertainty MLP on FB15k-237 ---")
    torch.manual_seed(42)
    unc_mlp = TransferableRCUE(emb_dim=emb_dim, hidden_dim=64)

    # Build coverage for FB15k-237
    fb_coverage_set = set()
    for h, r, t in fb_train:
        fb_coverage_set.add((int(h), int(r)))
        fb_coverage_set.add((int(t), int(r)))

    optimizer = torch.optim.Adam(unc_mlp.parameters(), lr=1e-3)

    for epoch in range(10):
        np.random.shuffle(fb_train)
        total_loss = 0
        for i in range(0, len(fb_train), 512):
            batch = fb_train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])

            h_emb = fb_ent_emb(h).detach()
            r_emb = fb_rel_emb(r).detach()
            t_emb = fb_ent_emb(t).detach()

            # Coverage
            cov = torch.tensor([
                1.0 if ((int(hh), int(rr)) in fb_coverage_set and (int(tt), int(rr)) in fb_coverage_set) else 0.0
                for hh, rr, tt in zip(batch[:, 0], batch[:, 1], batch[:, 2])
            ])

            optimizer.zero_grad()
            unc = unc_mlp.get_uncertainty(h_emb, r_emb, t_emb, cov)

            # Train to be low on positive samples
            loss = unc.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.4f}")

    # ========================================
    # Evaluate on FB15k-237 (in-domain)
    # ========================================
    print("\n--- Evaluating on FB15k-237 (in-domain) ---")
    fb_ent_emb.eval()
    fb_rel_emb.eval()
    unc_mlp.eval()

    fb_ood_mask = np.array([
        (int(h), int(r)) not in fb_coverage_set or (int(t), int(r)) not in fb_coverage_set
        for h, r, t in fb_test
    ])

    with torch.no_grad():
        h = torch.tensor(fb_test[:, 0])
        r = torch.tensor(fb_test[:, 1])
        t = torch.tensor(fb_test[:, 2])

        h_emb = fb_ent_emb(h)
        r_emb = fb_rel_emb(r)
        t_emb = fb_ent_emb(t)

        cov = torch.tensor([
            1.0 if ((int(hh), int(rr)) in fb_coverage_set and (int(tt), int(rr)) in fb_coverage_set) else 0.0
            for hh, rr, tt in fb_test
        ])

        fb_unc = unc_mlp.get_uncertainty(h_emb, r_emb, t_emb, cov).numpy()

    fb_auroc = roc_auc_score(fb_ood_mask.astype(int), fb_unc)
    print(f"FB15k-237 AUROC: {fb_auroc:.4f}")

    # ========================================
    # Evaluate on WN18RR (cross-domain transfer)
    # ========================================
    print("\n--- Evaluating on WN18RR (cross-domain transfer) ---")
    wn_ent_emb.eval()
    wn_rel_emb.eval()

    # Build coverage for WN18RR
    wn_coverage_set = set()
    for h, r, t in wn_train:
        wn_coverage_set.add((int(h), int(r)))
        wn_coverage_set.add((int(t), int(r)))

    wn_ood_mask = np.array([
        (int(h), int(r)) not in wn_coverage_set or (int(t), int(r)) not in wn_coverage_set
        for h, r, t in wn_test
    ])

    with torch.no_grad():
        h = torch.tensor(wn_test[:, 0])
        r = torch.tensor(wn_test[:, 1])
        t = torch.tensor(wn_test[:, 2])

        h_emb = wn_ent_emb(h)
        r_emb = wn_rel_emb(r)
        t_emb = wn_ent_emb(t)

        cov = torch.tensor([
            1.0 if ((int(hh), int(rr)) in wn_coverage_set and (int(tt), int(rr)) in wn_coverage_set) else 0.0
            for hh, rr, tt in wn_test
        ])

        wn_unc = unc_mlp.get_uncertainty(h_emb, r_emb, t_emb, cov).numpy()

    wn_auroc = roc_auc_score(wn_ood_mask.astype(int), wn_unc)
    print(f"WN18RR AUROC (transfer): {wn_auroc:.4f}")

    # ========================================
    # Results
    # ========================================
    print("\n" + "="*70)
    print("CROSS-DATASET TRANSFER RESULTS")
    print("="*70)
    print(f"FB15k-237 (in-domain):    AUROC = {fb_auroc:.4f}")
    print(f"WN18RR (cross-domain):    AUROC = {wn_auroc:.4f}")
    print(f"Transfer gap:             {fb_auroc - wn_auroc:.4f}")

    if wn_auroc > 0.7:
        print("\nCONCLUSION: Uncertainty MLP transfers across datasets!")
    else:
        print("\nCONCLUSION: Limited transfer - MLP is dataset-specific.")


if __name__ == "__main__":
    main()
