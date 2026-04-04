#!/usr/bin/env python3
"""
BEUrRE baseline implementation for comparison.
BEUrRE: Box Embeddings for Uncertain Relational data with Evidence.

Uses box embeddings where uncertainty = box volume.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_fb15k237, load_wn18rr
from src.models.relation_conditioned import RCUE, train_rcue


class BEUrRE(nn.Module):
    """
    Simplified BEUrRE-style box embedding model.

    Each entity is a box: (center, offset)
    Uncertainty = box volume = prod(2 * offset)
    """

    def __init__(self, num_entities, num_relations, dim=100, min_offset=0.01):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.min_offset = min_offset

        # Entity boxes: center and offset (half-width)
        self.entity_center = nn.Embedding(num_entities, dim)
        self.entity_offset = nn.Embedding(num_entities, dim)

        # Relation transformations
        self.relation_center = nn.Embedding(num_relations, dim)
        self.relation_offset = nn.Embedding(num_relations, dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_center.weight)
        nn.init.uniform_(self.entity_offset.weight, 0.0, 0.5)
        nn.init.xavier_uniform_(self.relation_center.weight)
        nn.init.uniform_(self.relation_offset.weight, 0.0, 0.3)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0

    def get_box(self, entity_ids, relation_ids=None):
        """Get box for entities, optionally transformed by relation."""
        center = self.entity_center(entity_ids)
        offset = F.softplus(self.entity_offset(entity_ids)) + self.min_offset

        if relation_ids is not None:
            # Transform box by relation
            r_center = self.relation_center(relation_ids)
            r_offset = F.softplus(self.relation_offset(relation_ids))
            center = center + r_center
            offset = offset * (1 + r_offset)

        return center, offset

    def forward(self, h, r, t):
        """Score = negative distance between transformed head box and tail box."""
        h_center, h_offset = self.get_box(h, r)
        t_center, t_offset = self.get_box(t)

        # Distance between box centers
        dist = torch.abs(h_center - t_center)

        # Penalize if outside box
        outside = F.relu(dist - h_offset - t_offset)

        # Score = negative distance (higher is better)
        score = -outside.sum(dim=-1)

        return score

    def get_uncertainty(self, h, r, t):
        """
        Uncertainty = sum of box volumes for head and tail.
        Volume ~ prod(offset), but we use sum(log(offset)) for stability.
        """
        _, h_offset = self.get_box(h, r)
        _, t_offset = self.get_box(t, r)

        # Log volume (sum of log offsets)
        h_log_vol = torch.log(h_offset + 1e-8).sum(dim=-1)
        t_log_vol = torch.log(t_offset + 1e-8).sum(dim=-1)

        return h_log_vol + t_log_vol


def train_beurre(model, train_triples, device, epochs=30, lr=1e-3):
    """Train BEUrRE model."""
    model = model.to(device)
    model.precompute_coverage(train_triples)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h_all = torch.tensor(train_triples[:, 0], dtype=torch.long)
    r_all = torch.tensor(train_triples[:, 1], dtype=torch.long)
    t_all = torch.tensor(train_triples[:, 2], dtype=torch.long)

    loader = DataLoader(TensorDataset(h_all, r_all, t_all), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)

            pos_scores = model(h, r, t)
            neg_scores = model(h, r, neg_t)

            loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=1.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    for ds_name, load_fn in [('FB15k-237', load_fb15k237), ('WN18RR', load_wn18rr)]:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        train_ds, _, test_ds = load_fn()
        train = train_ds.triples
        test = test_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations

        # Coverage
        coverage = set()
        for h, r, t in train:
            coverage.add((h, r))
            coverage.add((t, r))

        ood_mask = np.array([
            (h, r) not in coverage or (t, r) not in coverage
            for h, r, t in test
        ])

        print(f"OOD fraction: {ood_mask.mean()*100:.1f}%")

        h_t = torch.tensor(test[:, 0], device=device)
        r_t = torch.tensor(test[:, 1], device=device)
        t_t = torch.tensor(test[:, 2], device=device)

        results = {}

        # BEUrRE
        print("\n--- BEUrRE ---")
        torch.manual_seed(42)
        beurre = BEUrRE(n_ent, n_rel)
        beurre = train_beurre(beurre, train, device, epochs=30)

        beurre.eval()
        with torch.no_grad():
            unc = beurre.get_uncertainty(h_t, r_t, t_t).cpu().numpy()
        auroc = roc_auc_score(ood_mask, unc)
        results['BEUrRE'] = auroc
        print(f"  OOD AUROC: {auroc:.4f}")

        # RCUE for comparison
        print("\n--- RCUE ---")
        torch.manual_seed(42)
        rcue = RCUE(n_ent, n_rel, use_coverage=True)
        rcue = train_rcue(rcue, train, device, epochs=30, verbose=True)

        rcue.eval()
        with torch.no_grad():
            unc = rcue.get_uncertainty(h_t, r_t, t_t).cpu().numpy()
        auroc = roc_auc_score(ood_mask, unc)
        results['RCUE'] = auroc
        print(f"  OOD AUROC: {auroc:.4f}")

        # Summary
        print(f"\n{ds_name} Summary:")
        for name, auroc in results.items():
            print(f"  {name}: {auroc:.4f}")


if __name__ == "__main__":
    main()
