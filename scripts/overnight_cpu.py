#!/usr/bin/env python3
"""
Overnight CPU experiment for YAGO3-10 and ICEWS14.
Uses smaller batch size and optimizations for CPU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

print(f"Using CPU overnight mode")
print(f"PyTorch: {torch.__version__}")

# Force CPU
device = torch.device('cpu')

def load_yago310_efficient():
    """Load YAGO3-10 efficiently."""
    base_path = "data/raw/yago3-10"

    def load_triples(split):
        triples = []
        with open(f"{base_path}/{split}2id.txt") as f:
            n = int(f.readline().strip())  # First line is count
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
        return np.array(triples)

    train = load_triples("train")
    test = load_triples("test")

    # Get max entity and relation
    all_triples = np.vstack([train, test])
    n_ent = all_triples[:, [0, 2]].max() + 1
    n_rel = all_triples[:, 1].max() + 1

    return train, test, n_ent, n_rel

def load_icews14_efficient():
    """Load ICEWS14 efficiently."""
    base_path = "data/raw/icews14"

    entity2id = {}
    relation2id = {}

    def load_triples(split):
        triples = []
        with open(f"{base_path}/{split}.txt") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                h, r, t = parts[:3]
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return np.array(triples)

    train = load_triples("train")
    test = load_triples("test")

    return train, test, len(entity2id), len(relation2id)

class LightRCUE(torch.nn.Module):
    """Lightweight RCUE for CPU training."""

    def __init__(self, num_entities, num_relations, dim=50):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_emb = torch.nn.Embedding(num_entities, dim)
        self.relation_emb = torch.nn.Embedding(num_relations, dim)

        self.unc_net = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Softplus()
        )

        self.boost_logit = torch.nn.Parameter(torch.tensor(0.7))

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def precompute_coverage(self, triples):
        for h, r, t in triples:
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_entity_var(self, e, r):
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        base_var = self.unc_net(torch.cat([e_emb, r_emb], dim=-1)).squeeze(-1)
        cov = self.coverage[e, r]
        k = torch.exp(self.boost_logit)
        boost = 1.0 + k * (1.0 - cov)
        return base_var * boost

    def get_uncertainty(self, h, r, t):
        return self.get_entity_var(h, r) + self.get_entity_var(t, r)

def train_light(model, train, device, epochs=10, batch_size=512):
    """Train with smaller batch for CPU."""
    model = model.to(device)
    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    h_all = torch.tensor(train[:, 0], dtype=torch.long)
    r_all = torch.tensor(train[:, 1], dtype=torch.long)
    t_all = torch.tensor(train[:, 2], dtype=torch.long)

    n = len(train)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(n)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            h = h_all[batch_idx].to(device)
            r = r_all[batch_idx].to(device)
            t = t_all[batch_idx].to(device)

            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)

            pos_scores = model(h, r, t)
            neg_scores = model(h, r, neg_t)

            score_loss = torch.nn.functional.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=1.0
            )

            pos_unc = model.get_uncertainty(h, r, t)
            neg_unc = model.get_uncertainty(h, r, neg_t)
            unc_loss = torch.nn.functional.relu(pos_unc - neg_unc + 0.1).mean()

            loss = score_loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

class EnergyBaseline(torch.nn.Module):
    """Simple energy baseline."""

    def __init__(self, num_entities, num_relations, dim=50):
        super().__init__()
        self.num_entities = num_entities
        self.entity_emb = torch.nn.Embedding(num_entities, dim)
        self.relation_emb = torch.nn.Embedding(num_relations, dim)
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_uncertainty(self, h, r, t):
        return -self.forward(h, r, t)  # Negative score as uncertainty

def run_dataset(name, load_fn, epochs, seed=42):
    """Run single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train, test, n_ent, n_rel = load_fn()
    print(f"  Entities: {n_ent}, Relations: {n_rel}")
    print(f"  Train: {len(train)}, Test: {len(test)}")

    # Build coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])
    print(f"  OOD fraction: {ood_mask.mean():.1%}")

    # Energy baseline
    print("\n--- Energy Baseline ---")
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_light(energy, train, device, epochs=epochs, batch_size=512)

    energy.eval()
    h = torch.tensor(test[:, 0])
    r = torch.tensor(test[:, 1])
    t = torch.tensor(test[:, 2])

    with torch.no_grad():
        unc_energy = energy.get_uncertainty(h, r, t).numpy()
    auroc_energy = roc_auc_score(ood_mask, unc_energy)
    print(f"  Energy AUROC: {auroc_energy:.4f}")

    # RCUE
    print("\n--- RCUE ---")
    torch.manual_seed(seed)
    rcue = LightRCUE(n_ent, n_rel)
    rcue = train_light(rcue, train, device, epochs=epochs, batch_size=512)

    rcue.eval()
    with torch.no_grad():
        unc_rcue = rcue.get_uncertainty(h, r, t).numpy()
    auroc_rcue = roc_auc_score(ood_mask, unc_rcue)
    print(f"  RCUE AUROC: {auroc_rcue:.4f}")

    return auroc_energy, auroc_rcue

def main():
    print("="*60)
    print("OVERNIGHT CPU EXPERIMENT")
    print("YAGO3-10 and ICEWS14")
    print("="*60)

    results = {}

    # ICEWS14 first (smaller)
    e, r = run_dataset("ICEWS14", load_icews14_efficient, epochs=15)
    results["ICEWS14"] = (e, r)

    # YAGO3-10 (larger)
    e, r = run_dataset("YAGO3-10", load_yago310_efficient, epochs=10)
    results["YAGO3-10"] = (e, r)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, (energy, rcue) in results.items():
        print(f"{name}: Energy={energy:.4f}, RCUE={rcue:.4f}, Δ=+{rcue-energy:.4f}")

if __name__ == "__main__":
    main()
