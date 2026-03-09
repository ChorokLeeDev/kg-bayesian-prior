#!/usr/bin/env python3
"""
RelCondVar on ICEWS14 - Test if relation-conditioned variance helps on non-circular benchmark.

Hypothesis:
- Entity-level variance σ²(e) doesn't help (Theorem 1)
- Relation-conditioned variance σ²(e,r) MIGHT help because it encodes coverage implicitly

Expected:
- If σ²(e,r) correlates with (entity, relation) co-occurrence → could help
- On role-shift OOD, entities have coverage but unusual relation → σ²(e,r) should be high
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import argparse

def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples


class RelCondVar(nn.Module):
    """Relation-Conditioned Variance model."""

    def __init__(self, num_entities, num_relations, dim=100, hidden_dim=64):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity embeddings
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Variance network: σ²(e, r) = MLP([e; r])
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(self, h, r, t):
        """DistMult scoring."""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_variance(self, e, r):
        """Get σ²(e, r) - relation-conditioned variance."""
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        return self.var_net(combined).squeeze(-1)

    def get_triple_uncertainty(self, h, r, t):
        """Uncertainty for a triple = avg of σ²(h,r) and σ²(t,r)."""
        var_h = self.get_variance(h, r)
        var_t = self.get_variance(t, r)
        return (var_h + var_t) / 2


def train_model(model, train_triples, num_entities, epochs=30, batch_size=1024,
                lr=1e-3, aux_weight=0.1, device='cpu'):
    """Train RelCondVar with auxiliary variance objective."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Build (entity, relation) co-occurrence counts
    er_counts = defaultdict(int)
    for h, r, t in train_triples:
        er_counts[(h, r)] += 1
        er_counts[(t, r)] += 1

    # Normalize counts to [0, 1] for auxiliary objective
    max_count = max(er_counts.values()) if er_counts else 1

    train_tensor = torch.LongTensor(train_triples).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        perm = torch.randperm(len(train_tensor))
        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[perm[i:i+batch_size]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Positive scores
            pos_scores = model(h, r, t)

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            # BCE loss
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Auxiliary objective: variance should correlate with inverse frequency
            # High frequency (e, r) → low variance
            # Low frequency (e, r) → high variance
            var_h = model.get_variance(h, r)
            var_t = model.get_variance(t, r)

            # Target: 1 - normalized_count (more frequent = lower target variance)
            target_h = torch.tensor([1.0 - er_counts[(h_i.item(), r_i.item())] / max_count
                                     for h_i, r_i in zip(h, r)], device=device)
            target_t = torch.tensor([1.0 - er_counts[(t_i.item(), r_i.item())] / max_count
                                     for t_i, r_i in zip(t, r)], device=device)

            # MSE loss for variance alignment
            aux_loss = ((var_h - target_h) ** 2).mean() + ((var_t - target_t) ** 2).mean()

            loss = pos_loss + neg_loss + aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}", flush=True)

    return model


def create_role_shift_split(train_triples, test_triples, top_pct=20):
    """
    Role-shift OOD: entities using relations atypical for their profile.

    - Find entities' typical relation distribution
    - OOD = test triples where entity uses atypical relation
    - ρ > 0 because entity HAS coverage for that relation (just rare)
    """
    # Build entity-relation frequency profile
    entity_rel_freq = defaultdict(lambda: defaultdict(int))
    entity_total = defaultdict(int)

    for h, r, t in train_triples:
        entity_rel_freq[h][r] += 1
        entity_rel_freq[t][r] += 1
        entity_total[h] += 1
        entity_total[t] += 1

    # Coverage: which (entity, relation) pairs exist
    coverage = defaultdict(set)
    for h, r, t in train_triples:
        coverage[h].add(r)
        coverage[t].add(r)

    # For each entity, find "typical" relations (top 80% of its usage)
    entity_typical = {}
    for e, rel_counts in entity_rel_freq.items():
        total = sum(rel_counts.values())
        sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
        cumsum = 0
        typical = set()
        for rel, cnt in sorted_rels:
            cumsum += cnt
            typical.add(rel)
            if cumsum >= 0.8 * total:
                break
        entity_typical[e] = typical

    # Categorize test triples
    role_shift_ood = []  # Entity using atypical relation (but has coverage)
    standard_ood = []    # Novel context (no coverage)
    id_triples = []      # Normal

    for h, r, t in test_triples:
        h_covered = r in coverage.get(h, set())
        t_covered = r in coverage.get(t, set())

        h_typical = r in entity_typical.get(h, set())
        t_typical = r in entity_typical.get(t, set())

        if not (h_covered and t_covered):
            standard_ood.append((h, r, t))
        elif not h_typical or not t_typical:
            # Has coverage but atypical for at least one entity
            role_shift_ood.append((h, r, t))
        else:
            id_triples.append((h, r, t))

    print(f"\nSplit statistics:", flush=True)
    print(f"  Role-shift OOD (covered but atypical): {len(role_shift_ood)}", flush=True)
    print(f"  Standard OOD (no coverage): {len(standard_ood)}", flush=True)
    print(f"  ID: {len(id_triples)}", flush=True)

    # Compute ρ for role-shift
    rho = 1.0  # By construction, all role-shift have coverage
    print(f"  ρ (coverage overlap) for role-shift: {rho:.3f}", flush=True)

    return {
        'role_shift_ood': role_shift_ood,
        'standard_ood': standard_ood,
        'id': id_triples,
        'coverage': coverage
    }


def evaluate_ood(model, split_data, device='cpu'):
    """Evaluate RelCondVar on role-shift OOD."""
    model.eval()

    role_shift_ood = split_data['role_shift_ood']
    id_triples = split_data['id']
    coverage = split_data['coverage']

    if len(role_shift_ood) == 0:
        print("No role-shift OOD triples!", flush=True)
        return None

    def get_uncertainties(triples):
        u_relcond = []
        u_str = []

        with torch.no_grad():
            for h, r, t in triples:
                h_t = torch.LongTensor([h]).to(device)
                r_t = torch.LongTensor([r]).to(device)
                t_t = torch.LongTensor([t]).to(device)

                # RelCondVar uncertainty
                var = model.get_triple_uncertainty(h_t, r_t, t_t).item()
                u_relcond.append(var)

                # Structural (coverage)
                h_cov = 1 if r in coverage.get(h, set()) else 0
                t_cov = 1 if r in coverage.get(t, set()) else 0
                struct = 2 - h_cov - t_cov
                u_str.append(struct)

        return np.array(u_relcond), np.array(u_str)

    # Get uncertainties
    relcond_ood, str_ood = get_uncertainties(role_shift_ood)
    relcond_id, str_id = get_uncertainties(id_triples)

    # Labels: 1=OOD, 0=ID
    labels = np.concatenate([np.ones(len(role_shift_ood)), np.zeros(len(id_triples))])

    all_relcond = np.concatenate([relcond_ood, relcond_id])
    all_str = np.concatenate([str_ood, str_id])

    # AUROC
    relcond_auroc = roc_auc_score(labels, all_relcond)
    str_auroc = roc_auc_score(labels, all_str)
    cagp_auroc = roc_auc_score(labels, 0.5 * all_relcond + 0.5 * all_str)

    print(f"\n--- Role-Shift OOD (KEY RESULT, n={len(role_shift_ood)}) ---", flush=True)
    print(f"  ρ (coverage overlap): 1.000", flush=True)
    print(f"  RelCondVar AUROC:  {relcond_auroc:.3f}", flush=True)
    print(f"  Structural AUROC:  {str_auroc:.3f}", flush=True)
    print(f"  Combined AUROC:    {cagp_auroc:.3f}", flush=True)
    print(f"  ** RelCondVar - Structural: {relcond_auroc - str_auroc:+.3f} **", flush=True)

    # Variance analysis
    print(f"\n  OOD variance mean: {relcond_ood.mean():.4f} ± {relcond_ood.std():.4f}", flush=True)
    print(f"  ID variance mean:  {relcond_id.mean():.4f} ± {relcond_id.std():.4f}", flush=True)
    print(f"  Ratio: {relcond_ood.mean() / relcond_id.mean():.2f}x", flush=True)

    return {
        'relcond_auroc': relcond_auroc,
        'str_auroc': str_auroc,
        'cagp_auroc': cagp_auroc,
        'gap': relcond_auroc - str_auroc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw/icews14')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--aux_weight', type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("RelCondVar on ICEWS14 - Role-Shift OOD", flush=True)
    print("=" * 60, flush=True)
    print(f"Hypothesis: σ²(e,r) helps where σ²(e) fails", flush=True)
    print("=" * 60, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    train_triples = load_triples(f"{args.data_dir}/train.txt")
    test_triples = load_triples(f"{args.data_dir}/test.txt")

    all_triples = train_triples + test_triples
    entities = set()
    relations = set()
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    num_entities = max(entities) + 1
    num_relations = max(relations) + 1

    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}", flush=True)
    print(f"Entities: {num_entities}, Relations: {num_relations}", flush=True)

    # Create split
    print("\n" + "=" * 60, flush=True)
    print("Creating role-shift split...", flush=True)
    split_data = create_role_shift_split(train_triples, test_triples)

    if len(split_data['role_shift_ood']) == 0:
        print("ERROR: No role-shift OOD triples found.", flush=True)
        return

    # Run experiments
    all_results = []

    for seed in range(args.seeds):
        print(f"\n{'=' * 60}", flush=True)
        print(f"Seed {seed+1}/{args.seeds}", flush=True)
        print("=" * 60, flush=True)

        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)

        # Train model
        print("\nTraining RelCondVar...", flush=True)
        model = RelCondVar(num_entities, num_relations, args.dim)
        model = train_model(model, train_triples, num_entities,
                           epochs=args.epochs, aux_weight=args.aux_weight,
                           device=args.device)

        # Evaluate
        results = evaluate_ood(model, split_data, args.device)
        if results:
            all_results.append(results)

    # Aggregate
    if all_results:
        print(f"\n{'=' * 60}", flush=True)
        print("AGGREGATE RESULTS", flush=True)
        print("=" * 60, flush=True)

        relcond_aurocs = [r['relcond_auroc'] for r in all_results]
        str_aurocs = [r['str_auroc'] for r in all_results]
        gaps = [r['gap'] for r in all_results]

        print(f"\nRole-Shift OOD ({args.seeds} seeds):", flush=True)
        print(f"  RelCondVar: {np.mean(relcond_aurocs):.3f} ± {np.std(relcond_aurocs):.3f}", flush=True)
        print(f"  Structural: {np.mean(str_aurocs):.3f} ± {np.std(str_aurocs):.3f}", flush=True)
        print(f"  Gap:        {np.mean(gaps):+.3f} ± {np.std(gaps):.3f}", flush=True)

        if np.mean(gaps) > 0.05:
            print(f"\n*** SUCCESS: RelCondVar beats structural by {np.mean(gaps)*100:+.1f}pp! ***", flush=True)
        elif np.mean(gaps) > 0:
            print(f"\n*** MARGINAL: RelCondVar slightly better ({np.mean(gaps)*100:+.1f}pp) ***", flush=True)
        else:
            print(f"\n*** RelCondVar did not beat structural ({np.mean(gaps)*100:+.1f}pp) ***", flush=True)


if __name__ == "__main__":
    main()
