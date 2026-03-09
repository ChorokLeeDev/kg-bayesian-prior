#!/usr/bin/env python3
"""
Elegant Coverage Solutions - Phase 1 Quick Experiments

Goal: Find a way to learn coverage-aware uncertainty WITHOUT explicit hash table.

Experiments:
1. Coverage Reconstruction Loss: Train embedding to predict which relations entity has seen
2. Relation-Set Encoding: Encode entity by its relation vocabulary (NodePiece-style)
3. Contrastive Coverage: Learn to distinguish seen vs unseen (e, r) pairs

Success Criteria: AUROC > 0.70 on role-shift OOD without explicit coverage supervision
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def build_entity_relation_sets(triples):
    """Build mapping from entity to set of relations it appears with."""
    entity_relations = defaultdict(set)
    for h, r, t in triples:
        entity_relations[h].add(r)
        entity_relations[t].add(r)
    return entity_relations


def create_role_shift_split(train_triples, test_triples, top_pct=0.8):
    """Create role-shift OOD split."""
    entity_rel_freq = defaultdict(lambda: defaultdict(int))
    for h, r, t in train_triples:
        entity_rel_freq[h][r] += 1
        entity_rel_freq[t][r] += 1

    coverage = defaultdict(set)
    for h, r, t in train_triples:
        coverage[h].add(r)
        coverage[t].add(r)

    entity_typical = {}
    for e, rel_counts in entity_rel_freq.items():
        total = sum(rel_counts.values())
        sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
        cumsum = 0
        typical = set()
        for rel, cnt in sorted_rels:
            cumsum += cnt
            typical.add(rel)
            if cumsum >= top_pct * total:
                break
        entity_typical[e] = typical

    role_shift_ood = []
    id_triples = []

    for h, r, t in test_triples:
        h_covered = r in coverage.get(h, set())
        t_covered = r in coverage.get(t, set())
        h_typical = r in entity_typical.get(h, set())
        t_typical = r in entity_typical.get(t, set())

        if not (h_covered and t_covered):
            continue
        elif not h_typical or not t_typical:
            role_shift_ood.append((h, r, t))
        else:
            id_triples.append((h, r, t))

    # Calculate rho
    rho = len([1 for h, r, t in role_shift_ood
               if r in coverage.get(h, set()) and r in coverage.get(t, set())]) / max(len(role_shift_ood), 1)

    return {
        'role_shift_ood': role_shift_ood,
        'id': id_triples,
        'coverage': coverage,
        'entity_relations': build_entity_relation_sets(train_triples),
        'rho': rho
    }


# =============================================================================
# Approach 1: Coverage Reconstruction Loss
# =============================================================================

class CoverageReconstructionModel(nn.Module):
    """
    Learn embedding that can reconstruct which relations each entity has seen.

    Hypothesis: If φ(e) must predict relations_of(e), it will encode coverage info.
    Then σ²(e, r) can use this to detect unseen relations.
    """
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage reconstruction head: predict which relations entity has
        self.coverage_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_relations)
        )

        # Uncertainty head: combine entity + relation for variance
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def predict_coverage(self, e):
        """Predict which relations entity e has been seen with."""
        e_emb = self.entity_emb(e)
        return self.coverage_predictor(e_emb)

    def get_variance(self, e, r):
        """Get variance for (entity, relation) pair."""
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        return self.var_net(combined).squeeze(-1)

    def get_triple_uncertainty(self, h, r, t):
        var_h = self.get_variance(h, r)
        var_t = self.get_variance(t, r)
        return (var_h + var_t) / 2


def train_coverage_reconstruction(model, train_triples, entity_relations,
                                   num_entities, num_relations, epochs=30,
                                   coverage_weight=0.1, device='cpu'):
    """Train with coverage reconstruction loss."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    # Build coverage targets
    coverage_targets = torch.zeros(num_entities, num_relations, device=device)
    for e, rels in entity_relations.items():
        for r in rels:
            if e < num_entities and r < num_relations:
                coverage_targets[e, r] = 1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0
        total_link_loss = 0
        total_cov_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Link prediction loss
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            link_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            link_loss += -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Coverage reconstruction loss
            unique_entities = torch.unique(torch.cat([h, t]))
            cov_pred = model.predict_coverage(unique_entities)
            cov_target = coverage_targets[unique_entities]
            cov_loss = F.binary_cross_entropy_with_logits(cov_pred, cov_target)

            # Uncertainty margin loss (variance should be higher for neg samples)
            var_pos = model.get_triple_uncertainty(h, r, t)
            var_neg = model.get_triple_uncertainty(h, r, neg_t)
            margin_loss = F.relu(var_pos - var_neg + 0.1).mean()

            loss = link_loss + coverage_weight * cov_loss + 0.1 * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_link_loss += link_loss.item()
            total_cov_loss += cov_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f} "
                  f"(Link: {total_link_loss:.4f}, Cov: {total_cov_loss:.4f})")

    return model


# =============================================================================
# Approach 2: Relation-Set Encoding (NodePiece-style)
# =============================================================================

class RelationSetModel(nn.Module):
    """
    Encode entity by its relation vocabulary.

    Entity representation includes info about which relations it has seen.
    Uncertainty for (e, r) depends on whether r is in entity's relation set.
    """
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Relation set embedding: each entity gets a learned "relation mask"
        self.relation_mask = nn.Embedding(num_entities, num_relations)
        nn.init.zeros_(self.relation_mask.weight)  # Start with no relations

        # Combine entity + relation set for final representation
        self.entity_combiner = nn.Linear(dim + num_relations, dim)

        # Variance network
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),  # +1 for relation membership score
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

    def get_entity_repr(self, e):
        """Get entity representation including relation set info."""
        e_emb = self.entity_emb(e)
        rel_mask = torch.sigmoid(self.relation_mask(e))  # Soft relation membership
        return self.entity_combiner(torch.cat([e_emb, rel_mask], dim=-1))

    def forward(self, h, r, t):
        h_repr = self.get_entity_repr(h)
        r_emb = self.relation_emb(r)
        t_repr = self.get_entity_repr(t)
        return (h_repr * r_emb * t_repr).sum(dim=-1)

    def get_relation_membership(self, e, r):
        """How strongly is relation r in entity e's relation set?"""
        rel_mask = torch.sigmoid(self.relation_mask(e))  # [batch, num_relations]
        # Gather the score for the specific relation
        r_expanded = r.unsqueeze(-1)  # [batch, 1]
        membership = torch.gather(rel_mask, 1, r_expanded).squeeze(-1)
        return membership

    def get_variance(self, e, r):
        """Variance based on entity, relation, and membership score."""
        e_repr = self.get_entity_repr(e)
        r_emb = self.relation_emb(r)
        membership = self.get_relation_membership(e, r).unsqueeze(-1)
        combined = torch.cat([e_repr, r_emb, membership], dim=-1)
        return self.var_net(combined).squeeze(-1)

    def get_triple_uncertainty(self, h, r, t):
        var_h = self.get_variance(h, r)
        var_t = self.get_variance(t, r)
        return (var_h + var_t) / 2


def train_relation_set(model, train_triples, entity_relations,
                       num_entities, num_relations, epochs=30, device='cpu'):
    """Train relation set model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    # Build relation membership targets
    membership_targets = torch.zeros(num_entities, num_relations, device=device)
    for e, rels in entity_relations.items():
        for r in rels:
            if e < num_entities and r < num_relations:
                membership_targets[e, r] = 1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Link prediction loss
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            link_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            link_loss += -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Relation membership loss
            unique_entities = torch.unique(torch.cat([h, t]))
            pred_mask = torch.sigmoid(model.relation_mask(unique_entities))
            target_mask = membership_targets[unique_entities]
            membership_loss = F.binary_cross_entropy(pred_mask, target_mask)

            # Variance should be higher when relation NOT in entity's set
            var_pos = model.get_triple_uncertainty(h, r, t)
            var_neg = model.get_triple_uncertainty(h, r, neg_t)
            margin_loss = F.relu(var_pos - var_neg + 0.1).mean()

            loss = link_loss + 0.1 * membership_loss + 0.1 * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


# =============================================================================
# Approach 3: Contrastive Coverage Learning
# =============================================================================

class ContrastiveCoverageModel(nn.Module):
    """
    Learn to distinguish seen vs unseen (e, r) pairs via contrastive learning.
    """
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage score network
        self.coverage_scorer = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # Variance network
        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def coverage_score(self, e, r):
        """Score for whether (e, r) pair was seen in training."""
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        return self.coverage_scorer(combined).squeeze(-1)

    def get_variance(self, e, r):
        """Variance inversely related to coverage score."""
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        base_var = self.var_net(combined).squeeze(-1)
        # Modulate by inverse coverage score
        cov_score = torch.sigmoid(self.coverage_score(e, r))
        return base_var * (2 - cov_score)  # Higher variance when low coverage

    def get_triple_uncertainty(self, h, r, t):
        var_h = self.get_variance(h, r)
        var_t = self.get_variance(t, r)
        return (var_h + var_t) / 2


def train_contrastive_coverage(model, train_triples, entity_relations,
                                num_entities, num_relations, epochs=30, device='cpu'):
    """Train with contrastive coverage loss."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    all_relations = set(range(num_relations))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Link prediction loss
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            link_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            link_loss += -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Contrastive coverage loss
            # Positive: (h, r) pairs from training (these are seen)
            pos_coverage = model.coverage_score(h, r)

            # Negative: (h, r_unseen) pairs where entity h hasn't seen relation r_unseen
            neg_r = []
            for h_i in h:
                seen_rels = entity_relations.get(h_i.item(), set())
                unseen_rels = list(all_relations - seen_rels)
                if unseen_rels:
                    neg_r.append(np.random.choice(unseen_rels))
                else:
                    neg_r.append(r[0].item())
            neg_r = torch.LongTensor(neg_r).to(device)
            neg_coverage = model.coverage_score(h, neg_r)

            # Contrastive loss: positive coverage > negative coverage
            contrastive_loss = F.relu(neg_coverage - pos_coverage + 1.0).mean()

            loss = link_loss + 0.1 * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, split_data, device='cpu'):
    """Evaluate on role-shift OOD."""
    model.eval()
    role_shift_ood = split_data['role_shift_ood']
    id_triples = split_data['id']

    if len(role_shift_ood) == 0 or len(id_triples) == 0:
        return None

    def get_uncertainties(triples):
        uncertainties = []
        with torch.no_grad():
            for h, r, t in triples:
                h_t = torch.LongTensor([h]).to(device)
                r_t = torch.LongTensor([r]).to(device)
                t_t = torch.LongTensor([t]).to(device)
                var = model.get_triple_uncertainty(h_t, r_t, t_t).item()
                uncertainties.append(var)
        return np.array(uncertainties)

    ood_uncertainties = get_uncertainties(role_shift_ood)
    id_uncertainties = get_uncertainties(id_triples)

    labels = np.concatenate([np.ones(len(role_shift_ood)), np.zeros(len(id_triples))])
    scores = np.concatenate([ood_uncertainties, id_uncertainties])

    return roc_auc_score(labels, scores)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, required=True,
                        choices=['reconstruction', 'relation_set', 'contrastive'])
    parser.add_argument('--data_dir', default='data/raw/icews14')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Elegant Coverage Experiment: {args.approach}")
    print("=" * 60)

    # Load data
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

    print(f"Entities: {num_entities}, Relations: {num_relations}")

    # Create split
    split_data = create_role_shift_split(train_triples, test_triples)
    print(f"Role-shift OOD: {len(split_data['role_shift_ood'])}, ID: {len(split_data['id'])}")
    print(f"Coverage overlap (ρ): {split_data['rho']:.3f}")

    # Run experiments
    results = []
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed+1}/{args.seeds} ---")
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)

        if args.approach == 'reconstruction':
            model = CoverageReconstructionModel(num_entities, num_relations)
            model = train_coverage_reconstruction(
                model, train_triples, split_data['entity_relations'],
                num_entities, num_relations, epochs=args.epochs, device=args.device
            )
        elif args.approach == 'relation_set':
            model = RelationSetModel(num_entities, num_relations)
            model = train_relation_set(
                model, train_triples, split_data['entity_relations'],
                num_entities, num_relations, epochs=args.epochs, device=args.device
            )
        elif args.approach == 'contrastive':
            model = ContrastiveCoverageModel(num_entities, num_relations)
            model = train_contrastive_coverage(
                model, train_triples, split_data['entity_relations'],
                num_entities, num_relations, epochs=args.epochs, device=args.device
            )

        auroc = evaluate(model, split_data, args.device)
        if auroc is not None:
            results.append(auroc)
            print(f"  Role-shift OOD AUROC: {auroc:.3f}")

    print("\n" + "=" * 60)
    print(f"RESULTS - {args.approach}")
    print("=" * 60)
    print(f"AUROC: {np.mean(results):.3f} ± {np.std(results):.3f}")

    if np.mean(results) > 0.70:
        print("✓ SUCCESS: >0.70 AUROC achieved!")
    elif np.mean(results) > 0.60:
        print("~ PROMISING: >0.60 AUROC, needs tuning")
    else:
        print("✗ FAILED: No better than baseline")


if __name__ == "__main__":
    main()
