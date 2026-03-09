#!/usr/bin/env python3
"""
RelCondVar Ablations - Testing different approaches to escape the circularity problem.

Approaches:
A. Binary coverage aux (seen/unseen, not frequency)
B. Contrastive loss (positive vs negative relations)
C. Reconstruction loss (predict relation from entity+variance)
D. KL divergence (variance should diverge from prior for rare pairs)
E. Long training (100 epochs, no aux, just BCE + margin)
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


class RelCondVarAblation(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100, hidden_dim=64):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.var_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # For reconstruction approach
        self.relation_predictor = nn.Linear(dim + 1, num_relations)  # entity + variance -> relation

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_variance(self, e, r):
        e_emb = self.entity_emb(e)
        r_emb = self.relation_emb(r)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        return self.var_net(combined).squeeze(-1)

    def get_triple_uncertainty(self, h, r, t):
        var_h = self.get_variance(h, r)
        var_t = self.get_variance(t, r)
        return (var_h + var_t) / 2

    def predict_relation(self, e, var):
        e_emb = self.entity_emb(e)
        combined = torch.cat([e_emb, var.unsqueeze(-1)], dim=-1)
        return self.relation_predictor(combined)


def create_role_shift_split(train_triples, test_triples):
    entity_rel_freq = defaultdict(lambda: defaultdict(int))
    entity_total = defaultdict(int)

    for h, r, t in train_triples:
        entity_rel_freq[h][r] += 1
        entity_rel_freq[t][r] += 1
        entity_total[h] += 1
        entity_total[t] += 1

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
            if cumsum >= 0.8 * total:
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
            continue  # Skip no-coverage cases
        elif not h_typical or not t_typical:
            role_shift_ood.append((h, r, t))
        else:
            id_triples.append((h, r, t))

    return {
        'role_shift_ood': role_shift_ood,
        'id': id_triples,
        'coverage': coverage,
        'entity_rel_freq': entity_rel_freq
    }


def train_approach_A(model, train_triples, num_entities, coverage, epochs=30, device='cpu'):
    """Approach A: Binary coverage aux (0 if seen, 1 if unseen)"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Binary coverage aux: 0 if seen, 1 if unseen (but all training pairs are seen!)
            # For negative samples, check if (neg_t, r) was seen
            var_neg = model.get_variance(neg_t, r)
            target_neg = torch.tensor([0.0 if r.item() in coverage.get(nt.item(), set()) else 1.0
                                       for nt, r in zip(neg_t, r)], device=device)
            aux_loss = F.mse_loss(var_neg, target_neg)

            loss = pos_loss + neg_loss + 0.1 * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_approach_B(model, train_triples, num_entities, num_relations, epochs=30, device='cpu'):
    """Approach B: Contrastive loss - low variance for seen relations, high for unseen"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Build entity -> relations mapping
    entity_relations = defaultdict(set)
    for h, r, t in train_triples:
        entity_relations[h].add(r)
        entity_relations[t].add(r)

    train_tensor = torch.LongTensor(train_triples).to(device)
    all_relations = set(range(num_relations))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Contrastive: variance for seen relation should be LOW
            var_pos = model.get_variance(h, r)

            # Sample unseen relations for each entity
            neg_r = []
            for h_i in h:
                seen = entity_relations.get(h_i.item(), set())
                unseen = list(all_relations - seen)
                if unseen:
                    neg_r.append(np.random.choice(unseen))
                else:
                    neg_r.append(r[0].item())  # fallback
            neg_r = torch.LongTensor(neg_r).to(device)
            var_neg_r = model.get_variance(h, neg_r)

            # Contrastive: var_pos should be < var_neg_r
            margin = 0.5
            contrastive_loss = F.relu(var_pos - var_neg_r + margin).mean()

            loss = pos_loss + neg_loss + 0.1 * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_approach_C(model, train_triples, num_entities, num_relations, epochs=30, device='cpu'):
    """Approach C: Reconstruction - predict relation from entity+variance"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Reconstruction: predict relation from entity + variance
            var_h = model.get_variance(h, r)
            rel_logits = model.predict_relation(h, var_h)
            recon_loss = F.cross_entropy(rel_logits, r)

            loss = pos_loss + neg_loss + 0.1 * recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_approach_D(model, train_triples, num_entities, epochs=30, device='cpu'):
    """Approach D: KL divergence - variance should stay close to prior unless informative"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    prior_var = 1.0  # Prior variance

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # KL: penalize deviation from prior (encourages variance to be meaningful)
            var_h = model.get_variance(h, r)
            var_t = model.get_variance(t, r)
            # KL(learned || prior) ≈ (var - prior)^2 / prior for Gaussians
            kl_loss = ((var_h - prior_var)**2 + (var_t - prior_var)**2).mean()

            loss = pos_loss + neg_loss + 0.01 * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_approach_E(model, train_triples, num_entities, epochs=100, device='cpu'):
    """Approach E: Long training with margin loss only (no aux)"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_tensor = torch.LongTensor(train_triples).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[perm[i:i+1024]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # Margin loss: negative samples should have higher variance
            var_pos = model.get_triple_uncertainty(h, r, t)
            var_neg = model.get_triple_uncertainty(h, r, neg_t)
            margin_loss = F.relu(var_pos - var_neg + 0.1).mean()

            loss = pos_loss + neg_loss + 0.1 * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}", flush=True)

    return model


def evaluate(model, split_data, device='cpu'):
    model.eval()
    role_shift_ood = split_data['role_shift_ood']
    id_triples = split_data['id']
    coverage = split_data['coverage']

    if len(role_shift_ood) == 0 or len(id_triples) == 0:
        return None

    def get_uncertainties(triples):
        u_relcond = []
        with torch.no_grad():
            for h, r, t in triples:
                h_t = torch.LongTensor([h]).to(device)
                r_t = torch.LongTensor([r]).to(device)
                t_t = torch.LongTensor([t]).to(device)
                var = model.get_triple_uncertainty(h_t, r_t, t_t).item()
                u_relcond.append(var)
        return np.array(u_relcond)

    relcond_ood = get_uncertainties(role_shift_ood)
    relcond_id = get_uncertainties(id_triples)

    labels = np.concatenate([np.ones(len(role_shift_ood)), np.zeros(len(id_triples))])
    all_relcond = np.concatenate([relcond_ood, relcond_id])

    auroc = roc_auc_score(labels, all_relcond)
    return auroc


def run_approach(approach, train_triples, test_triples, num_entities, num_relations,
                 split_data, seeds=3, device='cpu'):
    results = []

    for seed in range(seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)

        model = RelCondVarAblation(num_entities, num_relations)

        if approach == 'A':
            model = train_approach_A(model, train_triples, num_entities,
                                    split_data['coverage'], epochs=30, device=device)
        elif approach == 'B':
            model = train_approach_B(model, train_triples, num_entities, num_relations,
                                    epochs=30, device=device)
        elif approach == 'C':
            model = train_approach_C(model, train_triples, num_entities, num_relations,
                                    epochs=30, device=device)
        elif approach == 'D':
            model = train_approach_D(model, train_triples, num_entities,
                                    epochs=30, device=device)
        elif approach == 'E':
            model = train_approach_E(model, train_triples, num_entities,
                                    epochs=100, device=device)

        auroc = evaluate(model, split_data, device)
        if auroc is not None:
            results.append(auroc)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, required=True, choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--data_dir', default='data/raw/icews14')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f"=" * 60, flush=True)
    print(f"Approach {args.approach}: ", end="", flush=True)
    if args.approach == 'A':
        print("Binary coverage aux", flush=True)
    elif args.approach == 'B':
        print("Contrastive loss", flush=True)
    elif args.approach == 'C':
        print("Reconstruction loss", flush=True)
    elif args.approach == 'D':
        print("KL divergence regularization", flush=True)
    elif args.approach == 'E':
        print("Long training (100 epochs) with margin only", flush=True)
    print("=" * 60, flush=True)

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

    print(f"Entities: {num_entities}, Relations: {num_relations}", flush=True)

    # Create split
    split_data = create_role_shift_split(train_triples, test_triples)
    print(f"Role-shift OOD: {len(split_data['role_shift_ood'])}, ID: {len(split_data['id'])}", flush=True)

    # Run approach
    results = run_approach(args.approach, train_triples, test_triples,
                          num_entities, num_relations, split_data,
                          seeds=args.seeds, device=args.device)

    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULTS - Approach {args.approach}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"AUROC: {np.mean(results):.3f} ± {np.std(results):.3f}", flush=True)
    print(f"Gap vs random (0.50): {np.mean(results) - 0.5:+.3f}", flush=True)

    if np.mean(results) > 0.6:
        print(f"\n*** PROMISING: >0.6 AUROC without frequency cheating! ***", flush=True)
    else:
        print(f"\n*** No signal without frequency information ***", flush=True)


if __name__ == "__main__":
    main()
