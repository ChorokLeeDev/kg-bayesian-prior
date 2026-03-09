#!/usr/bin/env python3
"""
GDELT Role-Shift OOD Experiment

Goal: Find a NON-CIRCULAR benchmark where semantic uncertainty helps (ρ > 0).

Role-Shift OOD Definition:
- An entity using a relation that is ATYPICAL for its historical profile
- "Atypical" = relation NOT in entity's top 80% most frequent relations
- This creates high ρ (coverage overlap) because entity HAS coverage for the relation,
  but the behavior is unusual for that entity

Expected outcome:
- Coverage alone: ~0.70 AUROC (has coverage but atypical)
- Semantic: ~0.85+ AUROC (low-freq entities have higher variance)
- CAGP: ~0.90 AUROC

This would prove semantic is NECESSARY on a non-circular benchmark.
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import json
import os

def load_triples(path):
    """Load triples from file. Format: h r t timestamp extra (tab-separated)"""
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples

def compute_entity_relation_profile(triples):
    """Compute which relations each entity typically uses."""
    entity_relation_counts = defaultdict(lambda: defaultdict(int))
    for h, r, t in triples:
        entity_relation_counts[h][r] += 1
        entity_relation_counts[t][r] += 1
    return entity_relation_counts

def get_typical_relations(entity_relation_counts, entity, top_pct=0.8):
    """Get the top X% most frequent relations for an entity."""
    rel_counts = entity_relation_counts[entity]
    if not rel_counts:
        return set()
    sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
    total = sum(rel_counts.values())
    cumsum = 0
    typical = set()
    for r, count in sorted_rels:
        cumsum += count
        typical.add(r)
        if cumsum / total >= top_pct:
            break
    return typical

def create_role_shift_split(train_triples, test_triples, entity_relation_counts, top_pct=0.8):
    """
    Create role-shift OOD split.

    OOD = test triples where entity uses an ATYPICAL relation (not in top 80%)
    ID = test triples where entity uses a TYPICAL relation

    Key: Entity still has COVERAGE for the relation (from other entities),
    but this specific entity rarely uses it.
    """
    # Build coverage matrix from training
    coverage = defaultdict(set)
    for h, r, t in train_triples:
        coverage[h].add(r)
        coverage[t].add(r)

    # Compute entity frequencies
    entity_freq = defaultdict(int)
    for h, r, t in train_triples:
        entity_freq[h] += 1
        entity_freq[t] += 1

    freq_threshold = np.percentile(list(entity_freq.values()), 25)

    role_shift_ood = []  # Atypical relation usage
    emerging_ood = []    # Low-frequency entities
    id_triples = []      # Normal ID

    for h, r, t in test_triples:
        h_typical = get_typical_relations(entity_relation_counts, h, top_pct)
        t_typical = get_typical_relations(entity_relation_counts, t, top_pct)

        is_emerging = min(entity_freq.get(h, 0), entity_freq.get(t, 0)) <= freq_threshold
        h_has_coverage = r in coverage[h]
        t_has_coverage = r in coverage[t]
        h_atypical = r not in h_typical and len(h_typical) > 0
        t_atypical = r not in t_typical and len(t_typical) > 0

        if is_emerging:
            emerging_ood.append((h, r, t))
        elif (h_atypical or t_atypical) and (h_has_coverage or t_has_coverage):
            # Key: has coverage BUT atypical behavior
            role_shift_ood.append((h, r, t))
        else:
            id_triples.append((h, r, t))

    return {
        'role_shift': role_shift_ood,
        'emerging': emerging_ood,
        'id': id_triples,
        'coverage': coverage,
        'entity_freq': entity_freq
    }

class VariationalKGE(nn.Module):
    """Simple variational KG embedding for uncertainty estimation."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.entity_mean = nn.Embedding(num_entities, dim)
        self.entity_logvar = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_logvar.weight, -2.0)  # Small initial variance
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t, sample=True):
        h_mean = self.entity_mean(h)
        t_mean = self.entity_mean(t)
        h_logvar = self.entity_logvar(h)
        t_logvar = self.entity_logvar(t)
        rel = self.relation_emb(r)

        if sample and self.training:
            h_std = torch.exp(0.5 * h_logvar)
            t_std = torch.exp(0.5 * t_logvar)
            h_emb = h_mean + h_std * torch.randn_like(h_std)
            t_emb = t_mean + t_std * torch.randn_like(t_std)
        else:
            h_emb = h_mean
            t_emb = t_mean

        # DistMult scoring
        score = (h_emb * rel * t_emb).sum(dim=-1)
        return score, h_logvar, t_logvar

    def get_uncertainty(self, h, r, t):
        """Get semantic uncertainty (mean variance)."""
        h_var = torch.exp(self.entity_logvar(h)).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar(t)).mean(dim=-1)
        return 0.5 * (h_var + t_var)

def train_model(model, train_triples, num_entities, epochs=20, batch_size=1024, lr=1e-3, device='cpu'):
    """Train variational KGE model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.LongTensor(train_triples)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = train_tensor[batch_idx].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(batch),), device=device)

            pos_score, h_logvar, t_logvar = model(h, r, t)
            neg_score, _, _ = model(h, r, neg_t)

            # BCE loss
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-10).mean()

            # KL regularization
            kl_loss = -0.5 * (1 + h_logvar - h_logvar.exp()).mean()
            kl_loss += -0.5 * (1 + t_logvar - t_logvar.exp()).mean()

            loss = pos_loss + neg_loss + 0.001 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

def compute_uncertainties(model, triples, coverage, device='cpu'):
    """Compute semantic and structural uncertainties."""
    model.eval()

    sem_unc = []
    str_unc = []

    with torch.no_grad():
        for h, r, t in triples:
            h_t = torch.LongTensor([h]).to(device)
            r_t = torch.LongTensor([r]).to(device)
            t_t = torch.LongTensor([t]).to(device)

            # Semantic uncertainty
            u_sem = model.get_uncertainty(h_t, r_t, t_t).item()
            sem_unc.append(u_sem)

            # Structural uncertainty (coverage)
            h_cov = 1 if r in coverage[h] else 0
            t_cov = 1 if r in coverage[t] else 0
            u_str = 2 - h_cov - t_cov
            str_unc.append(u_str)

    return np.array(sem_unc), np.array(str_unc)

def evaluate_auroc(ood_triples, id_triples, model, coverage, device='cpu'):
    """Compute AUROC for different uncertainty signals."""
    if len(ood_triples) == 0 or len(id_triples) == 0:
        return {'semantic': 0.5, 'structural': 0.5, 'cagp': 0.5, 'n_ood': 0, 'n_id': 0}

    all_triples = ood_triples + id_triples
    labels = [1] * len(ood_triples) + [0] * len(id_triples)

    sem_unc, str_unc = compute_uncertainties(model, all_triples, coverage, device)

    # Normalize semantic for combination
    sem_norm = (sem_unc - sem_unc.min()) / (sem_unc.max() - sem_unc.min() + 1e-10)
    str_norm = str_unc / 2.0  # Already in [0, 1]

    # CAGP combination
    cagp_unc = 0.5 * sem_norm + 0.5 * str_norm

    results = {
        'semantic': roc_auc_score(labels, sem_unc),
        'structural': roc_auc_score(labels, str_unc),
        'cagp': roc_auc_score(labels, cagp_unc),
        'n_ood': len(ood_triples),
        'n_id': len(id_triples),
        'rho': np.mean(str_unc[:len(ood_triples)] == 0)  # Coverage overlap
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw/gdelt', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--top_pct', type=float, default=0.8, help='Top % relations considered typical')
    args = parser.parse_args()

    print("=" * 60)
    print("GDELT Role-Shift OOD Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_triples = load_triples(os.path.join(args.data_dir, 'train.txt'))
    test_triples = load_triples(os.path.join(args.data_dir, 'test.txt'))

    print(f"Train: {len(train_triples)} triples")
    print(f"Test: {len(test_triples)} triples")

    # Get entity/relation counts
    all_triples = train_triples + test_triples
    num_entities = max(max(h, t) for h, _, t in all_triples) + 1
    num_relations = max(r for _, r, _ in all_triples) + 1
    print(f"Entities: {num_entities}, Relations: {num_relations}")

    # Compute entity-relation profiles from training
    entity_relation_counts = compute_entity_relation_profile(train_triples)

    # Create role-shift split
    print(f"\nCreating role-shift split (top {args.top_pct*100:.0f}% = typical)...")
    split = create_role_shift_split(train_triples, test_triples, entity_relation_counts, args.top_pct)

    print(f"Role-shift OOD: {len(split['role_shift'])} triples")
    print(f"Emerging OOD: {len(split['emerging'])} triples")
    print(f"ID: {len(split['id'])} triples")

    if len(split['role_shift']) < 100:
        print("\nWARNING: Too few role-shift triples. Adjusting threshold...")
        # Try with more lenient threshold
        for top_pct in [0.7, 0.6, 0.5]:
            split = create_role_shift_split(train_triples, test_triples, entity_relation_counts, top_pct)
            print(f"  top_pct={top_pct}: {len(split['role_shift'])} role-shift triples")
            if len(split['role_shift']) >= 100:
                break

    # Run experiments with multiple seeds
    all_results = []

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed + 1}/{args.seeds}")
        print(f"{'='*60}")

        torch.manual_seed(seed + 42)
        np.random.seed(seed + 42)

        # Train model
        print("\nTraining variational KGE...")
        model = VariationalKGE(num_entities, num_relations, args.dim)
        model = train_model(model, train_triples, num_entities,
                           epochs=args.epochs, device=args.device)

        # Evaluate on role-shift OOD (THE KEY RESULT)
        print("\n--- Role-Shift OOD (KEY RESULT) ---")
        role_shift_results = evaluate_auroc(
            split['role_shift'], split['id'], model, split['coverage'], args.device
        )
        print(f"  ρ (coverage overlap): {role_shift_results['rho']:.3f}")
        print(f"  Semantic AUROC:   {role_shift_results['semantic']:.3f}")
        print(f"  Structural AUROC: {role_shift_results['structural']:.3f}")
        print(f"  CAGP AUROC:       {role_shift_results['cagp']:.3f}")
        print(f"  Semantic - Structural: {role_shift_results['semantic'] - role_shift_results['structural']:+.3f}")

        # Evaluate on emerging OOD
        print("\n--- Emerging OOD ---")
        emerging_results = evaluate_auroc(
            split['emerging'], split['id'], model, split['coverage'], args.device
        )
        print(f"  Semantic AUROC:   {emerging_results['semantic']:.3f}")
        print(f"  Structural AUROC: {emerging_results['structural']:.3f}")
        print(f"  CAGP AUROC:       {emerging_results['cagp']:.3f}")

        all_results.append({
            'seed': seed,
            'role_shift': role_shift_results,
            'emerging': emerging_results
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Role-Shift OOD)")
    print("=" * 60)

    sem_scores = [r['role_shift']['semantic'] for r in all_results]
    str_scores = [r['role_shift']['structural'] for r in all_results]
    cagp_scores = [r['role_shift']['cagp'] for r in all_results]
    rho_scores = [r['role_shift']['rho'] for r in all_results]

    print(f"ρ (coverage overlap): {np.mean(rho_scores):.3f} ± {np.std(rho_scores):.3f}")
    print(f"Semantic AUROC:   {np.mean(sem_scores):.3f} ± {np.std(sem_scores):.3f}")
    print(f"Structural AUROC: {np.mean(str_scores):.3f} ± {np.std(str_scores):.3f}")
    print(f"CAGP AUROC:       {np.mean(cagp_scores):.3f} ± {np.std(cagp_scores):.3f}")
    print(f"\n** Semantic - Structural: {np.mean(sem_scores) - np.mean(str_scores):+.3f} **")

    if np.mean(sem_scores) > np.mean(str_scores) + 0.05:
        print("\n✅ SUCCESS: Semantic provides meaningful lift on non-circular benchmark!")
    else:
        print("\n⚠️  Semantic does not provide significant lift over structural.")

    # Save results
    output_path = 'outputs/gdelt_role_shift_results.json'
    os.makedirs('outputs', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'split_sizes': {
                'role_shift': len(split['role_shift']),
                'emerging': len(split['emerging']),
                'id': len(split['id'])
            },
            'results': all_results,
            'summary': {
                'rho': float(np.mean(rho_scores)),
                'semantic': float(np.mean(sem_scores)),
                'structural': float(np.mean(str_scores)),
                'cagp': float(np.mean(cagp_scores)),
                'semantic_gain': float(np.mean(sem_scores) - np.mean(str_scores))
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
