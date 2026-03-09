#!/usr/bin/env python3
"""
Cold-Start Relation OOD Experiment

Hypothesis: Entities with LIMITED relation diversity in training
have HIGH variance (poorly constrained embeddings).
When they appear with NEW relations at test time,
semantic uncertainty should detect this.

Key difference from Role-Shift:
- Role-shift: High-diversity entities → Low variance → FAILED
- Cold-start: LOW-diversity entities → High variance → Should work

OOD Definition:
- Entity appears with only 1-3 distinct relation types in training
- At test time, appears with a NEW relation type
- Coverage exists (other entities use that relation)
- But this specific entity has high uncertainty due to limited training context
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os

def load_triples(path):
    """Load triples from file."""
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples

def analyze_entity_relation_diversity(triples):
    """Count how many distinct relations each entity uses."""
    entity_relations = defaultdict(set)
    entity_freq = defaultdict(int)

    for h, r, t in triples:
        entity_relations[h].add(r)
        entity_relations[t].add(r)
        entity_freq[h] += 1
        entity_freq[t] += 1

    return entity_relations, entity_freq

def create_cold_start_split(train_triples, test_triples, max_train_relations=3, min_freq=5):
    """
    Create cold-start relation OOD split.

    Target: Entities that appear with LIMITED relation types in training (1-3),
    but have moderate frequency (not just rare entities).
    At test time, they appear with a NEW relation.

    This should have:
    - High ρ (coverage overlap) because other entities use the test relation
    - High semantic uncertainty because entity is poorly constrained
    """
    # Analyze training data
    train_entity_relations, train_entity_freq = analyze_entity_relation_diversity(train_triples)

    # Build coverage matrix
    coverage = defaultdict(set)
    for h, r, t in train_triples:
        coverage[h].add(r)
        coverage[t].add(r)

    # Find "cold-start" entities: moderate freq + limited relation diversity
    cold_start_entities = set()
    for e, rels in train_entity_relations.items():
        freq = train_entity_freq[e]
        n_rels = len(rels)
        # Moderate frequency (not too rare) + limited diversity
        if freq >= min_freq and n_rels <= max_train_relations:
            cold_start_entities.add(e)

    print(f"Cold-start entities (freq>={min_freq}, rels<={max_train_relations}): {len(cold_start_entities)}")

    # Categorize test triples
    cold_start_ood = []  # Cold-start entity with NEW relation
    emerging_ood = []     # Low-frequency entities
    id_triples = []       # Normal ID

    freq_threshold = np.percentile(list(train_entity_freq.values()), 25)

    for h, r, t in test_triples:
        h_freq = train_entity_freq.get(h, 0)
        t_freq = train_entity_freq.get(t, 0)
        is_emerging = min(h_freq, t_freq) <= freq_threshold

        h_is_cold_start = h in cold_start_entities
        t_is_cold_start = t in cold_start_entities
        h_new_relation = r not in train_entity_relations.get(h, set())
        t_new_relation = r not in train_entity_relations.get(t, set())
        h_has_coverage = r in coverage[h]  # Other entities use this relation with h
        t_has_coverage = r in coverage[t]

        if is_emerging:
            emerging_ood.append((h, r, t))
        elif (h_is_cold_start and h_new_relation) or (t_is_cold_start and t_new_relation):
            # At least one entity is cold-start AND using a new relation
            # But the relation itself exists in training (coverage from other entities)
            cold_start_ood.append((h, r, t))
        else:
            id_triples.append((h, r, t))

    return {
        'cold_start': cold_start_ood,
        'emerging': emerging_ood,
        'id': id_triples,
        'coverage': coverage,
        'entity_freq': train_entity_freq,
        'entity_relations': train_entity_relations,
        'cold_start_entities': cold_start_entities
    }

class VariationalKGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.entity_mean = nn.Embedding(num_entities, dim)
        self.entity_logvar = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_logvar.weight, -2.0)
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

        score = (h_emb * rel * t_emb).sum(dim=-1)
        return score, h_logvar, t_logvar

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar(h)).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar(t)).mean(dim=-1)
        return 0.5 * (h_var + t_var)

def train_model(model, train_triples, num_entities, epochs=20, batch_size=1024, lr=1e-3, device='cpu'):
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

            neg_t = torch.randint(0, num_entities, (len(batch),), device=device)

            pos_score, h_logvar, t_logvar = model(h, r, t)
            neg_score, _, _ = model(h, r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-10).mean()
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
    model.eval()
    sem_unc, str_unc = [], []

    with torch.no_grad():
        for h, r, t in triples:
            h_t = torch.LongTensor([h]).to(device)
            r_t = torch.LongTensor([r]).to(device)
            t_t = torch.LongTensor([t]).to(device)

            u_sem = model.get_uncertainty(h_t, r_t, t_t).item()
            sem_unc.append(u_sem)

            h_cov = 1 if r in coverage[h] else 0
            t_cov = 1 if r in coverage[t] else 0
            u_str = 2 - h_cov - t_cov
            str_unc.append(u_str)

    return np.array(sem_unc), np.array(str_unc)

def evaluate_auroc(ood_triples, id_triples, model, coverage, device='cpu'):
    if len(ood_triples) == 0 or len(id_triples) == 0:
        return {'semantic': 0.5, 'structural': 0.5, 'cagp': 0.5, 'n_ood': 0, 'n_id': 0, 'rho': 0}

    all_triples = ood_triples + id_triples
    labels = [1] * len(ood_triples) + [0] * len(id_triples)

    sem_unc, str_unc = compute_uncertainties(model, all_triples, coverage, device)

    # Check coverage overlap (ρ) for OOD triples
    ood_str = str_unc[:len(ood_triples)]
    rho = np.mean(ood_str == 0)  # Fraction with full coverage

    sem_norm = (sem_unc - sem_unc.min()) / (sem_unc.max() - sem_unc.min() + 1e-10)
    str_norm = str_unc / 2.0
    cagp_unc = 0.5 * sem_norm + 0.5 * str_norm

    return {
        'semantic': roc_auc_score(labels, sem_unc),
        'structural': roc_auc_score(labels, str_unc),
        'cagp': roc_auc_score(labels, cagp_unc),
        'n_ood': len(ood_triples),
        'n_id': len(id_triples),
        'rho': rho
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw/gdelt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--max_rels', type=int, default=3, help='Max relations for cold-start entity')
    parser.add_argument('--min_freq', type=int, default=10, help='Min frequency for cold-start entity')
    args = parser.parse_args()

    print("=" * 60)
    print("Cold-Start Relation OOD Experiment")
    print("=" * 60)
    print(f"Target: Entities with freq>={args.min_freq}, relations<={args.max_rels}")

    # Load data
    print("\nLoading data...")
    train_triples = load_triples(os.path.join(args.data_dir, 'train.txt'))
    test_triples = load_triples(os.path.join(args.data_dir, 'test.txt'))

    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    all_triples = train_triples + test_triples
    num_entities = max(max(h, t) for h, _, t in all_triples) + 1
    num_relations = max(r for _, r, _ in all_triples) + 1
    print(f"Entities: {num_entities}, Relations: {num_relations}")

    # Try different thresholds
    best_result = None
    best_gain = -999

    for max_rels in [2, 3, 4, 5]:
        for min_freq in [5, 10, 20, 50]:
            split = create_cold_start_split(train_triples, test_triples,
                                           max_train_relations=max_rels,
                                           min_freq=min_freq)
            n_cold = len(split['cold_start'])
            if n_cold >= 500:  # Need enough samples
                print(f"\n  max_rels={max_rels}, min_freq={min_freq}: {n_cold} cold-start triples")

    # Use default params for main run
    print(f"\n{'='*60}")
    print("Creating split with default params...")
    split = create_cold_start_split(train_triples, test_triples,
                                   max_train_relations=args.max_rels,
                                   min_freq=args.min_freq)

    print(f"Cold-start OOD: {len(split['cold_start'])}")
    print(f"Emerging OOD: {len(split['emerging'])}")
    print(f"ID: {len(split['id'])}")

    if len(split['cold_start']) < 100:
        print("\n⚠️ Too few cold-start triples. Trying more lenient params...")
        for max_rels in [5, 10, 15]:
            for min_freq in [3, 5, 10]:
                split = create_cold_start_split(train_triples, test_triples,
                                               max_train_relations=max_rels,
                                               min_freq=min_freq)
                if len(split['cold_start']) >= 500:
                    print(f"Using max_rels={max_rels}, min_freq={min_freq}")
                    break
            if len(split['cold_start']) >= 500:
                break

    # Run experiments
    all_results = []

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed + 1}/{args.seeds}")
        print(f"{'='*60}")

        torch.manual_seed(seed + 42)
        np.random.seed(seed + 42)

        print("\nTraining variational KGE...")
        model = VariationalKGE(num_entities, num_relations, args.dim)
        model = train_model(model, train_triples, num_entities,
                           epochs=args.epochs, device=args.device)

        # Analyze cold-start entity variances
        model.eval()
        cold_start_vars = []
        non_cold_vars = []

        with torch.no_grad():
            for e in split['cold_start_entities']:
                if e < num_entities:
                    var = torch.exp(model.entity_logvar(torch.LongTensor([e]))).mean().item()
                    cold_start_vars.append(var)

            # Sample some non-cold-start entities
            for e in list(split['entity_freq'].keys())[:1000]:
                if e not in split['cold_start_entities'] and e < num_entities:
                    var = torch.exp(model.entity_logvar(torch.LongTensor([e]))).mean().item()
                    non_cold_vars.append(var)

        print(f"\nCold-start entity variance: {np.mean(cold_start_vars):.4f} ± {np.std(cold_start_vars):.4f}")
        print(f"Non-cold-start entity variance: {np.mean(non_cold_vars):.4f} ± {np.std(non_cold_vars):.4f}")
        print(f"Ratio: {np.mean(cold_start_vars) / (np.mean(non_cold_vars) + 1e-10):.2f}x")

        # Evaluate
        print("\n--- Cold-Start OOD (KEY RESULT) ---")
        cold_results = evaluate_auroc(split['cold_start'], split['id'],
                                     model, split['coverage'], args.device)
        print(f"  ρ (coverage overlap): {cold_results['rho']:.3f}")
        print(f"  Semantic AUROC:   {cold_results['semantic']:.3f}")
        print(f"  Structural AUROC: {cold_results['structural']:.3f}")
        print(f"  CAGP AUROC:       {cold_results['cagp']:.3f}")
        print(f"  ** Semantic - Structural: {cold_results['semantic'] - cold_results['structural']:+.3f} **")

        print("\n--- Emerging OOD ---")
        emerging_results = evaluate_auroc(split['emerging'], split['id'],
                                         model, split['coverage'], args.device)
        print(f"  Semantic: {emerging_results['semantic']:.3f}, Structural: {emerging_results['structural']:.3f}")

        all_results.append({
            'seed': seed,
            'cold_start': cold_results,
            'emerging': emerging_results,
            'cold_start_variance': float(np.mean(cold_start_vars)),
            'non_cold_variance': float(np.mean(non_cold_vars))
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Cold-Start OOD)")
    print("=" * 60)

    sem_scores = [r['cold_start']['semantic'] for r in all_results]
    str_scores = [r['cold_start']['structural'] for r in all_results]
    cagp_scores = [r['cold_start']['cagp'] for r in all_results]
    rho_scores = [r['cold_start']['rho'] for r in all_results]

    print(f"ρ (coverage overlap): {np.mean(rho_scores):.3f}")
    print(f"Semantic AUROC:   {np.mean(sem_scores):.3f} ± {np.std(sem_scores):.3f}")
    print(f"Structural AUROC: {np.mean(str_scores):.3f} ± {np.std(str_scores):.3f}")
    print(f"CAGP AUROC:       {np.mean(cagp_scores):.3f} ± {np.std(cagp_scores):.3f}")

    gain = np.mean(sem_scores) - np.mean(str_scores)
    print(f"\n** Semantic - Structural: {gain:+.3f} **")

    if gain > 0.05:
        print("\n✅ SUCCESS: Semantic provides meaningful lift!")
    elif gain > 0:
        print("\n⚠️ Marginal: Semantic slightly better but not significant")
    else:
        print("\n❌ FAILED: Semantic does not help")

    # Save
    output_path = 'outputs/cold_start_results.json'
    os.makedirs('outputs', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'split_sizes': {
                'cold_start': len(split['cold_start']),
                'emerging': len(split['emerging']),
                'id': len(split['id'])
            },
            'results': all_results,
            'summary': {
                'rho': float(np.mean(rho_scores)),
                'semantic': float(np.mean(sem_scores)),
                'structural': float(np.mean(str_scores)),
                'cagp': float(np.mean(cagp_scores)),
                'semantic_gain': float(gain)
            }
        }, f, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
