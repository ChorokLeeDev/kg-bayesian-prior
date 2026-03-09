#!/usr/bin/env python3
"""
Covered-but-Rare OOD Experiment

Design: Find entities that ARE covered for a relation but have LOW overall frequency.
- These entities have high variance (poorly constrained embeddings)
- But coverage exists (U_str = 0)
- Semantic should detect them as OOD

Key difference from previous experiments:
- Role-shift: High-freq entities → Low variance → Failed
- Cold-start: No coverage (ρ=0) → Structural dominates
- THIS: Low-freq entities WITH coverage → ρ > 0, semantic should help
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys

PYTHONUNBUFFERED = True

def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
    return triples


def create_covered_rare_split(train_triples, test_triples, freq_threshold_pct=25):
    """
    Create OOD split where:
    - OOD = triples with LOW-FREQ entities that HAVE coverage
    - ID = triples with HIGH-FREQ entities

    This ensures ρ > 0 (OOD entities have coverage for query relation)
    """
    # Compute entity frequency from training
    entity_freq = defaultdict(int)
    for h, r, t in train_triples:
        entity_freq[h] += 1
        entity_freq[t] += 1

    # Compute coverage: which (entity, relation) pairs exist in training
    coverage = defaultdict(set)
    for h, r, t in train_triples:
        coverage[h].add(r)
        coverage[t].add(r)

    # Frequency threshold (bottom 25% = rare)
    all_freqs = [f for f in entity_freq.values() if f > 0]
    freq_threshold = np.percentile(all_freqs, freq_threshold_pct)

    print(f"Frequency threshold ({freq_threshold_pct}th pctl): {freq_threshold}")

    # Categorize test triples
    ood_covered_rare = []  # KEY: low-freq entities WITH coverage
    ood_no_coverage = []   # Standard novel-context
    id_triples = []

    for h, r, t in test_triples:
        min_freq = min(entity_freq.get(h, 0), entity_freq.get(t, 0))
        h_covered = r in coverage.get(h, set())
        t_covered = r in coverage.get(t, set())
        both_covered = h_covered and t_covered

        is_rare = min_freq <= freq_threshold and min_freq > 0  # Low-freq but seen

        if not both_covered:
            # No coverage - standard novel context
            ood_no_coverage.append((h, r, t))
        elif is_rare:
            # KEY CATEGORY: Covered but rare
            ood_covered_rare.append((h, r, t))
        else:
            # High-freq and covered - ID
            id_triples.append((h, r, t))

    print(f"\nSplit statistics:")
    print(f"  OOD (covered + rare): {len(ood_covered_rare)}")
    print(f"  OOD (no coverage):    {len(ood_no_coverage)}")
    print(f"  ID:                   {len(id_triples)}")

    # Compute ρ for covered-rare subset
    if len(ood_covered_rare) > 0:
        # ρ = fraction of OOD that has coverage (by design, all of them here)
        rho = 1.0  # All covered_rare have coverage by construction
        print(f"  ρ (coverage overlap) for covered_rare: {rho:.3f}")

    return {
        'ood_covered_rare': ood_covered_rare,
        'ood_no_coverage': ood_no_coverage,
        'id': id_triples,
        'entity_freq': entity_freq,
        'coverage': coverage,
        'freq_threshold': freq_threshold
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

        if sample and self.training:
            h_logvar = self.entity_logvar(h)
            t_logvar = self.entity_logvar(t)
            h_emb = h_mean + torch.exp(0.5 * h_logvar) * torch.randn_like(h_mean)
            t_emb = t_mean + torch.exp(0.5 * t_logvar) * torch.randn_like(t_mean)
        else:
            h_emb = h_mean
            t_emb = t_mean

        r_emb = self.relation_emb(r)
        score = (h_emb * r_emb * t_emb).sum(dim=-1)
        return score

    def get_entity_variance(self, entity_ids):
        logvar = self.entity_logvar(entity_ids)
        return torch.exp(logvar).mean(dim=-1)

    def kl_loss(self):
        mean = self.entity_mean.weight
        logvar = self.entity_logvar.weight
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl / mean.size(0)


def train_model(model, train_triples, num_entities, epochs=15, batch_size=1024, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.LongTensor(train_triples).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        perm = torch.randperm(len(train_tensor))
        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[perm[i:i+batch_size]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Positive scores
            pos_scores = model(h, r, t, sample=True)

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t, sample=True)

            # BCE loss
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()

            # KL loss
            kl = model.kl_loss()

            loss = pos_loss + neg_loss + 0.001 * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}", flush=True)

    return model


def evaluate_ood(model, split_data, device='cpu'):
    model.eval()

    ood_covered_rare = split_data['ood_covered_rare']
    ood_no_coverage = split_data['ood_no_coverage']
    id_triples = split_data['id']
    entity_freq = split_data['entity_freq']
    coverage = split_data['coverage']

    if len(ood_covered_rare) == 0:
        print("No covered-rare OOD triples found!")
        return None

    results = {}

    # Evaluate on covered-rare (KEY RESULT)
    print(f"\n--- Covered-Rare OOD (KEY RESULT, n={len(ood_covered_rare)}) ---", flush=True)

    # Compute uncertainties
    def get_uncertainties(triples):
        u_sem = []
        u_str = []

        with torch.no_grad():
            for h, r, t in triples:
                # Semantic: entity variance
                h_var = model.get_entity_variance(torch.LongTensor([h]).to(device)).item()
                t_var = model.get_entity_variance(torch.LongTensor([t]).to(device)).item()
                sem = (h_var + t_var) / 2

                # Structural: coverage
                h_cov = 1 if r in coverage.get(h, set()) else 0
                t_cov = 1 if r in coverage.get(t, set()) else 0
                struct = 2 - h_cov - t_cov  # 0=both covered, 2=neither

                u_sem.append(sem)
                u_str.append(struct)

        return np.array(u_sem), np.array(u_str)

    # Get uncertainties for all categories
    sem_ood_cr, str_ood_cr = get_uncertainties(ood_covered_rare)
    sem_id, str_id = get_uncertainties(id_triples)

    # Labels: 1=OOD, 0=ID
    labels = np.concatenate([np.ones(len(ood_covered_rare)), np.zeros(len(id_triples))])

    # Combined scores
    all_sem = np.concatenate([sem_ood_cr, sem_id])
    all_str = np.concatenate([str_ood_cr, str_id])

    # AUROC
    sem_auroc = roc_auc_score(labels, all_sem)
    str_auroc = roc_auc_score(labels, all_str)
    cagp_auroc = roc_auc_score(labels, 0.5 * all_sem + 0.5 * all_str)

    # ρ = fraction of OOD with coverage (should be 1.0 for covered-rare)
    rho = np.mean(str_ood_cr == 0)  # Coverage means str=0

    print(f"  ρ (coverage overlap): {rho:.3f}", flush=True)
    print(f"  Semantic AUROC:   {sem_auroc:.3f}", flush=True)
    print(f"  Structural AUROC: {str_auroc:.3f}", flush=True)
    print(f"  CAGP AUROC:       {cagp_auroc:.3f}", flush=True)
    print(f"  ** Semantic - Structural: {sem_auroc - str_auroc:+.3f} **", flush=True)

    # Variance analysis
    ood_entities = set()
    for h, r, t in ood_covered_rare:
        ood_entities.add(h)
        ood_entities.add(t)

    id_entities = set()
    for h, r, t in id_triples:
        id_entities.add(h)
        id_entities.add(t)
    id_entities = id_entities - ood_entities

    with torch.no_grad():
        ood_vars = model.get_entity_variance(torch.LongTensor(list(ood_entities)).to(device))
        id_vars = model.get_entity_variance(torch.LongTensor(list(id_entities)).to(device))

        print(f"\n  OOD entity variance: {ood_vars.mean():.4f} ± {ood_vars.std():.4f}", flush=True)
        print(f"  ID entity variance:  {id_vars.mean():.4f} ± {id_vars.std():.4f}", flush=True)
        print(f"  Ratio: {ood_vars.mean() / id_vars.mean():.2f}x", flush=True)

    results['covered_rare'] = {
        'rho': rho,
        'sem_auroc': sem_auroc,
        'str_auroc': str_auroc,
        'cagp_auroc': cagp_auroc,
        'gap': sem_auroc - str_auroc
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw/gdelt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--freq_pct', type=int, default=25, help='Frequency percentile for rare')
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Covered-but-Rare OOD Experiment", flush=True)
    print("=" * 60, flush=True)
    print(f"Target: Low-freq entities WITH coverage", flush=True)
    print(f"Expected: ρ > 0, semantic should help", flush=True)
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
    print("Creating covered-rare split...", flush=True)
    split_data = create_covered_rare_split(train_triples, test_triples, args.freq_pct)

    if len(split_data['ood_covered_rare']) == 0:
        print("ERROR: No covered-rare OOD triples. Try different freq_pct.", flush=True)
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
        print("\nTraining variational KGE...", flush=True)
        model = VariationalKGE(num_entities, num_relations, args.dim)
        model = train_model(model, train_triples, num_entities,
                           epochs=args.epochs, device=args.device)

        # Evaluate
        results = evaluate_ood(model, split_data, args.device)
        if results:
            all_results.append(results)

    # Aggregate
    if all_results:
        print(f"\n{'=' * 60}", flush=True)
        print("AGGREGATE RESULTS", flush=True)
        print("=" * 60, flush=True)

        sem_aurocs = [r['covered_rare']['sem_auroc'] for r in all_results]
        str_aurocs = [r['covered_rare']['str_auroc'] for r in all_results]
        gaps = [r['covered_rare']['gap'] for r in all_results]

        print(f"\nCovered-Rare OOD ({args.seeds} seeds):", flush=True)
        print(f"  ρ: {all_results[0]['covered_rare']['rho']:.3f}", flush=True)
        print(f"  Semantic:   {np.mean(sem_aurocs):.3f} ± {np.std(sem_aurocs):.3f}", flush=True)
        print(f"  Structural: {np.mean(str_aurocs):.3f} ± {np.std(str_aurocs):.3f}", flush=True)
        print(f"  Gap:        {np.mean(gaps):+.3f} ± {np.std(gaps):.3f}", flush=True)

        if np.mean(gaps) > 0:
            print(f"\n*** SUCCESS: Semantic beats structural by {np.mean(gaps)*100:+.1f}pp! ***", flush=True)
        else:
            print(f"\n*** Semantic did not beat structural ({np.mean(gaps)*100:+.1f}pp) ***", flush=True)


if __name__ == "__main__":
    main()
