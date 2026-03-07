#!/usr/bin/env python3
"""
Final deep dive: What exactly is different about U_sem's top-100 selections?

The mystery: U_sem has -0.85 freq-variance correlation on both datasets, but:
- FB15k-237: top-100 has 83% zero-evidence
- ICEWS14: top-100 has 7% zero-evidence

Let's look at the EXACT entities being selected and their coverage patterns.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import stats

from src.data.loaders import load_fb15k237, load_icews14


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class GPOnly(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)

    def forward(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2


def _kl_entity_gaussian(model):
    mean = model.entity_mean
    logvar = model.entity_logvar
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()


def train_model(model, triples, device, epochs=30):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(
        torch.tensor(triples[:, 0]),
        torch.tensor(triples[:, 1]),
        torch.tensor(triples[:, 2])
    ), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg = model(h, r, neg_t)
            loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
                   F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)) + \
                   0.001 * _kl_entity_gaussian(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def deep_analysis(dataset_name, loader, device, seed=42):
    print(f"\n{'='*80}")
    print(f"  {dataset_name}: U_sem TOP-100 DEEP DIVE")
    print(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Build coverage
    coverage = np.zeros((n_ent, n_rel), dtype=np.float32)
    for h, r, t in train:
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0

    # Entity frequency
    entity_freq = np.zeros(n_ent)
    for h, r, t in train:
        entity_freq[h] += 1
        entity_freq[t] += 1

    # Entity coverage count (how many relations each entity has evidence for)
    entity_cov_count = coverage.sum(axis=1)

    print(f"\n  Training U_sem...")
    model = GPOnly(n_ent, n_rel)
    model = train_model(model, train, device, epochs=30)

    # Get uncertainties for all test
    model.eval()
    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)
        unc = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -unc
    top100_idx = np.argsort(confidence)[::-1][:100]

    # Analyze top-100 triples
    top100_triples = test[top100_idx]
    top100_relations = top100_triples[:, 1]

    # What relations are in top-100?
    unique_rels, rel_counts = np.unique(top100_relations, return_counts=True)
    print(f"\n  Relations in top-100:")
    print(f"    Unique relations: {len(unique_rels)}")
    print(f"    Top 5 relations by count: {sorted(zip(rel_counts, unique_rels), reverse=True)[:5]}")

    # Entity coverage for entities in top-100
    top100_heads = top100_triples[:, 0]
    top100_tails = top100_triples[:, 2]
    top100_entities = np.unique(np.concatenate([top100_heads, top100_tails]))

    print(f"\n  Entities in top-100:")
    print(f"    Unique entities: {len(top100_entities)}")
    print(f"    Avg frequency: {entity_freq[top100_entities].mean():.1f}")
    print(f"    Avg coverage count: {entity_cov_count[top100_entities].mean():.1f} relations")

    # KEY: For each top-100 triple, is the SPECIFIC relation covered?
    # i.e., does (head, rel) or (tail, rel) have coverage?
    head_rel_covered = np.array([coverage[test[i, 0], test[i, 1]] for i in top100_idx])
    tail_rel_covered = np.array([coverage[test[i, 2], test[i, 1]] for i in top100_idx])
    both_covered = head_rel_covered * tail_rel_covered  # 1 if both, 0 otherwise

    zero_evidence = (head_rel_covered == 0) | (tail_rel_covered == 0)

    print(f"\n  Coverage analysis of top-100:")
    print(f"    Head-relation covered: {100*head_rel_covered.mean():.1f}%")
    print(f"    Tail-relation covered: {100*tail_rel_covered.mean():.1f}%")
    print(f"    Both covered: {100*both_covered.mean():.1f}%")
    print(f"    Zero evidence (at least one uncovered): {100*zero_evidence.mean():.1f}%")

    # WHY are these specific (entity, relation) pairs uncovered?
    # Is it because the RELATION is rare, or the ENTITY never saw this relation?
    rel_freq = np.zeros(n_rel)
    for h, r, t in train:
        rel_freq[r] += 1

    top100_rel_freq = rel_freq[top100_relations]
    print(f"\n  Relation frequencies in top-100:")
    print(f"    Avg relation frequency: {top100_rel_freq.mean():.1f}")
    print(f"    Min relation frequency: {top100_rel_freq.min():.1f}")
    print(f"    Max relation frequency: {top100_rel_freq.max():.1f}")

    # Compare to overall relation frequency
    print(f"    Overall avg relation frequency: {rel_freq.mean():.1f}")

    # The key insight: even though entities are high-frequency,
    # the specific (entity, relation) pair might not be covered
    # because the entity only appears with a SUBSET of relations

    # Compute: for top-100 entities, what fraction of relations are they covered for?
    top100_ent_coverage_rate = entity_cov_count[top100_entities] / n_rel
    print(f"\n  Coverage rate of top-100 entities:")
    print(f"    Mean: {100*top100_ent_coverage_rate.mean():.1f}% of relations")
    print(f"    Min: {100*top100_ent_coverage_rate.min():.1f}%")
    print(f"    Max: {100*top100_ent_coverage_rate.max():.1f}%")

    return {
        'zero_evidence_rate': zero_evidence.mean(),
        'top100_ent_coverage_rate_mean': top100_ent_coverage_rate.mean(),
        'top100_rel_freq_mean': top100_rel_freq.mean(),
        'overall_rel_freq_mean': rel_freq.mean(),
        'n_rel': n_rel,
    }


def main():
    device = setup_device()
    print(f"Device: {device}")

    print("\n" + "="*80)
    print("FINAL INVESTIGATION: U_sem ANOMALY")
    print("="*80)

    fb_stats = deep_analysis("FB15k-237", load_fb15k237, device)
    icews_stats = deep_analysis("ICEWS14", load_icews14, device)

    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    print(f"\n  FB15k-237:")
    print(f"    Zero-evidence rate: {100*fb_stats['zero_evidence_rate']:.1f}%")
    print(f"    Top-100 entities cover {100*fb_stats['top100_ent_coverage_rate_mean']:.1f}% of relations")
    print(f"    Top-100 avg relation freq: {fb_stats['top100_rel_freq_mean']:.1f} (overall: {fb_stats['overall_rel_freq_mean']:.1f})")

    print(f"\n  ICEWS14:")
    print(f"    Zero-evidence rate: {100*icews_stats['zero_evidence_rate']:.1f}%")
    print(f"    Top-100 entities cover {100*icews_stats['top100_ent_coverage_rate_mean']:.1f}% of relations")
    print(f"    Top-100 avg relation freq: {icews_stats['top100_rel_freq_mean']:.1f} (overall: {icews_stats['overall_rel_freq_mean']:.1f})")

    # The answer
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if icews_stats['top100_ent_coverage_rate_mean'] > fb_stats['top100_ent_coverage_rate_mean']:
        print(f"\n  ROOT CAUSE IDENTIFIED:")
        print(f"    On ICEWS14, the high-frequency entities in U_sem's top-100 have")
        print(f"    MUCH HIGHER relation coverage ({100*icews_stats['top100_ent_coverage_rate_mean']:.1f}% vs {100*fb_stats['top100_ent_coverage_rate_mean']:.1f}%).")
        print(f"    This means they're more likely to have training evidence for any given relation.")
    else:
        print(f"\n  UNEXPECTED: ICEWS14 entities have LOWER coverage rate.")
        print(f"    The difference must be in which specific relations appear in test queries.")

    print(f"\n  ADDITIONAL FACTOR:")
    fb_rel_ratio = fb_stats['top100_rel_freq_mean'] / fb_stats['overall_rel_freq_mean']
    icews_rel_ratio = icews_stats['top100_rel_freq_mean'] / icews_stats['overall_rel_freq_mean']
    print(f"    FB15k-237: top-100 rel freq / overall = {fb_rel_ratio:.2f}x")
    print(f"    ICEWS14: top-100 rel freq / overall = {icews_rel_ratio:.2f}x")

    if icews_rel_ratio > fb_rel_ratio:
        print(f"\n    ICEWS14's top-100 contains MORE COMMON relations,")
        print(f"    which are more likely to be covered.")


if __name__ == "__main__":
    main()
