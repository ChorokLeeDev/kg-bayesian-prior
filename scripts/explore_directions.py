#!/usr/bin/env python3
"""
Explore two directions:
1. Find HARM cases: Where zero-coverage leads to actual failures
2. Coverage as difficulty: Analyze why zero-coverage has higher accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def main():
    print("="*70)
    print("EXPLORING TWO DIRECTIONS")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"FB15k-237: {n_ent} entities, {n_rel} relations")

    # Coverage
    coverage_set = set()
    entity_rel_count = {}  # (e, r) -> count
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
        entity_rel_count[(int(h), int(r))] = entity_rel_count.get((int(h), int(r)), 0) + 1
        entity_rel_count[(int(t), int(r))] = entity_rel_count.get((int(t), int(r)), 0) + 1

    # Entity frequency
    entity_freq = {}
    for h, r, t in train:
        entity_freq[int(h)] = entity_freq.get(int(h), 0) + 1
        entity_freq[int(t)] = entity_freq.get(int(t), 0) + 1

    # Relation frequency
    rel_freq = {}
    for h, r, t in train:
        rel_freq[int(r)] = rel_freq.get(int(r), 0) + 1

    # Train Energy
    print("\n--- Training Energy ---")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h = torch.tensor(batch[:, 0])
            r = torch.tensor(batch[:, 1])
            t = torch.tensor(batch[:, 2])
            t_neg = torch.randint(0, n_ent, (len(batch),))

            optimizer.zero_grad()
            pos = energy(h, r, t)
            neg = energy(h, r, t_neg)
            loss = torch.clamp(1.0 - pos + neg, min=0).mean()
            loss.backward()
            optimizer.step()

    # Analyze test set
    print("\n--- Analyzing test set ---")
    energy.eval()

    results = []
    with torch.no_grad():
        for idx, (h, r, t) in enumerate(test):
            if idx % 3000 == 0:
                print(f"  {idx}/{len(test)}...")

            # Coverage status
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            is_zero_cov = not h_cov or not t_cov

            # Partial coverage
            partial_cov = (h_cov and not t_cov) or (not h_cov and t_cov)
            full_zero = not h_cov and not t_cov

            # Entity/relation frequencies
            h_freq = entity_freq.get(int(h), 0)
            t_freq = entity_freq.get(int(t), 0)
            r_freq = rel_freq.get(int(r), 0)

            # Rank
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)
            scores = energy(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = int((scores > true_score).sum() + 1)

            results.append({
                'h': h, 'r': r, 't': t,
                'zero_cov': is_zero_cov,
                'partial_cov': partial_cov,
                'full_zero': full_zero,
                'h_freq': h_freq,
                't_freq': t_freq,
                'r_freq': r_freq,
                'rank': rank,
                'hits10': rank <= 10,
                'hits1': rank == 1
            })

    # ========================================
    # DIRECTION 1: Find HARM cases
    # ========================================
    print("\n" + "="*70)
    print("DIRECTION 1: WHERE DOES ZERO-COVERAGE CAUSE HARM?")
    print("="*70)

    zero_cov = [r for r in results if r['zero_cov']]
    has_cov = [r for r in results if not r['zero_cov']]

    # Stratify by entity frequency
    print("\n--- By Entity Frequency ---")
    for freq_threshold in [10, 50, 100, 500]:
        zc_low_freq = [r for r in zero_cov if r['h_freq'] < freq_threshold or r['t_freq'] < freq_threshold]
        hc_low_freq = [r for r in has_cov if r['h_freq'] < freq_threshold or r['t_freq'] < freq_threshold]

        if zc_low_freq and hc_low_freq:
            zc_hits10 = sum(r['hits10'] for r in zc_low_freq) / len(zc_low_freq)
            hc_hits10 = sum(r['hits10'] for r in hc_low_freq) / len(hc_low_freq)
            print(f"Entity freq < {freq_threshold}:")
            print(f"  Zero-cov: {len(zc_low_freq)} samples, Hits@10 = {zc_hits10:.1%}")
            print(f"  Has-cov:  {len(hc_low_freq)} samples, Hits@10 = {hc_hits10:.1%}")

    # Stratify by relation frequency
    print("\n--- By Relation Frequency ---")
    rel_freqs_sorted = sorted(rel_freq.values())
    low_rel_threshold = rel_freqs_sorted[len(rel_freqs_sorted)//4]  # Bottom 25%
    high_rel_threshold = rel_freqs_sorted[3*len(rel_freqs_sorted)//4]  # Top 25%

    for name, threshold_fn in [("Rare relations", lambda r: r['r_freq'] < low_rel_threshold),
                                ("Common relations", lambda r: r['r_freq'] > high_rel_threshold)]:
        zc_subset = [r for r in zero_cov if threshold_fn(r)]
        hc_subset = [r for r in has_cov if threshold_fn(r)]

        if zc_subset and hc_subset:
            zc_hits10 = sum(r['hits10'] for r in zc_subset) / len(zc_subset)
            hc_hits10 = sum(r['hits10'] for r in hc_subset) / len(hc_subset)
            print(f"{name}:")
            print(f"  Zero-cov: {len(zc_subset)} samples, Hits@10 = {zc_hits10:.1%}")
            print(f"  Has-cov:  {len(hc_subset)} samples, Hits@10 = {hc_hits10:.1%}")

    # Full zero vs partial coverage
    print("\n--- Full Zero vs Partial Coverage ---")
    full_zero_samples = [r for r in results if r['full_zero']]
    partial_samples = [r for r in results if r['partial_cov']]

    if full_zero_samples:
        fz_hits10 = sum(r['hits10'] for r in full_zero_samples) / len(full_zero_samples)
        print(f"Full zero (neither h nor t covered): {len(full_zero_samples)} samples, Hits@10 = {fz_hits10:.1%}")

    if partial_samples:
        p_hits10 = sum(r['hits10'] for r in partial_samples) / len(partial_samples)
        print(f"Partial (one covered): {len(partial_samples)} samples, Hits@10 = {p_hits10:.1%}")

    # ========================================
    # DIRECTION 2: Coverage as Difficulty
    # ========================================
    print("\n" + "="*70)
    print("DIRECTION 2: IS COVERAGE A DIFFICULTY INDICATOR?")
    print("="*70)

    # Hypothesis: Zero-coverage queries are "easy" because they involve common patterns
    print("\n--- Average ranks by coverage ---")
    zc_avg_rank = np.mean([r['rank'] for r in zero_cov])
    hc_avg_rank = np.mean([r['rank'] for r in has_cov])
    print(f"Zero-coverage avg rank: {zc_avg_rank:.1f}")
    print(f"Has-coverage avg rank: {hc_avg_rank:.1f}")

    # Check if zero-cov has higher entity frequency overall
    print("\n--- Entity frequency by coverage ---")
    zc_avg_h_freq = np.mean([r['h_freq'] for r in zero_cov])
    zc_avg_t_freq = np.mean([r['t_freq'] for r in zero_cov])
    hc_avg_h_freq = np.mean([r['h_freq'] for r in has_cov])
    hc_avg_t_freq = np.mean([r['t_freq'] for r in has_cov])
    print(f"Zero-cov: avg h_freq={zc_avg_h_freq:.1f}, avg t_freq={zc_avg_t_freq:.1f}")
    print(f"Has-cov:  avg h_freq={hc_avg_h_freq:.1f}, avg t_freq={hc_avg_t_freq:.1f}")

    # Check relation frequency
    print("\n--- Relation frequency by coverage ---")
    zc_avg_r_freq = np.mean([r['r_freq'] for r in zero_cov])
    hc_avg_r_freq = np.mean([r['r_freq'] for r in has_cov])
    print(f"Zero-cov: avg r_freq={zc_avg_r_freq:.1f}")
    print(f"Has-cov:  avg r_freq={hc_avg_r_freq:.1f}")

    # ========================================
    # KEY ANALYSIS: When is zero-coverage BAD?
    # ========================================
    print("\n" + "="*70)
    print("KEY QUESTION: WHEN IS ZERO-COVERAGE ACTUALLY HARMFUL?")
    print("="*70)

    # Find cases where zero-cov has LOWER accuracy than has-cov
    print("\nSearching for harmful conditions...")

    # By specific relations
    rel_results = {}
    for r in results:
        rel = r['r']
        if rel not in rel_results:
            rel_results[rel] = {'zero_cov': [], 'has_cov': []}
        if r['zero_cov']:
            rel_results[rel]['zero_cov'].append(r)
        else:
            rel_results[rel]['has_cov'].append(r)

    harmful_relations = []
    for rel, data in rel_results.items():
        if len(data['zero_cov']) >= 20 and len(data['has_cov']) >= 20:
            zc_hits = sum(r['hits10'] for r in data['zero_cov']) / len(data['zero_cov'])
            hc_hits = sum(r['hits10'] for r in data['has_cov']) / len(data['has_cov'])
            if zc_hits < hc_hits - 0.1:  # 10pp worse
                harmful_relations.append((rel, zc_hits, hc_hits, len(data['zero_cov']), len(data['has_cov'])))

    if harmful_relations:
        print(f"\nFound {len(harmful_relations)} relations where zero-cov is harmful:")
        harmful_relations.sort(key=lambda x: x[1] - x[2])  # Sort by gap
        for rel, zc, hc, n_zc, n_hc in harmful_relations[:10]:
            print(f"  Relation {rel}: ZC={zc:.1%} ({n_zc}), HC={hc:.1%} ({n_hc}), gap={zc-hc:+.1%}")
    else:
        print("\nNo relations found where zero-coverage is significantly harmful!")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    zc_total_hits = sum(r['hits10'] for r in zero_cov) / len(zero_cov)
    hc_total_hits = sum(r['hits10'] for r in has_cov) / len(has_cov)
    print(f"Overall: Zero-cov Hits@10 = {zc_total_hits:.1%}, Has-cov = {hc_total_hits:.1%}")

    if zc_total_hits > hc_total_hits:
        print("\n>>> Zero-coverage has HIGHER accuracy overall!")
        print(">>> This suggests coverage is an INVERSE difficulty indicator.")
        print(">>> Paper direction: 'Coverage Paradox' - less evidence = better performance?")
    else:
        print("\n>>> Zero-coverage has LOWER accuracy overall.")
        print(">>> This supports the 'blind spot' narrative.")


if __name__ == "__main__":
    main()
