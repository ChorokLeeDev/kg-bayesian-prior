#!/usr/bin/env python3
"""
Controlled comparison: Is the coverage paradox real, or just entity frequency confound?
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237
from scipy import stats


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
    print("CONTROLLED COMPARISON: Is Coverage Effect Real?")
    print("="*70)

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Coverage and frequency
    coverage_set = set()
    entity_freq = {}
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
        entity_freq[int(h)] = entity_freq.get(int(h), 0) + 1
        entity_freq[int(t)] = entity_freq.get(int(t), 0) + 1

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

            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set

            # Coverage type
            if h_cov and t_cov:
                cov_type = 'full_cov'
            elif not h_cov and not t_cov:
                cov_type = 'full_zero'
            else:
                cov_type = 'partial'

            h_freq = entity_freq.get(int(h), 0)
            t_freq = entity_freq.get(int(t), 0)
            combined_freq = h_freq + t_freq

            # Rank
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)
            scores = energy(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = int((scores > true_score).sum() + 1)

            results.append({
                'cov_type': cov_type,
                'h_freq': h_freq,
                't_freq': t_freq,
                'combined_freq': combined_freq,
                'rank': rank,
                'hits10': rank <= 10
            })

    # ========================================
    # UNCONTROLLED COMPARISON (original)
    # ========================================
    print("\n" + "="*70)
    print("UNCONTROLLED COMPARISON (original finding)")
    print("="*70)

    for cov_type in ['full_cov', 'partial', 'full_zero']:
        subset = [r for r in results if r['cov_type'] == cov_type]
        if subset:
            hits10 = sum(r['hits10'] for r in subset) / len(subset)
            avg_freq = np.mean([r['combined_freq'] for r in subset])
            print(f"{cov_type:12}: n={len(subset):5}, Hits@10={hits10:.1%}, avg_freq={avg_freq:.0f}")

    # ========================================
    # CONTROLLED COMPARISON: Same frequency bands
    # ========================================
    print("\n" + "="*70)
    print("CONTROLLED COMPARISON: Within frequency bands")
    print("="*70)

    # Define frequency bands
    all_freqs = [r['combined_freq'] for r in results]
    freq_percentiles = np.percentile(all_freqs, [25, 50, 75])

    bands = [
        ('Low freq (0-25%)', 0, freq_percentiles[0]),
        ('Med-low (25-50%)', freq_percentiles[0], freq_percentiles[1]),
        ('Med-high (50-75%)', freq_percentiles[1], freq_percentiles[2]),
        ('High freq (75-100%)', freq_percentiles[2], float('inf'))
    ]

    print(f"\nFrequency percentiles: 25%={freq_percentiles[0]:.0f}, 50%={freq_percentiles[1]:.0f}, 75%={freq_percentiles[2]:.0f}")

    for band_name, low, high in bands:
        print(f"\n{band_name}:")
        for cov_type in ['full_cov', 'partial', 'full_zero']:
            subset = [r for r in results if r['cov_type'] == cov_type and low <= r['combined_freq'] < high]
            if len(subset) >= 20:
                hits10 = sum(r['hits10'] for r in subset) / len(subset)
                print(f"  {cov_type:12}: n={len(subset):4}, Hits@10={hits10:.1%}")
            else:
                print(f"  {cov_type:12}: n={len(subset):4} (too few)")

    # ========================================
    # STATISTICAL TEST: Coverage effect within same frequency
    # ========================================
    print("\n" + "="*70)
    print("STATISTICAL TEST: Coverage effect controlling for frequency")
    print("="*70)

    # Match partial and full_cov by frequency
    partial = [r for r in results if r['cov_type'] == 'partial']
    full_cov = [r for r in results if r['cov_type'] == 'full_cov']

    # For each partial sample, find a full_cov sample with similar frequency
    matched_partial = []
    matched_full = []

    np.random.seed(42)
    for p in partial:
        # Find full_cov samples within 10% frequency
        candidates = [f for f in full_cov if abs(f['combined_freq'] - p['combined_freq']) < p['combined_freq'] * 0.2]
        if candidates:
            match = np.random.choice(len(candidates))
            matched_partial.append(p)
            matched_full.append(candidates[match])

    print(f"\nMatched pairs: {len(matched_partial)}")

    if len(matched_partial) >= 100:
        partial_hits = sum(r['hits10'] for r in matched_partial) / len(matched_partial)
        full_hits = sum(r['hits10'] for r in matched_full) / len(matched_full)

        print(f"Matched partial Hits@10: {partial_hits:.1%}")
        print(f"Matched full_cov Hits@10: {full_hits:.1%}")

        # McNemar's test or simple proportion test
        partial_correct = [r['hits10'] for r in matched_partial]
        full_correct = [r['hits10'] for r in matched_full]

        # Bootstrap confidence interval
        n_bootstrap = 1000
        diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(partial_correct), len(partial_correct), replace=True)
            p_mean = np.mean([partial_correct[i] for i in idx])
            f_mean = np.mean([full_correct[i] for i in idx])
            diffs.append(p_mean - f_mean)

        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        print(f"\nDifference (partial - full): {partial_hits - full_hits:+.1%}")
        print(f"95% CI: [{ci_low:+.1%}, {ci_high:+.1%}]")

        if ci_low > 0:
            print(">>> SIGNIFICANT: Partial > Full even after controlling for frequency!")
        elif ci_high < 0:
            print(">>> SIGNIFICANT: Full > Partial after controlling")
        else:
            print(">>> NOT SIGNIFICANT: Difference could be due to frequency confound")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("""
Key question: Is the coverage paradox real, or just entity frequency confound?

If controlled comparison shows:
- Partial > Full within same frequency band → REAL coverage effect
- Partial ~ Full within same frequency band → CONFOUND (frequency explains it)
    """)


if __name__ == "__main__":
    main()
