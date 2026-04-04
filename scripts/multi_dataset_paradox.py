#!/usr/bin/env python3
"""
Multi-dataset validation: Does the coverage paradox hold on WN18RR and YAGO3-10?
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def analyze_dataset(name, train, test, n_ent, n_rel):
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")
    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Coverage and frequency
    coverage_set = set()
    entity_freq = {}
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
        entity_freq[int(h)] = entity_freq.get(int(h), 0) + 1
        entity_freq[int(t)] = entity_freq.get(int(t), 0) + 1

    # Train Energy
    print("\n--- Training Energy (10 epochs) ---")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)

    for epoch in range(10):
        np.random.shuffle(train)
        total_loss = 0
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
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.2f}")

    # Analyze test set (subsample for speed)
    test_sub = test[:3000] if len(test) > 3000 else test
    print(f"\n--- Analyzing {len(test_sub)} test samples ---")
    energy.eval()

    results = []
    with torch.no_grad():
        for idx, (h, r, t) in enumerate(test_sub):
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set

            if h_cov and t_cov:
                cov_type = 'full_cov'
            elif not h_cov and not t_cov:
                cov_type = 'full_zero'
            else:
                cov_type = 'partial'

            combined_freq = entity_freq.get(int(h), 0) + entity_freq.get(int(t), 0)

            # Rank
            h_exp = torch.full((n_ent,), h, dtype=torch.long)
            r_exp = torch.full((n_ent,), r, dtype=torch.long)
            all_t = torch.arange(n_ent)
            scores = energy(h_exp, r_exp, all_t).numpy()
            true_score = scores[t]
            rank = int((scores > true_score).sum() + 1)

            results.append({
                'cov_type': cov_type,
                'combined_freq': combined_freq,
                'hits10': rank <= 10
            })

    # Results
    print("\n--- UNCONTROLLED COMPARISON ---")
    for cov_type in ['full_cov', 'partial', 'full_zero']:
        subset = [r for r in results if r['cov_type'] == cov_type]
        if subset:
            hits10 = sum(r['hits10'] for r in subset) / len(subset)
            avg_freq = np.mean([r['combined_freq'] for r in subset])
            print(f"{cov_type:12}: n={len(subset):5}, Hits@10={hits10:.1%}, avg_freq={avg_freq:.0f}")

    # Controlled comparison
    print("\n--- CONTROLLED COMPARISON (frequency bands) ---")
    all_freqs = [r['combined_freq'] for r in results]
    freq_percentiles = np.percentile(all_freqs, [25, 50, 75])

    bands = [
        ('Low (0-25%)', 0, freq_percentiles[0]),
        ('Med-low (25-50%)', freq_percentiles[0], freq_percentiles[1]),
        ('Med-high (50-75%)', freq_percentiles[1], freq_percentiles[2]),
        ('High (75-100%)', freq_percentiles[2], float('inf'))
    ]

    for band_name, low, high in bands:
        full_cov = [r for r in results if r['cov_type'] == 'full_cov' and low <= r['combined_freq'] < high]
        partial = [r for r in results if r['cov_type'] == 'partial' and low <= r['combined_freq'] < high]

        if len(full_cov) >= 20 and len(partial) >= 20:
            fc_hits = sum(r['hits10'] for r in full_cov) / len(full_cov)
            p_hits = sum(r['hits10'] for r in partial) / len(partial)
            diff = p_hits - fc_hits
            print(f"{band_name:20}: Full={fc_hits:.1%} (n={len(full_cov)}), Partial={p_hits:.1%} (n={len(partial)}), Δ={diff:+.1%}")
        else:
            print(f"{band_name:20}: Insufficient samples")

    # Matched-pair analysis
    print("\n--- MATCHED-PAIR ANALYSIS ---")
    partial = [r for r in results if r['cov_type'] == 'partial']
    full_cov = [r for r in results if r['cov_type'] == 'full_cov']

    matched_partial = []
    matched_full = []
    np.random.seed(42)

    for p in partial:
        candidates = [f for f in full_cov if abs(f['combined_freq'] - p['combined_freq']) < max(p['combined_freq'] * 0.2, 10)]
        if candidates:
            match = np.random.choice(len(candidates))
            matched_partial.append(p)
            matched_full.append(candidates[match])

    if len(matched_partial) >= 50:
        p_hits = sum(r['hits10'] for r in matched_partial) / len(matched_partial)
        f_hits = sum(r['hits10'] for r in matched_full) / len(matched_full)

        # Bootstrap CI
        diffs = []
        for _ in range(1000):
            idx = np.random.choice(len(matched_partial), len(matched_partial), replace=True)
            p_mean = np.mean([matched_partial[i]['hits10'] for i in idx])
            f_mean = np.mean([matched_full[i]['hits10'] for i in idx])
            diffs.append(p_mean - f_mean)

        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

        print(f"Matched pairs: {len(matched_partial)}")
        print(f"Partial: {p_hits:.1%}, Full: {f_hits:.1%}")
        print(f"Difference: {p_hits - f_hits:+.1%}, 95% CI: [{ci_low:+.1%}, {ci_high:+.1%}]")

        if ci_low > 0:
            print(">>> PARADOX CONFIRMED: Partial > Full")
        elif ci_high < 0:
            print(">>> REVERSED: Full > Partial")
        else:
            print(">>> INCONCLUSIVE: No significant difference")
    else:
        print(f"Only {len(matched_partial)} matched pairs (insufficient)")

    return results


def main():
    print("="*70)
    print("MULTI-DATASET VALIDATION: Coverage Paradox")
    print("="*70)

    # FB15k-237
    fb_train_ds, _, fb_test_ds = load_fb15k237()
    analyze_dataset(
        "FB15k-237",
        fb_train_ds.triples,
        fb_test_ds.triples,
        fb_train_ds.num_entities,
        fb_train_ds.num_relations
    )

    # WN18RR
    wn_train_ds, _, wn_test_ds = load_wn18rr()
    analyze_dataset(
        "WN18RR",
        wn_train_ds.triples,
        wn_test_ds.triples,
        wn_train_ds.num_entities,
        wn_train_ds.num_relations
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("If paradox holds on both datasets, finding is robust.")
    print("If paradox reverses or disappears on WN18RR, it may be FB15k-specific.")


if __name__ == "__main__":
    main()
