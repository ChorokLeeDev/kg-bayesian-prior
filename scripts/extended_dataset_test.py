#!/usr/bin/env python3
"""
Extended multi-dataset validation: YAGO3-10 and ICEWS14
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class EnergyBaseline(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)


def load_yago310():
    """Load YAGO3-10 dataset."""
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/YAGO3-10")

    # Load entity and relation mappings
    entity2id = {}
    with open(data_dir / "entity2id.txt") as f:
        n_ent = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity2id[parts[0]] = int(parts[1])

    relation2id = {}
    with open(data_dir / "relation2id.txt") as f:
        n_rel = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation2id[parts[0]] = int(parts[1])

    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            n = int(f.readline().strip())
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, t, r = parts
                    if h in entity2id and t in entity2id and r in relation2id:
                        triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)

    train = load_triples("train2id.txt")
    test = load_triples("test2id.txt")

    return train, test, n_ent, n_rel


def load_icews14():
    """Load ICEWS14 dataset."""
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/ICEWS14")

    entity2id = {}
    relation2id = {}

    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entity2id:
                        entity2id[h] = len(entity2id)
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)
                    if t not in entity2id:
                        entity2id[t] = len(entity2id)
                    triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)

    train = load_triples("train.txt")
    test = load_triples("test.txt")

    return train, test, len(entity2id), len(relation2id)


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
            result = "PARADOX"
        elif ci_high < 0:
            result = "REVERSED"
        else:
            result = "INCONCLUSIVE"
        print(f">>> {result}")
        return result, p_hits - f_hits
    else:
        print(f"Only {len(matched_partial)} matched pairs (insufficient)")
        return "INSUFFICIENT", 0


def main():
    print("="*70)
    print("EXTENDED MULTI-DATASET VALIDATION: Coverage Paradox")
    print("="*70)

    results = {}

    # YAGO3-10
    try:
        train, test, n_ent, n_rel = load_yago310()
        result, diff = analyze_dataset("YAGO3-10", train, test, n_ent, n_rel)
        results["YAGO3-10"] = (result, diff, n_rel)
    except Exception as e:
        print(f"YAGO3-10 failed: {e}")

    # ICEWS14
    try:
        train, test, n_ent, n_rel = load_icews14()
        result, diff = analyze_dataset("ICEWS14", train, test, n_ent, n_rel)
        results["ICEWS14"] = (result, diff, n_rel)
    except Exception as e:
        print(f"ICEWS14 failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: All 4 Datasets")
    print("="*70)
    print(f"{'Dataset':<15} {'Relations':<10} {'Result':<15} {'Δ (P-F)':<10}")
    print("-"*50)

    # Add previous results
    print(f"{'FB15k-237':<15} {'237':<10} {'PARADOX':<15} {'+7.5%':<10}")
    print(f"{'WN18RR':<15} {'11':<10} {'REVERSED':<15} {'-41.7%':<10}")

    for name, (result, diff, n_rel) in results.items():
        print(f"{name:<15} {n_rel:<10} {result:<15} {diff:+.1%}")

    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Does relation density predict paradox?")
    print("="*70)


if __name__ == "__main__":
    main()
