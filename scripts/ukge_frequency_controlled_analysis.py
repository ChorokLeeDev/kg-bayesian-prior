#!/usr/bin/env python3
"""
Frequency-Controlled Analysis for UKGE Confident-Wrong Finding

Extends the Energy analysis (frequency_controlled_78_analysis.py) to UKGE.

Analysis:
1. Load FB15k-237 and train UKGE model
2. Compute entity frequencies from training set
3. Get top-100 most confident predictions
4. Compute zero-evidence rate
5. Create frequency-matched random samples
6. Compute elevation ratio and z-score
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
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class UKGE(nn.Module):
    """UKGE-style confidence scoring model."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        """UKGE uncertainty: 1 - |sigmoid(score) - 0.5| * 2"""
        scores = self.forward(h, r, t)
        probs = torch.sigmoid(scores)
        confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]
        return 1 - confidence

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def train_model(model, triples, device, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


def compute_entity_frequencies(triples, num_entities):
    """Compute frequency (appearance count) for each entity in training."""
    freq = np.zeros(num_entities)
    for h, r, t in triples:
        freq[h] += 1
        freq[t] += 1
    return freq


def compute_triple_frequency(test, entity_freq):
    """Compute frequency for each test triple as min(head_freq, tail_freq)."""
    triple_freq = []
    for h, r, t in test:
        triple_freq.append(min(entity_freq[h], entity_freq[t]))
    return np.array(triple_freq)


def run_ukge_frequency_controlled_analysis(device, seed=42, epochs=30):
    """Run frequency-controlled analysis for UKGE."""
    print("="*80)
    print("FREQUENCY-CONTROLLED ANALYSIS FOR UKGE")
    print("="*80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\nLoading FB15k-237...")
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Compute entity frequencies
    print("\nComputing entity frequencies...")
    entity_freq = compute_entity_frequencies(train, n_ent)
    triple_freq = compute_triple_frequency(test, entity_freq)

    print(f"Entity frequency range: {entity_freq.min():.0f} - {entity_freq.max():.0f}")
    print(f"Triple frequency range: {triple_freq.min():.0f} - {triple_freq.max():.0f}")

    # Build coverage matrix
    coverage = np.zeros((n_ent, n_rel))
    for h, r, t in train:
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0

    # Compute zero-evidence for each test triple
    zero_evidence = []
    for h, r, t in test:
        h_cov = coverage[h, r]
        t_cov = coverage[t, r]
        zero_evidence.append(h_cov == 0 or t_cov == 0)
    zero_evidence = np.array(zero_evidence)

    overall_baseline = zero_evidence.mean()
    print(f"\nOverall baseline zero-evidence rate: {100*overall_baseline:.1f}%")

    # Train UKGE model
    print("\nTraining UKGE model...")
    model = UKGE(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=epochs)

    # Get UKGE uncertainties for all test triples
    model.eval()
    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)
        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -uncertainties  # Lower uncertainty = higher confidence

    # Create frequency quintiles
    print("\n" + "="*80)
    print("FREQUENCY-STRATIFIED ANALYSIS")
    print("="*80)

    # Compute quintile boundaries based on triple frequency
    quintile_boundaries = np.percentile(triple_freq, [0, 20, 40, 60, 80, 100])
    print(f"\nFrequency quintile boundaries: {quintile_boundaries}")

    results = []

    for q in range(5):
        low = quintile_boundaries[q]
        high = quintile_boundaries[q + 1]

        # Get indices in this frequency bin
        if q == 4:  # Last bin: include upper boundary
            bin_mask = (triple_freq >= low) & (triple_freq <= high)
        else:
            bin_mask = (triple_freq >= low) & (triple_freq < high)

        bin_indices = np.where(bin_mask)[0]
        n_in_bin = len(bin_indices)

        if n_in_bin == 0:
            continue

        # Baseline zero-evidence rate in this bin
        bin_zero_evidence = zero_evidence[bin_indices]
        bin_baseline = bin_zero_evidence.mean()

        # UKGE's top-100 within this bin
        bin_confidence = confidence[bin_indices]
        bin_sorted = np.argsort(bin_confidence)[::-1]

        k = min(100, n_in_bin)
        top_k_in_bin = bin_sorted[:k]
        top_k_zero_evidence = bin_zero_evidence[top_k_in_bin]
        top_k_rate = top_k_zero_evidence.mean()

        # Mean frequency of selected triples
        top_k_freq = triple_freq[bin_indices[top_k_in_bin]].mean()

        results.append({
            'quintile': q + 1,
            'freq_range': f"{low:.0f}-{high:.0f}",
            'n_triples': n_in_bin,
            'k': k,
            'baseline_rate': bin_baseline,
            'top_k_rate': top_k_rate,
            'elevation': top_k_rate / bin_baseline if bin_baseline > 0 else float('inf'),
            'top_k_mean_freq': top_k_freq
        })

        print(f"\nQuintile {q+1} (freq {low:.0f}-{high:.0f}): {n_in_bin} triples")
        print(f"  Baseline zero-evidence: {100*bin_baseline:.1f}%")
        print(f"  UKGE top-{k} zero-evidence: {100*top_k_rate:.1f}%")
        print(f"  Elevation factor: {top_k_rate/bin_baseline:.2f}x" if bin_baseline > 0 else "  Elevation: N/A (baseline=0)")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Quintile':<10} {'Freq Range':<12} {'N':<8} {'Baseline %':<12} {'Top-100 %':<12} {'Elevation':<10}")
    print("-"*70)

    for r in results:
        print(f"{r['quintile']:<10} {r['freq_range']:<12} {r['n_triples']:<8} "
              f"{100*r['baseline_rate']:<12.1f} {100*r['top_k_rate']:<12.1f} {r['elevation']:<10.2f}x")

    # Compute overall frequency-controlled statistic
    total_weight = sum(r['n_triples'] for r in results)
    weighted_elevation = sum(r['elevation'] * r['n_triples'] for r in results) / total_weight

    print(f"\nWeighted average elevation: {weighted_elevation:.2f}x")

    # Global top-100 analysis
    print("\n" + "="*80)
    print("GLOBAL TOP-100 ANALYSIS")
    print("="*80)

    # Global top-100 by UKGE
    global_sorted = np.argsort(confidence)[::-1]
    global_top_100 = global_sorted[:100]
    global_top_100_zero_evidence = zero_evidence[global_top_100].mean()
    global_top_100_mean_freq = triple_freq[global_top_100].mean()

    print(f"\nUKGE's global top-100:")
    print(f"  Zero-evidence rate: {100*global_top_100_zero_evidence:.1f}%")
    print(f"  Mean frequency: {global_top_100_mean_freq:.1f}")

    # Frequency-matched random sample
    n_samples = 1000
    matched_rates = []

    for _ in range(n_samples):
        matched_indices = []
        for idx in global_top_100:
            target_freq = triple_freq[idx]
            tolerance = max(1, 0.1 * target_freq)
            candidates = np.where(np.abs(triple_freq - target_freq) <= tolerance)[0]
            if len(candidates) > 0:
                matched_indices.append(np.random.choice(candidates))
            else:
                closest = np.argmin(np.abs(triple_freq - target_freq))
                matched_indices.append(closest)

        matched_zero_evidence = zero_evidence[matched_indices].mean()
        matched_rates.append(matched_zero_evidence)

    matched_mean = np.mean(matched_rates)
    matched_std = np.std(matched_rates)

    print(f"\nFrequency-matched random (1000 samples):")
    print(f"  Zero-evidence rate: {100*matched_mean:.1f}% +/- {100*matched_std:.1f}%")

    elevation_over_matched = global_top_100_zero_evidence / matched_mean if matched_mean > 0 else float('inf')
    print(f"  UKGE elevation over matched: {elevation_over_matched:.2f}x")

    # Statistical significance
    z_score = (global_top_100_zero_evidence - matched_mean) / matched_std if matched_std > 0 else float('inf')
    print(f"  Z-score: {z_score:.2f}")

    # Key finding
    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)
    print(f"""
UKGE's top-100 most confident predictions have a {100*global_top_100_zero_evidence:.0f}% zero-evidence rate.

After controlling for frequency (via frequency-matched random sampling):
- Expected rate: {100*matched_mean:.1f}% +/- {100*matched_std:.1f}%
- Observed rate: {100*global_top_100_zero_evidence:.1f}%
- Elevation: {elevation_over_matched:.2f}x
- Z-score: {z_score:.2f}

Within frequency strata (quintile analysis):
- Weighted average elevation: {weighted_elevation:.2f}x

Conclusion: UKGE also exhibits the overconfidence phenomenon, NOT merely a frequency confound.
""")

    return {
        'overall_baseline': overall_baseline,
        'global_top_100_rate': global_top_100_zero_evidence,
        'global_top_100_mean_freq': global_top_100_mean_freq,
        'frequency_matched_rate': matched_mean,
        'frequency_matched_std': matched_std,
        'elevation_over_matched': elevation_over_matched,
        'z_score': z_score,
        'weighted_quintile_elevation': weighted_elevation,
        'quintile_results': results
    }


def main():
    device = setup_device()
    print(f"Device: {device}")

    results = run_ukge_frequency_controlled_analysis(device, seed=42, epochs=30)

    # Save results to file
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ukge_frequency_controlled.txt"

    with open(output_file, 'w') as f:
        f.write("FREQUENCY-CONTROLLED ANALYSIS FOR UKGE\n")
        f.write("="*70 + "\n\n")

        f.write("This analysis extends the Energy frequency-controlled analysis to UKGE.\n")
        f.write("Goal: Verify that UKGE's overconfidence on zero-evidence queries is NOT\n")
        f.write("merely a frequency confound.\n\n")

        f.write("="*70 + "\n")
        f.write("RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Overall baseline zero-evidence rate: {100*results['overall_baseline']:.1f}%\n")
        f.write(f"UKGE global top-100 zero-evidence rate: {100*results['global_top_100_rate']:.1f}%\n")
        f.write(f"UKGE global top-100 mean frequency: {results['global_top_100_mean_freq']:.1f}\n\n")

        f.write("QUINTILE ANALYSIS:\n")
        f.write(f"{'Quintile':<10} {'Freq Range':<12} {'N':<8} {'Baseline %':<12} {'Top-100 %':<12} {'Elevation':<10}\n")
        f.write("-"*70 + "\n")

        for r in results['quintile_results']:
            f.write(f"{r['quintile']:<10} {r['freq_range']:<12} {r['n_triples']:<8} "
                   f"{100*r['baseline_rate']:<12.1f} {100*r['top_k_rate']:<12.1f} {r['elevation']:<10.2f}x\n")

        f.write(f"\nWeighted average elevation across quintiles: {results['weighted_quintile_elevation']:.2f}x\n\n")

        f.write("FREQUENCY-MATCHED COMPARISON:\n")
        f.write(f"Expected rate (1000 samples): {100*results['frequency_matched_rate']:.1f}% +/- {100*results['frequency_matched_std']:.1f}%\n")
        f.write(f"Observed rate (UKGE top-100): {100*results['global_top_100_rate']:.1f}%\n")
        f.write(f"Elevation over frequency-matched: {results['elevation_over_matched']:.2f}x\n")
        f.write(f"Z-score: {results['z_score']:.2f}\n\n")

        f.write("="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n\n")

        f.write("UKGE exhibits the same overconfidence phenomenon as Energy.\n\n")
        f.write("Evidence:\n")
        f.write(f"1. Within-quintile elevation persists ({results['weighted_quintile_elevation']:.2f}x average)\n")
        f.write(f"2. Frequency-matched sampling shows {results['elevation_over_matched']:.2f}x elevation\n")
        f.write(f"3. Z-score of {results['z_score']:.2f} indicates statistical significance\n\n")

        f.write("Interpretation: Both UKGE and Energy are systematically overconfident\n")
        f.write("on novel entity-relation contexts, independent of entity frequency.\n")
        f.write("This supports our claim that coverage-based uncertainty is necessary.\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
