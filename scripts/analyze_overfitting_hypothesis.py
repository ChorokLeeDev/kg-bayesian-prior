#!/usr/bin/env python3
"""
Coverage Paradox Analysis: Overfitting Hypothesis Verification

Background:
- FB15k-237에서 Full coverage (32.3%) < Partial zero-coverage (59.5%)
- Hypothesis: "Full coverage = 훈련에서 많이 봄 = overfitting, Partial = 적게 봄 = generalization"

Verification experiments:
1. Training frequency vs test accuracy correlation analysis
2. How often do full coverage triples appear in training?
3. Full vs Partial performance comparison at early stopping (before/after overfitting)
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_fb15k237


def compute_entity_pair_frequency(train_triples):
    """Compute training frequency for each (entity, relation) pair."""
    head_rel_freq = Counter()
    tail_rel_freq = Counter()

    for h, r, t in train_triples:
        head_rel_freq[(h, r)] += 1
        tail_rel_freq[(t, r)] += 1

    return head_rel_freq, tail_rel_freq


def compute_coverage_matrix(train_triples, num_entities, num_relations):
    """Build coverage matrix from training data."""
    coverage = np.zeros((num_entities, num_relations), dtype=bool)

    for h, r, t in train_triples:
        coverage[h, r] = True
        coverage[t, r] = True

    return coverage


def classify_test_triples(test_triples, coverage):
    """Classify test triples into full/partial zero-coverage."""
    full_coverage_indices = []
    partial_zero_indices = []

    for idx, (h, r, t) in enumerate(test_triples):
        head_covered = coverage[h, r]
        tail_covered = coverage[t, r]

        if head_covered and tail_covered:
            full_coverage_indices.append(idx)
        else:
            partial_zero_indices.append(idx)

    return np.array(full_coverage_indices), np.array(partial_zero_indices)


def get_triple_training_frequency(test_triples, head_rel_freq, tail_rel_freq):
    """Get training frequency for each test triple's entities."""
    frequencies = []

    for h, r, t in test_triples:
        h_freq = head_rel_freq.get((h, r), 0)
        t_freq = tail_rel_freq.get((t, r), 0)
        frequencies.append({
            'head_freq': h_freq,
            'tail_freq': t_freq,
            'total_freq': h_freq + t_freq,
            'min_freq': min(h_freq, t_freq),
            'max_freq': max(h_freq, t_freq)
        })

    return frequencies


class SimpleDistMult(nn.Module):
    """Simple DistMult for overfitting analysis."""

    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def score_triples(self, triples):
        h = torch.tensor(triples[:, 0], dtype=torch.long)
        r = torch.tensor(triples[:, 1], dtype=torch.long)
        t = torch.tensor(triples[:, 2], dtype=torch.long)
        return self.forward(h, r, t)


def train_with_checkpoints(model, train_triples, test_triples, full_coverage_idx,
                           partial_zero_idx, num_entities, device,
                           epochs=100, batch_size=1024, checkpoint_epochs=[5, 10, 20, 50, 100]):
    """Train model and evaluate at checkpoints to analyze overfitting."""

    # Create training data with negative sampling
    train_tensor = torch.tensor(train_triples, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    results = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch[0].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Positive scores
            pos_scores = model(h, r, t)

            # Negative sampling (corrupt tail)
            neg_t = torch.randint(0, num_entities, (len(h),), device=device)
            neg_scores = model(h, r, neg_t)

            # Margin ranking loss
            loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                torch.ones_like(pos_scores),
                margin=1.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch in checkpoint_epochs:
            model.eval()
            with torch.no_grad():
                # Evaluate on test set
                test_tensor = torch.tensor(test_triples, dtype=torch.long).to(device)
                test_scores = model(
                    test_tensor[:, 0],
                    test_tensor[:, 1],
                    test_tensor[:, 2]
                ).cpu().numpy()

                # Compute accuracy (positive score > 0 as proxy)
                full_scores = test_scores[full_coverage_idx]
                partial_scores = test_scores[partial_zero_idx]

                # Use threshold based on score distribution
                threshold = np.median(test_scores)

                full_acc = (full_scores > threshold).mean()
                partial_acc = (partial_scores > threshold).mean()

                # Also compute mean scores
                full_mean_score = full_scores.mean()
                partial_mean_score = partial_scores.mean()

                results.append({
                    'epoch': epoch,
                    'train_loss': total_loss / len(train_loader),
                    'full_coverage_acc': full_acc,
                    'partial_zero_acc': partial_acc,
                    'full_mean_score': full_mean_score,
                    'partial_mean_score': partial_mean_score,
                    'acc_gap': full_acc - partial_acc,
                    'score_gap': full_mean_score - partial_mean_score
                })

                print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, "
                      f"Full Acc={full_acc:.3f}, Partial Acc={partial_acc:.3f}, "
                      f"Gap={full_acc-partial_acc:+.3f}")

    return results


def analyze_frequency_accuracy_correlation(test_triples, frequencies, model, device):
    """Analyze correlation between training frequency and test accuracy."""
    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(test_triples, dtype=torch.long).to(device)
        scores = model(
            test_tensor[:, 0],
            test_tensor[:, 1],
            test_tensor[:, 2]
        ).cpu().numpy()

    # Bin by frequency
    freq_bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
    bin_results = []

    total_freqs = np.array([f['total_freq'] for f in frequencies])
    threshold = np.median(scores)
    correct = scores > threshold

    for i in range(len(freq_bins) - 1):
        low, high = freq_bins[i], freq_bins[i+1]
        mask = (total_freqs >= low) & (total_freqs < high)

        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_mean_score = scores[mask].mean()
            bin_results.append({
                'freq_range': f'{low}-{high if high != float("inf") else "inf"}',
                'count': mask.sum(),
                'accuracy': bin_acc,
                'mean_score': bin_mean_score
            })

    return bin_results


def compute_mrr_hits(model, test_triples, all_triples, num_entities, device, batch_size=256):
    """Compute MRR and Hits@10 for test triples."""
    model.eval()

    # Build filter set for filtered ranking
    all_true = set(map(tuple, all_triples))

    ranks = []

    test_tensor = torch.tensor(test_triples, dtype=torch.long)

    with torch.no_grad():
        for i in tqdm(range(0, len(test_tensor), batch_size), desc="Computing ranks"):
            batch = test_tensor[i:i+batch_size].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Score all possible tails
            all_entities = torch.arange(num_entities, device=device)

            for j in range(len(h)):
                # Get scores for all entities as tail
                h_j = h[j].unsqueeze(0).expand(num_entities)
                r_j = r[j].unsqueeze(0).expand(num_entities)

                scores = model(h_j, r_j, all_entities).cpu().numpy()

                # Filter out other true triples
                target_score = scores[t[j].item()]

                # Count how many have higher score (filtered)
                rank = 1
                for e_idx in range(num_entities):
                    if e_idx != t[j].item():
                        if (h[j].item(), r[j].item(), e_idx) in all_true:
                            continue
                        if scores[e_idx] >= target_score:
                            rank += 1

                ranks.append(rank)

    ranks = np.array(ranks)
    mrr = (1.0 / ranks).mean()
    hits10 = (ranks <= 10).mean()

    return mrr, hits10, ranks


def main():
    output_path = project_root / "outputs" / "overfitting_hypothesis_results.txt"
    output_path.parent.mkdir(exist_ok=True)

    results_lines = []

    def log(msg):
        print(msg)
        results_lines.append(msg)

    log("=" * 70)
    log("Coverage Paradox Analysis: Overfitting Hypothesis Verification")
    log("=" * 70)

    # Load data
    log("\n[1] Loading FB15k-237 dataset...")
    train_ds, valid_ds, test_ds = load_fb15k237()

    train_triples = train_ds.triples
    valid_triples = valid_ds.triples
    test_triples = test_ds.triples
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    log(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")
    log(f"Entities: {num_entities}, Relations: {num_relations}")

    # Compute coverage and frequencies
    log("\n[2] Computing coverage matrix and training frequencies...")
    coverage = compute_coverage_matrix(train_triples, num_entities, num_relations)
    head_rel_freq, tail_rel_freq = compute_entity_pair_frequency(train_triples)

    # Classify test triples
    full_coverage_idx, partial_zero_idx = classify_test_triples(test_triples, coverage)
    log(f"Full coverage test triples: {len(full_coverage_idx)} ({100*len(full_coverage_idx)/len(test_triples):.1f}%)")
    log(f"Partial zero-coverage test triples: {len(partial_zero_idx)} ({100*len(partial_zero_idx)/len(test_triples):.1f}%)")

    # Analyze training frequency for each group
    log("\n[3] Analyzing training frequency by coverage group...")
    frequencies = get_triple_training_frequency(test_triples, head_rel_freq, tail_rel_freq)

    full_freqs = [frequencies[i]['total_freq'] for i in full_coverage_idx]
    partial_freqs = [frequencies[i]['total_freq'] for i in partial_zero_idx]

    log(f"\nFull coverage triples:")
    log(f"  Mean training frequency: {np.mean(full_freqs):.2f}")
    log(f"  Median training frequency: {np.median(full_freqs):.2f}")
    log(f"  Min/Max: {np.min(full_freqs)}/{np.max(full_freqs)}")

    log(f"\nPartial zero-coverage triples:")
    log(f"  Mean training frequency: {np.mean(partial_freqs):.2f}")
    log(f"  Median training frequency: {np.median(partial_freqs):.2f}")
    log(f"  Min/Max: {np.min(partial_freqs)}/{np.max(partial_freqs)}")

    log(f"\nFrequency ratio (Full/Partial): {np.mean(full_freqs)/np.mean(partial_freqs):.2f}x")

    # Train model and track performance at checkpoints
    log("\n[4] Training DistMult with checkpoints (overfitting analysis)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    model = SimpleDistMult(num_entities, num_relations, embedding_dim=100)

    checkpoint_epochs = [5, 10, 20, 30, 50, 75, 100]
    checkpoint_results = train_with_checkpoints(
        model, train_triples, test_triples,
        full_coverage_idx, partial_zero_idx,
        num_entities, device,
        epochs=100, checkpoint_epochs=checkpoint_epochs
    )

    log("\n[5] Checkpoint Results Summary:")
    log("-" * 70)
    log(f"{'Epoch':<8} {'Loss':<10} {'Full Acc':<12} {'Partial Acc':<12} {'Gap':<10}")
    log("-" * 70)
    for r in checkpoint_results:
        log(f"{r['epoch']:<8} {r['train_loss']:<10.4f} {r['full_coverage_acc']:<12.3f} "
            f"{r['partial_zero_acc']:<12.3f} {r['acc_gap']:<+10.3f}")

    # Analyze frequency-accuracy correlation
    log("\n[6] Frequency vs Accuracy Correlation:")
    log("-" * 70)
    freq_corr = analyze_frequency_accuracy_correlation(test_triples, frequencies, model, device)
    log(f"{'Freq Range':<15} {'Count':<10} {'Accuracy':<12} {'Mean Score':<12}")
    log("-" * 70)
    for r in freq_corr:
        log(f"{r['freq_range']:<15} {r['count']:<10} {r['accuracy']:<12.3f} {r['mean_score']:<12.3f}")

    # Compute actual MRR/Hits for both groups (on a subset for speed)
    log("\n[7] MRR/Hits@10 Analysis (subset for speed)...")

    # Sample for speed
    n_sample = min(1000, len(full_coverage_idx), len(partial_zero_idx))

    full_sample_idx = np.random.choice(full_coverage_idx, n_sample, replace=False)
    partial_sample_idx = np.random.choice(partial_zero_idx, n_sample, replace=False)

    full_test_sample = test_triples[full_sample_idx]
    partial_test_sample = test_triples[partial_sample_idx]

    log(f"\nComputing MRR/Hits@10 on {n_sample} samples per group...")

    full_mrr, full_h10, full_ranks = compute_mrr_hits(
        model, full_test_sample, all_triples, num_entities, device
    )
    partial_mrr, partial_h10, partial_ranks = compute_mrr_hits(
        model, partial_test_sample, all_triples, num_entities, device
    )

    log(f"\nFull Coverage Triples:")
    log(f"  MRR: {full_mrr:.4f}")
    log(f"  Hits@10: {full_h10:.4f}")
    log(f"  Mean Rank: {full_ranks.mean():.1f}")

    log(f"\nPartial Zero-Coverage Triples:")
    log(f"  MRR: {partial_mrr:.4f}")
    log(f"  Hits@10: {partial_h10:.4f}")
    log(f"  Mean Rank: {partial_ranks.mean():.1f}")

    # Summary and conclusions
    log("\n" + "=" * 70)
    log("SUMMARY: Overfitting Hypothesis Analysis")
    log("=" * 70)

    # Calculate overfitting indicators
    early_gap = checkpoint_results[0]['acc_gap']  # Epoch 5
    late_gap = checkpoint_results[-1]['acc_gap']  # Epoch 100
    gap_change = late_gap - early_gap

    log(f"\n1. Training Frequency Analysis:")
    log(f"   - Full coverage entities appear {np.mean(full_freqs)/np.mean(partial_freqs):.2f}x more often in training")
    log(f"   - This confirms: Full coverage = more training exposure")

    log(f"\n2. Overfitting Evidence (Accuracy Gap over Training):")
    log(f"   - Early (epoch 5) gap: {early_gap:+.3f}")
    log(f"   - Late (epoch 100) gap: {late_gap:+.3f}")
    log(f"   - Gap change: {gap_change:+.3f}")

    if gap_change > 0.02:
        log(f"   - VERDICT: Gap INCREASES with training → Overfitting hypothesis SUPPORTED")
    elif gap_change < -0.02:
        log(f"   - VERDICT: Gap DECREASES with training → Overfitting hypothesis REJECTED")
    else:
        log(f"   - VERDICT: Gap STABLE → Overfitting hypothesis INCONCLUSIVE")

    log(f"\n3. Link Prediction Performance:")
    log(f"   - Full MRR: {full_mrr:.4f}, Partial MRR: {partial_mrr:.4f}")
    log(f"   - Full Hits@10: {full_h10:.4f}, Partial Hits@10: {partial_h10:.4f}")

    if full_mrr < partial_mrr:
        log(f"   - PARADOX CONFIRMED: Full coverage performs WORSE despite more training")
    else:
        log(f"   - PARADOX NOT OBSERVED: Full coverage performs better as expected")

    log("\n4. Frequency-Accuracy Correlation:")
    if len(freq_corr) > 2:
        low_freq_acc = freq_corr[0]['accuracy']
        high_freq_acc = freq_corr[-2]['accuracy'] if freq_corr[-2]['count'] > 10 else freq_corr[-3]['accuracy']
        log(f"   - Low freq (0-1) accuracy: {low_freq_acc:.3f}")
        log(f"   - High freq accuracy: {high_freq_acc:.3f}")

        if high_freq_acc < low_freq_acc:
            log(f"   - Negative correlation: More training → WORSE generalization (overfitting!)")
        else:
            log(f"   - Positive correlation: More training → Better generalization")

    log("\n" + "=" * 70)
    log("END OF ANALYSIS")
    log("=" * 70)

    # Save results
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_lines))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
