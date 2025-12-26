"""
Continuous Coverage Quick Test

Tests 3 key coverage variants for rapid validation:
1. Binary (baseline)
2. Log-scaled (theoretically motivated - matches GP variance relationship)
3. TF-IDF (balances frequency with relation rarity)

Quick settings:
- Single dataset (FB15k-237)
- Fewer epochs (20 vs 50)
- Smaller batch size for faster iterations
- Single OOD setting (temporal - more realistic than random)

Runtime: ~15-20 minutes on CPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coverage_augmented_gpkge import CoverageAugmentedGPKGE
from src.data.loaders import load_fb15k237


class ContinuousCoverageModel(CoverageAugmentedGPKGE):
    """Lightweight continuous coverage model for quick testing."""

    def __init__(self, *args, coverage_mode='binary', **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_mode = coverage_mode

        # Only create buffers for the 3 modes we're testing
        self.register_buffer('coverage_raw', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_log', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_tfidf', torch.zeros(self.num_entities, self.num_relations))

    def precompute_coverage(self, triples):
        """Precompute the 3 coverage variants."""
        # Count raw co-occurrences
        for h_idx, r_idx, t_idx in triples:
            # Binary (inherited from parent)
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

            # Raw counts
            self.coverage_raw[h_idx, r_idx] += 1.0
            self.coverage_raw[t_idx, r_idx] += 1.0

        # Log-scaled
        self.coverage_log = torch.log1p(self.coverage_raw)

        # TF-IDF
        entities_per_relation = (self.coverage_raw > 0).sum(dim=0, keepdim=True).float()
        entities_per_relation = torch.clamp(entities_per_relation, min=1.0)
        idf = torch.log(self.num_entities / entities_per_relation)
        self.coverage_tfidf = self.coverage_raw * idf

        # Normalize TF-IDF to [0, 1] per relation
        tfidf_max = self.coverage_tfidf.max(dim=0, keepdim=True)[0]
        tfidf_max = torch.clamp(tfidf_max, min=1e-8)
        self.coverage_tfidf = self.coverage_tfidf / tfidf_max

        # Print statistics
        print(f"\nCoverage statistics ({self.coverage_mode}):")
        print(f"  Binary non-zero: {self.coverage.sum().item():.0f}")
        print(f"  Raw range: [{self.coverage_raw.min():.0f}, {self.coverage_raw.max():.0f}]")
        print(f"  Log range: [{self.coverage_log.min():.2f}, {self.coverage_log.max():.2f}]")
        print(f"  TF-IDF range: [{self.coverage_tfidf.min():.2f}, {self.coverage_tfidf.max():.2f}]")

    def get_coverage_uncertainty(self, heads, relations, tails, use_frequency=None):
        """Compute coverage uncertainty based on selected mode."""
        mode = self.coverage_mode

        if mode == 'binary':
            h_cov = self.coverage[heads, relations]
            t_cov = self.coverage[tails, relations]

        elif mode == 'log':
            h_log = self.coverage_log[heads, relations]
            t_log = self.coverage_log[tails, relations]
            max_log = self.coverage_log.max() + 1e-8
            h_cov = h_log / max_log
            t_cov = t_log / max_log

        elif mode == 'tfidf':
            h_cov = self.coverage_tfidf[heads, relations]
            t_cov = self.coverage_tfidf[tails, relations]

        else:
            raise ValueError(f"Unknown coverage mode: {mode}")

        # Convert to uncertainty: higher coverage = lower uncertainty
        uncertainty = 2.0 - h_cov - t_cov
        return uncertainty


def load_dataset():
    """Load FB15k-237 dataset."""
    print("Loading FB15k-237...")
    train_ds, valid_ds, test_ds = load_fb15k237()

    # Extract triples as indexed tuples
    train = [(h, r, t) for h, r, t in train_ds.triples]
    test = [(h, r, t) for h, r, t in test_ds.triples]

    print(f"Train: {len(train)} triples")
    print(f"Test: {len(test)} triples")
    print(f"Entities: {train_ds.num_entities}, Relations: {train_ds.num_relations}")

    return {
        'train': train,
        'test': test,
        'num_entities': train_ds.num_entities,
        'num_relations': train_ds.num_relations
    }


def create_dataloader(triples, batch_size=1024):
    """Create dataloader from indexed triples."""
    heads = torch.tensor([h for h, _, _ in triples], dtype=torch.long)
    relations = torch.tensor([r for _, r, _ in triples], dtype=torch.long)
    tails = torch.tensor([t for _, _, t in triples], dtype=torch.long)

    dataset = TensorDataset(heads, relations, tails)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_temporal_ood(data, train_ratio=0.7):
    """
    Simulate temporal OOD by splitting training chronologically.

    Better approach: Use held-out validation for ID, late training for OOD.
    This ensures ID triples were actually seen during training.
    """
    train = data['train']
    split_idx = int(len(train) * train_ratio)

    # Train on early portion
    train_early = train[:split_idx]

    # OOD: late training data (never seen during training)
    train_late = train[split_idx:]

    # ID: Use early training data itself (definitely seen)
    # Take a sample to match OOD size for fair comparison
    import random
    random.seed(42)
    test_id = random.sample(train_early, min(len(train_late), len(train_early)))

    print(f"\nTemporal split:")
    print(f"  Train (early 70%): {len(train_early)} triples")
    print(f"  ID (sampled from train_early): {len(test_id)} triples")
    print(f"  OOD (late 30%): {len(train_late)} triples")

    return train_early, test_id, train_late


def evaluate_ood_detection(model, id_triples, ood_triples, device):
    """Compute OOD detection metrics."""
    model.eval()

    with torch.no_grad():
        # ID uncertainties
        h_id = torch.tensor([h for h, _, _ in id_triples], device=device)
        r_id = torch.tensor([r for _, r, _ in id_triples], device=device)
        t_id = torch.tensor([t for _, _, t in id_triples], device=device)
        id_uncertainty = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        # OOD uncertainties
        h_ood = torch.tensor([h for h, _, _ in ood_triples], device=device)
        r_ood = torch.tensor([r for _, r, _ in ood_triples], device=device)
        t_ood = torch.tensor([t for _, _, t in ood_triples], device=device)
        ood_uncertainty = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

    # Compute metrics
    labels = np.concatenate([np.zeros(len(id_uncertainty)), np.ones(len(ood_uncertainty))])
    scores = np.concatenate([id_uncertainty, ood_uncertainty])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    # Additional stats
    id_mean = np.mean(id_uncertainty)
    ood_mean = np.mean(ood_uncertainty)
    separation = ood_mean - id_mean

    return {
        'auroc': auroc,
        'aupr': aupr,
        'id_mean': id_mean,
        'ood_mean': ood_mean,
        'separation': separation
    }


def train_model(model, train_loader, device, epochs=20, lr=0.001, kl_weight=0.01):
    """Train the model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_h, batch_r, batch_t in train_loader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            # Positive scores
            pos_scores = model(batch_h, batch_r, batch_t, use_sampling=True)

            # Negative sampling
            neg_t = torch.randint(0, model.num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t, use_sampling=True)

            # BCE loss
            loss = criterion(pos_scores, torch.ones_like(pos_scores))
            loss += criterion(neg_scores, torch.zeros_like(neg_scores))

            # KL regularization
            loss += kl_weight * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    """Run quick continuous coverage test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Configuration
    coverage_modes = ['binary', 'log', 'tfidf']
    dim = 100
    epochs = 20  # Quick training
    batch_size = 1024  # Smaller for faster iterations

    # Load data
    data = load_dataset()

    # Generate temporal split
    train_triples, test_id, test_ood = generate_temporal_ood(data, train_ratio=0.7)

    results = []

    for coverage_mode in coverage_modes:
        print(f"\n{'='*70}")
        print(f"Testing Coverage Mode: {coverage_mode.upper()}")
        print(f"{'='*70}")

        # Initialize model
        model = ContinuousCoverageModel(
            num_entities=data['num_entities'],
            num_relations=data['num_relations'],
            dim=dim,
            coverage_mode=coverage_mode,
            initial_alpha=0.5,
            learn_alpha=True
        ).to(device)

        # Precompute coverage
        print("Precomputing coverage matrices...")
        model.precompute_coverage(train_triples)

        # Create dataloader
        train_loader = create_dataloader(train_triples, batch_size=batch_size)

        # Train
        start_time = time.time()
        train_model(model, train_loader, device, epochs=epochs)
        train_time = time.time() - start_time

        # Evaluate
        print(f"\nEvaluating temporal OOD detection...")
        metrics = evaluate_ood_detection(model, test_id, test_ood, device)

        # Store results (convert numpy types to Python types for JSON)
        result = {
            'coverage_mode': coverage_mode,
            'auroc': float(metrics['auroc']),
            'aupr': float(metrics['aupr']),
            'id_uncertainty_mean': float(metrics['id_mean']),
            'ood_uncertainty_mean': float(metrics['ood_mean']),
            'separation': float(metrics['separation']),
            'learned_alpha': float(model.get_alpha().item()),
            'train_time_seconds': float(train_time),
            'epochs': int(epochs)
        }
        results.append(result)

        print(f"\nResults:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPR: {metrics['aupr']:.4f}")
        print(f"  ID uncertainty: {metrics['id_mean']:.3f}")
        print(f"  OOD uncertainty: {metrics['ood_mean']:.3f}")
        print(f"  Separation: {metrics['separation']:.3f}")
        print(f"  Learned α: {result['learned_alpha']:.3f}")
        print(f"  Train time: {train_time:.1f}s")

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'continuous_coverage_quick.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Temporal OOD Detection")
    print(f"{'='*70}")
    print(f"{'Mode':<10} {'AUROC':<8} {'AUPR':<8} {'Sep.':<8} {'Alpha':<8} {'Time':<8}")
    print('-'*70)

    baseline = None
    for r in results:
        if r['coverage_mode'] == 'binary':
            baseline = r
        print(f"{r['coverage_mode']:<10} "
              f"{r['auroc']:<8.4f} "
              f"{r['aupr']:<8.4f} "
              f"{r['separation']:<8.3f} "
              f"{r['learned_alpha']:<8.3f} "
              f"{r['train_time_seconds']:<8.1f}")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    if baseline:
        best = max(results, key=lambda x: x['auroc'])
        improvement = best['auroc'] - baseline['auroc']

        print(f"\nBaseline (binary): AUROC = {baseline['auroc']:.4f}")
        print(f"Best ({best['coverage_mode']}): AUROC = {best['auroc']:.4f}")
        print(f"Improvement: {improvement:+.4f} ({improvement/baseline['auroc']*100:+.1f}%)")

        if improvement > 0.02:
            print("\n✓ FINDING: Continuous coverage significantly improves performance!")
            print("  → Recommendation: Use log-scaled or TF-IDF coverage in final model")
            print("  → Update paper methodology to use continuous coverage")
        elif improvement > -0.02:
            print("\n✓ FINDING: Binary and continuous coverage perform similarly")
            print("  → Recommendation: Keep binary coverage (simpler is better)")
            print("  → Add ablation to Appendix B to justify choice")
        else:
            print("\n✓ FINDING: Binary coverage outperforms continuous variants")
            print("  → Recommendation: Keep binary coverage")
            print("  → Investigate why: possible training frequency overfitting")

        # Alpha analysis
        print(f"\nLearned α values:")
        for r in results:
            alpha = r['learned_alpha']
            print(f"  {r['coverage_mode']:<10}: α = {alpha:.3f} "
                  f"({'relies more on coverage' if alpha < 0.5 else 'relies more on GP variance'})")

    print(f"\n{'='*70}")
    print(f"Quick test complete! Total runtime: {sum(r['train_time_seconds'] for r in results):.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
