"""
Continuous Coverage Ablation Study

Addresses reviewer concern: "Binary coverage doesn't capture co-occurrence frequency"

This script compares different continuous coverage formulations:
1. Binary (baseline): c(e,r) ∈ {0,1}
2. Raw counts: c(e,r) = count of (e,r) co-occurrences
3. Log-scaled: c(e,r) = log(1 + count)
4. Normalized: c(e,r) = count / max_count
5. Inverse frequency: c(e,r) = 1 / (1 + count)  [high freq = low uncertainty]
6. TF-IDF style: c(e,r) = count * log(N_entities / entities_with_r)

Key question: Does richer frequency information improve OOD detection?
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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coverage_augmented_gpkge import CoverageAugmentedGPKGE
from src.data.loaders import load_fb15k237, load_wn18rr


class ContinuousCoverageModel(CoverageAugmentedGPKGE):
    """Extended CAGP with multiple continuous coverage formulations."""

    def __init__(self, *args, coverage_mode='binary', **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_mode = coverage_mode

        # Additional coverage representations
        self.register_buffer('coverage_raw', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_log', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_normalized', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_inverse', torch.zeros(self.num_entities, self.num_relations))
        self.register_buffer('coverage_tfidf', torch.zeros(self.num_entities, self.num_relations))

    def precompute_coverage(self, triples, entity_to_idx=None, relation_to_idx=None):
        """Precompute all coverage variants. Triples can be indexed or need mapping."""
        # First, count raw co-occurrences
        for h, r, t in triples:
            # If mappings provided, triples are strings; otherwise they're already indices
            if entity_to_idx is not None and relation_to_idx is not None:
                h_idx = entity_to_idx[h]
                r_idx = relation_to_idx[r]
                t_idx = entity_to_idx[t]
            else:
                h_idx, r_idx, t_idx = h, r, t

            # Binary (original)
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

            # Raw counts
            self.coverage_raw[h_idx, r_idx] += 1.0
            self.coverage_raw[t_idx, r_idx] += 1.0

        # Compute derived representations
        # 1. Log-scaled
        self.coverage_log = torch.log1p(self.coverage_raw)  # log(1 + count)

        # 2. Normalized by max count per relation
        max_counts = self.coverage_raw.max(dim=0, keepdim=True)[0]
        max_counts = torch.clamp(max_counts, min=1.0)  # Avoid division by zero
        self.coverage_normalized = self.coverage_raw / max_counts

        # 3. Inverse frequency (high freq = low uncertainty)
        # uncertainty = 2 - inverse_freq gives higher values for rare (e,r)
        self.coverage_inverse = 1.0 / (1.0 + self.coverage_raw)

        # 4. TF-IDF style
        # TF: raw count, IDF: log(N_entities / entities_with_relation)
        entities_per_relation = (self.coverage_raw > 0).sum(dim=0, keepdim=True).float()
        entities_per_relation = torch.clamp(entities_per_relation, min=1.0)
        idf = torch.log(self.num_entities / entities_per_relation)
        self.coverage_tfidf = self.coverage_raw * idf

        # Normalize TF-IDF to [0, 1] range per relation for stability
        tfidf_max = self.coverage_tfidf.max(dim=0, keepdim=True)[0]
        tfidf_max = torch.clamp(tfidf_max, min=1e-8)
        self.coverage_tfidf = self.coverage_tfidf / tfidf_max

        print(f"\nCoverage statistics ({self.coverage_mode}):")
        print(f"  Binary coverage: {self.coverage.sum().item():.0f} non-zero entries")
        print(f"  Raw count range: [{self.coverage_raw.min():.0f}, {self.coverage_raw.max():.0f}]")
        print(f"  Log-scaled range: [{self.coverage_log.min():.2f}, {self.coverage_log.max():.2f}]")
        print(f"  Normalized range: [{self.coverage_normalized.min():.2f}, {self.coverage_normalized.max():.2f}]")
        print(f"  Inverse range: [{self.coverage_inverse.min():.4f}, {self.coverage_inverse.max():.4f}]")
        print(f"  TF-IDF range: [{self.coverage_tfidf.min():.2f}, {self.coverage_tfidf.max():.2f}]")

    def get_coverage_uncertainty(self, heads, relations, tails, use_frequency=None):
        """
        Compute coverage-based uncertainty using selected mode.

        For all continuous modes, uncertainty formula is:
        U_coverage = 2 - f(h,r) - f(t,r)

        where f(e,r) is the coverage function (higher value = more observed = lower uncertainty)
        """
        if use_frequency is not None:
            # Legacy compatibility
            mode = 'raw' if use_frequency else 'binary'
        else:
            mode = self.coverage_mode

        if mode == 'binary':
            h_cov = self.coverage[heads, relations]
            t_cov = self.coverage[tails, relations]

        elif mode == 'raw':
            # Normalize raw counts to [0, 1] range
            h_raw = self.coverage_raw[heads, relations]
            t_raw = self.coverage_raw[tails, relations]
            max_freq = self.coverage_raw.max() + 1e-8
            h_cov = h_raw / max_freq
            t_cov = t_raw / max_freq

        elif mode == 'log':
            # Log-scaled counts, normalized
            h_log = self.coverage_log[heads, relations]
            t_log = self.coverage_log[tails, relations]
            max_log = self.coverage_log.max() + 1e-8
            h_cov = h_log / max_log
            t_cov = t_log / max_log

        elif mode == 'normalized':
            # Already normalized by max per relation
            h_cov = self.coverage_normalized[heads, relations]
            t_cov = self.coverage_normalized[tails, relations]

        elif mode == 'inverse':
            # Inverse frequency (already in [0, 1] range after 1/(1+count))
            # But we want higher values for more observed, so take complement
            h_inv = self.coverage_inverse[heads, relations]
            t_inv = self.coverage_inverse[tails, relations]
            h_cov = 1.0 - h_inv
            t_cov = 1.0 - t_inv

        elif mode == 'tfidf':
            # TF-IDF already normalized to [0, 1]
            h_cov = self.coverage_tfidf[heads, relations]
            t_cov = self.coverage_tfidf[tails, relations]

        else:
            raise ValueError(f"Unknown coverage mode: {mode}")

        # Convert to uncertainty: 2 - coverage gives range [0, 2]
        # High coverage (well-observed) → low uncertainty
        # Low coverage (rare/novel) → high uncertainty
        uncertainty = 2.0 - h_cov - t_cov

        return uncertainty


def load_dataset(name: str):
    """Load dataset and return necessary components."""
    if name == "fb15k-237":
        train_ds, valid_ds, test_ds = load_fb15k237()
    elif name == "wn18rr":
        train_ds, valid_ds, test_ds = load_wn18rr()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Extract triples as lists of (h, r, t) with original indices
    def extract_triples(dataset):
        return [(h, r, t) for h, r, t in dataset.triples]

    train = extract_triples(train_ds)
    valid = extract_triples(valid_ds)
    test = extract_triples(test_ds)

    # Get mappings (they're the same across all splits)
    entity_to_idx = train_ds.entity_to_id
    relation_to_idx = train_ds.relation_to_id

    return {
        'train': train,
        'valid': valid,
        'test': test,
        'entity_to_idx': entity_to_idx,
        'relation_to_idx': relation_to_idx,
        'idx_to_entity': train_ds.id_to_entity,
        'idx_to_relation': train_ds.id_to_relation,
        'num_entities': train_ds.num_entities,
        'num_relations': train_ds.num_relations
    }


def create_dataloader(triples, batch_size=2048):
    """Convert triples to tensor dataloader. Triples are already indexed."""
    heads = torch.tensor([h for h, _, _ in triples], dtype=torch.long)
    relations = torch.tensor([r for _, r, _ in triples], dtype=torch.long)
    tails = torch.tensor([t for _, _, t in triples], dtype=torch.long)

    dataset = TensorDataset(heads, relations, tails)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_temporal_ood(data, train_ratio=0.7):
    """
    Simulate temporal OOD by splitting training data chronologically.

    Train on first 70% of triples, test OOD detection on remaining 30%.
    This creates realistic emerging entities + novel contexts.
    """
    train = data['train']
    split_idx = int(len(train) * train_ratio)

    train_early = train[:split_idx]
    train_late = train[split_idx:]

    # Triples are already indexed
    test_id = data['test']
    test_ood = train_late

    return train_early, test_id, test_ood


def generate_random_ood(test_triples, num_entities):
    """Generate OOD samples via random tail corruption. Triples are already indexed."""
    ood_triples = []

    for h, r, t in test_triples:
        # h, r, t are already indices
        t_corrupted = np.random.randint(0, num_entities)
        ood_triples.append((h, r, t_corrupted))

    return ood_triples


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

    return {'auroc': auroc, 'aupr': aupr}


def train_model(model, train_loader, device, epochs=50, lr=0.001, kl_weight=0.01):
    """Train the model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0

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

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    """Run continuous coverage ablation study."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Configuration
    datasets = ['fb15k-237', 'wn18rr']
    coverage_modes = ['binary', 'raw', 'log', 'normalized', 'inverse', 'tfidf']
    dim = 100
    epochs = 30  # Reduced for quick testing; use 50 for paper results

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        data = load_dataset(dataset_name)
        print(f"Entities: {data['num_entities']}, Relations: {data['num_relations']}")
        print(f"Train: {len(data['train'])}, Test: {len(data['test'])}\n")

        for coverage_mode in coverage_modes:
            print(f"\n{'-'*70}")
            print(f"Coverage Mode: {coverage_mode}")
            print(f"{'-'*70}")

            # Initialize model
            model = ContinuousCoverageModel(
                num_entities=data['num_entities'],
                num_relations=data['num_relations'],
                dim=dim,
                coverage_mode=coverage_mode,
                initial_alpha=0.5,
                learn_alpha=True
            ).to(device)

            # Precompute coverage (all variants)
            # Triples are already indexed, so no need for mappings
            model.precompute_coverage(data['train'])

            # Create dataloader
            train_loader = create_dataloader(data['train'])

            # Train
            print(f"\nTraining...")
            start_time = time.time()
            train_model(model, train_loader, device, epochs=epochs)
            train_time = time.time() - start_time
            print(f"Training time: {train_time:.1f}s")

            # Evaluate on both OOD settings
            results = {
                'dataset': dataset_name,
                'coverage_mode': coverage_mode,
                'train_time': train_time,
                'learned_alpha': model.get_alpha().item()
            }

            # 1. Random corruption OOD (easy)
            print(f"\nEvaluating random corruption OOD...")
            test_id = data['test']  # Already indexed
            test_ood_random = generate_random_ood(
                data['test'],
                data['num_entities']
            )

            metrics_random = evaluate_ood_detection(model, test_id, test_ood_random, device)
            results['random_auroc'] = metrics_random['auroc']
            results['random_aupr'] = metrics_random['aupr']
            print(f"  AUROC: {metrics_random['auroc']:.4f}, AUPR: {metrics_random['aupr']:.4f}")

            # 2. Temporal OOD (harder, more realistic)
            print(f"\nEvaluating temporal OOD...")
            _, test_id_temporal, test_ood_temporal = generate_temporal_ood(
                data,
                train_ratio=0.7
            )

            metrics_temporal = evaluate_ood_detection(model, test_id_temporal, test_ood_temporal, device)
            results['temporal_auroc'] = metrics_temporal['auroc']
            results['temporal_aupr'] = metrics_temporal['aupr']
            print(f"  AUROC: {metrics_temporal['auroc']:.4f}, AUPR: {metrics_temporal['aupr']:.4f}")

            all_results.append(results)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'continuous_coverage_ablation.json'

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Random Corruption OOD")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Mode':<12} {'AUROC':<8} {'AUPR':<8} {'Alpha':<8}")
    print('-'*70)
    for r in all_results:
        print(f"{r['dataset']:<15} {r['coverage_mode']:<12} "
              f"{r['random_auroc']:.4f}   {r['random_aupr']:.4f}   "
              f"{r['learned_alpha']:.3f}")

    print(f"\n{'='*70}")
    print("SUMMARY: Temporal OOD (More Realistic)")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Mode':<12} {'AUROC':<8} {'AUPR':<8}")
    print('-'*70)
    for r in all_results:
        print(f"{r['dataset']:<15} {r['coverage_mode']:<12} "
              f"{r['temporal_auroc']:.4f}   {r['temporal_aupr']:.4f}")

    # Analysis: Which modes work best?
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        best_random = max(dataset_results, key=lambda x: x['random_auroc'])
        best_temporal = max(dataset_results, key=lambda x: x['temporal_auroc'])
        binary_result = [r for r in dataset_results if r['coverage_mode'] == 'binary'][0]

        print(f"\n{dataset}:")
        print(f"  Best on random OOD: {best_random['coverage_mode']} "
              f"(AUROC={best_random['random_auroc']:.4f})")
        print(f"  Best on temporal OOD: {best_temporal['coverage_mode']} "
              f"(AUROC={best_temporal['temporal_auroc']:.4f})")
        print(f"  Binary baseline: "
              f"Random={binary_result['random_auroc']:.4f}, "
              f"Temporal={binary_result['temporal_auroc']:.4f}")

        improvement_random = best_random['random_auroc'] - binary_result['random_auroc']
        improvement_temporal = best_temporal['temporal_auroc'] - binary_result['temporal_auroc']

        print(f"  Improvement: Random={improvement_random:+.4f}, Temporal={improvement_temporal:+.4f}")


if __name__ == "__main__":
    main()
