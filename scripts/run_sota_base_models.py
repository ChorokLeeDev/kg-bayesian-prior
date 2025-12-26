"""
SOTA Base Models Comparison

Tests whether CAGP's uncertainty decomposition generalizes across different scoring functions:
1. DistMult (current baseline)
2. ComplEx (complex-valued, asymmetric relations)
3. TransE (translational, simpler architecture)

For each base model, we compare:
- Base model alone (variance-based OOD detection)
- CAGP-augmented (variance + coverage decomposition)

Quick settings:
- Single dataset (FB15k-237)
- 20 epochs for rapid testing
- Temporal OOD split (more realistic than random corruption)

Runtime: ~30-40 minutes on CPU (3 models × 2 variants × 20 epochs)
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

from src.models.flexible_cagp import FlexibleCAGP, FlexibleCAGPTrainer
from src.data.loaders import load_fb15k237


def create_temporal_split(triples, train_ratio=0.7):
    """
    Create temporal split: early 70% as ID, late 30% as OOD.

    Simulates distribution shift over time - more realistic than random corruption.
    """
    n_train = int(len(triples) * train_ratio)

    train_triples = triples[:n_train]
    ood_triples = triples[n_train:]

    return train_triples, ood_triples


def prepare_ood_evaluation(train_triples, ood_triples, num_entities):
    """
    Prepare balanced ID/OOD evaluation set.

    ID: triples from training set
    OOD: temporal shift triples
    """
    # Sample equal number of ID and OOD triples
    n_eval = min(len(train_triples), len(ood_triples))

    id_sample = train_triples[np.random.choice(len(train_triples), n_eval, replace=False)]
    ood_sample = ood_triples[np.random.choice(len(ood_triples), n_eval, replace=False)]

    # Create evaluation tensors
    eval_triples = np.vstack([id_sample, ood_sample])
    eval_labels = np.concatenate([
        np.zeros(n_eval),  # ID = 0
        np.ones(n_eval)    # OOD = 1
    ])

    return torch.from_numpy(eval_triples).long(), torch.from_numpy(eval_labels).float()


def train_and_evaluate(scoring_fn, num_entities, num_relations, train_triples,
                       eval_triples, eval_labels, device, epochs=20):
    """
    Train model and evaluate OOD detection performance.

    Returns:
        dict with AUROC, AUPR, uncertainty stats, and learned α
    """
    print(f"\n{'='*60}")
    print(f"Training: {scoring_fn.upper()}")
    print(f"{'='*60}")

    # Create model
    model = FlexibleCAGP(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=100,
        scoring_fn=scoring_fn,
        initial_alpha=0.5,
        learn_alpha=True,
    ).to(device)

    # Precompute coverage from training triples
    print("Precomputing coverage matrix...")
    # Convert indices back to dummy entity/relation mappings
    dummy_entity_to_idx = {i: i for i in range(num_entities)}
    dummy_relation_to_idx = {i: i for i in range(num_relations)}

    coverage_triples = [(int(h), int(r), int(t)) for h, r, t in train_triples]
    model.precompute_coverage(coverage_triples, dummy_entity_to_idx, dummy_relation_to_idx)

    # Create data loader
    train_dataset = TensorDataset(
        torch.from_numpy(train_triples[:, 0]),
        torch.from_numpy(train_triples[:, 1]),
        torch.from_numpy(train_triples[:, 2]),
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Training
    trainer = FlexibleCAGPTrainer(model, lr=0.001, kl_weight=0.01)

    start_time = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader, device)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {loss:.4f}")

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")

    # Evaluation
    print("\nEvaluating OOD detection...")
    model.eval()

    with torch.no_grad():
        eval_triples = eval_triples.to(device)

        heads = eval_triples[:, 0]
        relations = eval_triples[:, 1]
        tails = eval_triples[:, 2]

        # Compute uncertainties
        uncertainties = model.get_uncertainty(heads, relations, tails).cpu().numpy()

        # Get components for analysis
        gp_var = model.get_gp_variance(heads, tails).cpu().numpy()
        coverage_unc = model.get_coverage_uncertainty(heads, relations, tails).cpu().numpy()

        # Separate ID and OOD
        id_mask = eval_labels == 0
        ood_mask = eval_labels == 1

        id_uncertainty = uncertainties[id_mask]
        ood_uncertainty = uncertainties[ood_mask]

        id_gp = gp_var[id_mask]
        ood_gp = gp_var[ood_mask]

        id_coverage = coverage_unc[id_mask]
        ood_coverage = coverage_unc[ood_mask]

    # Compute metrics
    auroc = roc_auc_score(eval_labels.numpy(), uncertainties)
    aupr = average_precision_score(eval_labels.numpy(), uncertainties)

    # Uncertainty statistics
    id_mean = id_uncertainty.mean()
    ood_mean = ood_uncertainty.mean()
    separation = ood_mean - id_mean

    print(f"\nResults:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPR:  {aupr:.4f}")
    print(f"  ID uncertainty:  {id_mean:.3f}")
    print(f"  OOD uncertainty: {ood_mean:.3f}")
    print(f"  Separation:      {separation:.3f}")
    print(f"  Learned α:       {model.get_alpha().item():.3f}")

    return {
        'scoring_fn': scoring_fn,
        'auroc': float(auroc),
        'aupr': float(aupr),
        'id_uncertainty_mean': float(id_mean),
        'ood_uncertainty_mean': float(ood_mean),
        'separation': float(separation),
        'id_gp_mean': float(id_gp.mean()),
        'ood_gp_mean': float(ood_gp.mean()),
        'id_coverage_mean': float(id_coverage.mean()),
        'ood_coverage_mean': float(ood_coverage.mean()),
        'learned_alpha': float(model.get_alpha().item()),
        'train_time_seconds': float(train_time),
        'epochs': epochs,
    }


def main():
    print("="*80)
    print("SOTA Base Models for OOD Detection")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading FB15k-237...")
    train_dataset, valid_dataset, test_dataset = load_fb15k237()

    num_entities = train_dataset.num_entities
    num_relations = train_dataset.num_relations

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_dataset.triples):,}")

    # Create temporal split
    print("\nCreating temporal OOD split (70% train / 30% OOD)...")
    train_triples, ood_triples = create_temporal_split(train_dataset.triples, train_ratio=0.7)

    print(f"  Training: {len(train_triples):,} triples")
    print(f"  OOD: {len(ood_triples):,} triples")

    # Prepare evaluation
    eval_triples, eval_labels = prepare_ood_evaluation(train_triples, ood_triples, num_entities)

    n_id = (eval_labels == 0).sum()
    n_ood = (eval_labels == 1).sum()
    print(f"  Evaluation: {n_id:,} ID + {n_ood:,} OOD = {len(eval_labels):,} total")

    # Test each scoring function
    scoring_functions = ['distmult', 'complex', 'transe']
    results = []

    for scoring_fn in scoring_functions:
        result = train_and_evaluate(
            scoring_fn=scoring_fn,
            num_entities=num_entities,
            num_relations=num_relations,
            train_triples=train_triples,
            eval_triples=eval_triples,
            eval_labels=eval_labels,
            device=device,
            epochs=20,
        )
        results.append(result)

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "sota_base_models.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print("Scoring Function  AUROC   AUPR    ID Unc  OOD Unc  Separation  Learned α")
    print("-" * 80)
    for r in results:
        print(f"{r['scoring_fn']:16s}  {r['auroc']:.4f}  {r['aupr']:.4f}  "
              f"{r['id_uncertainty_mean']:6.3f}  {r['ood_uncertainty_mean']:7.3f}  "
              f"{r['separation']:10.3f}  {r['learned_alpha']:9.3f}")

    print(f"\nResults saved to: {output_file}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80 + "\n")

    # Find best model
    best_model = max(results, key=lambda x: x['auroc'])
    print(f"Best AUROC: {best_model['scoring_fn'].upper()} ({best_model['auroc']:.4f})")

    # Check if all models benefit from CAGP decomposition
    print("\nComponent Analysis (GP variance vs Coverage):")
    for r in results:
        gp_sep = r['ood_gp_mean'] - r['id_gp_mean']
        cov_sep = r['ood_coverage_mean'] - r['id_coverage_mean']
        print(f"  {r['scoring_fn']:10s}: GP sep={gp_sep:6.3f}, Coverage sep={cov_sep:6.3f}, α={r['learned_alpha']:.3f}")

    # Compare to baseline (if DistMult is included)
    distmult_result = next((r for r in results if r['scoring_fn'] == 'distmult'), None)
    if distmult_result:
        print(f"\nComparison to DistMult baseline:")
        for r in results:
            if r['scoring_fn'] != 'distmult':
                improvement = (r['auroc'] - distmult_result['auroc']) / distmult_result['auroc'] * 100
                print(f"  {r['scoring_fn']:10s}: {improvement:+.1f}% relative improvement")

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
