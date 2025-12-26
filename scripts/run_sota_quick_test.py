"""
SOTA Base Models - Quick Test

Quick validation of FlexibleCAGP with different scoring functions:
- 2 models: DistMult (baseline) + ComplEx (most different architecture)
- 5 epochs (vs 20 for full version)
- Smaller evaluation set

Runtime: ~5-7 minutes on CPU
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
    """Create temporal split: early 70% as ID, late 30% as OOD."""
    n_train = int(len(triples) * train_ratio)
    train_triples = triples[:n_train]
    ood_triples = triples[n_train:]
    return train_triples, ood_triples


def prepare_ood_evaluation(train_triples, ood_triples, n_eval=10000):
    """Prepare balanced ID/OOD evaluation set (smaller for quick test)."""
    n_eval = min(n_eval, len(train_triples), len(ood_triples))

    id_sample = train_triples[np.random.choice(len(train_triples), n_eval, replace=False)]
    ood_sample = ood_triples[np.random.choice(len(ood_triples), n_eval, replace=False)]

    eval_triples = np.vstack([id_sample, ood_sample])
    eval_labels = np.concatenate([
        np.zeros(n_eval),  # ID = 0
        np.ones(n_eval)    # OOD = 1
    ])

    return torch.from_numpy(eval_triples).long(), torch.from_numpy(eval_labels).float()


def train_and_evaluate(scoring_fn, num_entities, num_relations, train_triples,
                       eval_triples, eval_labels, device, epochs=5):
    """Train model and evaluate OOD detection (quick version)."""

    print(f"\n{'='*60}")
    print(f"Testing: {scoring_fn.upper()}")
    print(f"{'='*60}")

    # Create model
    model = FlexibleCAGP(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=50,  # Smaller dim for faster training
        scoring_fn=scoring_fn,
        initial_alpha=0.5,
        learn_alpha=True,
    ).to(device)

    # Precompute coverage
    print("Precomputing coverage matrix...")
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
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # Larger batch for speed

    # Training
    trainer = FlexibleCAGPTrainer(model, lr=0.001, kl_weight=0.01)

    start_time = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader, device)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {loss:.4f}")

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")

    # Evaluation
    print("Evaluating OOD detection...")
    model.eval()

    with torch.no_grad():
        eval_triples = eval_triples.to(device)

        heads = eval_triples[:, 0]
        relations = eval_triples[:, 1]
        tails = eval_triples[:, 2]

        # Compute uncertainties
        uncertainties = model.get_uncertainty(heads, relations, tails).cpu().numpy()
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

    id_mean = id_uncertainty.mean()
    ood_mean = ood_uncertainty.mean()
    separation = ood_mean - id_mean

    print(f"\nResults:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPR:  {aupr:.4f}")
    print(f"  Separation: {separation:.3f}")
    print(f"  Learned α:  {model.get_alpha().item():.3f}")

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
    print("SOTA Base Models - QUICK TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading FB15k-237...")
    train_dataset, valid_dataset, test_dataset = load_fb15k237()

    num_entities = train_dataset.num_entities
    num_relations = train_dataset.num_relations

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")

    # Create temporal split
    print("\nCreating temporal split...")
    train_triples, ood_triples = create_temporal_split(train_dataset.triples, train_ratio=0.7)

    print(f"  Training: {len(train_triples):,} triples")
    print(f"  OOD: {len(ood_triples):,} triples")

    # Smaller evaluation set for quick test
    eval_triples, eval_labels = prepare_ood_evaluation(train_triples, ood_triples, n_eval=10000)

    n_id = (eval_labels == 0).sum()
    n_ood = (eval_labels == 1).sum()
    print(f"  Evaluation: {n_id:,} ID + {n_ood:,} OOD")

    # Test 2 scoring functions
    scoring_functions = ['distmult', 'complex']
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
            epochs=5,  # Quick test
        )
        results.append(result)

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "sota_quick_test.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("QUICK TEST SUMMARY")
    print(f"{'='*80}\n")

    print("Model        AUROC   AUPR    Separation  Learned α")
    print("-" * 60)
    for r in results:
        print(f"{r['scoring_fn']:12s} {r['auroc']:.4f}  {r['aupr']:.4f}  "
              f"{r['separation']:10.3f}  {r['learned_alpha']:9.3f}")

    print(f"\n✅ Quick test complete! Results saved to: {output_file}")

    # Check if results look reasonable
    all_good = all(r['auroc'] > 0.7 for r in results)
    if all_good:
        print("\n✅ Both models achieve >0.7 AUROC - implementation looks good!")
        print("   You can now run the full experiment with all models and 20 epochs.")
    else:
        print("\n⚠️  Some models have low AUROC (<0.7) - may need debugging")

    print(f"\nNext step: Run scripts/run_sota_base_models.py for full results")


if __name__ == "__main__":
    main()
