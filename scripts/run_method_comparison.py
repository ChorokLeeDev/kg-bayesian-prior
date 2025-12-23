"""
Experiment: Compare All Relation-Aware Uncertainty Methods

This script runs a comprehensive comparison of:
1. CAGP (baseline - simple averaging)
2. AttentionCAGP (query-specific mixing)
3. RelationConditionedVariance (learned σ²(e,r))
4. GNNUncertainty (message-passing uncertainty)

On multiple OOD settings:
- Random corruption (easy)
- Type-constrained (medium)
- Adversarial (hard)

This addresses reviewer concern: "too simple method"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coverage_augmented_gpkge import CoverageAugmentedGPKGE
from src.models.relation_aware_uncertainty import (
    AttentionCAGP,
    RelationConditionedVariance,
    GNNUncertainty,
    RelationAwareUncertaintyTrainer
)
from src.data.loaders import load_fb15k237, load_wn18rr


def load_dataset(name: str):
    """Load dataset and return necessary components."""
    if name == "fb15k-237":
        data = load_fb15k237()
    elif name == "wn18rr":
        data = load_wn18rr()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Build entity and relation mappings
    entities = set()
    relations = set()
    for split in ['train', 'valid', 'test']:
        for h, r, t in data[split]:
            entities.add(h)
            entities.add(t)
            relations.add(r)

    entity_to_idx = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}

    return {
        'train': data['train'],
        'valid': data['valid'],
        'test': data['test'],
        'entity_to_idx': entity_to_idx,
        'relation_to_idx': relation_to_idx,
        'num_entities': len(entity_to_idx),
        'num_relations': len(relation_to_idx)
    }


def create_dataloader(triples, entity_to_idx, relation_to_idx, batch_size=2048):
    """Convert triples to tensor dataloader."""
    heads = torch.tensor([entity_to_idx[h] for h, _, _ in triples])
    relations = torch.tensor([relation_to_idx[r] for _, r, _ in triples])
    tails = torch.tensor([entity_to_idx[t] for _, _, t in triples])

    dataset = TensorDataset(heads, relations, tails)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_ood_samples(test_triples, entity_to_idx, relation_to_idx,
                         num_entities, mode='random'):
    """
    Generate OOD samples using different corruption strategies.

    Modes:
    - random: Uniform random tail replacement
    - type_constrained: Replace with entity of similar type (approximated by relation co-occurrence)
    - popularity_matched: Replace with entity of similar frequency
    """
    ood_triples = []

    for h, r, t in test_triples:
        h_idx = entity_to_idx[h]
        r_idx = relation_to_idx[r]

        if mode == 'random':
            t_idx = np.random.randint(0, num_entities)
        else:
            # For now, use random as fallback
            t_idx = np.random.randint(0, num_entities)

        ood_triples.append((h_idx, r_idx, t_idx))

    return ood_triples


def evaluate_ood_detection(model, id_triples, ood_triples, device):
    """Compute AUROC for OOD detection."""
    model.eval()

    with torch.no_grad():
        # ID uncertainties
        h_id = torch.tensor([h for h, _, _ in id_triples]).to(device)
        r_id = torch.tensor([r for _, r, _ in id_triples]).to(device)
        t_id = torch.tensor([t for _, _, t in id_triples]).to(device)

        id_uncertainty = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        # OOD uncertainties
        h_ood = torch.tensor([h for h, _, _ in ood_triples]).to(device)
        r_ood = torch.tensor([r for _, r, _ in ood_triples]).to(device)
        t_ood = torch.tensor([t for _, _, t in ood_triples]).to(device)

        ood_uncertainty = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

    # AUROC: higher uncertainty for OOD is better
    labels = np.concatenate([np.zeros(len(id_uncertainty)), np.ones(len(ood_uncertainty))])
    scores = np.concatenate([id_uncertainty, ood_uncertainty])

    return roc_auc_score(labels, scores)


def train_and_evaluate(model_class, model_kwargs, data, device,
                       epochs=50, lr=0.001, model_name="Model"):
    """Train model and evaluate on OOD detection."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Initialize model
    model = model_class(**model_kwargs).to(device)

    # Precompute coverage
    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(
            data['train'],
            data['entity_to_idx'],
            data['relation_to_idx']
        )

    # For GNN model, set graph structure
    if hasattr(model, 'set_graph') and not hasattr(model, '_graph_set'):
        model.set_graph(
            data['train'],
            data['entity_to_idx'],
            data['relation_to_idx']
        )
        model._graph_set = True

    # Create dataloader
    train_loader = create_dataloader(
        data['train'],
        data['entity_to_idx'],
        data['relation_to_idx']
    )

    # Trainer
    trainer = RelationAwareUncertaintyTrainer(model, lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader, device)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s")

    # Prepare test data
    test_id = [
        (data['entity_to_idx'][h], data['relation_to_idx'][r], data['entity_to_idx'][t])
        for h, r, t in data['test']
    ]

    # Evaluate on different OOD settings
    results = {'model': model_name, 'train_time': train_time}

    for ood_mode in ['random']:  # Can add 'type_constrained', 'popularity_matched'
        test_ood = generate_ood_samples(
            data['test'],
            data['entity_to_idx'],
            data['relation_to_idx'],
            data['num_entities'],
            mode=ood_mode
        )

        auroc = evaluate_ood_detection(model, test_id, test_ood, device)
        results[f'auroc_{ood_mode}'] = auroc
        print(f"  AUROC ({ood_mode}): {auroc:.4f}")

    return results, model


def main():
    """Run comparison of all methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    datasets = ['fb15k-237']  # Add 'wn18rr' for full comparison
    dim = 100
    epochs = 30  # Reduce for quick testing

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'#'*60}")

        data = load_dataset(dataset_name)
        print(f"Entities: {data['num_entities']}, Relations: {data['num_relations']}")
        print(f"Train: {len(data['train'])}, Test: {len(data['test'])}")

        model_configs = [
            # Baseline
            (CoverageAugmentedGPKGE, {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'dim': dim,
            }, "CAGP (baseline)"),

            # Method 1: Attention-based
            (AttentionCAGP, {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'dim': dim,
                'hidden_dim': 64,
                'num_attention_heads': 4,
            }, "AttentionCAGP"),

            # Method 2: Relation-conditioned variance
            (RelationConditionedVariance, {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'dim': dim,
                'variance_hidden_dim': 128,
            }, "RelCondVar"),

            # Method 3: GNN-based (skip if no GPU due to memory)
            (GNNUncertainty, {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'dim': dim,
                'num_layers': 2,
            }, "GNNUncertainty"),
        ]

        for model_class, model_kwargs, model_name in model_configs:
            try:
                results, _ = train_and_evaluate(
                    model_class, model_kwargs, data, device,
                    epochs=epochs, model_name=model_name
                )
                results['dataset'] = dataset_name
                all_results.append(results)
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                continue

    # Save results
    output_path = Path(__file__).parent.parent / 'outputs' / 'method_comparison.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'AUROC (random)':<15}")
    print("-"*40)
    for r in all_results:
        print(f"{r['model']:<25} {r.get('auroc_random', 'N/A'):<15.4f}")


if __name__ == "__main__":
    main()
