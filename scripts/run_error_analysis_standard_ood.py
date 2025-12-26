"""
Error Analysis for CAGP - Standard OOD Setting

Uses random tail corruption (standard OOD protocol) instead of pseudo-temporal split.
This provides meaningful failure analysis since random corruption creates actual distribution shift.

OOD Generation:
- ID: Training triples
- OOD: Corrupted triples (head, relation, random_tail)

This matches the paper's standard OOD evaluation (Table 3).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import json
from pathlib import Path
import sys
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coverage_augmented_gpkge import CoverageAugmentedGPKGE, CoverageAugmentedGPKGETrainer
from src.data.loaders import load_fb15k237


def create_standard_ood(triples, num_entities, n_samples=10000):
    """
    Create standard OOD by random tail corruption.

    This is the standard protocol: replace tail with random entity.
    Creates implausible triples that coverage should catch.
    """
    # Sample ID triples
    id_indices = np.random.choice(len(triples), n_samples, replace=False)
    id_triples = triples[id_indices]

    # Create OOD by corrupting tails
    ood_triples = id_triples.copy()
    ood_triples[:, 2] = np.random.randint(0, num_entities, size=n_samples)

    # Remove any that accidentally match ID
    valid_mask = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        if np.array_equal(id_triples[i], ood_triples[i]):
            # Retry corruption
            ood_triples[i, 2] = (ood_triples[i, 2] + 1) % num_entities

    return id_triples, ood_triples


def prepare_ood_evaluation(id_triples, ood_triples):
    """Prepare balanced evaluation set."""
    eval_triples = np.vstack([id_triples, ood_triples])
    eval_labels = np.concatenate([
        np.zeros(len(id_triples)),
        np.ones(len(ood_triples))
    ])

    return torch.from_numpy(eval_triples).long(), torch.from_numpy(eval_labels).float()


def compute_entity_statistics(triples, num_entities):
    """Compute per-entity statistics."""
    entity_degrees = defaultdict(int)
    entity_relations = defaultdict(set)

    for h, r, t in triples:
        entity_degrees[h] += 1
        entity_degrees[t] += 1
        entity_relations[h].add(r)
        entity_relations[t].add(r)

    degrees = np.array([entity_degrees.get(i, 0) for i in range(num_entities)])
    num_relations = np.array([len(entity_relations.get(i, set())) for i in range(num_entities)])

    return degrees, num_relations


def compute_relation_statistics(triples, num_relations):
    """Compute per-relation statistics."""
    relation_counts = defaultdict(int)

    for h, r, t in triples:
        relation_counts[r] += 1

    counts = np.array([relation_counts.get(i, 0) for i in range(num_relations)])
    return counts


def train_model(num_entities, num_relations, train_triples, device, epochs=50):
    """Train CAGP model."""
    print("Training CAGP model...")

    model = CoverageAugmentedGPKGE(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=100,
        initial_alpha=0.5,
        learn_alpha=True,
    ).to(device)

    # Precompute coverage
    dummy_entity_to_idx = {i: i for i in range(num_entities)}
    dummy_relation_to_idx = {i: i for i in range(num_relations)}
    coverage_triples = [(int(h), int(r), int(t)) for h, r, t in train_triples]
    model.precompute_coverage(coverage_triples, dummy_entity_to_idx, dummy_relation_to_idx)

    # Training
    train_dataset = TensorDataset(
        torch.from_numpy(train_triples[:, 0]),
        torch.from_numpy(train_triples[:, 1]),
        torch.from_numpy(train_triples[:, 2]),
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    trainer = CoverageAugmentedGPKGETrainer(model, lr=0.001, kl_weight=0.01)

    start = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader, device)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss:.4f}")

    train_time = time.time() - start
    print(f"  Training complete in {train_time/60:.1f} minutes")

    return model


def analyze_errors(model, eval_triples, eval_labels, entity_degrees, entity_num_relations,
                   relation_counts, device):
    """Comprehensive error analysis."""

    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    model.eval()

    with torch.no_grad():
        eval_triples = eval_triples.to(device)

        heads = eval_triples[:, 0]
        relations = eval_triples[:, 1]
        tails = eval_triples[:, 2]

        # Get uncertainties and components
        total_unc = model.get_uncertainty(heads, relations, tails).cpu().numpy()
        gp_var = model.get_gp_variance(heads, tails).cpu().numpy()
        coverage_unc = model.get_coverage_uncertainty(heads, relations, tails).cpu().numpy()

        # Get predictions (use median as threshold)
        threshold = np.median(total_unc)
        predictions = (total_unc > threshold).astype(int)

        # Get triple metadata
        head_degrees = entity_degrees[heads.cpu().numpy()]
        tail_degrees = entity_degrees[tails.cpu().numpy()]
        head_num_rels = entity_num_relations[heads.cpu().numpy()]
        tail_num_rels = entity_num_relations[tails.cpu().numpy()]
        rel_counts = relation_counts[relations.cpu().numpy()]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(eval_labels, predictions).ravel()

    print(f"\nConfusion Matrix:")
    print(f"  TN (ID correctly identified): {tn:6,}")
    print(f"  FP (ID flagged as OOD):       {fp:6,}")
    print(f"  FN (OOD missed):              {fn:6,}")
    print(f"  TP (OOD correctly detected):  {tp:6,}")

    accuracy = (tp + tn) / len(eval_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auroc = roc_auc_score(eval_labels, total_unc)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")

    # FALSE POSITIVE ANALYSIS
    print(f"\n{'='*80}")
    print("FALSE POSITIVE ANALYSIS (ID flagged as OOD)")
    print(f"{'='*80}")

    fp_mask = (eval_labels == 0) & (predictions == 1)

    if fp_mask.sum() > 0:
        print(f"\nCount: {fp_mask.sum():,} ({fp_mask.sum()/len(eval_labels)*100:.1f}% of all queries)")

        # Entity degree analysis
        fp_head_deg = head_degrees[fp_mask]
        fp_tail_deg = tail_degrees[fp_mask]
        id_head_deg = head_degrees[eval_labels==0]
        id_tail_deg = tail_degrees[eval_labels==0]

        print(f"\nEntity Degrees:")
        print(f"  FP head:    {fp_head_deg.mean():6.1f} ± {fp_head_deg.std():5.1f}  (ID avg: {id_head_deg.mean():6.1f})")
        print(f"  FP tail:    {fp_tail_deg.mean():6.1f} ± {fp_tail_deg.std():5.1f}  (ID avg: {id_tail_deg.mean():6.1f})")

        # Relation frequency
        fp_rel_counts = rel_counts[fp_mask]
        id_rel_counts = rel_counts[eval_labels==0]
        print(f"  FP relation freq: {fp_rel_counts.mean():6.1f}  (ID avg: {id_rel_counts.mean():6.1f})")

        # Component analysis
        fp_total = total_unc[fp_mask]
        fp_gp = gp_var[fp_mask]
        fp_cov = coverage_unc[fp_mask]

        print(f"\nUncertainty Components:")
        print(f"  Total:    {fp_total.mean():.4f} ± {fp_total.std():.4f}")
        print(f"  GP var:   {fp_gp.mean():.4f} ± {fp_gp.std():.4f}")
        print(f"  Coverage: {fp_cov.mean():.4f} ± {fp_cov.std():.4f}")

        # Component attribution
        high_gp = (fp_gp > gp_var[eval_labels==0].mean())
        high_cov = (fp_cov > coverage_unc[eval_labels==0].mean())

        print(f"\nFailure Attribution:")
        print(f"  High GP only:       {(high_gp & ~high_cov).sum():5,} ({(high_gp & ~high_cov).sum()/fp_mask.sum()*100:5.1f}%)")
        print(f"  High coverage only: {(~high_gp & high_cov).sum():5,} ({(~high_gp & high_cov).sum()/fp_mask.sum()*100:5.1f}%)")
        print(f"  Both high:          {(high_gp & high_cov).sum():5,} ({(high_gp & high_cov).sum()/fp_mask.sum()*100:5.1f}%)")

        # Pattern: low-degree entities
        low_degree_mask = (fp_head_deg < np.percentile(id_head_deg, 25)) | (fp_tail_deg < np.percentile(id_tail_deg, 25))
        print(f"\nPatterns:")
        print(f"  Low-degree entities: {low_degree_mask.sum():5,} ({low_degree_mask.sum()/fp_mask.sum()*100:5.1f}%)")
    else:
        print("\n✅ No false positives!")

    # FALSE NEGATIVE ANALYSIS
    print(f"\n{'='*80}")
    print("FALSE NEGATIVE ANALYSIS (OOD missed)")
    print(f"{'='*80}")

    fn_mask = (eval_labels == 1) & (predictions == 0)

    if fn_mask.sum() > 0:
        print(f"\nCount: {fn_mask.sum():,} ({fn_mask.sum()/len(eval_labels)*100:.1f}% of all queries)")

        # Entity degree analysis
        fn_head_deg = head_degrees[fn_mask]
        fn_tail_deg = tail_degrees[fn_mask]
        ood_head_deg = head_degrees[eval_labels==1]
        ood_tail_deg = tail_degrees[eval_labels==1]

        print(f"\nEntity Degrees:")
        print(f"  FN head:    {fn_head_deg.mean():6.1f} ± {fn_head_deg.std():5.1f}  (OOD avg: {ood_head_deg.mean():6.1f})")
        print(f"  FN tail:    {fn_tail_deg.mean():6.1f} ± {fn_tail_deg.std():5.1f}  (OOD avg: {ood_tail_deg.mean():6.1f})")

        # Component analysis
        fn_total = total_unc[fn_mask]
        fn_gp = gp_var[fn_mask]
        fn_cov = coverage_unc[fn_mask]

        print(f"\nUncertainty Components:")
        print(f"  Total:    {fn_total.mean():.4f} ± {fn_total.std():.4f}")
        print(f"  GP var:   {fn_gp.mean():.4f} ± {fn_gp.std():.4f}")
        print(f"  Coverage: {fn_cov.mean():.4f} ± {fn_cov.std():.4f}")

        # Component attribution
        low_gp = (fn_gp < gp_var[eval_labels==1].mean())
        low_cov = (fn_cov < coverage_unc[eval_labels==1].mean())

        print(f"\nFailure Attribution:")
        print(f"  Low GP only:        {(low_gp & ~low_cov).sum():5,} ({(low_gp & ~low_cov).sum()/fn_mask.sum()*100:5.1f}%)")
        print(f"  Low coverage only:  {(~low_gp & low_cov).sum():5,} ({(~low_gp & low_cov).sum()/fn_mask.sum()*100:5.1f}%)")
        print(f"  Both low:           {(low_gp & low_cov).sum():5,} ({(low_gp & low_cov).sum()/fn_mask.sum()*100:5.1f}%)")

        # Pattern: accidentally valid corruptions
        print(f"\nPatterns:")
        print(f"  Corrupted tails with coverage: {(fn_cov == 0).sum():5,} ({(fn_cov == 0).sum()/fn_mask.sum()*100:5.1f}%)")
        print(f"  → These corruptions happened to be observed (h,r) pairs")
    else:
        print("\n✅ No false negatives!")

    # KEY INSIGHTS
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    if auroc > 0.95:
        print(f"✅ Excellent discrimination (AUROC={auroc:.3f})")
    elif auroc > 0.85:
        print(f"✅ Good discrimination (AUROC={auroc:.3f})")
    else:
        print(f"⚠️  Moderate discrimination (AUROC={auroc:.3f})")

    if fp_rate > 0.1:
        print(f"\n⚠️  False positive rate: {fp_rate*100:.1f}%")
        print("   → Model flags some ID triples as OOD")
        print("   → Likely low-degree entities or rare relations")

    if fn_rate > 0.1:
        print(f"\n⚠️  False negative rate: {fn_rate*100:.1f}%")
        print("   → Model misses some OOD corruptions")
        print("   → Likely corruptions that coincidentally have coverage")

    return {
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auroc': float(auroc),
        },
        'threshold': float(threshold),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
    }


def main():
    print("="*80)
    print("CAGP ERROR ANALYSIS - Standard OOD")
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

    print(f"  Entities:  {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train:     {len(train_dataset.triples):,} triples")

    # Compute statistics on full training set
    print("\nComputing statistics...")
    entity_degrees, entity_num_relations = compute_entity_statistics(train_dataset.triples, num_entities)
    relation_counts = compute_relation_statistics(train_dataset.triples, num_relations)

    print(f"  Avg entity degree: {entity_degrees.mean():.1f}")
    print(f"  Avg entity relations: {entity_num_relations.mean():.1f}")

    # Create standard OOD (random tail corruption)
    print("\nCreating standard OOD (random tail corruption)...")
    id_triples, ood_triples = create_standard_ood(train_dataset.triples, num_entities, n_samples=20000)

    print(f"  ID samples:  {len(id_triples):,}")
    print(f"  OOD samples: {len(ood_triples):,}")

    # Train model on full training set
    model = train_model(num_entities, num_relations, train_dataset.triples, device, epochs=50)

    # Prepare evaluation
    eval_triples, eval_labels = prepare_ood_evaluation(id_triples, ood_triples)
    print(f"\nEvaluation: {len(eval_triples):,} triples ({(eval_labels==0).sum():,} ID + {(eval_labels==1).sum():,} OOD)")

    # Error analysis
    results = analyze_errors(
        model, eval_triples, eval_labels,
        entity_degrees, entity_num_relations, relation_counts,
        device
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_file = output_dir / "error_analysis_standard_ood.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
