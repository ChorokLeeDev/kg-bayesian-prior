"""
Error Analysis for CAGP

Systematic analysis of failure modes:
1. False Positives: ID triples flagged as OOD
2. False Negatives: OOD triples missed
3. Component failures: Coverage vs GP variance
4. Patterns: Relation types, entity degrees, edge cases

This analysis provides insights for:
- Discussion section (limitations)
- Reviewer responses (thoroughness)
- Future improvements
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coverage_augmented_gpkge import CoverageAugmentedGPKGE, CoverageAugmentedGPKGETrainer
from src.data.loaders import load_fb15k237


def create_temporal_split(triples, train_ratio=0.7):
    """Create temporal split."""
    n_train = int(len(triples) * train_ratio)
    return triples[:n_train], triples[n_train:]


def prepare_ood_evaluation(train_triples, ood_triples, num_entities):
    """Prepare balanced evaluation set with metadata."""
    n_eval = min(len(train_triples), len(ood_triples))

    id_indices = np.random.choice(len(train_triples), n_eval, replace=False)
    ood_indices = np.random.choice(len(ood_triples), n_eval, replace=False)

    id_sample = train_triples[id_indices]
    ood_sample = ood_triples[ood_indices]

    eval_triples = np.vstack([id_sample, ood_sample])
    eval_labels = np.concatenate([
        np.zeros(n_eval),
        np.ones(n_eval)
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

    # Convert to arrays
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


def train_model(num_entities, num_relations, train_triples, device, epochs=20):
    """Train CAGP model for error analysis."""
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    trainer = CoverageAugmentedGPKGETrainer(model, lr=0.001, kl_weight=0.01)

    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss:.4f}")

    return model


def analyze_errors(model, eval_triples, eval_labels, entity_degrees, entity_num_relations,
                   relation_counts, device, threshold=0.5):
    """Comprehensive error analysis."""

    print("\nAnalyzing prediction errors...")

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

        # Get predictions (threshold at median for balanced classification)
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
    print(f"  True Negatives (ID correctly identified):  {tn:5d}")
    print(f"  False Positives (ID flagged as OOD):      {fp:5d}")
    print(f"  False Negatives (OOD missed):             {fn:5d}")
    print(f"  True Positives (OOD correctly detected):  {tp:5d}")

    accuracy = (tp + tn) / len(eval_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics (at threshold={threshold:.3f}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # Analyze false positives (ID flagged as OOD)
    print(f"\n{'='*60}")
    print("FALSE POSITIVE ANALYSIS (ID → OOD)")
    print(f"{'='*60}")

    fp_mask = (eval_labels == 0) & (predictions == 1)

    if fp_mask.sum() > 0:
        print(f"\nTotal false positives: {fp_mask.sum()}")

        # Entity degree analysis
        fp_head_deg = head_degrees[fp_mask]
        fp_tail_deg = tail_degrees[fp_mask]

        print(f"\nEntity degrees (FP vs all ID):")
        print(f"  FP head degree: {fp_head_deg.mean():.1f} ± {fp_head_deg.std():.1f}")
        print(f"  All ID head:    {head_degrees[eval_labels==0].mean():.1f} ± {head_degrees[eval_labels==0].std():.1f}")
        print(f"  FP tail degree: {fp_tail_deg.mean():.1f} ± {fp_tail_deg.std():.1f}")
        print(f"  All ID tail:    {tail_degrees[eval_labels==0].mean():.1f} ± {tail_degrees[eval_labels==0].std():.1f}")

        # Component analysis
        fp_total = total_unc[fp_mask]
        fp_gp = gp_var[fp_mask]
        fp_cov = coverage_unc[fp_mask]

        print(f"\nUncertainty components (FP):")
        print(f"  Total:    {fp_total.mean():.3f} ± {fp_total.std():.3f}")
        print(f"  GP var:   {fp_gp.mean():.3f} ± {fp_gp.std():.3f}")
        print(f"  Coverage: {fp_cov.mean():.3f} ± {fp_cov.std():.3f}")

        # Which component is high?
        high_gp = (fp_gp > fp_gp.mean())
        high_cov = (fp_cov > fp_cov.mean())

        print(f"\nFailure attribution:")
        print(f"  High GP variance only:  {(high_gp & ~high_cov).sum()} ({(high_gp & ~high_cov).sum()/fp_mask.sum()*100:.1f}%)")
        print(f"  High coverage only:     {(~high_gp & high_cov).sum()} ({(~high_gp & high_cov).sum()/fp_mask.sum()*100:.1f}%)")
        print(f"  Both high:              {(high_gp & high_cov).sum()} ({(high_gp & high_cov).sum()/fp_mask.sum()*100:.1f}%)")
    else:
        print("\nNo false positives! Perfect ID detection.")

    # Analyze false negatives (OOD missed)
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE ANALYSIS (OOD → ID)")
    print(f"{'='*60}")

    fn_mask = (eval_labels == 1) & (predictions == 0)

    if fn_mask.sum() > 0:
        print(f"\nTotal false negatives: {fn_mask.sum()}")

        # Entity degree analysis
        fn_head_deg = head_degrees[fn_mask]
        fn_tail_deg = tail_degrees[fn_mask]

        print(f"\nEntity degrees (FN vs all OOD):")
        print(f"  FN head degree: {fn_head_deg.mean():.1f} ± {fn_head_deg.std():.1f}")
        print(f"  All OOD head:   {head_degrees[eval_labels==1].mean():.1f} ± {head_degrees[eval_labels==1].std():.1f}")
        print(f"  FN tail degree: {fn_tail_deg.mean():.1f} ± {fn_tail_deg.std():.1f}")
        print(f"  All OOD tail:   {tail_degrees[eval_labels==1].mean():.1f} ± {tail_degrees[eval_labels==1].std():.1f}")

        # Component analysis
        fn_total = total_unc[fn_mask]
        fn_gp = gp_var[fn_mask]
        fn_cov = coverage_unc[fn_mask]

        print(f"\nUncertainty components (FN):")
        print(f"  Total:    {fn_total.mean():.3f} ± {fn_total.std():.3f}")
        print(f"  GP var:   {fn_gp.mean():.3f} ± {fn_gp.std():.3f}")
        print(f"  Coverage: {fn_cov.mean():.3f} ± {fn_cov.std():.3f}")

        # Which component is low?
        low_gp = (fn_gp < gp_var[eval_labels==1].mean())
        low_cov = (fn_cov < coverage_unc[eval_labels==1].mean())

        print(f"\nFailure attribution:")
        print(f"  Low GP variance only:   {(low_gp & ~low_cov).sum()} ({(low_gp & ~low_cov).sum()/fn_mask.sum()*100:.1f}%)")
        print(f"  Low coverage only:      {(~low_gp & low_cov).sum()} ({(~low_gp & low_cov).sum()/fn_mask.sum()*100:.1f}%)")
        print(f"  Both low:               {(low_gp & low_cov).sum()} ({(low_gp & low_cov).sum()/fn_mask.sum()*100:.1f}%)")
    else:
        print("\nNo false negatives! Perfect OOD detection.")

    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}\n")

    # Determine dominant failure mode
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    if fp_rate > fn_rate:
        print(f"⚠️  Main issue: False positives ({fp_rate*100:.1f}% of ID flagged as OOD)")
        print("   → Model is too conservative (high uncertainty on ID)")
    elif fn_rate > fp_rate:
        print(f"⚠️  Main issue: False negatives ({fn_rate*100:.1f}% of OOD missed)")
        print("   → Model is too permissive (low uncertainty on OOD)")
    else:
        print("✅ Balanced error rates")

    # AUROC
    auroc = roc_auc_score(eval_labels, total_unc)
    print(f"\nOverall AUROC: {auroc:.4f}")

    if auroc > 0.95:
        print("✅ Excellent discrimination (>0.95)")
    elif auroc > 0.85:
        print("✅ Good discrimination (0.85-0.95)")
    elif auroc > 0.75:
        print("⚠️  Moderate discrimination (0.75-0.85)")
    else:
        print("❌ Poor discrimination (<0.75)")

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
    }


def main():
    print("="*80)
    print("CAGP ERROR ANALYSIS")
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
    train_triples, ood_triples = create_temporal_split(train_dataset.triples, train_ratio=0.7)

    print(f"\nTemporal split:")
    print(f"  Training: {len(train_triples):,}")
    print(f"  OOD: {len(ood_triples):,}")

    # Compute statistics
    print("\nComputing entity and relation statistics...")
    entity_degrees, entity_num_relations = compute_entity_statistics(train_triples, num_entities)
    relation_counts = compute_relation_statistics(train_triples, num_relations)

    print(f"  Entity degree: {entity_degrees.mean():.1f} ± {entity_degrees.std():.1f} (max: {entity_degrees.max()})")
    print(f"  Entity relations: {entity_num_relations.mean():.1f} ± {entity_num_relations.std():.1f}")
    print(f"  Relation frequency: {relation_counts.mean():.1f} ± {relation_counts.std():.1f}")

    # Train model
    model = train_model(num_entities, num_relations, train_triples, device, epochs=20)

    # Prepare evaluation
    eval_triples, eval_labels = prepare_ood_evaluation(train_triples, ood_triples, num_entities)

    # Error analysis
    results = analyze_errors(
        model, eval_triples, eval_labels,
        entity_degrees, entity_num_relations, relation_counts,
        device
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_file = output_dir / "error_analysis.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Error analysis complete! Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
