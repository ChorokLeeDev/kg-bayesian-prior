#!/usr/bin/env python3
"""
Proper train/val/test split experiment to verify no data leakage.

Key checks:
1. Coverage is computed from train only
2. Hyperparameters selected on validation, reported on test
3. No test data seen during any decision making
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_fb15k237
from src.models.relation_conditioned import RCUE, train_rcue


def verify_no_leakage():
    """Verify train/val/test separation is correct."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load with proper splits
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    valid = valid_ds.triples
    test = test_ds.triples

    print(f"\n{'='*60}")
    print("DATA SPLIT VERIFICATION")
    print(f"{'='*60}")
    print(f"Train: {len(train):,} triples")
    print(f"Valid: {len(valid):,} triples")
    print(f"Test:  {len(test):,} triples")

    # Build coverage from TRAIN ONLY
    coverage_train = set()
    for h, r, t in train:
        coverage_train.add((h, r))
        coverage_train.add((t, r))

    print(f"\nCoverage computed from train only: {len(coverage_train):,} (entity, relation) pairs")

    # OOD masks
    def get_ood_mask(triples, coverage):
        return np.array([
            (h, r) not in coverage or (t, r) not in coverage
            for h, r, t in triples
        ])

    ood_valid = get_ood_mask(valid, coverage_train)
    ood_test = get_ood_mask(test, coverage_train)

    print(f"\nOOD fraction (valid): {ood_valid.mean()*100:.1f}%")
    print(f"OOD fraction (test):  {ood_test.mean()*100:.1f}%")

    # Verify no overlap
    train_set = set(map(tuple, train))
    valid_set = set(map(tuple, valid))
    test_set = set(map(tuple, test))

    overlap_train_valid = len(train_set & valid_set)
    overlap_train_test = len(train_set & test_set)
    overlap_valid_test = len(valid_set & test_set)

    print(f"\nOverlap check:")
    print(f"  Train ∩ Valid: {overlap_train_valid}")
    print(f"  Train ∩ Test:  {overlap_train_test}")
    print(f"  Valid ∩ Test:  {overlap_valid_test}")

    assert overlap_train_valid == 0, "Train/Valid overlap!"
    assert overlap_train_test == 0, "Train/Test overlap!"
    print("✓ No data leakage between splits")

    # Hyperparameter selection on VALIDATION
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SELECTION (on validation)")
    print(f"{'='*60}")

    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    boost_values = [1.5, 2.0, 3.0, 5.0]
    val_results = {}

    for boost in boost_values:
        torch.manual_seed(42)
        model = RCUE(n_ent, n_rel, use_coverage=True)

        # Patch boost
        def patched_get_var(entity_ids, relation_ids, bv=boost):
            e_emb = model.entity_emb(entity_ids)
            r_emb = model.relation_emb(relation_ids)
            unc_input = torch.cat([e_emb, r_emb], dim=-1)
            base_variance = model.uncertainty_net(unc_input).squeeze(-1)
            cov = model.coverage[entity_ids, relation_ids]
            boost_factor = 1.0 + (bv - 1.0) * (1.0 - cov)
            return base_variance * boost_factor
        model.get_entity_variance = patched_get_var

        model = train_rcue(model, train, device, epochs=20, verbose=False)

        # Evaluate on VALIDATION
        model.eval()
        h = torch.tensor(valid[:, 0], device=device)
        r = torch.tensor(valid[:, 1], device=device)
        t = torch.tensor(valid[:, 2], device=device)

        with torch.no_grad():
            unc = model.get_uncertainty(h, r, t).cpu().numpy()

        val_auroc = roc_auc_score(ood_valid, unc)
        val_results[boost] = val_auroc
        print(f"  Boost={boost}x: Val AUROC={val_auroc:.4f}")

    # Select best on validation
    best_boost = max(val_results, key=val_results.get)
    print(f"\nBest boost (selected on validation): {best_boost}x")

    # Final evaluation on TEST (only once, after selection)
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION (with selected hyperparameters)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = RCUE(n_ent, n_rel, use_coverage=True)

    def final_get_var(entity_ids, relation_ids):
        e_emb = model.entity_emb(entity_ids)
        r_emb = model.relation_emb(relation_ids)
        unc_input = torch.cat([e_emb, r_emb], dim=-1)
        base_variance = model.uncertainty_net(unc_input).squeeze(-1)
        cov = model.coverage[entity_ids, relation_ids]
        boost_factor = 1.0 + (best_boost - 1.0) * (1.0 - cov)
        return base_variance * boost_factor
    model.get_entity_variance = final_get_var

    model = train_rcue(model, train, device, epochs=20, verbose=True)

    # TEST evaluation
    model.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()

    test_auroc = roc_auc_score(ood_test, unc)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Hyperparameter: boost = {best_boost}x (selected on validation)")
    print(f"Validation AUROC: {val_results[best_boost]:.4f}")
    print(f"Test AUROC:       {test_auroc:.4f}")
    print(f"\n✓ No data leakage: test seen only once, after all decisions made")


if __name__ == "__main__":
    verify_no_leakage()
