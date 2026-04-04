#!/usr/bin/env python3
"""
Compute Expected Calibration Error (ECE) for RCUE.
Shows whether uncertainty correlates with prediction error.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def expected_calibration_error(confidences, accuracies, n_bins=10):
    """
    Compute ECE given confidence scores and binary accuracy indicators.

    Args:
        confidences: Array of confidence scores (1 - normalized_uncertainty)
        accuracies: Binary array, 1 if prediction correct, 0 otherwise
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.sum() / total

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece

def run_ece_evaluation():
    """Run ECE evaluation on FB15k-237."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data.loaders import load_fb15k237
    from src.models.relation_conditioned import RCUE, train_rcue

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"\nFB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    # Train RCUE
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=20, verbose=True)

    rcue.eval()

    # Evaluate on test set
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    with torch.no_grad():
        # Get scores and uncertainties
        scores = rcue(h, r, t).cpu().numpy()
        uncertainties = rcue.get_uncertainty(h, r, t).cpu().numpy()

    # For ECE, we need:
    # 1. Confidence = 1 - normalized_uncertainty
    # 2. Accuracy = some measure of prediction correctness

    # Since we don't have ground truth for link prediction,
    # we use a proxy: is the score above threshold?
    # Higher score = more likely to be true triple

    # Normalize uncertainty to [0, 1]
    unc_min, unc_max = uncertainties.min(), uncertainties.max()
    norm_unc = (uncertainties - unc_min) / (unc_max - unc_min + 1e-8)
    confidences = 1 - norm_unc

    # Use positive score as proxy for "correct"
    # (test set contains true triples, so high score = correct)
    accuracies = (scores > np.median(scores)).astype(float)

    ece = expected_calibration_error(confidences, accuracies)
    print(f"\n{'='*50}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  - Lower is better (0 = perfectly calibrated)")
    print(f"{'='*50}")

    # Also report ID vs OOD calibration separately
    ood_mask = np.array([
        (h.item(), r.item()) not in coverage or (t.item(), r.item()) not in coverage
        for h, r, t in zip(test[:, 0], test[:, 1], test[:, 2])
    ])

    id_mask = ~ood_mask

    if id_mask.sum() > 100:
        ece_id = expected_calibration_error(confidences[id_mask], accuracies[id_mask])
        print(f"ECE (ID only): {ece_id:.4f}")

    if ood_mask.sum() > 100:
        ece_ood = expected_calibration_error(confidences[ood_mask], accuracies[ood_mask])
        print(f"ECE (OOD only): {ece_ood:.4f}")

    # Reliability diagram data
    print("\nReliability Diagram Data:")
    print("Bin\tConf\tAcc\tCount")
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            count = in_bin.sum()
            print(f"{i}\t{avg_conf:.3f}\t{avg_acc:.3f}\t{count}")

if __name__ == "__main__":
    run_ece_evaluation()
