#!/usr/bin/env python3
"""
Qualitative case study: Find examples where RCUE catches OOD but Energy misses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loaders import load_fb15k237
from src.models.relation_conditioned import RCUE, train_rcue
from scripts.rcue_experiment import EnergyBaseline, train_baseline


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load data with entity/relation names
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    # Get mappings if available
    try:
        entity_names = train_ds.entity_to_id
        relation_names = train_ds.relation_to_id
        # Invert mappings
        id_to_entity = {v: k for k, v in entity_names.items()}
        id_to_relation = {v: k for k, v in relation_names.items()}
        has_names = True
    except:
        has_names = False
        print("Entity/relation names not available, using IDs")

    # Build coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage or (t, r) not in coverage
        for h, r, t in test
    ])

    # Train models
    print("Training Energy baseline...")
    torch.manual_seed(42)
    energy = EnergyBaseline(n_ent, n_rel)
    energy = train_baseline(energy, train, device, epochs=30)

    print("Training RCUE...")
    torch.manual_seed(42)
    rcue = RCUE(n_ent, n_rel, use_coverage=True)
    rcue = train_rcue(rcue, train, device, epochs=30, verbose=False)

    # Get uncertainties
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    energy.eval()
    rcue.eval()
    with torch.no_grad():
        unc_energy = energy.get_uncertainty(h, r, t).cpu().numpy()
        unc_rcue = rcue.get_uncertainty(h, r, t).cpu().numpy()

    # Normalize to [0,1] for comparison
    unc_energy_norm = (unc_energy - unc_energy.min()) / (unc_energy.max() - unc_energy.min())
    unc_rcue_norm = (unc_rcue - unc_rcue.min()) / (unc_rcue.max() - unc_rcue.min())

    # Find cases: OOD where Energy is confident but RCUE is uncertain
    # These are "saves" by RCUE
    threshold_low = 0.3  # "confident" = below this
    threshold_high = 0.7  # "uncertain" = above this

    energy_confident = unc_energy_norm < threshold_low
    rcue_uncertain = unc_rcue_norm > threshold_high

    # OOD samples where Energy is wrong (confident) but RCUE is right (uncertain)
    rcue_saves = ood_mask & energy_confident & rcue_uncertain
    save_indices = np.where(rcue_saves)[0]

    print(f"\n{'='*60}")
    print("CASE STUDY: RCUE catches OOD that Energy misses")
    print(f"{'='*60}")
    print(f"Total OOD: {ood_mask.sum()}")
    print(f"Energy confident on OOD (false negatives): {(ood_mask & energy_confident).sum()}")
    print(f"RCUE catches these: {len(save_indices)}")

    print(f"\n{'='*60}")
    print("EXAMPLE TRIPLES (RCUE saves)")
    print(f"{'='*60}")

    # Show top 10 examples
    for i, idx in enumerate(save_indices[:10]):
        h_id, r_id, t_id = test[idx]

        if has_names:
            h_name = id_to_entity.get(h_id, f"e{h_id}")
            r_name = id_to_relation.get(r_id, f"r{r_id}")
            t_name = id_to_entity.get(t_id, f"e{t_id}")
        else:
            h_name, r_name, t_name = f"e{h_id}", f"r{r_id}", f"e{t_id}"

        h_cov = (h_id, r_id) in coverage
        t_cov = (t_id, r_id) in coverage

        print(f"\n{i+1}. ({h_name}, {r_name}, {t_name})")
        print(f"   Coverage: head={(h_id,r_id) in coverage}, tail={(t_id,r_id) in coverage}")
        print(f"   Energy unc: {unc_energy[idx]:.4f} (normalized: {unc_energy_norm[idx]:.4f})")
        print(f"   RCUE unc:   {unc_rcue[idx]:.4f} (normalized: {unc_rcue_norm[idx]:.4f})")
        print(f"   → Energy: CONFIDENT (wrong), RCUE: UNCERTAIN (correct)")

    # Also find opposite: ID where both are confident (expected behavior)
    id_mask = ~ood_mask
    both_confident = id_mask & energy_confident & (unc_rcue_norm < threshold_low)
    both_idx = np.where(both_confident)[0]

    print(f"\n{'='*60}")
    print("SANITY CHECK: ID triples where both are confident (expected)")
    print(f"{'='*60}")
    print(f"Both confident on ID: {len(both_idx)}")

    for i, idx in enumerate(both_idx[:5]):
        h_id, r_id, t_id = test[idx]

        if has_names:
            h_name = id_to_entity.get(h_id, f"e{h_id}")
            r_name = id_to_relation.get(r_id, f"r{r_id}")
            t_name = id_to_entity.get(t_id, f"e{t_id}")
        else:
            h_name, r_name, t_name = f"e{h_id}", f"r{r_id}", f"e{t_id}"

        print(f"\n{i+1}. ({h_name}, {r_name}, {t_name})")
        print(f"   Coverage: head={(h_id,r_id) in coverage}, tail={(t_id,r_id) in coverage}")
        print(f"   Energy unc: {unc_energy_norm[idx]:.4f}, RCUE unc: {unc_rcue_norm[idx]:.4f}")
        print(f"   → Both confident (correct for ID)")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    # Confusion matrix style
    energy_pred_ood = unc_energy_norm > 0.5
    rcue_pred_ood = unc_rcue_norm > 0.5

    print(f"\nEnergy:")
    print(f"  True Positives (OOD detected): {(ood_mask & energy_pred_ood).sum()}")
    print(f"  False Negatives (OOD missed): {(ood_mask & ~energy_pred_ood).sum()}")
    print(f"  True Negatives (ID correct): {(~ood_mask & ~energy_pred_ood).sum()}")
    print(f"  False Positives (ID flagged): {(~ood_mask & energy_pred_ood).sum()}")

    print(f"\nRCUE:")
    print(f"  True Positives (OOD detected): {(ood_mask & rcue_pred_ood).sum()}")
    print(f"  False Negatives (OOD missed): {(ood_mask & ~rcue_pred_ood).sum()}")
    print(f"  True Negatives (ID correct): {(~ood_mask & ~rcue_pred_ood).sum()}")
    print(f"  False Positives (ID flagged): {(~ood_mask & rcue_pred_ood).sum()}")

    print(f"\nRCUE saves {(ood_mask & ~energy_pred_ood & rcue_pred_ood).sum()} OOD samples that Energy misses")


if __name__ == "__main__":
    main()
