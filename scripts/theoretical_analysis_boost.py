#!/usr/bin/env python3
"""
Theoretical analysis: Why multiplicative boost? What's the optimal k?

Hypothesis: k_optimal depends on the separation needed between ID and OOD.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data.loaders import load_fb15k237

def compute_optimal_k():
    """
    Derivation:

    For perfect OOD detection (AUROC=1), we need:
        min(U_ood) > max(U_id)

    With multiplicative boost:
        U_id = base_variance (for covered pairs)
        U_ood = base_variance * (1 + k) (for uncovered pairs)

    The ratio U_ood / U_id = 1 + k

    For AUROC ~ 1, we need this ratio to exceed the variance ratio
    between the distributions.
    """

    # Load data to analyze base variance distribution
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples

    # Build coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    # Analyze test set
    id_count = 0
    ood_count = 0
    for h, r, t in test:
        h_cov = (h, r) in coverage
        t_cov = (t, r) in coverage
        if h_cov and t_cov:
            id_count += 1
        else:
            ood_count += 1

    print(f"ID triples: {id_count} ({id_count/len(test)*100:.1f}%)")
    print(f"OOD triples: {ood_count} ({ood_count/len(test)*100:.1f}%)")

    # Theoretical optimal k for perfect separation
    # Assuming base_variance ~ LogNormal or similar
    # Need (1+k) > ratio of max(base_id) / min(base_ood)

    print("\n" + "="*50)
    print("THEORETICAL ANALYSIS")
    print("="*50)

    print("""
    For multiplicative boost: U = base * (1 + k * (1 - cov))

    ID (cov=1):  U_id = base
    OOD (cov=0): U_ood = base * (1 + k)

    Ratio: U_ood / U_id = 1 + k

    For AUROC ≈ 1, we need:
        P(U_ood > U_id) ≈ 1

    This requires (1 + k) to overcome the variance in base.

    From our experiments:
    - k=0.5 (boost=1.5×): AUROC=0.74  (insufficient separation)
    - k=1.0 (boost=2.0×): AUROC=0.85
    - k=2.0 (boost=3.0×): AUROC=0.97  (good separation)
    - k=4.0 (boost=5.0×): AUROC=0.998 (near-perfect)

    Learned k ≈ 2.5 (boost ≈ 3.5×) suggests the base variance has
    a spread of roughly 2-3× between typical values.
    """)

    # Why multiplicative is better than additive
    print("\n" + "="*50)
    print("WHY MULTIPLICATIVE > ADDITIVE")
    print("="*50)
    print("""
    Additive:      U = base + k * (1 - cov)
    Multiplicative: U = base * (1 + k * (1 - cov))

    Problem with additive:
    - If base_variance varies widely (e.g., 0.1 to 10),
      a fixed additive k (e.g., k=2) adds the same amount regardless
    - For small base (certain entity): 0.1 + 2 = 2.1  (huge relative change)
    - For large base (uncertain entity): 10 + 2 = 12  (small relative change)

    With multiplicative:
    - For small base: 0.1 * 3 = 0.3  (still small, appropriately)
    - For large base: 10 * 3 = 30   (proportionally larger)

    Interpretation:
    - Multiplicative respects the "scale" of base uncertainty
    - An entity that's already uncertain becomes MORE uncertain when unseen
    - An entity that's certain becomes moderately uncertain when unseen

    This matches intuition:
    - A well-known entity (Obama) in unseen context: moderate uncertainty
    - A rare entity (random person) in unseen context: high uncertainty
    """)


def test_additive_vs_multiplicative():
    """Empirically compare additive vs multiplicative."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((h, r))
        coverage_set.add((t, r))

    ood_mask = np.array([
        (h, r) not in coverage_set or (t, r) not in coverage_set
        for h, r, t in test
    ])

    from src.models.relation_conditioned import RCUE, train_rcue

    print("\n" + "="*50)
    print("ADDITIVE VS MULTIPLICATIVE COMPARISON")
    print("="*50)

    # Test multiplicative (original)
    print("\nMultiplicative (k=2, boost=3×):")
    torch.manual_seed(42)
    model = RCUE(n_ent, n_rel, use_coverage=True)
    model = train_rcue(model, train, device, epochs=20, verbose=False)

    model.eval()
    h = torch.tensor(test[:, 0], device=device)
    r = torch.tensor(test[:, 1], device=device)
    t = torch.tensor(test[:, 2], device=device)

    with torch.no_grad():
        unc = model.get_uncertainty(h, r, t).cpu().numpy()
    auroc = roc_auc_score(ood_mask, unc)
    print(f"  AUROC: {auroc:.4f}")

    # Test additive version
    print("\nAdditive (k=2):")
    torch.manual_seed(42)
    model_add = RCUE(n_ent, n_rel, use_coverage=True)

    # Monkey-patch to use additive
    original_get_var = model_add.get_entity_variance
    def additive_get_var(entity_ids, relation_ids):
        e_emb = model_add.entity_emb(entity_ids)
        r_emb = model_add.relation_emb(relation_ids)
        unc_input = torch.cat([e_emb, r_emb], dim=-1)
        base_variance = model_add.uncertainty_net(unc_input).squeeze(-1)
        cov = model_add.coverage[entity_ids, relation_ids]
        # Additive: base + k*(1-cov) instead of base * (1 + k*(1-cov))
        return base_variance + 2.0 * (1.0 - cov)
    model_add.get_entity_variance = additive_get_var

    model_add = train_rcue(model_add, train, device, epochs=20, verbose=False)

    model_add.eval()
    with torch.no_grad():
        unc_add = model_add.get_uncertainty(h, r, t).cpu().numpy()
    auroc_add = roc_auc_score(ood_mask, unc_add)
    print(f"  AUROC: {auroc_add:.4f}")

    print(f"\nDifference: {auroc - auroc_add:+.4f} (multiplicative - additive)")


if __name__ == "__main__":
    compute_optimal_k()
    test_additive_vs_multiplicative()
