#!/usr/bin/env python3
"""
Coverage-only baseline: U(e,r) = 1 - cov(e,r)
This shows whether MLP contributes beyond simple hash table lookup.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def load_fb15k237():
    """Load FB15k-237 dataset."""
    base_path = "data/raw/fb15k-237"

    entity2id = {}
    relation2id = {}

    def load_triples(split):
        triples = []
        with open(f"{base_path}/{split}.txt") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    train = load_triples("train")
    test = load_triples("test")

    return train, test, len(entity2id), len(relation2id)

def load_wn18rr():
    """Load WN18RR dataset."""
    base_path = "data/raw/wn18rr"

    entity2id = {}
    relation2id = {}

    def load_triples(split):
        triples = []
        with open(f"{base_path}/{split}.txt") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    train = load_triples("train")
    test = load_triples("test")

    return train, test, len(entity2id), len(relation2id)

def compute_coverage(train_triples, num_entities, num_relations):
    """Build coverage matrix from training data."""
    coverage = torch.zeros(num_entities, num_relations)
    for h, r, t in train_triples:
        coverage[h, r] = 1
        coverage[t, r] = 1
    return coverage

def coverage_only_auroc(test_triples, coverage):
    """Compute AUROC using only coverage as uncertainty."""
    labels = []  # 1 = OOD (novel context), 0 = ID
    uncertainties = []

    for h, r, t in test_triples:
        cov_h = coverage[h, r].item()
        cov_t = coverage[t, r].item()

        # Novel context if either head or tail has zero coverage
        is_ood = (cov_h == 0) or (cov_t == 0)
        labels.append(1 if is_ood else 0)

        # Coverage-only uncertainty: U = 1 - min(cov_h, cov_t)
        # Or equivalently: U = max(1-cov_h, 1-cov_t)
        uncertainty = max(1 - cov_h, 1 - cov_t)
        uncertainties.append(uncertainty)

    labels = np.array(labels)
    uncertainties = np.array(uncertainties)

    # AUROC
    if len(np.unique(labels)) < 2:
        return 0.5, np.mean(labels)

    auroc = roc_auc_score(labels, uncertainties)
    ood_fraction = np.mean(labels)

    return auroc, ood_fraction

def main():
    print("=" * 60)
    print("Coverage-Only Baseline Experiment")
    print("U(e,r) = 1 - min(cov(h,r), cov(t,r))")
    print("=" * 60)

    # FB15k-237
    print("\n[FB15k-237]")
    train, test, n_ent, n_rel = load_fb15k237()
    print(f"  Entities: {n_ent}, Relations: {n_rel}")
    print(f"  Train: {len(train)}, Test: {len(test)}")

    coverage = compute_coverage(train, n_ent, n_rel)
    auroc, ood_frac = coverage_only_auroc(test, coverage)
    print(f"  OOD fraction: {ood_frac:.1%}")
    print(f"  Coverage-Only AUROC: {auroc:.4f}")

    # WN18RR
    print("\n[WN18RR]")
    train, test, n_ent, n_rel = load_wn18rr()
    print(f"  Entities: {n_ent}, Relations: {n_rel}")
    print(f"  Train: {len(train)}, Test: {len(test)}")

    coverage = compute_coverage(train, n_ent, n_rel)
    auroc, ood_frac = coverage_only_auroc(test, coverage)
    print(f"  OOD fraction: {ood_frac:.1%}")
    print(f"  Coverage-Only AUROC: {auroc:.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  - If Coverage-Only AUROC ~ 1.0, the MLP adds nothing")
    print("  - If Coverage-Only AUROC < RCUE AUROC, MLP contributes")
    print("  - Coverage-Only is a binary detector, RCUE provides gradations")
    print("=" * 60)

if __name__ == "__main__":
    main()
