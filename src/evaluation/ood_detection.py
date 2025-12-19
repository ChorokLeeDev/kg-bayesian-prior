"""
Out-of-Distribution (OOD) Detection Metrics

Evaluate how well uncertainty estimates distinguish:
- In-distribution (ID): samples similar to training data
- Out-of-distribution (OOD): samples from different distribution

Good uncertainty should be:
- Low for ID samples (model is confident, correctly)
- High for OOD samples (model knows it doesn't know)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve for OOD detection.

    Higher uncertainty should indicate OOD.

    Args:
        id_scores: Uncertainty scores for in-distribution samples
        ood_scores: Uncertainty scores for OOD samples

    Returns:
        AUROC (0.5 = random, 1.0 = perfect)
    """
    # Labels: 0 = ID, 1 = OOD
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    return roc_auc_score(labels, scores)


def compute_aupr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Area Under Precision-Recall Curve.

    Returns both AUPR-In (OOD as positive) and AUPR-Out (ID as positive).

    Args:
        id_scores: Uncertainty for ID samples
        ood_scores: Uncertainty for OOD samples

    Returns:
        Tuple of (AUPR_out, AUPR_in)
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    # AUPR with OOD as positive class
    aupr_out = average_precision_score(labels, scores)

    # AUPR with ID as positive class (use negative scores)
    aupr_in = average_precision_score(1 - labels, -scores)

    return aupr_out, aupr_in


def fpr_at_tpr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_threshold: float = 0.95,
) -> float:
    """
    Compute False Positive Rate at fixed True Positive Rate.

    Common metric: FPR@95TPR (FPR when we detect 95% of OOD samples)

    Args:
        id_scores: Uncertainty for ID samples
        ood_scores: Uncertainty for OOD samples
        tpr_threshold: Target TPR

    Returns:
        FPR at the specified TPR
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find threshold where TPR >= tpr_threshold
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]


def detection_error(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """
    Compute detection error at equal FPR and FNR.

    Detection Error = (FPR + FNR) / 2 at optimal threshold

    Lower is better.
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find point closest to equal error
    detection_errors = (fpr + fnr) / 2
    return detection_errors.min()


def evaluate_ood_detection(
    model,
    id_dataset,
    ood_dataset,
    batch_size: int = 128,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Full OOD detection evaluation.

    Args:
        model: Model with uncertainty estimation
        id_dataset: In-distribution dataset
        ood_dataset: Out-of-distribution dataset
        batch_size: Batch size
        device: Device to use

    Returns:
        Dict with all OOD metrics
    """
    import torch

    model.eval()
    model = model.to(device)

    def get_uncertainties(dataset):
        uncertainties = []
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch = dataset.triples[start:end]

                h = torch.tensor(batch[:, 0], device=device)
                r = torch.tensor(batch[:, 1], device=device)
                t = torch.tensor(batch[:, 2], device=device)

                if hasattr(model, 'predict_with_uncertainty'):
                    pred = model.predict_with_uncertainty(h, r, t)
                    unc = pred.get('total', pred.get('epistemic'))
                elif hasattr(model, 'get_entity_uncertainty'):
                    unc = (model.get_entity_uncertainty(h) + model.get_entity_uncertainty(t)) / 2
                else:
                    # Fallback: use negative score as proxy for uncertainty
                    unc = -model.score_triple(h, r, t)

                uncertainties.extend(unc.cpu().numpy().tolist())

        return np.array(uncertainties)

    id_uncertainties = get_uncertainties(id_dataset)
    ood_uncertainties = get_uncertainties(ood_dataset)

    auroc = compute_auroc(id_uncertainties, ood_uncertainties)
    aupr_out, aupr_in = compute_aupr(id_uncertainties, ood_uncertainties)
    fpr95 = fpr_at_tpr(id_uncertainties, ood_uncertainties, 0.95)
    det_error = detection_error(id_uncertainties, ood_uncertainties)

    return {
        "auroc": auroc,
        "aupr_out": aupr_out,
        "aupr_in": aupr_in,
        "fpr@95tpr": fpr95,
        "detection_error": det_error,
        "id_uncertainty_mean": id_uncertainties.mean(),
        "id_uncertainty_std": id_uncertainties.std(),
        "ood_uncertainty_mean": ood_uncertainties.mean(),
        "ood_uncertainty_std": ood_uncertainties.std(),
    }


def create_ood_dataset(
    train_dataset,
    test_dataset,
    ood_type: str = "random",
    num_samples: int = 1000,
) -> np.ndarray:
    """
    Create OOD samples for testing.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        ood_type: Type of OOD samples to create
            - "random": Random entity combinations
            - "unseen_entity": Entities not in training
            - "corrupted": Heavily corrupted triples
        num_samples: Number of OOD samples

    Returns:
        Array of OOD triples
    """
    num_entities = train_dataset.num_entities
    num_relations = train_dataset.num_relations

    if ood_type == "random":
        # Completely random triples
        ood_triples = np.random.randint(
            0, num_entities,
            size=(num_samples, 2)
        )
        relations = np.random.randint(0, num_relations, size=(num_samples, 1))
        ood_triples = np.hstack([
            ood_triples[:, 0:1],
            relations,
            ood_triples[:, 1:2]
        ])

    elif ood_type == "corrupted":
        # Take real triples and corrupt multiple positions
        base = test_dataset.triples[:num_samples].copy()
        for i in range(len(base)):
            # Corrupt both head and tail
            base[i, 0] = np.random.randint(num_entities)
            base[i, 2] = np.random.randint(num_entities)
        ood_triples = base

    elif ood_type == "new_relation_combo":
        # Create triples with unusual relation-entity combinations
        # (entities that don't typically appear with certain relations)
        ood_triples = []
        for r in range(num_relations):
            # Find entities that rarely appear with this relation
            relation_mask = train_dataset.triples[:, 1] == r
            common_heads = set(train_dataset.triples[relation_mask, 0])
            common_tails = set(train_dataset.triples[relation_mask, 2])

            uncommon_heads = list(set(range(num_entities)) - common_heads)
            uncommon_tails = list(set(range(num_entities)) - common_tails)

            if uncommon_heads and uncommon_tails:
                for _ in range(num_samples // num_relations):
                    h = np.random.choice(uncommon_heads)
                    t = np.random.choice(uncommon_tails)
                    ood_triples.append([h, r, t])

        ood_triples = np.array(ood_triples[:num_samples])

    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")

    return ood_triples
