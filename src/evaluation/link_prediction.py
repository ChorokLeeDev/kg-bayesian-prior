"""
Link Prediction Evaluation Metrics

Standard metrics for evaluating KGE models:
- Mean Reciprocal Rank (MRR)
- Hits@K (K = 1, 3, 10)
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm


def compute_ranks(
    scores: torch.Tensor,
    targets: torch.Tensor,
    filter_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute ranks of target entities.

    Args:
        scores: Scores for all entities, shape (batch_size, num_entities)
        targets: Target entity indices, shape (batch_size,)
        filter_mask: Boolean mask of entities to filter out (True = filter)

    Returns:
        Ranks of targets, shape (batch_size,)
    """
    batch_size = scores.size(0)

    # Apply filter mask (set filtered entities to -inf so they rank last)
    if filter_mask is not None:
        scores = scores.clone()
        scores[filter_mask] = float('-inf')

    # Get target scores
    target_scores = scores[torch.arange(batch_size), targets]

    # Count how many entities score higher than target
    ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

    return ranks


def compute_mrr(ranks: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = (1/N) * Σ (1/rank_i)
    """
    return (1.0 / ranks.float()).mean().item()


def compute_hits_at_k(
    ranks: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Hits@K.

    Hits@K = (1/N) * Σ 1[rank_i <= k]
    """
    return (ranks <= k).float().mean().item()


def evaluate_link_prediction(
    model,
    dataset,
    batch_size: int = 128,
    filter_triples: Optional[np.ndarray] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Full link prediction evaluation.

    Evaluates both head and tail prediction:
    - (?, r, t): predict head
    - (h, r, ?): predict tail

    Args:
        model: KGE model with score_heads and score_tails methods
        dataset: KGDataset with test triples
        batch_size: Batch size for evaluation
        filter_triples: All true triples for filtered ranking

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    model = model.to(device)

    # Build filter set for filtered ranking
    if filter_triples is not None:
        filter_set = set(map(tuple, filter_triples.tolist()))
    else:
        filter_set = set()

    all_ranks = []
    triples = dataset.triples

    with torch.no_grad():
        for start in tqdm(range(0, len(triples), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(triples))
            batch = triples[start:end]

            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            # Tail prediction: (h, r, ?)
            tail_scores = model.score_tails(h, r)

            # Filter known true tails (except the one we're predicting)
            for i in range(len(batch)):
                for e in range(dataset.num_entities):
                    if (batch[i, 0], batch[i, 1], e) in filter_set and e != batch[i, 2]:
                        tail_scores[i, e] = float('-inf')

            tail_ranks = compute_ranks(tail_scores, t)
            all_ranks.append(tail_ranks)

            # Head prediction: (?, r, t)
            head_scores = model.score_heads(r, t)

            # Filter known true heads
            for i in range(len(batch)):
                for e in range(dataset.num_entities):
                    if (e, batch[i, 1], batch[i, 2]) in filter_set and e != batch[i, 0]:
                        head_scores[i, e] = float('-inf')

            head_ranks = compute_ranks(head_scores, h)
            all_ranks.append(head_ranks)

    all_ranks = torch.cat(all_ranks)

    return {
        "mrr": compute_mrr(all_ranks),
        "hits@1": compute_hits_at_k(all_ranks, 1),
        "hits@3": compute_hits_at_k(all_ranks, 3),
        "hits@10": compute_hits_at_k(all_ranks, 10),
        "mean_rank": all_ranks.float().mean().item(),
    }


def evaluate_with_uncertainty(
    model,
    dataset,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, any]:
    """
    Evaluate link prediction with uncertainty information.

    Returns both accuracy metrics and uncertainty-related analyses.
    """
    model.eval()
    model = model.to(device)

    results = {
        "ranks": [],
        "uncertainties": [],
        "is_correct": [],
    }

    triples = dataset.triples

    with torch.no_grad():
        for start in tqdm(range(0, len(triples), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(triples))
            batch = triples[start:end]

            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            # Get predictions with uncertainty
            if hasattr(model, 'predict_with_uncertainty'):
                pred = model.predict_with_uncertainty(h, r, t)
                uncertainty = pred.get('total', pred.get('epistemic', torch.zeros_like(h.float())))
            else:
                uncertainty = torch.zeros_like(h.float())

            # Compute ranks for tail prediction
            tail_scores = model.score_tails(h, r)
            ranks = compute_ranks(tail_scores, t)

            results["ranks"].extend(ranks.cpu().tolist())
            results["uncertainties"].extend(uncertainty.cpu().tolist())
            results["is_correct"].extend((ranks == 1).cpu().tolist())

    # Compute correlation between uncertainty and error
    ranks = np.array(results["ranks"])
    uncertainties = np.array(results["uncertainties"])
    is_correct = np.array(results["is_correct"])

    # Spearman correlation: high uncertainty should correlate with high rank (worse)
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(uncertainties, ranks)

    return {
        "mrr": compute_mrr(torch.tensor(ranks)),
        "hits@1": compute_hits_at_k(torch.tensor(ranks), 1),
        "hits@10": compute_hits_at_k(torch.tensor(ranks), 10),
        "uncertainty_rank_correlation": corr,
        "correlation_p_value": p_value,
        "mean_uncertainty_correct": uncertainties[is_correct].mean() if is_correct.any() else 0,
        "mean_uncertainty_incorrect": uncertainties[~is_correct].mean() if (~is_correct).any() else 0,
    }
