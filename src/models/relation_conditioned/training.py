"""
Training and evaluation for RCUE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional

from .rcue import RCUE, RCUEWithAttention


def train_rcue(
    model: nn.Module,
    train_triples: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    margin: float = 1.0,
    unc_weight: float = 0.1,
    verbose: bool = True
) -> nn.Module:
    """
    Train RCUE model.

    Loss = L_score + unc_weight * L_uncertainty

    L_score: Margin ranking loss for link prediction
    L_uncertainty: Encourage higher uncertainty for negative samples
    """
    model = model.to(device)
    model.precompute_coverage(train_triples)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    h_all = torch.tensor(train_triples[:, 0], dtype=torch.long)
    r_all = torch.tensor(train_triples[:, 1], dtype=torch.long)
    t_all = torch.tensor(train_triples[:, 2], dtype=torch.long)

    dataset = TensorDataset(h_all, r_all, t_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_score_loss = 0.0
        total_unc_loss = 0.0

        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            # Negative sampling (corrupt tail)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)

            # Scores
            pos_scores = model(h, r, t)
            neg_scores = model(h, r, neg_t)

            # Margin ranking loss
            score_loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=margin
            )

            # Uncertainty loss: neg should have higher uncertainty
            pos_unc = model.get_uncertainty(h, r, t)
            neg_unc = model.get_uncertainty(h, r, neg_t)

            # We want neg_unc > pos_unc
            unc_loss = F.relu(pos_unc - neg_unc + 0.1).mean()

            # Total loss
            loss = score_loss + unc_weight * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_score_loss += score_loss.item()
            total_unc_loss += unc_loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={total_loss:.4f}, "
                  f"Score={total_score_loss:.4f}, "
                  f"Unc={total_unc_loss:.4f}")

    return model


def evaluate_ood_detection(
    model: nn.Module,
    test_triples: np.ndarray,
    device: torch.device,
    ood_mask: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate OOD detection using model's uncertainty.

    Args:
        model: Trained RCUE model
        test_triples: Test triples [N, 3]
        device: torch device
        ood_mask: Boolean array [N], True for OOD samples

    Returns:
        Dict with AUROC and other metrics
    """
    from sklearn.metrics import roc_auc_score

    model.eval()

    h = torch.tensor(test_triples[:, 0], dtype=torch.long, device=device)
    r = torch.tensor(test_triples[:, 1], dtype=torch.long, device=device)
    t = torch.tensor(test_triples[:, 2], dtype=torch.long, device=device)

    with torch.no_grad():
        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    # Higher uncertainty should indicate OOD
    auroc = roc_auc_score(ood_mask, uncertainties)

    return {
        'auroc': auroc,
        'mean_unc_id': uncertainties[~ood_mask].mean(),
        'mean_unc_ood': uncertainties[ood_mask].mean(),
    }


def evaluate_link_prediction(
    model: nn.Module,
    test_triples: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    Evaluate link prediction performance.

    Returns:
        Dict with MRR and Hits@k metrics
    """
    model.eval()

    ranks = []

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            h = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            r = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            t = torch.tensor(batch[:, 2], dtype=torch.long, device=device)

            # Score all tails
            scores = model.score_tails(h, r)  # [batch, num_entities]

            # Get rank of correct tail
            correct_scores = scores[torch.arange(len(h)), t]
            ranks_batch = (scores > correct_scores.unsqueeze(1)).sum(dim=1) + 1

            ranks.extend(ranks_batch.cpu().tolist())

    ranks = np.array(ranks)

    return {
        'mrr': (1.0 / ranks).mean(),
        'hits@1': (ranks == 1).mean(),
        'hits@3': (ranks <= 3).mean(),
        'hits@10': (ranks <= 10).mean(),
    }
