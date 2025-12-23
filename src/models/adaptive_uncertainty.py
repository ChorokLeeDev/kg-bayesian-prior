"""
Adaptive Uncertainty Estimation for KG Link Prediction

Based on experimental findings, combines multiple uncertainty signals
adaptively for robust OOD detection across different attack types.

Key Findings:
1. Ensemble negative score (Energy) works best for random/type-constrained OOD
2. Neighborhood isolation helps detect high-score adversarial attacks
3. Structural (Coverage) remains a strong baseline
4. Margin is effective for selective prediction (abstention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Literal


class AdaptiveUncertaintyKGE(nn.Module):
    """
    KG Embedding model with adaptive uncertainty estimation.

    Combines multiple uncertainty signals:
    - Semantic: GP variance from entity embeddings
    - Structural: Coverage-based uncertainty
    - Predictive: Score-based (Energy-like)
    - Neighborhood: Local consistency analysis
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 100,
        neighbor_k: int = 10,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.neighbor_k = neighbor_k

        # Entity embeddings (variational)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage matrix
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Learnable combination weights for different contexts
        # [semantic, structural, score_based, neighborhood]
        self.uncertainty_weights = nn.Parameter(torch.tensor([0.1, 0.3, 0.4, 0.2]))

    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """Compute triple scores using DistMult."""
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]
        return (h * r * t).sum(dim=-1)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Score all possible tails."""
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        hr = h * r
        return hr @ self.entity_mean.T

    # =========================================================================
    # Individual Uncertainty Components
    # =========================================================================

    def get_semantic_uncertainty(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """GP variance-based uncertainty."""
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_structural_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                    tails: torch.Tensor) -> torch.Tensor:
        """Coverage-based uncertainty."""
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_score_based_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                     tails: torch.Tensor) -> torch.Tensor:
        """
        Negative score as uncertainty (Energy-based).

        Low score = high uncertainty = potential OOD
        """
        scores = self.forward(heads, relations, tails)
        return -scores  # Negative score as uncertainty

    def get_rank_based_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                    tails: torch.Tensor) -> torch.Tensor:
        """Rank of tail among all candidates."""
        scores = self.score_all_tails(heads, relations)
        tail_scores = scores.gather(1, tails.unsqueeze(1)).squeeze(1)
        ranks = (scores > tail_scores.unsqueeze(1)).sum(dim=1).float() + 1
        return ranks

    def get_neighborhood_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                      tails: torch.Tensor) -> torch.Tensor:
        """
        Neighborhood isolation: gap between tail score and neighbor scores.

        High gap = tail is isolated from similar entities = potential OOD
        """
        # Get tail embeddings and find k-NN
        tail_embs = self.entity_mean[tails]
        dists = torch.cdist(tail_embs, self.entity_mean)
        dists.scatter_(1, tails.unsqueeze(1), float('inf'))
        _, nn_idx = torch.topk(dists, self.neighbor_k, dim=1, largest=False)

        # Score neighbors
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        hr = h * r
        nn_embs = self.entity_mean[nn_idx]
        nn_scores = (hr.unsqueeze(1) * nn_embs).sum(dim=-1)

        # Get tail score
        tail_scores = self.forward(heads, relations, tails)

        # Gap = tail_score - neighbor_mean
        # High gap might indicate isolated adversarial tail
        neighbor_mean = nn_scores.mean(dim=1)
        neighbor_std = nn_scores.std(dim=1) + 1e-8

        # Isolation = gap normalized by neighbor variance
        isolation = (tail_scores - neighbor_mean) / neighbor_std

        return isolation

    def get_prediction_margin(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Margin between top-1 and top-2 predictions."""
        scores = self.score_all_tails(heads, relations)
        topk, _ = torch.topk(scores, k=2, dim=-1)
        return topk[:, 0] - topk[:, 1]

    # =========================================================================
    # Combined Uncertainty
    # =========================================================================

    def get_uncertainty(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        return_components: bool = False,
        method: Literal['adaptive', 'structural', 'score', 'ensemble'] = 'adaptive',
    ) -> torch.Tensor:
        """
        Compute combined uncertainty.

        Args:
            method: 'adaptive' - learned weighted combination
                    'structural' - coverage only (baseline)
                    'score' - negative score only (Energy-like)
                    'ensemble' - for use with ensemble models
        """
        if method == 'structural':
            return self.get_structural_uncertainty(heads, relations, tails)

        if method == 'score':
            return self.get_score_based_uncertainty(heads, relations, tails)

        # Compute all components
        u_semantic = self.get_semantic_uncertainty(heads, tails)
        u_structural = self.get_structural_uncertainty(heads, relations, tails)
        u_score = self.get_score_based_uncertainty(heads, relations, tails)
        u_neighbor = self.get_neighborhood_uncertainty(heads, relations, tails)

        # Normalize each
        u_semantic_norm = u_semantic / (u_semantic.mean() + 1e-8)
        u_structural_norm = u_structural / (u_structural.mean() + 1e-8)
        u_score_norm = u_score / (u_score.abs().mean() + 1e-8)
        u_neighbor_norm = u_neighbor / (u_neighbor.abs().mean() + 1e-8)

        # Combine with weights
        weights = F.softmax(self.uncertainty_weights, dim=0)
        combined = (
            weights[0] * u_semantic_norm +
            weights[1] * u_structural_norm +
            weights[2] * u_score_norm +
            weights[3] * u_neighbor_norm
        )

        if return_components:
            return combined, {
                'semantic': u_semantic,
                'structural': u_structural,
                'score_based': u_score,
                'neighborhood': u_neighbor,
                'weights': weights.detach(),
            }

        return combined

    def should_abstain(self, heads: torch.Tensor, relations: torch.Tensor,
                       threshold: float = 0.1) -> torch.Tensor:
        """
        Decide whether to abstain from answering based on margin.

        Low margin = high uncertainty about the answer = abstain
        """
        margins = self.get_prediction_margin(heads, relations)
        return margins < threshold

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def precompute_coverage(self, triples):
        """Precompute coverage from training triples."""
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def kl_loss(self) -> torch.Tensor:
        """KL divergence for variational inference."""
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities

    def get_stats(self) -> Dict[str, float]:
        """Get model statistics."""
        weights = F.softmax(self.uncertainty_weights, dim=0).detach().cpu()
        return {
            'weight_semantic': weights[0].item(),
            'weight_structural': weights[1].item(),
            'weight_score': weights[2].item(),
            'weight_neighborhood': weights[3].item(),
        }


class EnsembleUncertaintyKGE:
    """
    Ensemble of KGE models for uncertainty estimation.

    Uses disagreement between models as additional uncertainty signal.
    """

    def __init__(self, models: List[AdaptiveUncertaintyKGE]):
        self.models = models
        self.n_models = len(models)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """Ensemble average score."""
        scores = torch.stack([m(heads, relations, tails) for m in self.models], dim=1)
        return scores.mean(dim=1)

    def get_score_variance(self, heads: torch.Tensor, relations: torch.Tensor,
                           tails: torch.Tensor) -> torch.Tensor:
        """Variance of scores across ensemble."""
        scores = torch.stack([m(heads, relations, tails) for m in self.models], dim=1)
        return scores.var(dim=1)

    def get_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                        tails: torch.Tensor,
                        method: Literal['neg_score', 'variance', 'combined'] = 'neg_score'
                        ) -> torch.Tensor:
        """
        Ensemble uncertainty.

        'neg_score': Use negative ensemble mean score (Energy-like) - BEST for random/type-constrained
        'variance': Use score variance - Helps for high-score attacks
        'combined': Weighted combination
        """
        mean_scores = self.forward(heads, relations, tails)
        variance = self.get_score_variance(heads, relations, tails)

        if method == 'neg_score':
            return -mean_scores
        elif method == 'variance':
            return variance
        else:  # combined
            neg_score_norm = -mean_scores / (mean_scores.abs().mean() + 1e-8)
            variance_norm = variance / (variance.mean() + 1e-8)
            return 0.7 * neg_score_norm + 0.3 * variance_norm

    def get_structural_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                    tails: torch.Tensor) -> torch.Tensor:
        """Get structural uncertainty from first model."""
        return self.models[0].get_structural_uncertainty(heads, relations, tails)

    def get_neighborhood_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                      tails: torch.Tensor) -> torch.Tensor:
        """Average neighborhood uncertainty across models."""
        uncs = torch.stack([m.get_neighborhood_uncertainty(heads, relations, tails)
                           for m in self.models], dim=1)
        return uncs.mean(dim=1)


def create_best_uncertainty_method(
    attack_type: Literal['random', 'high_score', 'embedding_similar', 'type_constrained', 'unknown']
) -> str:
    """
    Return the best uncertainty method for a given attack type.

    Based on experimental results:
    - Random: neg_score (AUROC 0.99)
    - High-score: neighborhood isolation (AUROC 0.67)
    - Embedding-similar: neg_score (AUROC 0.68)
    - Type-constrained: neg_score (AUROC 0.93)
    """
    if attack_type == 'random':
        return 'neg_score'  # Ensemble negative score
    elif attack_type == 'high_score':
        return 'neighborhood'  # Neighborhood isolation
    elif attack_type == 'embedding_similar':
        return 'neg_score'
    elif attack_type == 'type_constrained':
        return 'neg_score'
    else:  # unknown
        return 'combined'  # Use adaptive combination
