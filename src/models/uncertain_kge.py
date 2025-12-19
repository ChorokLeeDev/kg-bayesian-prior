"""
Uncertain KGE models with uncertainty quantification.

Implements:
1. MC Dropout for uncertainty estimation
2. Ensemble methods
3. Probabilistic embeddings (Gaussian)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .base import BaseKGEModel
from .distmult import DistMult
from .complex import ComplEx


class MCDropoutKGE(nn.Module):
    """
    Wrapper for any KGE model to enable MC Dropout uncertainty estimation.

    Uses dropout at inference time to generate multiple predictions,
    then computes mean and variance as uncertainty estimate.
    """

    def __init__(
        self,
        base_model: BaseKGEModel,
        dropout_rate: float = 0.1,
        num_samples: int = 10,
    ):
        """
        Args:
            base_model: Base KGE model (must have dropout layers)
            dropout_rate: Dropout rate to use
            num_samples: Number of MC samples for uncertainty estimation
        """
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples

        # Add dropout if not present
        self.dropout = nn.Dropout(dropout_rate)

    def enable_dropout(self):
        """Enable dropout during inference."""
        for m in self.base_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scores with uncertainty using MC Dropout.

        Returns:
            Tuple of (mean_score, uncertainty/variance)
        """
        self.enable_dropout()

        scores = []
        for _ in range(self.num_samples):
            score = self.base_model.score_triple(head, relation, tail)
            scores.append(score)

        scores = torch.stack(scores, dim=0)  # (num_samples, batch_size)

        mean_score = scores.mean(dim=0)
        uncertainty = scores.var(dim=0)

        return mean_score, uncertainty

    def predict_tails_with_uncertainty(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict all tail scores with uncertainty.

        Returns:
            Tuple of (mean_scores, uncertainties) each of shape (batch, num_entities)
        """
        self.enable_dropout()

        all_scores = []
        for _ in range(self.num_samples):
            scores = self.base_model.score_tails(head, relation)
            all_scores.append(scores)

        all_scores = torch.stack(all_scores, dim=0)

        mean_scores = all_scores.mean(dim=0)
        uncertainties = all_scores.var(dim=0)

        return mean_scores, uncertainties

    def forward(self, head, relation, tail):
        return self.base_model(head, relation, tail)


class EnsembleKGE(nn.Module):
    """
    Ensemble of KGE models for uncertainty estimation.

    Trains multiple independent models and uses disagreement as uncertainty.
    """

    def __init__(
        self,
        model_class: type,
        num_models: int = 5,
        **model_kwargs,
    ):
        """
        Args:
            model_class: Class of KGE model to ensemble
            num_models: Number of models in ensemble
            **model_kwargs: Arguments passed to each model
        """
        super().__init__()
        self.num_models = num_models

        self.models = nn.ModuleList([
            model_class(**model_kwargs)
            for _ in range(num_models)
        ])

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Average prediction across ensemble."""
        scores = torch.stack([
            model.score_triple(head, relation, tail)
            for model in self.models
        ], dim=0)
        return scores.mean(dim=0)

    def predict_with_uncertainty(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty from ensemble disagreement.

        Returns:
            Tuple of (mean_score, uncertainty)
        """
        scores = torch.stack([
            model.score_triple(head, relation, tail)
            for model in self.models
        ], dim=0)

        mean_score = scores.mean(dim=0)
        uncertainty = scores.var(dim=0)

        return mean_score, uncertainty

    def loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for all ensemble members."""
        total_loss = 0
        for model in self.models:
            total_loss += model.loss(positive_triples, negative_triples)
        return total_loss / self.num_models


class GaussianEmbeddingKGE(BaseKGEModel):
    """
    KGE model with Gaussian (probabilistic) entity embeddings.

    Each entity is represented as a Gaussian distribution:
    e ~ N(μ_e, σ_e²)

    This naturally captures entity-level uncertainty.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        min_variance: float = 0.1,
        max_variance: float = 10.0,
        **kwargs
    ):
        """
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            min_variance: Minimum variance (for numerical stability)
            max_variance: Maximum variance
        """
        super().__init__(num_entities, num_relations, embedding_dim)

        self.min_variance = min_variance
        self.max_variance = max_variance

        # Entity mean and log-variance
        self.entity_mean = nn.Embedding(num_entities, embedding_dim)
        self.entity_log_var = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings (deterministic)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # For base class compatibility
        self.entity_embeddings = self.entity_mean

        self.init_embeddings()

    def init_embeddings(self, init_range: float = 0.1):
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_log_var.weight, -1.0)  # Small initial variance
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def get_entity_distribution(
        self,
        entity_ids: torch.Tensor
    ) -> Normal:
        """Get Gaussian distribution for entities."""
        mean = self.entity_mean(entity_ids)
        log_var = self.entity_log_var(entity_ids)

        # Clamp variance
        var = torch.exp(log_var).clamp(self.min_variance, self.max_variance)
        std = torch.sqrt(var)

        return Normal(mean, std)

    def sample_entity_embedding(
        self,
        entity_ids: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample entity embeddings from their distributions."""
        dist = self.get_entity_distribution(entity_ids)
        samples = dist.rsample((num_samples,))  # (num_samples, batch, dim)
        return samples

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        use_mean: bool = True,
    ) -> torch.Tensor:
        """
        Compute scores using DistMult-style scoring.

        If use_mean=True, uses mean embeddings.
        If use_mean=False, samples from distributions.
        """
        if use_mean:
            h = self.entity_mean(head)
            t = self.entity_mean(tail)
        else:
            h = self.sample_entity_embedding(head, num_samples=1).squeeze(0)
            t = self.sample_entity_embedding(tail, num_samples=1).squeeze(0)

        r = self.relation_embeddings(relation)

        score = torch.sum(h * r * t, dim=-1)
        return score

    def predict_with_uncertainty(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty by sampling from entity distributions.

        Returns:
            Tuple of (mean_score, epistemic_uncertainty, aleatoric_uncertainty)
        """
        r = self.relation_embeddings(relation)

        # Sample entity embeddings
        h_samples = self.sample_entity_embedding(head, num_samples)  # (S, B, D)
        t_samples = self.sample_entity_embedding(tail, num_samples)  # (S, B, D)

        # Compute scores for all samples
        scores = torch.sum(h_samples * r.unsqueeze(0) * t_samples, dim=-1)  # (S, B)

        mean_score = scores.mean(dim=0)
        total_uncertainty = scores.var(dim=0)

        return mean_score, total_uncertainty

    def get_entity_uncertainty(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get uncertainty (variance) for specific entities."""
        log_var = self.entity_log_var(entity_ids)
        var = torch.exp(log_var).clamp(self.min_variance, self.max_variance)
        # Return average variance across dimensions
        return var.mean(dim=-1)

    def loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
        kl_weight: float = 0.001,
    ) -> torch.Tensor:
        """
        Compute loss with KL regularization.

        The KL term encourages the learned distributions to not deviate
        too far from a prior (standard normal).
        """
        # Score loss (BCE)
        pos_scores = self.score_triple(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
            use_mean=False,
        )
        neg_scores = self.score_triple(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
            use_mean=False,
        )

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])
        score_loss = F.binary_cross_entropy_with_logits(scores, labels)

        # KL regularization
        mean = self.entity_mean.weight
        log_var = self.entity_log_var.weight
        var = torch.exp(log_var)

        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl = 0.5 * torch.mean(var + mean**2 - 1 - log_var)

        return score_loss + kl_weight * kl


class UncertainKGE:
    """
    Factory class for creating uncertainty-aware KGE models.
    """

    @staticmethod
    def create(
        method: str,
        base_model: str = "distmult",
        **kwargs
    ) -> nn.Module:
        """
        Create an uncertainty-aware KGE model.

        Args:
            method: "mc_dropout", "ensemble", or "gaussian"
            base_model: Base model type for mc_dropout/ensemble
            **kwargs: Model-specific arguments

        Returns:
            Uncertainty-aware model
        """
        if method == "mc_dropout":
            if base_model == "distmult":
                model = DistMult(**kwargs)
            elif base_model == "complex":
                model = ComplEx(**kwargs)
            else:
                raise ValueError(f"Unknown base model: {base_model}")
            return MCDropoutKGE(model, **kwargs)

        elif method == "ensemble":
            if base_model == "distmult":
                model_class = DistMult
            elif base_model == "complex":
                model_class = ComplEx
            else:
                raise ValueError(f"Unknown base model: {base_model}")
            return EnsembleKGE(model_class, **kwargs)

        elif method == "gaussian":
            return GaussianEmbeddingKGE(**kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")
