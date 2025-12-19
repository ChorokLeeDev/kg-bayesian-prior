"""
Base class for Knowledge Graph Embedding models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class BaseKGEModel(ABC, nn.Module):
    """
    Abstract base class for KG embedding models.

    All KGE models should inherit from this class and implement:
    - score_triple: Compute score for a given triple
    - forward: Forward pass (usually calls score_triple)
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Entity and relation embeddings (to be defined in subclasses)
        self.entity_embeddings = None
        self.relation_embeddings = None

    @abstractmethod
    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score for triples.

        Args:
            head: Entity indices of shape (batch_size,)
            relation: Relation indices of shape (batch_size,)
            tail: Entity indices of shape (batch_size,)

        Returns:
            Scores of shape (batch_size,)
        """
        pass

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass, computes triple scores."""
        return self.score_triple(head, relation, tail)

    def score_heads(
        self,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (?, r, t) queries.

        Args:
            relation: Relation indices of shape (batch_size,)
            tail: Entity indices of shape (batch_size,)

        Returns:
            Scores of shape (batch_size, num_entities)
        """
        batch_size = relation.size(0)
        # Expand to score all entities as heads
        all_heads = torch.arange(self.num_entities, device=relation.device)
        all_heads = all_heads.unsqueeze(0).expand(batch_size, -1)

        relation = relation.unsqueeze(1).expand(-1, self.num_entities)
        tail = tail.unsqueeze(1).expand(-1, self.num_entities)

        # Reshape for scoring
        scores = self.score_triple(
            all_heads.reshape(-1),
            relation.reshape(-1),
            tail.reshape(-1),
        )
        return scores.reshape(batch_size, self.num_entities)

    def score_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (h, r, ?) queries.

        Args:
            head: Entity indices of shape (batch_size,)
            relation: Relation indices of shape (batch_size,)

        Returns:
            Scores of shape (batch_size, num_entities)
        """
        batch_size = head.size(0)
        # Expand to score all entities as tails
        all_tails = torch.arange(self.num_entities, device=head.device)
        all_tails = all_tails.unsqueeze(0).expand(batch_size, -1)

        head = head.unsqueeze(1).expand(-1, self.num_entities)
        relation = relation.unsqueeze(1).expand(-1, self.num_entities)

        # Reshape for scoring
        scores = self.score_triple(
            head.reshape(-1),
            relation.reshape(-1),
            all_tails.reshape(-1),
        )
        return scores.reshape(batch_size, self.num_entities)

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific entities."""
        return self.entity_embeddings(entity_ids)

    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific relations."""
        return self.relation_embeddings(relation_ids)

    def get_all_entity_embeddings(self) -> torch.Tensor:
        """Get all entity embeddings."""
        return self.entity_embeddings.weight

    def get_all_relation_embeddings(self) -> torch.Tensor:
        """Get all relation embeddings."""
        return self.relation_embeddings.weight

    def regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss on embeddings."""
        entity_reg = torch.mean(self.entity_embeddings.weight ** 2)
        relation_reg = torch.mean(self.relation_embeddings.weight ** 2)
        return entity_reg + relation_reg

    def init_embeddings(self, init_range: float = 0.1):
        """Initialize embeddings with uniform distribution."""
        nn.init.uniform_(self.entity_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.relation_embeddings.weight, -init_range, init_range)


class MarginRankingLoss(nn.Module):
    """Margin ranking loss for KGE training."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.

        Args:
            positive_scores: Scores for positive triples
            negative_scores: Scores for negative triples

        Returns:
            Loss value
        """
        return torch.mean(
            torch.clamp(self.margin - positive_scores + negative_scores, min=0)
        )


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy loss for KGE training."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss.

        Args:
            positive_scores: Scores for positive triples
            negative_scores: Scores for negative triples

        Returns:
            Loss value
        """
        scores = torch.cat([positive_scores, negative_scores])
        labels = torch.cat([
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores),
        ])
        return self.loss_fn(scores, labels)
