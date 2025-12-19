"""
DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases

Reference: Yang et al. (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseKGEModel


class DistMult(BaseKGEModel):
    """
    DistMult model.

    Score function: <h, r, t> = sum(h * r * t)

    Uses diagonal relation matrices for efficiency.
    Symmetric model - can't distinguish (h, r, t) from (t, r, h).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__(num_entities, num_relations, embedding_dim)

        self.dropout_rate = dropout

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self.init_embeddings()

    def init_embeddings(self, init_range: float = 0.1):
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DistMult scores.

        Score = <h, r, t> = sum(h * r * t)
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Apply dropout during training
        h = self.dropout(h)
        r = self.dropout(r)
        t = self.dropout(t)

        # Trilinear dot product
        score = torch.sum(h * r * t, dim=-1)

        return score

    def score_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently score all tails using matrix multiplication.

        Overrides base class for efficiency.
        """
        h = self.entity_embeddings(head)  # (batch, dim)
        r = self.relation_embeddings(relation)  # (batch, dim)

        # h * r gives us the query vector
        query = h * r  # (batch, dim)

        # Score against all entity embeddings
        all_entities = self.entity_embeddings.weight  # (num_entities, dim)
        scores = torch.mm(query, all_entities.t())  # (batch, num_entities)

        return scores

    def score_heads(
        self,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently score all heads using matrix multiplication.

        Overrides base class for efficiency.
        """
        r = self.relation_embeddings(relation)  # (batch, dim)
        t = self.entity_embeddings(tail)  # (batch, dim)

        # r * t gives us the query vector (DistMult is symmetric)
        query = r * t  # (batch, dim)

        # Score against all entity embeddings
        all_entities = self.entity_embeddings.weight  # (num_entities, dim)
        scores = torch.mm(query, all_entities.t())  # (batch, num_entities)

        return scores

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_triple(head, relation, tail)

    def loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
        use_bce: bool = True,
    ) -> torch.Tensor:
        """
        Compute loss (BCE or margin-based).

        Args:
            positive_triples: Positive triples (batch, 3)
            negative_triples: Negative triples (batch, 3)
            use_bce: Use BCE loss if True, margin loss otherwise

        Returns:
            Loss value
        """
        pos_scores = self.score_triple(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
        )
        neg_scores = self.score_triple(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
        )

        if use_bce:
            # BCE loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
        else:
            # Margin loss
            margin = 1.0
            loss = torch.mean(F.relu(margin - pos_scores + neg_scores))

        return loss

    def regularization_loss(self, lambda_reg: float = 0.001) -> torch.Tensor:
        """L2 regularization on embeddings."""
        reg = torch.mean(self.entity_embeddings.weight ** 2)
        reg += torch.mean(self.relation_embeddings.weight ** 2)
        return lambda_reg * reg
