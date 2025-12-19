"""
TransE: Translating Embeddings for Modeling Multi-relational Data

Reference: Bordes et al. (2013)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseKGEModel


class TransE(BaseKGEModel):
    """
    TransE model.

    Score function: -||h + r - t||_p

    The basic idea is that the relation acts as a translation in the
    embedding space: if (h, r, t) holds, then h + r ≈ t.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        p_norm: int = 1,
        margin: float = 1.0,
        **kwargs
    ):
        """
        Args:
            num_entities: Number of entities in the KG
            num_relations: Number of relation types
            embedding_dim: Dimension of embeddings
            p_norm: Which p-norm to use (1 or 2)
            margin: Margin for ranking loss
        """
        super().__init__(num_entities, num_relations, embedding_dim)

        self.p_norm = p_norm
        self.margin = margin

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        self.init_embeddings()

    def init_embeddings(self, init_range: float = 6.0):
        """Initialize embeddings following original paper."""
        # Uniform initialization
        bound = init_range / self.embedding_dim
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=-1
            )

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TransE scores.

        Score = -||h + r - t||_p

        Higher scores indicate more plausible triples.
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Normalize entities
        h = F.normalize(h, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

        # Score: negative distance
        score = -torch.norm(h + r - t, p=self.p_norm, dim=-1)

        return score

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
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.

        Args:
            positive_triples: Tensor of shape (batch, 3) with positive triples
            negative_triples: Tensor of shape (batch, 3) with negative triples

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

        # Margin ranking loss
        loss = torch.mean(
            F.relu(self.margin - pos_scores + neg_scores)
        )

        return loss
