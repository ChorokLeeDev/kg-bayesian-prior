"""
ComplEx: Complex Embeddings for Simple Link Prediction

Reference: Trouillon et al. (2016)

Uses complex-valued embeddings to model asymmetric relations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseKGEModel


class ComplEx(BaseKGEModel):
    """
    ComplEx model.

    Score function: Re(<h, r, conj(t)>)

    where embeddings are complex-valued (split into real and imaginary parts).
    This allows modeling asymmetric relations unlike DistMult.
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
            embedding_dim: Embedding dimension (for real part; total is 2x)
            dropout: Dropout rate
        """
        super().__init__(num_entities, num_relations, embedding_dim)

        self.dropout_rate = dropout

        # Complex embeddings: split into real and imaginary parts
        # Each embedding has 2 * embedding_dim parameters
        self.entity_embeddings_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_im = nn.Embedding(num_relations, embedding_dim)

        # For compatibility with base class
        self.entity_embeddings = self.entity_embeddings_re
        self.relation_embeddings = self.relation_embeddings_re

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self.init_embeddings()

    def init_embeddings(self, init_range: float = 0.1):
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.entity_embeddings_re.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_im.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_re.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_im.weight)

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ComplEx scores.

        Score = Re(<h, r, conj(t)>)
              = Re_h * Re_r * Re_t
              + Re_h * Im_r * Im_t
              + Im_h * Re_r * Im_t
              - Im_h * Im_r * Re_t
        """
        # Get real and imaginary parts
        h_re = self.dropout(self.entity_embeddings_re(head))
        h_im = self.dropout(self.entity_embeddings_im(head))
        r_re = self.dropout(self.relation_embeddings_re(relation))
        r_im = self.dropout(self.relation_embeddings_im(relation))
        t_re = self.dropout(self.entity_embeddings_re(tail))
        t_im = self.dropout(self.entity_embeddings_im(tail))

        # Complex dot product: Re(<h, r, conj(t)>)
        score = torch.sum(h_re * r_re * t_re, dim=-1)
        score += torch.sum(h_re * r_im * t_im, dim=-1)
        score += torch.sum(h_im * r_re * t_im, dim=-1)
        score -= torch.sum(h_im * r_im * t_re, dim=-1)

        return score

    def score_tails_efficient(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """Efficiently score all tails."""
        h_re = self.entity_embeddings_re(head)
        h_im = self.entity_embeddings_im(head)
        r_re = self.relation_embeddings_re(relation)
        r_im = self.relation_embeddings_im(relation)

        # All entity embeddings
        all_e_re = self.entity_embeddings_re.weight
        all_e_im = self.entity_embeddings_im.weight

        # Compute scores efficiently
        # Re(h * r) and Im(h * r)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Score = Re(hr * conj(t)) = hr_re * t_re + hr_im * t_im
        scores = torch.mm(hr_re, all_e_re.t()) + torch.mm(hr_im, all_e_im.t())

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
    ) -> torch.Tensor:
        """Compute BCE loss."""
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

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])

        return F.binary_cross_entropy_with_logits(scores, labels)

    def regularization_loss(self, lambda_reg: float = 0.001) -> torch.Tensor:
        """L3 regularization (as in original paper)."""
        reg = torch.mean(torch.abs(self.entity_embeddings_re.weight) ** 3)
        reg += torch.mean(torch.abs(self.entity_embeddings_im.weight) ** 3)
        reg += torch.mean(torch.abs(self.relation_embeddings_re.weight) ** 3)
        reg += torch.mean(torch.abs(self.relation_embeddings_im.weight) ** 3)
        return lambda_reg * reg

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get complex entity embeddings (concatenated real and imaginary)."""
        re = self.entity_embeddings_re(entity_ids)
        im = self.entity_embeddings_im(entity_ids)
        return torch.cat([re, im], dim=-1)
