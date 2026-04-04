"""
RCUE: Relation-Conditioned Uncertainty Estimation for Knowledge Graphs

Key idea: σ²(e) → σ²(e | r)
Entity uncertainty depends on which relation is being queried.

Architecture:
- Entity base embedding: μ_e ∈ R^d
- Relation embedding: r ∈ R^d
- Uncertainty network: σ²(e, r) = MLP([μ_e; r; coverage(e,r)])

This supersedes:
- KG2E: which has separate entity/relation Gaussians
- UKGE: which has entity-only variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RCUE(nn.Module):
    """
    Relation-Conditioned Uncertainty Estimation.

    For query (h, r, t):
    - Score: standard KGE scoring (DistMult, ComplEx, etc.)
    - Uncertainty: σ²(h|r) + σ²(t|r), where σ²(e|r) is relation-conditioned
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        use_coverage: bool = True,
        scoring: str = "distmult"
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.use_coverage = use_coverage
        self.scoring = scoring

        # Entity and relation embeddings (for scoring)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Uncertainty network: (entity_emb, relation_emb) -> variance
        # Coverage is used as multiplicative factor, not input
        input_dim = 2 * embedding_dim

        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive variance
        )

        # Learnable boost factor: k = exp(boost_logit)
        # boost(cov=0) = 1 + k, boost(cov=1) = 1
        self.boost_logit = nn.Parameter(torch.tensor(0.7))  # ~2.0 initial boost

        # Coverage matrix (precomputed from training data)
        self.register_buffer(
            'coverage',
            torch.zeros(num_entities, num_relations)
        )

    def precompute_coverage(self, triples: np.ndarray):
        """Build coverage matrix from training triples."""
        for i in range(len(triples)):
            h, r, t = triples[i, 0], triples[i, 1], triples[i, 2]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def get_entity_variance(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute relation-conditioned variance for entities.

        Args:
            entity_ids: [batch_size]
            relation_ids: [batch_size]

        Returns:
            variances: [batch_size]
        """
        # Get embeddings
        e_emb = self.entity_emb(entity_ids)  # [batch, d]
        r_emb = self.relation_emb(relation_ids)  # [batch, d]

        # MLP input: entity + relation only (no coverage as input)
        unc_input = torch.cat([e_emb, r_emb], dim=-1)  # [batch, 2d]

        # Compute base variance from MLP
        base_variance = self.uncertainty_net(unc_input).squeeze(-1)  # [batch]

        # Coverage as multiplicative factor:
        # - coverage=1 (seen): variance stays as is
        # - coverage=0 (unseen): variance boosted by factor (1 + k)
        if self.use_coverage:
            cov = self.coverage[entity_ids, relation_ids]  # [batch]
            # Learnable boost: k = exp(boost_logit)
            k = torch.exp(self.boost_logit)
            boost = 1.0 + k * (1.0 - cov)
            variance = base_variance * boost
        else:
            variance = base_variance

        return variance

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for triples.

        Args:
            h, r, t: [batch_size] entity/relation indices

        Returns:
            scores: [batch_size]
        """
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)

        if self.scoring == "distmult":
            scores = (h_emb * r_emb * t_emb).sum(dim=-1)
        elif self.scoring == "transe":
            scores = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

        return scores

    def get_uncertainty(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute relation-conditioned uncertainty for triples.

        U(h, r, t) = σ²(h | r) + σ²(t | r)

        Args:
            h, r, t: [batch_size] entity/relation indices

        Returns:
            uncertainty: [batch_size]
        """
        h_var = self.get_entity_variance(h, r)
        t_var = self.get_entity_variance(t, r)

        return h_var + t_var

    def score_tails(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Score all possible tails for (h, r, ?)."""
        h_emb = self.entity_emb(h)  # [batch, d]
        r_emb = self.relation_emb(r)  # [batch, d]
        all_t = self.entity_emb.weight  # [num_entities, d]

        if self.scoring == "distmult":
            # [batch, d] * [batch, d] -> [batch, d]
            hr = h_emb * r_emb
            # [batch, d] @ [d, num_entities] -> [batch, num_entities]
            scores = hr @ all_t.T
        elif self.scoring == "transe":
            # [batch, d] + [batch, d] -> [batch, d]
            hr = h_emb + r_emb
            # [batch, 1, d] - [1, num_entities, d] -> [batch, num_entities, d]
            diff = hr.unsqueeze(1) - all_t.unsqueeze(0)
            scores = -torch.norm(diff, p=2, dim=-1)

        return scores


class RCUEWithAttention(nn.Module):
    """
    RCUE with attention mechanism for relation-conditioned uncertainty.

    Instead of MLP, use attention to weight entity embedding dimensions
    based on the relation.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        num_heads: int = 4,
        use_coverage: bool = True,
        scoring: str = "distmult"
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.use_coverage = use_coverage
        self.scoring = scoring

        # Entity embeddings: mean and base log-variance
        self.entity_mean = nn.Embedding(num_entities, embedding_dim)
        self.entity_logvar_base = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # Relation-conditioned attention for variance
        # Query: relation, Key/Value: entity logvar dimensions
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Coverage projection (if used)
        if use_coverage:
            self.coverage_proj = nn.Linear(1, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_mean.weight)
        nn.init.constant_(self.entity_logvar_base.weight, -1.0)  # Start with low variance
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage matrix
        self.register_buffer(
            'coverage',
            torch.zeros(num_entities, num_relations)
        )

    def precompute_coverage(self, triples: np.ndarray):
        """Build coverage matrix from training triples."""
        for i in range(len(triples)):
            h, r, t = triples[i, 0], triples[i, 1], triples[i, 2]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def get_entity_variance(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute relation-conditioned variance using attention.

        The relation "attends" to the entity's base variance dimensions,
        weighting them based on relevance.
        """
        batch_size = entity_ids.shape[0]

        # Base log-variance: [batch, d]
        base_logvar = self.entity_logvar_base(entity_ids)

        # Relation as query: [batch, 1, d]
        r_emb = self.relation_emb(relation_ids).unsqueeze(1)

        # Entity logvar as key/value: [batch, 1, d]
        base_logvar_seq = base_logvar.unsqueeze(1)

        # Apply attention: relation attends to entity variance
        # Output: [batch, 1, d]
        attended_logvar, _ = self.attention(r_emb, base_logvar_seq, base_logvar_seq)
        attended_logvar = attended_logvar.squeeze(1)  # [batch, d]

        # Modulate by coverage if available
        if self.use_coverage:
            cov = self.coverage[entity_ids, relation_ids].unsqueeze(-1)  # [batch, 1]
            # High coverage -> lower variance, low coverage -> higher variance
            cov_mod = self.coverage_proj(1.0 - cov)  # [batch, d]
            attended_logvar = attended_logvar + cov_mod

        # Convert to variance and average across dimensions
        variance = torch.exp(attended_logvar).mean(dim=-1)  # [batch]

        return variance

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute scores for triples."""
        h_emb = self.entity_mean(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_mean(t)

        if self.scoring == "distmult":
            scores = (h_emb * r_emb * t_emb).sum(dim=-1)
        elif self.scoring == "transe":
            scores = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

        return scores

    def get_uncertainty(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute relation-conditioned uncertainty."""
        h_var = self.get_entity_variance(h, r)
        t_var = self.get_entity_variance(t, r)
        return h_var + t_var
