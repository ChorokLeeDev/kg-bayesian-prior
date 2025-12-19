"""
GGPN: Multi-Relational Graph Representation Learning with Bayesian Gaussian Process Network

Reference: Chen et al. (AAAI 2022)

This is a faithful reimplementation based on the paper description.

Key Ideas:
1. GP reformulated as Bayesian Linear Model for efficiency
2. Relation-aware kernel learned in data-driven way
3. Message passing with stochastic function values

The key insight from the paper:
- Instead of computing full GP inference (O(N³)), they use random Fourier features
  to approximate the kernel, converting GP to a Bayesian linear model (O(N))
- The kernel takes relations into account when computing entity similarity

NOTE: This implementation is for research comparison purposes.
      We will show this model lacks proper uncertainty calibration.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse

from .base import BaseKGEModel


class RelationAwareRandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for approximating relation-aware kernel.

    The kernel k(x, x') ≈ φ(x)^T φ(x') where φ are random features.

    For relation-aware case:
        k_r(x, x') = σ_r² exp(-||x - x'||² / (2ℓ_r²))

    Approximated by:
        φ_r(x) = sqrt(2/D) * [cos(ω_r^T x + b_r)]

    where ω_r ~ N(0, 1/ℓ_r² I) and b_r ~ Uniform(0, 2π)
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        num_relations: int,
        init_lengthscale: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_features = num_features
        self.num_relations = num_relations

        # Per-relation parameters
        self.log_lengthscale = nn.Parameter(
            torch.full((num_relations,), np.log(init_lengthscale))
        )
        self.log_variance = nn.Parameter(
            torch.zeros(num_relations)
        )

        # Random frequencies (fixed after initialization)
        # ω ~ N(0, I) - will be scaled by 1/lengthscale
        self.register_buffer(
            "omega",
            torch.randn(num_relations, num_features, input_dim)
        )
        # b ~ Uniform(0, 2π)
        self.register_buffer(
            "bias",
            torch.rand(num_relations, num_features) * 2 * np.pi
        )

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    def forward(
        self,
        x: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute random Fourier features for given inputs and relations.

        Args:
            x: Input features, shape (batch_size, input_dim)
            relation: Relation indices, shape (batch_size,)

        Returns:
            Random features, shape (batch_size, num_features)
        """
        batch_size = x.size(0)

        # Get relation-specific parameters
        ell = self.lengthscale[relation]  # (batch,)
        sigma_sq = self.variance[relation]  # (batch,)

        # Get relation-specific random features
        omega_r = self.omega[relation]  # (batch, num_features, input_dim)
        bias_r = self.bias[relation]  # (batch, num_features)

        # Scale frequencies by inverse lengthscale
        scaled_omega = omega_r / ell.unsqueeze(-1).unsqueeze(-1)  # (batch, D, d)

        # Compute features: φ(x) = sqrt(2σ²/D) * cos(ω^T x + b)
        projection = torch.bmm(scaled_omega, x.unsqueeze(-1)).squeeze(-1)  # (batch, D)
        features = torch.cos(projection + bias_r)  # (batch, D)

        # Scale by sqrt(2σ²/D)
        scale = torch.sqrt(2 * sigma_sq / self.num_features)  # (batch,)
        features = features * scale.unsqueeze(-1)

        return features


class GGPNLayer(nn.Module):
    """
    Single GGPN layer: aggregates neighbor messages using GP-style weighting.

    For each entity, aggregates information from neighbors with:
    - Relation-aware kernel weighting
    - Stochastic function values (Bayesian treatment)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        num_rff: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations

        # Random Fourier Features for kernel approximation
        self.rff = RelationAwareRandomFourierFeatures(
            input_dim=input_dim,
            num_features=num_rff,
            num_relations=num_relations,
        )

        # Bayesian linear model: y = Φw + ε
        # w ~ N(0, I), so posterior is also Gaussian
        # We learn the mean and maintain diagonal covariance
        self.weight_mean = nn.Parameter(torch.randn(num_rff, output_dim) * 0.01)
        self.weight_log_var = nn.Parameter(torch.full((num_rff, output_dim), -2.0))

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Attention for neighbor aggregation
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2 + num_relations, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Relation embeddings for attention
        self.relation_embed = nn.Embedding(num_relations, num_relations)
        nn.init.eye_(self.relation_embed.weight)

    @property
    def weight_var(self) -> torch.Tensor:
        return torch.exp(self.weight_log_var)

    def sample_weights(self, num_samples: int = 1) -> torch.Tensor:
        """Sample weights from posterior: w ~ N(μ, σ²)"""
        std = torch.sqrt(self.weight_var)
        eps = torch.randn(num_samples, *self.weight_mean.shape, device=self.weight_mean.device)
        return self.weight_mean.unsqueeze(0) + std.unsqueeze(0) * eps

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        use_mean: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with neighbor aggregation.

        Args:
            x: Node features, shape (num_nodes, input_dim)
            edge_index: Edge indices, shape (2, num_edges)
            edge_type: Edge types, shape (num_edges,)
            use_mean: If True, use mean weights; else sample

        Returns:
            Updated node features, shape (num_nodes, output_dim)
        """
        num_nodes = x.size(0)
        src, dst = edge_index

        # Get source node features
        x_src = x[src]  # (num_edges, input_dim)
        x_dst = x[dst]  # (num_edges, input_dim)

        # Compute random Fourier features
        phi = self.rff(x_src, edge_type)  # (num_edges, num_rff)

        # Apply Bayesian linear model: y = Φw
        if use_mean:
            messages = phi @ self.weight_mean  # (num_edges, output_dim)
        else:
            w = self.sample_weights(1).squeeze(0)
            messages = phi @ w

        # Compute attention weights
        rel_embed = self.relation_embed(edge_type)  # (num_edges, num_relations)
        attn_input = torch.cat([x_src, x_dst, rel_embed], dim=-1)
        attn_weights = self.attention(attn_input).squeeze(-1)  # (num_edges,)
        attn_weights = F.leaky_relu(attn_weights, 0.2)

        # Softmax over neighbors (per destination node)
        attn_weights = self._scatter_softmax(attn_weights, dst, num_nodes)

        # Aggregate messages
        weighted_messages = messages * attn_weights.unsqueeze(-1)
        output = torch.zeros(num_nodes, self.output_dim, device=x.device)
        output.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_messages), weighted_messages)

        # Residual connection if dimensions match
        if self.input_dim == self.output_dim:
            output = output + x

        output = self.layer_norm(output)
        output = self.dropout(output)

        return output

    def _scatter_softmax(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over scattered values."""
        # Subtract max for numerical stability
        max_vals = torch.zeros(num_nodes, device=src.device)
        max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
        src = src - max_vals[index]

        # Compute exp and sum
        exp_src = torch.exp(src)
        sum_exp = torch.zeros(num_nodes, device=src.device)
        sum_exp.scatter_add_(0, index, exp_src)

        return exp_src / (sum_exp[index] + 1e-10)


class GGPN(BaseKGEModel):
    """
    GGPN: Gaussian Process Network for Multi-Relational Graphs.

    Architecture:
    1. Initial entity embeddings
    2. Multiple GGPN layers for message passing
    3. Scoring function (DistMult-style)

    Key difference from standard GNN:
    - Uses GP kernel (via RFF) for message weighting
    - Bayesian treatment of layer weights

    LIMITATION (what we'll demonstrate):
    - Outputs point estimates, not uncertainty
    - No calibration guarantee
    - Can't decompose epistemic/aleatoric uncertainty
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 2,
        num_rff: int = 100,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

        # Relation embeddings for scoring
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # GGPN layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(GGPNLayer(
            input_dim=embedding_dim,
            output_dim=hidden_dim,
            num_relations=num_relations,
            num_rff=num_rff,
            dropout=dropout,
        ))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GGPNLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_relations=num_relations,
                num_rff=num_rff,
                dropout=dropout,
            ))

        # Output projection if needed
        if hidden_dim != embedding_dim:
            self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.output_proj = None

        # Store graph structure
        self._edge_index = None
        self._edge_type = None
        self._entity_repr = None

    def set_graph(self, kg_dataset):
        """Set graph structure for message passing."""
        # Build edge index from triples
        triples = kg_dataset.triples

        # Add reverse edges for message passing
        src = np.concatenate([triples[:, 0], triples[:, 2]])
        dst = np.concatenate([triples[:, 2], triples[:, 0]])
        rel = np.concatenate([triples[:, 1], triples[:, 1] + kg_dataset.num_relations])

        self._edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        self._edge_type = torch.tensor(rel, dtype=torch.long)

        # Update relation embeddings if we added reverse relations
        if self.relation_embeddings.num_embeddings < kg_dataset.num_relations * 2:
            old_weight = self.relation_embeddings.weight.data
            self.relation_embeddings = nn.Embedding(kg_dataset.num_relations * 2, self.embedding_dim)
            self.relation_embeddings.weight.data[:old_weight.size(0)] = old_weight
            nn.init.xavier_uniform_(self.relation_embeddings.weight.data[old_weight.size(0):])

        self._entity_repr = None  # Reset cached representations

    def encode(self, use_mean: bool = True) -> torch.Tensor:
        """
        Encode all entities using GGPN layers.

        Returns:
            Entity representations, shape (num_entities, embedding_dim)
        """
        if self._edge_index is None:
            # No graph structure, return raw embeddings
            return self.entity_embeddings.weight

        device = self.entity_embeddings.weight.device
        edge_index = self._edge_index.to(device)
        edge_type = self._edge_type.to(device)

        x = self.entity_embeddings.weight

        for layer in self.layers:
            x = layer(x, edge_index, edge_type, use_mean=use_mean)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using DistMult-style scoring.

        NOTE: Returns point estimates only - no uncertainty!
        """
        # Get entity representations
        if self._entity_repr is None:
            self._entity_repr = self.encode(use_mean=True)

        h = self._entity_repr[head]
        t = self._entity_repr[tail]
        r = self.relation_embeddings(relation)

        # DistMult scoring
        score = torch.sum(h * r * t, dim=-1)

        return score

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        # Reset cached representations for fresh encoding
        self._entity_repr = None
        return self.score_triple(head, relation, tail)

    def loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
        kl_weight: float = 0.001,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with KL regularization for Bayesian weights.
        """
        # Reset for fresh encoding
        self._entity_repr = None

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

        # BCE loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])
        likelihood_loss = F.binary_cross_entropy_with_logits(scores, labels)

        # KL regularization for Bayesian weights
        kl_loss = 0
        for layer in self.layers:
            # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
            kl = 0.5 * torch.sum(
                layer.weight_var + layer.weight_mean**2 - 1 - layer.weight_log_var
            )
            kl_loss = kl_loss + kl

        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "total": total_loss,
            "likelihood": likelihood_loss,
            "kl": kl_loss,
        }

    def get_confidence(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get confidence scores (sigmoid of scores).

        NOTE: This is NOT calibrated uncertainty!
        It's just the model's predicted probability.
        """
        scores = self.score_triple(head, relation, tail)
        return torch.sigmoid(scores)

    def score_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (h, r, ?) queries.

        Uses DistMult-style efficient scoring with cached entity representations.
        """
        # Get entity representations
        if self._entity_repr is None:
            self._entity_repr = self.encode(use_mean=True)

        batch_size = head.size(0)
        h = self._entity_repr[head]  # (batch, dim)
        r = self.relation_embeddings(relation)  # (batch, dim)

        # Query vector for DistMult
        query = h * r  # (batch, dim)

        # Score against all entities
        scores = torch.mm(query, self._entity_repr.t())  # (batch, num_entities)

        return scores

    def score_heads(
        self,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (?, r, t) queries.
        """
        # Get entity representations
        if self._entity_repr is None:
            self._entity_repr = self.encode(use_mean=True)

        batch_size = relation.size(0)
        t = self._entity_repr[tail]  # (batch, dim)
        r = self.relation_embeddings(relation)  # (batch, dim)

        # Query vector for DistMult (symmetric)
        query = r * t  # (batch, dim)

        # Score against all entities
        scores = torch.mm(query, self._entity_repr.t())  # (batch, num_entities)

        return scores

    # Methods for uncertainty comparison (these will show GGPN's limitations)

    def predict_with_mc_samples(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate uncertainty via MC sampling of Bayesian weights.

        NOTE: This is an approximation - GGPN wasn't designed for this.
        We add this to show that even with MC sampling, calibration is poor.
        """
        scores_list = []

        for _ in range(num_samples):
            # Encode with sampled weights
            self._entity_repr = None

            device = self.entity_embeddings.weight.device
            edge_index = self._edge_index.to(device) if self._edge_index is not None else None
            edge_type = self._edge_type.to(device) if self._edge_type is not None else None

            x = self.entity_embeddings.weight

            if edge_index is not None:
                for layer in self.layers:
                    x = layer(x, edge_index, edge_type, use_mean=False)  # Sample!

            if self.output_proj is not None:
                x = self.output_proj(x)

            h = x[head]
            t = x[tail]
            r = self.relation_embeddings(relation)

            score = torch.sum(h * r * t, dim=-1)
            scores_list.append(score)

        scores = torch.stack(scores_list, dim=0)
        mean_score = scores.mean(dim=0)
        var_score = scores.var(dim=0)

        return mean_score, var_score
