"""
Relation-Aware Uncertainty Methods

This module addresses the reviewer concern that CAGP is "too simple" by providing
three progressively more sophisticated approaches to relation-aware uncertainty:

1. AttentionCAGP: Query-specific mixing weights via attention
2. RelationConditionedVariance: Learn σ²(e,r) = MLP([e;r])
3. GNNUncertainty: Propagate uncertainty through KG structure

All methods maintain the semantic-structural decomposition insight while adding
principled learnable components.

Requirements:
    - PyTorch >= 1.12 (for scatter_reduce_ in GNNUncertainty)
    - PyTorch >= 2.0 recommended for MPS support and performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# =============================================================================
# Method 1: Attention-Based Mixing (Quick Win)
# =============================================================================

class AttentionCAGP(nn.Module):
    """
    Attention-based Coverage-Augmented GP-KGE.

    Instead of a single global α, learns query-specific mixing weights:
        α(h, r, t) = σ(MLP([h_emb; r_emb; t_emb; gp_var; cov_unc]))

    This allows the model to dynamically weight GP vs coverage based on
    the specific query characteristics.

    Key insight: Some queries benefit more from GP (rare entities),
    others from coverage (familiar entities in new contexts).
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 hidden_dim: int = 64, num_attention_heads: int = 4):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity embeddings (variational)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage matrix
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Attention network for mixing weights
        # Input: [h_emb, r_emb, t_emb, gp_var_scalar, cov_unc_scalar]
        attention_input_dim = 3 * dim + 2

        self.attention_net = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output α ∈ (0, 1)
        )

        # Multi-head attention alternative (more expressive)
        self.use_multihead = num_attention_heads > 1
        if self.use_multihead:
            self.head_weights = nn.Linear(attention_input_dim, num_attention_heads)
            self.head_alphas = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(attention_input_dim, hidden_dim // num_attention_heads),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // num_attention_heads, 1),
                    nn.Sigmoid()
                ) for _ in range(num_attention_heads)
            ])

    def forward(self, heads, relations, tails, use_sampling=True):
        """Score triples using DistMult."""
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices):
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def get_gp_variance(self, heads, tails):
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_coverage_uncertainty(self, heads, relations, tails):
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_attention_alpha(self, heads, relations, tails):
        """
        Compute query-specific mixing weight α(h, r, t).

        The attention mechanism considers:
        - Entity embeddings (semantic content)
        - Relation embedding (query context)
        - Current uncertainty values (meta-information)
        """
        h_emb = self.entity_mean[heads]
        r_emb = self.relation_emb(relations)
        t_emb = self.entity_mean[tails]

        gp_var = self.get_gp_variance(heads, tails).unsqueeze(-1)
        cov_unc = self.get_coverage_uncertainty(heads, relations, tails).unsqueeze(-1)

        # Concatenate all features
        features = torch.cat([h_emb, r_emb, t_emb, gp_var, cov_unc], dim=-1)

        if self.use_multihead:
            # Multi-head attention: weighted combination of head-specific alphas
            head_logits = self.head_weights(features)  # (B, num_heads)
            head_weights = F.softmax(head_logits, dim=-1)

            alphas = torch.stack([head(features).squeeze(-1) for head in self.head_alphas], dim=-1)
            alpha = (head_weights * alphas).sum(dim=-1)
        else:
            alpha = self.attention_net(features).squeeze(-1)

        return alpha

    def get_uncertainty(self, heads, relations, tails):
        """
        Combined uncertainty with attention-based mixing.

        U = α(h,r,t) * GP_var + (1 - α(h,r,t)) * Coverage_unc
        """
        gp_var = self.get_gp_variance(heads, tails)
        cov_unc = self.get_coverage_uncertainty(heads, relations, tails)

        # Normalize GP variance to coverage scale with robust statistics
        # Use detached running statistics to avoid issues with all-OOD batches
        gp_mean = gp_var.mean().detach().clamp(min=1e-6)
        cov_mean = cov_unc.mean().detach().clamp(min=1e-6)
        gp_var_norm = gp_var / gp_mean * cov_mean

        # Query-specific mixing
        alpha = self.get_attention_alpha(heads, relations, tails)

        uncertainty = alpha * gp_var_norm + (1 - alpha) * cov_unc
        return uncertainty

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        """Precompute coverage matrix from training triples."""
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

    def kl_loss(self):
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities


# =============================================================================
# Method 2: Relation-Conditioned Variance (Principled)
# =============================================================================

class RelationConditionedVariance(nn.Module):
    """
    Learns relation-specific entity variance: σ²(e, r) = MLP([e_emb; r_emb])

    This directly addresses the reviewer's concern that GP variance is
    relation-agnostic. Instead of learning a single σ²_e per entity,
    we learn a function that outputs variance conditioned on the relation.

    Key insight: An entity's uncertainty depends on the relational context.
    Einstein's uncertainty for (Einstein, born_in, ?) differs from
    (Einstein, discovered, ?).
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 variance_hidden_dim: int = 128, min_variance: float = 1e-4,
                 max_variance: float = 10.0):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.min_variance = min_variance
        self.max_variance = max_variance

        # Entity embeddings (mean only - variance is relation-conditioned)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Relation-conditioned variance network
        # Input: [entity_emb, relation_emb]
        # Output: scalar variance
        self.variance_net = nn.Sequential(
            nn.Linear(2 * dim, variance_hidden_dim),
            nn.LayerNorm(variance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(variance_hidden_dim, variance_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(variance_hidden_dim // 2, 1),
        )

        # Initialize to output small positive values
        with torch.no_grad():
            self.variance_net[-1].bias.fill_(-2.0)  # exp(-2) ≈ 0.14

        # Entity-level base variance (fallback for unseen relations)
        self.entity_base_logvar = nn.Parameter(torch.zeros(num_entities) - 1.0)

        # Coverage matrix (structural signal)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, heads, relations, tails, use_sampling=True):
        """Score triples."""
        if use_sampling and self.training:
            h = self._sample(heads, relations)
            t = self._sample(tails, relations)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, entity_ids, relations):
        """Sample with relation-conditioned variance."""
        mean = self.entity_mean[entity_ids]
        var = self.get_entity_relation_variance(entity_ids, relations)
        std = torch.sqrt(var).unsqueeze(-1).expand_as(mean)
        return mean + std * torch.randn_like(mean)

    def get_entity_relation_variance(self, entity_ids, relations):
        """
        Compute σ²(e, r) = softplus(MLP([e_emb; r_emb])) + min_var

        The variance is:
        - Low when entity frequently appears with relation (well-learned)
        - High when entity rarely/never appears with relation (uncertain)
        """
        e_emb = self.entity_mean[entity_ids]
        r_emb = self.relation_emb(relations)

        combined = torch.cat([e_emb, r_emb], dim=-1)
        raw_var = self.variance_net(combined).squeeze(-1)

        # Bounded variance: softplus ensures positive, clamp ensures bounds
        variance = F.softplus(raw_var) + self.min_variance
        variance = torch.clamp(variance, max=self.max_variance)

        return variance

    def get_uncertainty(self, heads, relations, tails):
        """
        Combined uncertainty using relation-conditioned variance.

        This is the KEY INNOVATION: variance now depends on (entity, relation),
        not just entity.
        """
        # Relation-conditioned semantic uncertainty
        h_var = self.get_entity_relation_variance(heads, relations)
        t_var = self.get_entity_relation_variance(tails, relations)
        semantic_unc = (h_var + t_var) / 2

        # Structural uncertainty (coverage)
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        structural_unc = 2.0 - h_seen - t_seen

        # Learned combination with robust normalization
        # Use detached and clamped statistics to handle edge cases
        semantic_mean = semantic_unc.mean().detach().clamp(min=1e-6)
        structural_mean = structural_unc.mean().detach().clamp(min=1e-6)
        semantic_norm = semantic_unc / semantic_mean * structural_mean

        return 0.5 * semantic_norm + 0.5 * structural_unc

    def get_gp_variance(self, heads, tails, relations=None):
        """
        Get variance - now relation-aware if relations provided.
        Falls back to base variance if relations not provided.
        """
        if relations is not None:
            h_var = self.get_entity_relation_variance(heads, relations)
            t_var = self.get_entity_relation_variance(tails, relations)
        else:
            h_var = torch.exp(self.entity_base_logvar[heads])
            t_var = torch.exp(self.entity_base_logvar[tails])
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

    def kl_loss(self):
        """KL regularization on base variances."""
        base_var = torch.exp(self.entity_base_logvar)
        kl = -0.5 * torch.sum(
            1 + self.entity_base_logvar - self.entity_mean.pow(2).mean(dim=-1) - base_var
        )
        return kl / self.num_entities


# =============================================================================
# Method 3: GNN-Based Uncertainty Propagation (Most Novel)
# =============================================================================

class MessagePassingUncertainty(nn.Module):
    """
    Single layer of uncertainty message passing.

    Entities gather uncertainty information from their neighbors,
    weighted by relation-specific attention.
    """

    def __init__(self, dim: int, num_relations: int):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations

        # Relation-specific message transformation
        self.relation_transform = nn.Embedding(num_relations, dim * dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # Uncertainty update
        self.uncertainty_update = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

    def forward(self, entity_emb: torch.Tensor, entity_unc: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Propagate uncertainty through edges.

        Args:
            entity_emb: (N, dim) entity embeddings
            entity_unc: (N,) entity uncertainties
            edge_index: (2, E) edge indices [source, target]
            edge_type: (E,) relation types

        Returns:
            Updated uncertainties (N,)
        """
        source, target = edge_index

        # Compute messages
        source_emb = entity_emb[source]
        target_emb = entity_emb[target]
        source_unc = entity_unc[source]

        # Relation-specific transformation
        W = self.relation_transform(edge_type).view(-1, self.dim, self.dim)
        transformed = torch.bmm(source_emb.unsqueeze(1), W).squeeze(1)

        # Attention weights
        attention_input = torch.cat([source_emb, target_emb, transformed], dim=-1)
        attention_scores = self.attention(attention_input).squeeze(-1)

        # Normalize per target node (scatter softmax)
        attention_weights = self._scatter_softmax(attention_scores, target, entity_emb.size(0))

        # Aggregate uncertainty messages
        weighted_unc = attention_weights * source_unc
        aggregated_unc = torch.zeros(entity_emb.size(0), device=entity_emb.device)
        aggregated_unc.scatter_add_(0, target, weighted_unc)

        # Count neighbors for normalization
        neighbor_count = torch.zeros(entity_emb.size(0), device=entity_emb.device)
        neighbor_count.scatter_add_(0, target, torch.ones_like(weighted_unc))
        neighbor_count = neighbor_count.clamp(min=1)

        aggregated_unc = aggregated_unc / neighbor_count

        # Update uncertainty: combine current and aggregated
        update_input = torch.cat([entity_emb, aggregated_unc.unsqueeze(-1)], dim=-1)
        new_unc = self.uncertainty_update(update_input).squeeze(-1)

        # Residual connection
        return 0.5 * entity_unc + 0.5 * new_unc

    def _scatter_softmax(self, scores: torch.Tensor, indices: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Compute softmax over groups defined by indices."""
        # Subtract max for numerical stability
        max_scores = torch.zeros(num_nodes, device=scores.device)
        max_scores.scatter_reduce_(0, indices, scores, reduce='amax', include_self=False)
        scores = scores - max_scores[indices]

        # Compute exp and sum
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(num_nodes, device=scores.device)
        sum_exp.scatter_add_(0, indices, exp_scores)

        return exp_scores / (sum_exp[indices] + 1e-10)


class GNNUncertainty(nn.Module):
    """
    GNN-based uncertainty quantification.

    Key idea: Uncertainty should propagate through the graph structure.
    An entity connected to many uncertain entities should itself be uncertain.
    An entity connected to many certain entities should be more certain.

    This captures higher-order structural patterns that simple coverage misses.

    Architecture:
    1. Initial uncertainty from GP variance + coverage
    2. Message passing layers propagate uncertainty
    3. Final uncertainty is relation-conditioned readout
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.num_layers = num_layers

        # Entity embeddings (variational)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Initial uncertainty encoder
        self.init_uncertainty = nn.Sequential(
            nn.Linear(dim + 2, dim),  # entity_emb + gp_var + coverage
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingUncertainty(dim, num_relations)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Relation-conditioned readout
        self.readout = nn.Sequential(
            nn.Linear(dim + dim + 1, dim),  # entity + relation + propagated_unc
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()
        )

        # Buffers
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer('edge_type', torch.zeros(0, dtype=torch.long))

        # Cached propagated uncertainty
        self._cached_uncertainty = None

    def forward(self, heads, relations, tails, use_sampling=True):
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices):
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def set_graph(self, triples, entity_to_idx, relation_to_idx):
        """Set graph structure for message passing."""
        sources, targets, relations = [], [], []

        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]

            # Add both directions for undirected message passing
            sources.extend([h_idx, t_idx])
            targets.extend([t_idx, h_idx])
            relations.extend([r_idx, r_idx])

            # Update coverage
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)
        self.edge_type = torch.tensor(relations, dtype=torch.long)

        # Invalidate cache
        self._cached_uncertainty = None

    def propagate_uncertainty(self):
        """
        Run message passing to propagate uncertainty.

        This is the core innovation: uncertainty flows through the graph,
        capturing structural patterns beyond simple coverage.
        """
        device = self.entity_mean.device

        # Initial uncertainty: GP variance + average coverage
        gp_var = torch.exp(self.entity_logvar).mean(dim=-1)  # (N,)
        avg_coverage = self.coverage.mean(dim=-1)  # (N,)

        init_input = torch.cat([
            self.entity_mean,
            gp_var.unsqueeze(-1),
            avg_coverage.unsqueeze(-1)
        ], dim=-1)

        uncertainty = self.init_uncertainty(init_input).squeeze(-1)

        # Message passing
        edge_index = self.edge_index.to(device)
        edge_type = self.edge_type.to(device)

        for layer in self.mp_layers:
            uncertainty = layer(self.entity_mean, uncertainty, edge_index, edge_type)
            uncertainty = self.dropout(uncertainty)

        self._cached_uncertainty = uncertainty
        return uncertainty

    def get_uncertainty(self, heads, relations, tails):
        """
        Get relation-conditioned uncertainty using propagated values.
        """
        # Ensure uncertainty is propagated
        if self._cached_uncertainty is None:
            self.propagate_uncertainty()

        device = heads.device
        prop_unc = self._cached_uncertainty.to(device)

        h_emb = self.entity_mean[heads]
        t_emb = self.entity_mean[tails]
        r_emb = self.relation_emb(relations)

        h_prop_unc = prop_unc[heads].unsqueeze(-1)
        t_prop_unc = prop_unc[tails].unsqueeze(-1)

        # Relation-conditioned readout
        h_input = torch.cat([h_emb, r_emb, h_prop_unc], dim=-1)
        t_input = torch.cat([t_emb, r_emb, t_prop_unc], dim=-1)

        h_unc = self.readout(h_input).squeeze(-1)
        t_unc = self.readout(t_input).squeeze(-1)

        return (h_unc + t_unc) / 2

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        """Alias for set_graph to maintain API compatibility."""
        self.set_graph(triples, entity_to_idx, relation_to_idx)

    def kl_loss(self):
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities


# =============================================================================
# Unified Trainer for All Methods
# =============================================================================

class RelationAwareUncertaintyTrainer:
    """Unified trainer for all relation-aware uncertainty models."""

    def __init__(self, model: nn.Module, lr: float = 0.001,
                 kl_weight: float = 0.01, uncertainty_weight: float = 0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight
        self.uncertainty_weight = uncertainty_weight

    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0

        # Invalidate uncertainty cache for GNN model
        if hasattr(self.model, '_cached_uncertainty'):
            self.model._cached_uncertainty = None

        for batch_h, batch_r, batch_t in dataloader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            # Positive scores
            pos_scores = self.model(batch_h, batch_r, batch_t, use_sampling=True)

            # Negative sampling
            neg_t = torch.randint(0, self.model.num_entities, batch_t.shape, device=device)
            neg_scores = self.model(batch_h, batch_r, neg_t, use_sampling=True)

            # BCE loss
            loss = self.criterion(pos_scores, torch.ones_like(pos_scores))
            loss += self.criterion(neg_scores, torch.zeros_like(neg_scores))

            # KL regularization
            if hasattr(self.model, 'kl_loss'):
                loss += self.kl_weight * self.model.kl_loss()

            # Uncertainty regularization: OOD samples should have higher uncertainty
            if self.uncertainty_weight > 0:
                pos_unc = self.model.get_uncertainty(batch_h, batch_r, batch_t)
                neg_unc = self.model.get_uncertainty(batch_h, batch_r, neg_t)

                # Margin loss: neg uncertainty should be higher
                margin = 0.5
                unc_loss = F.relu(margin + pos_unc - neg_unc).mean()
                loss += self.uncertainty_weight * unc_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
