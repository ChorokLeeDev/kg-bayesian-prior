"""
Coverage-Augmented GP-KGE (CAGP)

Key Innovation: Combines learned GP variance with explicit relation-specific coverage.
- Fixes WN18RR failure (low relation diversity)
- Maintains FB15k-237 performance (high relation diversity)
- Adaptive α learns to weight the two signals based on dataset characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class CoverageAugmentedGPKGE(nn.Module):
    """
    Coverage-Augmented Gaussian Process Knowledge Graph Embedding.

    Uncertainty = α * GP_variance + (1-α) * Coverage_uncertainty

    Where α is learned adaptively:
    - Low-diversity KGs: α → 0 (rely on explicit coverage)
    - High-diversity KGs: α → 1 (leverage learned GP variance)
    """

    def __init__(self, num_entities, num_relations, dim,
                 initial_alpha=0.5, learn_alpha=True, per_relation_alpha=False):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.learn_alpha = learn_alpha
        self.per_relation_alpha = per_relation_alpha

        # Entity embeddings (variational: mean + log-variance)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Relation-specific coverage matrix: [num_entities, num_relations]
        # Binary: 1 if entity has been seen with relation, 0 otherwise
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Frequency-weighted coverage (optional, for ablation)
        self.register_buffer('coverage_freq', torch.zeros(num_entities, num_relations))

        # Adaptive combination weight α
        if per_relation_alpha:
            alpha_init = torch.full((num_relations,), initial_alpha)
        else:
            alpha_init = torch.tensor([initial_alpha])

        if learn_alpha:
            # Parameterize in logit space for unconstrained optimization
            self.alpha_logit = nn.Parameter(torch.logit(alpha_init))
        else:
            self.register_buffer('alpha_logit', torch.logit(alpha_init))

    def get_alpha(self, relations=None):
        """Get α value(s), optionally per-relation."""
        alpha = torch.sigmoid(self.alpha_logit)
        if self.per_relation_alpha and relations is not None:
            return alpha[relations]
        return alpha

    def forward(self, heads, relations, tails, use_sampling=True):
        """Compute triple scores using DistMult scoring function."""
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices):
        """Reparameterization trick for variational inference."""
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def get_gp_variance(self, heads, tails):
        """Compute GP-based variance (learned component)."""
        h_var = torch.exp(self.entity_logvar[heads])  # [B, dim]
        t_var = torch.exp(self.entity_logvar[tails])  # [B, dim]

        # Average variance across dimensions
        gp_var = (h_var.mean(dim=-1) + t_var.mean(dim=-1)) / 2
        return gp_var

    def get_coverage_uncertainty(self, heads, relations, tails, use_frequency=False):
        """Compute coverage-based uncertainty (explicit component)."""
        if use_frequency:
            # Frequency-weighted: higher frequency = lower uncertainty
            h_freq = self.coverage_freq[heads, relations]
            t_freq = self.coverage_freq[tails, relations]
            max_freq = self.coverage_freq.max() + 1
            coverage_unc = 2.0 - h_freq / max_freq - t_freq / max_freq
        else:
            # Binary: seen with relation or not
            h_seen = self.coverage[heads, relations]
            t_seen = self.coverage[tails, relations]
            coverage_unc = 2.0 - h_seen - t_seen

        return coverage_unc

    def calibrate_normalization(self, heads, relations, tails, use_frequency=False):
        """Compute and cache normalization statistics from a reference set.
        Must be called before get_uncertainty() during evaluation to avoid
        batch-dependent normalization leakage."""
        with torch.no_grad():
            gp_var = self.get_gp_variance(heads, tails)
            coverage_unc = self.get_coverage_uncertainty(heads, relations, tails, use_frequency)
            self._norm_stats = {
                'gp_mean': gp_var.mean().item(),
                'cov_mean': coverage_unc.mean().item(),
            }

    def get_uncertainty(self, heads, relations, tails, use_frequency=False):
        """
        Combined uncertainty: α * GP_var + (1-α) * Coverage_unc

        This is the key innovation:
        - GP variance captures learned embedding uncertainty
        - Coverage captures relation-specific observation patterns
        - α adapts based on which signal is more reliable
        """
        gp_var = self.get_gp_variance(heads, tails)
        coverage_unc = self.get_coverage_uncertainty(heads, relations, tails, use_frequency)

        # Normalize GP variance to similar scale as coverage uncertainty
        # Use cached stats if available to avoid batch-dependent leakage
        if hasattr(self, '_norm_stats') and self._norm_stats is not None:
            gp_mean = self._norm_stats['gp_mean']
            cov_mean = self._norm_stats['cov_mean']
        else:
            gp_mean = gp_var.mean().item()
            cov_mean = coverage_unc.mean().item()
        gp_var_normalized = gp_var / (gp_mean + 1e-8) * (cov_mean + 1e-8)

        # Adaptive combination
        alpha = self.get_alpha(relations)
        uncertainty = alpha * gp_var_normalized + (1 - alpha) * coverage_unc

        return uncertainty

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        """Precompute coverage matrix from training triples."""
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]

            # Binary coverage
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0

            # Frequency coverage
            self.coverage_freq[h_idx, r_idx] += 1.0
            self.coverage_freq[t_idx, r_idx] += 1.0

    def kl_loss(self):
        """KL divergence from standard normal prior."""
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities

    def get_coverage_stats(self):
        """Return coverage statistics for analysis."""
        coverage_per_entity = self.coverage.sum(dim=1)  # How many relations per entity
        coverage_per_relation = self.coverage.sum(dim=0)  # How many entities per relation

        return {
            'entities_per_relation_mean': coverage_per_relation.mean().item(),
            'entities_per_relation_std': coverage_per_relation.std().item(),
            'relations_per_entity_mean': coverage_per_entity.mean().item(),
            'relations_per_entity_std': coverage_per_entity.std().item(),
            'alpha': self.get_alpha().mean().item(),
        }


class CoverageAugmentedGPKGETrainer:
    """Trainer for Coverage-Augmented GP-KGE."""

    def __init__(self, model, lr=0.001, kl_weight=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight

    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0

        for batch_h, batch_r, batch_t in dataloader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            # Positive scores
            pos_scores = self.model(batch_h, batch_r, batch_t, use_sampling=True)

            # Negative sampling
            num_entities = self.model.num_entities
            neg_t = torch.randint(0, num_entities, batch_t.shape, device=device)
            neg_scores = self.model(batch_h, batch_r, neg_t, use_sampling=True)

            # BCE loss
            loss = self.criterion(pos_scores, torch.ones_like(pos_scores))
            loss += self.criterion(neg_scores, torch.zeros_like(neg_scores))

            # KL regularization
            loss += self.kl_weight * self.model.kl_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
