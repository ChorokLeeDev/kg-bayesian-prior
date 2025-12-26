"""
Flexible Coverage-Augmented GP-KGE (FlexibleCAGP)

Supports multiple scoring functions: DistMult, ComplEx, TransE, RotatE
Maintains CAGP's uncertainty decomposition while allowing different base models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class FlexibleCAGP(nn.Module):
    """
    Coverage-Augmented GP-KGE with pluggable scoring functions.

    Supports:
    - DistMult: (h * r * t).sum()
    - ComplEx: Re(<h, r, conj(t)>)
    - TransE: -||h + r - t||_p

    Uncertainty = α * GP_variance + (1-α) * Coverage_uncertainty
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        scoring_fn: Literal['distmult', 'complex', 'transe'] = 'distmult',
        initial_alpha: float = 0.5,
        learn_alpha: bool = True,
        per_relation_alpha: bool = False,
        transe_p_norm: int = 1,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.scoring_fn = scoring_fn
        self.learn_alpha = learn_alpha
        self.per_relation_alpha = per_relation_alpha
        self.transe_p_norm = transe_p_norm

        # Entity embeddings (variational: mean + log-variance)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Scoring-function-specific embeddings
        if scoring_fn == 'distmult':
            # Simple relation embeddings
            self.relation_emb = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.relation_emb.weight)

        elif scoring_fn == 'complex':
            # Complex embeddings: need imaginary parts too
            self.entity_mean_im = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
            self.entity_logvar_im = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

            self.relation_emb_re = nn.Embedding(num_relations, dim)
            self.relation_emb_im = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.relation_emb_re.weight)
            nn.init.xavier_uniform_(self.relation_emb_im.weight)

        elif scoring_fn == 'transe':
            # TransE uses normalized relation embeddings
            self.relation_emb = nn.Embedding(num_relations, dim)
            bound = 6.0 / dim
            nn.init.uniform_(self.relation_emb.weight, -bound, bound)
            # Normalize
            with torch.no_grad():
                self.relation_emb.weight.data = F.normalize(
                    self.relation_emb.weight.data, p=2, dim=-1
                )

        # Coverage matrix: [num_entities, num_relations]
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('coverage_freq', torch.zeros(num_entities, num_relations))

        # Adaptive α parameter
        if per_relation_alpha:
            alpha_init = torch.full((num_relations,), initial_alpha)
        else:
            alpha_init = torch.tensor([initial_alpha])

        if learn_alpha:
            self.alpha_logit = nn.Parameter(torch.logit(alpha_init))
        else:
            self.register_buffer('alpha_logit', torch.logit(alpha_init))

    def get_alpha(self, relations=None):
        """Get α value(s), optionally per-relation."""
        alpha = torch.sigmoid(self.alpha_logit)
        if self.per_relation_alpha and relations is not None:
            return alpha[relations]
        return alpha

    def _sample(self, indices, mean, logvar):
        """Reparameterization trick for variational inference."""
        m = mean[indices]
        std = torch.exp(0.5 * logvar[indices])
        return m + std * torch.randn_like(std)

    def forward(self, heads, relations, tails, use_sampling=True):
        """Compute triple scores using specified scoring function."""
        if self.scoring_fn == 'distmult':
            return self._forward_distmult(heads, relations, tails, use_sampling)
        elif self.scoring_fn == 'complex':
            return self._forward_complex(heads, relations, tails, use_sampling)
        elif self.scoring_fn == 'transe':
            return self._forward_transe(heads, relations, tails, use_sampling)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_fn}")

    def _forward_distmult(self, heads, relations, tails, use_sampling):
        """DistMult: (h * r * t).sum()"""
        if use_sampling and self.training:
            h = self._sample(heads, self.entity_mean, self.entity_logvar)
            t = self._sample(tails, self.entity_mean, self.entity_logvar)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _forward_complex(self, heads, relations, tails, use_sampling):
        """ComplEx: Re(<h, r, conj(t)>)"""
        if use_sampling and self.training:
            h_re = self._sample(heads, self.entity_mean, self.entity_logvar)
            h_im = self._sample(heads, self.entity_mean_im, self.entity_logvar_im)
            t_re = self._sample(tails, self.entity_mean, self.entity_logvar)
            t_im = self._sample(tails, self.entity_mean_im, self.entity_logvar_im)
        else:
            h_re = self.entity_mean[heads]
            h_im = self.entity_mean_im[heads]
            t_re = self.entity_mean[tails]
            t_im = self.entity_mean_im[tails]

        r_re = self.relation_emb_re(relations)
        r_im = self.relation_emb_im(relations)

        # Complex dot product: Re(<h, r, conj(t)>)
        score = (h_re * r_re * t_re).sum(dim=-1)
        score += (h_re * r_im * t_im).sum(dim=-1)
        score += (h_im * r_re * t_im).sum(dim=-1)
        score -= (h_im * r_im * t_re).sum(dim=-1)

        return score

    def _forward_transe(self, heads, relations, tails, use_sampling):
        """TransE: -||h + r - t||_p"""
        if use_sampling and self.training:
            h = self._sample(heads, self.entity_mean, self.entity_logvar)
            t = self._sample(tails, self.entity_mean, self.entity_logvar)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        # Normalize entities (TransE requirement)
        h = F.normalize(h, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

        r = self.relation_emb(relations)

        # Negative distance
        score = -torch.norm(h + r - t, p=self.transe_p_norm, dim=-1)

        return score

    def get_gp_variance(self, heads, tails):
        """Compute GP-based variance (learned component)."""
        if self.scoring_fn == 'complex':
            # For ComplEx, combine real and imaginary variances
            h_var_re = torch.exp(self.entity_logvar[heads])
            h_var_im = torch.exp(self.entity_logvar_im[heads])
            t_var_re = torch.exp(self.entity_logvar[tails])
            t_var_im = torch.exp(self.entity_logvar_im[tails])

            gp_var = (h_var_re.mean(dim=-1) + h_var_im.mean(dim=-1) +
                     t_var_re.mean(dim=-1) + t_var_im.mean(dim=-1)) / 4
        else:
            # For DistMult and TransE
            h_var = torch.exp(self.entity_logvar[heads])
            t_var = torch.exp(self.entity_logvar[tails])
            gp_var = (h_var.mean(dim=-1) + t_var.mean(dim=-1)) / 2

        return gp_var

    def get_coverage_uncertainty(self, heads, relations, tails, use_frequency=False):
        """Compute coverage-based uncertainty (explicit component)."""
        if use_frequency:
            h_freq = self.coverage_freq[heads, relations]
            t_freq = self.coverage_freq[tails, relations]
            max_freq = self.coverage_freq.max() + 1
            coverage_unc = 2.0 - h_freq / max_freq - t_freq / max_freq
        else:
            h_seen = self.coverage[heads, relations]
            t_seen = self.coverage[tails, relations]
            coverage_unc = 2.0 - h_seen - t_seen

        return coverage_unc

    def get_uncertainty(self, heads, relations, tails, use_frequency=False):
        """
        Combined uncertainty: α * GP_var + (1-α) * Coverage_unc
        """
        gp_var = self.get_gp_variance(heads, tails)
        coverage_unc = self.get_coverage_uncertainty(heads, relations, tails, use_frequency)

        # Normalize GP variance to similar scale as coverage uncertainty
        gp_var_normalized = gp_var / (gp_var.mean() + 1e-8) * coverage_unc.mean()

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
        if self.scoring_fn == 'complex':
            # KL for both real and imaginary parts
            kl_re = -0.5 * torch.sum(
                1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
            )
            kl_im = -0.5 * torch.sum(
                1 + self.entity_logvar_im - self.entity_mean_im.pow(2) - self.entity_logvar_im.exp()
            )
            return (kl_re + kl_im) / (2 * self.num_entities)
        else:
            kl = -0.5 * torch.sum(
                1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
            )
            return kl / self.num_entities

    def get_coverage_stats(self):
        """Return coverage statistics for analysis."""
        coverage_per_entity = self.coverage.sum(dim=1)
        coverage_per_relation = self.coverage.sum(dim=0)

        return {
            'entities_per_relation_mean': coverage_per_relation.mean().item(),
            'entities_per_relation_std': coverage_per_relation.std().item(),
            'relations_per_entity_mean': coverage_per_entity.mean().item(),
            'relations_per_entity_std': coverage_per_entity.std().item(),
            'alpha': self.get_alpha().mean().item(),
            'scoring_fn': self.scoring_fn,
        }


class FlexibleCAGPTrainer:
    """Trainer for Flexible CAGP with any scoring function."""

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
