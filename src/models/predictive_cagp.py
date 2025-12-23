"""
Predictive CAGP: Three-Component Uncertainty Decomposition

Extends CAGP with predictive uncertainty (entropy/margin) for adversarial OOD detection.

Uncertainty = Semantic (GP) + Structural (Coverage) + Predictive (Entropy/Margin)

Key insight: Adversarial OOD samples create distinctive prediction distributions
- Normal queries: one dominant answer (low entropy, high margin)
- Adversarial queries: multiple competing answers (high entropy, low margin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional


class PredictiveCAGP(nn.Module):
    """
    Three-Component Uncertainty: Semantic + Structural + Predictive.

    The predictive component captures when the model is uncertain about
    which tail entity to predict, even when it knows the entities well
    (low semantic uncertainty) and has seen them in this context
    (low structural uncertainty).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 100,
        predictive_type: Literal['entropy', 'margin', 'both'] = 'entropy',
        temperature: float = 1.0,
        learn_temperature: bool = True,
        learn_weights: bool = True,
        initial_alpha: float = 0.33,
        initial_beta: float = 0.33,
        initial_gamma: float = 0.33,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.predictive_type = predictive_type

        # Entity embeddings (variational: mean + log-variance)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage matrix
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('coverage_freq', torch.zeros(num_entities, num_relations))

        # Temperature for softmax in entropy computation
        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

        # Learnable weights for three components
        if learn_weights:
            # Use unconstrained parameters, apply softmax later
            self.weight_logits = nn.Parameter(torch.tensor([
                torch.logit(torch.tensor(initial_alpha)),
                torch.logit(torch.tensor(initial_beta)),
                torch.logit(torch.tensor(initial_gamma)),
            ]))
        else:
            self.register_buffer('weight_logits', torch.tensor([
                torch.logit(torch.tensor(initial_alpha)),
                torch.logit(torch.tensor(initial_beta)),
                torch.logit(torch.tensor(initial_gamma)),
            ]))

        # Pre-computed entity representations for efficient scoring
        self._cached_entity_emb = None

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    @property
    def weights(self) -> torch.Tensor:
        """Get normalized weights (alpha, beta, gamma)."""
        return F.softmax(self.weight_logits, dim=0)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor, use_sampling: bool = True) -> torch.Tensor:
        """Compute triple scores using DistMult."""
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """
        Score all possible tails for given (head, relation) pairs.

        Args:
            heads: [batch_size] head entity indices
            relations: [batch_size] relation indices

        Returns:
            [batch_size, num_entities] scores for all tail candidates
        """
        h = self.entity_mean[heads]  # [B, dim]
        r = self.relation_emb(relations)  # [B, dim]
        hr = h * r  # [B, dim]

        # Score against all entities
        scores = hr @ self.entity_mean.T  # [B, num_entities]
        return scores

    def get_semantic_uncertainty(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """GP variance-based uncertainty."""
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_structural_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                    tails: torch.Tensor) -> torch.Tensor:
        """Coverage-based uncertainty."""
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_prediction_entropy(self, heads: torch.Tensor, relations: torch.Tensor,
                               top_k: Optional[int] = None) -> torch.Tensor:
        """
        Compute prediction entropy over all (or top-k) tail candidates.

        High entropy = uncertain about which tail to predict = potential adversarial OOD
        """
        scores = self.score_all_tails(heads, relations)  # [B, num_entities]

        if top_k is not None:
            # Approximate with top-k only
            topk_scores, _ = torch.topk(scores, k=top_k, dim=-1)
            probs = F.softmax(topk_scores / self.temperature, dim=-1)
        else:
            probs = F.softmax(scores / self.temperature, dim=-1)

        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        return entropy

    def get_prediction_margin(self, heads: torch.Tensor, relations: torch.Tensor,
                              k: int = 2) -> torch.Tensor:
        """
        Compute prediction margin (top1 - top2 score).

        Low margin = competing candidates = potential adversarial OOD
        """
        scores = self.score_all_tails(heads, relations)  # [B, num_entities]
        topk_scores, _ = torch.topk(scores, k=k, dim=-1)  # [B, k]

        margin = topk_scores[:, 0] - topk_scores[:, 1]
        return margin

    def get_topk_density(self, heads: torch.Tensor, relations: torch.Tensor,
                         k: int = 10) -> torch.Tensor:
        """
        Sum of top-k probabilities.

        Low density = diffuse predictions = potential OOD
        """
        scores = self.score_all_tails(heads, relations)
        probs = F.softmax(scores / self.temperature, dim=-1)
        topk_probs, _ = torch.topk(probs, k=k, dim=-1)
        density = topk_probs.sum(dim=-1)
        return density

    def get_predictive_uncertainty(self, heads: torch.Tensor, relations: torch.Tensor,
                                   top_k: Optional[int] = None) -> torch.Tensor:
        """Get query-level predictive uncertainty based on configured type."""
        if self.predictive_type == 'entropy':
            return self.get_prediction_entropy(heads, relations, top_k)
        elif self.predictive_type == 'margin':
            # Negative margin (higher = more uncertain)
            return -self.get_prediction_margin(heads, relations)
        else:  # 'both'
            entropy = self.get_prediction_entropy(heads, relations, top_k)
            margin = self.get_prediction_margin(heads, relations)
            # Combine: normalize each and average
            entropy_norm = entropy / (entropy.mean() + 1e-8)
            neg_margin_norm = -margin / (margin.abs().mean() + 1e-8)
            return (entropy_norm + neg_margin_norm) / 2

    def get_tail_rank(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        """
        Get rank of the given tail among all possible tails.

        Higher rank = more uncertain about this specific tail.
        """
        scores = self.score_all_tails(heads, relations)  # [B, num_entities]
        tail_scores = scores.gather(1, tails.unsqueeze(1)).squeeze(1)  # [B]

        # Count how many entities have higher score than the given tail
        ranks = (scores > tail_scores.unsqueeze(1)).sum(dim=1).float() + 1  # [B]
        return ranks

    def get_score_gap(self, heads: torch.Tensor, relations: torch.Tensor,
                      tails: torch.Tensor) -> torch.Tensor:
        """
        Get gap between top-1 score and given tail's score.

        Higher gap = tail is less likely = more uncertain.
        """
        scores = self.score_all_tails(heads, relations)  # [B, num_entities]
        top1_scores = scores.max(dim=1).values  # [B]
        tail_scores = scores.gather(1, tails.unsqueeze(1)).squeeze(1)  # [B]

        gap = top1_scores - tail_scores
        return gap

    def get_tail_percentile(self, heads: torch.Tensor, relations: torch.Tensor,
                            tails: torch.Tensor) -> torch.Tensor:
        """
        Get percentile rank of tail (0-1, where 1 means top-ranked).

        Lower percentile = tail is unusual = potential OOD.
        """
        ranks = self.get_tail_rank(heads, relations, tails)
        percentiles = 1.0 - ranks / self.num_entities
        return percentiles

    def get_triple_predictive_uncertainty(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        method: Literal['rank', 'gap', 'percentile', 'combined'] = 'combined',
    ) -> torch.Tensor:
        """
        Get triple-level predictive uncertainty.

        Unlike query-level entropy/margin, this considers the specific tail.
        """
        if method == 'rank':
            return self.get_tail_rank(heads, relations, tails)
        elif method == 'gap':
            return self.get_score_gap(heads, relations, tails)
        elif method == 'percentile':
            return 1.0 - self.get_tail_percentile(heads, relations, tails)
        else:  # 'combined'
            # Combine rank-based uncertainty with query-level entropy
            rank = self.get_tail_rank(heads, relations, tails)
            entropy = self.get_prediction_entropy(heads, relations)

            # Normalize
            rank_norm = rank / (rank.mean() + 1e-8)
            entropy_norm = entropy / (entropy.mean() + 1e-8)

            # High rank in high-entropy query = very uncertain
            # Multiplicative combination captures this interaction
            return rank_norm * (1 + 0.5 * entropy_norm)

    def get_uncertainty(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        return_components: bool = False,
        top_k: Optional[int] = None,
        use_triple_level_predictive: bool = True,
    ) -> torch.Tensor:
        """
        Compute combined three-component uncertainty.

        U = alpha * U_semantic + beta * U_structural + gamma * U_predictive

        Args:
            use_triple_level_predictive: If True, use tail-specific predictive uncertainty
                                         (rank/gap). If False, use query-level entropy.
        """
        # 1. Semantic uncertainty (GP variance)
        u_semantic = self.get_semantic_uncertainty(heads, tails)

        # 2. Structural uncertainty (coverage)
        u_structural = self.get_structural_uncertainty(heads, relations, tails)

        # 3. Predictive uncertainty
        if use_triple_level_predictive:
            # Triple-level: considers the specific tail
            u_predictive = self.get_triple_predictive_uncertainty(
                heads, relations, tails, method='combined'
            )
        else:
            # Query-level: only considers (h, r)
            u_predictive = self.get_predictive_uncertainty(heads, relations, top_k)

        # Normalize each component
        u_semantic_norm = u_semantic / (u_semantic.mean() + 1e-8)
        u_structural_norm = u_structural / (u_structural.mean() + 1e-8)
        u_predictive_norm = u_predictive / (u_predictive.mean() + 1e-8)

        # Combine with learned weights
        alpha, beta, gamma = self.weights
        combined = (alpha * u_semantic_norm +
                   beta * u_structural_norm +
                   gamma * u_predictive_norm)

        if return_components:
            return combined, {
                'semantic': u_semantic,
                'structural': u_structural,
                'predictive': u_predictive,
                'semantic_norm': u_semantic_norm,
                'structural_norm': u_structural_norm,
                'predictive_norm': u_predictive_norm,
                'weights': self.weights.detach(),
            }

        return combined

    def get_uncertainty_components(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get all uncertainty components separately."""
        _, components = self.get_uncertainty(
            heads, relations, tails, return_components=True, top_k=top_k
        )
        return components

    def precompute_coverage(self, triples, entity_to_idx=None, relation_to_idx=None):
        """Precompute coverage from training triples."""
        if entity_to_idx is not None:
            # Triples are strings, need conversion
            for h, r, t in triples:
                h_idx = entity_to_idx[h]
                r_idx = relation_to_idx[r]
                t_idx = entity_to_idx[t]
                self.coverage[h_idx, r_idx] = 1.0
                self.coverage[t_idx, r_idx] = 1.0
                self.coverage_freq[h_idx, r_idx] += 1.0
                self.coverage_freq[t_idx, r_idx] += 1.0
        else:
            # Triples are already indices (numpy array)
            for i in range(len(triples)):
                h, r, t = triples[i]
                self.coverage[h, r] = 1.0
                self.coverage[t, r] = 1.0
                self.coverage_freq[h, r] += 1.0
                self.coverage_freq[t, r] += 1.0

    def kl_loss(self) -> torch.Tensor:
        """KL divergence from standard normal prior."""
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities

    def get_stats(self) -> Dict[str, float]:
        """Get model statistics for analysis."""
        weights = self.weights.detach().cpu()
        return {
            'alpha_semantic': weights[0].item(),
            'beta_structural': weights[1].item(),
            'gamma_predictive': weights[2].item(),
            'temperature': self.temperature.item(),
            'coverage_density': (self.coverage.sum() / self.coverage.numel()).item(),
        }


class PredictiveCAGPTrainer:
    """Trainer for Predictive CAGP."""

    def __init__(self, model: PredictiveCAGP, lr: float = 0.001, kl_weight: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight

    def train_epoch(self, dataloader, device: str = 'cuda') -> float:
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            if len(batch) == 3:
                batch_h, batch_r, batch_t = batch
            else:
                batch_h, batch_r, batch_t = batch[0], batch[1], batch[2]

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
            loss += self.kl_weight * self.model.kl_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
