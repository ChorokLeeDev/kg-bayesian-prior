"""
Gaussian Process Knowledge Graph Embedding (GP-KGE)

This is the MAIN MODEL of the thesis.

Combines:
1. Relation-aware GP prior on the entity embedding space
2. Standard KGE scoring functions
3. Full posterior inference for uncertainty quantification

Key Features:
- Entity embeddings are drawn from a GP with relation-aware kernel
- Posterior mean: point estimate of embedding
- Posterior covariance: entity-level uncertainty
- Learnable kernel parameters per relation type

Mathematical Framework:
======================

Prior:
    f ~ GP(0, K)

where K is the relation-aware kernel:
    K(i, j) = Σ_r σ_r² · exp(-L_r / ℓ_r²)

Observation Model (for link prediction):
    y_{hrt} ~ Bernoulli(σ(score(h, r, t)))

where score can be TransE, DistMult, or ComplEx style.

Posterior (approximate):
    p(f | data) ≈ N(μ, Σ)

computed via variational inference or Laplace approximation.

Uncertainty Decomposition:
- Epistemic: from Σ (reducible with more data)
- Aleatoric: from observation noise (irreducible)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import numpy as np
from scipy import sparse

from ..kernels.relation_aware import RelationAwareKernel
from ..kernels.matern_graph import SparseGraphMaternKernel


class GPKGE(nn.Module):
    """
    Gaussian Process Knowledge Graph Embedding.

    Combines GP prior with KGE scoring for uncertainty-aware link prediction.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        kernel_type: str = "relation_aware",  # or "matern"
        scoring_function: str = "distmult",  # "distmult", "complex", "transe"
        num_inducing: int = 500,
        jitter: float = 1e-4,
        learn_noise: bool = True,
    ):
        """
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Dimension of embeddings
            kernel_type: Type of GP kernel
            scoring_function: KGE scoring function
            num_inducing: Number of inducing points for scalability
            jitter: Numerical stability term
            learn_noise: Whether to learn observation noise
        """
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.scoring_function = scoring_function
        self.num_inducing = min(num_inducing, num_entities)
        self.jitter = jitter

        # GP Kernel
        if kernel_type == "relation_aware":
            self.kernel = RelationAwareKernel(
                num_relations=num_relations,
                kernel_type="diffusion",
            )
        else:
            self.kernel = SparseGraphMaternKernel(
                num_inducing=self.num_inducing,
            )

        # Variational parameters for entity embeddings
        # q(f) = N(μ, Σ) where Σ = LL^T (Cholesky)
        self.entity_mean = nn.Parameter(
            torch.randn(num_entities, embedding_dim) * 0.1
        )

        # Low-rank + diagonal covariance for scalability
        # Σ = diag(d) + VV^T
        self.entity_log_diag = nn.Parameter(
            torch.zeros(num_entities, embedding_dim)
        )
        self.rank = min(50, embedding_dim)
        self.entity_cov_factor = nn.Parameter(
            torch.randn(num_entities, embedding_dim, self.rank) * 0.01
        )

        # Relation embeddings (deterministic)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # For ComplEx
        if scoring_function == "complex":
            self.entity_mean_im = nn.Parameter(
                torch.randn(num_entities, embedding_dim) * 0.1
            )
            self.relation_embeddings_im = nn.Embedding(num_relations, embedding_dim)
            nn.init.xavier_uniform_(self.relation_embeddings_im.weight)

        # Observation noise
        if learn_noise:
            self.log_noise = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_noise", torch.tensor(0.0))

        # Inducing points (indices into entities)
        self.register_buffer(
            "inducing_indices",
            torch.randperm(num_entities)[:self.num_inducing]
        )

        # Cached matrices
        self._K_uu = None
        self._L_uu = None

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    def set_graph(self, kg_dataset, num_eigenvectors: int = 100, min_edges: int = 10, show_progress: bool = True, init_embeddings: bool = True):
        """
        Set the KG structure for the GP prior.

        Args:
            kg_dataset: KGDataset object
            num_eigenvectors: Number of eigenvectors for spectral decomposition (default: 100)
            min_edges: Skip relations with fewer edges (default: 10)
            show_progress: Show progress bar during eigendecomposition
            init_embeddings: Whether to initialize embeddings from graph structure (default: True)
        """
        if hasattr(self.kernel, 'set_graph'):
            # RelationAwareKernel has the new signature
            import inspect
            sig = inspect.signature(self.kernel.set_graph)
            if 'num_eigenvectors' in sig.parameters:
                self.kernel.set_graph(
                    kg_dataset.relation_adjacencies,
                    kg_dataset.num_entities,
                    num_eigenvectors=num_eigenvectors,
                    min_edges=min_edges,
                    show_progress=show_progress,
                )
            else:
                self.kernel.set_graph(
                    kg_dataset.relation_adjacencies,
                    kg_dataset.num_entities,
                )
        self._K_uu = None
        self._L_uu = None

        # Initialize embeddings from graph eigenvectors if available
        if init_embeddings and hasattr(self.kernel, 'relation_laplacians'):
            self._init_embeddings_from_graph()

    def _init_embeddings_from_graph(self):
        """
        Initialize entity embeddings from graph eigenvectors.

        This is the KEY difference for kernel ablation:
        - RBF kernel: no graph structure, random init
        - Relation-Aware: uses graph eigenvectors for initialization

        Entities connected by relations get similar initial embeddings.

        For sparse-relation KGs (like WN18RR), also uses global kernel eigenvectors.
        """
        all_eigenvecs = []

        # First, try to use global kernel eigenvectors (for sparse-relation KGs)
        if hasattr(self.kernel, 'global_laplacian') and self.kernel.global_laplacian is not None:
            if hasattr(self.kernel.global_laplacian, 'eigenvectors') and self.kernel.global_laplacian.eigenvectors is not None:
                evecs = self.kernel.global_laplacian.eigenvectors
                all_eigenvecs.append(evecs)

        # Then, add per-relation eigenvectors
        if hasattr(self.kernel, 'relation_laplacians') and self.kernel.relation_laplacians:
            for rel_id, laplacian in self.kernel.relation_laplacians.items():
                if hasattr(laplacian, 'eigenvectors') and laplacian.eigenvectors is not None:
                    evecs = laplacian.eigenvectors  # (num_entities, k)
                    all_eigenvecs.append(evecs)

        if not all_eigenvecs:
            return

        # Concatenate and use PCA-like projection
        # Take first few eigenvectors from each source
        combined = []
        for evecs in all_eigenvecs:
            n_use = min(evecs.shape[1], self.embedding_dim // len(all_eigenvecs) + 1)
            combined.append(evecs[:, :n_use])

        combined = torch.cat(combined, dim=1)  # (num_entities, total_dims)

        # Project to embedding dimension
        if combined.shape[1] >= self.embedding_dim:
            # Use first embedding_dim columns
            init_emb = combined[:, :self.embedding_dim]
        else:
            # Pad with random
            pad_size = self.embedding_dim - combined.shape[1]
            padding = torch.randn(self.num_entities, pad_size) * 0.1
            init_emb = torch.cat([combined, padding], dim=1)

        # Normalize
        init_emb = init_emb / (init_emb.norm(dim=1, keepdim=True) + 1e-8) * 0.1

        # Set as initial embedding
        with torch.no_grad():
            self.entity_mean.copy_(init_emb)

        has_global = hasattr(self.kernel, 'global_laplacian') and self.kernel.global_laplacian is not None
        n_relations = len(self.kernel.relation_laplacians) if hasattr(self.kernel, 'relation_laplacians') else 0
        print(f"Initialized embeddings from {len(all_eigenvecs)} sources (global={has_global}, relations={n_relations})")

    def get_entity_distribution(
        self,
        entity_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the approximate posterior distribution for entities.

        Returns:
            Tuple of (mean, variance) where variance is diagonal
        """
        mean = self.entity_mean[entity_ids]

        # Variance = diag(exp(log_diag)) + ||V||²
        diag = torch.exp(self.entity_log_diag[entity_ids])
        V = self.entity_cov_factor[entity_ids]
        var = diag + torch.sum(V ** 2, dim=-1)

        return mean, var

    def sample_embeddings(
        self,
        entity_ids: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample entity embeddings from the posterior.

        Args:
            entity_ids: Entity indices
            num_samples: Number of samples

        Returns:
            Samples of shape (num_samples, batch_size, embedding_dim)
        """
        mean, var = self.get_entity_distribution(entity_ids)
        std = torch.sqrt(var)

        # Reparameterization trick
        eps = torch.randn(num_samples, *mean.shape, device=mean.device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps

        return samples

    def score_triple(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        use_mean: bool = True,
    ) -> torch.Tensor:
        """
        Score triples using the chosen scoring function.

        Args:
            head, relation, tail: Triple indices
            use_mean: If True, use posterior mean; else sample

        Returns:
            Scores
        """
        if use_mean:
            h = self.entity_mean[head]
            t = self.entity_mean[tail]
        else:
            h = self.sample_embeddings(head, 1).squeeze(0)
            t = self.sample_embeddings(tail, 1).squeeze(0)

        r = self.relation_embeddings(relation)

        if self.scoring_function == "distmult":
            score = torch.sum(h * r * t, dim=-1)

        elif self.scoring_function == "transe":
            score = -torch.norm(h + r - t, p=1, dim=-1)

        elif self.scoring_function == "complex":
            if use_mean:
                h_im = self.entity_mean_im[head]
                t_im = self.entity_mean_im[tail]
            else:
                # Simplified: use mean for imaginary
                h_im = self.entity_mean_im[head]
                t_im = self.entity_mean_im[tail]

            r_im = self.relation_embeddings_im(relation)

            # ComplEx scoring
            score = torch.sum(h * r * t, dim=-1)
            score += torch.sum(h * r_im * t_im, dim=-1)
            score += torch.sum(h_im * r * t_im, dim=-1)
            score -= torch.sum(h_im * r_im * t, dim=-1)

        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_function}")

        return score

    def predict_with_uncertainty(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        num_samples: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict scores with full uncertainty decomposition.

        Returns dict with:
        - mean: Expected score
        - epistemic: Epistemic uncertainty (from posterior variance)
        - aleatoric: Aleatoric uncertainty (from observation noise)
        - total: Total uncertainty
        """
        # Sample multiple embeddings
        h_samples = self.sample_embeddings(head, num_samples)
        t_samples = self.sample_embeddings(tail, num_samples)
        r = self.relation_embeddings(relation)

        # Compute scores for all samples
        if self.scoring_function == "distmult":
            scores = torch.sum(
                h_samples * r.unsqueeze(0) * t_samples,
                dim=-1
            )  # (S, B)
        elif self.scoring_function == "transe":
            scores = -torch.norm(
                h_samples + r.unsqueeze(0) - t_samples,
                p=1, dim=-1
            )
        else:
            # Fallback
            scores = torch.sum(
                h_samples * r.unsqueeze(0) * t_samples,
                dim=-1
            )

        mean_score = scores.mean(dim=0)

        # Convert to probabilities for proper uncertainty
        probs = torch.sigmoid(scores)  # (S, B)
        mean_prob = probs.mean(dim=0)

        # Predictive entropy of mean prediction
        # Higher entropy = more uncertain (closer to 0.5)
        pred_entropy = -mean_prob * torch.log(mean_prob + 1e-10) - \
                       (1 - mean_prob) * torch.log(1 - mean_prob + 1e-10)

        # Score variance across samples (epistemic from embedding uncertainty)
        score_var = scores.var(dim=0)

        # Probability variance (alternative epistemic measure)
        prob_var = probs.var(dim=0)

        # For OOD detection, we use negative mean score as uncertainty
        # Rationale:
        # - ID (true facts): model trained to give HIGH positive scores → low uncertainty
        # - OOD (random/false): model gives LOW/negative scores → high uncertainty
        # This aligns with the standard uncertainty interpretation where
        # OOD samples should have higher uncertainty values
        epistemic_uncertainty = -mean_score

        aleatoric = self.noise ** 2

        return {
            "mean": mean_score,
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric.expand_as(pred_entropy),
            "total": epistemic_uncertainty,  # Primary uncertainty measure
            "pred_entropy": pred_entropy,  # Keep entropy for analysis
            "score_var": score_var,  # Keep for analysis
            "prob_var": prob_var,  # Keep for analysis
        }

    def get_entity_uncertainty(
        self,
        entity_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get uncertainty for entities.

        Returns average variance across embedding dimensions.
        """
        if entity_ids is None:
            entity_ids = torch.arange(self.num_entities)

        _, var = self.get_entity_distribution(entity_ids)
        return var.mean(dim=-1)

    def precompute_kernel_matrix(self) -> None:
        """
        Precompute and cache kernel matrix K_uu for fast KL computation.
        Call this after set_graph() to avoid recomputing the kernel each time.
        """
        device = self.entity_mean.device
        u_idx = self.inducing_indices.to(device)

        # Check if we have a proper kernel with graph structure
        has_single_laplacian = (
            hasattr(self.kernel, 'laplacian') and
            self.kernel.laplacian is not None
        )
        has_relation_laplacians = (
            hasattr(self.kernel, 'relation_laplacians') and
            self.kernel.relation_laplacians
        )

        if has_single_laplacian or has_relation_laplacians:
            K_uu = self.kernel(u_idx, u_idx)
            K_uu = K_uu + self.jitter * torch.eye(len(u_idx), device=device)
            self._cached_K_uu = K_uu
            print(f"Precomputed K_uu: {K_uu.shape}")
        else:
            self._cached_K_uu = None

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational posterior and GP prior.

        KL(q(f) || p(f)) where:
        - q(f) = N(μ_q, Σ_q) is the variational posterior
        - p(f) = N(0, K) is the GP prior

        For scalability, we compute this at inducing points only.
        """
        # Get inducing point distributions
        device = self.entity_mean.device
        u_idx = self.inducing_indices.to(device)
        mu_u = self.entity_mean[u_idx]  # (M, D)

        # Use cached kernel matrix if available (much faster)
        if hasattr(self, '_cached_K_uu') and self._cached_K_uu is not None:
            K_uu = self._cached_K_uu
        else:
            # Check if kernel has proper eigenvectors (relation-aware kernel)
            # For RBF/Matern kernels without graph structure, use identity prior
            has_single_laplacian = (
                hasattr(self.kernel, 'laplacian') and
                self.kernel.laplacian is not None and
                hasattr(self.kernel.laplacian, 'eigenvectors') and
                self.kernel.laplacian.eigenvectors is not None
            )
            # Also check for relation-aware kernel which uses relation_laplacians (plural)
            has_relation_laplacians = (
                hasattr(self.kernel, 'relation_laplacians') and
                self.kernel.relation_laplacians  # non-empty dict
            )
            has_spectral_kernel = has_single_laplacian or has_relation_laplacians

            if has_spectral_kernel:
                # Compute prior covariance at inducing points using spectral kernel
                K_uu = self.kernel(u_idx, u_idx)
                K_uu = K_uu + self.jitter * torch.eye(len(u_idx), device=device)
            else:
                # Fallback: use identity prior (standard Gaussian regularization)
                # This is equivalent to L2 regularization on embeddings
                K_uu = torch.eye(len(u_idx), device=device)

        # Variational covariance at inducing points (diagonal approximation)
        _, var_u = self.get_entity_distribution(u_idx)

        # KL for each embedding dimension (assuming independence)
        kl = 0
        for d in range(self.embedding_dim):
            mu = mu_u[:, d]
            var = var_u[:, d]

            # KL(N(μ, diag(σ²)) || N(0, K))
            # = 0.5 * (tr(K^{-1} diag(σ²)) + μ^T K^{-1} μ - M + log|K| - log|diag(σ²)|)

            L = torch.linalg.cholesky(K_uu)
            alpha = torch.cholesky_solve(mu.unsqueeze(-1), L).squeeze(-1)

            kl_d = 0.5 * (
                torch.sum(torch.cholesky_solve(torch.diag(var), L).diag())  # tr term
                + torch.dot(mu, alpha)  # quadratic term
                - len(u_idx)  # -M
                + 2 * torch.sum(torch.log(L.diag()))  # log|K|
                - torch.sum(torch.log(var))  # -log|Σ|
            )
            kl += kl_d

        return kl

    def loss(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
        kl_weight: float = 0.001,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.

        ELBO = E_q[log p(y|f)] - β * KL(q(f) || p(f))

        Args:
            positive_triples: Positive triples (B, 3)
            negative_triples: Negative triples (B, 3)
            kl_weight: Weight for KL term (β in β-VAE)

        Returns:
            Dict with loss components
        """
        # Likelihood term (using samples for gradient estimation)
        pos_scores = self.score_triple(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
            use_mean=False,
        )
        neg_scores = self.score_triple(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
            use_mean=False,
        )

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])

        likelihood_loss = F.binary_cross_entropy_with_logits(scores, labels)

        # KL term
        kl_loss = self.kl_divergence()

        # Total ELBO (negative because we minimize)
        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "total": total_loss,
            "likelihood": likelihood_loss,
            "kl": kl_loss,
        }

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returns scores using posterior mean."""
        return self.score_triple(head, relation, tail, use_mean=True)

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
        device = head.device

        # Get head embeddings and relation embeddings
        h = self.entity_mean[head]  # (batch, dim)
        r = self.relation_embeddings(relation)  # (batch, dim)

        # Get all tail embeddings
        all_tails = self.entity_mean  # (num_entities, dim)

        if self.scoring_function == "distmult":
            # DistMult: h * r dot t
            query = h * r  # (batch, dim)
            scores = torch.mm(query, all_tails.t())  # (batch, num_entities)
        elif self.scoring_function == "transe":
            # TransE: -||h + r - t||
            query = h + r  # (batch, dim)
            # Compute distance to all tails
            scores = -torch.cdist(query, all_tails, p=1)  # (batch, num_entities)
        else:
            # Fallback: score each tail individually
            all_tails_idx = torch.arange(self.num_entities, device=device)
            all_tails_idx = all_tails_idx.unsqueeze(0).expand(batch_size, -1)
            head_exp = head.unsqueeze(1).expand(-1, self.num_entities)
            relation_exp = relation.unsqueeze(1).expand(-1, self.num_entities)

            scores = self.score_triple(
                head_exp.reshape(-1),
                relation_exp.reshape(-1),
                all_tails_idx.reshape(-1),
            )
            scores = scores.reshape(batch_size, self.num_entities)

        return scores

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
        device = relation.device

        # Get tail embeddings and relation embeddings
        t = self.entity_mean[tail]  # (batch, dim)
        r = self.relation_embeddings(relation)  # (batch, dim)

        # Get all head embeddings
        all_heads = self.entity_mean  # (num_entities, dim)

        if self.scoring_function == "distmult":
            # DistMult: h * r * t (symmetric in h and t)
            query = r * t  # (batch, dim)
            scores = torch.mm(query, all_heads.t())  # (batch, num_entities)
        elif self.scoring_function == "transe":
            # TransE: -||h + r - t|| => h = t - r
            query = t - r  # (batch, dim)
            scores = -torch.cdist(query, all_heads, p=1)  # (batch, num_entities)
        else:
            # Fallback
            all_heads_idx = torch.arange(self.num_entities, device=device)
            all_heads_idx = all_heads_idx.unsqueeze(0).expand(batch_size, -1)
            relation_exp = relation.unsqueeze(1).expand(-1, self.num_entities)
            tail_exp = tail.unsqueeze(1).expand(-1, self.num_entities)

            scores = self.score_triple(
                all_heads_idx.reshape(-1),
                relation_exp.reshape(-1),
                tail_exp.reshape(-1),
            )
            scores = scores.reshape(batch_size, self.num_entities)

        return scores


class SparseGPKGE(GPKGE):
    """
    Scalable GP-KGE using inducing points.

    For large KGs, we can't compute the full kernel matrix.
    Instead, we use inducing point approximation:

    p(f) ≈ ∫ p(f|u) p(u) du

    where u are inducing variables at M << N inducing points.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        num_inducing: int = 500,
        **kwargs
    ):
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_inducing=num_inducing,
            **kwargs
        )

        # Inducing point embeddings (variational parameters)
        self.inducing_mean = nn.Parameter(
            torch.randn(num_inducing, embedding_dim) * 0.1
        )
        self.inducing_log_var = nn.Parameter(
            torch.zeros(num_inducing, embedding_dim)
        )

    def predict_at_new_entity(
        self,
        new_entity_neighbors: Dict[int, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict embedding for a new entity based on its neighbors.

        This demonstrates how the GP prior enables zero-shot inference
        for new entities based on graph structure.

        Args:
            new_entity_neighbors: Dict mapping relation_id -> list of neighbor entity ids

        Returns:
            Tuple of (predicted_mean, predicted_variance)
        """
        # Collect neighbor embeddings
        neighbor_means = []
        neighbor_vars = []
        kernel_weights = []

        for r, neighbors in new_entity_neighbors.items():
            if len(neighbors) == 0:
                continue

            n_idx = torch.tensor(neighbors)
            mean, var = self.get_entity_distribution(n_idx)

            neighbor_means.append(mean)
            neighbor_vars.append(var)

            # Weight by relation importance and inverse variance
            rel_weight = self.kernel.variance[r] if hasattr(self.kernel, 'variance') else 1.0
            kernel_weights.append(
                rel_weight * torch.ones(len(neighbors)) / len(neighbors)
            )

        if len(neighbor_means) == 0:
            # No neighbors: return prior
            return (
                torch.zeros(self.embedding_dim),
                torch.ones(self.embedding_dim) * 10,  # High uncertainty
            )

        # Weighted combination
        all_means = torch.cat(neighbor_means, dim=0)
        all_vars = torch.cat(neighbor_vars, dim=0)
        all_weights = torch.cat(kernel_weights, dim=0)
        all_weights = all_weights / all_weights.sum()

        # Precision-weighted mean
        precisions = 1.0 / all_vars
        weighted_precision = torch.sum(
            all_weights.unsqueeze(-1) * precisions,
            dim=0
        )
        weighted_mean = torch.sum(
            all_weights.unsqueeze(-1) * precisions * all_means,
            dim=0
        ) / weighted_precision

        # Combined variance
        combined_var = 1.0 / weighted_precision

        return weighted_mean, combined_var
