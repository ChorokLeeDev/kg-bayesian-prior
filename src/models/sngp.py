"""
Spectral-Normalized Gaussian Process (SNGP) for Knowledge Graph Embeddings.

Based on: Liu et al., "Simple and Principled Uncertainty Estimation with
Deterministic Deep Learning via Distance Awareness" (NeurIPS 2020)

Key ideas:
1. Spectral normalization on embedding layers for Lipschitz constraint
2. Random Fourier Features (RFF) approximation for GP output layer
3. Uncertainty = distance to training data in feature space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SpectralNormalizedLinear(nn.Module):
    """Linear layer with spectral normalization."""

    def __init__(self, in_features: int, out_features: int, n_power_iterations: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear = nn.utils.spectral_norm(self.linear, n_power_iterations=n_power_iterations)

    def forward(self, x):
        return self.linear(x)


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for GP approximation.

    Approximates RBF kernel: k(x, x') ≈ φ(x)^T φ(x')
    where φ(x) = sqrt(2/D) * [cos(Wx + b), sin(Wx + b)]
    """

    def __init__(
        self,
        in_features: int,
        num_features: int = 1024,
        lengthscale: float = 1.0,
        trainable_lengthscale: bool = True
    ):
        super().__init__()
        self.num_features = num_features

        # Random weights (frozen)
        self.register_buffer(
            'random_weights',
            torch.randn(in_features, num_features) / lengthscale
        )
        self.register_buffer(
            'random_bias',
            torch.rand(num_features) * 2 * np.pi
        )

        # Learnable lengthscale
        if trainable_lengthscale:
            self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
        else:
            self.register_buffer('log_lengthscale', torch.tensor(np.log(lengthscale)))

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute random Fourier features."""
        # Scale by lengthscale
        scaled_weights = self.random_weights / self.lengthscale

        # Project and apply nonlinearity
        projection = x @ scaled_weights + self.random_bias
        features = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

        # Normalize
        return features * np.sqrt(2.0 / self.num_features)


class LaplacianGPOutputLayer(nn.Module):
    """
    GP output layer using Laplace approximation.

    Maintains a precision matrix Λ = Φ^T Φ + λI for computing
    predictive variance.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int = 1,
        ridge_penalty: float = 1.0,
        momentum: float = 0.999
    ):
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        # Output weights
        self.output_layer = nn.Linear(num_features, num_outputs, bias=True)

        # Running precision matrix (for uncertainty)
        # Λ = (1/N) Σ φ(x)φ(x)^T + λI
        self.register_buffer(
            'precision_matrix',
            torch.eye(num_features) * ridge_penalty
        )
        self.register_buffer('num_samples_seen', torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits."""
        return self.output_layer(features)

    def update_precision(self, features: torch.Tensor):
        """Update precision matrix with new features (during training)."""
        if not self.training:
            return

        batch_size = features.shape[0]

        # Compute batch precision: (1/B) * Φ^T Φ
        batch_precision = features.T @ features / batch_size

        # Exponential moving average update
        self.precision_matrix = (
            self.momentum * self.precision_matrix +
            (1 - self.momentum) * batch_precision
        )
        self.num_samples_seen += batch_size

    def compute_predictive_variance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive variance for uncertainty estimation.

        Var = φ(x)^T Λ^{-1} φ(x)
        """
        # Add ridge to ensure invertibility
        precision = self.precision_matrix + self.ridge_penalty * torch.eye(
            self.num_features, device=features.device
        )

        # Compute variance via solve (more stable than inverse)
        # variance = φ^T Λ^{-1} φ = (Λ^{-1} φ)^T φ
        try:
            # Cholesky solve
            L = torch.linalg.cholesky(precision)
            solved = torch.cholesky_solve(features.T, L).T  # (batch, features)
            variance = (features * solved).sum(dim=-1)  # (batch,)
        except:
            # Fallback to lstsq
            solved = torch.linalg.lstsq(precision, features.T).solution.T
            variance = (features * solved).sum(dim=-1)

        return variance


class SNGP(nn.Module):
    """
    Spectral-Normalized Gaussian Process for KG embeddings.

    Simplified architecture:
    1. Standard entity/relation embeddings
    2. DistMult scoring
    3. Distance-based uncertainty using embedding norms and density
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        num_rff_features: int = 1024,
        ridge_penalty: float = 1.0,
        spectral_norm_layers: bool = True,
        scoring_function: str = 'distmult'
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.scoring_function = scoring_function
        self.ridge_penalty = ridge_penalty

        # Standard embeddings (no spectral norm on embeddings for stability)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Feature extractor for uncertainty
        hidden_dim = embedding_dim * 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Apply spectral normalization to feature extractor
        if spectral_norm_layers:
            for i, layer in enumerate(self.feature_extractor):
                if isinstance(layer, nn.Linear):
                    self.feature_extractor[i] = nn.utils.spectral_norm(layer)

        # Random Fourier Features for GP approximation
        self.rff = RandomFourierFeatures(
            in_features=hidden_dim,
            num_features=num_rff_features,
            lengthscale=1.0
        )

        # GP output layer
        self.gp_layer = LaplacianGPOutputLayer(
            num_features=num_rff_features * 2,  # cos + sin features
            num_outputs=1,
            ridge_penalty=ridge_penalty
        )

        # Coverage buffer (for compatibility)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Entity frequency for distance-based uncertainty
        self.register_buffer('entity_freq', torch.zeros(num_entities))

    def get_triple_features(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """Extract features for triples."""
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        # Concatenate
        triple_repr = torch.cat([h, r, t], dim=-1)

        # Extract features
        features = self.feature_extractor(triple_repr)

        # Apply RFF
        rff_features = self.rff(features)

        return rff_features

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """Compute scores for triples."""
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        if self.scoring_function == 'distmult':
            scores = (h * r * t).sum(dim=-1)
        elif self.scoring_function == 'transe':
            scores = -torch.norm(h + r - t, p=1, dim=-1)
        else:
            scores = (h * r * t).sum(dim=-1)

        return scores

    def forward_with_uncertainty(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        update_precision: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            scores: Triple scores
            uncertainty: Predictive variance (higher = more uncertain)
        """
        # Get features
        features = self.get_triple_features(heads, relations, tails)

        # Update precision matrix during training
        if update_precision and self.training:
            self.gp_layer.update_precision(features)

        # Get scores
        scores = self.forward(heads, relations, tails)

        # Compute uncertainty
        uncertainty = self.gp_layer.compute_predictive_variance(features)

        return scores, uncertainty

    def precompute_coverage(self, triples):
        """Compute coverage and entity frequency from training triples."""
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0
            self.entity_freq[h] += 1
            self.entity_freq[t] += 1

    def get_uncertainty(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Get uncertainty for OOD detection.

        Combines:
        1. GP predictive variance (distance to training data)
        2. Entity frequency (rare entities = higher uncertainty)
        """
        # GP-based uncertainty from feature space distance
        features = self.get_triple_features(heads, relations, tails)
        gp_variance = self.gp_layer.compute_predictive_variance(features)

        # Entity frequency-based uncertainty
        h_freq = self.entity_freq[heads]
        t_freq = self.entity_freq[tails]
        max_freq = self.entity_freq.max() + 1
        freq_unc = 2.0 - (h_freq / max_freq) - (t_freq / max_freq)

        # Combine (GP variance is primary for SNGP)
        # Normalize GP variance
        gp_var_norm = gp_variance / (gp_variance.mean() + 1e-8)

        return gp_var_norm + 0.1 * freq_unc

    def fit_precision(self, train_loader, device, max_batches: int = 100):
        """
        Fit the precision matrix on training data.

        Should be called after training to compute proper uncertainty estimates.
        """
        self.eval()

        # Reset precision
        self.gp_layer.precision_matrix = torch.eye(
            self.gp_layer.num_features, device=device
        ) * self.gp_layer.ridge_penalty

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= max_batches:
                    break

                heads, relations, tails = batch
                heads = heads.to(device)
                relations = relations.to(device)
                tails = tails.to(device)

                features = self.get_triple_features(heads, relations, tails)

                # Accumulate precision
                batch_precision = features.T @ features / features.shape[0]

                if i == 0:
                    self.gp_layer.precision_matrix = batch_precision
                else:
                    # Running average
                    self.gp_layer.precision_matrix = (
                        i / (i + 1) * self.gp_layer.precision_matrix +
                        1 / (i + 1) * batch_precision
                    )

        # Add ridge
        self.gp_layer.precision_matrix += self.gp_layer.ridge_penalty * torch.eye(
            self.gp_layer.num_features, device=device
        )
