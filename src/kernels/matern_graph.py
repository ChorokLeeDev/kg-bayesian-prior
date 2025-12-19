"""
Matérn Gaussian Process on Graphs

Reference: Borovitskiy et al. (2021) - Matérn Gaussian Processes on Graphs

The Matérn kernel on graphs is defined via the spectral decomposition:
    K = σ² (2ν/κ² + L)^{-ν}

where:
- L is the graph Laplacian
- ν controls smoothness
- κ is the lengthscale (inverse)
- σ² is the variance

Special cases:
- ν → ∞: Squared exponential (diffusion kernel)
- ν = 1/2: Exponential kernel (Laplacian kernel)
"""

from typing import Optional
import torch
import torch.nn as nn
from scipy import sparse
import numpy as np

from .base import BaseGraphKernel, GraphLaplacian


class GraphMaternKernel(BaseGraphKernel):
    """
    Matérn kernel on graphs.

    K = σ² (2ν/κ² + L)^{-ν}

    This is equivalent to the Matérn covariance in Euclidean space,
    but uses the graph Laplacian to define distances.
    """

    def __init__(
        self,
        nu: float = 2.5,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        learnable: bool = True,
    ):
        """
        Args:
            nu: Smoothness parameter (common values: 0.5, 1.5, 2.5)
            lengthscale: Lengthscale parameter (larger = smoother)
            variance: Output variance
            learnable: Whether parameters are learnable
        """
        super().__init__()

        # Store nu directly (discrete choices are common)
        self.nu = nu

        if learnable:
            # Use log parameterization for positivity
            self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
            self.log_variance = nn.Parameter(torch.tensor(np.log(variance)))
        else:
            self.register_buffer("log_lengthscale", torch.tensor(np.log(lengthscale)))
            self.register_buffer("log_variance", torch.tensor(np.log(variance)))

        self.laplacian = GraphLaplacian(normalized=True)
        self._kernel_matrix = None

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def kappa(self) -> torch.Tensor:
        """Inverse lengthscale."""
        return 1.0 / self.lengthscale

    def set_graph(self, adjacency: sparse.csr_matrix, **kwargs):
        """Set the graph structure."""
        self.laplacian.set_graph(adjacency, **kwargs)
        self._kernel_matrix = None

    def _spectral_density(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density S(λ).

        For Matérn: S(λ) = (2ν/κ² + λ)^{-ν}
        """
        kappa_sq = self.kappa ** 2
        return torch.pow(2 * self.nu / kappa_sq + eigenvalues, -self.nu)

    def _compute_kernel_matrix(self) -> torch.Tensor:
        """Compute full kernel matrix."""
        if self._kernel_matrix is not None:
            return self._kernel_matrix

        def matern_func(eigenvalues):
            S = self._spectral_density(eigenvalues)
            # Normalize so that K(i,i) ≈ variance
            S = S / S[0]  # Normalize by smallest eigenvalue component
            return self.variance * S

        K = self.laplacian.apply_function(matern_func)
        self._kernel_matrix = K
        return K

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute kernel values."""
        K = self._compute_kernel_matrix()

        if x2 is None:
            x2 = x1

        return K[x1.unsqueeze(1), x2.unsqueeze(0)]

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal."""
        K = self._compute_kernel_matrix()
        return K[x, x]


class SparseGraphMaternKernel(BaseGraphKernel):
    """
    Sparse Matérn kernel using inducing points for scalability.

    Uses Nyström approximation:
        K ≈ K_nm K_mm^{-1} K_mn

    where m << n inducing points are selected.
    """

    def __init__(
        self,
        nu: float = 2.5,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        num_inducing: int = 100,
        learnable: bool = True,
    ):
        """
        Args:
            nu: Smoothness parameter
            lengthscale: Lengthscale
            variance: Output variance
            num_inducing: Number of inducing points
            learnable: Whether parameters are learnable
        """
        super().__init__()

        self.nu = nu
        self.num_inducing = num_inducing

        if learnable:
            self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
            self.log_variance = nn.Parameter(torch.tensor(np.log(variance)))
        else:
            self.register_buffer("log_lengthscale", torch.tensor(np.log(lengthscale)))
            self.register_buffer("log_variance", torch.tensor(np.log(variance)))

        self.laplacian = GraphLaplacian(normalized=True)

        # Inducing point indices (will be set when graph is provided)
        self.register_buffer("inducing_indices", None)

        # Cached matrices
        self._K_mm = None
        self._K_mm_inv = None

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def kappa(self) -> torch.Tensor:
        return 1.0 / self.lengthscale

    def set_graph(
        self,
        adjacency: sparse.csr_matrix,
        inducing_indices: Optional[torch.Tensor] = None,
    ):
        """
        Set graph and select inducing points.

        Args:
            adjacency: Graph adjacency matrix
            inducing_indices: Predefined inducing point indices.
                            If None, selects randomly.
        """
        num_nodes = adjacency.shape[0]
        self.laplacian.set_graph(adjacency, num_eigenvectors=min(1000, num_nodes - 1))

        # Select inducing points
        if inducing_indices is None:
            # Random selection (could use k-DPP or other methods)
            indices = torch.randperm(num_nodes)[:self.num_inducing]
            self.inducing_indices = indices.sort().values
        else:
            self.inducing_indices = inducing_indices

        self._K_mm = None
        self._K_mm_inv = None

    def _spectral_density(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Compute spectral density for Matérn."""
        kappa_sq = self.kappa ** 2
        S = torch.pow(2 * self.nu / kappa_sq + eigenvalues, -self.nu)
        S = S / S[0]  # Normalize
        return self.variance * S

    def _compute_K_mm(self) -> torch.Tensor:
        """Compute kernel between inducing points."""
        if self._K_mm is not None:
            return self._K_mm

        U_m = self.laplacian.eigenvectors[self.inducing_indices]  # (m, k)
        weights = self._spectral_density(self.laplacian.eigenvalues)  # (k,)

        K_mm = (U_m * weights.unsqueeze(0)) @ U_m.T
        self._K_mm = K_mm

        # Also compute inverse
        self._K_mm_inv = torch.linalg.inv(K_mm + 1e-6 * torch.eye(K_mm.size(0)))

        return K_mm

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel using Nyström approximation.

        K(x1, x2) ≈ K(x1, m) K(m, m)^{-1} K(m, x2)
        """
        if x2 is None:
            x2 = x1

        K_mm = self._compute_K_mm()
        K_mm_inv = self._K_mm_inv

        U_1 = self.laplacian.eigenvectors[x1]  # (n1, k)
        U_2 = self.laplacian.eigenvectors[x2]  # (n2, k)
        U_m = self.laplacian.eigenvectors[self.inducing_indices]  # (m, k)

        weights = self._spectral_density(self.laplacian.eigenvalues)

        K_1m = (U_1 * weights.unsqueeze(0)) @ U_m.T  # (n1, m)
        K_m2 = (U_m * weights.unsqueeze(0)) @ U_2.T  # (m, n2)

        # Nyström: K ≈ K_1m @ K_mm_inv @ K_m2
        K = K_1m @ K_mm_inv @ K_m2

        return K

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal efficiently."""
        K_mm = self._compute_K_mm()
        K_mm_inv = self._K_mm_inv

        U_x = self.laplacian.eigenvectors[x]
        U_m = self.laplacian.eigenvectors[self.inducing_indices]
        weights = self._spectral_density(self.laplacian.eigenvalues)

        K_xm = (U_x * weights.unsqueeze(0)) @ U_m.T  # (n, m)

        # Diagonal of K_xm @ K_mm_inv @ K_mx
        # = sum_j (K_xm @ K_mm_inv)_ij * K_mx_ji
        # = sum_j (K_xm @ K_mm_inv)_ij * K_xm_ij
        L = K_xm @ K_mm_inv  # (n, m)
        return torch.sum(L * K_xm, dim=-1)
