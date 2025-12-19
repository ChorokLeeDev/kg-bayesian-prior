"""
Diffusion Kernel on Graphs

Reference: Kondor & Lafferty (2002) - Diffusion Kernels on Graphs

The diffusion kernel is defined as:
    K = exp(-β L)

where L is the graph Laplacian and β > 0 is a diffusion parameter.

Interpretation: K_ij represents the probability of a random walk
from node i reaching node j after time β.
"""

from typing import Optional
import torch
import torch.nn as nn
from scipy import sparse
import numpy as np

from .base import BaseGraphKernel, GraphLaplacian


class DiffusionKernel(BaseGraphKernel):
    """
    Diffusion kernel on graphs.

    K = exp(-β L)

    Properties:
    - Always positive semi-definite
    - Nodes connected by short paths have higher similarity
    - β controls the "spread" of the diffusion
      - Small β: only immediate neighbors matter
      - Large β: distant nodes also connected
    """

    def __init__(
        self,
        beta: float = 1.0,
        learnable: bool = True,
        normalized_laplacian: bool = True,
    ):
        """
        Args:
            beta: Diffusion time parameter
            learnable: Whether beta is learnable
            normalized_laplacian: Whether to use normalized Laplacian
        """
        super().__init__()

        if learnable:
            # Use log parameterization for positivity
            self.log_beta = nn.Parameter(torch.tensor(np.log(beta)))
        else:
            self.register_buffer("log_beta", torch.tensor(np.log(beta)))

        self.laplacian = GraphLaplacian(normalized=normalized_laplacian)
        self._kernel_matrix = None

    @property
    def beta(self) -> torch.Tensor:
        return torch.exp(self.log_beta)

    def set_graph(self, adjacency: sparse.csr_matrix, **kwargs):
        """Set the graph structure."""
        self.laplacian.set_graph(adjacency, **kwargs)
        self._kernel_matrix = None  # Reset cached kernel

    def _compute_kernel_matrix(self) -> torch.Tensor:
        """Compute the full kernel matrix K = exp(-β L)."""
        if self._kernel_matrix is not None:
            return self._kernel_matrix

        # Apply exp(-β λ) to eigenvalues
        def diffusion_func(eigenvalues):
            return torch.exp(-self.beta * eigenvalues)

        K = self.laplacian.apply_function(diffusion_func)
        self._kernel_matrix = K
        return K

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel values K(x1, x2).

        Args:
            x1: Node indices, shape (n1,)
            x2: Node indices, shape (n2,). If None, compute K(x1, x1)

        Returns:
            Kernel matrix of shape (n1, n2)
        """
        K = self._compute_kernel_matrix()

        if x2 is None:
            x2 = x1

        # Extract submatrix
        return K[x1.unsqueeze(1), x2.unsqueeze(0)]

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal of K(x, x)."""
        K = self._compute_kernel_matrix()
        return K[x, x]


class LazyDiffusionKernel(BaseGraphKernel):
    """
    Memory-efficient diffusion kernel for large graphs.

    Instead of storing the full kernel matrix, computes entries on-demand
    using the spectral decomposition.
    """

    def __init__(
        self,
        beta: float = 1.0,
        learnable: bool = True,
        num_eigenvectors: int = 500,
    ):
        """
        Args:
            beta: Diffusion parameter
            learnable: Whether beta is learnable
            num_eigenvectors: Number of eigenvectors to use (approximation)
        """
        super().__init__()

        if learnable:
            self.log_beta = nn.Parameter(torch.tensor(np.log(beta)))
        else:
            self.register_buffer("log_beta", torch.tensor(np.log(beta)))

        self.num_eigenvectors = num_eigenvectors
        self.laplacian = GraphLaplacian(normalized=True)

    @property
    def beta(self) -> torch.Tensor:
        return torch.exp(self.log_beta)

    def set_graph(self, adjacency: sparse.csr_matrix):
        """Set graph with limited eigendecomposition."""
        self.laplacian.set_graph(adjacency, num_eigenvectors=self.num_eigenvectors)

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute kernel values using low-rank approximation."""
        if x2 is None:
            x2 = x1

        # Get eigenvector rows for the nodes
        U1 = self.laplacian.eigenvectors[x1]  # (n1, k)
        U2 = self.laplacian.eigenvectors[x2]  # (n2, k)

        # Compute exp(-β λ)
        weights = torch.exp(-self.beta * self.laplacian.eigenvalues)  # (k,)

        # K = U1 @ diag(weights) @ U2^T
        K = (U1 * weights.unsqueeze(0)) @ U2.T

        return K

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient diagonal computation."""
        U = self.laplacian.eigenvectors[x]  # (n, k)
        weights = torch.exp(-self.beta * self.laplacian.eigenvalues)
        return torch.sum(U ** 2 * weights.unsqueeze(0), dim=-1)
