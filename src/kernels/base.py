"""
Base class for graph kernels.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from scipy import sparse
import numpy as np


class BaseGraphKernel(ABC, nn.Module):
    """
    Abstract base class for graph kernels.

    Graph kernels define similarity between nodes based on graph structure.
    They serve as covariance functions for Gaussian Processes on graphs.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel matrix K(x1, x2).

        Args:
            x1: First set of node indices, shape (n1,)
            x2: Second set of node indices, shape (n2,). If None, compute K(x1, x1)

        Returns:
            Kernel matrix of shape (n1, n2)
        """
        pass

    @abstractmethod
    def set_graph(self, adjacency: sparse.csr_matrix):
        """
        Set the graph structure for the kernel.

        Args:
            adjacency: Sparse adjacency matrix
        """
        pass

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal of K(x, x).

        Default implementation computes full matrix and extracts diagonal.
        Override for efficiency if possible.

        Args:
            x: Node indices, shape (n,)

        Returns:
            Diagonal of shape (n,)
        """
        K = self.forward(x, x)
        return torch.diag(K)


class GraphLaplacian(nn.Module):
    """
    Utility class for computing graph Laplacian and its spectral decomposition.
    """

    def __init__(self, normalized: bool = True):
        super().__init__()
        self.normalized = normalized

        # Will be set when graph is provided
        self.register_buffer("eigenvalues", None)
        self.register_buffer("eigenvectors", None)
        self.register_buffer("L", None)
        self.num_nodes = 0

    def set_graph(
        self,
        adjacency: sparse.csr_matrix,
        num_eigenvectors: Optional[int] = None,
    ):
        """
        Set graph and compute spectral decomposition.

        Args:
            adjacency: Sparse adjacency matrix
            num_eigenvectors: Number of eigenvectors to compute (for large graphs)
        """
        self.num_nodes = adjacency.shape[0]

        # Compute Laplacian
        degree = np.array(adjacency.sum(axis=1)).flatten()

        if self.normalized:
            # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
            degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
            degree_inv_sqrt[degree == 0] = 0
            D_inv_sqrt = sparse.diags(degree_inv_sqrt)
            L = sparse.eye(self.num_nodes) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        else:
            # Unnormalized Laplacian: L = D - A
            D = sparse.diags(degree)
            L = D - adjacency

        L = L.tocsr()

        # Store dense Laplacian (for small graphs)
        if self.num_nodes <= 5000:
            self.L = torch.tensor(L.toarray(), dtype=torch.float32)

            # Full eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(self.L)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        else:
            # For large graphs, use sparse operations or approximate
            from scipy.sparse.linalg import eigsh

            k = num_eigenvectors or min(1000, self.num_nodes - 1)
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

            self.eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)
            self.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
            self.L = None  # Don't store full Laplacian for large graphs

    def apply_function(
        self,
        func,
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply a function to the Laplacian: f(L) = U f(Λ) U^T

        Args:
            func: Function to apply to eigenvalues
            node_indices: If provided, return only rows/columns for these nodes

        Returns:
            Result of f(L)
        """
        if self.eigenvalues is None:
            raise RuntimeError("Graph not set. Call set_graph first.")

        # Apply function to eigenvalues
        f_eigenvalues = func(self.eigenvalues)

        # Reconstruct: f(L) = U f(Λ) U^T
        if node_indices is None:
            # Full matrix
            result = self.eigenvectors @ torch.diag(f_eigenvalues) @ self.eigenvectors.T
        else:
            # Subset of nodes
            U_subset = self.eigenvectors[node_indices]
            result = U_subset @ torch.diag(f_eigenvalues) @ U_subset.T

        return result
