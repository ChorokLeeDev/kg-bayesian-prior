"""
Relation-Aware Gaussian Process Kernel for Knowledge Graphs

This is the CORE CONTRIBUTION of the thesis.

Key Idea: Different relation types induce different notions of similarity.
- "supplier" relationship: entities share business characteristics → strong smoothness
- "competitor" relationship: entities might be anti-correlated → different kernel
- "located_in" relationship: only geographic features shared → weak smoothness

Mathematical Formulation:
========================

For a Knowledge Graph G = (E, R, T) with:
- E: entities
- R: relation types
- T: set of triples (h, r, t)

We define a relation-aware kernel:

    K(i, j) = Σ_r  w_r · K_r(i, j)

where:
- K_r is a relation-specific kernel
- w_r are learnable mixing weights

Each K_r is defined via its own Laplacian L_r (from relation-r subgraph):

    K_r = σ_r² · exp(-L_r / ℓ_r²)

Parameters per relation:
- ℓ_r: lengthscale (how far information propagates along relation r)
- σ_r²: variance (importance of relation r)

This allows the model to learn that:
- "part_of" relations have short lengthscale (local structure)
- "similar_to" relations have long lengthscale (global structure)

Entity-Level Uncertainty:
========================

The posterior covariance Σ directly gives entity-level uncertainty:

    p(f | data) = N(μ, Σ)

For entity i:
- μ_i: mean representation
- Σ_ii: uncertainty (variance) of entity i

Uncertainty is HIGH when:
- Entity has few connections
- Connections are to uncertain entities
- Relation types have low variance weight
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import numpy as np

from .base import BaseGraphKernel, GraphLaplacian


class RelationAwareKernel(BaseGraphKernel):
    """
    Relation-aware kernel for Knowledge Graphs.

    K(i, j) = K_global(i, j) + Σ_r  σ_r² · K_r(i, j | ℓ_r)

    where K_r is computed from the relation-r subgraph, and K_global uses
    all edges regardless of relation type (fallback for sparse-relation KGs).

    This kernel captures the heterogeneous structure of KGs by learning
    different smoothness assumptions for different relation types.

    For relation-sparse KGs (like WN18RR with 11 relations), the global kernel
    provides meaningful structure even when per-relation eigendecomposition fails.
    """

    def __init__(
        self,
        num_relations: int,
        kernel_type: str = "diffusion",  # "diffusion" or "matern"
        nu: float = 2.5,  # For Matérn
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
        share_lengthscale: bool = False,
        aggregation: str = "sum",  # "sum", "attention", or "product"
        use_global_kernel: bool = True,  # NEW: whether to add global kernel
    ):
        """
        Args:
            num_relations: Number of relation types in the KG
            kernel_type: Type of base kernel ("diffusion" or "matern")
            nu: Smoothness parameter for Matérn kernel
            init_lengthscale: Initial lengthscale for all relations
            init_variance: Initial variance for all relations
            share_lengthscale: If True, share one lengthscale across relations
            aggregation: How to combine relation kernels
            use_global_kernel: If True, add a global kernel using all edges (helps sparse KGs)
        """
        super().__init__()

        self.num_relations = num_relations
        self.kernel_type = kernel_type
        self.nu = nu
        self.aggregation = aggregation
        self.share_lengthscale = share_lengthscale
        self.use_global_kernel = use_global_kernel

        # Per-relation parameters
        if share_lengthscale:
            self.log_lengthscale = nn.Parameter(
                torch.tensor(np.log(init_lengthscale))
            )
        else:
            self.log_lengthscale = nn.Parameter(
                torch.full((num_relations,), np.log(init_lengthscale))
            )

        self.log_variance = nn.Parameter(
            torch.full((num_relations,), np.log(init_variance))
        )

        # Global kernel parameters (for all edges combined)
        if use_global_kernel:
            self.log_global_lengthscale = nn.Parameter(
                torch.tensor(np.log(init_lengthscale))
            )
            self.log_global_variance = nn.Parameter(
                torch.tensor(np.log(init_variance))
            )

        # For attention-based aggregation
        if aggregation == "attention":
            self.attention_weights = nn.Parameter(torch.zeros(num_relations))

        # Per-relation Laplacians (set when graph is provided)
        self.relation_laplacians: Dict[int, GraphLaplacian] = {}
        # Global Laplacian (all edges combined)
        self.global_laplacian: Optional[GraphLaplacian] = None
        self.num_entities = 0

        # Cached kernel matrices
        self._relation_kernels: Dict[int, torch.Tensor] = {}

    @property
    def lengthscale(self) -> torch.Tensor:
        """Get lengthscales for all relations."""
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        """Get variances for all relations."""
        return torch.exp(self.log_variance)

    def set_graph(
        self,
        relation_adjacencies: Dict[int, sparse.csr_matrix],
        num_entities: int,
        num_eigenvectors: int = 100,
        min_edges: int = 10,
        show_progress: bool = True,
    ):
        """
        Set the Knowledge Graph structure.

        Args:
            relation_adjacencies: Dict mapping relation_id -> adjacency matrix
            num_entities: Total number of entities
            num_eigenvectors: Number of eigenvectors for spectral decomposition (default: 100)
            min_edges: Skip relations with fewer edges than this (default: 10)
            show_progress: Show progress bar during eigendecomposition
        """
        from tqdm import tqdm

        self.num_entities = num_entities
        self.relation_laplacians = {}
        self._relation_kernels = {}
        self.global_laplacian = None

        # Step 1: Build global adjacency (all edges combined)
        if self.use_global_kernel:
            if show_progress:
                print("Building global adjacency matrix (all relations combined)...")

            # Combine all relation adjacencies
            global_adj = sparse.csr_matrix((num_entities, num_entities))
            for adj in relation_adjacencies.values():
                global_adj = global_adj + adj

            # Make symmetric and binary
            global_adj = global_adj + global_adj.T
            global_adj.data = np.clip(global_adj.data, 0, 1)

            if show_progress:
                print(f"  Global graph: {global_adj.nnz} edges")

            # Compute global Laplacian
            self.global_laplacian = GraphLaplacian(normalized=True)
            self.global_laplacian.set_graph(global_adj, num_eigenvectors=num_eigenvectors)

            if self.global_laplacian.eigenvectors is not None:
                if show_progress:
                    print(f"  Global eigendecomp: SUCCESS ({self.global_laplacian.eigenvectors.shape[1]} eigenvectors)")
            else:
                if show_progress:
                    print("  Global eigendecomp: FAILED")
                self.global_laplacian = None

        # Step 2: Build per-relation Laplacians
        # Filter relations with enough edges
        valid_relations = {
            r: adj for r, adj in relation_adjacencies.items()
            if adj.nnz >= min_edges
        }

        if show_progress:
            print(f"Computing per-relation spectral decomposition for {len(valid_relations)}/{len(relation_adjacencies)} relations...")
            print(f"  (skipping {len(relation_adjacencies) - len(valid_relations)} sparse relations with <{min_edges} edges)")
            iterator = tqdm(valid_relations.items(), desc="Eigendecomp", unit="rel")
        else:
            iterator = valid_relations.items()

        success_count = 0
        for r, adj in iterator:
            # Make symmetric (undirected)
            adj_sym = adj + adj.T
            adj_sym.data = np.clip(adj_sym.data, 0, 1)

            laplacian = GraphLaplacian(normalized=True)
            laplacian.set_graph(adj_sym, num_eigenvectors=num_eigenvectors)

            # Only add if eigendecomp succeeded
            if laplacian.eigenvectors is not None:
                self.relation_laplacians[r] = laplacian
                success_count += 1

        if show_progress:
            print(f"Per-relation spectral decomposition: {success_count}/{len(valid_relations)} relations succeeded")
            if self.use_global_kernel and self.global_laplacian is not None:
                print(f"Global kernel: ENABLED (fallback for {len(relation_adjacencies) - success_count} failed relations)")

    def _compute_relation_kernel(self, r: int) -> torch.Tensor:
        """Compute kernel matrix for a single relation."""
        if r not in self.relation_laplacians:
            # Return identity for missing relations
            device = self.log_variance.device
            return torch.eye(self.num_entities, device=device)

        laplacian = self.relation_laplacians[r]

        # Get relation-specific parameters
        if self.share_lengthscale:
            ell = self.lengthscale
        else:
            ell = self.lengthscale[r]
        sigma_sq = self.variance[r]

        # Get device from parameters
        device = sigma_sq.device

        if self.kernel_type == "diffusion":
            # K_r = σ_r² · exp(-L_r / ℓ_r²)
            def kernel_func(eigenvalues):
                eigenvalues = eigenvalues.to(device)
                return sigma_sq * torch.exp(-eigenvalues / (ell ** 2))
        else:
            # Matérn: K_r = σ_r² · (2ν/ℓ_r² + L_r)^{-ν}
            def kernel_func(eigenvalues):
                eigenvalues = eigenvalues.to(device)
                S = torch.pow(2 * self.nu / (ell ** 2) + eigenvalues, -self.nu)
                S = S / S[0]  # Normalize
                return sigma_sq * S

        K_r = laplacian.apply_function(kernel_func)
        return K_r

    def _compute_global_kernel_subset(self, x1: torch.Tensor, x2: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute global kernel for a SUBSET of nodes (memory efficient).

        Instead of computing full N×N kernel, only compute for requested indices.
        K[x1, x2] = U[x1] @ diag(f(λ)) @ U[x2].T

        Args:
            x1: First set of node indices
            x2: Second set of node indices

        Returns:
            Kernel matrix of shape (len(x1), len(x2))
        """
        if not self.use_global_kernel or self.global_laplacian is None:
            return None

        if self.global_laplacian.eigenvectors is None:
            return None

        device = self.log_global_variance.device
        ell = torch.exp(self.log_global_lengthscale)
        sigma_sq = torch.exp(self.log_global_variance)

        # Get eigenvectors for the subset of nodes
        U1 = self.global_laplacian.eigenvectors[x1.cpu()].to(device)  # (n1, k)
        U2 = self.global_laplacian.eigenvectors[x2.cpu()].to(device)  # (n2, k)
        eigenvalues = self.global_laplacian.eigenvalues.to(device)

        if self.kernel_type == "diffusion":
            f_lambda = sigma_sq * torch.exp(-eigenvalues / (ell ** 2))
        else:
            # Matérn
            f_lambda = torch.pow(2 * self.nu / (ell ** 2) + eigenvalues, -self.nu)
            f_lambda = f_lambda / f_lambda[0]
            f_lambda = sigma_sq * f_lambda

        # K[x1, x2] = U1 @ diag(f_λ) @ U2.T
        K_global = U1 @ torch.diag(f_lambda) @ U2.T
        return K_global

    def _compute_relation_kernel_subset(self, r: int, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute relation kernel for a SUBSET of nodes (memory efficient)."""
        if r not in self.relation_laplacians:
            device = self.log_variance.device
            return torch.zeros(len(x1), len(x2), device=device)

        laplacian = self.relation_laplacians[r]

        if self.share_lengthscale:
            ell = self.lengthscale
        else:
            ell = self.lengthscale[r]
        sigma_sq = self.variance[r]

        device = sigma_sq.device

        # Get eigenvectors for the subset
        U1 = laplacian.eigenvectors[x1.cpu()].to(device)
        U2 = laplacian.eigenvectors[x2.cpu()].to(device)
        eigenvalues = laplacian.eigenvalues.to(device)

        if self.kernel_type == "diffusion":
            f_lambda = sigma_sq * torch.exp(-eigenvalues / (ell ** 2))
        else:
            f_lambda = torch.pow(2 * self.nu / (ell ** 2) + eigenvalues, -self.nu)
            f_lambda = f_lambda / f_lambda[0]
            f_lambda = sigma_sq * f_lambda

        K_r = U1 @ torch.diag(f_lambda) @ U2.T
        return K_r

    def _compute_full_kernel(self) -> torch.Tensor:
        """Compute the full aggregated kernel matrix."""
        device = self.log_variance.device

        # Start with global kernel if available
        K_global = self._compute_global_kernel()

        # Compute all relation kernels
        relation_kernels = []
        for r in range(self.num_relations):
            if r in self.relation_laplacians:
                K_r = self._compute_relation_kernel(r)
            else:
                K_r = torch.zeros(self.num_entities, self.num_entities, device=device)
            relation_kernels.append(K_r)

        # Stack: (num_relations, num_entities, num_entities)
        K_stack = torch.stack(relation_kernels, dim=0)

        # Aggregate per-relation kernels
        if self.aggregation == "sum":
            # Simple sum
            K = K_stack.sum(dim=0)

        elif self.aggregation == "attention":
            # Attention-weighted sum
            weights = F.softmax(self.attention_weights, dim=0)
            K = torch.einsum("r,rij->ij", weights, K_stack)

        elif self.aggregation == "product":
            # Product (log-sum)
            # K = exp(Σ_r log(K_r)) - careful with zeros
            K_stack = K_stack.clamp(min=1e-10)
            K = torch.exp(torch.log(K_stack).sum(dim=0))

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Add global kernel contribution
        if K_global is not None:
            K = K + K_global

        return K

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel values for specified indices.

        Uses efficient subset computation to avoid building full N×N kernel.
        Memory: O(n1 * n2 * k) instead of O(N²)
        """
        if x2 is None:
            x2 = x1

        device = self.log_variance.device

        # Use efficient subset computation
        # Start with global kernel
        K_global = self._compute_global_kernel_subset(x1, x2)

        # Add per-relation kernels
        K = torch.zeros(len(x1), len(x2), device=device)
        for r in range(self.num_relations):
            if r in self.relation_laplacians:
                K_r = self._compute_relation_kernel_subset(r, x1, x2)
                K = K + K_r

        # Add global kernel
        if K_global is not None:
            K = K + K_global

        return K

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal of kernel (efficient)."""
        # For diagonal, use forward with same indices
        K = self.forward(x, x)
        return torch.diag(K)

    def get_relation_importance(self) -> torch.Tensor:
        """Get learned importance of each relation type."""
        if self.aggregation == "attention":
            return F.softmax(self.attention_weights, dim=0)
        else:
            return self.variance / self.variance.sum()

    def get_relation_lengthscales(self) -> torch.Tensor:
        """Get learned lengthscales."""
        return self.lengthscale


class HierarchicalRelationKernel(BaseGraphKernel):
    """
    Hierarchical relation-aware kernel with relation embeddings.

    Instead of independent parameters per relation, learns relation embeddings
    and derives kernel parameters from them. This enables:
    - Parameter sharing between similar relations
    - Better generalization to rare relations
    - Interpretable relation space

    K(i, j) = Σ_r  f_σ(e_r) · K(i, j | f_ℓ(e_r))

    where:
    - e_r is the learned embedding of relation r
    - f_σ, f_ℓ are networks that map embeddings to kernel parameters
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim: int = 32,
        kernel_type: str = "diffusion",
        nu: float = 2.5,
    ):
        """
        Args:
            num_relations: Number of relation types
            relation_dim: Dimension of relation embeddings
            kernel_type: Base kernel type
            nu: Matérn smoothness
        """
        super().__init__()

        self.num_relations = num_relations
        self.relation_dim = relation_dim
        self.kernel_type = kernel_type
        self.nu = nu

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)

        # Networks to predict kernel parameters from embeddings
        self.lengthscale_net = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Softplus(),  # Ensure positive
        )

        self.variance_net = nn.Sequential(
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Softplus(),
        )

        # Per-relation Laplacians
        self.relation_laplacians: Dict[int, GraphLaplacian] = {}
        self.num_entities = 0

    def get_relation_params(self, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get kernel parameters for a relation."""
        e_r = self.relation_embeddings(torch.tensor(r))
        ell = self.lengthscale_net(e_r).squeeze() + 0.1  # Min lengthscale
        sigma_sq = self.variance_net(e_r).squeeze() + 0.01
        return ell, sigma_sq

    def set_graph(
        self,
        relation_adjacencies: Dict[int, sparse.csr_matrix],
        num_entities: int,
    ):
        """Set graph structure."""
        self.num_entities = num_entities
        self.relation_laplacians = {}

        for r, adj in relation_adjacencies.items():
            adj_sym = adj + adj.T
            adj_sym.data = np.clip(adj_sym.data, 0, 1)

            laplacian = GraphLaplacian(normalized=True)
            laplacian.set_graph(adj_sym)
            self.relation_laplacians[r] = laplacian

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute kernel using hierarchical parameters."""
        if x2 is None:
            x2 = x1

        K_total = torch.zeros(len(x1), len(x2))

        for r in range(self.num_relations):
            if r not in self.relation_laplacians:
                continue

            ell, sigma_sq = self.get_relation_params(r)
            laplacian = self.relation_laplacians[r]

            if self.kernel_type == "diffusion":
                def kernel_func(eigenvalues):
                    return sigma_sq * torch.exp(-eigenvalues / (ell ** 2))
            else:
                def kernel_func(eigenvalues):
                    S = torch.pow(2 * self.nu / (ell ** 2) + eigenvalues, -self.nu)
                    S = S / S[0]
                    return sigma_sq * S

            K_r = laplacian.apply_function(kernel_func)
            K_total += K_r[x1.unsqueeze(1), x2.unsqueeze(0)]

        return K_total

    def get_relation_similarity(self) -> torch.Tensor:
        """Compute similarity matrix between relation types."""
        embeddings = self.relation_embeddings.weight  # (R, D)
        # Cosine similarity
        norm_emb = F.normalize(embeddings, p=2, dim=-1)
        return norm_emb @ norm_emb.T


class AdaptiveRelationKernel(BaseGraphKernel):
    """
    Relation kernel that adapts based on entity context.

    The kernel parameters can vary based on the entities involved,
    not just the relation type. This captures that the same relation
    might have different semantics for different entity types.

    For example: "located_in" for (Company, City) vs (Person, Country)
    might have different smoothness properties.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int = 64,
        relation_dim: int = 32,
        kernel_type: str = "diffusion",
    ):
        super().__init__()

        self.num_entities_param = num_entities
        self.num_relations = num_relations
        self.kernel_type = kernel_type

        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)

        # Adaptive parameter network
        input_dim = 2 * entity_dim + relation_dim
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # lengthscale and variance
            nn.Softplus(),
        )

        # Base relation kernel for structure
        self.base_kernel = RelationAwareKernel(
            num_relations=num_relations,
            kernel_type=kernel_type,
        )

    def set_graph(self, relation_adjacencies, num_entities):
        """Set graph structure."""
        self.base_kernel.set_graph(relation_adjacencies, num_entities)
        self.num_entities = num_entities

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adaptive kernel.

        For full generality, this would compute pairwise adaptive parameters.
        For efficiency, we use the base kernel and modulate by entity features.
        """
        # Get base structure from relation kernel
        K_base = self.base_kernel(x1, x2)

        # Modulate by entity embeddings
        e1 = self.entity_embeddings(x1)  # (n1, d)
        if x2 is None:
            e2 = e1
        else:
            e2 = self.entity_embeddings(x2)  # (n2, d)

        # Compute entity-pair similarity as modulation
        modulation = torch.sigmoid(torch.mm(e1, e2.T))

        return K_base * modulation
