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

    K(i, j) = Σ_r  σ_r² · K_r(i, j | ℓ_r)

    where K_r is computed from the relation-r subgraph.

    This kernel captures the heterogeneous structure of KGs by learning
    different smoothness assumptions for different relation types.
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
        """
        super().__init__()

        self.num_relations = num_relations
        self.kernel_type = kernel_type
        self.nu = nu
        self.aggregation = aggregation
        self.share_lengthscale = share_lengthscale

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

        # For attention-based aggregation
        if aggregation == "attention":
            self.attention_weights = nn.Parameter(torch.zeros(num_relations))

        # Per-relation Laplacians (set when graph is provided)
        self.relation_laplacians: Dict[int, GraphLaplacian] = {}
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

        # Filter relations with enough edges
        valid_relations = {
            r: adj for r, adj in relation_adjacencies.items()
            if adj.nnz >= min_edges
        }

        if show_progress:
            print(f"Computing spectral decomposition for {len(valid_relations)}/{len(relation_adjacencies)} relations...")
            print(f"  (skipping {len(relation_adjacencies) - len(valid_relations)} sparse relations with <{min_edges} edges)")
            iterator = tqdm(valid_relations.items(), desc="Eigendecomp", unit="rel")
        else:
            iterator = valid_relations.items()

        for r, adj in iterator:
            # Make symmetric (undirected)
            adj_sym = adj + adj.T
            adj_sym.data = np.clip(adj_sym.data, 0, 1)

            laplacian = GraphLaplacian(normalized=True)
            laplacian.set_graph(adj_sym, num_eigenvectors=num_eigenvectors)
            self.relation_laplacians[r] = laplacian

        if show_progress:
            print(f"Spectral decomposition complete: {len(self.relation_laplacians)} relations processed")

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

    def _compute_full_kernel(self) -> torch.Tensor:
        """Compute the full aggregated kernel matrix."""
        device = self.log_variance.device

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

        # Aggregate
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

        return K

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute kernel values."""
        K = self._compute_full_kernel()

        if x2 is None:
            x2 = x1

        return K[x1.unsqueeze(1), x2.unsqueeze(0)]

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal of kernel."""
        K = self._compute_full_kernel()
        return K[x, x]

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
