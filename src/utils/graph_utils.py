"""
Graph utility functions for Knowledge Graphs.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
import torch


def build_adjacency_matrix(
    triples: np.ndarray,
    num_entities: int,
    symmetric: bool = True,
) -> sparse.csr_matrix:
    """
    Build adjacency matrix from triples.

    Args:
        triples: Array of (head, relation, tail) triples
        num_entities: Number of entities
        symmetric: Whether to make the matrix symmetric

    Returns:
        Sparse adjacency matrix
    """
    heads = triples[:, 0]
    tails = triples[:, 2]
    data = np.ones(len(triples))

    adj = sparse.csr_matrix(
        (data, (heads, tails)),
        shape=(num_entities, num_entities)
    )

    if symmetric:
        adj = adj + adj.T
        adj.data = np.clip(adj.data, 0, 1)

    return adj


def build_relation_adjacencies(
    triples: np.ndarray,
    num_entities: int,
    num_relations: int,
) -> Dict[int, sparse.csr_matrix]:
    """
    Build per-relation adjacency matrices.

    Args:
        triples: Array of triples
        num_entities: Number of entities
        num_relations: Number of relations

    Returns:
        Dict mapping relation_id -> adjacency matrix
    """
    relation_adjs = {}

    for r in range(num_relations):
        mask = triples[:, 1] == r
        rel_triples = triples[mask]

        if len(rel_triples) > 0:
            heads = rel_triples[:, 0]
            tails = rel_triples[:, 2]
            data = np.ones(len(rel_triples))

            adj = sparse.csr_matrix(
                (data, (heads, tails)),
                shape=(num_entities, num_entities)
            )
            relation_adjs[r] = adj

    return relation_adjs


def compute_laplacian(
    adjacency: sparse.csr_matrix,
    normalized: bool = True,
) -> sparse.csr_matrix:
    """
    Compute graph Laplacian.

    Args:
        adjacency: Adjacency matrix
        normalized: Whether to use normalized Laplacian

    Returns:
        Laplacian matrix
    """
    n = adjacency.shape[0]
    degree = np.array(adjacency.sum(axis=1)).flatten()

    if normalized:
        # L = I - D^{-1/2} A D^{-1/2}
        degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        degree_inv_sqrt[degree == 0] = 0
        D_inv_sqrt = sparse.diags(degree_inv_sqrt)
        L = sparse.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # L = D - A
        D = sparse.diags(degree)
        L = D - adjacency

    return L.tocsr()


def compute_shortest_paths(
    adjacency: sparse.csr_matrix,
    max_distance: int = 5,
) -> np.ndarray:
    """
    Compute shortest path distances between all pairs.

    Uses BFS with early stopping at max_distance.

    Args:
        adjacency: Adjacency matrix
        max_distance: Maximum distance to compute

    Returns:
        Distance matrix (inf for disconnected pairs)
    """
    from scipy.sparse.csgraph import shortest_path

    distances = shortest_path(
        adjacency,
        method='BF',
        directed=False,
        unweighted=True,
        limit=max_distance
    )

    return distances


def select_inducing_points(
    adjacency: sparse.csr_matrix,
    num_inducing: int,
    method: str = "kmeans",
) -> np.ndarray:
    """
    Select inducing points for sparse GP.

    Args:
        adjacency: Adjacency matrix
        num_inducing: Number of inducing points
        method: Selection method
            - "random": Random selection
            - "degree": Select high-degree nodes
            - "kmeans": K-means on graph features

    Returns:
        Array of inducing point indices
    """
    n = adjacency.shape[0]

    if method == "random":
        return np.random.choice(n, num_inducing, replace=False)

    elif method == "degree":
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        return np.argsort(degrees)[-num_inducing:]

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        from scipy.sparse.linalg import eigsh

        # Use spectral embedding
        L = compute_laplacian(adjacency, normalized=True)
        k = min(50, n - 1)
        _, eigvecs = eigsh(L, k=k, which='SM')

        # K-means on eigenvectors
        kmeans = KMeans(n_clusters=num_inducing, random_state=42)
        kmeans.fit(eigvecs)

        # Select point closest to each centroid
        inducing = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(eigvecs - center, axis=1)
            inducing.append(np.argmin(distances))

        return np.array(inducing)

    else:
        raise ValueError(f"Unknown method: {method}")


def graph_statistics(
    adjacency: sparse.csr_matrix,
) -> Dict[str, float]:
    """
    Compute various graph statistics.

    Returns:
        Dict with num_nodes, num_edges, avg_degree, density, etc.
    """
    n = adjacency.shape[0]
    num_edges = adjacency.nnz // 2  # Assuming symmetric

    degrees = np.array(adjacency.sum(axis=1)).flatten()

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "avg_degree": degrees.mean(),
        "max_degree": degrees.max(),
        "min_degree": degrees.min(),
        "density": num_edges / (n * (n - 1) / 2),
        "degree_std": degrees.std(),
    }


def entity_neighborhood(
    entity_id: int,
    triples: np.ndarray,
    hops: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get neighborhood of an entity.

    Args:
        entity_id: Entity to get neighborhood for
        triples: All triples
        hops: Number of hops to expand

    Returns:
        Tuple of (neighbor_entities, relevant_triples)
    """
    current_entities = {entity_id}
    all_entities = {entity_id}
    relevant_triples = []

    for _ in range(hops):
        new_entities = set()
        for e in current_entities:
            # Find triples involving e
            mask = (triples[:, 0] == e) | (triples[:, 2] == e)
            e_triples = triples[mask]

            for t in e_triples:
                relevant_triples.append(t)
                new_entities.add(t[0])
                new_entities.add(t[2])

        current_entities = new_entities - all_entities
        all_entities.update(new_entities)

    return np.array(list(all_entities)), np.array(relevant_triples)
