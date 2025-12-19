"""
Knowledge Graph Dataset class for handling KG data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx
from scipy import sparse


@dataclass
class KGTriple:
    """A single (head, relation, tail) triple."""
    head: int
    relation: int
    tail: int
    confidence: float = 1.0  # For uncertain KGs


class KGDataset(Dataset):
    """
    Knowledge Graph Dataset.

    Handles loading, processing, and serving of KG triples.
    Supports both standard and uncertain KGs (with confidence scores).
    """

    def __init__(
        self,
        triples: np.ndarray,
        num_entities: int,
        num_relations: int,
        confidence_scores: Optional[np.ndarray] = None,
        entity_to_id: Optional[Dict[str, int]] = None,
        relation_to_id: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            triples: Array of shape (N, 3) with (head, relation, tail) indices
            num_entities: Total number of entities
            num_relations: Total number of relations
            confidence_scores: Optional array of confidence scores per triple
            entity_to_id: Mapping from entity names to IDs
            relation_to_id: Mapping from relation names to IDs
        """
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.confidence_scores = confidence_scores
        self.entity_to_id = entity_to_id or {}
        self.relation_to_id = relation_to_id or {}

        # Inverse mappings
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

        # Build graph structure
        self._build_graph()

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        triple = self.triples[idx]
        item = {
            "head": torch.tensor(triple[0], dtype=torch.long),
            "relation": torch.tensor(triple[1], dtype=torch.long),
            "tail": torch.tensor(triple[2], dtype=torch.long),
        }
        if self.confidence_scores is not None:
            item["confidence"] = torch.tensor(
                self.confidence_scores[idx], dtype=torch.float
            )
        return item

    def _build_graph(self):
        """Build NetworkX graph and adjacency structures."""
        # Multi-relational graph
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(range(self.num_entities))

        for triple in self.triples:
            h, r, t = triple
            self.graph.add_edge(h, t, relation=r)

        # Per-relation adjacency matrices (sparse)
        self.relation_adjacencies = {}
        for r in range(self.num_relations):
            mask = self.triples[:, 1] == r
            edges = self.triples[mask][:, [0, 2]]

            if len(edges) > 0:
                rows, cols = edges[:, 0], edges[:, 1]
                data = np.ones(len(rows))
                adj = sparse.csr_matrix(
                    (data, (rows, cols)),
                    shape=(self.num_entities, self.num_entities)
                )
                self.relation_adjacencies[r] = adj

    def get_adjacency_matrix(
        self,
        relation: Optional[int] = None,
        symmetric: bool = True
    ) -> sparse.csr_matrix:
        """
        Get adjacency matrix.

        Args:
            relation: If specified, get adjacency for specific relation only
            symmetric: Whether to symmetrize the adjacency

        Returns:
            Sparse adjacency matrix
        """
        if relation is not None:
            adj = self.relation_adjacencies.get(
                relation,
                sparse.csr_matrix((self.num_entities, self.num_entities))
            )
        else:
            # Aggregate all relations
            adj = sparse.csr_matrix((self.num_entities, self.num_entities))
            for r_adj in self.relation_adjacencies.values():
                adj = adj + r_adj

        if symmetric:
            adj = adj + adj.T
            adj.data = np.clip(adj.data, 0, 1)  # Binary

        return adj

    def get_laplacian(
        self,
        relation: Optional[int] = None,
        normalized: bool = True
    ) -> sparse.csr_matrix:
        """
        Compute graph Laplacian.

        Args:
            relation: If specified, compute for specific relation only
            normalized: Whether to use normalized Laplacian

        Returns:
            Sparse Laplacian matrix
        """
        adj = self.get_adjacency_matrix(relation, symmetric=True)
        degree = np.array(adj.sum(axis=1)).flatten()

        if normalized:
            # L = I - D^{-1/2} A D^{-1/2}
            degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
            degree_inv_sqrt[degree == 0] = 0
            D_inv_sqrt = sparse.diags(degree_inv_sqrt)
            L = sparse.eye(self.num_entities) - D_inv_sqrt @ adj @ D_inv_sqrt
        else:
            # L = D - A
            D = sparse.diags(degree)
            L = D - adj

        return L.tocsr()

    def get_entity_degrees(self) -> np.ndarray:
        """Get degree of each entity (total connections)."""
        adj = self.get_adjacency_matrix(symmetric=True)
        return np.array(adj.sum(axis=1)).flatten()

    def get_entity_relation_degrees(self) -> Dict[int, np.ndarray]:
        """Get per-relation degree of each entity."""
        degrees = {}
        for r in range(self.num_relations):
            adj = self.get_adjacency_matrix(relation=r, symmetric=True)
            degrees[r] = np.array(adj.sum(axis=1)).flatten()
        return degrees

    def sample_negative(
        self,
        triple: np.ndarray,
        num_negatives: int = 1,
        mode: str = "tail"
    ) -> np.ndarray:
        """
        Sample negative triples by corrupting head or tail.

        Args:
            triple: Original triple (h, r, t)
            num_negatives: Number of negative samples
            mode: "head", "tail", or "both"

        Returns:
            Array of negative triples
        """
        h, r, t = triple
        negatives = []

        for _ in range(num_negatives):
            if mode == "tail" or (mode == "both" and np.random.random() < 0.5):
                # Corrupt tail
                neg_t = np.random.randint(self.num_entities)
                while neg_t == t:  # Avoid same as original
                    neg_t = np.random.randint(self.num_entities)
                negatives.append([h, r, neg_t])
            else:
                # Corrupt head
                neg_h = np.random.randint(self.num_entities)
                while neg_h == h:
                    neg_h = np.random.randint(self.num_entities)
                negatives.append([neg_h, r, t])

        return np.array(negatives)

    def get_train_triples_for_entity(self, entity_id: int) -> np.ndarray:
        """Get all triples involving a specific entity."""
        mask = (self.triples[:, 0] == entity_id) | (self.triples[:, 2] == entity_id)
        return self.triples[mask]

    @property
    def edge_index(self) -> torch.Tensor:
        """Get edge index in PyTorch Geometric format."""
        edges = self.triples[:, [0, 2]]  # head, tail
        return torch.tensor(edges.T, dtype=torch.long)

    @property
    def edge_type(self) -> torch.Tensor:
        """Get edge types (relations) for PyTorch Geometric."""
        return torch.tensor(self.triples[:, 1], dtype=torch.long)


def train_val_test_split(
    dataset: KGDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[KGDataset, KGDataset, KGDataset]:
    """
    Split dataset into train/val/test.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)

    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    def create_split(indices):
        triples = dataset.triples[indices]
        confidence = None
        if dataset.confidence_scores is not None:
            confidence = dataset.confidence_scores[indices]
        return KGDataset(
            triples=triples,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            confidence_scores=confidence,
            entity_to_id=dataset.entity_to_id,
            relation_to_id=dataset.relation_to_id,
        )

    return (
        create_split(train_indices),
        create_split(val_indices),
        create_split(test_indices),
    )
