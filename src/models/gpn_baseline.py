"""
Graph Posterior Network (GPN) Baseline for KG OOD Detection

Based on: Stadler et al. (2021) - Graph Posterior Network
Adapted for knowledge graph triple classification and OOD detection.

GPN propagates uncertainty through graph structure but remains relation-agnostic
in entity embeddings, making it a good test of whether explicit coverage
decomposition is necessary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GPNForKG(nn.Module):
    """
    Graph Posterior Network adapted for Knowledge Graph OOD detection.

    Architecture:
    - Node embeddings: GNN layers propagate information
    - Evidential uncertainty: Dirichlet distribution over triple labels
    - Uncertainty quantification: Based on evidence mass, not coverage

    This tests whether graph-aware uncertainty (GPN) can match
    coverage-based uncertainty (CAGP) on temporal OOD.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 100,
        num_gnn_layers: int = 2,
        gnn_type: str = 'gcn',  # 'gcn' or 'gat'
        uncertainty_budget: float = 1.0,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.uncertainty_budget = uncertainty_budget

        # Entity embeddings (will be refined by GNN)
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # GNN layers for propagating entity information
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(dim, dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(dim, dim, heads=4, concat=False))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Evidence predictor: outputs (alpha, beta) for Dirichlet
        # Higher evidence = lower uncertainty
        self.evidence_net = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),  # [positive_evidence, negative_evidence]
            nn.Softplus()  # Ensure positive evidence
        )

        # For coverage tracking (not used in uncertainty, only for comparison)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def propagate_gnn(self, edge_index):
        """
        Propagate entity embeddings through GNN.

        Args:
            edge_index: [2, num_edges] - graph connectivity

        Returns:
            Refined entity embeddings after GNN propagation
        """
        x = self.entity_emb.weight

        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        return x

    def forward(self, heads, relations, tails, entity_emb=None):
        """
        Compute triple scores using DistMult.

        Args:
            heads, relations, tails: Triple indices
            entity_emb: Optional pre-computed entity embeddings from GNN
        """
        if entity_emb is None:
            h = self.entity_emb(heads)
            t = self.entity_emb(tails)
        else:
            h = entity_emb[heads]
            t = entity_emb[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def get_evidence(self, heads, relations, tails, entity_emb=None):
        """
        Compute evidential parameters for uncertainty quantification.

        Returns:
            alpha: Positive evidence (higher = more confident in positive)
            beta: Negative evidence (higher = more confident in negative)
        """
        if entity_emb is None:
            h = self.entity_emb(heads)
            t = self.entity_emb(tails)
        else:
            h = entity_emb[heads]
            t = entity_emb[tails]

        r = self.relation_emb(relations)

        # Concatenate triple representation
        triple_repr = torch.cat([h, r, t], dim=-1)

        # Predict evidence (Dirichlet concentration parameters)
        evidence = self.evidence_net(triple_repr)  # [batch, 2]
        alpha = evidence[:, 0] + 1  # Positive evidence
        beta = evidence[:, 1] + 1   # Negative evidence

        return alpha, beta

    def get_uncertainty(self, heads, relations, tails, entity_emb=None):
        """
        Compute epistemic uncertainty based on evidence.

        GPN uncertainty: Lower total evidence = higher uncertainty
        This captures "how much evidence the model has seen" for this pattern,
        but does NOT explicitly track entity-relation co-occurrence like CAGP.

        Formula: uncertainty = K / (alpha + beta)
        where K is uncertainty budget
        """
        alpha, beta = self.get_evidence(heads, relations, tails, entity_emb)

        # Total evidence
        total_evidence = alpha + beta

        # Epistemic uncertainty: inverse of total evidence
        uncertainty = self.uncertainty_budget / total_evidence

        return uncertainty

    def precompute_coverage(self, triples):
        """Track coverage for comparison (not used in GPN uncertainty)."""
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0

    def evidential_loss(self, heads, relations, tails, labels, entity_emb=None):
        """
        Evidential loss for training GPN.

        Encourages high evidence on true labels, low on false.
        """
        alpha, beta = self.get_evidence(heads, relations, tails, entity_emb)

        # Expected probability under Dirichlet
        probs = alpha / (alpha + beta)

        # Cross-entropy loss
        ce_loss = F.binary_cross_entropy(probs, labels)

        # Evidence regularization: penalize overconfidence on training data
        # This helps OOD detection by limiting in-distribution evidence
        evidence_reg = torch.mean(alpha + beta)

        return ce_loss + 0.001 * evidence_reg


def build_kg_graph(triples, num_entities):
    """
    Build undirected graph from KG triples for GNN propagation.

    GPN treats all edges equally (relation-agnostic connectivity).
    This is a key limitation we're testing.
    """
    edges = []
    for h, r, t in triples:
        edges.append([h, t])
        edges.append([t, h])  # Undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class GPNTrainer:
    """Trainer for GPN baseline."""

    def __init__(self, model, edge_index, lr=0.001, device='cpu'):
        self.model = model
        self.edge_index = edge_index.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        # Propagate GNN once per epoch
        with torch.no_grad():
            entity_emb = self.model.propagate_gnn(self.edge_index)

        for batch_h, batch_r, batch_t in dataloader:
            batch_h = batch_h.to(self.device)
            batch_r = batch_r.to(self.device)
            batch_t = batch_t.to(self.device)

            # Positive samples
            pos_loss = self.model.evidential_loss(
                batch_h, batch_r, batch_t,
                labels=torch.ones(len(batch_h), device=self.device),
                entity_emb=entity_emb
            )

            # Negative samples
            neg_t = torch.randint(0, self.model.num_entities, batch_t.shape, device=self.device)
            neg_loss = self.model.evidential_loss(
                batch_h, batch_r, neg_t,
                labels=torch.zeros(len(batch_h), device=self.device),
                entity_emb=entity_emb
            )

            loss = pos_loss + neg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
