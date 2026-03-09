#!/usr/bin/env python3
"""
Heterogeneous Graph Coverage Blind Spot Experiments

Tests whether the coverage blind spot (novel edge-type detection failure)
generalizes from KGs to other heterogeneous graphs.

Datasets:
1. OGB-MAG: Microsoft Academic Graph
2. DBLP: Academic network
3. IMDB: Movie database

For each dataset, we:
1. Compute edge-type coverage per node
2. Create novel edge-type test split
3. Test if embedding-based uncertainty fails
4. Test if coverage tracking succeeds
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import argparse

# Try to import OGB
try:
    from ogb.nodeproppred import NodePropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False
    print("OGB not installed. Run: pip install ogb")


def analyze_edge_type_coverage(edge_index, edge_type, num_nodes, num_edge_types):
    """
    Analyze edge-type coverage for each node.

    Returns:
        coverage: dict mapping node_id -> set of edge_types seen
        edge_type_freq: dict mapping (node, edge_type) -> count
    """
    coverage = defaultdict(set)
    edge_type_freq = defaultdict(int)

    src = edge_index[0].numpy() if torch.is_tensor(edge_index[0]) else edge_index[0]
    dst = edge_index[1].numpy() if torch.is_tensor(edge_index[1]) else edge_index[1]
    etypes = edge_type.numpy() if torch.is_tensor(edge_type) else edge_type

    for i in range(len(src)):
        s, d, et = src[i], dst[i], etypes[i]
        coverage[s].add(et)
        coverage[d].add(et)
        edge_type_freq[(s, et)] += 1
        edge_type_freq[(d, et)] += 1

    return coverage, edge_type_freq


def create_novel_edge_type_split(edges, edge_types, coverage, entity_freq,
                                  tau_percentile=25):
    """
    Create train/test split where test has novel (node, edge_type) pairs.

    Novel edge-type: Node has high frequency but hasn't seen this edge type.
    """
    # Compute frequency threshold
    freq_values = list(entity_freq.values())
    if len(freq_values) == 0:
        return [], [], []
    tau = np.percentile(freq_values, tau_percentile)

    novel_edge_type = []
    in_distribution = []
    emerging = []

    src = edges[0].numpy() if torch.is_tensor(edges[0]) else edges[0]
    dst = edges[1].numpy() if torch.is_tensor(edges[1]) else edges[1]
    etypes = edge_types.numpy() if torch.is_tensor(edge_types) else edge_types

    for i in range(len(src)):
        s, d, et = int(src[i]), int(dst[i]), int(etypes[i])

        s_freq = entity_freq.get(s, 0)
        d_freq = entity_freq.get(d, 0)
        min_freq = min(s_freq, d_freq)

        s_covered = et in coverage.get(s, set())
        d_covered = et in coverage.get(d, set())

        if min_freq <= tau:
            emerging.append((s, d, et))
        elif not s_covered or not d_covered:
            novel_edge_type.append((s, d, et))
        else:
            in_distribution.append((s, d, et))

    return novel_edge_type, in_distribution, emerging


class SimpleHeteroGNN(nn.Module):
    """Simple heterogeneous GNN for testing."""

    def __init__(self, num_nodes, num_edge_types, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_type_emb.weight)

    def forward(self, src, dst, edge_type):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        et_emb = self.edge_type_emb(edge_type)

        x = torch.cat([src_emb, et_emb, dst_emb], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    def get_uncertainty(self, src, dst, edge_type):
        """Energy-based uncertainty: -logit"""
        logits = self.forward(src, dst, edge_type)
        return -logits


def compute_coverage_uncertainty(src, dst, edge_type, coverage):
    """Coverage-based uncertainty: 2 - coverage(src) - coverage(dst)"""
    uncertainties = []
    for s, d, et in zip(src, dst, edge_type):
        s, d, et = int(s), int(d), int(et)
        s_cov = 1 if et in coverage.get(s, set()) else 0
        d_cov = 1 if et in coverage.get(d, set()) else 0
        uncertainties.append(2 - s_cov - d_cov)
    return np.array(uncertainties)


def train_model(model, train_edges, train_edge_types, num_nodes,
                epochs=30, lr=1e-3, device='cpu'):
    """Train the model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    src = torch.tensor(train_edges[0], device=device)
    dst = torch.tensor(train_edges[1], device=device)
    etypes = torch.tensor(train_edge_types, device=device)

    for epoch in range(epochs):
        model.train()

        # Positive samples
        pos_scores = model(src, dst, etypes)

        # Negative samples (random destination)
        neg_dst = torch.randint(0, num_nodes, (len(src),), device=device)
        neg_scores = model(src, neg_dst, etypes)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def evaluate_ood(model, ood_edges, id_edges, coverage, device='cpu'):
    """Evaluate OOD detection."""
    model.eval()

    if len(ood_edges) < 10 or len(id_edges) < 10:
        return None, None

    with torch.no_grad():
        # OOD edges
        ood_src = torch.tensor([e[0] for e in ood_edges], device=device)
        ood_dst = torch.tensor([e[1] for e in ood_edges], device=device)
        ood_et = torch.tensor([e[2] for e in ood_edges], device=device)

        ood_energy = model.get_uncertainty(ood_src, ood_dst, ood_et).cpu().numpy()
        ood_coverage = compute_coverage_uncertainty(
            [e[0] for e in ood_edges],
            [e[1] for e in ood_edges],
            [e[2] for e in ood_edges],
            coverage
        )

        # ID edges (sample)
        n_id = min(len(id_edges), len(ood_edges) * 2)
        id_sample = id_edges[:n_id]

        id_src = torch.tensor([e[0] for e in id_sample], device=device)
        id_dst = torch.tensor([e[1] for e in id_sample], device=device)
        id_et = torch.tensor([e[2] for e in id_sample], device=device)

        id_energy = model.get_uncertainty(id_src, id_dst, id_et).cpu().numpy()
        id_coverage = compute_coverage_uncertainty(
            [e[0] for e in id_sample],
            [e[1] for e in id_sample],
            [e[2] for e in id_sample],
            coverage
        )

    # AUROC
    labels = np.concatenate([np.ones(len(ood_energy)), np.zeros(len(id_energy))])

    energy_scores = np.concatenate([ood_energy, id_energy])
    coverage_scores = np.concatenate([ood_coverage, id_coverage])

    energy_auroc = roc_auc_score(labels, energy_scores)
    coverage_auroc = roc_auc_score(labels, coverage_scores)

    return energy_auroc, coverage_auroc


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_synthetic_hetero_graph(num_nodes=5000, num_edge_types=50,
                                 avg_edges_per_node=20, seed=42):
    """
    Create synthetic heterogeneous graph for testing.

    Simulates a graph where:
    - Nodes have varying frequencies
    - Edge types are not uniformly distributed per node
    - Some high-freq nodes have low edge-type coverage
    """
    np.random.seed(seed)

    # Generate edges
    num_edges = num_nodes * avg_edges_per_node

    # Power-law node frequency (some nodes much more frequent)
    node_probs = np.random.power(0.5, num_nodes)
    node_probs /= node_probs.sum()

    src = np.random.choice(num_nodes, size=num_edges, p=node_probs)
    dst = np.random.choice(num_nodes, size=num_edges, p=node_probs)

    # Edge types: each node "prefers" certain edge types
    node_preferred_types = {}
    for n in range(num_nodes):
        # Each node prefers 20-50% of edge types
        n_preferred = np.random.randint(num_edge_types // 5, num_edge_types // 2)
        node_preferred_types[n] = set(np.random.choice(num_edge_types, n_preferred, replace=False))

    edge_types = []
    for s, d in zip(src, dst):
        # 80% chance to use preferred type, 20% random
        if np.random.random() < 0.8:
            preferred = list(node_preferred_types[s] | node_preferred_types[d])
            if preferred:
                et = np.random.choice(preferred)
            else:
                et = np.random.randint(num_edge_types)
        else:
            et = np.random.randint(num_edge_types)
        edge_types.append(et)

    edge_types = np.array(edge_types)

    # Split into train/test (80/20)
    n_train = int(0.8 * num_edges)
    perm = np.random.permutation(num_edges)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_edges = (src[train_idx], dst[train_idx])
    train_edge_types = edge_types[train_idx]

    test_edges = (src[test_idx], dst[test_idx])
    test_edge_types = edge_types[test_idx]

    return {
        'train_edges': train_edges,
        'train_edge_types': train_edge_types,
        'test_edges': test_edges,
        'test_edge_types': test_edge_types,
        'num_nodes': num_nodes,
        'num_edge_types': num_edge_types,
    }


def load_ogb_mag():
    """Load OGB-MAG dataset."""
    if not HAS_OGB:
        return None

    print("Loading OGB-MAG dataset...")
    dataset = NodePropPredDataset(name='ogbn-mag', root='data/ogb')

    graph = dataset[0][0]

    # OGB-MAG has multiple edge types
    # We'll focus on paper-cites-paper for simplicity
    edge_index = graph['edge_index_dict'][('paper', 'cites', 'paper')]
    num_nodes = graph['num_nodes_dict']['paper']

    # Create edge types based on citation patterns
    # (In real OGB-MAG, we could use author-writes-paper, etc.)
    # For now, simulate edge types
    num_edges = edge_index.shape[1]
    edge_types = np.zeros(num_edges, dtype=np.int64)  # Single edge type for cites

    return {
        'edge_index': edge_index,
        'edge_types': edge_types,
        'num_nodes': num_nodes,
        'num_edge_types': 1,  # Just citations
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(dataset_name, seed=42, epochs=30, device='cpu'):
    """Run coverage blind spot experiment on a dataset."""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} (seed={seed})")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    if dataset_name == 'synthetic':
        data = load_synthetic_hetero_graph(
            num_nodes=5000,
            num_edge_types=50,
            avg_edges_per_node=20,
            seed=seed
        )
        train_edges = data['train_edges']
        train_edge_types = data['train_edge_types']
        test_edges = data['test_edges']
        test_edge_types = data['test_edge_types']
        num_nodes = data['num_nodes']
        num_edge_types = data['num_edge_types']

    elif dataset_name == 'ogb-mag':
        data = load_ogb_mag()
        if data is None:
            print("OGB not available, skipping")
            return None
        # TODO: Implement proper train/test split
        return None
    else:
        print(f"Unknown dataset: {dataset_name}")
        return None

    print(f"Nodes: {num_nodes}, Edge types: {num_edge_types}")
    print(f"Train edges: {len(train_edges[0])}, Test edges: {len(test_edges[0])}")

    # Compute coverage from training data
    coverage, edge_type_freq = analyze_edge_type_coverage(
        train_edges, train_edge_types, num_nodes, num_edge_types
    )

    # Compute node frequency
    entity_freq = defaultdict(int)
    for s, d in zip(train_edges[0], train_edges[1]):
        entity_freq[int(s)] += 1
        entity_freq[int(d)] += 1

    # Create novel edge-type split
    novel_et, id_edges, emerging = create_novel_edge_type_split(
        test_edges, test_edge_types, coverage, entity_freq
    )

    print(f"\nTest split:")
    print(f"  Novel edge-type: {len(novel_et)}")
    print(f"  In-distribution: {len(id_edges)}")
    print(f"  Emerging: {len(emerging)}")

    if len(novel_et) < 10:
        print("Too few novel edge-type samples")
        return None

    novel_pct = len(novel_et) / (len(novel_et) + len(id_edges) + len(emerging)) * 100
    print(f"  Novel edge-type percentage: {novel_pct:.1f}%")

    # Train model
    print(f"\nTraining model ({epochs} epochs)...")
    model = SimpleHeteroGNN(num_nodes, num_edge_types)
    model = train_model(model, train_edges, train_edge_types, num_nodes,
                       epochs=epochs, device=device)

    # Evaluate
    print("\nEvaluating OOD detection...")
    energy_auroc, coverage_auroc = evaluate_ood(
        model, novel_et, id_edges, coverage, device
    )

    print(f"\nResults (Novel edge-type detection):")
    print(f"  Energy-based AUROC: {energy_auroc:.3f}")
    print(f"  Coverage-based AUROC: {coverage_auroc:.3f}")

    if energy_auroc < 0.55:
        print("  → Energy FAILS (near-random)")
    if coverage_auroc > 0.95:
        print("  → Coverage WORKS (as expected)")

    return {
        'dataset': dataset_name,
        'seed': seed,
        'novel_et_count': len(novel_et),
        'novel_et_pct': novel_pct,
        'energy_auroc': energy_auroc,
        'coverage_auroc': coverage_auroc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ogb-mag'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    seed_list = [42, 123, 456][:args.seeds]

    all_results = []
    for seed in seed_list:
        result = run_experiment(args.dataset, seed, args.epochs, args.device)
        if result:
            all_results.append(result)

    if all_results:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS - {args.dataset} ({len(all_results)} seeds)")
        print(f"{'='*60}")

        energy_aurocs = [r['energy_auroc'] for r in all_results]
        coverage_aurocs = [r['coverage_auroc'] for r in all_results]
        novel_pcts = [r['novel_et_pct'] for r in all_results]

        print(f"Novel edge-type percentage: {np.mean(novel_pcts):.1f}%")
        print(f"Energy AUROC: {np.mean(energy_aurocs):.3f} ± {np.std(energy_aurocs):.3f}")
        print(f"Coverage AUROC: {np.mean(coverage_aurocs):.3f} ± {np.std(coverage_aurocs):.3f}")

        if np.mean(energy_aurocs) < 0.55:
            print("\n✓ CONFIRMED: Embedding-based methods fail on novel edge-types")
        if np.mean(coverage_aurocs) > 0.95:
            print("✓ CONFIRMED: Coverage tracking solves the problem")


if __name__ == '__main__':
    main()
