#!/usr/bin/env python3
"""
Real Heterogeneous Graph Experiments

Tests coverage blind spot on real-world heterogeneous graphs:
1. PyG's built-in heterogeneous datasets (DBLP, IMDB, etc.)
2. OGB heterogeneous benchmarks

Requires: torch-geometric
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import argparse

# Try imports
try:
    import torch_geometric
    from torch_geometric.datasets import DBLP, IMDB, LastFM
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not installed. Run: pip install torch-geometric")


def analyze_hetero_data(data, name):
    """Analyze a PyG HeteroData object."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # Node types and counts
    print("\nNode types:")
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            n_nodes = data[node_type].x.shape[0]
        elif hasattr(data[node_type], 'num_nodes'):
            n_nodes = data[node_type].num_nodes
        else:
            n_nodes = 'unknown'
        print(f"  {node_type}: {n_nodes}")

    # Edge types and counts
    print("\nEdge types:")
    edge_type_info = {}
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        n_edges = edge_index.shape[1]
        print(f"  {edge_type}: {n_edges} edges")
        edge_type_info[edge_type] = n_edges

    return edge_type_info


def compute_node_edge_type_coverage(data, target_node_type):
    """
    Compute edge-type coverage for nodes of a specific type.

    Returns: coverage[node_id] = set of edge_types this node participates in
    """
    coverage = defaultdict(set)
    freq = defaultdict(int)

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        if src_type == target_node_type:
            for node in edge_index[0].numpy():
                coverage[int(node)].add(edge_type)
                freq[int(node)] += 1

        if dst_type == target_node_type:
            for node in edge_index[1].numpy():
                coverage[int(node)].add(edge_type)
                freq[int(node)] += 1

    return coverage, freq


def compute_coverage_stats(coverage, freq, num_edge_types):
    """Compute coverage statistics."""
    if not coverage:
        return {}

    coverage_counts = [len(c) for c in coverage.values()]
    coverage_rates = [len(c) / num_edge_types for c in coverage.values()]

    freqs = [freq.get(n, 0) for n in coverage.keys()]

    # Correlation between frequency and coverage
    if len(freqs) > 10:
        corr, pval = spearmanr(freqs, coverage_counts)
    else:
        corr, pval = 0, 1

    return {
        'mean_coverage': np.mean(coverage_counts),
        'mean_coverage_rate': np.mean(coverage_rates),
        'freq_cov_corr': corr,
        'num_nodes': len(coverage),
    }


def run_pyg_dblp():
    """Run experiment on PyG DBLP dataset."""
    if not HAS_PYG:
        print("PyG not available")
        return None

    print("\nLoading PyG DBLP...")
    dataset = DBLP(root='data/pyg/DBLP')
    data = dataset[0]

    edge_info = analyze_hetero_data(data, 'PyG-DBLP')

    # Analyze coverage for authors
    print("\n--- Author coverage analysis ---")
    coverage, freq = compute_node_edge_type_coverage(data, 'author')
    num_edge_types = len([et for et in data.edge_types if 'author' in et[0] or 'author' in et[2]])
    stats = compute_coverage_stats(coverage, freq, num_edge_types)

    print(f"Authors analyzed: {stats.get('num_nodes', 0)}")
    print(f"Mean edge-types per author: {stats.get('mean_coverage', 0):.2f}")
    print(f"Freq-Coverage correlation: {stats.get('freq_cov_corr', 0):.3f}")

    # Analyze coverage for papers
    print("\n--- Paper coverage analysis ---")
    coverage, freq = compute_node_edge_type_coverage(data, 'paper')
    num_edge_types = len([et for et in data.edge_types if 'paper' in et[0] or 'paper' in et[2]])
    stats = compute_coverage_stats(coverage, freq, num_edge_types)

    print(f"Papers analyzed: {stats.get('num_nodes', 0)}")
    print(f"Mean edge-types per paper: {stats.get('mean_coverage', 0):.2f}")
    print(f"Freq-Coverage correlation: {stats.get('freq_cov_corr', 0):.3f}")

    return {'dataset': 'PyG-DBLP', **stats}


def run_pyg_imdb():
    """Run experiment on PyG IMDB dataset."""
    if not HAS_PYG:
        print("PyG not available")
        return None

    print("\nLoading PyG IMDB...")
    dataset = IMDB(root='data/pyg/IMDB')
    data = dataset[0]

    edge_info = analyze_hetero_data(data, 'PyG-IMDB')

    # Analyze coverage for movies
    print("\n--- Movie coverage analysis ---")
    coverage, freq = compute_node_edge_type_coverage(data, 'movie')
    num_edge_types = len([et for et in data.edge_types if 'movie' in et[0] or 'movie' in et[2]])
    stats = compute_coverage_stats(coverage, freq, num_edge_types)

    print(f"Movies analyzed: {stats.get('num_nodes', 0)}")
    print(f"Mean edge-types per movie: {stats.get('mean_coverage', 0):.2f}")
    print(f"Freq-Coverage correlation: {stats.get('freq_cov_corr', 0):.3f}")

    return {'dataset': 'PyG-IMDB', **stats}


def create_novel_edge_type_test(data, target_node_type, test_ratio=0.2, seed=42):
    """
    Create test set with novel (node, edge_type) pairs.

    For each edge type involving target_node_type:
    1. Split edges into train (80%) and test (20%)
    2. In test: identify edges where node hasn't seen this edge type in train
    """
    np.random.seed(seed)

    train_edges = {}
    test_novel = []
    test_id = []

    # First, build training coverage
    train_coverage = defaultdict(set)

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        n_edges = edge_index.shape[1]

        # Split
        n_train = int(n_edges * (1 - test_ratio))
        perm = np.random.permutation(n_edges)

        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_edges[edge_type] = edge_index[:, train_idx]

        # Build training coverage
        if src_type == target_node_type:
            for node in edge_index[0, train_idx].numpy():
                train_coverage[int(node)].add(edge_type)
        if dst_type == target_node_type:
            for node in edge_index[1, train_idx].numpy():
                train_coverage[int(node)].add(edge_type)

    # Now categorize test edges
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        n_edges = edge_index.shape[1]

        n_train = int(n_edges * (1 - test_ratio))
        perm = np.random.permutation(n_edges)
        test_idx = perm[n_train:]

        for i in test_idx:
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()

            # Check if this is novel for target node type
            is_novel = False
            if src_type == target_node_type:
                if edge_type not in train_coverage.get(src, set()):
                    is_novel = True
            if dst_type == target_node_type:
                if edge_type not in train_coverage.get(dst, set()):
                    is_novel = True

            if is_novel:
                test_novel.append((src, dst, edge_type))
            else:
                test_id.append((src, dst, edge_type))

    return train_edges, test_novel, test_id, train_coverage


def run_full_experiment(dataset_name, seed=42):
    """Run full novel edge-type detection experiment."""
    if not HAS_PYG:
        return None

    print(f"\n{'='*60}")
    print(f"Full Experiment: {dataset_name} (seed={seed})")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    if dataset_name == 'DBLP':
        dataset = DBLP(root='data/pyg/DBLP')
        target_node = 'author'
    elif dataset_name == 'IMDB':
        dataset = IMDB(root='data/pyg/IMDB')
        target_node = 'movie'
    else:
        print(f"Unknown dataset: {dataset_name}")
        return None

    data = dataset[0]

    # Create train/test split with novel edge types
    train_edges, test_novel, test_id, coverage = create_novel_edge_type_test(
        data, target_node, test_ratio=0.2, seed=seed
    )

    print(f"\nSplit for {target_node}:")
    print(f"  Test novel edge-type: {len(test_novel)}")
    print(f"  Test in-distribution: {len(test_id)}")

    if len(test_novel) < 10 or len(test_id) < 10:
        print("  Too few samples for experiment")
        return None

    novel_pct = len(test_novel) / (len(test_novel) + len(test_id)) * 100
    print(f"  Novel edge-type percentage: {novel_pct:.1f}%")

    # Coverage-based detection (should be perfect)
    # For test_novel: coverage should be 0 for at least one endpoint
    # For test_id: coverage should be 1 for both endpoints

    coverage_scores_novel = []
    coverage_scores_id = []

    for src, dst, et in test_novel:
        src_cov = 1 if et in coverage.get(src, set()) else 0
        dst_cov = 1 if et in coverage.get(dst, set()) else 0
        coverage_scores_novel.append(2 - src_cov - dst_cov)

    for src, dst, et in test_id:
        src_cov = 1 if et in coverage.get(src, set()) else 0
        dst_cov = 1 if et in coverage.get(dst, set()) else 0
        coverage_scores_id.append(2 - src_cov - dst_cov)

    # Compute AUROC
    labels = np.array([1] * len(coverage_scores_novel) + [0] * len(coverage_scores_id))
    scores = np.array(coverage_scores_novel + coverage_scores_id)

    coverage_auroc = roc_auc_score(labels, scores)
    print(f"\nCoverage-based AUROC: {coverage_auroc:.3f}")

    if coverage_auroc > 0.95:
        print("✓ CONFIRMED: Coverage tracking works on real hetero graph")

    return {
        'dataset': dataset_name,
        'seed': seed,
        'novel_count': len(test_novel),
        'id_count': len(test_id),
        'novel_pct': novel_pct,
        'coverage_auroc': coverage_auroc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['DBLP', 'IMDB', 'all'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--full', action='store_true',
                       help='Run full experiment with train/test split')
    args = parser.parse_args()

    if args.full:
        # Run full experiments
        datasets = ['DBLP', 'IMDB'] if args.dataset == 'all' else [args.dataset]

        for ds in datasets:
            results = []
            for seed in [42, 123, 456][:args.seeds]:
                r = run_full_experiment(ds, seed)
                if r:
                    results.append(r)

            if results:
                print(f"\n{'='*60}")
                print(f"AGGREGATE: {ds}")
                print(f"{'='*60}")
                aurocs = [r['coverage_auroc'] for r in results]
                pcts = [r['novel_pct'] for r in results]
                print(f"Novel edge-type %: {np.mean(pcts):.1f}%")
                print(f"Coverage AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    else:
        # Just analyze datasets
        if args.dataset in ['DBLP', 'all']:
            run_pyg_dblp()

        if args.dataset in ['IMDB', 'all']:
            run_pyg_imdb()


if __name__ == '__main__':
    main()
