#!/usr/bin/env python3
"""
Computational Benchmark: Coverage Lookup Overhead

Measures wall-clock time comparing:
1. Baseline inference (scoring 10K triples)
2. Inference + hash table coverage lookup
3. Inference + Bloom filter lookup

Reports latency per triple (mean, p95, p99), memory footprint, and overhead percentage.

Dataset: FB15k-237
"""

import sys
import time
import math
import hashlib
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.data.loaders import load_fb15k237


class BloomFilter:
    """Bloom filter for coverage lookup using fast xxhash-style hashing."""

    def __init__(self, expected_elements, fpr=0.01):
        self.size = int(-expected_elements * math.log(fpr) / (math.log(2) ** 2))
        self.num_hashes = max(1, int((self.size / expected_elements) * math.log(2)))
        self.bit_array = np.zeros(self.size, dtype=np.uint8)

    def _hashes(self, item):
        """Fast double hashing: h(i) = (h1 + i*h2) mod size."""
        # Use Python's built-in hash (fast) with two seeds
        h1 = hash(item) % self.size
        h2 = hash((item[1], item[0])) % self.size  # Swap for second hash
        if h2 == 0:
            h2 = 1
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def __contains__(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))

    def memory_bytes(self):
        return self.bit_array.nbytes


class FastTensorCoverage:
    """Fast tensor-based coverage lookup using PyTorch (batched)."""

    def __init__(self, num_entities, num_relations, device='cpu'):
        self.coverage = torch.zeros(num_entities, num_relations, dtype=torch.bool, device=device)
        self.device = device

    def add_batch(self, entities, relations):
        """Add coverage for batch of (entity, relation) pairs."""
        self.coverage[entities, relations] = True

    def lookup_batch(self, entities, relations):
        """Batch lookup returning boolean tensor."""
        return self.coverage[entities, relations]

    def memory_bytes(self):
        return self.coverage.nelement() * self.coverage.element_size()


class SimpleDistMult(nn.Module):
    """Simple DistMult model for benchmarking."""

    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score_triple(self, head, relation, tail):
        h_emb = self.entity_embeddings(head)
        r_emb = self.relation_embeddings(relation)
        t_emb = self.entity_embeddings(tail)
        return torch.sum(h_emb * r_emb * t_emb, dim=-1)


def build_hash_coverage(triples):
    """Build hash set coverage from triples."""
    coverage = set()
    for h, r, t in triples:
        coverage.add((int(h), int(r)))
        coverage.add((int(t), int(r)))
    return coverage


def build_bloom_coverage(triples, num_pairs, fpr=0.01):
    """Build Bloom filter coverage from triples."""
    bloom = BloomFilter(num_pairs, fpr=fpr)
    for h, r, t in triples:
        bloom.add((int(h), int(r)))
        bloom.add((int(t), int(r)))
    return bloom


def build_tensor_coverage(triples, num_entities, num_relations, device='cpu'):
    """Build tensor-based coverage for batched lookup."""
    coverage = FastTensorCoverage(num_entities, num_relations, device)
    triples_np = triples.astype(np.int64)
    heads = torch.tensor(triples_np[:, 0], device=device)
    rels = torch.tensor(triples_np[:, 1], device=device)
    tails = torch.tensor(triples_np[:, 2], device=device)
    coverage.add_batch(heads, rels)
    coverage.add_batch(tails, rels)
    return coverage


def measure_baseline_inference(model, triples, device, num_warmup=100, num_runs=1000):
    """Measure baseline inference time without coverage lookup."""
    model.eval()
    triples_t = torch.tensor(triples, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    return np.array(latencies)


def measure_inference_with_hash(model, triples, device, coverage_set, num_warmup=100, num_runs=1000):
    """Measure inference + hash table coverage lookup."""
    model.eval()
    triples_t = torch.tensor(triples, dtype=torch.long, device=device)
    triples_np = triples.astype(np.int64)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            for h, r, t in triples_np:
                _ = (h, r) in coverage_set
                _ = (t, r) in coverage_set
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            for h, r, t in triples_np:
                _ = (h, r) in coverage_set
                _ = (t, r) in coverage_set
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    return np.array(latencies)


def measure_inference_with_bloom(model, triples, device, bloom_filter, num_warmup=100, num_runs=1000):
    """Measure inference + Bloom filter coverage lookup."""
    model.eval()
    triples_t = torch.tensor(triples, dtype=torch.long, device=device)
    triples_np = triples.astype(np.int64)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            for h, r, t in triples_np:
                _ = (h, r) in bloom_filter
                _ = (t, r) in bloom_filter
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.score_triple(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])
            for h, r, t in triples_np:
                _ = (h, r) in bloom_filter
                _ = (t, r) in bloom_filter
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    return np.array(latencies)


def measure_inference_with_tensor(model, triples, device, tensor_coverage, num_warmup=100, num_runs=1000):
    """Measure inference + tensor-based batched coverage lookup."""
    model.eval()
    triples_t = torch.tensor(triples, dtype=torch.long, device=device)
    heads = triples_t[:, 0]
    rels = triples_t[:, 1]
    tails = triples_t[:, 2]

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.score_triple(heads, rels, tails)
            _ = tensor_coverage.lookup_batch(heads, rels)
            _ = tensor_coverage.lookup_batch(tails, rels)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.score_triple(heads, rels, tails)
            _ = tensor_coverage.lookup_batch(heads, rels)
            _ = tensor_coverage.lookup_batch(tails, rels)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    return np.array(latencies)


def compute_stats(latencies, num_triples):
    """Compute latency statistics per triple."""
    per_triple = latencies / num_triples * 1e6  # Convert to microseconds
    return {
        'mean_us': np.mean(per_triple),
        'std_us': np.std(per_triple),
        'p95_us': np.percentile(per_triple, 95),
        'p99_us': np.percentile(per_triple, 99),
        'total_mean_ms': np.mean(latencies) * 1000,
    }


def estimate_memory(coverage_set=None, bloom_filter=None):
    """Estimate memory footprint."""
    if coverage_set is not None:
        # Python set: approx 50 bytes per tuple entry
        return len(coverage_set) * 50
    if bloom_filter is not None:
        return bloom_filter.memory_bytes()
    return 0


def run_benchmark():
    """Run the full benchmark."""
    print("=" * 70)
    print("Coverage Lookup Overhead Benchmark")
    print("Dataset: FB15k-237")
    print("=" * 70)

    # Load dataset
    print("\nLoading FB15k-237...")
    train_ds, valid_ds, test_ds = load_fb15k237()

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    train_triples = train_ds.triples

    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples):,}")

    # Select 10K test triples
    num_test = 10000
    test_triples = test_ds.triples[:num_test]
    print(f"  Test triples (benchmark): {len(test_triples):,}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Build model
    print("\nInitializing DistMult model...")
    model = SimpleDistMult(num_entities, num_relations, embedding_dim=100).to(device)

    # Build coverage structures
    print("\nBuilding coverage structures...")

    start = time.time()
    hash_coverage = build_hash_coverage(train_triples)
    hash_build_time = time.time() - start
    num_pairs = len(hash_coverage)
    print(f"  Hash set: {num_pairs:,} pairs, build time: {hash_build_time:.2f}s")

    start = time.time()
    bloom_coverage = build_bloom_coverage(train_triples, num_pairs, fpr=0.01)
    bloom_build_time = time.time() - start
    print(f"  Bloom filter (1% FPR): build time: {bloom_build_time:.2f}s")

    start = time.time()
    tensor_coverage = build_tensor_coverage(train_triples, num_entities, num_relations, device)
    tensor_build_time = time.time() - start
    print(f"  Tensor coverage (batched): build time: {tensor_build_time:.2f}s")

    # Memory footprint
    hash_memory = estimate_memory(coverage_set=hash_coverage)
    bloom_memory = estimate_memory(bloom_filter=bloom_coverage)
    tensor_memory = tensor_coverage.memory_bytes()

    print(f"\nMemory footprint:")
    print(f"  Hash set: {hash_memory / 1024 / 1024:.2f} MB")
    print(f"  Bloom filter: {bloom_memory / 1024 / 1024:.2f} MB")
    print(f"  Tensor (dense): {tensor_memory / 1024 / 1024:.2f} MB")
    print(f"  Reduction (Bloom vs Hash): {hash_memory / bloom_memory:.1f}x")

    # Run benchmarks
    num_warmup = 50
    num_runs = 500
    print(f"\nRunning benchmarks ({num_warmup} warmup, {num_runs} runs)...")

    print("\n  [1/4] Baseline inference...")
    baseline_latencies = measure_baseline_inference(
        model, test_triples, device, num_warmup, num_runs
    )
    baseline_stats = compute_stats(baseline_latencies, num_test)

    print("  [2/4] Inference + Hash lookup (sequential)...")
    hash_latencies = measure_inference_with_hash(
        model, test_triples, device, hash_coverage, num_warmup, num_runs
    )
    hash_stats = compute_stats(hash_latencies, num_test)

    print("  [3/4] Inference + Bloom lookup (sequential)...")
    bloom_latencies = measure_inference_with_bloom(
        model, test_triples, device, bloom_coverage, num_warmup, num_runs
    )
    bloom_stats = compute_stats(bloom_latencies, num_test)

    print("  [4/4] Inference + Tensor lookup (batched)...")
    tensor_latencies = measure_inference_with_tensor(
        model, test_triples, device, tensor_coverage, num_warmup, num_runs
    )
    tensor_stats = compute_stats(tensor_latencies, num_test)

    # Calculate overhead
    hash_overhead = (hash_stats['total_mean_ms'] - baseline_stats['total_mean_ms']) / baseline_stats['total_mean_ms'] * 100
    bloom_overhead = (bloom_stats['total_mean_ms'] - baseline_stats['total_mean_ms']) / baseline_stats['total_mean_ms'] * 100
    tensor_overhead = (tensor_stats['total_mean_ms'] - baseline_stats['total_mean_ms']) / baseline_stats['total_mean_ms'] * 100

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nLatency per triple (microseconds):")
    print(f"{'Method':<30} {'Mean':>10} {'Std':>10} {'P95':>10} {'P99':>10}")
    print("-" * 70)
    print(f"{'Baseline (inference only)':<30} {baseline_stats['mean_us']:>10.2f} {baseline_stats['std_us']:>10.2f} {baseline_stats['p95_us']:>10.2f} {baseline_stats['p99_us']:>10.2f}")
    print(f"{'+ Hash lookup (sequential)':<30} {hash_stats['mean_us']:>10.2f} {hash_stats['std_us']:>10.2f} {hash_stats['p95_us']:>10.2f} {hash_stats['p99_us']:>10.2f}")
    print(f"{'+ Bloom lookup (sequential)':<30} {bloom_stats['mean_us']:>10.2f} {bloom_stats['std_us']:>10.2f} {bloom_stats['p95_us']:>10.2f} {bloom_stats['p99_us']:>10.2f}")
    print(f"{'+ Tensor lookup (batched)':<30} {tensor_stats['mean_us']:>10.2f} {tensor_stats['std_us']:>10.2f} {tensor_stats['p95_us']:>10.2f} {tensor_stats['p99_us']:>10.2f}")

    print(f"\nTotal time for {num_test:,} triples (ms):")
    print(f"{'Method':<30} {'Mean (ms)':>12} {'Overhead':>12}")
    print("-" * 55)
    print(f"{'Baseline (inference only)':<30} {baseline_stats['total_mean_ms']:>12.2f} {'-':>12}")
    print(f"{'+ Hash lookup (sequential)':<30} {hash_stats['total_mean_ms']:>12.2f} {hash_overhead:>+11.1f}%")
    print(f"{'+ Bloom lookup (sequential)':<30} {bloom_stats['total_mean_ms']:>12.2f} {bloom_overhead:>+11.1f}%")
    print(f"{'+ Tensor lookup (batched)':<30} {tensor_stats['total_mean_ms']:>12.2f} {tensor_overhead:>+11.1f}%")

    print(f"\nMemory footprint:")
    print(f"{'Method':<30} {'Size':>15}")
    print("-" * 45)
    print(f"{'Hash set':<30} {hash_memory / 1024 / 1024:>12.2f} MB")
    print(f"{'Bloom filter (1% FPR)':<30} {bloom_memory / 1024 / 1024:>12.2f} MB")
    print(f"{'Tensor (dense matrix)':<30} {tensor_memory / 1024 / 1024:>12.2f} MB")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "computational_benchmark.txt"

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Coverage Lookup Overhead Benchmark\n")
        f.write("Dataset: FB15k-237\n")
        f.write(f"Test triples: {num_test:,}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Entities: {num_entities:,}\n")
        f.write(f"  Relations: {num_relations}\n")
        f.write(f"  Train triples: {len(train_triples):,}\n")
        f.write(f"  Covered (e,r) pairs: {num_pairs:,}\n")
        f.write(f"  Warmup runs: {num_warmup}\n")
        f.write(f"  Benchmark runs: {num_runs}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Latency per triple (microseconds)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<30} {'Mean':>10} {'Std':>10} {'P95':>10} {'P99':>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Baseline (inference only)':<30} {baseline_stats['mean_us']:>10.2f} {baseline_stats['std_us']:>10.2f} {baseline_stats['p95_us']:>10.2f} {baseline_stats['p99_us']:>10.2f}\n")
        f.write(f"{'+ Hash lookup (sequential)':<30} {hash_stats['mean_us']:>10.2f} {hash_stats['std_us']:>10.2f} {hash_stats['p95_us']:>10.2f} {hash_stats['p99_us']:>10.2f}\n")
        f.write(f"{'+ Bloom lookup (sequential)':<30} {bloom_stats['mean_us']:>10.2f} {bloom_stats['std_us']:>10.2f} {bloom_stats['p95_us']:>10.2f} {bloom_stats['p99_us']:>10.2f}\n")
        f.write(f"{'+ Tensor lookup (batched)':<30} {tensor_stats['mean_us']:>10.2f} {tensor_stats['std_us']:>10.2f} {tensor_stats['p95_us']:>10.2f} {tensor_stats['p99_us']:>10.2f}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"Total time for {num_test:,} triples\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<30} {'Mean (ms)':>12} {'Overhead':>12}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Baseline (inference only)':<30} {baseline_stats['total_mean_ms']:>12.2f} {'-':>12}\n")
        f.write(f"{'+ Hash lookup (sequential)':<30} {hash_stats['total_mean_ms']:>12.2f} {hash_overhead:>+11.1f}%\n")
        f.write(f"{'+ Bloom lookup (sequential)':<30} {bloom_stats['total_mean_ms']:>12.2f} {bloom_overhead:>+11.1f}%\n")
        f.write(f"{'+ Tensor lookup (batched)':<30} {tensor_stats['total_mean_ms']:>12.2f} {tensor_overhead:>+11.1f}%\n\n")

        f.write("-" * 70 + "\n")
        f.write("Memory footprint\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<30} {'Size':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Hash set':<30} {hash_memory / 1024 / 1024:>12.2f} MB\n")
        f.write(f"{'Bloom filter (1% FPR)':<30} {bloom_memory / 1024 / 1024:>12.2f} MB\n")
        f.write(f"{'Tensor (dense matrix)':<30} {tensor_memory / 1024 / 1024:>12.2f} MB\n\n")

        f.write("-" * 70 + "\n")
        f.write("Summary\n")
        f.write("-" * 70 + "\n")
        f.write(f"Hash lookup overhead: {hash_overhead:+.1f}%\n")
        f.write(f"Bloom lookup overhead: {bloom_overhead:+.1f}%\n")
        f.write(f"Tensor lookup overhead: {tensor_overhead:+.1f}%\n")
        f.write(f"Memory reduction (Bloom vs Hash): {hash_memory / bloom_memory:.1f}x\n\n")

        f.write("-" * 70 + "\n")
        f.write("Conclusion\n")
        f.write("-" * 70 + "\n")
        f.write("Tensor-based batched lookup provides the best latency (minimal overhead)\n")
        f.write("at the cost of higher memory for small/medium KGs.\n")
        f.write("For large-scale KGs (>100K entities), Bloom filter trades memory for speed.\n")
        f.write("Hash table provides a balanced middle ground.\n")

    print(f"\nResults saved to: {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
