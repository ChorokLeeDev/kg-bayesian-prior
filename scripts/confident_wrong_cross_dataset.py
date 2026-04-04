#!/usr/bin/env python3
"""
Cross-dataset confident-wrong analysis.

Validates the 78% finding on multiple datasets:
- FB15k-237 (original)
- WN18RR
- ICEWS14

For each dataset:
- Train Energy model
- Get top-100 most confident predictions
- Calculate % with zero coverage
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EnergyModel(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def train_model(model, train_triples, device, epochs=30, batch_size=1024, lr=1e-3):
    """Train Energy model with negative sampling."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_entities = model.entity_emb.num_embeddings
    train_tensor = torch.tensor(train_triples, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_tensor))
        total_loss = 0

        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[perm[i:i+batch_size]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(batch),), device=device)

            pos_score = model(h, r, t)
            neg_score = model(h, r, neg_t)

            # Margin loss
            loss = torch.clamp(1.0 - pos_score + neg_score, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def analyze_confident_wrong(model, test_triples, device, top_k=100):
    """Analyze top-K most confident predictions.

    For Energy model: confidence = score (higher is more confident)
    This is because Energy scoring uses score directly as confidence,
    and uncertainty = -score.
    """
    model.eval()
    test_tensor = torch.tensor(test_triples, dtype=torch.long, device=device)

    with torch.no_grad():
        h, r, t = test_tensor[:, 0], test_tensor[:, 1], test_tensor[:, 2]
        scores = model(h, r, t)
        # Energy-based: higher score = more confident (lower uncertainty)
        # Sort descending by score = most confident first
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        # Check coverage for top-K
        zero_coverage_count = 0
        for idx in top_indices:
            hi, ri, ti = h[idx].item(), r[idx].item(), t[idx].item()
            h_cov = model.coverage[hi, ri].item()
            t_cov = model.coverage[ti, ri].item()
            if h_cov == 0 or t_cov == 0:
                zero_coverage_count += 1

        # Baseline: overall zero-coverage rate
        baseline_count = 0
        for i in range(len(test_tensor)):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            h_cov = model.coverage[hi, ri].item()
            t_cov = model.coverage[ti, ri].item()
            if h_cov == 0 or t_cov == 0:
                baseline_count += 1

        baseline_rate = baseline_count / len(test_tensor) * 100
        top_k_rate = zero_coverage_count / top_k * 100

    return top_k_rate, baseline_rate


def load_icews14():
    """Load ICEWS14 dataset."""
    data_path = project_root / "data" / "raw" / "icews14"

    # Try different possible file names
    for train_name in ["train.txt", "train.tsv"]:
        train_file = data_path / train_name
        if train_file.exists():
            break
    else:
        raise FileNotFoundError(f"ICEWS14 train file not found in {data_path}")

    test_file = data_path / train_name.replace("train", "test")

    # Build entity/relation mappings
    entities = set()
    relations = set()

    def read_triples(filepath):
        triples = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
                    triples.append((h, r, t))
        return triples

    train_raw = read_triples(train_file)
    test_raw = read_triples(test_file)

    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}

    train_triples = np.array([[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in train_raw])
    test_triples = np.array([[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in test_raw if h in ent2id and t in ent2id and r in rel2id])

    return train_triples, test_triples, len(ent2id), len(rel2id)


def run_experiment(dataset_name, load_fn, seeds=[42, 123, 456], epochs=30):
    """Run confident-wrong analysis on a dataset with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    device = setup_device()
    print(f"Device: {device}")

    # Load data
    if dataset_name == "ICEWS14":
        train_triples, test_triples, num_entities, num_relations = load_fn()
    else:
        # load_fb15k237 and load_wn18rr return (train_dataset, valid_dataset, test_dataset)
        train_ds, _, test_ds = load_fn()
        train_triples = train_ds.triples
        test_triples = test_ds.triples
        num_entities = train_ds.num_entities
        num_relations = train_ds.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    results_top100 = []
    results_baseline = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = EnergyModel(num_entities, num_relations)
        model.precompute_coverage(train_triples)

        model = train_model(model, train_triples, device, epochs=epochs)

        top_k_rate, baseline_rate = analyze_confident_wrong(model, test_triples, device, top_k=100)

        print(f"  Top-100 zero-coverage: {top_k_rate:.1f}%")
        print(f"  Baseline: {baseline_rate:.1f}%")
        print(f"  Elevation: {top_k_rate/baseline_rate:.2f}x")

        results_top100.append(top_k_rate)
        results_baseline.append(baseline_rate)

    mean_top100 = np.mean(results_top100)
    std_top100 = np.std(results_top100)
    mean_baseline = np.mean(results_baseline)

    print(f"\n{dataset_name} Summary:")
    print(f"  Top-100: {mean_top100:.1f}% ± {std_top100:.1f}%")
    print(f"  Baseline: {mean_baseline:.1f}%")
    print(f"  Elevation: {mean_top100/mean_baseline:.2f}x")

    return {
        'dataset': dataset_name,
        'top100_mean': mean_top100,
        'top100_std': std_top100,
        'baseline': mean_baseline,
        'elevation': mean_top100 / mean_baseline
    }


def main():
    print("Cross-Dataset Confident-Wrong Analysis")
    print(f"Date: {datetime.now().isoformat()}")
    print("="*60)

    results = []

    # FB15k-237
    results.append(run_experiment("FB15k-237", load_fb15k237, seeds=[42, 123, 456], epochs=30))

    # WN18RR
    results.append(run_experiment("WN18RR", load_wn18rr, seeds=[42, 123, 456], epochs=30))

    # ICEWS14
    try:
        results.append(run_experiment("ICEWS14", load_icews14, seeds=[42, 123, 456], epochs=30))
    except FileNotFoundError as e:
        print(f"\nICEWS14 skipped: {e}")

    # Summary table
    print("\n" + "="*60)
    print("CROSS-DATASET SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'Top-100':<15} {'Baseline':<10} {'Elevation':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['dataset']:<15} {r['top100_mean']:.1f}%±{r['top100_std']:.1f}% {r['baseline']:.1f}% {r['elevation']:.2f}x")

    # Save results
    output_path = project_root / "outputs" / "confident_wrong_cross_dataset.txt"
    with open(output_path, 'w') as f:
        f.write(f"Cross-Dataset Confident-Wrong Analysis\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Seeds: [42, 123, 456]\n\n")
        f.write(f"{'Dataset':<15} {'Top-100':<15} {'Baseline':<10} {'Elevation':<10}\n")
        f.write("-"*50 + "\n")
        for r in results:
            f.write(f"{r['dataset']:<15} {r['top100_mean']:.1f}%±{r['top100_std']:.1f}% {r['baseline']:.1f}% {r['elevation']:.2f}x\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
