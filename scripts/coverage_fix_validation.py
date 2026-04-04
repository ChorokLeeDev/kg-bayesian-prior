#!/usr/bin/env python3
"""
Validate that coverage fix improves prediction quality.

Question: If we filter out zero-coverage predictions,
do the remaining predictions have better accuracy?

Experiment:
1. Train Energy model
2. For test triples, rank all candidate tails
3. Measure Hits@1 (correct tail ranked first) for:
   - All test triples
   - Only test triples WITH coverage (the "fix")
   - Only test triples WITHOUT coverage
4. If fix works: Hits@1(with coverage) > Hits@1(all)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EnergyModel(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0

    def score_tails(self, h, r):
        """Score all tails for query (h, r, ?)"""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        all_t = self.entity_emb.weight
        return (h_emb * r_emb) @ all_t.T


def train_model(model, triples, device, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(torch.tensor(triples[:, 0]),
                      torch.tensor(triples[:, 1]),
                      torch.tensor(triples[:, 2])),
        batch_size=1024, shuffle=True
    )

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def evaluate_with_coverage_filter(model, test_triples, device):
    """
    Evaluate Hits@1 with and without coverage filtering.
    """
    model.eval()

    results = {'all': [], 'with_cov': [], 'no_cov': []}

    with torch.no_grad():
        for i in range(len(test_triples)):
            h, r, t = test_triples[i]

            # Check coverage
            h_cov = model.coverage[h, r].item()
            t_cov = model.coverage[t, r].item()
            has_coverage = (h_cov > 0 and t_cov > 0)

            # Score all tails
            h_tensor = torch.tensor([h], device=device)
            r_tensor = torch.tensor([r], device=device)
            scores = model.score_tails(h_tensor, r_tensor).squeeze(0)

            # Rank of correct tail
            correct_score = scores[t]
            rank = (scores > correct_score).sum().item() + 1
            hit = 1 if rank == 1 else 0

            results['all'].append(hit)
            if has_coverage:
                results['with_cov'].append(hit)
            else:
                results['no_cov'].append(hit)

    return {
        'all_hits1': np.mean(results['all']) * 100,
        'all_count': len(results['all']),
        'cov_hits1': np.mean(results['with_cov']) * 100 if results['with_cov'] else 0,
        'cov_count': len(results['with_cov']),
        'nocov_hits1': np.mean(results['no_cov']) * 100 if results['no_cov'] else 0,
        'nocov_count': len(results['no_cov']),
    }


def main():
    print("Coverage Fix Validation")
    print(f"Date: {datetime.now().isoformat()}")
    print("="*60)
    print("\nQuestion: Does filtering zero-coverage improve prediction quality?")

    device = setup_device()
    print(f"Device: {device}")

    # Load data
    train_ds, _, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"\nFB15k-237: {n_ent} entities, {n_rel} relations")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Run multiple seeds
    seeds = [42, 123, 456]
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = EnergyModel(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)

        result = evaluate_with_coverage_filter(model, test, device)
        all_results.append(result)

        print(f"  Hits@1 (all): {result['all_hits1']:.2f}% (n={result['all_count']})")
        print(f"  Hits@1 (with coverage): {result['cov_hits1']:.2f}% (n={result['cov_count']})")
        print(f"  Hits@1 (no coverage): {result['nocov_hits1']:.2f}% (n={result['nocov_count']})")

    # Aggregate
    mean_all = np.mean([r['all_hits1'] for r in all_results])
    mean_cov = np.mean([r['cov_hits1'] for r in all_results])
    mean_nocov = np.mean([r['nocov_hits1'] for r in all_results])

    std_all = np.std([r['all_hits1'] for r in all_results])
    std_cov = np.std([r['cov_hits1'] for r in all_results])
    std_nocov = np.std([r['nocov_hits1'] for r in all_results])

    improvement = mean_cov - mean_all

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Hits@1 (all test):        {mean_all:.2f}% ± {std_all:.2f}%")
    print(f"Hits@1 (with coverage):   {mean_cov:.2f}% ± {std_cov:.2f}%")
    print(f"Hits@1 (no coverage):     {mean_nocov:.2f}% ± {std_nocov:.2f}%")
    print(f"\nImprovement from fix:     {improvement:+.2f}pp")

    if improvement > 0:
        print("\n>>> COVERAGE FIX IMPROVES PREDICTION QUALITY <<<")
    else:
        print("\n>>> Coverage fix does not improve Hits@1 <<<")

    # Save
    output_path = project_root / "outputs" / "coverage_fix_validation.txt"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"Coverage Fix Validation\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(f"Hits@1 (all):          {mean_all:.2f}% ± {std_all:.2f}%\n")
        f.write(f"Hits@1 (with coverage): {mean_cov:.2f}% ± {std_cov:.2f}%\n")
        f.write(f"Hits@1 (no coverage):   {mean_nocov:.2f}% ± {std_nocov:.2f}%\n")
        f.write(f"\nImprovement: {improvement:+.2f}pp\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
