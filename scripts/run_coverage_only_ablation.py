#!/usr/bin/env python3
"""
Coverage-Only Ablation: Proving GP Adds Nothing

Critical experiment for NeurIPS submission.

Hypothesis: If coverage-only ≈ CAGP, then GP component is worthless.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def load_triples(path):
    """Load triples from a TSV file."""
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_dataset(name):
    """Load dataset from local data directory."""
    # Get the project root (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Map dataset names to local directories
    local_paths = {
        'WN18RR': os.path.join(project_root, 'data', 'raw', 'wn18rr'),
        'FB15k-237': os.path.join(project_root, 'data', 'raw', 'fb15k-237'),
    }

    if name not in local_paths:
        raise ValueError(f"Unknown dataset: {name}")

    dataset_dir = local_paths[name]
    train_file = os.path.join(dataset_dir, 'train.txt')
    test_file = os.path.join(dataset_dir, 'test.txt')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")

    train = load_triples(train_file)
    test = load_triples(test_file)

    entities = set()
    relations = set()
    for h, r, t in train + test:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    return {
        'train': train, 'test': test,
        'entities': list(entities), 'relations': list(relations)
    }


class CoverageOnlyDetector:
    """
    No learning. Just tracks coverage.

    U = 2 - coverage[head, relation] - coverage[tail, relation]
    """
    def __init__(self, num_entities, num_relations):
        self.coverage = torch.zeros(num_entities, num_relations)

    def fit(self, triples, entity_to_idx, relation_to_idx):
        """Build coverage matrix from training triples."""
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0
        return self

    def get_uncertainty(self, heads, relations, tails):
        """Pure coverage-based uncertainty."""
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen


def evaluate_auroc(detector, test_triples, entity_to_idx, relation_to_idx, num_entities):
    """Evaluate AUROC for OOD detection."""
    heads = torch.tensor([entity_to_idx.get(h, 0) for h, r, t in test_triples])
    relations = torch.tensor([relation_to_idx.get(r, 0) for h, r, t in test_triples])
    tails = torch.tensor([entity_to_idx.get(t, 0) for h, r, t in test_triples])

    # ID uncertainty
    id_unc = detector.get_uncertainty(heads, relations, tails).numpy()

    # OOD uncertainty (random tails)
    neg_tails = torch.randint(0, num_entities, tails.shape)
    ood_unc = detector.get_uncertainty(heads, relations, neg_tails).numpy()

    labels = np.concatenate([np.ones(len(id_unc)), np.zeros(len(ood_unc))])
    scores = np.concatenate([-id_unc, -ood_unc])  # Lower uncertainty = higher confidence

    return roc_auc_score(labels, scores)


# Previous CAGP results for comparison
CAGP_RESULTS = {
    'WN18RR': {'mean': 0.871, 'std': 0.001},
    'FB15k-237': {'mean': 0.960, 'std': 0.000},
}

DATASETS = ['WN18RR', 'FB15k-237']
SEEDS = [42, 123, 456]

results = {}

for dataset_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print('='*60)

    data = load_dataset(dataset_name)
    print(f"Train: {len(data['train'])}, Test: {len(data['test'])}")
    print(f"Entities: {len(data['entities'])}, Relations: {len(data['relations'])}")

    ent2idx = {e: i for i, e in enumerate(data['entities'])}
    rel2idx = {r: i for i, r in enumerate(data['relations'])}

    aurocs = []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fit coverage detector (no training, just counting)
        detector = CoverageOnlyDetector(len(ent2idx), len(rel2idx))
        detector.fit(data['train'], ent2idx, rel2idx)

        # Evaluate
        auroc = evaluate_auroc(detector, data['test'], ent2idx, rel2idx, len(ent2idx))
        aurocs.append(auroc)
        print(f"  Seed {seed}: AUROC = {auroc:.4f}")

    results[dataset_name] = {
        'mean': float(np.mean(aurocs)),
        'std': float(np.std(aurocs)),
        'relations': len(data['relations'])
    }
    print(f"  => Mean: {results[dataset_name]['mean']:.4f} +/- {results[dataset_name]['std']:.4f}")

# Print comparison table
print("\n" + "="*70)
print("COVERAGE-ONLY vs CAGP COMPARISON")
print("="*70)
print(f"{'Dataset':<15} {'Relations':<10} {'Coverage-Only':<18} {'CAGP':<18} {'Diff'}")
print("-"*70)

for name in results:
    cov = results[name]
    cagp = CAGP_RESULTS.get(name, {'mean': 0, 'std': 0})
    diff = cov['mean'] - cagp['mean']

    print(f"{name:<15} {cov['relations']:<10} {cov['mean']:.4f} +/- {cov['std']:.3f}    "
          f"{cagp['mean']:.4f} +/- {cagp['std']:.3f}    {diff:+.4f}")

print("-"*70)
print("\nINTERPRETATION:")
print("If Coverage-Only approx CAGP, then GP component adds NOTHING.")
print("This proves the negative result: complex GP methods are unnecessary.")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

for name in results:
    cov = results[name]
    cagp = CAGP_RESULTS.get(name, {'mean': 0, 'std': 0})
    diff = abs(cov['mean'] - cagp['mean'])

    if diff < 0.02:
        verdict = "GP adds NOTHING (<2% difference)"
    elif diff < 0.05:
        verdict = "GP adds minimal value (2-5% difference)"
    else:
        verdict = "GP adds some value (>5% difference)"

    print(f"{name}: {verdict}")

# Save results
output = {
    'experiment': 'coverage_only_ablation',
    'purpose': 'Prove GP component adds nothing beyond coverage',
    'results': results,
    'comparison_to_cagp': CAGP_RESULTS,
    'conclusion': 'See ANALYSIS above'
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/coverage_only_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to outputs/coverage_only_results.json")
