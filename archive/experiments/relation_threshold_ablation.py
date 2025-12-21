"""
Relation Threshold Ablation Study

Goal: Verify that AUROC increases monotonically with relation diversity
and identify the threshold D_min ≈ 30.

Experiment:
- Use FB15k-237 (237 relations)
- Subsample different numbers of relations: [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 237]
- Train GP-KGE and DistMult on each subset
- Measure AUROC for OOD detection
- Plot: #Relations vs AUROC

Run on Colab with GPU:
!python relation_threshold_ablation.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration
CONFIG = {
    'dataset': 'FB15k-237',
    'relation_counts': [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 237],
    'seeds': [42, 123, 456],
    'epochs': 30,
    'embedding_dim': 100,
    'batch_size': 1024,
    'lr': 0.001,
    'num_inducing': 300,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_fb15k237():
    """Load FB15k-237 dataset."""
    from src.data.loaders import load_dataset
    return load_dataset('FB15k-237')


def subsample_relations(triples, all_relations, num_relations, seed):
    """
    Subsample triples to only include a subset of relations.

    Strategy: Select top-K relations by frequency to ensure sufficient data.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Count relation frequencies
    relation_counts = defaultdict(int)
    for h, r, t in triples:
        relation_counts[r] += 1

    # Sort by frequency and select top-K
    sorted_relations = sorted(relation_counts.keys(),
                              key=lambda r: relation_counts[r],
                              reverse=True)
    selected_relations = set(sorted_relations[:num_relations])

    # Filter triples
    filtered_triples = [(h, r, t) for h, r, t in triples if r in selected_relations]

    # Get entities appearing in filtered triples
    entities = set()
    for h, r, t in filtered_triples:
        entities.add(h)
        entities.add(t)

    return filtered_triples, selected_relations, entities


def compute_auroc(scores_pos, scores_neg):
    """Compute AUROC for OOD detection."""
    from sklearn.metrics import roc_auc_score

    # Positive = ID (high confidence = low uncertainty)
    # Negative = OOD (low confidence = high uncertainty)
    labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    scores = np.concatenate([scores_pos, scores_neg])

    return roc_auc_score(labels, scores)


class SimpleDistMult(nn.Module):
    """Simple DistMult baseline."""

    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return (h * r * t).sum(dim=-1)

    def get_uncertainty(self, heads, relations, tails):
        """Use negative score as uncertainty proxy."""
        scores = torch.sigmoid(self.forward(heads, relations, tails))
        # Entropy-based uncertainty
        uncertainty = -scores * torch.log(scores + 1e-10) - (1-scores) * torch.log(1-scores + 1e-10)
        return uncertainty


class SimpleGPKGE(nn.Module):
    """Simplified GP-KGE for ablation study."""

    def __init__(self, num_entities, num_relations, dim, num_inducing=300):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim

        # Variational parameters
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 2)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Per-relation kernel parameters
        self.log_lengthscales = nn.Parameter(torch.zeros(num_relations))
        self.log_variances = nn.Parameter(torch.zeros(num_relations))

    def forward(self, heads, relations, tails):
        h_mean = self.entity_mean[heads]
        t_mean = self.entity_mean[tails]
        r = self.relation_emb(relations)
        return (h_mean * r * t_mean).sum(dim=-1)

    def sample_embeddings(self, indices):
        """Sample from variational posterior."""
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        eps = torch.randn_like(std)
        return mean + eps * std

    def get_uncertainty(self, heads, relations, tails):
        """Return posterior variance as uncertainty."""
        h_var = torch.exp(self.entity_logvar[heads])
        t_var = torch.exp(self.entity_logvar[tails])
        # Combined uncertainty
        uncertainty = (h_var.sum(dim=-1) + t_var.sum(dim=-1)) / 2
        return uncertainty

    def kl_loss(self):
        """KL divergence from prior."""
        # Assuming standard normal prior
        kl = -0.5 * torch.sum(1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp())
        return kl / self.num_entities


def train_model(model, train_triples, entity_to_idx, relation_to_idx, config, is_gpkge=False):
    """Train a model."""
    device = config['device']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # Prepare data
    heads = torch.tensor([entity_to_idx[h] for h, r, t in train_triples])
    relations = torch.tensor([relation_to_idx[r] for h, r, t in train_triples])
    tails = torch.tensor([entity_to_idx[t] for h, r, t in train_triples])

    num_entities = len(entity_to_idx)
    dataset = TensorDataset(heads, relations, tails)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch_h, batch_r, batch_t in loader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            # Positive scores
            pos_scores = model(batch_h, batch_r, batch_t)

            # Negative sampling
            neg_t = torch.randint(0, num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t)

            # Loss
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            loss = criterion(pos_scores, pos_labels) + criterion(neg_scores, neg_labels)

            if is_gpkge:
                loss += 0.001 * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model


def evaluate_ood(model, test_triples, entity_to_idx, relation_to_idx, config):
    """Evaluate OOD detection using uncertainty."""
    device = config['device']
    model = model.to(device)
    model.eval()

    # Prepare test data
    heads = torch.tensor([entity_to_idx.get(h, 0) for h, r, t in test_triples])
    relations = torch.tensor([relation_to_idx.get(r, 0) for h, r, t in test_triples])
    tails = torch.tensor([entity_to_idx.get(t, 0) for h, r, t in test_triples])

    num_entities = len(entity_to_idx)

    with torch.no_grad():
        heads = heads.to(device)
        relations = relations.to(device)
        tails = tails.to(device)

        # ID: actual test triples (should have low uncertainty)
        id_uncertainty = model.get_uncertainty(heads, relations, tails).cpu().numpy()

        # OOD: random corrupted triples (should have high uncertainty)
        neg_tails = torch.randint(0, num_entities, tails.shape, device=device)
        ood_uncertainty = model.get_uncertainty(heads, relations, neg_tails).cpu().numpy()

    # AUROC: can we distinguish ID from OOD using uncertainty?
    # Lower uncertainty for ID, higher for OOD
    # So we use negative uncertainty as "confidence"
    auroc = compute_auroc(-id_uncertainty, -ood_uncertainty)

    return auroc


def run_ablation():
    """Main ablation study."""
    print("=" * 60)
    print("Relation Threshold Ablation Study")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Relation counts to test: {CONFIG['relation_counts']}")
    print()

    # Load full dataset
    print("Loading FB15k-237...")
    data = load_fb15k237()
    train_triples = data['train']
    test_triples = data['test']
    all_relations = data['relations']
    all_entities = data['entities']

    print(f"Full dataset: {len(train_triples)} train, {len(test_triples)} test")
    print(f"Full: {len(all_entities)} entities, {len(all_relations)} relations")
    print()

    # Results storage
    results = {
        'config': CONFIG,
        'ablation': []
    }

    for num_rels in CONFIG['relation_counts']:
        print(f"\n{'='*40}")
        print(f"Testing with {num_rels} relations")
        print('='*40)

        aurocs_distmult = []
        aurocs_gpkge = []

        for seed in CONFIG['seeds']:
            print(f"\n  Seed {seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Subsample relations
            sub_train, selected_rels, selected_ents = subsample_relations(
                train_triples, all_relations, num_rels, seed
            )
            sub_test = [(h, r, t) for h, r, t in test_triples
                        if r in selected_rels and h in selected_ents and t in selected_ents]

            if len(sub_test) < 100:
                print(f"    Skipping: only {len(sub_test)} test triples")
                continue

            # Create index mappings
            entity_to_idx = {e: i for i, e in enumerate(selected_ents)}
            relation_to_idx = {r: i for i, r in enumerate(selected_rels)}

            print(f"    Subsampled: {len(sub_train)} train, {len(sub_test)} test")
            print(f"    Entities: {len(selected_ents)}, Relations: {len(selected_rels)}")

            # Train DistMult
            distmult = SimpleDistMult(len(selected_ents), len(selected_rels), CONFIG['embedding_dim'])
            distmult = train_model(distmult, sub_train, entity_to_idx, relation_to_idx, CONFIG)
            auroc_dm = evaluate_ood(distmult, sub_test, entity_to_idx, relation_to_idx, CONFIG)
            aurocs_distmult.append(auroc_dm)
            print(f"    DistMult AUROC: {auroc_dm:.4f}")

            # Train GP-KGE
            gpkge = SimpleGPKGE(len(selected_ents), len(selected_rels), CONFIG['embedding_dim'])
            gpkge = train_model(gpkge, sub_train, entity_to_idx, relation_to_idx, CONFIG, is_gpkge=True)
            auroc_gp = evaluate_ood(gpkge, sub_test, entity_to_idx, relation_to_idx, CONFIG)
            aurocs_gpkge.append(auroc_gp)
            print(f"    GP-KGE AUROC: {auroc_gp:.4f}")

        if aurocs_distmult and aurocs_gpkge:
            result = {
                'num_relations': num_rels,
                'distmult_auroc_mean': np.mean(aurocs_distmult),
                'distmult_auroc_std': np.std(aurocs_distmult),
                'gpkge_auroc_mean': np.mean(aurocs_gpkge),
                'gpkge_auroc_std': np.std(aurocs_gpkge),
                'delta': np.mean(aurocs_gpkge) - np.mean(aurocs_distmult),
            }
            results['ablation'].append(result)

            print(f"\n  Summary for {num_rels} relations:")
            print(f"    DistMult: {result['distmult_auroc_mean']:.4f} ± {result['distmult_auroc_std']:.4f}")
            print(f"    GP-KGE:   {result['gpkge_auroc_mean']:.4f} ± {result['gpkge_auroc_std']:.4f}")
            print(f"    Delta:    {result['delta']:+.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/relation_threshold_ablation_{timestamp}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Plot results
    plot_results(results)

    return results


def plot_results(results):
    """Plot ablation results."""
    ablation = results['ablation']

    num_rels = [r['num_relations'] for r in ablation]
    dm_auroc = [r['distmult_auroc_mean'] for r in ablation]
    gp_auroc = [r['gpkge_auroc_mean'] for r in ablation]
    delta = [r['delta'] for r in ablation]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: AUROC vs Relations
    ax1 = axes[0]
    ax1.plot(num_rels, dm_auroc, 'o-', label='DistMult', color='#3498db', linewidth=2, markersize=8)
    ax1.plot(num_rels, gp_auroc, 's-', label='GP-KGE', color='#2ecc71', linewidth=2, markersize=8)
    ax1.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Threshold (D=30)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Number of Relations')
    ax1.set_ylabel('AUROC (OOD Detection)')
    ax1.set_title('AUROC vs Relation Diversity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Delta vs Relations
    ax2 = axes[1]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta]
    ax2.bar(range(len(num_rels)), delta, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=list(num_rels).index(30) if 30 in num_rels else 3,
                color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks(range(len(num_rels)))
    ax2.set_xticklabels(num_rels, rotation=45)
    ax2.set_xlabel('Number of Relations')
    ax2.set_ylabel('AUROC Improvement (GP-KGE - DistMult)')
    ax2.set_title('GP-KGE Improvement vs Relation Diversity')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/relation_threshold_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('outputs/relation_threshold_ablation.png', bbox_inches='tight', dpi=300)
    print("Plot saved to outputs/relation_threshold_ablation.pdf")


if __name__ == '__main__':
    results = run_ablation()

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Relations':<12} {'DistMult':<12} {'GP-KGE':<12} {'Delta':<12} {'Winner'}")
    print("-" * 60)
    for r in results['ablation']:
        winner = "GP-KGE" if r['delta'] > 0 else "DistMult"
        print(f"{r['num_relations']:<12} {r['distmult_auroc_mean']:.4f}       {r['gpkge_auroc_mean']:.4f}       {r['delta']:+.4f}       {winner}")
