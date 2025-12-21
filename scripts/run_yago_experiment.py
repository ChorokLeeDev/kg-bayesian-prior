#!/usr/bin/env python3
"""
YAGO3-10 Experiment: Validate synergy on third dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import os
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Configuration
CONFIG = {
    'epochs': 30,
    'embedding_dim': 100,
    'batch_size': 2048,
    'lr': 0.001,
    'kl_weight': 0.01,
    'seeds': [42, 123, 456],
}


def load_yago():
    """Load YAGO3-10 dataset."""
    # Try to download if not exists
    cache_dir = os.path.expanduser("~/.kg_cache/yago")
    os.makedirs(cache_dir, exist_ok=True)

    train_file = os.path.join(cache_dir, "train.txt")
    test_file = os.path.join(cache_dir, "test.txt")

    if not os.path.exists(train_file):
        print("Downloading YAGO3-10...")
        import urllib.request
        base_url = "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10"
        urllib.request.urlretrieve(f"{base_url}/train.txt", train_file)
        urllib.request.urlretrieve(f"{base_url}/test.txt", test_file)
        print("Download complete.")

    def load_triples(path):
        triples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
        return triples

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
    """Pure coverage-based uncertainty (no learning)."""
    def __init__(self, num_entities, num_relations):
        self.coverage = torch.zeros(num_entities, num_relations)

    def fit(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            self.coverage[entity_to_idx[h], relation_to_idx[r]] = 1.0
            self.coverage[entity_to_idx[t], relation_to_idx[r]] = 1.0
        return self

    def get_uncertainty(self, heads, relations, tails):
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen


class VanillaGPKGE(nn.Module):
    """GP-KGE with entity-level variance."""
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.num_entities = num_entities
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, heads, relations, tails, use_sampling=True):
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]
        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices):
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def get_uncertainty(self, heads, relations, tails):
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def kl_loss(self):
        kl = -0.5 * torch.sum(1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp())
        return kl / self.num_entities


class CAGP(nn.Module):
    """Coverage-Augmented GP-KGE."""
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, heads, relations, tails, use_sampling=True):
        if use_sampling and self.training:
            h = self._sample(heads)
            t = self._sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]
        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def _sample(self, indices):
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def get_uncertainty(self, heads, relations, tails):
        # GP variance (semantic)
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2

        # Coverage (structural)
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        cov_unc = 2.0 - h_seen - t_seen

        # Normalize GP variance to same scale
        gp_var_norm = gp_var / (gp_var.mean() + 1e-8) * (cov_unc.mean() + 1e-8)

        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * gp_var_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            self.coverage[entity_to_idx[h], relation_to_idx[r]] = 1.0
            self.coverage[entity_to_idx[t], relation_to_idx[r]] = 1.0

    def kl_loss(self):
        kl = -0.5 * torch.sum(1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp())
        return kl / self.num_entities

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit).item()


def train_model(model, triples, entity_to_idx, relation_to_idx, epochs, is_gp=False):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss()

    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(triples, entity_to_idx, relation_to_idx)

    heads = torch.tensor([entity_to_idx[h] for h, r, t in triples])
    relations = torch.tensor([relation_to_idx[r] for h, r, t in triples])
    tails = torch.tensor([entity_to_idx[t] for h, r, t in triples])

    num_entities = len(entity_to_idx)
    dataset = TensorDataset(heads, relations, tails)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_h, batch_r, batch_t in loader:
            batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)

            pos_scores = model(batch_h, batch_r, batch_t, use_sampling=is_gp)
            neg_t = torch.randint(0, num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t, use_sampling=is_gp)

            loss = criterion(pos_scores, torch.ones_like(pos_scores)) + \
                   criterion(neg_scores, torch.zeros_like(neg_scores))

            if is_gp and hasattr(model, 'kl_loss'):
                loss += CONFIG['kl_weight'] * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def evaluate_auroc(model, test_triples, entity_to_idx, relation_to_idx, is_coverage_only=False):
    if not is_coverage_only:
        model.eval()

    heads = torch.tensor([entity_to_idx.get(h, 0) for h, r, t in test_triples])
    relations = torch.tensor([relation_to_idx.get(r, 0) for h, r, t in test_triples])
    tails = torch.tensor([entity_to_idx.get(t, 0) for h, r, t in test_triples])

    if not is_coverage_only:
        heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)

    with torch.no_grad():
        id_unc = model.get_uncertainty(heads, relations, tails)
        if not is_coverage_only:
            id_unc = id_unc.cpu()
        id_unc = id_unc.numpy()

        neg_tails = torch.randint(0, len(entity_to_idx), tails.shape)
        if not is_coverage_only:
            neg_tails = neg_tails.to(device)
        ood_unc = model.get_uncertainty(heads, relations, neg_tails)
        if not is_coverage_only:
            ood_unc = ood_unc.cpu()
        ood_unc = ood_unc.numpy()

    labels = np.concatenate([np.ones(len(id_unc)), np.zeros(len(ood_unc))])
    scores = np.concatenate([-id_unc, -ood_unc])
    return roc_auc_score(labels, scores)


def main():
    print("Loading YAGO3-10...")
    data = load_yago()
    print(f"Train: {len(data['train'])}, Test: {len(data['test'])}")
    print(f"Entities: {len(data['entities'])}, Relations: {len(data['relations'])}")

    ent2idx = {e: i for i, e in enumerate(data['entities'])}
    rel2idx = {r: i for i, r in enumerate(data['relations'])}

    results = {
        'CoverageOnly': [],
        'VanillaGPKGE': [],
        'CAGP': []
    }
    alphas = []

    for seed in CONFIG['seeds']:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print('='*50)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Coverage Only (no training)
        print("  Coverage Only...")
        cov_detector = CoverageOnlyDetector(len(ent2idx), len(rel2idx))
        cov_detector.fit(data['train'], ent2idx, rel2idx)
        auroc_cov = evaluate_auroc(cov_detector, data['test'], ent2idx, rel2idx, is_coverage_only=True)
        results['CoverageOnly'].append(auroc_cov)
        print(f"    AUROC: {auroc_cov:.4f}")

        # Vanilla GP-KGE
        print("  VanillaGPKGE...")
        gp = VanillaGPKGE(len(ent2idx), len(rel2idx), CONFIG['embedding_dim'])
        gp = train_model(gp, data['train'], ent2idx, rel2idx, CONFIG['epochs'], is_gp=True)
        auroc_gp = evaluate_auroc(gp, data['test'], ent2idx, rel2idx)
        results['VanillaGPKGE'].append(auroc_gp)
        print(f"    AUROC: {auroc_gp:.4f}")

        # CAGP
        print("  CAGP...")
        cagp = CAGP(len(ent2idx), len(rel2idx), CONFIG['embedding_dim'])
        cagp = train_model(cagp, data['train'], ent2idx, rel2idx, CONFIG['epochs'], is_gp=True)
        auroc_cagp = evaluate_auroc(cagp, data['test'], ent2idx, rel2idx)
        results['CAGP'].append(auroc_cagp)
        alphas.append(cagp.get_alpha())
        print(f"    AUROC: {auroc_cagp:.4f} (alpha={cagp.get_alpha():.3f})")

    # Summary
    print("\n" + "="*70)
    print("YAGO3-10 RESULTS (37 relations)")
    print("="*70)

    for method in results:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"{method:<20} {mean:.4f} +/- {std:.4f}")

    print(f"\nLearned alpha: {np.mean(alphas):.3f} +/- {np.std(alphas):.3f}")

    # Synergy analysis
    cov_mean = np.mean(results['CoverageOnly'])
    gp_mean = np.mean(results['VanillaGPKGE'])
    cagp_mean = np.mean(results['CAGP'])
    best_single = max(cov_mean, gp_mean)
    synergy = cagp_mean - best_single

    print(f"\n{'='*70}")
    print("SYNERGY ANALYSIS")
    print("="*70)
    print(f"Best single component: {best_single:.4f}")
    print(f"CAGP: {cagp_mean:.4f}")
    print(f"Synergy (CAGP - best single): {synergy:+.4f} ({synergy/best_single*100:+.1f}%)")

    if synergy > 0.05:
        print("\n** SYNERGY CONFIRMED: CAGP >> best single component **")
    elif synergy > 0.02:
        print("\n** MODERATE SYNERGY: CAGP > best single component **")
    else:
        print("\n** WEAK/NO SYNERGY: CAGP ~ best single component **")

    # Save results
    output = {
        'dataset': 'YAGO3-10',
        'num_relations': len(data['relations']),
        'num_entities': len(data['entities']),
        'results': {m: {'mean': float(np.mean(results[m])), 'std': float(np.std(results[m]))} for m in results},
        'learned_alpha': {'mean': float(np.mean(alphas)), 'std': float(np.std(alphas))},
        'synergy': float(synergy),
        'config': CONFIG
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/yago_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to outputs/yago_results.json")


if __name__ == "__main__":
    main()
