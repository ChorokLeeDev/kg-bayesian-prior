#!/usr/bin/env python3
"""
YAGO3-10 Quick Experiment: Single seed, fewer epochs.
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

# QUICK CONFIG
CONFIG = {
    'epochs': 15,  # Reduced
    'embedding_dim': 50,  # Reduced
    'batch_size': 4096,  # Increased for speed
    'lr': 0.001,
    'kl_weight': 0.01,
    'seeds': [42],  # Single seed
}


def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_yago():
    cache_dir = os.path.expanduser("~/.kg_cache/yago")
    train = load_triples(os.path.join(cache_dir, "train.txt"))
    test = load_triples(os.path.join(cache_dir, "test.txt"))

    entities = set()
    relations = set()
    for h, r, t in train + test:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    return {'train': train, 'test': test, 'entities': list(entities), 'relations': list(relations)}


class CoverageOnlyDetector:
    def __init__(self, num_entities, num_relations):
        self.coverage = torch.zeros(num_entities, num_relations)

    def fit(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            self.coverage[entity_to_idx[h], relation_to_idx[r]] = 1.0
            self.coverage[entity_to_idx[t], relation_to_idx[r]] = 1.0
        return self

    def get_uncertainty(self, heads, relations, tails):
        return 2.0 - self.coverage[heads, relations] - self.coverage[tails, relations]


class VanillaGPKGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.num_entities = num_entities
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, heads, relations, tails, use_sampling=True):
        if use_sampling and self.training:
            h = self.entity_mean[heads] + torch.exp(0.5 * self.entity_logvar[heads]) * torch.randn_like(self.entity_mean[heads])
            t = self.entity_mean[tails] + torch.exp(0.5 * self.entity_logvar[tails]) * torch.randn_like(self.entity_mean[tails])
        else:
            h, t = self.entity_mean[heads], self.entity_mean[tails]
        return (h * self.relation_emb(relations) * t).sum(dim=-1)

    def get_uncertainty(self, heads, relations, tails):
        return (torch.exp(self.entity_logvar[heads]).mean(-1) + torch.exp(self.entity_logvar[tails]).mean(-1)) / 2

    def kl_loss(self):
        return -0.5 * torch.sum(1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()) / self.num_entities


class CAGP(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.num_entities = num_entities
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, heads, relations, tails, use_sampling=True):
        if use_sampling and self.training:
            h = self.entity_mean[heads] + torch.exp(0.5 * self.entity_logvar[heads]) * torch.randn_like(self.entity_mean[heads])
            t = self.entity_mean[tails] + torch.exp(0.5 * self.entity_logvar[tails]) * torch.randn_like(self.entity_mean[tails])
        else:
            h, t = self.entity_mean[heads], self.entity_mean[tails]
        return (h * self.relation_emb(relations) * t).sum(dim=-1)

    def get_uncertainty(self, heads, relations, tails):
        gp_var = (torch.exp(self.entity_logvar[heads]).mean(-1) + torch.exp(self.entity_logvar[tails]).mean(-1)) / 2
        cov_unc = 2.0 - self.coverage[heads, relations] - self.coverage[tails, relations]
        gp_var_norm = gp_var / (gp_var.mean() + 1e-8) * (cov_unc.mean() + 1e-8)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * gp_var_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            self.coverage[entity_to_idx[h], relation_to_idx[r]] = 1.0
            self.coverage[entity_to_idx[t], relation_to_idx[r]] = 1.0

    def kl_loss(self):
        return -0.5 * torch.sum(1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()) / self.num_entities

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit).item()


def train_model(model, triples, ent2idx, rel2idx, epochs, is_gp=False):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss()

    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(triples, ent2idx, rel2idx)

    heads = torch.tensor([ent2idx[h] for h, r, t in triples])
    relations = torch.tensor([rel2idx[r] for h, r, t in triples])
    tails = torch.tensor([ent2idx[t] for h, r, t in triples])
    loader = DataLoader(TensorDataset(heads, relations, tails), batch_size=CONFIG['batch_size'], shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_h, batch_r, batch_t in loader:
            batch_h, batch_r, batch_t = batch_h.to(device), batch_r.to(device), batch_t.to(device)
            pos = model(batch_h, batch_r, batch_t, use_sampling=is_gp)
            neg = model(batch_h, batch_r, torch.randint(0, len(ent2idx), batch_t.shape, device=device), use_sampling=is_gp)
            loss = criterion(pos, torch.ones_like(pos)) + criterion(neg, torch.zeros_like(neg))
            if is_gp:
                loss += CONFIG['kl_weight'] * model.kl_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{epochs}")
    return model


def evaluate(model, test, ent2idx, rel2idx, is_cov=False):
    if not is_cov:
        model.eval()
    heads = torch.tensor([ent2idx.get(h, 0) for h, r, t in test])
    relations = torch.tensor([rel2idx.get(r, 0) for h, r, t in test])
    tails = torch.tensor([ent2idx.get(t, 0) for h, r, t in test])
    if not is_cov:
        heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)
    with torch.no_grad():
        id_unc = model.get_uncertainty(heads, relations, tails)
        neg_tails = torch.randint(0, len(ent2idx), tails.shape)
        if not is_cov:
            neg_tails = neg_tails.to(device)
        ood_unc = model.get_uncertainty(heads, relations, neg_tails)
        if not is_cov:
            id_unc, ood_unc = id_unc.cpu(), ood_unc.cpu()
    labels = np.concatenate([np.ones(len(id_unc)), np.zeros(len(ood_unc))])
    scores = np.concatenate([-id_unc.numpy(), -ood_unc.numpy()])
    return roc_auc_score(labels, scores)


def main():
    print("Loading YAGO3-10...")
    data = load_yago()
    print(f"Train: {len(data['train'])}, Test: {len(data['test'])}")
    print(f"Entities: {len(data['entities'])}, Relations: {len(data['relations'])}")

    ent2idx = {e: i for i, e in enumerate(data['entities'])}
    rel2idx = {r: i for i, r in enumerate(data['relations'])}

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("\n1. Coverage Only...")
    cov = CoverageOnlyDetector(len(ent2idx), len(rel2idx))
    cov.fit(data['train'], ent2idx, rel2idx)
    auroc_cov = evaluate(cov, data['test'], ent2idx, rel2idx, is_cov=True)
    print(f"   AUROC: {auroc_cov:.4f}")

    print("\n2. VanillaGPKGE...")
    gp = VanillaGPKGE(len(ent2idx), len(rel2idx), CONFIG['embedding_dim'])
    gp = train_model(gp, data['train'], ent2idx, rel2idx, CONFIG['epochs'], is_gp=True)
    auroc_gp = evaluate(gp, data['test'], ent2idx, rel2idx)
    print(f"   AUROC: {auroc_gp:.4f}")

    print("\n3. CAGP...")
    cagp = CAGP(len(ent2idx), len(rel2idx), CONFIG['embedding_dim'])
    cagp = train_model(cagp, data['train'], ent2idx, rel2idx, CONFIG['epochs'], is_gp=True)
    auroc_cagp = evaluate(cagp, data['test'], ent2idx, rel2idx)
    alpha = cagp.get_alpha()
    print(f"   AUROC: {auroc_cagp:.4f} (alpha={alpha:.3f})")

    print("\n" + "="*60)
    print("YAGO3-10 RESULTS (37 relations)")
    print("="*60)
    print(f"{'Method':<20} {'AUROC':<10}")
    print("-"*30)
    print(f"{'CoverageOnly':<20} {auroc_cov:.4f}")
    print(f"{'VanillaGPKGE':<20} {auroc_gp:.4f}")
    print(f"{'CAGP':<20} {auroc_cagp:.4f}")

    best_single = max(auroc_cov, auroc_gp)
    synergy = auroc_cagp - best_single
    print(f"\nSynergy: {synergy:+.4f} ({synergy/best_single*100:+.1f}%)")

    if synergy > 0.05:
        print("** SYNERGY CONFIRMED **")
    elif synergy > 0.02:
        print("** MODERATE SYNERGY **")
    else:
        print("** WEAK/NO SYNERGY **")

    # Save
    output = {
        'dataset': 'YAGO3-10',
        'num_relations': len(data['relations']),
        'results': {
            'CoverageOnly': auroc_cov,
            'VanillaGPKGE': auroc_gp,
            'CAGP': auroc_cagp
        },
        'alpha': alpha,
        'synergy': synergy
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/yago_quick_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to outputs/yago_quick_results.json")


if __name__ == "__main__":
    main()
