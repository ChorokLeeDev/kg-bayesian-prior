#!/usr/bin/env python3
"""Minimal baseline + coverage test (5 epochs)."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from collections import defaultdict

from src.data.loaders import load_wn18rr

print("Loading data...")
train_ds, _, test_ds = load_wn18rr()
train = train_ds.triples
test = test_ds.triples
n_ent, n_rel = train_ds.num_entities, train_ds.num_relations

print(f"Entities: {n_ent}, Relations: {n_rel}")
print(f"Train: {len(train)}, Test: {len(test)}\n")

device = torch.device('cpu')

class SimpleBaseline(nn.Module):
    def __init__(self, ne, nr, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(ne, dim)
        self.relation_emb = nn.Embedding(nr, dim)
        self.num_entities = ne
        self.num_relations = nr
        self.register_buffer('coverage', torch.zeros(ne, nr))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_unc(self, h, r, t):
        return -self.forward(h, r, t)

    def precompute_cov(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0

print("Training baseline (5 epochs)...")
torch.manual_seed(42)
np.random.seed(42)

model = SimpleBaseline(n_ent, n_rel).to(device)
model.precompute_cov(train)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

heads = torch.tensor(train[:, 0])
rels = torch.tensor(train[:, 1])
tails = torch.tensor(train[:, 2])
loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=512, shuffle=True)

for epoch in range(5):
    for h, r, t in loader:
        h, r, t = h.to(device), r.to(device), t.to(device)
        pos = model(h, r, t)
        neg_t = torch.randint(0, n_ent, t.shape, device=device)
        neg = model(h, r, neg_t)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
               F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}/5")

print("\nComputing uncertainties...")
model.eval()
with torch.no_grad():
    h = torch.tensor(test[:, 0]).to(device)
    r = torch.tensor(test[:, 1]).to(device)
    t = torch.tensor(test[:, 2]).to(device)
    base_unc = model.get_unc(h, r, t).cpu().numpy()

# Coverage uncertainty
cov = model.coverage.cpu().numpy()
cov_unc = np.zeros(len(test))
for i in range(len(test)):
    cov_unc[i] = 2.0 - cov[test[i, 0], test[i, 1]] - cov[test[i, 2], test[i, 1]]

# Normalize and combine
base_norm = (base_unc - base_unc.mean()) / (base_unc.std() + 1e-8)
base_norm = base_norm * cov_unc.std() + cov_unc.mean()
combined = 0.5 * base_norm + 0.5 * cov_unc

# Evaluate temporal OOD
freq = defaultdict(int)
for i in range(len(train)):
    freq[train[i, 0]] += 1
    freq[train[i, 2]] += 1
thresh = np.percentile(list(freq.values()), 25)

id_idx, ood_idx = [], []
for i in range(len(test)):
    h_i, r_i, t_i = test[i]
    is_emerging = freq.get(h_i, 0) <= thresh or freq.get(t_i, 0) <= thresh
    has_cov = cov[h_i, r_i] > 0 and cov[t_i, r_i] > 0
    
    if is_emerging or not has_cov:
        ood_idx.append(i)
    else:
        id_idx.append(i)

print(f"ID: {len(id_idx)}, OOD: {len(ood_idx)}\n")

def eval_auroc(unc_scores):
    id_unc = unc_scores[id_idx]
    ood_unc = unc_scores[ood_idx]
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

base_auroc = eval_auroc(base_unc)
cov_auroc = eval_auroc(cov_unc)
comb_auroc = eval_auroc(combined)

print(f"Results (WN18RR, seed=42, 5 epochs):")
print(f"  Baseline:         {base_auroc:.4f}")
print(f"  Coverage:         {cov_auroc:.4f}")
print(f"  Baseline+Coverage:{comb_auroc:.4f}")
print(f"  Improvement:      {comb_auroc - base_auroc:+.4f}")

results = {
    'dataset': 'WN18RR',
    'epochs': 5,
    'seed': 42,
    'baseline_auroc': float(base_auroc),
    'coverage_auroc': float(cov_auroc),
    'combined_auroc': float(comb_auroc),
    'improvement': float(comb_auroc - base_auroc),
}

output_dir = Path("/sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior/outputs")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "baseline_coverage_minimal.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {output_dir / 'baseline_coverage_minimal.json'}")
