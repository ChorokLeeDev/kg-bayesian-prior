#!/usr/bin/env python3
"""Verify Definition 2.3 on WN18RR (in addition to FB15k-237)."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_wn18rr

print("="*60)
print("Definition 2.3 Verification on WN18RR")
print("="*60)

# Load data
train_ds, valid_ds, test_ds = load_wn18rr()
train_triples = train_ds.triples
num_entities = train_ds.num_entities
num_relations = train_ds.num_relations

print(f"\nWN18RR: {num_entities} entities, {num_relations} relations")

# Build coverage and frequency
coverage = np.zeros((num_entities, num_relations))
freq = np.zeros(num_entities)
for h, r, t in train_triples:
    coverage[h, r] = 1
    coverage[t, r] = 1
    freq[h] += 1
    freq[t] += 1

# Train simple embedding model
class SimpleKGE(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, h, r, t):
        return (self.ent_emb(h) * self.rel_emb(r) * self.ent_emb(t)).sum(-1)

print("\nTraining DistMult embedding...")
model = SimpleKGE(num_entities, num_relations)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

train_h = torch.tensor(train_triples[:, 0])
train_r = torch.tensor(train_triples[:, 1])
train_t = torch.tensor(train_triples[:, 2])

for epoch in range(30):
    model.train()
    perm = torch.randperm(len(train_triples))
    total_loss = 0
    for i in range(0, len(train_triples), 256):
        idx = perm[i:i+256]
        h, r, t = train_h[idx], train_r[idx], train_t[idx]
        pos = model(h, r, t)
        neg_t = torch.randint(0, num_entities, (len(h),))
        neg = model(h, r, neg_t)
        loss = -torch.log(torch.sigmoid(pos) + 1e-6).mean() - torch.log(1 - torch.sigmoid(neg) + 1e-6).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss:.2f}")

# Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model.ent_emb.weight.numpy()

# Split relations: 80% train, 20% held-out
np.random.seed(42)
rel_perm = np.random.permutation(num_relations)
n_train_rel = int(0.8 * num_relations)
train_rels = set(rel_perm[:n_train_rel])
test_rels = set(rel_perm[n_train_rel:])

print(f"\nRelation split: {len(train_rels)} train, {len(test_rels)} held-out")

# Prepare data for coverage prediction
X_train, y_train = [], []
X_test, y_test = [], []

for e in range(num_entities):
    for r in range(num_relations):
        features = np.concatenate([embeddings[e], [np.log1p(freq[e])]])
        label = coverage[e, r]
        if r in train_rels:
            X_train.append(features)
            y_train.append(label)
        else:
            X_test.append(features)
            y_test.append(label)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Subsample for speed
n_sample = min(50000, len(X_train))
idx = np.random.choice(len(X_train), n_sample, replace=False)
X_train_sub, y_train_sub = X_train[idx], y_train[idx]

idx_test = np.random.choice(len(X_test), min(20000, len(X_test)), replace=False)
X_test_sub, y_test_sub = X_test[idx_test], y_test[idx_test]

# Frequency-only baseline
print("\nTraining coverage predictors...")
freq_only = LogisticRegression(max_iter=500, n_jobs=-1)
freq_only.fit(X_train_sub[:, -1:], y_train_sub)

emb_freq = LogisticRegression(max_iter=500, n_jobs=-1)
emb_freq.fit(X_train_sub, y_train_sub)

# Evaluate
freq_auc_train = roc_auc_score(y_train_sub, freq_only.predict_proba(X_train_sub[:, -1:])[:, 1])
freq_auc_test = roc_auc_score(y_test_sub, freq_only.predict_proba(X_test_sub[:, -1:])[:, 1])

emb_auc_train = roc_auc_score(y_train_sub, emb_freq.predict_proba(X_train_sub)[:, 1])
emb_auc_test = roc_auc_score(y_test_sub, emb_freq.predict_proba(X_test_sub)[:, 1])

print("\n" + "="*60)
print("RESULTS: Definition 2.3 Verification (WN18RR)")
print("="*60)
print(f"\n{'Predictor':<25} {'Seen Rels AUC':<15} {'Held-out Rels AUC'}")
print("-"*55)
print(f"{'Frequency only':<25} {freq_auc_train:<15.2f} {freq_auc_test:.2f}")
print(f"{'Embedding + Frequency':<25} {emb_auc_train:<15.2f} {emb_auc_test:.2f}")

delta = emb_auc_test - freq_auc_test
print(f"\nΔ AUC (held-out relations): {delta:+.2f}")
print(f"\n{'✓' if abs(delta) < 0.05 else '✗'} Definition 2.3 {'confirmed' if abs(delta) < 0.05 else 'violated'}: embeddings provide {'no' if abs(delta) < 0.05 else ''} additional coverage info beyond frequency")
