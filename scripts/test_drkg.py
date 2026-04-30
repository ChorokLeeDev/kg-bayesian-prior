#!/usr/bin/env python3
"""Test Coverage Paradox on DRKG biomedical dataset."""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

def load_drkg(path, max_triples=500000):
    """Load DRKG dataset (sample if too large)."""
    print("Loading DRKG...")

    # Read all triples
    triples = []
    entities = set()
    relations = set()

    with open(os.path.join(path, 'drkg.tsv')) as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  Read {i/1e6:.1f}M lines...")
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                entities.add(h)
                entities.add(t)
                relations.add(r)
                triples.append((h, r, t))

    print(f"Total: {len(triples)} triples, {len(entities)} entities, {len(relations)} relations")

    # Create mappings
    e2id = {e: i for i, e in enumerate(sorted(entities))}
    r2id = {r: i for i, r in enumerate(sorted(relations))}

    # Convert to IDs
    triples_id = [(e2id[h], r2id[r], e2id[t]) for h, r, t in triples]

    # Sample if needed
    if len(triples_id) > max_triples:
        print(f"Sampling {max_triples} triples...")
        np.random.seed(42)
        idx = np.random.choice(len(triples_id), max_triples, replace=False)
        triples_id = [triples_id[i] for i in idx]

    # Split 80/10/10
    np.random.seed(42)
    np.random.shuffle(triples_id)
    n = len(triples_id)
    train = triples_id[:int(0.8*n)]
    test = triples_id[int(0.9*n):]

    return train, test, len(e2id), len(r2id)

class ComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent_re = nn.Embedding(n_ent, dim)
        self.ent_im = nn.Embedding(n_ent, dim)
        self.rel_re = nn.Embedding(n_rel, dim)
        self.rel_im = nn.Embedding(n_rel, dim)
        for emb in [self.ent_re, self.ent_im, self.rel_re, self.rel_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, h, r, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        return (h_re*r_re*t_re + h_re*r_im*t_im + h_im*r_re*t_im - h_im*r_im*t_re).sum(-1)

    def score_all_tails(self, h, r):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        hr_re = h_re*r_re - h_im*r_im
        hr_im = h_re*r_im + h_im*r_re
        return hr_re @ self.ent_re.weight.t() + hr_im @ self.ent_im.weight.t()

def compute_coverage(train, n_ent, n_rel):
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))
    return coverage

def train_model(model, train, n_ent, epochs=30, batch_size=2048, device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train_t = torch.tensor(train, dtype=torch.long, device=device)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train), device=device)
        total_loss = 0

        for i in range(0, len(train), batch_size):
            idx = perm[i:i+batch_size]
            batch = train_t[idx]
            h, r, t = batch[:,0], batch[:,1], batch[:,2]

            pos = model(h, r, t)
            neg_t = torch.randint(0, n_ent, (len(batch), 10), device=device)
            neg = model(h.unsqueeze(1).expand(-1,10).reshape(-1),
                       r.unsqueeze(1).expand(-1,10).reshape(-1),
                       neg_t.reshape(-1)).view(-1, 10)

            loss = torch.relu(1.0 - pos.unsqueeze(1) + neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}: {total_loss:.2f}")

    return model

def evaluate_by_coverage(model, test, coverage, n_ent, k=10, device='cpu'):
    model.eval()
    results = {'full': [], 'partial': [], 'zero': []}

    with torch.no_grad():
        for h, r, t in test[:5000]:  # Sample for speed
            h_cov = (h, r) in coverage
            t_cov = (t, r) in coverage

            if h_cov and t_cov:
                cov_type = 'full'
            elif h_cov or t_cov:
                cov_type = 'partial'
            else:
                cov_type = 'zero'

            h_t = torch.tensor([h], device=device)
            r_t = torch.tensor([r], device=device)
            scores = model.score_all_tails(h_t, r_t)[0]
            rank = (scores > scores[t]).sum().item() + 1
            results[cov_type].append(1 if rank <= k else 0)

    return results

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    drkg_path = '/Users/i767700/Github/kg-bayesian-prior/data/raw/biomedical'

    print("="*50)
    print("DRKG (Biomedical Knowledge Graph)")
    print("="*50)

    train, test, n_ent, n_rel = load_drkg(drkg_path, max_triples=500000)
    print(f"\nUsing: {len(train)} train, {len(test)} test")
    print(f"Entities: {n_ent}, Relations: {n_rel}")

    coverage = compute_coverage(train, n_ent, n_rel)

    # Coverage distribution
    full = partial = zero = 0
    for h, r, t in test[:5000]:
        h_cov = (h, r) in coverage
        t_cov = (t, r) in coverage
        if h_cov and t_cov: full += 1
        elif h_cov or t_cov: partial += 1
        else: zero += 1

    total = full + partial + zero
    print(f"\nCoverage distribution (test sample):")
    print(f"  Full: {full} ({100*full/total:.1f}%)")
    print(f"  Partial: {partial} ({100*partial/total:.1f}%)")
    print(f"  Zero: {zero} ({100*zero/total:.1f}%)")

    # Train
    print(f"\nTraining (30 epochs)...")
    model = ComplEx(n_ent, n_rel, dim=100)
    model = train_model(model, train, n_ent, epochs=30, device=device)

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_by_coverage(model, test, coverage, n_ent, k=10, device=device)

    print(f"\nDRKG RESULTS")
    print("-"*40)

    for cov_type in ['full', 'partial', 'zero']:
        hits = results[cov_type]
        if hits:
            acc = 100 * sum(hits) / len(hits)
            err = 100 - acc
            print(f"{cov_type.capitalize():8}: {acc:5.1f}% Hits@10 (Err: {err:.1f}%)")

    # Check paradox
    full_hits = results['full']
    partial_hits = results['partial']
    if full_hits and partial_hits:
        full_acc = sum(full_hits) / len(full_hits)
        partial_acc = sum(partial_hits) / len(partial_hits)

        if partial_acc > full_acc:
            print(f"\n>>> PARADOX: Partial ({100*partial_acc:.1f}%) > Full ({100*full_acc:.1f}%)")
        else:
            print(f"\n>>> NORMAL: Full ({100*full_acc:.1f}%) >= Partial ({100*partial_acc:.1f}%)")

if __name__ == '__main__':
    main()
