#!/usr/bin/env python3
"""Test Coverage Paradox on new temporal datasets (WIKI, YAGO temporal)."""
import sys
import os
import numpy as np
import torch
import torch.nn as nn

def load_temporal_dataset(base_path, name):
    """Load temporal dataset from RE-Net format."""
    path = os.path.join(base_path, name)

    # Load triples (format: h r t time time) - IDs are already numeric
    def load_triples(filename):
        triples = []
        max_e = 0
        max_r = 0
        with open(os.path.join(path, filename)) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, r, t))
                    max_e = max(max_e, h, t)
                    max_r = max(max_r, r)
        return triples, max_e + 1, max_r + 1

    train, _, _ = load_triples('train.txt')
    test, max_e, max_r = load_triples('test.txt')

    # Get actual entity/relation counts from train as well
    for h, r, t in train:
        max_e = max(max_e, h + 1, t + 1)
        max_r = max(max_r, r + 1)

    return train, test, max_e, max_r

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
    """Compute (entity, relation) coverage from training data."""
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))
    return coverage

def train_model(model, train, n_ent, epochs=50, batch_size=1024, device='cpu'):
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
        for h, r, t in test:
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = './data/raw/temporal/renet_data/data'

    datasets = ['WIKI', 'YAGO']

    for name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print('='*50)

        try:
            train, test, n_ent, n_rel = load_temporal_dataset(base_path, name)
            print(f"Entities: {n_ent}, Relations: {n_rel}")
            print(f"Train: {len(train)}, Test: {len(test)}")

            coverage = compute_coverage(train, n_ent, n_rel)

            # Count coverage distribution
            full = partial = zero = 0
            for h, r, t in test:
                h_cov = (h, r) in coverage
                t_cov = (t, r) in coverage
                if h_cov and t_cov: full += 1
                elif h_cov or t_cov: partial += 1
                else: zero += 1

            total = full + partial + zero
            print(f"\nCoverage distribution:")
            print(f"  Full: {full} ({100*full/total:.1f}%)")
            print(f"  Partial: {partial} ({100*partial/total:.1f}%)")
            print(f"  Zero: {zero} ({100*zero/total:.1f}%)")

            # Train
            print(f"\nTraining (quick, 30 epochs)...")
            model = ComplEx(n_ent, n_rel, dim=100)
            model = train_model(model, train, n_ent, epochs=30, device=device)

            # Evaluate
            print("\nEvaluating...")
            results = evaluate_by_coverage(model, test, coverage, n_ent, k=10, device=device)

            print(f"\n{name} RESULTS")
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

        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
