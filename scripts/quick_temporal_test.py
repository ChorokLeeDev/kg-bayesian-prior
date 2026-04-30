#!/usr/bin/env python3
"""Quick test Coverage Paradox on WIKI and YAGO temporal datasets."""
import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Flush output
import functools
print = functools.partial(print, flush=True)

class ComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, dim=50):  # smaller dim for speed
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

def load_triples(path, filename, max_triples=None):
    triples = []
    max_e = max_r = 0
    with open(os.path.join(path, filename)) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, r, t))
                max_e = max(max_e, h, t)
                max_r = max(max_r, r)
                if max_triples and len(triples) >= max_triples:
                    break
    return triples, max_e + 1, max_r + 1

def quick_test(name, train, test, n_ent, n_rel, device='cpu'):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Coverage
    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    # Count coverage types
    full = partial = zero = 0
    for h, r, t in test:
        h_cov = (h, r) in coverage
        t_cov = (t, r) in coverage
        if h_cov and t_cov: full += 1
        elif h_cov or t_cov: partial += 1
        else: zero += 1

    total = full + partial + zero
    print(f"\nCoverage: Full={full} ({100*full/total:.1f}%), Partial={partial} ({100*partial/total:.1f}%), Zero={zero} ({100*zero/total:.1f}%)")

    # Quick train (10 epochs)
    print("\nTraining (10 epochs)...")
    model = ComplEx(n_ent, n_rel, dim=50).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train_t = torch.tensor(train[:100000], dtype=torch.long, device=device)  # limit for speed

    for ep in range(10):
        model.train()
        perm = torch.randperm(len(train_t), device=device)
        total_loss = 0
        for i in range(0, len(train_t), 2048):
            batch = train_t[perm[i:i+2048]]
            h, r, t = batch[:,0], batch[:,1], batch[:,2]
            pos = model(h, r, t)
            neg_t = torch.randint(0, n_ent, (len(batch), 5), device=device)
            neg = model(h.unsqueeze(1).expand(-1,5).reshape(-1),
                       r.unsqueeze(1).expand(-1,5).reshape(-1),
                       neg_t.reshape(-1)).view(-1, 5)
            loss = torch.relu(1.0 - pos.unsqueeze(1) + neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Ep {ep+1}: {total_loss:.2f}")

    # Evaluate (sample 1000 test)
    print("\nEvaluating...")
    model.eval()
    results = {'full': [], 'partial': [], 'zero': []}
    test_sample = test[:2000]

    with torch.no_grad():
        for h, r, t in test_sample:
            h_cov = (h, r) in coverage
            t_cov = (t, r) in coverage
            cov_type = 'full' if (h_cov and t_cov) else ('partial' if (h_cov or t_cov) else 'zero')

            scores = model.score_all_tails(
                torch.tensor([h], device=device),
                torch.tensor([r], device=device)
            )[0]
            rank = (scores > scores[t]).sum().item() + 1
            results[cov_type].append(1 if rank <= 10 else 0)

    print(f"\n{name} RESULTS")
    print("-"*40)
    for ct in ['full', 'partial', 'zero']:
        if results[ct]:
            acc = 100 * sum(results[ct]) / len(results[ct])
            print(f"{ct.capitalize():8}: {acc:5.1f}% Hits@10 (n={len(results[ct])})")

    # Paradox check
    if results['full'] and results['partial']:
        f_acc = sum(results['full']) / len(results['full'])
        p_acc = sum(results['partial']) / len(results['partial'])
        if p_acc > f_acc:
            print(f"\n>>> PARADOX: Partial > Full")
        else:
            print(f"\n>>> NORMAL: Full >= Partial")

    return results

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    base = '/Users/i767700/Github/kg-bayesian-prior/data/raw/temporal/renet_data/data'

    for name in ['WIKI', 'YAGO']:
        path = os.path.join(base, name)
        train, n_e1, n_r1 = load_triples(path, 'train.txt')
        test, n_e2, n_r2 = load_triples(path, 'test.txt')
        n_ent = max(n_e1, n_e2)
        n_rel = max(n_r1, n_r2)
        quick_test(name, train, test, n_ent, n_rel, device)

if __name__ == '__main__':
    main()
