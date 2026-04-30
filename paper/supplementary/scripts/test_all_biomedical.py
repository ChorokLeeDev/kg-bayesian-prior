#!/usr/bin/env python3
"""Test Coverage Paradox on all newly downloaded biomedical datasets."""
import os
import numpy as np
import torch
import torch.nn as nn
import functools
print = functools.partial(print, flush=True)

class ComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, dim=50):
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

def load_tsv(path, max_train=150000, test_ratio=0.1):
    """Load dataset from all_triples.tsv format."""
    entities = {}
    relations = {}
    triples = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                if h not in entities: entities[h] = len(entities)
                if t not in entities: entities[t] = len(entities)
                if r not in relations: relations[r] = len(relations)
                triples.append((entities[h], relations[r], entities[t]))

    # Sample if too large
    if len(triples) > max_train / (1 - test_ratio):
        np.random.seed(42)
        idx = np.random.choice(len(triples), int(max_train / (1 - test_ratio)), replace=False)
        triples = [triples[i] for i in idx]

    # Split
    np.random.seed(42)
    np.random.shuffle(triples)
    split = int(len(triples) * (1 - test_ratio))
    train = triples[:split]
    test = triples[split:]

    return train, test, len(entities), len(relations)

def quick_test(name, train, test, n_ent, n_rel, device='cpu'):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    coverage = set()
    for h, r, t in train:
        coverage.add((h, r))
        coverage.add((t, r))

    full = partial = zero = 0
    for h, r, t in test:
        h_cov = (h, r) in coverage
        t_cov = (t, r) in coverage
        if h_cov and t_cov: full += 1
        elif h_cov or t_cov: partial += 1
        else: zero += 1

    total = full + partial + zero
    print(f"Coverage: Full={full} ({100*full/total:.1f}%), Partial={partial} ({100*partial/total:.1f}%), Zero={zero} ({100*zero/total:.1f}%)")

    print("Training (10 epochs)...")
    model = ComplEx(n_ent, n_rel, dim=50).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train_t = torch.tensor(train[:100000], dtype=torch.long, device=device)

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
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1}: {total_loss:.2f}")

    print("Evaluating...")
    model.eval()
    results = {'full': [], 'partial': [], 'zero': []}
    test_sample = test[:2000]

    with torch.no_grad():
        for h, r, t in test_sample:
            h_cov = (h, r) in coverage
            t_cov = (t, r) in coverage
            cov_type = 'full' if (h_cov and t_cov) else ('partial' if (h_cov or t_cov) else 'zero')
            scores = model.score_all_tails(torch.tensor([h], device=device), torch.tensor([r], device=device))[0]
            rank = (scores > scores[t]).sum().item() + 1
            results[cov_type].append(1 if rank <= 10 else 0)

    print(f"\n{name} RESULTS")
    print("-"*40)
    for ct in ['full', 'partial', 'zero']:
        if results[ct]:
            acc = 100 * sum(results[ct]) / len(results[ct])
            print(f"{ct.capitalize():8}: {acc:5.1f}% Hits@10 (n={len(results[ct])})")

    if results['full'] and results['partial']:
        f_acc = sum(results['full']) / len(results['full'])
        p_acc = sum(results['partial']) / len(results['partial'])
        if p_acc > f_acc:
            print(f">>> PARADOX: Partial > Full")
            return True
        else:
            print(f">>> NORMAL: Full >= Partial")
            return False
    return None

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    base = './data/raw'

    # Biomedical datasets to test
    bio_datasets = [
        ('primekg', 'PrimeKG'),
        ('hetionet', 'Hetionet'),
        ('pharmkg', 'PharmKG'),
    ]

    # Additional datasets
    other_datasets = [
        ('conceptnet', 'ConceptNet'),
        ('nell', 'NELL'),
        ('kinship', 'Kinship'),
        ('nations', 'Nations'),
    ]

    results = {}

    for folder, name in bio_datasets + other_datasets:
        path = os.path.join(base, folder)
        tsv_files = [f for f in os.listdir(path) if f.endswith('.tsv')] if os.path.exists(path) else []

        if not tsv_files:
            print(f"\nSkipping {name}: no .tsv file found")
            continue

        tsv_path = os.path.join(path, tsv_files[0])
        try:
            train, test, n_ent, n_rel = load_tsv(tsv_path)
            paradox = quick_test(name, train, test, n_ent, n_rel, device)
            results[name] = paradox
        except Exception as e:
            print(f"Error with {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, paradox in results.items():
        status = "PARADOX" if paradox else "NORMAL" if paradox is False else "N/A"
        print(f"  {name}: {status}")

if __name__ == '__main__':
    main()
