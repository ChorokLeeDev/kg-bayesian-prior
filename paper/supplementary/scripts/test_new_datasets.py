#!/usr/bin/env python3
"""Test Coverage Paradox on newly downloaded datasets (OpenBioLink, NELL-995)."""
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

def load_openbiolink(base_path, max_train=200000):
    """Load OpenBioLink dataset."""
    train_path = os.path.join(base_path, 'HQ_DIR/train_test_data/train_sample.csv')
    test_path = os.path.join(base_path, 'HQ_DIR/train_test_data/test_sample.csv')

    entities = {}
    relations = {}

    def load_csv(path, max_lines=None):
        triples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entities: entities[h] = len(entities)
                    if t not in entities: entities[t] = len(entities)
                    if r not in relations: relations[r] = len(relations)
                    triples.append((entities[h], relations[r], entities[t]))
        return triples

    train = load_csv(train_path, max_train)
    test = load_csv(test_path, 20000)

    return train, test, len(entities), len(relations)

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

    # Quick train
    print("\nTraining (15 epochs)...")
    model = ComplEx(n_ent, n_rel, dim=50).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train_t = torch.tensor(train[:100000], dtype=torch.long, device=device)

    for ep in range(15):
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

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    results = {'full': [], 'partial': [], 'zero': []}
    test_sample = test[:3000]

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

    if results['full'] and results['partial']:
        f_acc = sum(results['full']) / len(results['full'])
        p_acc = sum(results['partial']) / len(results['partial'])
        if p_acc > f_acc:
            print(f"\n>>> PARADOX: Partial ({100*p_acc:.1f}%) > Full ({100*f_acc:.1f}%)")
        else:
            print(f"\n>>> NORMAL: Full ({100*f_acc:.1f}%) >= Partial ({100*p_acc:.1f}%)")

    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    base = './data/raw'

    # OpenBioLink
    print("\n" + "="*60)
    print("OPENBIOLINK (Biomedical)")
    print("="*60)
    try:
        train, test, n_ent, n_rel = load_openbiolink(os.path.join(base, 'openbiolink'))
        quick_test('OpenBioLink', train, test, n_ent, n_rel, device)
    except Exception as e:
        print(f"Error: {e}")

    # DRKG (already converted)
    print("\n" + "="*60)
    print("DRKG (Biomedical - converted)")
    print("="*60)
    drkg_path = os.path.join(base, 'drkg')
    if os.path.exists(os.path.join(drkg_path, 'train.txt')):
        try:
            entities = {}
            relations = {}

            def load_txt(path, max_lines=200000):
                triples = []
                with open(path) as f:
                    for i, line in enumerate(f):
                        if i >= max_lines: break
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]
                            if h not in entities: entities[h] = len(entities)
                            if t not in entities: entities[t] = len(entities)
                            if r not in relations: relations[r] = len(relations)
                            triples.append((entities[h], relations[r], entities[t]))
                return triples

            train = load_txt(os.path.join(drkg_path, 'train.txt'))
            test = load_txt(os.path.join(drkg_path, 'test.txt'), 20000)
            quick_test('DRKG-converted', train, test, len(entities), len(relations), device)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
