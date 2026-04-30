#!/usr/bin/env python3
"""Test Coverage Paradox across multiple KGE models: ComplEx, TransE, DistMult, RotatE."""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import functools
print = functools.partial(print, flush=True)

# ============================================================
# Models
# ============================================================

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


class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, r, t):
        h_emb = self.ent(h)
        r_emb = self.rel(r)
        t_emb = self.ent(t)
        # Negative L2 distance (higher = better)
        return -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

    def score_all_tails(self, h, r):
        h_emb = self.ent(h)  # (batch, dim)
        r_emb = self.rel(r)  # (batch, dim)
        hr = h_emb + r_emb   # (batch, dim)
        # Score all tails: -||hr - t||^2
        all_t = self.ent.weight  # (n_ent, dim)
        # (batch, 1, dim) - (1, n_ent, dim) -> (batch, n_ent, dim)
        diff = hr.unsqueeze(1) - all_t.unsqueeze(0)
        return -torch.norm(diff, p=2, dim=-1)  # (batch, n_ent)


class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, r, t):
        h_emb = self.ent(h)
        r_emb = self.rel(r)
        t_emb = self.ent(t)
        return (h_emb * r_emb * t_emb).sum(-1)

    def score_all_tails(self, h, r):
        h_emb = self.ent(h)
        r_emb = self.rel(r)
        hr = h_emb * r_emb
        return hr @ self.ent.weight.t()


class RotatE(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100, gamma=12.0):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.ent_re = nn.Embedding(n_ent, dim)
        self.ent_im = nn.Embedding(n_ent, dim)
        self.rel_phase = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)
        nn.init.uniform_(self.rel_phase.weight, -np.pi, np.pi)

    def forward(self, h, r, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        phase = self.rel_phase(r)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        # h * r (complex multiplication)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # ||h*r - t||
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        dist = torch.sqrt(diff_re**2 + diff_im**2 + 1e-9).sum(-1)
        return self.gamma - dist

    def score_all_tails(self, h, r):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        phase = self.rel_phase(r)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # Score all tails
        all_t_re = self.ent_re.weight
        all_t_im = self.ent_im.weight
        # (batch, 1, dim) - (1, n_ent, dim)
        diff_re = hr_re.unsqueeze(1) - all_t_re.unsqueeze(0)
        diff_im = hr_im.unsqueeze(1) - all_t_im.unsqueeze(0)
        dist = torch.sqrt(diff_re**2 + diff_im**2 + 1e-9).sum(-1)
        return self.gamma - dist


# ============================================================
# Data Loading
# ============================================================

def load_dataset(name, base_path):
    """Load FB15k-237 or WN18RR."""
    path = os.path.join(base_path, name)

    entities = {}
    relations = {}

    def load_file(filename):
        triples = []
        with open(os.path.join(path, filename)) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entities: entities[h] = len(entities)
                    if t not in entities: entities[t] = len(entities)
                    if r not in relations: relations[r] = len(relations)
                    triples.append((entities[h], relations[r], entities[t]))
        return triples

    train = load_file('train.txt')
    test = load_file('test.txt')

    return train, test, len(entities), len(relations)


# ============================================================
# Training & Evaluation
# ============================================================

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
            print(f"    Epoch {ep+1}: loss={total_loss:.2f}")

    return model


def evaluate_by_coverage(model, test, coverage, n_ent, k=10, device='cpu', max_test=3000):
    model.eval()
    results = {'full': [], 'partial': [], 'zero': []}

    test_sample = test[:max_test]

    with torch.no_grad():
        for h, r, t in test_sample:
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


def compute_auroc(results):
    """Compute AUROC for coverage predicting correctness."""
    # Full=1, Partial=0 as coverage indicator
    # If paradox: partial > full, so AUROC < 0.5
    y_true = []
    y_score = []

    for hit in results['full']:
        y_true.append(hit)
        y_score.append(1)  # full coverage

    for hit in results['partial']:
        y_true.append(hit)
        y_score.append(0)  # partial coverage

    if len(set(y_true)) < 2:
        return 0.5

    # Simple AUROC calculation
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs
    concordant = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if y_true[i] > y_true[j] and y_score[i] > y_score[j]:
                concordant += 1
            elif y_true[i] > y_true[j] and y_score[i] < y_score[j]:
                concordant -= 1

    auroc = 0.5 + concordant / (2 * n_pos * n_neg)
    return auroc


# ============================================================
# Main
# ============================================================

def main():
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Started: {datetime.now()}")

    base_path = '/Users/i767700/Github/kg-bayesian-prior/data/raw'

    datasets = ['FB15k-237', 'WN18RR']
    model_classes = {
        'ComplEx': ComplEx,
        'TransE': TransE,
        'DistMult': DistMult,
        'RotatE': RotatE,
    }

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)

        train, test, n_ent, n_rel = load_dataset(dataset_name, base_path)
        print(f"Entities: {n_ent}, Relations: {n_rel}")
        print(f"Train: {len(train)}, Test: {len(test)}")

        # Compute coverage
        coverage = set()
        for h, r, t in train:
            coverage.add((h, r))
            coverage.add((t, r))

        # Count coverage distribution
        full = partial = zero = 0
        for h, r, t in test:
            h_cov = (h, r) in coverage
            t_cov = (t, r) in coverage
            if h_cov and t_cov: full += 1
            elif h_cov or t_cov: partial += 1
            else: zero += 1

        total = full + partial + zero
        print(f"Coverage dist: Full={full} ({100*full/total:.1f}%), Partial={partial} ({100*partial/total:.1f}%), Zero={zero} ({100*zero/total:.1f}%)")

        all_results[dataset_name] = {}

        for model_name, ModelClass in model_classes.items():
            print(f"\n  --- {model_name} ---")

            model = ModelClass(n_ent, n_rel, dim=100)
            model = train_model(model, train, n_ent, epochs=50, device=device)

            results = evaluate_by_coverage(model, test, coverage, n_ent, device=device)

            # Compute metrics
            full_acc = 100 * sum(results['full']) / len(results['full']) if results['full'] else 0
            partial_acc = 100 * sum(results['partial']) / len(results['partial']) if results['partial'] else 0
            zero_acc = 100 * sum(results['zero']) / len(results['zero']) if results['zero'] else 0

            auroc = compute_auroc(results)
            paradox = partial_acc > full_acc

            print(f"    Full:    {full_acc:5.1f}% (n={len(results['full'])})")
            print(f"    Partial: {partial_acc:5.1f}% (n={len(results['partial'])})")
            print(f"    Zero:    {zero_acc:5.1f}% (n={len(results['zero'])})")
            print(f"    AUROC:   {auroc:.3f}")
            print(f"    Paradox: {'YES' if paradox else 'NO'}")

            all_results[dataset_name][model_name] = {
                'full': full_acc,
                'partial': partial_acc,
                'zero': zero_acc,
                'auroc': auroc,
                'paradox': paradox,
                'n_full': len(results['full']),
                'n_partial': len(results['partial']),
                'n_zero': len(results['zero']),
            }

    # Save results
    output_path = '/Users/i767700/Github/kg-bayesian-prior/outputs/multi_model_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Coverage Paradox Across Models")
    print("="*80)
    print(f"{'Dataset':<12} {'Model':<10} {'Full':<8} {'Partial':<8} {'AUROC':<8} {'Paradox':<8}")
    print("-"*80)

    for dataset_name in datasets:
        for model_name in model_classes.keys():
            r = all_results[dataset_name][model_name]
            print(f"{dataset_name:<12} {model_name:<10} {r['full']:>6.1f}% {r['partial']:>6.1f}% {r['auroc']:>6.3f}   {'YES' if r['paradox'] else 'NO'}")
        print()

    print(f"\nFinished: {datetime.now()}")


if __name__ == '__main__':
    main()
