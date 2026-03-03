#!/usr/bin/env python3
"""
GDELT Temporal KG Pipeline for CAGP evaluation.

Downloads GDELT subset, trains CAGP + baselines, evaluates temporal OOD.
Uses pykeen's GDELT dataset or downloads from standard source.

Output: outputs/gdelt_results.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import gc
import time
import urllib.request
import os

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "gdelt"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "gdelt_results.json"
LOG_PATH = Path(__file__).parent.parent / "outputs" / "gdelt_pipeline.log"

EPOCHS = 30
SEEDS = [42, 123, 456]
DIM = 100
BATCH_SIZE = 512
LR = 0.001


def log(msg):
    """Log to both stdout and file."""
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def download_gdelt():
    """Download GDELT dataset. Try pykeen first, then manual download."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (DATA_DIR / "train.txt").exists():
        log("GDELT data already exists.")
        return True

    # Try pykeen
    try:
        log("Trying pykeen GDELT download...")
        from pykeen.datasets import GDELT as PykeenGDELT
        ds = PykeenGDELT(create_inverse_triples=False)

        # Extract triples
        for split_name, tf in [('train', ds.training), ('valid', ds.validation), ('test', ds.testing)]:
            triples = tf.mapped_triples.numpy()
            np.savetxt(DATA_DIR / f"{split_name}.txt", triples, fmt='%d', delimiter='\t')

        # Save stats
        n_ent = ds.num_entities
        n_rel = ds.num_relations
        with open(DATA_DIR / "stat.txt", 'w') as f:
            f.write(f"{n_ent}\t{n_rel}\n")

        log(f"GDELT downloaded via pykeen: {n_ent} entities, {n_rel} relations")
        log(f"  Train: {len(ds.training.mapped_triples)}, Valid: {len(ds.validation.mapped_triples)}, Test: {len(ds.testing.mapped_triples)}")
        return True

    except Exception as e:
        log(f"pykeen GDELT failed: {e}")

    # Try manual download from common sources
    try:
        log("Trying manual GDELT download...")
        base_url = "https://raw.githubusercontent.com/INK-USC/RE-Net/master/data/GDELT"
        for fname in ["train.txt", "valid.txt", "test.txt", "stat.txt"]:
            dest = DATA_DIR / fname
            if not dest.exists():
                url = f"{base_url}/{fname}"
                log(f"  Downloading {fname} from {url}...")
                urllib.request.urlretrieve(url, dest)
        log("GDELT downloaded from RE-Net repository.")
        return True
    except Exception as e:
        log(f"Manual download also failed: {e}")

    # Try yet another source
    try:
        log("Trying TiRGN repository...")
        base_url = "https://raw.githubusercontent.com/Liyyy2122/TiRGN/main/data/GDELT"
        for fname in ["train.txt", "valid.txt", "test.txt", "stat.txt"]:
            dest = DATA_DIR / fname
            if not dest.exists():
                url = f"{base_url}/{fname}"
                log(f"  Downloading {fname} from {url}...")
                urllib.request.urlretrieve(url, dest)
        log("GDELT downloaded from TiRGN repository.")
        return True
    except Exception as e:
        log(f"TiRGN download also failed: {e}")
        return False


def load_gdelt():
    """Load GDELT triples."""
    stat_path = DATA_DIR / "stat.txt"

    if stat_path.exists():
        with open(stat_path) as f:
            parts = f.readline().strip().split('\t')
            num_entities = int(parts[0])
            num_relations = int(parts[1])
    else:
        num_entities = 0
        num_relations = 0

    def load_triples(path):
        triples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    s, r, o = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append([s, r, o])
        return np.array(triples)

    train = load_triples(DATA_DIR / "train.txt")
    test = load_triples(DATA_DIR / "test.txt")

    # Update entity/relation counts
    all_t = np.concatenate([train, test])
    num_entities = max(num_entities, all_t[:, 0].max() + 1, all_t[:, 2].max() + 1)
    num_relations = max(num_relations, all_t[:, 1].max() + 1)

    log(f"GDELT loaded: {num_entities} entities, {num_relations} relations")
    log(f"  Train: {len(train)}, Test: {len(test)}")

    return {
        'train': torch.tensor(train, dtype=torch.long),
        'test': torch.tensor(test, dtype=torch.long),
        'num_entities': int(num_entities),
        'num_relations': int(num_relations),
    }


# ---- Models (same as other scripts) ----

class CoverageOnly(nn.Module):
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        return 2.0 - self.coverage[h, r] - self.coverage[t, r]

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0].item(), triples[i, 1].item(), triples[i, 2].item()
            if h < self.num_entities and r < self.num_relations:
                self.coverage[h, r] = 1
            if t < self.num_entities and r < self.num_relations:
                self.coverage[t, r] = 1


class CAGP(nn.Module):
    def __init__(self, num_entities, num_relations, dim=DIM, alpha=0.5):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.alpha = alpha
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 2.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        h_mean, t_mean = self.entity_mean[h], self.entity_mean[t]
        h_std = torch.exp(0.5 * self.entity_logvar[h])
        t_std = torch.exp(0.5 * self.entity_logvar[t])
        h_emb = h_mean + h_std * torch.randn_like(h_std) if self.training else h_mean
        t_emb = t_mean + t_std * torch.randn_like(t_std) if self.training else t_mean
        r_emb = self.relation_emb(r)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        sem = 0.5 * (self.entity_logvar[h].exp().mean(-1) + self.entity_logvar[t].exp().mean(-1))
        struct = 2.0 - self.coverage[h, r] - self.coverage[t, r]
        sem_norm = sem * 1e-8 / (sem.mean() + 1e-8)
        return self.alpha * sem_norm + (1 - self.alpha) * struct

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0].item(), triples[i, 1].item(), triples[i, 2].item()
            if h < self.num_entities and r < self.num_relations:
                self.coverage[h, r] = 1
            if t < self.num_entities and r < self.num_relations:
                self.coverage[t, r] = 1


class GPOnly(nn.Module):
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 2.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        h_mean, t_mean = self.entity_mean[h], self.entity_mean[t]
        h_std = torch.exp(0.5 * self.entity_logvar[h])
        t_std = torch.exp(0.5 * self.entity_logvar[t])
        h_emb = h_mean + h_std * torch.randn_like(h_std) if self.training else h_mean
        t_emb = t_mean + t_std * torch.randn_like(t_std) if self.training else t_mean
        r_emb = self.relation_emb(r)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        return 0.5 * (self.entity_logvar[h].exp().mean(-1) + self.entity_logvar[t].exp().mean(-1))

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0].item(), triples[i, 1].item(), triples[i, 2].item()
            if h < self.num_entities and r < self.num_relations:
                self.coverage[h, r] = 1
            if t < self.num_entities and r < self.num_relations:
                self.coverage[t, r] = 1


class EnergyBaseline(nn.Module):
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        scores = self.forward(h, r, t)
        return -scores  # negative score = high uncertainty

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0].item(), triples[i, 1].item(), triples[i, 2].item()
            if h < self.num_entities and r < self.num_relations:
                self.coverage[h, r] = 1
            if t < self.num_entities and r < self.num_relations:
                self.coverage[t, r] = 1


# ---- Training & Evaluation ----

def train_model(model, triples, device, epochs=EPOCHS, unc_weight=0.1, batch_size=BATCH_SIZE):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    h_all, r_all, t_all = triples[:, 0], triples[:, 1], triples[:, 2]
    dataset = TensorDataset(h_all, r_all, t_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) +
                    F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))

            if hasattr(model, 'entity_logvar'):
                kl = -0.5 * torch.mean(1 + model.entity_logvar - model.entity_mean.pow(2) - model.entity_logvar.exp())
                loss += 0.001 * kl

            if unc_weight > 0 and hasattr(model, 'get_uncertainty'):
                try:
                    unc_pos = model.get_uncertainty(h, r, t)
                    unc_neg = model.get_uncertainty(h, r, neg_t)
                    margin_loss = F.relu(0.3 - (unc_neg - unc_pos)).mean()
                    loss += unc_weight * margin_loss
                except Exception:
                    pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            log(f"    Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


def evaluate_ood(model, train_triples, test_triples, device):
    model.eval()
    if hasattr(model, 'precompute_coverage'):
        model.precompute_coverage(train_triples)

    freq = torch.zeros(model.num_entities, dtype=torch.long)
    for col in [0, 2]:
        for e in train_triples[:, col]:
            freq[e.item()] += 1

    tau = int(np.percentile(freq[freq > 0].numpy(), 25))

    h_test, r_test, t_test = test_triples[:, 0], test_triples[:, 1], test_triples[:, 2]
    min_freq = torch.minimum(freq[h_test], freq[t_test])
    is_emerging = min_freq <= tau

    cov_h = model.coverage[h_test, r_test] if hasattr(model, 'coverage') else torch.ones(len(h_test))
    cov_t = model.coverage[t_test, r_test] if hasattr(model, 'coverage') else torch.ones(len(h_test))
    is_novel = (~is_emerging) & ((cov_h == 0) | (cov_t == 0))
    is_ood = is_emerging | is_novel

    if is_ood.sum() == 0 or (~is_ood).sum() == 0:
        return {'overall': float('nan'), 'emerging': float('nan'), 'novel': float('nan')}

    with torch.no_grad():
        uncertainties = model.get_uncertainty(h_test.to(device), r_test.to(device), t_test.to(device)).cpu().numpy()

    labels = is_ood.numpy().astype(int)
    results = {'overall': float(roc_auc_score(labels, uncertainties))}

    for cat_name, cat_mask in [('emerging', is_emerging), ('novel', is_novel)]:
        if cat_mask.sum() > 0:
            mask = cat_mask | (~is_ood)
            if cat_mask[mask].sum() > 0 and (~is_ood)[mask].sum() > 0:
                results[cat_name] = float(roc_auc_score(cat_mask[mask].numpy(), uncertainties[mask]))

    return results


def main():
    log(f"\n{'='*60}")
    log(f"GDELT Pipeline - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")

    # Step 1: Download
    if not download_gdelt():
        log("FATAL: Could not download GDELT data. Exiting.")
        return

    # Step 2: Load
    data = load_gdelt()
    train_triples = data['train']
    test_triples = data['test']
    n_ent = data['num_entities']
    n_rel = data['num_relations']
    device = torch.device('cpu')

    all_results = {}
    start_time = time.time()

    methods = {
        'CAGP': lambda: CAGP(n_ent, n_rel),
        'CoverageOnly': lambda: CoverageOnly(n_ent, n_rel),
        'GPOnly': lambda: GPOnly(n_ent, n_rel),
        'Energy': lambda: EnergyBaseline(n_ent, n_rel),
    }

    for seed in SEEDS:
        log(f"\n--- Seed {seed} ---")
        seed_results = {}

        for method_name, model_fn in methods.items():
            log(f"  Training {method_name}...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_fn().to(device)
            unc_w = 0.1 if method_name == 'CAGP' else 0.0
            model = train_model(model, train_triples, device, epochs=EPOCHS, unc_weight=unc_w)
            model.precompute_coverage(train_triples)

            r = evaluate_ood(model, train_triples, test_triples, device)
            seed_results[method_name] = r
            log(f"    {method_name}: overall={r.get('overall', 'N/A'):.4f}, emerging={r.get('emerging', 'N/A')}, novel={r.get('novel', 'N/A')}")

            del model; gc.collect()

        all_results[f'seed_{seed}'] = seed_results

    # Aggregate
    summary = {}
    for method_name in methods:
        for metric in ['overall', 'emerging', 'novel']:
            vals = []
            for seed in SEEDS:
                v = all_results[f'seed_{seed}'].get(method_name, {}).get(metric, float('nan'))
                if not np.isnan(v):
                    vals.append(v)
            if vals:
                summary[f'{method_name}_{metric}'] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }

    elapsed = time.time() - start_time
    final = {
        'per_seed': all_results,
        'summary': summary,
        'metadata': {
            'dataset': 'GDELT',
            'num_entities': n_ent,
            'num_relations': n_rel,
            'train_size': len(train_triples),
            'test_size': len(test_triples),
            'epochs': EPOCHS,
            'seeds': SEEDS,
            'elapsed_seconds': elapsed,
        }
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final, f, indent=2)

    log(f"\nDone! Elapsed: {elapsed:.0f}s")
    log(f"Results saved to {OUTPUT_PATH}")
    log(f"\nSummary:")
    for k, v in sorted(summary.items()):
        log(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == '__main__':
    main()
