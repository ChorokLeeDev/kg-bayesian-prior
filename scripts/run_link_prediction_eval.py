#!/usr/bin/env python3
"""
Link Prediction Evaluation for CAGP vs Vanilla DistMult

Evaluates whether CAGP maintains link prediction quality (MRR, Hits@1, Hits@10)
compared to vanilla DistMult baseline.

Expected: MRR ~0.24, Hits@10 ~0.38 for FB15k-237 (matching literature).
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from tqdm import tqdm
import time

from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Base model with score_heads and score_tails support
# ============================================================

class BaseDistMult(nn.Module):
    """Base class providing score_heads/score_tails for link prediction."""

    def __init__(self, num_entities, num_relations):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations

    def score_triple(self, h, r, t):
        """Subclasses must implement this."""
        raise NotImplementedError

    def forward(self, h, r, t):
        return self.score_triple(h, r, t)

    def score_heads(self, relation, tail):
        """Score all possible heads for (?, r, t) queries. Expects batch_size=1."""
        # For DistMult: score = sum(h * r * t), so for all heads:
        # scores[e] = sum(entity[e] * relation * entity[tail])
        r_emb = self._get_entity_emb_for_scoring('relation', relation)  # (1, dim)
        t_emb = self._get_entity_emb_for_scoring('entity', tail)       # (1, dim)
        all_h = self._get_all_entity_embs()                            # (num_entities, dim)
        # (num_entities, dim) * (1, dim) * (1, dim) -> (num_entities, dim) -> sum -> (num_entities,)
        scores = (all_h * r_emb * t_emb).sum(dim=-1)
        return scores.unsqueeze(0)  # (1, num_entities)

    def score_tails(self, head, relation):
        """Score all possible tails for (h, r, ?) queries. Expects batch_size=1."""
        h_emb = self._get_entity_emb_for_scoring('entity', head)       # (1, dim)
        r_emb = self._get_entity_emb_for_scoring('relation', relation)  # (1, dim)
        all_t = self._get_all_entity_embs()                            # (num_entities, dim)
        scores = (h_emb * r_emb * all_t).sum(dim=-1)
        return scores.unsqueeze(0)

    def _get_entity_emb_for_scoring(self, emb_type, idx):
        """Get embedding for scoring (subclasses override for mean vs sampled)."""
        raise NotImplementedError

    def _get_all_entity_embs(self):
        """Get all entity embeddings for scoring."""
        raise NotImplementedError


# ============================================================
# Model Implementations
# ============================================================

class VanillaDistMult(BaseDistMult):
    """Vanilla DistMult baseline (no uncertainty, no variance)."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__(num_entities, num_relations)
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score_triple(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def _get_entity_emb_for_scoring(self, emb_type, idx):
        if emb_type == 'relation':
            return self.relation_emb(idx)
        return self.entity_emb(idx)

    def _get_all_entity_embs(self):
        return self.entity_emb.weight


class CAGPDistMult(BaseDistMult):
    """CAGP with variational embeddings for link prediction."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__(num_entities, num_relations)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def score_triple(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def _get_entity_emb_for_scoring(self, emb_type, idx):
        if emb_type == 'relation':
            return self.relation_emb(idx)
        return self.entity_mean[idx]

    def _get_all_entity_embs(self):
        return self.entity_mean

    def get_uncertainty(self, h, r, t):
        """For OOD evaluation (not used in link prediction)."""
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        gp_var = (h_var + t_var) / 2
        cov_unc = 2.0 - self.coverage[h, r] - self.coverage[t, r]
        return 0.5 * gp_var + 0.5 * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Training
# ============================================================

def _kl_entity_gaussian(model):
    """KL(q(e)||N(0,1)) for models with explicit entity mean/logvar parameters."""
    if not (hasattr(model, 'entity_mean') and hasattr(model, 'entity_logvar')):
        return None
    mean = model.entity_mean
    logvar = model.entity_logvar
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()


def train_model(model, triples, device, epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1):
    """Train model with standard KGE objective + optional KL + uncertainty margin."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            # KL regularization (only for CAGP)
            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl

            # Uncertainty margin (only for CAGP)
            if hasattr(model, 'get_uncertainty'):
                pos_unc = model.get_uncertainty(h, r, t)
                neg_unc = model.get_uncertainty(h, r, neg_t)
                unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
                loss = loss + unc_weight * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


# ============================================================
# Link Prediction Evaluation
# ============================================================

def compute_filtered_mrr(model, test_triples, all_triples, num_entities, device, batch_size=64):
    """
    Compute filtered MRR, Hits@1, Hits@10 for link prediction.

    Uses pre-built filter dicts for O(1) lookup per known triple instead of
    iterating over all entities. Memory-efficient: scores one triple at a time
    on the entity dimension, uses small batch sizes.

    Returns: dict with mrr, hits@1, hits@10, mean_rank
    """
    model.eval()
    model = model.to(device)

    # Build filter dicts: (h,r) -> set of known tails, (r,t) -> set of known heads
    from collections import defaultdict
    tail_filter = defaultdict(set)  # (h, r) -> {t1, t2, ...}
    head_filter = defaultdict(set)  # (r, t) -> {h1, h2, ...}
    for triple in all_triples:
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        tail_filter[(h, r)].add(t)
        head_filter[(r, t)].add(h)

    all_ranks = []
    n_done = 0

    with torch.no_grad():
        for start in range(0, len(test_triples), batch_size):
            end = min(start + batch_size, len(test_triples))
            batch = test_triples[start:end]
            bs = len(batch)

            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            # --- Tail prediction: (h, r, ?) ---
            # Score all entities as potential tails
            # Do one triple at a time to save memory
            for i in range(bs):
                hi, ri, ti = int(batch[i, 0]), int(batch[i, 1]), int(batch[i, 2])
                h_i = torch.tensor([hi], device=device)
                r_i = torch.tensor([ri], device=device)

                # Score all tails: shape (1, num_entities)
                scores = model.score_tails(h_i, r_i).squeeze(0)  # (num_entities,)

                # Filter: set known tails (except true one) to -inf
                known = tail_filter.get((hi, ri), set())
                filter_idx = [e for e in known if e != ti]
                if filter_idx:
                    scores[torch.tensor(filter_idx, device=device)] = float('-inf')

                # Rank of true tail
                true_score = scores[ti]
                rank = (scores > true_score).sum().item() + 1
                all_ranks.append(rank)

            # --- Head prediction: (?, r, t) ---
            for i in range(bs):
                hi, ri, ti = int(batch[i, 0]), int(batch[i, 1]), int(batch[i, 2])
                r_i = torch.tensor([ri], device=device)
                t_i = torch.tensor([ti], device=device)

                scores = model.score_heads(r_i, t_i).squeeze(0)

                known = head_filter.get((ri, ti), set())
                filter_idx = [e for e in known if e != hi]
                if filter_idx:
                    scores[torch.tensor(filter_idx, device=device)] = float('-inf')

                true_score = scores[hi]
                rank = (scores > true_score).sum().item() + 1
                all_ranks.append(rank)

            n_done += bs
            if n_done % 1000 == 0 or end == len(test_triples):
                print(f"  Evaluated {n_done}/{len(test_triples)} triples...", flush=True)

    all_ranks = torch.tensor(all_ranks, dtype=torch.float)

    return {
        "mrr": (1.0 / all_ranks).mean().item(),
        "hits@1": (all_ranks <= 1).float().mean().item(),
        "hits@10": (all_ranks <= 10).float().mean().item(),
        "mean_rank": all_ranks.mean().item(),
    }


# ============================================================
# Main
# ============================================================

def run_evaluation(dataset_name, loader, device, seed=42, epochs=30, lr=0.001):
    """Run link prediction eval on one dataset."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_ds, valid_ds, test_ds = loader()
    train = train_ds.triples
    valid = valid_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # All known triples for filtering
    all_triples = np.concatenate([train, valid, test])

    results = {}

    # Train vanilla DistMult
    print(f"\n--- Vanilla DistMult ---")
    t0 = time.time()
    vanilla = VanillaDistMult(n_ent, n_rel)
    vanilla = train_model(vanilla, train, device, epochs=epochs, lr=lr, kl_beta=0.0, unc_weight=0.0)
    vanilla_lp = compute_filtered_mrr(vanilla, test, all_triples, n_ent, device)
    vanilla_time = time.time() - t0

    print(f"  MRR: {vanilla_lp['mrr']:.4f}")
    print(f"  Hits@1: {vanilla_lp['hits@1']:.4f}")
    print(f"  Hits@10: {vanilla_lp['hits@10']:.4f}")
    print(f"  Mean Rank: {vanilla_lp['mean_rank']:.1f}")
    print(f"  Time: {vanilla_time:.1f}s")

    results['vanilla'] = {
        **vanilla_lp,
        'time_seconds': vanilla_time,
    }

    # Train CAGP
    print(f"\n--- CAGP ---")
    t0 = time.time()
    cagp = CAGPDistMult(n_ent, n_rel)
    cagp.precompute_coverage(train)
    cagp = train_model(cagp, train, device, epochs=epochs, lr=lr, kl_beta=0.001, unc_weight=0.1)
    cagp_lp = compute_filtered_mrr(cagp, test, all_triples, n_ent, device)
    cagp_time = time.time() - t0

    print(f"  MRR: {cagp_lp['mrr']:.4f}")
    print(f"  Hits@1: {cagp_lp['hits@1']:.4f}")
    print(f"  Hits@10: {cagp_lp['hits@10']:.4f}")
    print(f"  Mean Rank: {cagp_lp['mean_rank']:.1f}")
    print(f"  Time: {cagp_time:.1f}s")

    results['cagp'] = {
        **cagp_lp,
        'time_seconds': cagp_time,
    }

    # Compute degradation
    mrr_diff = cagp_lp['mrr'] - vanilla_lp['mrr']
    hits10_diff = cagp_lp['hits@10'] - vanilla_lp['hits@10']

    print(f"\n--- Comparison ---")
    print(f"  MRR diff: {mrr_diff:+.4f} ({mrr_diff/vanilla_lp['mrr']*100:+.1f}%)")
    print(f"  Hits@10 diff: {hits10_diff:+.4f} ({hits10_diff/vanilla_lp['hits@10']*100:+.1f}%)")

    results['comparison'] = {
        'mrr_diff': float(mrr_diff),
        'mrr_diff_pct': float(mrr_diff / vanilla_lp['mrr'] * 100),
        'hits@10_diff': float(hits10_diff),
        'hits@10_diff_pct': float(hits10_diff / vanilla_lp['hits@10'] * 100),
    }

    return results


def main():
    device = setup_device()
    print(f"Device: {device}")

    all_results = {
        'config': {
            'seed': 42,
            'epochs': 30,
            'lr': 0.001,
            'batch_size': 1024,
        }
    }

    # FB15k-237
    all_results['fb15k237'] = run_evaluation(
        'FB15k-237',
        load_fb15k237,
        device,
        seed=42,
        epochs=30,
        lr=0.001,
    )

    # WN18RR
    all_results['wn18rr'] = run_evaluation(
        'WN18RR',
        load_wn18rr,
        device,
        seed=42,
        epochs=30,
        lr=0.001,
    )

    # Save results
    output_path = project_root / 'outputs' / 'link_prediction_eval.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'Model':<15} {'MRR':>10} {'Hits@1':>10} {'Hits@10':>10}")
    print("-"*80)
    for ds in ['fb15k237', 'wn18rr']:
        for model in ['vanilla', 'cagp']:
            r = all_results[ds][model]
            print(f"{ds:<15} {model:<15} {r['mrr']:>10.4f} {r['hits@1']:>10.4f} {r['hits@10']:>10.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
