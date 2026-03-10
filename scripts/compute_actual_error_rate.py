#!/usr/bin/env python3
"""
Compute ACTUAL ERROR RATE for Energy's top-100 confident predictions with zero coverage.

Reviewer critique: The "78% confident-wrong" statistic only shows zero-coverage rate,
not actual ERROR rate. This script computes what fraction of Energy's most confident
zero-coverage predictions are ACTUALLY WRONG (incorrect link predictions).

Key question: Among Energy's top-100 most confident predictions that have zero coverage,
what fraction are incorrect link predictions?

For each test triple (h, r, t):
- We know the ground truth tail entity (t is correct)
- The model predicts scores for ALL possible tail entities
- A prediction is "wrong" if the model ranks some incorrect entity higher than t
- We use reciprocal rank and check if the true tail is ranked in top-K

Definition of "error":
1. STRICT: true tail is not ranked #1 (typical MRR evaluation)
2. RELAXED: true tail is not ranked in top-10 (Hits@10 style)
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
import time

from src.data.loaders import load_fb15k237, load_icews14, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EnergyBased(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        return -self.forward(h, r, t)

    def score_all_tails(self, h, r):
        """Score all possible tail entities for a given (h, r) pair."""
        h_emb = self.entity_emb(h)  # [batch, dim]
        r_emb = self.relation_emb(r)  # [batch, dim]
        all_t_emb = self.entity_emb.weight  # [n_ent, dim]
        # Broadcast: [batch, 1, dim] * [batch, 1, dim] * [1, n_ent, dim]
        scores = (h_emb.unsqueeze(1) * r_emb.unsqueeze(1) * all_t_emb.unsqueeze(0)).sum(-1)
        return scores  # [batch, n_ent]

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def train_model(model, triples, device, epochs=30, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


def compute_ranks(model, test_triples, device, batch_size=100):
    """
    Compute the rank of true tail entity for each test triple.
    Returns array of ranks (1-indexed: rank=1 means correct prediction).
    """
    model.eval()
    ranks = []

    n_test = len(test_triples)

    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch = test_triples[start:end]

            h = torch.tensor(batch[:, 0]).to(device)
            r = torch.tensor(batch[:, 1]).to(device)
            t = torch.tensor(batch[:, 2]).to(device)

            # Score all tail entities
            scores = model.score_all_tails(h, r)  # [batch, n_ent]

            # Get rank of true tail (higher score = better)
            true_scores = scores[torch.arange(len(batch)), t.cpu()].cpu().numpy()

            # For each sample, count how many entities have higher or equal score
            for i in range(len(batch)):
                score_i = scores[i].cpu().numpy()
                true_score_i = true_scores[i]
                # Rank = 1 + number of entities with strictly higher score
                rank = 1 + (score_i > true_score_i).sum()
                ranks.append(rank)

    return np.array(ranks)


def compute_error_metrics(ranks, k_values=[1, 10]):
    """
    Compute error metrics from ranks.

    Returns:
        - MRR: Mean Reciprocal Rank
        - error_rate_strict: fraction where rank > 1
        - error_rate_at_k: fraction where rank > k
    """
    mrr = np.mean(1.0 / ranks)
    hits_at_k = {k: np.mean(ranks <= k) for k in k_values}
    error_at_k = {k: 1.0 - hits_at_k[k] for k in k_values}

    return {
        'mrr': mrr,
        'hits_at_k': hits_at_k,
        'error_at_k': error_at_k,
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks),
    }


def analyze_confident_wrong_with_errors(model, test_triples, device, k=100):
    """
    The key analysis:
    1. Find top-K most confident predictions
    2. Among those, identify which have zero coverage
    3. Compute the ACTUAL ERROR RATE for those zero-coverage predictions

    Returns detailed breakdown.
    """
    model.eval()
    cov = model.coverage.cpu().numpy()
    n_test = len(test_triples)

    # Step 1: Compute uncertainties for all test triples
    print("  Computing uncertainties...")
    with torch.no_grad():
        h = torch.tensor(test_triples[:, 0]).to(device)
        r = torch.tensor(test_triples[:, 1]).to(device)
        t = torch.tensor(test_triples[:, 2]).to(device)
        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -uncertainties  # Higher = more confident

    # Step 2: Identify zero-coverage (novel context) triples
    zero_coverage = []
    for i in range(n_test):
        h_cov = cov[test_triples[i, 0], test_triples[i, 1]]
        t_cov = cov[test_triples[i, 2], test_triples[i, 1]]
        zero_coverage.append(h_cov == 0 or t_cov == 0)
    zero_coverage = np.array(zero_coverage)

    # Step 3: Sort by confidence and select top-K
    sorted_indices = np.argsort(confidence)[::-1]
    top_k_indices = sorted_indices[:k]

    # Step 4: Compute ranks for ALL test triples (needed for error analysis)
    print("  Computing ranks (this may take a while)...")
    ranks = compute_ranks(model, test_triples, device)

    # Step 5: Analyze different subsets
    results = {}

    # All test triples
    results['all_test'] = {
        'n': n_test,
        'zero_cov_rate': zero_coverage.mean(),
        **compute_error_metrics(ranks)
    }

    # Top-K most confident
    top_k_ranks = ranks[top_k_indices]
    top_k_zero_cov = zero_coverage[top_k_indices]
    results['top_k_confident'] = {
        'n': k,
        'zero_cov_rate': top_k_zero_cov.mean(),
        **compute_error_metrics(top_k_ranks)
    }

    # Top-K confident AND zero coverage (THE KEY METRIC)
    top_k_and_zero_cov_mask = top_k_zero_cov
    top_k_and_zero_cov_indices = top_k_indices[top_k_and_zero_cov_mask]
    if len(top_k_and_zero_cov_indices) > 0:
        results['top_k_confident_zero_cov'] = {
            'n': len(top_k_and_zero_cov_indices),
            **compute_error_metrics(ranks[top_k_and_zero_cov_indices])
        }
    else:
        results['top_k_confident_zero_cov'] = {'n': 0}

    # Random K for comparison
    np.random.seed(42)
    random_k_indices = np.random.choice(n_test, k, replace=False)
    random_k_ranks = ranks[random_k_indices]
    random_k_zero_cov = zero_coverage[random_k_indices]
    results['random_k'] = {
        'n': k,
        'zero_cov_rate': random_k_zero_cov.mean(),
        **compute_error_metrics(random_k_ranks)
    }

    # All zero-coverage triples (regardless of confidence)
    zero_cov_indices = np.where(zero_coverage)[0]
    if len(zero_cov_indices) > 0:
        results['all_zero_cov'] = {
            'n': len(zero_cov_indices),
            **compute_error_metrics(ranks[zero_cov_indices])
        }

    # All non-zero-coverage triples
    nonzero_cov_indices = np.where(~zero_coverage)[0]
    if len(nonzero_cov_indices) > 0:
        results['all_nonzero_cov'] = {
            'n': len(nonzero_cov_indices),
            **compute_error_metrics(ranks[nonzero_cov_indices])
        }

    return results, confidence, zero_coverage, ranks


def run_analysis(dataset_name, loader, device, epochs=30, seed=42):
    """Run full error rate analysis on a dataset."""
    print(f"\n{'='*80}")
    print(f"  ERROR RATE ANALYSIS: {dataset_name}")
    print(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Train Energy model
    print("\n  Training Energy model...")
    t0 = time.time()
    model = EnergyBased(n_ent, n_rel)
    model.precompute_coverage(train)
    model = train_model(model, train, device, epochs=epochs)
    print(f"    Time: {time.time()-t0:.1f}s")

    # Analyze
    print("\n  Analyzing confidence vs errors...")
    results, confidence, zero_cov, ranks = analyze_confident_wrong_with_errors(
        model, test, device, k=100
    )

    return results


def print_results_table(results, dataset_name):
    """Print a nice summary table."""
    print(f"\n{'='*80}")
    print(f"  RESULTS: {dataset_name}")
    print(f"{'='*80}")

    print("\n" + "-"*80)
    print(f"{'Subset':<35} | {'N':>6} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6} | {'Err@1':>6} | {'Err@10':>6}")
    print("-"*80)

    for key in ['all_test', 'top_k_confident', 'top_k_confident_zero_cov', 'random_k', 'all_zero_cov', 'all_nonzero_cov']:
        r = results[key]
        if r['n'] == 0:
            print(f"{key:<35} | {r['n']:>6} |   N/A  |   N/A  |   N/A  |   N/A  |   N/A  ")
            continue

        mrr = r.get('mrr', 0)
        h1 = r.get('hits_at_k', {}).get(1, 0)
        h10 = r.get('hits_at_k', {}).get(10, 0)
        e1 = r.get('error_at_k', {}).get(1, 1)
        e10 = r.get('error_at_k', {}).get(10, 1)

        print(f"{key:<35} | {r['n']:>6} | {mrr:>6.3f} | {h1:>6.1%} | {h10:>6.1%} | {e1:>6.1%} | {e10:>6.1%}")

    print("-"*80)


def main():
    device = setup_device()
    print(f"Device: {device}")

    output_lines = []

    def log(s):
        print(s)
        output_lines.append(s)

    log("="*80)
    log("ACTUAL ERROR RATE ANALYSIS")
    log("="*80)
    log("")
    log("Reviewer critique: The '78% confident-wrong' statistic only shows")
    log("zero-coverage rate, not actual ERROR rate.")
    log("")
    log("This analysis computes the ACTUAL ERROR RATE for Energy's most")
    log("confident predictions that have zero coverage.")
    log("")
    log("Key question: Among Energy's top-100 most confident predictions")
    log("with zero coverage, what fraction are ACTUALLY WRONG?")
    log("")

    # Run on FB15k-237 (the main dataset in the claim)
    fb_results = run_analysis(
        "FB15k-237",
        load_fb15k237,
        device,
        epochs=30,
        seed=42
    )

    # Print detailed results
    print_results_table(fb_results, "FB15k-237")

    # Summary for output file
    log("")
    log("="*80)
    log("KEY FINDINGS: FB15k-237")
    log("="*80)
    log("")

    # The critical numbers
    all_test = fb_results['all_test']
    top100 = fb_results['top_k_confident']
    top100_zc = fb_results['top_k_confident_zero_cov']
    all_zc = fb_results['all_zero_cov']
    all_nzc = fb_results['all_nonzero_cov']

    log(f"1. Zero-coverage rate in top-100 confident: {top100['zero_cov_rate']:.1%}")
    log(f"   (This is the ~78-83% statistic in the paper)")
    log("")

    if top100_zc['n'] > 0:
        log(f"2. ERROR RATE among top-100 confident with zero coverage:")
        log(f"   - N = {top100_zc['n']} triples")
        log(f"   - Error@1 (not rank 1): {top100_zc['error_at_k'][1]:.1%}")
        log(f"   - Error@10 (not in top 10): {top100_zc['error_at_k'][10]:.1%}")
        log(f"   - MRR: {top100_zc['mrr']:.3f}")
        log("")

    log(f"3. For comparison - baseline error rates:")
    log(f"   a) ALL test triples:")
    log(f"      - Error@1: {all_test['error_at_k'][1]:.1%}")
    log(f"      - Error@10: {all_test['error_at_k'][10]:.1%}")
    log(f"      - MRR: {all_test['mrr']:.3f}")
    log(f"")
    log(f"   b) ALL zero-coverage triples (n={all_zc['n']}):")
    log(f"      - Error@1: {all_zc['error_at_k'][1]:.1%}")
    log(f"      - Error@10: {all_zc['error_at_k'][10]:.1%}")
    log(f"      - MRR: {all_zc['mrr']:.3f}")
    log(f"")
    log(f"   c) ALL non-zero-coverage triples (n={all_nzc['n']}):")
    log(f"      - Error@1: {all_nzc['error_at_k'][1]:.1%}")
    log(f"      - Error@10: {all_nzc['error_at_k'][10]:.1%}")
    log(f"      - MRR: {all_nzc['mrr']:.3f}")

    log("")
    log("="*80)
    log("VERDICT")
    log("="*80)
    log("")

    if top100_zc['n'] > 0:
        err_rate = top100_zc['error_at_k'][1]
        baseline_err = all_test['error_at_k'][1]
        zc_err = all_zc['error_at_k'][1]

        if err_rate > 0.5:
            log("The claim IS VALID:")
            log(f"  - Energy's top-100 confident zero-coverage predictions have")
            log(f"    {err_rate:.1%} error rate (Error@1)")
            log(f"  - This is {'HIGHER' if err_rate > baseline_err else 'LOWER'} than baseline ({baseline_err:.1%})")
            log(f"  - Zero-coverage triples overall have {zc_err:.1%} error rate")
        else:
            log("The claim MAY BE MISLEADING:")
            log(f"  - Error rate is only {err_rate:.1%}")
            log(f"  - While {top100['zero_cov_rate']:.1%} have zero coverage,")
            log(f"    the model is still often correct despite no training evidence")

    log("")
    log("="*80)

    # Save to file
    output_path = Path(project_root) / "outputs" / "confident_wrong_error_rate.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
