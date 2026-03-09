#!/usr/bin/env python3
"""
Verification of the "83% zero-evidence in top confident" claim.

This script investigates whether the 83% claim holds under scrutiny, specifically:
1. Is the finding confounded by entity frequency?
2. What happens when we control for frequency?
3. Why does U_sem behave differently on ICEWS14 (7% vs 83%)?

Key questions:
- Are top-confident predictions biased toward high-frequency entities?
- Does the 83% hold when controlling for frequency?
- What explains the U_sem ICEWS14 anomaly?
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
from scipy import stats
import time

from src.data.loaders import load_fb15k237, load_icews14


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model definitions (same as original)
# ============================================================

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

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class GPOnly(nn.Module):
    """U_sem baseline - pure semantic uncertainty without coverage."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        if self.training:
            h_std = torch.exp(0.5 * self.entity_logvar[h])
            t_std = torch.exp(0.5 * self.entity_logvar[t])
            h_emb = self.entity_mean[h] + h_std * torch.randn_like(h_std)
            t_emb = self.entity_mean[t] + t_std * torch.randn_like(t_std)
        else:
            h_emb = self.entity_mean[h]
            t_emb = self.entity_mean[t]
        return (h_emb * self.relation_emb(r) * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def _kl_entity_gaussian(model):
    if not (hasattr(model, 'entity_mean') and hasattr(model, 'entity_logvar')):
        return None
    mean = model.entity_mean
    logvar = model.entity_logvar
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()


def train_model(model, triples, device, epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1):
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

            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl

            if hasattr(model, 'entity_logvar'):
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
            print(f"    Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    return model


# ============================================================
# Analysis functions
# ============================================================

def compute_entity_frequencies(train_triples, n_ent):
    """Compute how many times each entity appears in training."""
    freq = np.zeros(n_ent)
    for i in range(len(train_triples)):
        freq[train_triples[i, 0]] += 1
        freq[train_triples[i, 2]] += 1
    return freq


def compute_triple_frequency(test_triples, entity_freq):
    """Compute average entity frequency for each test triple."""
    triple_freq = np.zeros(len(test_triples))
    for i in range(len(test_triples)):
        h, _, t = test_triples[i]
        triple_freq[i] = (entity_freq[h] + entity_freq[t]) / 2
    return triple_freq


def get_zero_evidence_mask(test_triples, coverage):
    """Return boolean mask: True if triple has zero evidence (novel context)."""
    zero_ev = np.zeros(len(test_triples), dtype=bool)
    for i in range(len(test_triples)):
        h, r, t = test_triples[i]
        if coverage[h, r] == 0 or coverage[t, r] == 0:
            zero_ev[i] = True
    return zero_ev


def analyze_top_confident(model, test, train, device, k=100):
    """
    Analyze top-K most confident predictions.
    Returns dict with detailed statistics.
    """
    model.eval()
    cov = model.coverage.cpu().numpy()
    n_ent = model.num_entities

    # Entity frequencies
    entity_freq = compute_entity_frequencies(train, n_ent)
    triple_freq = compute_triple_frequency(test, entity_freq)

    # Zero evidence mask
    zero_ev = get_zero_evidence_mask(test, cov)

    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)
        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

    confidence = -uncertainties
    sorted_idx = np.argsort(confidence)[::-1]

    # Top-K and random-K statistics
    top_k_idx = sorted_idx[:k]
    random_k_idx = np.random.choice(len(test), k, replace=False)

    results = {
        'top_k_zero_ev_rate': zero_ev[top_k_idx].mean(),
        'random_k_zero_ev_rate': zero_ev[random_k_idx].mean(),
        'baseline_zero_ev_rate': zero_ev.mean(),
        'top_k_avg_freq': triple_freq[top_k_idx].mean(),
        'random_k_avg_freq': triple_freq[random_k_idx].mean(),
        'all_avg_freq': triple_freq.mean(),
        'top_k_freq_std': triple_freq[top_k_idx].std(),
        'all_freq_std': triple_freq.std(),
    }

    return results, zero_ev, triple_freq, confidence, entity_freq


def frequency_controlled_analysis(test, zero_ev, triple_freq, confidence, k=100):
    """
    Analyze zero-evidence rate controlling for frequency.
    Split entities into quartiles by frequency and check each.
    """
    # Quartile boundaries
    q25, q75 = np.percentile(triple_freq, [25, 75])

    low_freq_mask = triple_freq <= q25
    high_freq_mask = triple_freq >= q75
    mid_freq_mask = (triple_freq > q25) & (triple_freq < q75)

    results = {}

    for name, mask in [('low_freq', low_freq_mask), ('high_freq', high_freq_mask), ('mid_freq', mid_freq_mask)]:
        if mask.sum() < k:
            results[name] = {'n': mask.sum(), 'insufficient': True}
            continue

        # Get top-K confident within this frequency band
        conf_in_band = confidence.copy()
        conf_in_band[~mask] = -np.inf
        top_k_in_band = np.argsort(conf_in_band)[::-1][:k]

        results[name] = {
            'n': mask.sum(),
            'baseline_zero_ev': zero_ev[mask].mean(),
            'top_k_zero_ev': zero_ev[top_k_in_band].mean(),
            'avg_freq': triple_freq[mask].mean(),
        }

    return results


def bootstrap_baseline(zero_ev, k=100, n_bootstrap=1000):
    """Bootstrap confidence interval for baseline zero-evidence rate."""
    rates = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(zero_ev), k, replace=False)
        rates.append(zero_ev[sample_idx].mean())
    return np.mean(rates), np.std(rates), np.percentile(rates, [2.5, 97.5])


def compute_freq_variance_correlation(train, model, n_ent, device):
    """
    Compute Spearman correlation between entity frequency and variance.
    This explains why U_sem may behave differently across datasets.
    """
    entity_freq = compute_entity_frequencies(train, n_ent)

    with torch.no_grad():
        variances = torch.exp(model.entity_logvar).mean(dim=-1).cpu().numpy()

    # Only consider entities that appear in training
    nonzero_mask = entity_freq > 0
    freq_nz = entity_freq[nonzero_mask]
    var_nz = variances[nonzero_mask]

    spearman_corr, spearman_p = stats.spearmanr(freq_nz, var_nz)
    pearson_corr, pearson_p = stats.pearsonr(freq_nz, var_nz)

    return {
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'pearson': pearson_corr,
        'pearson_p': pearson_p,
        'n_entities': nonzero_mask.sum(),
    }


def run_full_verification(dataset_name, loader, device, epochs=30, seed=42):
    """Run full verification analysis on a dataset."""
    print(f"\n{'='*80}")
    print(f"  VERIFICATION ANALYSIS: {dataset_name}")
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

    # Train models
    models = {}
    for name, cls in [('Energy', EnergyBased), ('U_sem', GPOnly)]:
        print(f"\n  Training {name}...")
        t0 = time.time()
        model = cls(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=epochs)
        models[name] = model
        print(f"    Time: {time.time()-t0:.1f}s")

    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  ANALYSIS: {name} on {dataset_name}")
        print(f"{'='*60}")

        # Basic analysis
        basic, zero_ev, triple_freq, confidence, entity_freq = analyze_top_confident(
            model, test, train, device, k=100
        )

        # Frequency-controlled analysis
        freq_controlled = frequency_controlled_analysis(
            test, zero_ev, triple_freq, confidence, k=100
        )

        # Bootstrap baseline
        boot_mean, boot_std, boot_ci = bootstrap_baseline(zero_ev, k=100, n_bootstrap=1000)

        # Freq-variance correlation (only for U_sem)
        if name == 'U_sem':
            freq_var_corr = compute_freq_variance_correlation(train, model, n_ent, device)
        else:
            freq_var_corr = None

        # Print results
        print(f"\n  1. TOP-100 vs RANDOM-100 COMPARISON")
        print(f"     Top-100 zero-evidence rate:    {100*basic['top_k_zero_ev_rate']:.1f}%")
        print(f"     Random-100 zero-evidence rate: {100*basic['random_k_zero_ev_rate']:.1f}%")
        print(f"     Baseline (all test):           {100*basic['baseline_zero_ev_rate']:.1f}%")
        print(f"     Bootstrap 95% CI for baseline: [{100*boot_ci[0]:.1f}%, {100*boot_ci[1]:.1f}%]")

        print(f"\n  2. FREQUENCY ANALYSIS")
        print(f"     Top-100 avg entity freq:       {basic['top_k_avg_freq']:.1f}")
        print(f"     Random-100 avg entity freq:    {basic['random_k_avg_freq']:.1f}")
        print(f"     All test avg entity freq:      {basic['all_avg_freq']:.1f}")
        freq_bias = basic['top_k_avg_freq'] / basic['all_avg_freq']
        print(f"     Frequency bias (top/all):      {freq_bias:.2f}x")

        print(f"\n  3. FREQUENCY-CONTROLLED ZERO-EVIDENCE RATES")
        for band in ['low_freq', 'mid_freq', 'high_freq']:
            info = freq_controlled[band]
            if info.get('insufficient'):
                print(f"     {band}: insufficient data (n={info['n']})")
            else:
                print(f"     {band} (n={info['n']}, avg_freq={info['avg_freq']:.1f}):")
                print(f"       Baseline zero-ev: {100*info['baseline_zero_ev']:.1f}%")
                print(f"       Top-100 zero-ev:  {100*info['top_k_zero_ev']:.1f}%")

        if freq_var_corr:
            print(f"\n  4. FREQUENCY-VARIANCE CORRELATION (U_sem only)")
            print(f"     Spearman(freq, variance): {freq_var_corr['spearman']:.3f} (p={freq_var_corr['spearman_p']:.2e})")
            print(f"     Pearson(freq, variance):  {freq_var_corr['pearson']:.3f} (p={freq_var_corr['pearson_p']:.2e})")
            print(f"     Entities with freq > 0:   {freq_var_corr['n_entities']}")

        all_results[name] = {
            'basic': basic,
            'freq_controlled': freq_controlled,
            'bootstrap': {'mean': boot_mean, 'std': boot_std, 'ci': boot_ci},
            'freq_var_corr': freq_var_corr,
        }

    return all_results


def main():
    device = setup_device()
    print(f"Device: {device}")

    print("\n" + "="*80)
    print("VERIFICATION OF '83% ZERO-EVIDENCE IN TOP CONFIDENT' CLAIM")
    print("="*80)
    print("\nQuestions to answer:")
    print("  1. Is 83% zero-evidence in Energy's top-100 a frequency confound?")
    print("  2. Does the finding hold when controlling for entity frequency?")
    print("  3. Why does U_sem show 7% on ICEWS14 vs 83% on FB15k-237?")

    # Run on FB15k-237
    fb_results = run_full_verification(
        "FB15k-237",
        load_fb15k237,
        device,
        epochs=30,
        seed=42
    )

    # Run on ICEWS14
    icews_results = run_full_verification(
        "ICEWS14",
        load_icews14,
        device,
        epochs=30,
        seed=42
    )

    # Summary and verdict
    print("\n" + "="*80)
    print("SUMMARY AND VERDICT")
    print("="*80)

    print("\n  FB15k-237:")
    for name in ['Energy', 'U_sem']:
        r = fb_results[name]
        print(f"    {name}:")
        print(f"      Top-100 zero-ev: {100*r['basic']['top_k_zero_ev_rate']:.1f}%")
        print(f"      Freq bias: {r['basic']['top_k_avg_freq'] / r['basic']['all_avg_freq']:.2f}x")
        if r['freq_var_corr']:
            print(f"      Freq-var corr: {r['freq_var_corr']['spearman']:.3f}")

    print("\n  ICEWS14:")
    for name in ['Energy', 'U_sem']:
        r = icews_results[name]
        print(f"    {name}:")
        print(f"      Top-100 zero-ev: {100*r['basic']['top_k_zero_ev_rate']:.1f}%")
        print(f"      Freq bias: {r['basic']['top_k_avg_freq'] / r['basic']['all_avg_freq']:.2f}x")
        if r['freq_var_corr']:
            print(f"      Freq-var corr: {r['freq_var_corr']['spearman']:.3f}")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    # Check if Energy's 83% claim holds
    fb_energy_top100 = fb_results['Energy']['basic']['top_k_zero_ev_rate']
    fb_baseline = fb_results['Energy']['basic']['baseline_zero_ev_rate']
    fb_bootstrap_ci = fb_results['Energy']['bootstrap']['ci']

    print(f"\n  Q1: Does Energy's top-100 have significantly higher zero-evidence than baseline?")
    print(f"      Top-100: {100*fb_energy_top100:.1f}% vs Baseline: {100*fb_baseline:.1f}%")
    print(f"      Bootstrap 95% CI: [{100*fb_bootstrap_ci[0]:.1f}%, {100*fb_bootstrap_ci[1]:.1f}%]")
    if fb_energy_top100 > fb_bootstrap_ci[1]:
        print(f"      -> YES, significantly higher (outside 95% CI)")
    else:
        print(f"      -> NO, within random variation")

    print(f"\n  Q2: Is it a frequency confound?")
    fb_freq_bias = fb_results['Energy']['basic']['top_k_avg_freq'] / fb_results['Energy']['basic']['all_avg_freq']
    print(f"      Frequency bias: {fb_freq_bias:.2f}x")
    # Check frequency-controlled results
    fb_high_freq = fb_results['Energy']['freq_controlled']['high_freq']
    fb_low_freq = fb_results['Energy']['freq_controlled']['low_freq']
    if not fb_high_freq.get('insufficient') and not fb_low_freq.get('insufficient'):
        print(f"      High-freq band top-100 zero-ev: {100*fb_high_freq['top_k_zero_ev']:.1f}%")
        print(f"      Low-freq band top-100 zero-ev: {100*fb_low_freq['top_k_zero_ev']:.1f}%")
        if fb_high_freq['top_k_zero_ev'] > fb_high_freq['baseline_zero_ev'] * 1.5:
            print(f"      -> Finding HOLDS even in high-freq entities")
        else:
            print(f"      -> Finding is PARTIALLY a frequency confound")

    print(f"\n  Q3: Why does U_sem differ between FB15k-237 and ICEWS14?")
    fb_usem_corr = fb_results['U_sem']['freq_var_corr']['spearman']
    icews_usem_corr = icews_results['U_sem']['freq_var_corr']['spearman']
    print(f"      FB15k-237 freq-var correlation: {fb_usem_corr:.3f}")
    print(f"      ICEWS14 freq-var correlation:   {icews_usem_corr:.3f}")
    if abs(fb_usem_corr - icews_usem_corr) > 0.2:
        print(f"      -> Different freq-variance relationships explain the discrepancy")
    else:
        print(f"      -> Similar correlations; difference may be due to coverage structure")

    fb_usem_top100 = fb_results['U_sem']['basic']['top_k_zero_ev_rate']
    icews_usem_top100 = icews_results['U_sem']['basic']['top_k_zero_ev_rate']
    print(f"      FB15k-237 U_sem top-100 zero-ev: {100*fb_usem_top100:.1f}%")
    print(f"      ICEWS14 U_sem top-100 zero-ev:   {100*icews_usem_top100:.1f}%")


if __name__ == "__main__":
    main()
