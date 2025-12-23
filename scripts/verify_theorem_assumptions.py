"""
Verify Theorem Assumptions (A1)-(A6) empirically.

This script trains actual GP-KGE models on each dataset and extracts
the learned variances to verify theorem assumptions with real measurements.

Key improvement over simulation: We train actual models rather than
simulating variance based on frequency, avoiding circular reasoning.
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy import stats
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237, load_wn18rr, load_yago310


class SimpleGPKGE(nn.Module):
    """
    Simplified GP-KGE model for assumption verification.

    Uses variational inference with learned entity variances.
    This is the actual model - no simulation.
    """

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Entity embeddings: mean and log-variance (variational)
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)

        # Relation embeddings
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def sample(self, indices):
        """Reparameterization trick for variational inference."""
        mean = self.entity_mean[indices]
        std = torch.exp(0.5 * self.entity_logvar[indices])
        return mean + std * torch.randn_like(std)

    def forward(self, heads, relations, tails, use_sampling=True):
        """DistMult scoring."""
        if use_sampling and self.training:
            h = self.sample(heads)
            t = self.sample(tails)
        else:
            h = self.entity_mean[heads]
            t = self.entity_mean[tails]

        r = self.relation_emb(relations)
        return (h * r * t).sum(dim=-1)

    def kl_loss(self):
        """KL divergence from standard normal prior."""
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities

    def get_entity_variance(self):
        """Return mean variance per entity (averaged across dimensions)."""
        with torch.no_grad():
            var = torch.exp(self.entity_logvar)  # [num_entities, dim]
            return var.mean(dim=-1).cpu().numpy()  # [num_entities]


def train_model(train_triples, num_entities, num_relations,
                dim=100, epochs=30, batch_size=2048, lr=1e-3, kl_weight=0.01,
                device='cpu', verbose=True):
    """
    Train a GP-KGE model and return learned entity variances.

    This trains an actual model - no simulation or shortcuts.
    """
    model = SimpleGPKGE(num_entities, num_relations, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Create dataloader
    heads = torch.tensor(train_triples[:, 0], dtype=torch.long)
    relations = torch.tensor(train_triples[:, 1], dtype=torch.long)
    tails = torch.tensor(train_triples[:, 2], dtype=torch.long)

    dataset = TensorDataset(heads, relations, tails)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)

    for epoch in pbar:
        total_loss = 0
        for batch_h, batch_r, batch_t in dataloader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            # Positive scores
            pos_scores = model(batch_h, batch_r, batch_t, use_sampling=True)

            # Negative sampling (corrupt tails)
            neg_t = torch.randint(0, num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t, use_sampling=True)

            # BCE loss
            loss = criterion(pos_scores, torch.ones_like(pos_scores))
            loss += criterion(neg_scores, torch.zeros_like(neg_scores))

            # KL regularization (this is key - encourages variance learning)
            loss += kl_weight * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            pbar.set_postfix({'loss': total_loss / len(dataloader)})

    return model.get_entity_variance()


def compute_entity_frequency(triples, num_entities):
    """Compute frequency of each entity in triples."""
    freq = np.zeros(num_entities)
    for h, r, t in triples:
        freq[h] += 1
        freq[t] += 1
    return freq


def compute_coverage_matrix(triples, num_entities, num_relations):
    """Compute binary coverage matrix: C[e, r] = 1 if entity e seen with relation r."""
    coverage = np.zeros((num_entities, num_relations))
    for h, r, t in triples:
        coverage[h, r] = 1
        coverage[t, r] = 1
    return coverage


def create_ood_splits(train_triples, test_triples, coverage, freq, tau_percentile=20):
    """
    Create emerging entity and novel context OOD splits.

    Args:
        train_triples: Training triples
        test_triples: Test triples (used as ID)
        coverage: Coverage matrix from training
        freq: Entity frequency from training
        tau_percentile: Percentile threshold for "low frequency" entities

    Returns:
        id_triples: In-distribution test triples
        emerging_triples: Triples with low-frequency entities
        novel_ctx_triples: Triples with high-freq entities but uncovered relations
    """
    tau = np.percentile(freq[freq > 0], tau_percentile)

    id_triples = []
    emerging_triples = []
    novel_ctx_triples = []

    for h, r, t in test_triples:
        min_freq = min(freq[h], freq[t])
        h_covered = coverage[h, r] == 1
        t_covered = coverage[t, r] == 1
        both_covered = h_covered and t_covered

        if min_freq < tau:
            # Emerging entity: at least one entity is low-frequency
            emerging_triples.append((h, r, t))
        elif not both_covered:
            # Novel context: high-freq entities but missing coverage
            novel_ctx_triples.append((h, r, t))
        else:
            # ID: high-freq and both covered
            id_triples.append((h, r, t))

    return (
        np.array(id_triples) if id_triples else np.array([]).reshape(0, 3),
        np.array(emerging_triples) if emerging_triples else np.array([]).reshape(0, 3),
        np.array(novel_ctx_triples) if novel_ctx_triples else np.array([]).reshape(0, 3)
    )


def compute_semantic_uncertainty(triples, variance):
    """Compute semantic uncertainty for triples: (var_h + var_t) / 2"""
    if len(triples) == 0:
        return np.array([])
    heads = triples[:, 0].astype(int)
    tails = triples[:, 2].astype(int)
    return (variance[heads] + variance[tails]) / 2


def compute_structural_uncertainty(triples, coverage):
    """Compute structural uncertainty: 2 - c(h,r) - c(t,r)"""
    if len(triples) == 0:
        return np.array([])
    heads = triples[:, 0].astype(int)
    rels = triples[:, 1].astype(int)
    tails = triples[:, 2].astype(int)
    return 2.0 - coverage[heads, rels] - coverage[tails, rels]


def normalize_to_range(x, target_min=0, target_max=2, ref_min=None, ref_max=None):
    """Normalize array to [target_min, target_max] range."""
    if len(x) == 0:
        return x
    if ref_min is None:
        ref_min = x.min()
    if ref_max is None:
        ref_max = x.max()
    if ref_max - ref_min < 1e-8:
        return np.full_like(x, (target_min + target_max) / 2)
    x_clipped = np.clip(x, ref_min, ref_max)
    return target_min + (x_clipped - ref_min) / (ref_max - ref_min) * (target_max - target_min)


def compute_auroc(scores_pos, scores_neg):
    """Compute AUROC: P(score_pos > score_neg)"""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return 0.5

    from scipy.stats import mannwhitneyu
    try:
        stat, _ = mannwhitneyu(scores_pos, scores_neg, alternative='greater')
        auroc = stat / (len(scores_pos) * len(scores_neg))
    except ValueError:
        auroc = 0.5

    return auroc


def compute_kl_divergence(p_samples, q_samples, num_bins=50):
    """Compute KL divergence between two sample distributions using histograms."""
    if len(p_samples) == 0 or len(q_samples) == 0:
        return float('inf')

    all_samples = np.concatenate([p_samples, q_samples])
    bins = np.linspace(all_samples.min(), all_samples.max(), num_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))

    return kl


def verify_assumptions(dataset_name, train_dataset, test_dataset, device='cpu',
                       train_epochs=30, use_cache=True):
    """Verify all assumptions for a single dataset using trained model variances."""

    print(f"\n{'='*60}")
    print(f"Verifying assumptions for: {dataset_name}")
    print(f"{'='*60}")

    train_triples = train_dataset.triples
    test_triples = test_dataset.triples
    num_entities = train_dataset.num_entities
    num_relations = train_dataset.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train triples: {len(train_triples)}, Test triples: {len(test_triples)}")

    # Check for cached variance
    cache_path = Path(__file__).parent.parent / "outputs" / f"variance_{dataset_name.replace('-', '_').lower()}.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        print(f"Loading cached variance from {cache_path}")
        variance = np.load(cache_path)
    else:
        # Train actual model and extract variance
        print(f"\nTraining GP-KGE model ({train_epochs} epochs)...")
        variance = train_model(
            train_triples,
            num_entities,
            num_relations,
            epochs=train_epochs,
            device=device,
            verbose=True
        )
        # Cache the variance
        np.save(cache_path, variance)
        print(f"Saved variance to {cache_path}")

    # Compute basic statistics
    freq = compute_entity_frequency(train_triples, num_entities)
    coverage = compute_coverage_matrix(train_triples, num_entities, num_relations)

    # Create OOD splits
    id_triples, emerging_triples, novel_ctx_triples = create_ood_splits(
        train_triples, test_triples, coverage, freq, tau_percentile=20
    )

    print(f"\nOOD Split sizes:")
    print(f"  ID triples: {len(id_triples)}")
    print(f"  Emerging entity triples: {len(emerging_triples)}")
    print(f"  Novel context triples: {len(novel_ctx_triples)}")

    results = {}

    # =========================================================================
    # (A1) Variance-frequency monotonicity - NOW WITH REAL LEARNED VARIANCE
    # =========================================================================
    mask = freq > 0
    spearman_corr, spearman_p = stats.spearmanr(freq[mask], variance[mask])

    results['A1_spearman_rho'] = spearman_corr
    results['A1_p_value'] = spearman_p

    print(f"\n(A1) Variance-frequency monotonicity (LEARNED VARIANCE):")
    print(f"  Spearman rho: {spearman_corr:.3f}")
    print(f"  p-value: {spearman_p:.2e}")
    print(f"  Status: {'PASS' if spearman_corr < -0.3 and spearman_p < 0.01 else 'WEAK'}")

    # =========================================================================
    # (A2) ID coverage: P(c(h,r)=c(t,r)=1 | ID)
    # =========================================================================
    if len(id_triples) > 0:
        id_structural = compute_structural_uncertainty(id_triples, coverage)
        id_coverage_rate = np.mean(id_structural == 0)
    else:
        id_coverage_rate = 1.0

    results['A2_coverage_rate'] = id_coverage_rate

    print(f"\n(A2) ID coverage rate:")
    print(f"  P(both covered | ID): {id_coverage_rate:.3f}")

    # =========================================================================
    # (A3) Frequency overlap: KL divergence between novel-ctx and ID frequencies
    # =========================================================================
    if len(id_triples) > 0 and len(novel_ctx_triples) > 0:
        id_freqs = np.concatenate([freq[id_triples[:, 0].astype(int)],
                                    freq[id_triples[:, 2].astype(int)]])
        novel_freqs = np.concatenate([freq[novel_ctx_triples[:, 0].astype(int)],
                                       freq[novel_ctx_triples[:, 2].astype(int)]])
        kl_div = compute_kl_divergence(novel_freqs, id_freqs)
    else:
        kl_div = 0.0

    results['A3_kl_divergence'] = kl_div

    print(f"\n(A3) Frequency overlap:")
    print(f"  KL(freq_novel || freq_ID): {kl_div:.3f}")

    # =========================================================================
    # (A4) Bounded semantic gap: Delta = max[U_sem(ID) - U_sem(novel)]
    # =========================================================================
    if len(id_triples) > 0 and len(novel_ctx_triples) > 0:
        id_sem = compute_semantic_uncertainty(id_triples, variance)
        novel_sem = compute_semantic_uncertainty(novel_ctx_triples, variance)

        all_sem = np.concatenate([id_sem, novel_sem])
        ref_min, ref_max = all_sem.min(), all_sem.max()

        id_sem_norm = normalize_to_range(id_sem, 0, 2, ref_min, ref_max)
        novel_sem_norm = normalize_to_range(novel_sem, 0, 2, ref_min, ref_max)

        id_high = np.percentile(id_sem_norm, 95)
        novel_low = np.percentile(novel_sem_norm, 5)
        delta = max(0, id_high - novel_low)

        delta_max = max(0, id_sem_norm.max() - novel_sem_norm.min())
        max_valid_alpha = 1.0 / (1.0 + delta) if delta < 10 else 0.0
    else:
        delta = 0.0
        delta_max = 0.0
        max_valid_alpha = 1.0

    results['A4_delta'] = delta
    results['A4_delta_max'] = delta_max
    results['A4_max_valid_alpha'] = max_valid_alpha

    print(f"\n(A4) Bounded semantic gap:")
    print(f"  Delta (95th-5th percentile): {delta:.3f}")
    print(f"  Requirement: Delta < 1? {'PASS' if delta < 1 else 'FAIL'}")
    print(f"  Max valid alpha (1/(1+Delta)): {max_valid_alpha:.3f}")

    # =========================================================================
    # (A5) Non-degenerate coverage: rho = P(U_str=0 | emerging)
    # =========================================================================
    if len(emerging_triples) > 0:
        emerging_structural = compute_structural_uncertainty(emerging_triples, coverage)
        rho = np.mean(emerging_structural == 0)
        predicted_auroc = 1.0 - rho

        if len(id_triples) > 0:
            id_structural = compute_structural_uncertainty(id_triples, coverage)
            observed_auroc = compute_auroc(emerging_structural, id_structural)
        else:
            observed_auroc = predicted_auroc
    else:
        rho = 0.0
        predicted_auroc = 1.0
        observed_auroc = 1.0

    results['A5_rho'] = rho
    results['A5_predicted_auroc'] = predicted_auroc
    results['A5_observed_auroc'] = observed_auroc

    print(f"\n(A5) Non-degenerate coverage:")
    print(f"  rho (P(U_str=0 | emerging)): {rho:.3f}")
    print(f"  Predicted AUROC (1-rho): {predicted_auroc:.3f}")
    print(f"  Observed AUROC: {observed_auroc:.3f}")

    # =========================================================================
    # (A6) Semantic separation: AUROC(U_sem, ID, emerging)
    # =========================================================================
    if len(id_triples) > 0 and len(emerging_triples) > 0:
        id_sem = compute_semantic_uncertainty(id_triples, variance)
        emerging_sem = compute_semantic_uncertainty(emerging_triples, variance)
        semantic_auroc = compute_auroc(emerging_sem, id_sem)
    else:
        semantic_auroc = 1.0

    results['A6_semantic_auroc'] = semantic_auroc

    print(f"\n(A6) Semantic separation:")
    print(f"  AUROC(U_sem, ID, emerging): {semantic_auroc:.3f}")

    # =========================================================================
    # Verify theorem predictions
    # =========================================================================
    print(f"\n--- Theorem Prediction Verification ---")

    # Part (i): Semantic on novel contexts should be ~0.5
    if len(id_triples) > 0 and len(novel_ctx_triples) > 0:
        id_sem = compute_semantic_uncertainty(id_triples, variance)
        novel_sem = compute_semantic_uncertainty(novel_ctx_triples, variance)
        sem_novel_auroc = compute_auroc(novel_sem, id_sem)
        results['verify_part_i_auroc'] = sem_novel_auroc
        print(f"  Part (i) - Semantic AUROC on novel contexts: {sem_novel_auroc:.3f} (should be ~0.5)")

    # Part (iii): Structural on novel contexts should be 1.0
    if len(id_triples) > 0 and len(novel_ctx_triples) > 0:
        id_str = compute_structural_uncertainty(id_triples, coverage)
        novel_str = compute_structural_uncertainty(novel_ctx_triples, coverage)
        str_novel_auroc = compute_auroc(novel_str, id_str)
        results['verify_part_iii_auroc'] = str_novel_auroc
        print(f"  Part (iii) - Structural AUROC on novel contexts: {str_novel_auroc:.3f} (should be 1.0)")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify theorem assumptions with trained models')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per dataset')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--no-cache', action='store_true', help='Disable variance caching')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("="*60)
    print("THEOREM ASSUMPTION VERIFICATION")
    print("Using TRAINED MODEL VARIANCES (not simulation)")
    print("="*60)

    all_results = {}

    # Load and verify each dataset
    datasets = [
        ('FB15k-237', load_fb15k237),
        ('WN18RR', load_wn18rr),
        ('YAGO3-10', load_yago310),
    ]

    for name, loader in datasets:
        try:
            print(f"\nLoading {name}...")
            train, valid, test = loader()
            results = verify_assumptions(
                name, train, test,
                device=device,
                train_epochs=args.epochs,
                use_cache=not args.no_cache
            )
            all_results[name] = results
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR PAPER (TRAINED MODEL VARIANCES)")
    print("="*80)

    print("\n%-40s %12s %12s %12s" % ("Metric", "FB15k-237", "WN18RR", "YAGO3-10"))
    print("-"*80)

    metrics = [
        ('A1_spearman_rho', '(A1) Spearman rho(freq, var)'),
        ('A2_coverage_rate', '(A2) ID coverage rate'),
        ('A3_kl_divergence', '(A3) KL divergence'),
        ('A4_delta', '(A4) Delta'),
        ('A4_max_valid_alpha', '(A4) Max valid alpha'),
        ('A5_rho', '(A5) rho'),
        ('A5_predicted_auroc', '(A5) Predicted AUROC'),
        ('A5_observed_auroc', '(A5) Observed AUROC'),
        ('A6_semantic_auroc', '(A6) Semantic AUROC'),
        ('verify_part_i_auroc', 'Part (i): Sem on novel'),
        ('verify_part_iii_auroc', 'Part (iii): Str on novel'),
    ]

    for key, label in metrics:
        values = []
        for ds in ['FB15k-237', 'WN18RR', 'YAGO3-10']:
            if ds in all_results and key in all_results[ds]:
                values.append(f"{all_results[ds][key]:.3f}")
            else:
                values.append("N/A")
        print("%-40s %12s %12s %12s" % (label, values[0], values[1], values[2]))

    # Save results to JSON
    output_path = Path(__file__).parent.parent / "outputs" / "theorem_assumptions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_results = {
        k: {k2: convert_numpy(v2) for k2, v2 in v.items()}
        for k, v in all_results.items()
    }

    # Add metadata about methodology
    json_results['_metadata'] = {
        'variance_source': 'trained_model',
        'model_type': 'SimpleGPKGE (variational)',
        'training_epochs': args.epochs,
        'note': 'Variances extracted from trained GP-KGE models, not simulated'
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE (for paper)")
    print("="*80)

    print(r"""
\begin{table}[t]
\centering
\caption{\textbf{Empirical verification of theorem assumptions} using learned variances from trained GP-KGE models.}
\label{tab:assumptions}
\small
\begin{tabular}{llccc}
\toprule
 & & FB15k & WN18RR & YAGO \\
\midrule
\multicolumn{5}{l}{\textit{(A1) Variance-frequency monotonicity}} \\""")

    print(f"& Spearman $\\rho$ & {all_results.get('FB15k-237', {}).get('A1_spearman_rho', 0):.2f} & {all_results.get('WN18RR', {}).get('A1_spearman_rho', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A1_spearman_rho', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{(A2) ID coverage}} \\")
    print(f"& Coverage rate & {all_results.get('FB15k-237', {}).get('A2_coverage_rate', 0):.2f} & {all_results.get('WN18RR', {}).get('A2_coverage_rate', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A2_coverage_rate', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{(A3) Frequency overlap}} \\")
    print(f"& KL divergence & {all_results.get('FB15k-237', {}).get('A3_kl_divergence', 0):.2f} & {all_results.get('WN18RR', {}).get('A3_kl_divergence', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A3_kl_divergence', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{(A4) Bounded semantic gap}} \\")
    print(f"& $\\Delta$ & {all_results.get('FB15k-237', {}).get('A4_delta', 0):.2f} & {all_results.get('WN18RR', {}).get('A4_delta', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A4_delta', 0):.2f} \\\\")
    print(f"& Max $\\alpha$ & {all_results.get('FB15k-237', {}).get('A4_max_valid_alpha', 0):.2f} & {all_results.get('WN18RR', {}).get('A4_max_valid_alpha', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A4_max_valid_alpha', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{(A5) Non-degenerate coverage}} \\")
    print(f"& $\\rho$ & {all_results.get('FB15k-237', {}).get('A5_rho', 0):.2f} & {all_results.get('WN18RR', {}).get('A5_rho', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A5_rho', 0):.2f} \\\\")
    print(f"& Predicted AUROC & {all_results.get('FB15k-237', {}).get('A5_predicted_auroc', 0):.2f} & {all_results.get('WN18RR', {}).get('A5_predicted_auroc', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A5_predicted_auroc', 0):.2f} \\\\")
    print(f"& Observed AUROC & {all_results.get('FB15k-237', {}).get('A5_observed_auroc', 0):.2f} & {all_results.get('WN18RR', {}).get('A5_observed_auroc', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A5_observed_auroc', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{(A6) Semantic separation}} \\")
    print(f"& AUROC & {all_results.get('FB15k-237', {}).get('A6_semantic_auroc', 0):.2f} & {all_results.get('WN18RR', {}).get('A6_semantic_auroc', 0):.2f} & {all_results.get('YAGO3-10', {}).get('A6_semantic_auroc', 0):.2f} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{Theorem predictions}} \\")
    print(f"& Part (i): Sem. on novel $\\approx 0.5$ & {all_results.get('FB15k-237', {}).get('verify_part_i_auroc', 0):.2f} & {all_results.get('WN18RR', {}).get('verify_part_i_auroc', 0):.2f} & {all_results.get('YAGO3-10', {}).get('verify_part_i_auroc', 0):.2f} \\\\")
    print(f"& Part (iii): Str. on novel $= 1.0$ & {all_results.get('FB15k-237', {}).get('verify_part_iii_auroc', 0):.2f} & {all_results.get('WN18RR', {}).get('verify_part_iii_auroc', 0):.2f} & {all_results.get('YAGO3-10', {}).get('verify_part_iii_auroc', 0):.2f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
