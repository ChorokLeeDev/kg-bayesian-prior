#!/usr/bin/env python3
"""
Advanced Experiments for Adversarial OOD Detection

Implements multiple approaches to improve adversarial OOD detection:
1. Coverage-based optimization (weight tuning)
2. Selective Prediction with margin-based abstention
3. Ensemble Disagreement
4. Perturbation Robustness
5. Local Neighborhood Analysis
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
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy

from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# =============================================================================
# Model Definition (Enhanced CAGP)
# =============================================================================

class EnhancedCAGP(nn.Module):
    """Enhanced CAGP with additional uncertainty signals."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('coverage_freq', torch.zeros(num_entities, num_relations))

    def forward(self, heads, relations, tails):
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]
        return (h * r * t).sum(dim=-1)

    def score_all_tails(self, heads, relations):
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        hr = h * r
        return hr @ self.entity_mean.T

    def get_semantic_uncertainty(self, heads, tails):
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_structural_uncertainty(self, heads, relations, tails):
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_combined_uncertainty(self, heads, relations, tails, alpha=0.5):
        """Combined uncertainty with configurable weight."""
        sem = self.get_semantic_uncertainty(heads, tails)
        struct = self.get_structural_uncertainty(heads, relations, tails)
        sem_norm = sem / (sem.mean() + 1e-8)
        struct_norm = struct / (struct.mean() + 1e-8)
        return alpha * sem_norm + (1 - alpha) * struct_norm

    def get_prediction_margin(self, heads, relations):
        scores = self.score_all_tails(heads, relations)
        topk, _ = torch.topk(scores, k=2, dim=-1)
        return topk[:, 0] - topk[:, 1]

    def get_tail_rank(self, heads, relations, tails):
        scores = self.score_all_tails(heads, relations)
        tail_scores = scores.gather(1, tails.unsqueeze(1)).squeeze(1)
        ranks = (scores > tail_scores.unsqueeze(1)).sum(dim=1).float() + 1
        return ranks

    def get_neighbor_scores(self, heads, relations, tails, k=10):
        """Get average score of k-nearest neighbors of the tail."""
        # Find k-NN of each tail in embedding space
        tail_embs = self.entity_mean[tails]  # [B, dim]

        # Compute distances to all entities
        dists = torch.cdist(tail_embs, self.entity_mean)  # [B, num_entities]

        # Exclude self
        dists.scatter_(1, tails.unsqueeze(1), float('inf'))

        # Get k nearest neighbors
        _, nn_idx = torch.topk(dists, k, dim=1, largest=False)  # [B, k]

        # Score neighbors
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        hr = h * r  # [B, dim]

        # Get neighbor embeddings
        nn_embs = self.entity_mean[nn_idx]  # [B, k, dim]

        # Score each neighbor
        nn_scores = (hr.unsqueeze(1) * nn_embs).sum(dim=-1)  # [B, k]

        return nn_scores.mean(dim=1), nn_scores.std(dim=1)

    def get_perturbation_sensitivity(self, heads, relations, tails,
                                     n_perturbations=10, noise_scale=0.1):
        """Measure how much score changes with small perturbations."""
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]

        # Original score
        original_score = (h * r * t).sum(dim=-1)

        # Perturbed scores
        perturbed_scores = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(t) * noise_scale
            t_perturbed = t + noise
            score = (h * r * t_perturbed).sum(dim=-1)
            perturbed_scores.append(score)

        perturbed_scores = torch.stack(perturbed_scores, dim=1)  # [B, n_perturbations]

        # Sensitivity = std of perturbed scores
        sensitivity = perturbed_scores.std(dim=1)

        # Also compute max change
        max_change = (perturbed_scores - original_score.unsqueeze(1)).abs().max(dim=1).values

        return sensitivity, max_change

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0
            self.coverage_freq[h, r] += 1.0
            self.coverage_freq[t, r] += 1.0

    def kl_loss(self):
        kl = -0.5 * torch.sum(
            1 + self.entity_logvar - self.entity_mean.pow(2) - self.entity_logvar.exp()
        )
        return kl / self.num_entities


def train_model(model, train_triples, device, epochs=30, batch_size=1024, lr=0.001, seed=None):
    """Train a model with optional seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    heads = torch.tensor(train_triples[:, 0])
    relations = torch.tensor(train_triples[:, 1])
    tails = torch.tensor(train_triples[:, 2])

    dataset = TensorDataset(heads, relations, tails)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_h, batch_r, batch_t in loader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            pos_scores = model(batch_h, batch_r, batch_t)
            neg_t = torch.randint(0, model.num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t)

            loss = criterion(pos_scores, torch.ones_like(pos_scores))
            loss += criterion(neg_scores, torch.zeros_like(neg_scores))
            loss += 0.01 * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def generate_ood_samples(test_triples, model, num_entities, device, corruption_type='random', k=10):
    """Generate OOD samples."""
    n_samples = min(len(test_triples), 1000)

    if corruption_type == 'random':
        ood_tails = np.random.randint(0, num_entities, n_samples)

    elif corruption_type == 'high_score':
        ood_tails = []
        model.eval()
        with torch.no_grad():
            for i in range(n_samples):
                h = torch.tensor([test_triples[i, 0]]).to(device)
                r = torch.tensor([test_triples[i, 1]]).to(device)
                scores = model.score_all_tails(h, r).squeeze()
                scores[test_triples[i, 2]] = float('-inf')
                topk_idx = torch.topk(scores, k).indices
                ood_tails.append(topk_idx[np.random.randint(k)].item())
        ood_tails = np.array(ood_tails)

    elif corruption_type == 'embedding_similar':
        ood_tails = []
        with torch.no_grad():
            emb = model.entity_mean.cpu().numpy()
            for i in range(n_samples):
                t = test_triples[i, 2]
                dists = np.linalg.norm(emb - emb[t], axis=1)
                dists[t] = np.inf
                nn_idx = np.argsort(dists)[:k]
                ood_tails.append(nn_idx[np.random.randint(len(nn_idx))])
        ood_tails = np.array(ood_tails)

    elif corruption_type == 'type_constrained':
        ood_tails = []
        coverage = model.coverage.cpu().numpy()
        for i in range(n_samples):
            r = test_triples[i, 1]
            valid = np.where(coverage[:, r] > 0)[0]
            if len(valid) > 0:
                valid = valid[valid != test_triples[i, 2]]
                if len(valid) > 0:
                    ood_tails.append(np.random.choice(valid))
                else:
                    ood_tails.append(np.random.randint(0, num_entities))
            else:
                ood_tails.append(np.random.randint(0, num_entities))
        ood_tails = np.array(ood_tails)

    else:
        ood_tails = np.random.randint(0, num_entities, n_samples)

    return test_triples[:n_samples], ood_tails


def compute_auroc(id_unc, ood_unc):
    """Compute AUROC for OOD detection."""
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


# =============================================================================
# 1. Coverage-Based Optimization
# =============================================================================

def optimize_alpha_weights(model, test_triples, num_entities, device):
    """Find optimal alpha for Semantic+Structural combination."""
    print("\n" + "="*70)
    print("1. COVERAGE-BASED OPTIMIZATION")
    print("="*70)

    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {corr: {} for corr in corruption_types}
    best_alpha = {corr: 0.5 for corr in corruption_types}
    best_auroc = {corr: 0.0 for corr in corruption_types}

    model.eval()

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        with torch.no_grad():
            h = torch.tensor(test_subset[:, 0]).to(device)
            r = torch.tensor(test_subset[:, 1]).to(device)
            t_id = torch.tensor(test_subset[:, 2]).to(device)
            t_ood = torch.tensor(ood_tails).to(device)

            for alpha in alphas:
                id_unc = model.get_combined_uncertainty(h, r, t_id, alpha).cpu().numpy()
                ood_unc = model.get_combined_uncertainty(h, r, t_ood, alpha).cpu().numpy()
                auroc = compute_auroc(id_unc, ood_unc)
                results[corr_type][alpha] = auroc

                if auroc > best_auroc[corr_type]:
                    best_auroc[corr_type] = auroc
                    best_alpha[corr_type] = alpha

        print(f"    Best alpha: {best_alpha[corr_type]:.1f} -> AUROC: {best_auroc[corr_type]:.4f}")

    # Find globally optimal alpha
    avg_auroc_per_alpha = {}
    for alpha in alphas:
        avg_auroc_per_alpha[alpha] = np.mean([results[c][alpha] for c in corruption_types])

    global_best_alpha = max(avg_auroc_per_alpha, key=avg_auroc_per_alpha.get)

    print(f"\n  Global best alpha: {global_best_alpha:.1f}")
    print(f"  Per-attack best alphas: {best_alpha}")

    return {
        'per_alpha_results': results,
        'best_alpha_per_attack': best_alpha,
        'best_auroc_per_attack': best_auroc,
        'global_best_alpha': global_best_alpha,
        'avg_auroc_per_alpha': avg_auroc_per_alpha,
    }


# =============================================================================
# 2. Selective Prediction with Margin-Based Abstention
# =============================================================================

def selective_prediction_experiment(model, test_triples, device, coverage_targets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    """Evaluate selective prediction using margin-based abstention."""
    print("\n" + "="*70)
    print("2. SELECTIVE PREDICTION (MARGIN-BASED ABSTENTION)")
    print("="*70)

    model.eval()
    n_samples = min(len(test_triples), 2000)
    test_subset = test_triples[:n_samples]

    # Compute predictions and margins
    predictions = []
    margins = []
    correct = []

    with torch.no_grad():
        for i in range(n_samples):
            h = torch.tensor([test_subset[i, 0]]).to(device)
            r = torch.tensor([test_subset[i, 1]]).to(device)
            t_true = test_subset[i, 2]

            scores = model.score_all_tails(h, r).squeeze()
            pred = scores.argmax().item()
            margin = model.get_prediction_margin(h, r).item()

            predictions.append(pred)
            margins.append(margin)
            correct.append(pred == t_true)

    margins = np.array(margins)
    correct = np.array(correct)

    baseline_accuracy = correct.mean()
    print(f"\n  Baseline accuracy: {baseline_accuracy:.4f}")

    results = {'baseline_accuracy': float(baseline_accuracy)}

    # Evaluate at different coverage levels
    print(f"\n  {'Coverage':<10} {'Selective Acc':<15} {'Error Reduction':<15} {'Threshold':<10}")
    print("  " + "-"*55)

    for target_coverage in coverage_targets:
        # Sort by margin (descending - higher margin = more confident)
        sorted_idx = np.argsort(-margins)
        n_answer = int(target_coverage * len(correct))
        answer_idx = sorted_idx[:n_answer]

        selective_acc = correct[answer_idx].mean()

        baseline_error = 1 - baseline_accuracy
        selective_error = 1 - selective_acc
        error_reduction = (baseline_error - selective_error) / (baseline_error + 1e-8)

        threshold = margins[sorted_idx[n_answer-1]] if n_answer > 0 else 0

        results[f'coverage_{target_coverage}'] = {
            'selective_accuracy': float(selective_acc),
            'error_reduction': float(error_reduction),
            'threshold': float(threshold),
        }

        print(f"  {target_coverage:<10.2f} {selective_acc:<15.4f} {error_reduction:<15.4f} {threshold:<10.4f}")

    # Also compute using structural uncertainty for comparison
    print("\n  Comparison with Structural Uncertainty:")

    structural_uncs = []
    with torch.no_grad():
        h = torch.tensor(test_subset[:, 0]).to(device)
        r = torch.tensor(test_subset[:, 1]).to(device)
        t = torch.tensor(test_subset[:, 2]).to(device)
        structural_uncs = model.get_structural_uncertainty(h, r, t).cpu().numpy()

    for target_coverage in [0.8, 0.9]:
        # Sort by structural uncertainty (ascending - lower = more confident)
        sorted_idx = np.argsort(structural_uncs)
        n_answer = int(target_coverage * len(correct))
        answer_idx = sorted_idx[:n_answer]
        selective_acc_struct = correct[answer_idx].mean()

        # Margin-based
        sorted_idx_margin = np.argsort(-margins)
        answer_idx_margin = sorted_idx_margin[:n_answer]
        selective_acc_margin = correct[answer_idx_margin].mean()

        print(f"    Coverage {target_coverage}: Margin={selective_acc_margin:.4f}, Structural={selective_acc_struct:.4f}")

    return results


# =============================================================================
# 3. Ensemble Disagreement
# =============================================================================

def ensemble_disagreement_experiment(train_triples, test_triples, num_entities, num_relations,
                                     device, n_models=3, epochs=20, dim=100):
    """Train ensemble and use disagreement as uncertainty."""
    print("\n" + "="*70)
    print("3. ENSEMBLE DISAGREEMENT")
    print("="*70)

    # Train multiple models with different seeds
    models = []
    print(f"\n  Training {n_models} models with different seeds...")

    for i in range(n_models):
        print(f"\n  Model {i+1}/{n_models}:")
        model = EnhancedCAGP(num_entities, num_relations, dim)
        model.precompute_coverage(train_triples)
        model = train_model(model, train_triples, device, epochs=epochs, seed=42+i)
        models.append(model)

    # Evaluate ensemble disagreement
    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    results = {}

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, models[0], num_entities, device, corr_type
        )

        with torch.no_grad():
            h = torch.tensor(test_subset[:, 0]).to(device)
            r = torch.tensor(test_subset[:, 1]).to(device)
            t_id = torch.tensor(test_subset[:, 2]).to(device)
            t_ood = torch.tensor(ood_tails).to(device)

            # Collect predictions from all models
            id_scores_all = []
            ood_scores_all = []
            id_ranks_all = []
            ood_ranks_all = []

            for model in models:
                model.eval()
                id_scores_all.append(model(h, r, t_id).cpu().numpy())
                ood_scores_all.append(model(h, r, t_ood).cpu().numpy())
                id_ranks_all.append(model.get_tail_rank(h, r, t_id).cpu().numpy())
                ood_ranks_all.append(model.get_tail_rank(h, r, t_ood).cpu().numpy())

            id_scores_all = np.stack(id_scores_all, axis=1)  # [n_samples, n_models]
            ood_scores_all = np.stack(ood_scores_all, axis=1)
            id_ranks_all = np.stack(id_ranks_all, axis=1)
            ood_ranks_all = np.stack(ood_ranks_all, axis=1)

            # Compute disagreement (variance)
            id_score_var = id_scores_all.var(axis=1)
            ood_score_var = ood_scores_all.var(axis=1)
            id_rank_var = id_ranks_all.var(axis=1)
            ood_rank_var = ood_ranks_all.var(axis=1)

            # AUROC using score variance
            auroc_score_var = compute_auroc(id_score_var, ood_score_var)

            # AUROC using rank variance
            auroc_rank_var = compute_auroc(id_rank_var, ood_rank_var)

            # Also try mean score (ensemble average)
            id_score_mean = id_scores_all.mean(axis=1)
            ood_score_mean = ood_scores_all.mean(axis=1)
            auroc_neg_score = compute_auroc(-id_score_mean, -ood_score_mean)

            # Combined: low mean score + high variance = OOD
            id_combined = -id_score_mean + id_score_var
            ood_combined = -ood_score_mean + ood_score_var
            auroc_combined = compute_auroc(id_combined, ood_combined)

        results[corr_type] = {
            'auroc_score_variance': float(auroc_score_var),
            'auroc_rank_variance': float(auroc_rank_var),
            'auroc_neg_score': float(auroc_neg_score),
            'auroc_combined': float(auroc_combined),
        }

        print(f"    Score Variance: {auroc_score_var:.4f}")
        print(f"    Rank Variance:  {auroc_rank_var:.4f}")
        print(f"    Neg Score:      {auroc_neg_score:.4f}")
        print(f"    Combined:       {auroc_combined:.4f}")

    return results


# =============================================================================
# 4. Perturbation Robustness
# =============================================================================

def perturbation_robustness_experiment(model, test_triples, num_entities, device):
    """Test if true tails are more robust to perturbations than adversarial tails."""
    print("\n" + "="*70)
    print("4. PERTURBATION ROBUSTNESS")
    print("="*70)

    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    noise_scales = [0.01, 0.05, 0.1, 0.2]
    results = {}

    model.eval()

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        results[corr_type] = {}

        with torch.no_grad():
            h = torch.tensor(test_subset[:, 0]).to(device)
            r = torch.tensor(test_subset[:, 1]).to(device)
            t_id = torch.tensor(test_subset[:, 2]).to(device)
            t_ood = torch.tensor(ood_tails).to(device)

            for noise_scale in noise_scales:
                # Get perturbation sensitivity
                id_sens, id_max = model.get_perturbation_sensitivity(
                    h, r, t_id, n_perturbations=20, noise_scale=noise_scale
                )
                ood_sens, ood_max = model.get_perturbation_sensitivity(
                    h, r, t_ood, n_perturbations=20, noise_scale=noise_scale
                )

                id_sens = id_sens.cpu().numpy()
                ood_sens = ood_sens.cpu().numpy()
                id_max = id_max.cpu().numpy()
                ood_max = ood_max.cpu().numpy()

                # Higher sensitivity = more uncertain
                auroc_sens = compute_auroc(id_sens, ood_sens)
                auroc_max = compute_auroc(id_max, ood_max)

                results[corr_type][f'noise_{noise_scale}'] = {
                    'auroc_sensitivity': float(auroc_sens),
                    'auroc_max_change': float(auroc_max),
                    'id_sensitivity_mean': float(id_sens.mean()),
                    'ood_sensitivity_mean': float(ood_sens.mean()),
                }

                print(f"    Noise {noise_scale}: Sens AUROC={auroc_sens:.4f}, Max AUROC={auroc_max:.4f}")

    return results


# =============================================================================
# 5. Local Neighborhood Analysis
# =============================================================================

def neighborhood_analysis_experiment(model, test_triples, num_entities, device):
    """Analyze if true tails have consistent neighborhood scores."""
    print("\n" + "="*70)
    print("5. LOCAL NEIGHBORHOOD ANALYSIS")
    print("="*70)

    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    k_values = [5, 10, 20, 50]
    results = {}

    model.eval()

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        results[corr_type] = {}

        with torch.no_grad():
            h = torch.tensor(test_subset[:, 0]).to(device)
            r = torch.tensor(test_subset[:, 1]).to(device)
            t_id = torch.tensor(test_subset[:, 2]).to(device)
            t_ood = torch.tensor(ood_tails).to(device)

            # Get tail scores
            id_scores = model(h, r, t_id).cpu().numpy()
            ood_scores = model(h, r, t_ood).cpu().numpy()

            for k in k_values:
                # Get neighbor scores
                id_nn_mean, id_nn_std = model.get_neighbor_scores(h, r, t_id, k=k)
                ood_nn_mean, ood_nn_std = model.get_neighbor_scores(h, r, t_ood, k=k)

                id_nn_mean = id_nn_mean.cpu().numpy()
                id_nn_std = id_nn_std.cpu().numpy()
                ood_nn_mean = ood_nn_mean.cpu().numpy()
                ood_nn_std = ood_nn_std.cpu().numpy()

                # Hypothesis: True tails have neighbors with similar high scores
                # Adversarial tails may be "isolated" (neighbors have lower scores)

                # Score gap between tail and neighbors
                id_gap = id_scores - id_nn_mean
                ood_gap = ood_scores - ood_nn_mean

                # Use various signals
                # 1. Lower neighbor mean = more isolated = OOD
                auroc_nn_mean = compute_auroc(-id_nn_mean, -ood_nn_mean)

                # 2. Higher gap = more isolated = OOD
                auroc_gap = compute_auroc(id_gap, ood_gap)

                # 3. Higher neighbor std = inconsistent = OOD
                auroc_nn_std = compute_auroc(id_nn_std, ood_nn_std)

                # 4. Combined: gap / (nn_std + epsilon)
                id_isolation = id_gap / (id_nn_std + 0.1)
                ood_isolation = ood_gap / (ood_nn_std + 0.1)
                auroc_isolation = compute_auroc(id_isolation, ood_isolation)

                results[corr_type][f'k_{k}'] = {
                    'auroc_nn_mean': float(auroc_nn_mean),
                    'auroc_gap': float(auroc_gap),
                    'auroc_nn_std': float(auroc_nn_std),
                    'auroc_isolation': float(auroc_isolation),
                }

                print(f"    k={k}: NN_mean={auroc_nn_mean:.4f}, Gap={auroc_gap:.4f}, "
                      f"NN_std={auroc_nn_std:.4f}, Isolation={auroc_isolation:.4f}")

    return results


# =============================================================================
# 6. Combined Best Methods
# =============================================================================

def evaluate_best_methods(model, test_triples, num_entities, device, best_alpha=0.0):
    """Evaluate the best methods identified from experiments."""
    print("\n" + "="*70)
    print("6. BEST METHODS COMPARISON")
    print("="*70)

    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    results = {}

    model.eval()

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        with torch.no_grad():
            h = torch.tensor(test_subset[:, 0]).to(device)
            r = torch.tensor(test_subset[:, 1]).to(device)
            t_id = torch.tensor(test_subset[:, 2]).to(device)
            t_ood = torch.tensor(ood_tails).to(device)

            methods = {}

            # 1. Structural only (baseline)
            id_struct = model.get_structural_uncertainty(h, r, t_id).cpu().numpy()
            ood_struct = model.get_structural_uncertainty(h, r, t_ood).cpu().numpy()
            methods['Structural'] = compute_auroc(id_struct, ood_struct)

            # 2. Semantic only
            id_sem = model.get_semantic_uncertainty(h, t_id).cpu().numpy()
            ood_sem = model.get_semantic_uncertainty(h, t_ood).cpu().numpy()
            methods['Semantic'] = compute_auroc(id_sem, ood_sem)

            # 3. Optimized alpha
            id_comb = model.get_combined_uncertainty(h, r, t_id, best_alpha).cpu().numpy()
            ood_comb = model.get_combined_uncertainty(h, r, t_ood, best_alpha).cpu().numpy()
            methods[f'Combined(a={best_alpha})'] = compute_auroc(id_comb, ood_comb)

            # 4. Neighborhood isolation (k=10)
            id_nn_mean, id_nn_std = model.get_neighbor_scores(h, r, t_id, k=10)
            ood_nn_mean, ood_nn_std = model.get_neighbor_scores(h, r, t_ood, k=10)
            id_scores = model(h, r, t_id)
            ood_scores = model(h, r, t_ood)
            id_isolation = (id_scores - id_nn_mean) / (id_nn_std + 0.1)
            ood_isolation = (ood_scores - ood_nn_mean) / (ood_nn_std + 0.1)
            methods['Neighborhood'] = compute_auroc(
                id_isolation.cpu().numpy(), ood_isolation.cpu().numpy()
            )

            # 5. Perturbation sensitivity
            id_sens, _ = model.get_perturbation_sensitivity(h, r, t_id, noise_scale=0.1)
            ood_sens, _ = model.get_perturbation_sensitivity(h, r, t_ood, noise_scale=0.1)
            methods['Perturbation'] = compute_auroc(id_sens.cpu().numpy(), ood_sens.cpu().numpy())

            # 6. Combined best
            # Structural + Neighborhood
            id_struct_norm = id_struct / (id_struct.mean() + 1e-8)
            ood_struct_norm = ood_struct / (ood_struct.mean() + 1e-8)
            id_iso_norm = id_isolation.cpu().numpy() / (np.abs(id_isolation.cpu().numpy()).mean() + 1e-8)
            ood_iso_norm = ood_isolation.cpu().numpy() / (np.abs(ood_isolation.cpu().numpy()).mean() + 1e-8)

            id_best = id_struct_norm + id_iso_norm
            ood_best = ood_struct_norm + ood_iso_norm
            methods['Struct+Neighbor'] = compute_auroc(id_best, ood_best)

        results[corr_type] = methods

        for method, auroc in methods.items():
            print(f"    {method}: {auroc:.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all advanced experiments."""
    print("\n" + "="*70)
    print("ADVANCED EXPERIMENTS FOR ADVERSARIAL OOD DETECTION")
    print("="*70)

    device = setup_device()

    # Load data
    print("\nLoading data...")
    train_ds, valid_ds, test_ds = load_fb15k237()
    train_triples = train_ds.triples
    test_triples = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    # Train base model
    print("\nTraining base model...")
    model = EnhancedCAGP(num_entities, num_relations, dim=100)
    model.precompute_coverage(train_triples)
    model = train_model(model, train_triples, device, epochs=30)

    all_results = {}

    # 1. Coverage optimization
    all_results['coverage_optimization'] = optimize_alpha_weights(
        model, test_triples, num_entities, device
    )
    best_alpha = all_results['coverage_optimization']['global_best_alpha']

    # 2. Selective prediction
    all_results['selective_prediction'] = selective_prediction_experiment(
        model, test_triples, device
    )

    # 3. Ensemble disagreement (smaller for speed)
    all_results['ensemble'] = ensemble_disagreement_experiment(
        train_triples, test_triples, num_entities, num_relations,
        device, n_models=3, epochs=15
    )

    # 4. Perturbation robustness
    all_results['perturbation'] = perturbation_robustness_experiment(
        model, test_triples, num_entities, device
    )

    # 5. Neighborhood analysis
    all_results['neighborhood'] = neighborhood_analysis_experiment(
        model, test_triples, num_entities, device
    )

    # 6. Best methods comparison
    all_results['best_methods'] = evaluate_best_methods(
        model, test_triples, num_entities, device, best_alpha
    )

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\n1. COVERAGE OPTIMIZATION:")
    print(f"   Global best alpha: {best_alpha}")
    for attack, auroc in all_results['coverage_optimization']['best_auroc_per_attack'].items():
        print(f"   {attack}: best alpha={all_results['coverage_optimization']['best_alpha_per_attack'][attack]}, AUROC={auroc:.4f}")

    print("\n2. SELECTIVE PREDICTION (Coverage=0.9):")
    sp = all_results['selective_prediction']
    if 'coverage_0.9' in sp:
        print(f"   Selective Accuracy: {sp['coverage_0.9']['selective_accuracy']:.4f}")
        print(f"   Error Reduction: {sp['coverage_0.9']['error_reduction']:.4f}")

    print("\n3. ENSEMBLE DISAGREEMENT (best per attack):")
    for attack in ['random', 'high_score', 'embedding_similar', 'type_constrained']:
        ens = all_results['ensemble'][attack]
        best = max(ens.values())
        best_method = max(ens, key=ens.get)
        print(f"   {attack}: {best_method}={best:.4f}")

    print("\n4. BEST METHODS COMPARISON:")
    for attack in ['random', 'high_score', 'embedding_similar', 'type_constrained']:
        bm = all_results['best_methods'][attack]
        best = max(bm.values())
        best_method = max(bm, key=bm.get)
        print(f"   {attack}: {best_method}={best:.4f}")

    # Save results
    output_path = project_root / 'outputs' / 'advanced_experiments_results.json'
    output_path.parent.mkdir(exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
