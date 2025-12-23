#!/usr/bin/env python3
"""
Full Experiment Suite for EMNLP Submission

Runs all experiments needed for the paper:
1. Main OOD detection (random corruption)
2. Adversarial OOD (targeted corruptions)
3. Temporal OOD (frequency-based)
4. QA Abstention (downstream task)
5. Method comparison (all approaches)

This script is designed to run on CPU for smaller experiments
and can be adapted for GPU/Colab for full-scale experiments.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import time
from collections import defaultdict

# Import our modules
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


class SimpleCAGP(nn.Module):
    """Simplified CAGP for quick experiments."""

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
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, heads, relations, tails):
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]
        return (h * r * t).sum(dim=-1)

    def get_gp_variance(self, heads, tails):
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_coverage_uncertainty(self, heads, relations, tails):
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_uncertainty(self, heads, relations, tails):
        gp_var = self.get_gp_variance(heads, tails)
        cov_unc = self.get_coverage_uncertainty(heads, relations, tails)
        gp_var_norm = gp_var / (gp_var.mean() + 1e-8) * (cov_unc.mean() + 1e-8)
        alpha = torch.sigmoid(self.alpha)
        return alpha * gp_var_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples):
        """Compute coverage from numpy triples array."""
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


class AttentionCAGP(nn.Module):
    """Attention-based CAGP."""

    def __init__(self, num_entities, num_relations, dim=100, hidden_dim=64):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(3 * dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, heads, relations, tails):
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]
        return (h * r * t).sum(dim=-1)

    def get_gp_variance(self, heads, tails):
        h_var = torch.exp(self.entity_logvar[heads]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[tails]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_coverage_uncertainty(self, heads, relations, tails):
        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        return 2.0 - h_seen - t_seen

    def get_uncertainty(self, heads, relations, tails):
        gp_var = self.get_gp_variance(heads, tails)
        cov_unc = self.get_coverage_uncertainty(heads, relations, tails)

        h_emb = self.entity_mean[heads]
        r_emb = self.relation_emb(relations)
        t_emb = self.entity_mean[tails]

        features = torch.cat([
            h_emb, r_emb, t_emb,
            gp_var.unsqueeze(-1),
            cov_unc.unsqueeze(-1)
        ], dim=-1)

        alpha = self.attention_net(features).squeeze(-1)

        gp_var_norm = gp_var / (gp_var.mean() + 1e-8) * (cov_unc.mean() + 1e-8)
        return alpha * gp_var_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


class RelCondVar(nn.Module):
    """Relation-Conditioned Variance model."""

    def __init__(self, num_entities, num_relations, dim=100, hidden_dim=128):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

        # Variance network
        self.variance_net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, heads, relations, tails):
        h = self.entity_mean[heads]
        r = self.relation_emb(relations)
        t = self.entity_mean[tails]
        return (h * r * t).sum(dim=-1)

    def get_entity_relation_variance(self, entities, relations):
        e_emb = self.entity_mean[entities]
        r_emb = self.relation_emb(relations)
        combined = torch.cat([e_emb, r_emb], dim=-1)
        raw_var = self.variance_net(combined).squeeze(-1)
        return torch.nn.functional.softplus(raw_var) + 1e-4

    def get_uncertainty(self, heads, relations, tails):
        h_var = self.get_entity_relation_variance(heads, relations)
        t_var = self.get_entity_relation_variance(tails, relations)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


class ScoreBasedUncertainty(nn.Module):
    """UKGE/Energy-style score-based uncertainty."""

    def __init__(self, num_entities, num_relations, dim=100, method='ukge'):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.method = method

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return (h * r * t).sum(dim=-1)

    def get_uncertainty(self, heads, relations, tails):
        scores = self.forward(heads, relations, tails)
        if self.method == 'ukge':
            # UKGE: uncertainty = 1 - |prob - 0.5| * 2
            probs = torch.sigmoid(scores)
            confidence = torch.abs(probs - 0.5) * 2
            return 1 - confidence
        else:  # energy
            # Energy: -score as uncertainty
            return -scores

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i]
            self.coverage[h, r] = 1.0
            self.coverage[t, r] = 1.0


def train_model(model, train_triples, device, epochs=30, batch_size=1024, lr=0.001):
    """Train a model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Create dataloader
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

            # Positive scores
            pos_scores = model(batch_h, batch_r, batch_t)

            # Negative sampling
            neg_t = torch.randint(0, model.num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t)

            # Loss
            loss = criterion(pos_scores, torch.ones_like(pos_scores))
            loss += criterion(neg_scores, torch.zeros_like(neg_scores))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def evaluate_ood(model, test_triples, num_entities, device, corruption='random', k=10):
    """Evaluate OOD detection."""
    model.eval()

    # Generate OOD samples
    if corruption == 'random':
        ood_tails = np.random.randint(0, num_entities, len(test_triples))
    elif corruption == 'popularity_matched':
        # Match popularity (frequency) - for now, use random as approximation
        ood_tails = np.random.randint(0, num_entities, len(test_triples))
    elif corruption == 'embedding_similar':
        # Find k-NN in embedding space
        with torch.no_grad():
            if hasattr(model, 'entity_mean'):
                emb = model.entity_mean.cpu().numpy()
            elif hasattr(model, 'entity_emb'):
                emb = model.entity_emb.weight.cpu().numpy()
            else:
                ood_tails = np.random.randint(0, num_entities, len(test_triples))
                emb = None

            if emb is not None:
                ood_tails = []
                for i in range(len(test_triples)):
                    t = test_triples[i, 2]
                    dists = np.linalg.norm(emb - emb[t], axis=1)
                    dists[t] = np.inf
                    nn_idx = np.argsort(dists)[:k]
                    ood_tails.append(nn_idx[np.random.randint(k)])
                ood_tails = np.array(ood_tails)
    elif corruption == 'relation_plausible':
        # Sample from entities seen with this relation
        ood_tails = []
        coverage = model.coverage.cpu().numpy()
        for i in range(len(test_triples)):
            r = test_triples[i, 1]
            valid_entities = np.where(coverage[:, r] > 0)[0]
            if len(valid_entities) > 0:
                ood_tails.append(np.random.choice(valid_entities))
            else:
                ood_tails.append(np.random.randint(0, num_entities))
        ood_tails = np.array(ood_tails)
    elif corruption == 'high_score':
        # Find entities with high model scores
        with torch.no_grad():
            ood_tails = []
            for i in range(min(len(test_triples), 1000)):  # Limit for speed
                h = torch.tensor([test_triples[i, 0]]).to(device)
                r = torch.tensor([test_triples[i, 1]]).to(device)

                # Score all tails
                all_t = torch.arange(num_entities).to(device)
                h_exp = h.expand(num_entities)
                r_exp = r.expand(num_entities)
                scores = model(h_exp, r_exp, all_t)

                # Exclude true tail, get top-k
                t_true = test_triples[i, 2]
                scores[t_true] = float('-inf')
                top_k = torch.topk(scores, k).indices
                ood_tails.append(top_k[np.random.randint(k)].item())

            # Fill rest with random
            ood_tails.extend([np.random.randint(0, num_entities)
                             for _ in range(len(test_triples) - len(ood_tails))])
            ood_tails = np.array(ood_tails)
    else:
        ood_tails = np.random.randint(0, num_entities, len(test_triples))

    # Compute uncertainties
    with torch.no_grad():
        # ID
        h_id = torch.tensor(test_triples[:, 0]).to(device)
        r_id = torch.tensor(test_triples[:, 1]).to(device)
        t_id = torch.tensor(test_triples[:, 2]).to(device)
        id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        # OOD
        t_ood = torch.tensor(ood_tails).to(device)
        ood_unc = model.get_uncertainty(h_id, r_id, t_ood).cpu().numpy()

    # AUROC
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])

    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5

    return auroc


def evaluate_temporal_ood(model, train_triples, test_triples, num_entities, device):
    """Evaluate on temporal-like OOD (frequency-based)."""
    model.eval()

    # Compute entity frequencies
    entity_freq = defaultdict(int)
    for i in range(len(train_triples)):
        entity_freq[train_triples[i, 0]] += 1
        entity_freq[train_triples[i, 2]] += 1

    freq_threshold = np.percentile(list(entity_freq.values()), 25)

    # Categorize test triples
    new_entity_idx = []
    new_pair_idx = []
    coverage = model.coverage.cpu().numpy()

    for i in range(len(test_triples)):
        h, r, t = test_triples[i]
        h_freq = entity_freq.get(h, 0)
        t_freq = entity_freq.get(t, 0)

        if h_freq < freq_threshold or t_freq < freq_threshold:
            new_entity_idx.append(i)
        elif coverage[h, r] == 0 or coverage[t, r] == 0:
            new_pair_idx.append(i)

    results = {}

    # Evaluate new entity OOD
    if len(new_entity_idx) > 100:
        test_subset = test_triples[new_entity_idx[:1000]]
        results['new_entity'] = {
            'n': len(new_entity_idx),
            'auroc': evaluate_ood(model, test_subset, num_entities, device, 'random')
        }

    # Evaluate new pair OOD
    if len(new_pair_idx) > 100:
        test_subset = test_triples[new_pair_idx[:1000]]
        results['new_pair'] = {
            'n': len(new_pair_idx),
            'auroc': evaluate_ood(model, test_subset, num_entities, device, 'random')
        }

    return results


def evaluate_qa_abstention(model, train_triples, test_triples, num_entities, device, coverage_target=0.85):
    """Evaluate QA abstention task."""
    model.eval()

    # Compute uncertainties for all test triples
    with torch.no_grad():
        h = torch.tensor(test_triples[:, 0]).to(device)
        r = torch.tensor(test_triples[:, 1]).to(device)
        t = torch.tensor(test_triples[:, 2]).to(device)

        uncertainties = model.get_uncertainty(h, r, t).cpu().numpy()

        # Get predictions
        correct = []
        for i in range(min(len(test_triples), 2000)):  # Limit for speed
            h_i = torch.tensor([test_triples[i, 0]]).to(device)
            r_i = torch.tensor([test_triples[i, 1]]).to(device)

            # Score all tails
            all_t = torch.arange(num_entities).to(device)
            h_exp = h_i.expand(num_entities)
            r_exp = r_i.expand(num_entities)
            scores = model(h_exp, r_exp, all_t)

            pred = scores.argmax().item()
            correct.append(pred == test_triples[i, 2])

        correct = np.array(correct)
        uncertainties = uncertainties[:len(correct)]

    # Find threshold for target coverage
    sorted_idx = np.argsort(uncertainties)
    n_answer = int(coverage_target * len(correct))
    answer_idx = sorted_idx[:n_answer]

    # Compute selective accuracy
    selective_acc = correct[answer_idx].mean()
    baseline_acc = correct.mean()

    error_reduction = (baseline_acc - selective_acc) / (1 - baseline_acc + 1e-8) if baseline_acc < 1 else 0
    # Actually it should be error reduction, so:
    baseline_error = 1 - baseline_acc
    selective_error = 1 - selective_acc
    error_reduction = (baseline_error - selective_error) / (baseline_error + 1e-8) if baseline_error > 0 else 0

    return {
        'coverage': coverage_target,
        'selective_accuracy': selective_acc,
        'baseline_accuracy': baseline_acc,
        'error_reduction': error_reduction
    }


def run_experiments(dataset_name='fb15k-237', epochs=30, dim=100):
    """Run all experiments on a dataset."""
    print(f"\n{'='*60}")
    print(f"Running experiments on {dataset_name}")
    print(f"{'='*60}\n")

    device = setup_device()

    # Load data
    print("Loading data...")
    if dataset_name == 'fb15k-237':
        train_ds, valid_ds, test_ds = load_fb15k237()
    elif dataset_name == 'wn18rr':
        train_ds, valid_ds, test_ds = load_wn18rr()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_triples = train_ds.triples
    test_triples = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    # Models to evaluate
    models = {
        'CAGP': SimpleCAGP(num_entities, num_relations, dim),
        'AttentionCAGP': AttentionCAGP(num_entities, num_relations, dim),
        'RelCondVar': RelCondVar(num_entities, num_relations, dim),
        'UKGE': ScoreBasedUncertainty(num_entities, num_relations, dim, 'ukge'),
        'Energy': ScoreBasedUncertainty(num_entities, num_relations, dim, 'energy'),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        # Precompute coverage
        model.precompute_coverage(train_triples)

        # Train
        print("  Training...")
        model = train_model(model, train_triples, device, epochs=epochs)

        model_results = {'model': model_name}

        # Random OOD
        print("  Evaluating random OOD...")
        model_results['random_ood'] = evaluate_ood(
            model, test_triples[:2000], num_entities, device, 'random'
        )
        print(f"    AUROC: {model_results['random_ood']:.4f}")

        # Adversarial OOD (only for decomposition-based methods)
        if model_name in ['CAGP', 'AttentionCAGP', 'RelCondVar']:
            print("  Evaluating adversarial OOD...")
            for corruption in ['embedding_similar', 'relation_plausible', 'high_score']:
                auroc = evaluate_ood(
                    model, test_triples[:1000], num_entities, device, corruption
                )
                model_results[f'{corruption}_ood'] = auroc
                print(f"    {corruption}: {auroc:.4f}")

        # Temporal OOD
        print("  Evaluating temporal OOD...")
        temporal_results = evaluate_temporal_ood(
            model, train_triples, test_triples, num_entities, device
        )
        model_results['temporal'] = temporal_results
        for k, v in temporal_results.items():
            print(f"    {k}: AUROC={v['auroc']:.4f} (n={v['n']})")

        # QA Abstention
        print("  Evaluating QA abstention...")
        qa_results = evaluate_qa_abstention(
            model, train_triples, test_triples[:2000], num_entities, device
        )
        model_results['qa_abstention'] = qa_results
        print(f"    Selective Acc: {qa_results['selective_accuracy']:.4f}")
        print(f"    Error Reduction: {qa_results['error_reduction']:.4f}")

        results[model_name] = model_results

    return results


def main():
    """Main entry point."""
    all_results = {}

    # Run on FB15k-237 (main dataset)
    fb_results = run_experiments('fb15k-237', epochs=30, dim=100)
    all_results['fb15k-237'] = fb_results

    # Run on WN18RR (secondary)
    wn_results = run_experiments('wn18rr', epochs=30, dim=100)
    all_results['wn18rr'] = wn_results

    # Save results
    output_path = project_root / 'outputs' / 'full_experiment_results.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Random OOD AUROC")
    print("="*80)
    print(f"{'Model':<20} {'FB15k-237':<15} {'WN18RR':<15}")
    print("-"*50)
    for model_name in ['CAGP', 'AttentionCAGP', 'RelCondVar', 'UKGE', 'Energy']:
        fb = fb_results.get(model_name, {}).get('random_ood', 'N/A')
        wn = wn_results.get(model_name, {}).get('random_ood', 'N/A')
        fb_str = f"{fb:.4f}" if isinstance(fb, float) else fb
        wn_str = f"{wn:.4f}" if isinstance(wn, float) else wn
        print(f"{model_name:<20} {fb_str:<15} {wn_str:<15}")


if __name__ == "__main__":
    main()
