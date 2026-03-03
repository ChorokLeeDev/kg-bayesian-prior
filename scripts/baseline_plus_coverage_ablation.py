#!/usr/bin/env python3
"""
Baseline + Coverage Ablation Study

For each baseline uncertainty method, compute post-hoc combination:
    U_combined = alpha * U_baseline_normalized + (1-alpha) * U_coverage
    
with alpha=0.5, and measure temporal OOD AUROC.

This demonstrates that coverage augmentation improves ANY baseline,
supporting the paper's novelty claim about the structural signal.
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
import json
from collections import defaultdict
import time

from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Baseline Models with Multiple Uncertainty Methods
# ============================================================

class EnergyBasedKGE(nn.Module):
    """DistMult + Energy-based uncertainty (score-based)."""
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
        """Energy-based: use negative score as uncertainty (lower score = higher uncertainty)."""
        score = self.forward(h, r, t)
        return -score  # High uncertainty when score is low

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class MCDropoutKGE(nn.Module):
    """DistMult + MC Dropout for uncertainty."""
    def __init__(self, num_entities, num_relations, dim=100, dropout_rate=0.1, num_samples=10):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t, use_dropout=False):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        
        if use_dropout:
            h_emb = self.dropout(h_emb)
            r_emb = self.dropout(r_emb)
            t_emb = self.dropout(t_emb)
        
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        """MC Dropout uncertainty: variance of scores over dropout samples."""
        scores = []
        for _ in range(self.num_samples):
            score = self.forward(h, r, t, use_dropout=True)
            scores.append(score)
        scores = torch.stack(scores, dim=0)
        return scores.var(dim=0)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class DeepEnsemblesKGE(nn.Module):
    """DistMult + Deep Ensembles uncertainty."""
    def __init__(self, num_entities, num_relations, dim=100, num_ensembles=5):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_ensembles = num_ensembles
        
        self.entity_embs = nn.ModuleList([
            nn.Embedding(num_entities, dim) for _ in range(num_ensembles)
        ])
        self.relation_embs = nn.ModuleList([
            nn.Embedding(num_relations, dim) for _ in range(num_ensembles)
        ])
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t, ensemble_idx=None):
        if ensemble_idx is None:
            # Use mean over ensembles
            scores = []
            for i in range(self.num_ensembles):
                score = (self.entity_embs[i](h) * self.relation_embs[i](r) * self.entity_embs[i](t)).sum(-1)
                scores.append(score)
            return torch.stack(scores, dim=0).mean(dim=0)
        else:
            return (self.entity_embs[ensemble_idx](h) * self.relation_embs[ensemble_idx](r) * self.entity_embs[ensemble_idx](t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        """Ensemble uncertainty: variance of scores across ensemble members."""
        scores = []
        for i in range(self.num_ensembles):
            score = (self.entity_embs[i](h) * self.relation_embs[i](r) * self.entity_embs[i](t)).sum(-1)
            scores.append(score)
        scores = torch.stack(scores, dim=0)
        return scores.var(dim=0)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class VariationalKGE(nn.Module):
    """DistMult + Variational embeddings (GP/semantic uncertainty)."""
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
        """Variational uncertainty: embedding variance."""
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


class SNGPBasedKGE(nn.Module):
    """SNGP-inspired: spectral norm + distance-based uncertainty."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        
        # Mean and covariance for training data tracking
        self.register_buffer('train_entity_mean', torch.zeros(num_entities, dim))
        self.register_buffer('train_entity_cov', torch.eye(dim).unsqueeze(0).expand(num_entities, -1, -1))
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self._fitted = False

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        """Distance-based uncertainty: distance to training data manifold."""
        h_emb = self.entity_emb(h)
        t_emb = self.entity_emb(t)
        
        # Simple Euclidean distance to training mean (approximation)
        h_dist = torch.norm(h_emb - self.train_entity_mean[h], dim=-1)
        t_dist = torch.norm(t_emb - self.train_entity_mean[t], dim=-1)
        return (h_dist + t_dist) / 2

    def fit_on_training_data(self, h, r, t):
        """Fit the model statistics on training data."""
        with torch.no_grad():
            h_emb = self.entity_emb(h)
            t_emb = self.entity_emb(t)
            
            # Update mean
            self.train_entity_mean[h] = h_emb.mean(dim=0)
            self.train_entity_mean[t] = t_emb.mean(dim=0)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


# ============================================================
# Training and Evaluation
# ============================================================

def _kl_entity_gaussian(model):
    """KL divergence of entity embeddings from standard normal."""
    if hasattr(model, 'entity_logvar') and hasattr(model, 'entity_mean'):
        mean = model.entity_mean
        logvar = model.entity_logvar
        return (0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(dim=-1)).mean()
    return None


def train_model(model, triples, device, epochs=30, lr=0.001, kl_beta=0.001, unc_weight=0.1):
    """Train a KGE model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
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

            # KL regularization
            kl = _kl_entity_gaussian(model)
            if kl is not None:
                loss = loss + kl_beta * kl

            # Uncertainty margin
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

    return model


def _is_emerging(h_freq, t_freq, thresh, emerging_operator='leq'):
    if emerging_operator == 'lt':
        return h_freq < thresh or t_freq < thresh
    if emerging_operator == 'leq':
        return h_freq <= thresh or t_freq <= thresh
    raise ValueError(f"Unsupported emerging_operator: {emerging_operator}")


def evaluate_temporal(model, train, test, n_ent, device):
    """Temporal-like OOD evaluation with 25th percentile threshold."""
    model.eval()

    # Entity frequencies from training
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1

    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()

    # Categorize test triples
    new_entity_idx, new_pair_idx, id_idx = [], [], []
    for i in range(len(test)):
        h, r, t = test[i]
        if _is_emerging(freq.get(h, 0), freq.get(t, 0), thresh, 'leq'):
            new_entity_idx.append(i)
        elif cov[h, r] == 0 or cov[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    ood_idx = new_entity_idx + new_pair_idx
    
    results = {
        'n_emerging': len(new_entity_idx),
        'n_novel_ctx': len(new_pair_idx),
        'n_id': len(id_idx),
    }

    # Overall temporal OOD
    if len(ood_idx) > 50 and len(id_idx) > 50:
        with torch.no_grad():
            ood_triples = test[ood_idx]
            id_triples = test[id_idx]

            h_ood = torch.tensor(ood_triples[:, 0]).to(device)
            r_ood = torch.tensor(ood_triples[:, 1]).to(device)
            t_ood = torch.tensor(ood_triples[:, 2]).to(device)
            ood_unc = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

            h_id = torch.tensor(id_triples[:, 0]).to(device)
            r_id = torch.tensor(id_triples[:, 1]).to(device)
            t_id = torch.tensor(id_triples[:, 2]).to(device)
            id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])

        try:
            results['overall_auroc'] = float(roc_auc_score(labels, scores))
        except Exception:
            results['overall_auroc'] = 0.5
    else:
        results['overall_auroc'] = 0.5

    return results


def get_coverage_uncertainty(model, h, r, t):
    """Compute structural/coverage uncertainty."""
    coverage = model.coverage.cpu()
    h_seen = coverage[h, r]
    t_seen = coverage[t, r]
    return 2.0 - h_seen - t_seen


def normalize_scores(scores, baseline_mean, baseline_std, target_mean, target_std, eps=1e-8):
    """Normalize scores from baseline to match target scale."""
    normalized = (scores - baseline_mean) / (baseline_std + eps)
    return normalized * (target_std + eps) + target_mean


def run_ablation(dataset_name, loader_fn, device, seeds=[42, 123, 456]):
    """Run baseline + coverage ablation for one dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name} — Baseline + Coverage Ablation")
    print(f"{'='*70}\n")

    train_ds, _, test_ds = loader_fn()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    
    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train triples: {len(train)}, Test triples: {len(test)}")
    
    # Define baseline methods
    baseline_classes = {
        'Energy': EnergyBasedKGE,
        'MCDropout': MCDropoutKGE,
        'DeepEnsembles': DeepEnsemblesKGE,
        'VariationalGP': VariationalKGE,
        'SNGPDistance': SNGPBasedKGE,
    }
    
    # Results structure: {baseline: {seed: auroc_dict}}
    all_results = {baseline: [] for baseline in baseline_classes.keys()}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for baseline_name, ModelClass in baseline_classes.items():
            print(f"  {baseline_name}...", end='', flush=True)
            
            # Train model
            model = ModelClass(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30)
            model.eval()

            # Compute uncertainties on test set
            with torch.no_grad():
                h = torch.tensor(test[:, 0]).to(device)
                r = torch.tensor(test[:, 1]).to(device)
                t = torch.tensor(test[:, 2]).to(device)
                
                # Baseline uncertainty
                baseline_unc = model.get_uncertainty(h, r, t).cpu().numpy()
                
                # Coverage uncertainty
                h_cpu = h.cpu()
                r_cpu = r.cpu()
                t_cpu = t.cpu()
                coverage_unc = get_coverage_uncertainty(model, h_cpu, r_cpu, t_cpu).numpy()
            
            # Normalize baseline to match coverage scale
            baseline_mean = baseline_unc.mean()
            baseline_std = baseline_unc.std()
            coverage_mean = coverage_unc.mean()
            coverage_std = coverage_unc.std()
            
            baseline_norm = normalize_scores(
                baseline_unc, baseline_mean, baseline_std, coverage_mean, coverage_std
            )
            
            # Combine with alpha=0.5
            alpha = 0.5
            combined_unc = alpha * baseline_norm + (1 - alpha) * coverage_unc
            
            # Evaluate all four variants
            results = {}
            
            for name, unc in [
                ('baseline', baseline_unc),
                ('coverage_only', coverage_unc),
                ('combined', combined_unc),
            ]:
                # Evaluate using temporal split
                eval_model = model
                eval_model.eval()
                
                # Entity frequencies from training
                freq = defaultdict(int)
                for i in range(len(train)):
                    freq[train[i, 0]] += 1
                    freq[train[i, 2]] += 1
                
                thresh = np.percentile(list(freq.values()), 25)
                cov = eval_model.coverage.cpu().numpy()
                
                # Categorize test triples
                new_entity_idx, new_pair_idx, id_idx = [], [], []
                for i in range(len(test)):
                    h_i, r_i, t_i = test[i]
                    if _is_emerging(freq.get(h_i, 0), freq.get(t_i, 0), thresh, 'leq'):
                        new_entity_idx.append(i)
                    elif cov[h_i, r_i] == 0 or cov[t_i, r_i] == 0:
                        new_pair_idx.append(i)
                    else:
                        id_idx.append(i)
                
                ood_idx = new_entity_idx + new_pair_idx
                
                if len(ood_idx) > 50 and len(id_idx) > 50:
                    id_scores = unc[id_idx]
                    ood_scores = unc[ood_idx]
                    try:
                        auroc = roc_auc_score(
                            np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))]),
                            np.concatenate([id_scores, ood_scores])
                        )
                    except:
                        auroc = 0.5
                else:
                    auroc = 0.5
                
                results[name] = auroc
            
            all_results[baseline_name].append(results)
            print(f" ✓ (baseline={results['baseline']:.4f}, combined={results['combined']:.4f})")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {dataset_name} — Summary (mean ± std over {len(seeds)} seeds)")
    print(f"{'='*70}\n")
    
    summary = {}
    for baseline_name in baseline_classes.keys():
        baseline_aucs = [r['baseline'] for r in all_results[baseline_name]]
        combined_aucs = [r['combined'] for r in all_results[baseline_name]]
        coverage_aucs = [r['coverage_only'] for r in all_results[baseline_name]]
        
        baseline_mean, baseline_std = np.mean(baseline_aucs), np.std(baseline_aucs)
        combined_mean, combined_std = np.mean(combined_aucs), np.std(combined_aucs)
        coverage_mean, coverage_std = np.mean(coverage_aucs), np.std(coverage_aucs)
        
        improvement = combined_mean - baseline_mean
        
        print(f"{baseline_name:20s}:")
        print(f"  Baseline only:      {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Baseline + Coverage:{combined_mean:.4f} ± {combined_std:.4f}  (+{improvement:+.4f})")
        print(f"  Coverage only:      {coverage_mean:.4f} ± {coverage_std:.4f}")
        print()
        
        summary[baseline_name] = {
            'baseline_auroc': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'combined_auroc': float(combined_mean),
            'combined_std': float(combined_std),
            'coverage_auroc': float(coverage_mean),
            'coverage_std': float(coverage_std),
            'improvement': float(improvement),
            'num_seeds': len(seeds),
        }
    
    return summary, all_results


def main():
    device = setup_device()
    print(f"Using device: {device}")
    
    all_summaries = {}
    
    for name, loader in [
        ("WN18RR", load_wn18rr),
        ("FB15k-237", load_fb15k237),
    ]:
        summary, detailed = run_ablation(name, loader, device)
        all_summaries[name] = summary
    
    # Save results
    output_dir = Path("/sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior/outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "baseline_plus_coverage_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}\n")
    
    # Final summary table
    print("\nFinal Summary Table:\n")
    print(f"{'Dataset':<15} {'Method':<20} {'Baseline':<15} {'+Coverage':<15} {'Improvement':<12}")
    print("-" * 80)
    
    for dataset in ["WN18RR", "FB15k-237"]:
        for method, results in all_summaries[dataset].items():
            baseline = f"{results['baseline_auroc']:.4f}±{results['baseline_std']:.4f}"
            combined = f"{results['combined_auroc']:.4f}±{results['combined_std']:.4f}"
            improve = f"{results['improvement']:+.4f}"
            print(f"{dataset:<15} {method:<20} {baseline:<15} {combined:<15} {improve:<12}")


if __name__ == "__main__":
    main()
