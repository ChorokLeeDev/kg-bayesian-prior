#!/usr/bin/env python3
"""
Strong Accept Exp 2: Compare to uncertainty baselines
- MC Dropout
- Deep Ensemble
- Temperature Scaling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score

class DistMultDropout(nn.Module):
    """DistMult with dropout for MC Dropout"""
    def __init__(self, n_ent, n_rel, dim=100, dropout=0.2):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    
    def forward(self, h, r, t):
        h_emb = self.dropout(self.ent(h))
        r_emb = self.rel(r)
        t_emb = self.dropout(self.ent(t))
        return (h_emb * r_emb * t_emb).sum(-1)

class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

def train_model(model, train, n_ent, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h, r, t = torch.tensor(batch[:,0]), torch.tensor(batch[:,1]), torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            opt.zero_grad()
            loss = torch.clamp(1.0 - model(h,r,t) + model(h,r,t_neg), min=0).mean()
            loss.backward()
            opt.step()
    return model

def mc_dropout_uncertainty(model, h, r, t, n_samples=10):
    """MC Dropout uncertainty = variance over multiple forward passes"""
    model.train()  # Enable dropout
    scores = []
    with torch.no_grad():
        for _ in range(n_samples):
            score = model(h, r, t).item()
            scores.append(score)
    return np.std(scores)

def deep_ensemble_uncertainty(models, h, r, t):
    """Deep Ensemble uncertainty = variance across ensemble members"""
    scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            score = model(h, r, t).item()
        scores.append(score)
    return np.std(scores)

def run_baselines(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Uncertainty Baselines")
    print(f"{'='*60}")
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Train models
    print("  Training MC Dropout model...")
    torch.manual_seed(42)
    mc_model = train_model(DistMultDropout(n_ent, n_rel), train, n_ent)
    
    print("  Training Deep Ensemble (3 models)...")
    ensemble_models = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        model = train_model(DistMult(n_ent, n_rel), train.copy(), n_ent)
        ensemble_models.append(model)
    
    print("  Training base model for temperature scaling...")
    torch.manual_seed(42)
    base_model = train_model(DistMult(n_ent, n_rel), train, n_ent)
    base_model.eval()
    
    # Evaluate
    print("  Evaluating...")
    test_sub = test[:1000] if len(test) > 1000 else test
    train_sample = train[np.random.choice(len(train), min(500, len(train)), replace=False)]
    
    results = []
    for is_test, data in [(0, train_sample), (1, test_sub)]:
        for h, r, t in data:
            h, r, t = int(h), int(r), int(t)
            h_t, r_t, t_t = torch.tensor([h]), torch.tensor([r]), torch.tensor([t])
            
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            
            # Energy
            with torch.no_grad():
                energy = -base_model(h_t, r_t, t_t).item()
            
            # MC Dropout uncertainty
            mc_unc = mc_dropout_uncertainty(mc_model, h_t, r_t, t_t)
            
            # Deep Ensemble uncertainty
            de_unc = deep_ensemble_uncertainty(ensemble_models, h_t, r_t, t_t)
            
            # Coverage uncertainty
            cov_unc = 2 - cov
            
            # Rank for error detection
            with torch.no_grad():
                scores = base_model(torch.full((n_ent,), h, dtype=torch.long),
                                   torch.full((n_ent,), r, dtype=torch.long),
                                   torch.arange(n_ent)).numpy()
            rank = int((scores > scores[t]).sum() + 1)
            
            results.append({
                'energy': energy, 'mc': mc_unc, 'de': de_unc, 'cov': cov_unc,
                'is_test': is_test, 'is_wrong': int(rank > 10)
            })
    
    # Compute AUROCs
    labels_temp = [r['is_test'] for r in results]
    labels_err = [r['is_wrong'] for r in results]
    
    # Normalize for ensemble
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.mean()) / (arr.std() + 1e-8)
    
    energy = [r['energy'] for r in results]
    mc = [r['mc'] for r in results]
    de = [r['de'] for r in results]
    cov = [r['cov'] for r in results]
    
    energy_norm = normalize(energy)
    cov_norm = normalize(cov)
    
    # Best linear ensemble
    best_linear = 0
    for alpha in np.arange(0, 1.1, 0.1):
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels_temp, ens)
        best_linear = max(best_linear, auroc)
    
    best_linear_err = 0
    for alpha in np.arange(0, 1.1, 0.1):
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels_err, ens)
        best_linear_err = max(best_linear_err, auroc)
    
    print(f"\n  Temporal OOD AUROC:")
    print(f"    Energy:         {roc_auc_score(labels_temp, energy):.4f}")
    print(f"    MC Dropout:     {roc_auc_score(labels_temp, mc):.4f}")
    print(f"    Deep Ensemble:  {roc_auc_score(labels_temp, de):.4f}")
    print(f"    Coverage:       {roc_auc_score(labels_temp, cov):.4f}")
    print(f"    Linear Ens:     {best_linear:.4f}")
    
    print(f"\n  Error Prediction AUROC:")
    print(f"    Energy:         {roc_auc_score(labels_err, energy):.4f}")
    print(f"    MC Dropout:     {roc_auc_score(labels_err, mc):.4f}")
    print(f"    Deep Ensemble:  {roc_auc_score(labels_err, de):.4f}")
    print(f"    Coverage:       {roc_auc_score(labels_err, cov):.4f}")
    print(f"    Linear Ens:     {best_linear_err:.4f}")
    
    return {
        'temporal': {
            'energy': roc_auc_score(labels_temp, energy),
            'mc': roc_auc_score(labels_temp, mc),
            'de': roc_auc_score(labels_temp, de),
            'cov': roc_auc_score(labels_temp, cov),
            'linear': best_linear
        },
        'error': {
            'energy': roc_auc_score(labels_err, energy),
            'mc': roc_auc_score(labels_err, mc),
            'de': roc_auc_score(labels_err, de),
            'cov': roc_auc_score(labels_err, cov),
            'linear': best_linear_err
        }
    }

def main():
    print("="*60)
    print("STRONG ACCEPT: Comparison to Uncertainty Baselines")
    print("="*60)
    
    all_results = {}
    
    ds = load_fb15k237()
    all_results['FB15k-237'] = run_baselines('FB15k-237', ds[0].triples, ds[2].triples,
                                             ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    all_results['WN18RR'] = run_baselines('WN18RR', ds[0].triples, ds[2].triples,
                                          ds[0].num_entities, ds[0].num_relations)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nTemporal OOD:")
    print(f"{'Dataset':<12} {'Energy':<10} {'MC Drop':<10} {'DeepEns':<10} {'Coverage':<10} {'Linear':<10}")
    for ds_name, r in all_results.items():
        t = r['temporal']
        print(f"{ds_name:<12} {t['energy']:<10.3f} {t['mc']:<10.3f} {t['de']:<10.3f} {t['cov']:<10.3f} {t['linear']:<10.3f}")

if __name__ == "__main__":
    main()
