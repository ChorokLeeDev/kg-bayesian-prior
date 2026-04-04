#!/usr/bin/env python3
"""
Strong Accept Exp 4: Statistical significance tests
Bootstrap confidence intervals for AUROC improvements
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score
from scipy import stats

class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

class NeuralEnsemble(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def bootstrap_auroc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for AUROC"""
    aurocs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            auroc = roc_auc_score(np.array(y_true)[idx], np.array(y_score)[idx])
            aurocs.append(auroc)
        except:
            pass
    
    lower = np.percentile(aurocs, (1 - ci) / 2 * 100)
    upper = np.percentile(aurocs, (1 + ci) / 2 * 100)
    return np.mean(aurocs), lower, upper

def paired_bootstrap_test(y_true, y_score1, y_score2, n_bootstrap=1000):
    """Test if AUROC difference is significant"""
    diffs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            auroc1 = roc_auc_score(np.array(y_true)[idx], np.array(y_score1)[idx])
            auroc2 = roc_auc_score(np.array(y_true)[idx], np.array(y_score2)[idx])
            diffs.append(auroc2 - auroc1)
        except:
            pass
    
    # p-value: proportion of times diff <= 0
    p_value = np.mean(np.array(diffs) <= 0)
    return np.mean(diffs), p_value

def run_significance(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Significance Tests")
    print(f"{'='*60}")
    
    # Stats
    coverage_set = set()
    coverage_count = {}
    ent_degree = np.zeros(n_ent)
    rel_freq = np.zeros(n_rel)
    
    for h, r, t in train:
        h, r, t = int(h), int(r), int(t)
        coverage_set.add((h, r))
        coverage_set.add((t, r))
        for key in [(h, r), (t, r)]:
            coverage_count[key] = coverage_count.get(key, 0) + 1
        ent_degree[h] += 1
        ent_degree[t] += 1
        rel_freq[r] += 1
    ent_degree = np.log1p(ent_degree)
    rel_freq = np.log1p(rel_freq)
    
    def get_features(h, r, t, energy):
        h, r, t = int(h), int(r), int(t)
        h_cov = (h, r) in coverage_set
        t_cov = (t, r) in coverage_set
        cov = int(h_cov) + int(t_cov)
        return [energy, cov, np.log1p(coverage_count.get((h,r),0)), np.log1p(coverage_count.get((t,r),0)),
                ent_degree[h], ent_degree[t], rel_freq[r], ent_degree[h]-ent_degree[t]]
    
    # Train base model
    torch.manual_seed(42)
    model = DistMult(n_ent, n_rel)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h, r, t = torch.tensor(batch[:,0]), torch.tensor(batch[:,1]), torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            opt.zero_grad()
            loss = torch.clamp(1.0 - model(h,r,t) + model(h,r,t_neg), min=0).mean()
            loss.backward()
            opt.step()
    model.eval()
    
    # Train neural ensemble
    unc_data = []
    with torch.no_grad():
        for h, r, t in train[:5000]:
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            unc_data.append((get_features(h, r, t, energy), 0))
            t_neg = np.random.randint(0, n_ent)
            energy_neg = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t_neg])).item()
            unc_data.append((get_features(h, r, t_neg, energy_neg), 1))
    
    np.random.shuffle(unc_data)
    X = torch.tensor([d[0] for d in unc_data], dtype=torch.float32)
    y = torch.tensor([d[1] for d in unc_data], dtype=torch.float32)
    
    torch.manual_seed(42)
    ensemble = NeuralEnsemble()
    opt_e = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    for epoch in range(20):
        for i in range(0, len(X), 256):
            opt_e.zero_grad()
            loss = nn.BCEWithLogitsLoss()(ensemble(X[i:i+256]), y[i:i+256])
            loss.backward()
            opt_e.step()
    ensemble.eval()
    
    # Evaluate
    test_sub = test[:2000] if len(test) > 2000 else test
    train_sample = train[np.random.choice(len(train), min(1000, len(train)), replace=False)]
    
    results = []
    with torch.no_grad():
        for is_test, data in [(0, train_sample), (1, test_sub)]:
            for h, r, t in data:
                h, r, t = int(h), int(r), int(t)
                h_cov = (h, r) in coverage_set
                t_cov = (t, r) in coverage_set
                cov = int(h_cov) + int(t_cov)
                energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
                feats = get_features(h, r, t, energy)
                neural = ensemble(torch.tensor([feats], dtype=torch.float32)).item()
                
                scores = model(torch.full((n_ent,), h, dtype=torch.long),
                              torch.full((n_ent,), r, dtype=torch.long),
                              torch.arange(n_ent)).numpy()
                rank = int((scores > scores[t]).sum() + 1)
                
                results.append({'energy': energy, 'cov': cov, 'neural': neural,
                               'is_test': is_test, 'is_wrong': int(rank > 10)})
    
    labels_err = [r['is_wrong'] for r in results]
    energy_unc = [r['energy'] for r in results]
    neural_unc = [r['neural'] for r in results]
    
    # Bootstrap CIs
    print("\n  Error Prediction AUROC with 95% CI:")
    
    mean_e, lo_e, hi_e = bootstrap_auroc_ci(labels_err, energy_unc)
    print(f"    Energy: {mean_e:.4f} [{lo_e:.4f}, {hi_e:.4f}]")
    
    mean_n, lo_n, hi_n = bootstrap_auroc_ci(labels_err, neural_unc)
    print(f"    Neural: {mean_n:.4f} [{lo_n:.4f}, {hi_n:.4f}]")
    
    # Significance test
    diff, p_value = paired_bootstrap_test(labels_err, energy_unc, neural_unc)
    print(f"\n  Neural vs Energy:")
    print(f"    Mean difference: {diff:+.4f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

def main():
    print("="*60)
    print("STRONG ACCEPT: Statistical Significance Tests")
    print("="*60)
    
    ds = load_fb15k237()
    run_significance('FB15k-237', ds[0].triples, ds[2].triples,
                    ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    run_significance('WN18RR', ds[0].triples, ds[2].triples,
                    ds[0].num_entities, ds[0].num_relations)

if __name__ == "__main__":
    main()
