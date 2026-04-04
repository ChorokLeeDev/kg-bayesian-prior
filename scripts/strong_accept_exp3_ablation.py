#!/usr/bin/env python3
"""
Strong Accept Exp 3: Ablation study for Neural Ensemble
Which features matter most?
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score

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
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_ablation(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Feature Ablation")
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
    
    # Train base model
    torch.manual_seed(42)
    base_model = DistMult(n_ent, n_rel)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h, r, t = torch.tensor(batch[:,0]), torch.tensor(batch[:,1]), torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            opt.zero_grad()
            loss = torch.clamp(1.0 - base_model(h,r,t) + base_model(h,r,t_neg), min=0).mean()
            loss.backward()
            opt.step()
    base_model.eval()
    
    # Feature sets to test
    feature_sets = {
        'All (8 features)': ['energy', 'cov', 'h_cnt', 't_cnt', 'h_deg', 't_deg', 'rel_freq', 'deg_diff'],
        'No energy': ['cov', 'h_cnt', 't_cnt', 'h_deg', 't_deg', 'rel_freq', 'deg_diff'],
        'No coverage': ['energy', 'h_deg', 't_deg', 'rel_freq', 'deg_diff'],
        'Energy + Coverage only': ['energy', 'cov'],
        'Energy only': ['energy'],
        'Coverage only': ['cov'],
    }
    
    def get_feature_vector(h, r, t, energy, feature_names):
        h, r, t = int(h), int(r), int(t)
        h_cov = (h, r) in coverage_set
        t_cov = (t, r) in coverage_set
        cov = int(h_cov) + int(t_cov)
        h_cnt = coverage_count.get((h, r), 0)
        t_cnt = coverage_count.get((t, r), 0)
        
        feature_map = {
            'energy': energy,
            'cov': cov,
            'h_cnt': np.log1p(h_cnt),
            't_cnt': np.log1p(t_cnt),
            'h_deg': ent_degree[h],
            't_deg': ent_degree[t],
            'rel_freq': rel_freq[r],
            'deg_diff': ent_degree[h] - ent_degree[t],
        }
        return [feature_map[f] for f in feature_names]
    
    results = {}
    
    for set_name, features in feature_sets.items():
        print(f"\n  Testing: {set_name}")
        
        # Collect training data
        unc_data = []
        with torch.no_grad():
            for h, r, t in train[:5000]:
                energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
                unc_data.append((get_feature_vector(h, r, t, energy, features), 0))
                t_neg = np.random.randint(0, n_ent)
                energy_neg = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t_neg])).item()
                unc_data.append((get_feature_vector(h, r, t_neg, energy_neg, features), 1))
        
        np.random.shuffle(unc_data)
        X = torch.tensor([d[0] for d in unc_data], dtype=torch.float32)
        y = torch.tensor([d[1] for d in unc_data], dtype=torch.float32)
        
        # Train neural ensemble
        torch.manual_seed(42)
        ensemble = NeuralEnsemble(len(features))
        opt_e = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
        for epoch in range(20):
            for i in range(0, len(X), 256):
                opt_e.zero_grad()
                loss = nn.BCEWithLogitsLoss()(ensemble(X[i:i+256]), y[i:i+256])
                loss.backward()
                opt_e.step()
        ensemble.eval()
        
        # Evaluate
        test_sub = test[:1500] if len(test) > 1500 else test
        train_sample = train[np.random.choice(len(train), min(750, len(train)), replace=False)]
        
        eval_results = []
        with torch.no_grad():
            for is_test, data in [(0, train_sample), (1, test_sub)]:
                for h, r, t in data:
                    energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
                    feats = get_feature_vector(h, r, t, energy, features)
                    neural = ensemble(torch.tensor([feats], dtype=torch.float32)).item()
                    
                    scores = base_model(torch.full((n_ent,), int(h), dtype=torch.long),
                                       torch.full((n_ent,), int(r), dtype=torch.long),
                                       torch.arange(n_ent)).numpy()
                    rank = int((scores > scores[int(t)]).sum() + 1)
                    
                    eval_results.append({'neural': neural, 'is_test': is_test, 'is_wrong': int(rank > 10)})
        
        labels_err = [r['is_wrong'] for r in eval_results]
        neural_unc = [r['neural'] for r in eval_results]
        auroc = roc_auc_score(labels_err, neural_unc)
        
        results[set_name] = auroc
        print(f"    Error Prediction AUROC: {auroc:.4f}")
    
    return results

def main():
    print("="*60)
    print("STRONG ACCEPT: Feature Ablation Study")
    print("="*60)
    
    all_results = {}
    
    ds = load_fb15k237()
    all_results['FB15k-237'] = run_ablation('FB15k-237', ds[0].triples, ds[2].triples,
                                            ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    all_results['WN18RR'] = run_ablation('WN18RR', ds[0].triples, ds[2].triples,
                                         ds[0].num_entities, ds[0].num_relations)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Error Prediction AUROC by Feature Set")
    print("="*60)
    
    feature_sets = ['All (8 features)', 'No energy', 'No coverage', 'Energy + Coverage only', 'Energy only', 'Coverage only']
    print(f"{'Feature Set':<25} {'FB15k-237':<12} {'WN18RR':<12}")
    print("-"*50)
    for fs in feature_sets:
        fb = all_results['FB15k-237'].get(fs, 0)
        wn = all_results['WN18RR'].get(fs, 0)
        print(f"{fs:<25} {fb:<12.4f} {wn:<12.4f}")

if __name__ == "__main__":
    main()
