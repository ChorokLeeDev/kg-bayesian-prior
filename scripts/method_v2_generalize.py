#!/usr/bin/env python3
"""
Generalizability test: Neural Ensemble on ICEWS14 + multiple seeds
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score

def load_icews14():
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/ICEWS14")
    entity2id, relation2id = {}, {}
    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entity2id: entity2id[h] = len(entity2id)
                    if r not in relation2id: relation2id[r] = len(relation2id)
                    if t not in entity2id: entity2id[t] = len(entity2id)
                    triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)
    train = load_triples("train.txt")
    test = load_triples("test.txt")
    return train, test, len(entity2id), len(relation2id)

class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

class NeuralEnsemble(nn.Module):
    def __init__(self, input_dim=8):
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

def run_experiment(name, train, test, n_ent, n_rel, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Coverage
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
        h_cnt = coverage_count.get((h, r), 0)
        t_cnt = coverage_count.get((t, r), 0)
        return [energy, cov, np.log1p(h_cnt), np.log1p(t_cnt),
                ent_degree[h], ent_degree[t], rel_freq[r], ent_degree[h]-ent_degree[t]]
    
    # Train base
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
    
    # AUROCs
    energy_unc = np.array([r['energy'] for r in results])
    cov_unc = np.array([2 - r['cov'] for r in results])
    neural_unc = [r['neural'] for r in results]
    
    energy_norm = (energy_unc - energy_unc.mean()) / (energy_unc.std() + 1e-8)
    cov_norm = (cov_unc - cov_unc.mean()) / (cov_unc.std() + 1e-8)
    
    labels_temp = [r['is_test'] for r in results]
    labels_err = [r['is_wrong'] for r in results]
    
    auroc_energy_temp = roc_auc_score(labels_temp, energy_unc)
    auroc_cov_temp = roc_auc_score(labels_temp, cov_unc)
    auroc_neural_temp = roc_auc_score(labels_temp, neural_unc)
    best_linear_temp = max(roc_auc_score(labels_temp, a*cov_norm + (1-a)*energy_norm) for a in np.arange(0,1.1,0.1))
    
    auroc_energy_err = roc_auc_score(labels_err, energy_unc)
    auroc_cov_err = roc_auc_score(labels_err, cov_unc)
    auroc_neural_err = roc_auc_score(labels_err, neural_unc)
    best_linear_err = max(roc_auc_score(labels_err, a*cov_norm + (1-a)*energy_norm) for a in np.arange(0,1.1,0.1))
    
    return {
        'temporal': {'energy': auroc_energy_temp, 'cov': auroc_cov_temp, 'linear': best_linear_temp, 'neural': auroc_neural_temp},
        'error': {'energy': auroc_energy_err, 'cov': auroc_cov_err, 'linear': best_linear_err, 'neural': auroc_neural_err}
    }

def main():
    print("="*60)
    print("GENERALIZABILITY TEST: Multiple seeds + ICEWS14")
    print("="*60)
    
    datasets = {}
    
    # Load datasets
    ds = load_fb15k237()
    datasets['FB15k-237'] = (ds[0].triples, ds[2].triples, ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    datasets['WN18RR'] = (ds[0].triples, ds[2].triples, ds[0].num_entities, ds[0].num_relations)
    
    train, test, n_ent, n_rel = load_icews14()
    datasets['ICEWS14'] = (train, test, n_ent, n_rel)
    
    seeds = [42, 123, 456]
    
    all_results = {}
    for name, (train, test, n_ent, n_rel) in datasets.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        
        seed_results = []
        for seed in seeds:
            print(f"  Seed {seed}...")
            r = run_experiment(name, train, test, n_ent, n_rel, seed)
            seed_results.append(r)
            print(f"    Temporal: E={r['temporal']['energy']:.3f}, L={r['temporal']['linear']:.3f}, N={r['temporal']['neural']:.3f}")
            print(f"    Error: E={r['error']['energy']:.3f}, L={r['error']['linear']:.3f}, N={r['error']['neural']:.3f}")
        
        # Aggregate
        all_results[name] = {
            'temporal': {
                'energy': np.mean([r['temporal']['energy'] for r in seed_results]),
                'linear': np.mean([r['temporal']['linear'] for r in seed_results]),
                'neural': np.mean([r['temporal']['neural'] for r in seed_results]),
                'neural_std': np.std([r['temporal']['neural'] for r in seed_results]),
            },
            'error': {
                'energy': np.mean([r['error']['energy'] for r in seed_results]),
                'linear': np.mean([r['error']['linear'] for r in seed_results]),
                'neural': np.mean([r['error']['neural'] for r in seed_results]),
                'neural_std': np.std([r['error']['neural'] for r in seed_results]),
            }
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (mean over 3 seeds)")
    print("="*60)
    
    print("\nTemporal OOD:")
    print(f"{'Dataset':<12} {'Energy':<10} {'Linear':<10} {'Neural':<15} {'Δ vs best':<10}")
    for name, r in all_results.items():
        t = r['temporal']
        best = max(t['energy'], t['linear'])
        delta = t['neural'] - best
        print(f"{name:<12} {t['energy']:<10.3f} {t['linear']:<10.3f} {t['neural']:.3f}±{t['neural_std']:.3f}  {delta:+.3f}")
    
    print("\nPrediction Error OOD:")
    print(f"{'Dataset':<12} {'Energy':<10} {'Linear':<10} {'Neural':<15} {'Δ vs best':<10}")
    for name, r in all_results.items():
        e = r['error']
        best = max(e['energy'], e['linear'])
        delta = e['neural'] - best
        print(f"{name:<12} {e['energy']:<10.3f} {e['linear']:<10.3f} {e['neural']:.3f}±{e['neural_std']:.3f}  {delta:+.3f}")

if __name__ == "__main__":
    main()
