#!/usr/bin/env python3
"""
Non-circular OOD: Prediction Error
OOD = model makes wrong prediction (practical uncertainty)
Good uncertainty should correlate with errors
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr
from sklearn.metrics import roc_auc_score

class SimpleModel(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

def run_prediction_ood(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Prediction Error OOD (wrong = OOD)")
    print(f"{'='*60}")
    
    # Coverage from train
    coverage_set = set()
    coverage_count = {}
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
        for key in [(int(h), int(r)), (int(t), int(r))]:
            coverage_count[key] = coverage_count.get(key, 0) + 1
    
    # Train model
    torch.manual_seed(42)
    model = SimpleModel(n_ent, n_rel)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
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
    test_sub = test[:1500] if len(test) > 1500 else test
    
    results = []
    with torch.no_grad():
        for h, r, t in test_sub:
            h, r, t = int(h), int(r), int(t)
            
            # Coverage
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            h_cnt = coverage_count.get((h, r), 0)
            t_cnt = coverage_count.get((t, r), 0)
            
            # Energy
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            
            # Rank (for hits@10)
            h_t = torch.full((n_ent,), h, dtype=torch.long)
            r_t = torch.full((n_ent,), r, dtype=torch.long)
            scores = model(h_t, r_t, torch.arange(n_ent)).numpy()
            rank = int((scores > scores[t]).sum() + 1)
            
            # OOD = wrong prediction (not in top 10)
            is_wrong = rank > 10
            
            results.append({
                'cov': cov, 'h_cnt': h_cnt, 't_cnt': t_cnt,
                'energy': energy, 'rank': rank, 'wrong': int(is_wrong)
            })
    
    labels = [r['wrong'] for r in results]
    wrong_rate = sum(labels) / len(labels)
    print(f"Wrong prediction rate (not hits@10): {wrong_rate:.1%}")
    
    print("\nUncertainty methods (higher = more uncertain = should predict errors):")
    
    # Energy
    energy_unc = [r['energy'] for r in results]
    auroc_energy = roc_auc_score(labels, energy_unc)
    print(f"  Energy only: AUROC={auroc_energy:.4f}")
    
    # Coverage binary
    cov_unc = [2 - r['cov'] for r in results]
    auroc_cov = roc_auc_score(labels, cov_unc)
    print(f"  Coverage binary: AUROC={auroc_cov:.4f}")
    
    # Coverage log-count
    log_unc = [-(np.log1p(r['h_cnt']) + np.log1p(r['t_cnt'])) for r in results]
    auroc_log = roc_auc_score(labels, log_unc)
    print(f"  Coverage log-count: AUROC={auroc_log:.4f}")
    
    # Min count
    min_unc = [-min(r['h_cnt'], r['t_cnt']) for r in results]
    auroc_min = roc_auc_score(labels, min_unc)
    print(f"  Coverage min-count: AUROC={auroc_min:.4f}")
    
    # Ensembles
    energy_arr = np.array(energy_unc)
    cov_arr = np.array(cov_unc)
    log_arr = np.array(log_unc)
    
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    log_norm = (log_arr - log_arr.mean()) / (log_arr.std() + 1e-8)
    
    best_ens = 0
    best_config = ""
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        # Energy + Binary
        ens1 = alpha * cov_norm + (1-alpha) * energy_norm
        auroc1 = roc_auc_score(labels, ens1)
        if auroc1 > best_ens:
            best_ens = auroc1
            best_config = f"α={alpha} (cov+energy)"
        
        # Energy + Log
        ens2 = alpha * log_norm + (1-alpha) * energy_norm
        auroc2 = roc_auc_score(labels, ens2)
        if auroc2 > best_ens:
            best_ens = auroc2
            best_config = f"α={alpha} (log+energy)"
    
    print(f"  Best ensemble ({best_config}): AUROC={best_ens:.4f}")
    
    # Coverage stratified analysis
    print("\nCoverage stratified error rates:")
    for cov_level in [0, 1, 2]:
        subset = [r for r in results if r['cov'] == cov_level]
        if subset:
            err_rate = sum(r['wrong'] for r in subset) / len(subset)
            print(f"  Coverage={cov_level}: {len(subset)} samples, error rate={err_rate:.1%}")
    
    return {
        'energy': auroc_energy, 'cov': auroc_cov, 'log': auroc_log, 
        'min': auroc_min, 'ensemble': best_ens
    }

def main():
    print("="*60)
    print("NON-CIRCULAR OOD: PREDICTION ERROR")
    print("="*60)
    
    results = {}
    
    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = run_prediction_ood('FB15k-237', ds[0].triples, ds[2].triples,
                                               ds[0].num_entities, ds[0].num_relations)
    
    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = run_prediction_ood('WN18RR', ds[0].triples, ds[2].triples,
                                            ds[0].num_entities, ds[0].num_relations)
    
    print("\n" + "="*60)
    print("SUMMARY: Predicting Wrong Predictions")
    print("="*60)
    print(f"{'Dataset':<12} {'Energy':<10} {'Cov-bin':<10} {'Cov-log':<10} {'Cov-min':<10} {'Ensemble':<10}")
    for name, r in results.items():
        print(f"{name:<12} {r['energy']:<10.4f} {r['cov']:<10.4f} {r['log']:<10.4f} {r['min']:<10.4f} {r['ensemble']:<10.4f}")

if __name__ == "__main__":
    main()
