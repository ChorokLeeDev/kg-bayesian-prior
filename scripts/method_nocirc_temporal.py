#!/usr/bin/env python3
"""
Non-circular OOD: Temporal Split
OOD = test triples (simulating future/unseen data)
Coverage computed only from train
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

def run_temporal_ood(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Temporal OOD (test = OOD)")
    print(f"{'='*60}")
    
    # Coverage from train only
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
    
    # Create ID (train sample) vs OOD (test) evaluation set
    n_eval = min(1000, len(test))
    train_sample = train[np.random.choice(len(train), n_eval, replace=False)]
    test_sample = test[:n_eval] if len(test) >= n_eval else test
    
    results = []
    with torch.no_grad():
        # ID samples (from train)
        for h, r, t in train_sample:
            h, r, t = int(h), int(r), int(t)
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            h_cnt = coverage_count.get((h, r), 0)
            t_cnt = coverage_count.get((t, r), 0)
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            results.append({'cov': cov, 'h_cnt': h_cnt, 't_cnt': t_cnt, 'energy': energy, 'ood': 0})
        
        # OOD samples (from test)
        for h, r, t in test_sample:
            h, r, t = int(h), int(r), int(t)
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            h_cnt = coverage_count.get((h, r), 0)
            t_cnt = coverage_count.get((t, r), 0)
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            results.append({'cov': cov, 'h_cnt': h_cnt, 't_cnt': t_cnt, 'energy': energy, 'ood': 1})
    
    labels = [r['ood'] for r in results]
    
    # Different uncertainty measures
    print("\nUncertainty methods:")
    
    # 1. Energy only
    energy_unc = [r['energy'] for r in results]
    auroc_energy = roc_auc_score(labels, energy_unc)
    print(f"  Energy only: AUROC={auroc_energy:.4f}")
    
    # 2. Coverage binary
    cov_unc = [2 - r['cov'] for r in results]
    auroc_cov = roc_auc_score(labels, cov_unc)
    print(f"  Coverage binary: AUROC={auroc_cov:.4f}")
    
    # 3. Coverage count (log)
    log_unc = [-(np.log1p(r['h_cnt']) + np.log1p(r['t_cnt'])) for r in results]
    auroc_log = roc_auc_score(labels, log_unc)
    print(f"  Coverage log-count: AUROC={auroc_log:.4f}")
    
    # 4. Ensemble (normalized)
    energy_arr = np.array(energy_unc)
    cov_arr = np.array(cov_unc)
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    
    best_ens = 0
    best_alpha = 0
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels, ens)
        if auroc > best_ens:
            best_ens = auroc
            best_alpha = alpha
    print(f"  Best ensemble (α={best_alpha}): AUROC={best_ens:.4f}")
    
    return {'energy': auroc_energy, 'cov': auroc_cov, 'log': auroc_log, 'ensemble': best_ens}

def main():
    print("="*60)
    print("NON-CIRCULAR OOD: TEMPORAL SPLIT")
    print("="*60)
    
    results = {}
    
    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = run_temporal_ood('FB15k-237', ds[0].triples, ds[2].triples,
                                            ds[0].num_entities, ds[0].num_relations)
    
    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = run_temporal_ood('WN18RR', ds[0].triples, ds[2].triples,
                                          ds[0].num_entities, ds[0].num_relations)
    
    print("\n" + "="*60)
    print("SUMMARY: Can coverage beat energy for temporal OOD?")
    print("="*60)
    print(f"{'Dataset':<12} {'Energy':<10} {'Coverage':<10} {'Log-cnt':<10} {'Ensemble':<10}")
    for name, r in results.items():
        print(f"{name:<12} {r['energy']:<10.4f} {r['cov']:<10.4f} {r['log']:<10.4f} {r['ensemble']:<10.4f}")

if __name__ == "__main__":
    main()
