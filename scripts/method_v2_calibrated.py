#!/usr/bin/env python3
"""
Method V2c: Calibrated Coverage-Energy
Learn temperature scaling per coverage level
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
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

def run_calibrated(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Calibrated Coverage-Energy")
    print(f"{'='*60}")
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
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
    
    # Collect calibration data
    val_data = {0: [], 1: [], 2: []}  # By coverage level
    
    with torch.no_grad():
        for h, r, t in train[:10000]:
            h, r, t = int(h), int(r), int(t)
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            
            energy_pos = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            
            # Negative sample
            t_neg = np.random.randint(0, n_ent)
            energy_neg = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t_neg])).item()
            
            val_data[cov].append((energy_pos, 0))  # 0 = correct
            val_data[cov].append((energy_neg, 1))  # 1 = wrong
    
    # Learn temperature per coverage level
    temperatures = {}
    for cov in [0, 1, 2]:
        if len(val_data[cov]) < 100:
            temperatures[cov] = 1.0
            continue
        
        energies = torch.tensor([d[0] for d in val_data[cov]], dtype=torch.float32)
        labels = torch.tensor([d[1] for d in val_data[cov]], dtype=torch.float32)
        
        # Optimize temperature
        temp = nn.Parameter(torch.tensor(1.0))
        opt_temp = torch.optim.LBFGS([temp], lr=0.1, max_iter=50)
        
        def closure():
            opt_temp.zero_grad()
            scaled = energies / temp
            loss = nn.BCEWithLogitsLoss()(scaled, labels)
            loss.backward()
            return loss
        
        opt_temp.step(closure)
        temperatures[cov] = max(0.1, temp.item())  # Prevent negative
    
    print(f"Learned temperatures: {temperatures}")
    
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
                
                # Calibrated energy
                temp = temperatures[cov]
                calibrated = energy / temp
                
                # Combined: coverage + calibrated energy
                combined = (2 - cov) * 0.5 + calibrated * 0.5
                
                scores = model(torch.full((n_ent,), h, dtype=torch.long),
                              torch.full((n_ent,), r, dtype=torch.long),
                              torch.arange(n_ent)).numpy()
                rank = int((scores > scores[t]).sum() + 1)
                
                results.append({
                    'energy': energy, 'cov': cov, 'calibrated': calibrated, 'combined': combined,
                    'is_test': is_test, 'is_wrong': int(rank > 10)
                })
    
    # AUROCs
    energy_unc = [r['energy'] for r in results]
    cov_unc = [2 - r['cov'] for r in results]
    calib_unc = [r['calibrated'] for r in results]
    combined_unc = [r['combined'] for r in results]
    
    labels_temp = [r['is_test'] for r in results]
    labels_err = [r['is_wrong'] for r in results]
    
    print(f"\nTemporal OOD:")
    print(f"  Energy: {roc_auc_score(labels_temp, energy_unc):.4f}")
    print(f"  Coverage: {roc_auc_score(labels_temp, cov_unc):.4f}")
    print(f"  Calibrated: {roc_auc_score(labels_temp, calib_unc):.4f}")
    print(f"  Combined: {roc_auc_score(labels_temp, combined_unc):.4f}")
    
    print(f"\nPrediction Error:")
    print(f"  Energy: {roc_auc_score(labels_err, energy_unc):.4f}")
    print(f"  Coverage: {roc_auc_score(labels_err, cov_unc):.4f}")
    print(f"  Calibrated: {roc_auc_score(labels_err, calib_unc):.4f}")
    print(f"  Combined: {roc_auc_score(labels_err, combined_unc):.4f}")

def main():
    print("="*60)
    print("METHOD V2c: Calibrated Coverage-Energy")
    print("="*60)
    
    ds = load_fb15k237()
    run_calibrated('FB15k-237', ds[0].triples, ds[2].triples,
                   ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    run_calibrated('WN18RR', ds[0].triples, ds[2].triples,
                   ds[0].num_entities, ds[0].num_relations)

if __name__ == "__main__":
    main()
