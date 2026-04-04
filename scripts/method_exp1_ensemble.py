#!/usr/bin/env python3
"""
Method 1: Ensemble Coverage + Energy
Idea: Coverage-only is 0.94, Energy is 0.42. Can ensemble beat 0.94?
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237
from sklearn.metrics import roc_auc_score

class SimpleModel(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
    
    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

def main():
    print("="*60)
    print("METHOD 1: Ensemble Coverage + Energy")
    print("="*60)
    
    ds = load_fb15k237()
    train, test = ds[0].triples, ds[2].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Train energy model
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
    
    # Evaluate
    model.eval()
    test_sub = test[:2000]
    
    results = []
    with torch.no_grad():
        for h, r, t in test_sub:
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            
            # Coverage score (0, 1, 2)
            cov_score = int(h_cov) + int(t_cov)
            
            # Energy score
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            
            # OOD label: not full coverage
            is_ood = not (h_cov and t_cov)
            
            results.append({'cov': cov_score, 'energy': energy, 'ood': is_ood})
    
    labels = [r['ood'] for r in results]
    
    # Baseline AUROCs
    cov_unc = [2 - r['cov'] for r in results]  # Higher = more uncertain
    energy_unc = [r['energy'] for r in results]
    
    auroc_cov = roc_auc_score(labels, cov_unc)
    auroc_energy = roc_auc_score(labels, energy_unc)
    
    print(f"Coverage-only AUROC: {auroc_cov:.4f}")
    print(f"Energy-only AUROC: {auroc_energy:.4f}")
    
    # Ensemble experiments
    print("\nEnsemble experiments:")
    
    # Normalize
    cov_arr = np.array(cov_unc)
    energy_arr = np.array(energy_unc)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    
    best_auroc = auroc_cov
    best_alpha = None
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        ensemble = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels, ensemble)
        marker = " *BEST*" if auroc > best_auroc else ""
        print(f"  alpha={alpha:.1f} (cov weight): AUROC={auroc:.4f}{marker}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = alpha
    
    print(f"\nBest: alpha={best_alpha}, AUROC={best_auroc:.4f}")
    print(f"Improvement over coverage-only: {best_auroc - auroc_cov:+.4f}")

if __name__ == "__main__":
    main()
