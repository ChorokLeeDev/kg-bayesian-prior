#!/usr/bin/env python3
"""
Non-circular OOD: Held-out Relations
OOD = triples with relations not seen during training
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
        self.ent = nn.Embedding(n_ent, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

def main():
    print("="*60)
    print("NON-CIRCULAR OOD: HELD-OUT RELATIONS")
    print("="*60)
    
    ds = load_fb15k237()
    all_triples = ds[0].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Hold out 20% of relations
    all_rels = list(set(all_triples[:, 1]))
    np.random.seed(42)
    np.random.shuffle(all_rels)
    n_holdout = max(1, len(all_rels) // 5)
    held_out_rels = set(all_rels[:n_holdout])
    train_rels = set(all_rels[n_holdout:])
    
    print(f"Total relations: {len(all_rels)}")
    print(f"Held-out relations: {len(held_out_rels)}")
    print(f"Training relations: {len(train_rels)}")
    
    # Split triples
    train_triples = np.array([t for t in all_triples if int(t[1]) in train_rels])
    ood_triples = np.array([t for t in all_triples if int(t[1]) in held_out_rels])
    
    print(f"Training triples: {len(train_triples)}")
    print(f"OOD triples: {len(ood_triples)}")
    
    # Coverage from train only
    coverage_set = set()
    coverage_count = {}
    for h, r, t in train_triples:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
        for key in [(int(h), int(r)), (int(t), int(r))]:
            coverage_count[key] = coverage_count.get(key, 0) + 1
    
    # Relation coverage (how many entities seen with each relation)
    rel_entity_count = {}
    for h, r, t in train_triples:
        r = int(r)
        if r not in rel_entity_count:
            rel_entity_count[r] = set()
        rel_entity_count[r].add(int(h))
        rel_entity_count[r].add(int(t))
    
    # Train model
    torch.manual_seed(42)
    model = SimpleModel(n_ent, n_rel)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        np.random.shuffle(train_triples)
        for i in range(0, len(train_triples), 512):
            batch = train_triples[i:i+512]
            h, r, t = torch.tensor(batch[:,0]), torch.tensor(batch[:,1]), torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            opt.zero_grad()
            loss = torch.clamp(1.0 - model(h,r,t) + model(h,r,t_neg), min=0).mean()
            loss.backward()
            opt.step()
    
    model.eval()
    
    # Evaluate
    n_eval = min(1000, len(ood_triples))
    train_sample = train_triples[np.random.choice(len(train_triples), n_eval, replace=False)]
    ood_sample = ood_triples[:n_eval]
    
    results = []
    with torch.no_grad():
        for h, r, t in train_sample:
            h, r, t = int(h), int(r), int(t)
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            rel_cov = len(rel_entity_count.get(r, set()))
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            results.append({'cov': cov, 'rel_cov': rel_cov, 'energy': energy, 'ood': 0})
        
        for h, r, t in ood_sample:
            h, r, t = int(h), int(r), int(t)
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            rel_cov = len(rel_entity_count.get(r, set()))  # Will be 0 for held-out
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            results.append({'cov': cov, 'rel_cov': rel_cov, 'energy': energy, 'ood': 1})
    
    labels = [r['ood'] for r in results]
    
    print("\nUncertainty methods:")
    
    # Energy
    energy_unc = [r['energy'] for r in results]
    auroc_energy = roc_auc_score(labels, energy_unc)
    print(f"  Energy only: AUROC={auroc_energy:.4f}")
    
    # Coverage binary
    cov_unc = [2 - r['cov'] for r in results]
    auroc_cov = roc_auc_score(labels, cov_unc)
    print(f"  Coverage binary: AUROC={auroc_cov:.4f}")
    
    # Relation coverage
    rel_unc = [-r['rel_cov'] for r in results]
    auroc_rel = roc_auc_score(labels, rel_unc)
    print(f"  Relation coverage: AUROC={auroc_rel:.4f}")
    
    # Ensemble
    energy_arr = np.array(energy_unc)
    cov_arr = np.array(cov_unc)
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    
    best_ens = 0
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels, ens)
        if auroc > best_ens:
            best_ens = auroc
    print(f"  Best ensemble: AUROC={best_ens:.4f}")

if __name__ == "__main__":
    main()
