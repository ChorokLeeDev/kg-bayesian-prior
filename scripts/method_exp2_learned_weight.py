#!/usr/bin/env python3
"""
Method 2: Learned per-relation coverage weight
Idea: Some relations benefit more from coverage than others
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237
from sklearn.metrics import roc_auc_score

class RelationWeightedUncertainty(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        # Per-relation coverage importance weight
        self.rel_cov_weight = nn.Parameter(torch.zeros(n_rel))
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
    
    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)
    
    def get_uncertainty(self, h, r, t, coverage):
        """coverage: 0, 1, or 2"""
        energy = -self.forward(h, r, t)
        # Per-relation weight for coverage
        cov_weight = torch.sigmoid(self.rel_cov_weight[r])  # 0-1
        cov_unc = (2 - coverage.float()) * cov_weight
        return energy * 0.1 + cov_unc

def main():
    print("="*60)
    print("METHOD 2: Per-Relation Coverage Weight")
    print("="*60)
    
    ds = load_fb15k237()
    train, test = ds[0].triples, ds[2].triples
    n_ent, n_rel = ds[0].num_entities, ds[0].num_relations
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    def get_coverage(triples):
        covs = []
        for h, r, t in triples:
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            covs.append(int(h_cov) + int(t_cov))
        return torch.tensor(covs)
    
    # Train
    torch.manual_seed(42)
    model = RelationWeightedUncertainty(n_ent, n_rel)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training...")
    for epoch in range(15):
        np.random.shuffle(train)
        for i in range(0, len(train), 512):
            batch = train[i:i+512]
            h, r, t = torch.tensor(batch[:,0]), torch.tensor(batch[:,1]), torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            cov_pos = get_coverage(batch)
            cov_neg = get_coverage(np.column_stack([batch[:,0], batch[:,1], t_neg.numpy()]))
            
            opt.zero_grad()
            # Score loss
            score_loss = torch.clamp(1.0 - model(h,r,t) + model(h,r,t_neg), min=0).mean()
            # Uncertainty loss: neg should have higher uncertainty
            unc_pos = model.get_uncertainty(h, r, t, cov_pos)
            unc_neg = model.get_uncertainty(h, r, t_neg, cov_neg)
            unc_loss = torch.clamp(0.5 - unc_neg + unc_pos, min=0).mean()
            
            loss = score_loss + 0.1 * unc_loss
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
            cov = int(h_cov) + int(t_cov)
            
            unc = model.get_uncertainty(
                torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                torch.tensor([cov])
            ).item()
            
            is_ood = not (h_cov and t_cov)
            results.append({'unc': unc, 'cov': cov, 'ood': is_ood})
    
    labels = [r['ood'] for r in results]
    
    # Baselines
    cov_unc = [2 - r['cov'] for r in results]
    auroc_cov = roc_auc_score(labels, cov_unc)
    
    # Learned
    learned_unc = [r['unc'] for r in results]
    auroc_learned = roc_auc_score(labels, learned_unc)
    
    print(f"\nCoverage-only AUROC: {auroc_cov:.4f}")
    print(f"Learned weight AUROC: {auroc_learned:.4f}")
    print(f"Improvement: {auroc_learned - auroc_cov:+.4f}")
    
    # Show top relations by learned weight
    weights = torch.sigmoid(model.rel_cov_weight).detach().numpy()
    top_rels = np.argsort(weights)[-5:]
    print(f"\nTop 5 relations by coverage importance: {top_rels}")
    print(f"Their weights: {weights[top_rels]}")

if __name__ == "__main__":
    main()
