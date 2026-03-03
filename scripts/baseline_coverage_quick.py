#!/usr/bin/env python3
"""Quick test: baseline + coverage on one seed."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from collections import defaultdict

from src.data.loaders import load_wn18rr, load_fb15k237


class BaselineKGE(nn.Module):
    """Simple DistMult baseline."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def get_uncertainty(self, h, r, t):
        """Score-based uncertainty."""
        return -self.forward(h, r, t)

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def train_model(model, triples, device, epochs=30):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])
    
    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            
            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)
            
            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    
    return model


def evaluate_temporal(model, train, test, n_ent, device, unc_scores):
    """Temporal OOD evaluation using pre-computed uncertainty scores."""
    model.eval()
    
    # Entity frequencies
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    
    thresh = np.percentile(list(freq.values()), 25)
    cov = model.coverage.cpu().numpy()
    
    # Categorize
    id_idx, ood_idx = [], []
    for i in range(len(test)):
        h, r, t = test[i]
        h_freq = freq.get(h, 0)
        t_freq = freq.get(t, 0)
        is_emerging = h_freq <= thresh or t_freq <= thresh
        has_coverage = cov[h, r] > 0 and cov[t, r] > 0
        
        if is_emerging or not has_coverage:
            ood_idx.append(i)
        else:
            id_idx.append(i)
    
    if len(id_idx) > 50 and len(ood_idx) > 50:
        id_unc = unc_scores[id_idx]
        ood_unc = unc_scores[ood_idx]
        
        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])
        
        try:
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = 0.5
    else:
        auroc = 0.5
    
    return auroc


def main():
    device = torch.device('cpu')
    print(f"Device: {device}\n")
    
    results = {}
    
    for ds_name, loader_fn in [("WN18RR", load_wn18rr), ("FB15k-237", load_fb15k237)]:
        print(f"{'='*60}")
        print(f"{ds_name}")
        print(f"{'='*60}")
        
        train_ds, _, test_ds = loader_fn()
        train = train_ds.triples
        test = test_ds.triples
        n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
        
        print(f"Entities: {n_ent}, Relations: {n_rel}")
        print(f"Train: {len(train)}, Test: {len(test)}\n")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Train baseline
        print("Training baseline...")
        model = BaselineKGE(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=30)
        model.eval()
        
        # Compute uncertainties
        print("Computing uncertainties...")
        with torch.no_grad():
            h = torch.tensor(test[:, 0]).to(device)
            r = torch.tensor(test[:, 1]).to(device)
            t = torch.tensor(test[:, 2]).to(device)
            
            baseline_unc = model.get_uncertainty(h, r, t).cpu().numpy()
            
            # Coverage uncertainty
            cov = model.coverage.cpu()
            coverage_unc = np.zeros(len(test))
            for i in range(len(test)):
                coverage_unc[i] = 2.0 - cov[test[i, 0], test[i, 1]] - cov[test[i, 2], test[i, 1]]
        
        # Normalize and combine
        baseline_norm = (baseline_unc - baseline_unc.mean()) / (baseline_unc.std() + 1e-8)
        baseline_norm = baseline_norm * coverage_unc.std() + coverage_unc.mean()
        
        combined = 0.5 * baseline_norm + 0.5 * coverage_unc
        
        # Evaluate
        print("Evaluating...")
        baseline_auroc = evaluate_temporal(model, train, test, n_ent, device, baseline_unc)
        coverage_auroc = evaluate_temporal(model, train, test, n_ent, device, coverage_unc)
        combined_auroc = evaluate_temporal(model, train, test, n_ent, device, combined)
        
        results[ds_name] = {
            'baseline': baseline_auroc,
            'coverage': coverage_auroc,
            'combined': combined_auroc,
        }
        
        print(f"\nResults (seed=42):")
        print(f"  Baseline:        {baseline_auroc:.4f}")
        print(f"  Coverage:        {coverage_auroc:.4f}")
        print(f"  Combined:        {combined_auroc:.4f}")
        print(f"  Improvement:     {combined_auroc - baseline_auroc:+.4f}")
        print()
    
    # Save
    output_dir = Path("/sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior/outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "baseline_coverage_quick.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {output_dir / 'baseline_coverage_quick.json'}")


if __name__ == "__main__":
    main()
