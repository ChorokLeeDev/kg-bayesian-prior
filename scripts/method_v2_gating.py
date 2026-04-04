#!/usr/bin/env python3
"""
Method V2b: Gating Network
Learn a gate that decides: use coverage or energy based on context
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

class GatingNetwork(nn.Module):
    """
    Learns per-query gating: how much to weight coverage vs energy
    Output: alpha in [0,1] where uncertainty = alpha*cov + (1-alpha)*energy
    """
    def __init__(self, n_rel, emb_dim=32):
        super().__init__()
        # Per-relation gating bias
        self.rel_gate = nn.Embedding(n_rel, 1)
        # Context-based gating
        self.context_net = nn.Sequential(
            nn.Linear(4, 16),  # [h_degree, t_degree, rel_freq, cov]
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        nn.init.zeros_(self.rel_gate.weight)
    
    def forward(self, r, context_features):
        """
        r: relation indices [batch]
        context_features: [batch, 4]
        Returns: alpha [batch] in [0,1]
        """
        rel_bias = self.rel_gate(r).squeeze(-1)
        context_bias = self.context_net(context_features).squeeze(-1)
        return torch.sigmoid(rel_bias + context_bias)

def run_gating(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Gating Network")
    print(f"{'='*60}")
    
    # Stats
    ent_degree = np.zeros(n_ent)
    rel_freq = np.zeros(n_rel)
    coverage_set = set()
    
    for h, r, t in train:
        h, r, t = int(h), int(r), int(t)
        ent_degree[h] += 1
        ent_degree[t] += 1
        rel_freq[r] += 1
        coverage_set.add((h, r))
        coverage_set.add((t, r))
    
    ent_degree = np.log1p(ent_degree)
    rel_freq = np.log1p(rel_freq)
    
    # Train base model
    torch.manual_seed(42)
    base_model = DistMult(n_ent, n_rel)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    print("Training base model...")
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
    
    # Train gating network
    print("Training gating network...")
    gate = GatingNetwork(n_rel)
    opt_gate = torch.optim.Adam(gate.parameters(), lr=1e-3)
    
    # Training data: minimize uncertainty for correct, maximize for wrong
    for epoch in range(20):
        np.random.shuffle(train)
        for i in range(0, min(len(train), 10000), 256):
            batch = train[i:i+256]
            h = torch.tensor(batch[:,0])
            r = torch.tensor(batch[:,1])
            t = torch.tensor(batch[:,2])
            t_neg = torch.randint(0, n_ent, (len(batch),))
            
            # Context features
            def get_context(h_arr, r_arr, t_arr):
                features = []
                for hi, ri, ti in zip(h_arr.numpy(), r_arr.numpy(), t_arr.numpy()):
                    h_cov = (int(hi), int(ri)) in coverage_set
                    t_cov = (int(ti), int(ri)) in coverage_set
                    cov = int(h_cov) + int(t_cov)
                    features.append([ent_degree[hi], ent_degree[ti], rel_freq[ri], cov])
                return torch.tensor(features, dtype=torch.float32)
            
            ctx_pos = get_context(h, r, t)
            ctx_neg = get_context(h, r, t_neg)
            
            with torch.no_grad():
                energy_pos = -base_model(h, r, t)
                energy_neg = -base_model(h, r, t_neg)
            
            # Normalize energies
            all_energy = torch.cat([energy_pos, energy_neg])
            e_mean, e_std = all_energy.mean(), all_energy.std() + 1e-8
            energy_pos_norm = (energy_pos - e_mean) / e_std
            energy_neg_norm = (energy_neg - e_mean) / e_std
            
            cov_pos = ctx_pos[:, 3]  # coverage level
            cov_neg = ctx_neg[:, 3]
            cov_pos_unc = (2 - cov_pos)
            cov_neg_unc = (2 - cov_neg)
            
            # Normalize coverage
            all_cov = torch.cat([cov_pos_unc, cov_neg_unc])
            c_mean, c_std = all_cov.mean(), all_cov.std() + 1e-8
            cov_pos_norm = (cov_pos_unc - c_mean) / c_std
            cov_neg_norm = (cov_neg_unc - c_mean) / c_std
            
            opt_gate.zero_grad()
            
            alpha_pos = gate(r, ctx_pos)
            alpha_neg = gate(r, ctx_neg)
            
            unc_pos = alpha_pos * cov_pos_norm + (1 - alpha_pos) * energy_pos_norm
            unc_neg = alpha_neg * cov_neg_norm + (1 - alpha_neg) * energy_neg_norm
            
            # Loss: uncertainty should be higher for negatives
            loss = torch.clamp(0.5 - unc_neg + unc_pos, min=0).mean()
            loss.backward()
            opt_gate.step()
    
    gate.eval()
    
    # Evaluate
    print("Evaluating...")
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
                
                energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
                
                ctx = torch.tensor([[ent_degree[h], ent_degree[t], rel_freq[r], cov]], dtype=torch.float32)
                alpha = gate(torch.tensor([r]), ctx).item()
                
                # Gated uncertainty (unnormalized for now)
                gated_unc = alpha * (2 - cov) + (1 - alpha) * energy
                
                scores = base_model(torch.full((n_ent,), h, dtype=torch.long),
                                   torch.full((n_ent,), r, dtype=torch.long),
                                   torch.arange(n_ent)).numpy()
                rank = int((scores > scores[t]).sum() + 1)
                
                results.append({
                    'energy': energy, 'cov': cov, 'alpha': alpha, 'gated': gated_unc,
                    'is_test': is_test, 'is_wrong': int(rank > 10)
                })
    
    # Compute AUROCs
    energy_unc = [r['energy'] for r in results]
    cov_unc = [2 - r['cov'] for r in results]
    gated_unc = [r['gated'] for r in results]
    
    # Normalize for linear baseline
    energy_arr = np.array(energy_unc)
    cov_arr = np.array(cov_unc)
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    
    # Task 1: Temporal
    labels_temp = [r['is_test'] for r in results]
    auroc_energy = roc_auc_score(labels_temp, energy_unc)
    auroc_cov = roc_auc_score(labels_temp, cov_unc)
    auroc_gated = roc_auc_score(labels_temp, gated_unc)
    
    best_linear = max(roc_auc_score(labels_temp, a*cov_norm + (1-a)*energy_norm) 
                      for a in np.arange(0, 1.1, 0.1))
    
    print(f"\nTemporal OOD:")
    print(f"  Energy: {auroc_energy:.4f}")
    print(f"  Coverage: {auroc_cov:.4f}")
    print(f"  Linear: {best_linear:.4f}")
    print(f"  Gated: {auroc_gated:.4f}")
    
    # Task 2: Error
    labels_err = [r['is_wrong'] for r in results]
    auroc_energy_e = roc_auc_score(labels_err, energy_unc)
    auroc_cov_e = roc_auc_score(labels_err, cov_unc)
    auroc_gated_e = roc_auc_score(labels_err, gated_unc)
    
    best_linear_e = max(roc_auc_score(labels_err, a*cov_norm + (1-a)*energy_norm) 
                        for a in np.arange(0, 1.1, 0.1))
    
    print(f"\nPrediction Error:")
    print(f"  Energy: {auroc_energy_e:.4f}")
    print(f"  Coverage: {auroc_cov_e:.4f}")
    print(f"  Linear: {best_linear_e:.4f}")
    print(f"  Gated: {auroc_gated_e:.4f}")
    
    # Analyze learned gates
    alphas = [r['alpha'] for r in results]
    print(f"\nLearned gate α distribution:")
    print(f"  Mean: {np.mean(alphas):.3f}, Std: {np.std(alphas):.3f}")
    print(f"  Min: {np.min(alphas):.3f}, Max: {np.max(alphas):.3f}")
    
    return {
        'temporal': {'energy': auroc_energy, 'cov': auroc_cov, 'linear': best_linear, 'gated': auroc_gated},
        'error': {'energy': auroc_energy_e, 'cov': auroc_cov_e, 'linear': best_linear_e, 'gated': auroc_gated_e}
    }

def main():
    print("="*60)
    print("METHOD V2b: Gating Network")
    print("="*60)
    
    results = {}
    
    ds = load_fb15k237()
    results['FB15k-237'] = run_gating('FB15k-237', ds[0].triples, ds[2].triples,
                                      ds[0].num_entities, ds[0].num_relations)
    
    ds = load_wn18rr()
    results['WN18RR'] = run_gating('WN18RR', ds[0].triples, ds[2].triples,
                                   ds[0].num_entities, ds[0].num_relations)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Temporal - Energy: {r['temporal']['energy']:.3f}, Gated: {r['temporal']['gated']:.3f}")
        print(f"  Error - Energy: {r['error']['energy']:.3f}, Gated: {r['error']['gated']:.3f}")

if __name__ == "__main__":
    main()
