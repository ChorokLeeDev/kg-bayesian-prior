#!/usr/bin/env python3
"""
Method V2: Neural Ensemble
Learn when to trust coverage vs energy based on context features
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

class NeuralUncertaintyEnsemble(nn.Module):
    """
    Learns to combine coverage and energy based on context.
    Input: [energy, coverage, rel_freq, h_degree, t_degree, ...]
    Output: uncertainty score
    """
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
    
    def forward(self, features):
        return self.net(features).squeeze(-1)

def compute_features(train, n_ent, n_rel):
    """Precompute entity/relation statistics"""
    # Entity degree
    ent_degree = np.zeros(n_ent)
    # Relation frequency
    rel_freq = np.zeros(n_rel)
    # Coverage set
    coverage_set = set()
    coverage_count = {}
    
    for h, r, t in train:
        h, r, t = int(h), int(r), int(t)
        ent_degree[h] += 1
        ent_degree[t] += 1
        rel_freq[r] += 1
        coverage_set.add((h, r))
        coverage_set.add((t, r))
        for key in [(h, r), (t, r)]:
            coverage_count[key] = coverage_count.get(key, 0) + 1
    
    # Normalize
    ent_degree = np.log1p(ent_degree)
    rel_freq = np.log1p(rel_freq)
    
    return ent_degree, rel_freq, coverage_set, coverage_count

def get_features(h, r, t, energy, ent_degree, rel_freq, coverage_set, coverage_count):
    """Get feature vector for a triple"""
    h, r, t = int(h), int(r), int(t)
    
    h_cov = (h, r) in coverage_set
    t_cov = (t, r) in coverage_set
    cov = int(h_cov) + int(t_cov)
    
    h_count = coverage_count.get((h, r), 0)
    t_count = coverage_count.get((t, r), 0)
    
    features = [
        energy,                          # Energy score
        cov,                             # Coverage level (0/1/2)
        np.log1p(h_count),              # Head coverage count
        np.log1p(t_count),              # Tail coverage count
        ent_degree[h],                   # Head degree
        ent_degree[t],                   # Tail degree
        rel_freq[r],                     # Relation frequency
        ent_degree[h] - ent_degree[t],  # Degree difference
    ]
    return features

def run_neural_ensemble(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Neural Uncertainty Ensemble")
    print(f"{'='*60}")
    
    # Precompute features
    ent_degree, rel_freq, coverage_set, coverage_count = compute_features(train, n_ent, n_rel)
    
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
    
    # Collect training data for uncertainty ensemble
    # Use validation-like split: train on first 80% of train, validate on rest
    n_train_unc = int(len(train) * 0.8)
    train_unc = train[:n_train_unc]
    val_unc = train[n_train_unc:]
    
    print("Collecting uncertainty training data...")
    unc_train_data = []
    
    with torch.no_grad():
        # Positive examples (from training data) - should have LOW uncertainty
        for h, r, t in train_unc[:5000]:
            energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            features = get_features(h, r, t, energy, ent_degree, rel_freq, coverage_set, coverage_count)
            unc_train_data.append((features, 0))  # 0 = low uncertainty (correct)
        
        # Negative examples (corrupted triples) - should have HIGH uncertainty
        for h, r, t in train_unc[:5000]:
            t_neg = np.random.randint(0, n_ent)
            energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t_neg])).item()
            features = get_features(h, r, t_neg, energy, ent_degree, rel_freq, coverage_set, coverage_count)
            unc_train_data.append((features, 1))  # 1 = high uncertainty (wrong)
    
    # Train neural ensemble
    print("Training neural ensemble...")
    ensemble = NeuralUncertaintyEnsemble()
    opt_ens = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    
    np.random.shuffle(unc_train_data)
    X_train = torch.tensor([d[0] for d in unc_train_data], dtype=torch.float32)
    y_train = torch.tensor([d[1] for d in unc_train_data], dtype=torch.float32)
    
    for epoch in range(20):
        for i in range(0, len(X_train), 256):
            X_batch = X_train[i:i+256]
            y_batch = y_train[i:i+256]
            
            opt_ens.zero_grad()
            pred = ensemble(X_batch)
            loss = nn.BCEWithLogitsLoss()(pred, y_batch)
            loss.backward()
            opt_ens.step()
    
    ensemble.eval()
    
    # Evaluate on test set
    print("Evaluating...")
    test_sub = test[:2000] if len(test) > 2000 else test
    n_train_sample = min(1000, len(train))
    train_sample = train[np.random.choice(len(train), n_train_sample, replace=False)]
    
    results = []
    with torch.no_grad():
        # Train samples (ID)
        for h, r, t in train_sample:
            energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            features = get_features(h, r, t, energy, ent_degree, rel_freq, coverage_set, coverage_count)
            neural_unc = ensemble(torch.tensor([features], dtype=torch.float32)).item()
            
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            cov = int(h_cov) + int(t_cov)
            
            # Compute rank
            scores = base_model(torch.full((n_ent,), h, dtype=torch.long),
                               torch.full((n_ent,), r, dtype=torch.long),
                               torch.arange(n_ent)).numpy()
            rank = int((scores > scores[int(t)]).sum() + 1)
            
            results.append({
                'energy': energy, 'cov': cov, 'neural': neural_unc,
                'is_test': 0, 'is_wrong': int(rank > 10)
            })
        
        # Test samples (OOD)
        for h, r, t in test_sub:
            energy = -base_model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            features = get_features(h, r, t, energy, ent_degree, rel_freq, coverage_set, coverage_count)
            neural_unc = ensemble(torch.tensor([features], dtype=torch.float32)).item()
            
            h_cov = (int(h), int(r)) in coverage_set
            t_cov = (int(t), int(r)) in coverage_set
            cov = int(h_cov) + int(t_cov)
            
            scores = base_model(torch.full((n_ent,), h, dtype=torch.long),
                               torch.full((n_ent,), r, dtype=torch.long),
                               torch.arange(n_ent)).numpy()
            rank = int((scores > scores[int(t)]).sum() + 1)
            
            results.append({
                'energy': energy, 'cov': cov, 'neural': neural_unc,
                'is_test': 1, 'is_wrong': int(rank > 10)
            })
    
    # Compute AUROCs
    energy_unc = [r['energy'] for r in results]
    cov_unc = [2 - r['cov'] for r in results]
    neural_unc = [r['neural'] for r in results]
    
    # Normalize for linear ensemble
    energy_arr = np.array(energy_unc)
    cov_arr = np.array(cov_unc)
    energy_norm = (energy_arr - energy_arr.mean()) / (energy_arr.std() + 1e-8)
    cov_norm = (cov_arr - cov_arr.mean()) / (cov_arr.std() + 1e-8)
    
    # Task 1: Temporal OOD
    labels_temp = [r['is_test'] for r in results]
    
    auroc_energy_temp = roc_auc_score(labels_temp, energy_unc)
    auroc_cov_temp = roc_auc_score(labels_temp, cov_unc)
    auroc_neural_temp = roc_auc_score(labels_temp, neural_unc)
    
    # Best linear ensemble
    best_linear = 0
    for alpha in np.arange(0, 1.1, 0.1):
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels_temp, ens)
        if auroc > best_linear:
            best_linear = auroc
    
    print(f"\nTemporal OOD:")
    print(f"  Energy: {auroc_energy_temp:.4f}")
    print(f"  Coverage: {auroc_cov_temp:.4f}")
    print(f"  Linear Ensemble: {best_linear:.4f}")
    print(f"  Neural Ensemble: {auroc_neural_temp:.4f}")
    print(f"  Neural vs best baseline: {auroc_neural_temp - max(auroc_energy_temp, auroc_cov_temp):+.4f}")
    
    # Task 2: Prediction Error
    labels_err = [r['is_wrong'] for r in results]
    
    auroc_energy_err = roc_auc_score(labels_err, energy_unc)
    auroc_cov_err = roc_auc_score(labels_err, cov_unc)
    auroc_neural_err = roc_auc_score(labels_err, neural_unc)
    
    best_linear_err = 0
    for alpha in np.arange(0, 1.1, 0.1):
        ens = alpha * cov_norm + (1-alpha) * energy_norm
        auroc = roc_auc_score(labels_err, ens)
        if auroc > best_linear_err:
            best_linear_err = auroc
    
    print(f"\nPrediction Error OOD:")
    print(f"  Energy: {auroc_energy_err:.4f}")
    print(f"  Coverage: {auroc_cov_err:.4f}")
    print(f"  Linear Ensemble: {best_linear_err:.4f}")
    print(f"  Neural Ensemble: {auroc_neural_err:.4f}")
    print(f"  Neural vs best baseline: {auroc_neural_err - max(auroc_energy_err, auroc_cov_err):+.4f}")
    
    return {
        'temporal': {'energy': auroc_energy_temp, 'cov': auroc_cov_temp, 
                    'linear': best_linear, 'neural': auroc_neural_temp},
        'error': {'energy': auroc_energy_err, 'cov': auroc_cov_err,
                 'linear': best_linear_err, 'neural': auroc_neural_err}
    }

def main():
    print("="*60)
    print("METHOD V2: Neural Uncertainty Ensemble")
    print("="*60)
    
    results = {}
    
    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = run_neural_ensemble('FB15k-237', ds[0].triples, ds[2].triples,
                                               ds[0].num_entities, ds[0].num_relations)
    
    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = run_neural_ensemble('WN18RR', ds[0].triples, ds[2].triples,
                                            ds[0].num_entities, ds[0].num_relations)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Neural Ensemble Results")
    print("="*60)
    
    print("\nTemporal OOD:")
    print(f"{'Dataset':<12} {'Energy':<10} {'Coverage':<10} {'Linear':<10} {'Neural':<10}")
    for name, r in results.items():
        t = r['temporal']
        print(f"{name:<12} {t['energy']:<10.4f} {t['cov']:<10.4f} {t['linear']:<10.4f} {t['neural']:<10.4f}")
    
    print("\nPrediction Error OOD:")
    print(f"{'Dataset':<12} {'Energy':<10} {'Coverage':<10} {'Linear':<10} {'Neural':<10}")
    for name, r in results.items():
        e = r['error']
        print(f"{name:<12} {e['energy']:<10.4f} {e['cov']:<10.4f} {e['linear']:<10.4f} {e['neural']:<10.4f}")

if __name__ == "__main__":
    main()
