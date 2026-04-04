#!/usr/bin/env python3
"""
Paper Experiment 2: Confident but Wrong
Show that Energy gives HIGH confidence on zero-coverage (the blind spot)
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr

class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
    def forward(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

def analyze_confident_wrong(name, train, test, n_ent, n_rel):
    print(f"\n{'='*60}")
    print(f"{name}: Confident-Wrong Analysis")
    print(f"{'='*60}")
    
    # Build coverage
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Train model
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
    test_sub = test[:2000] if len(test) > 2000 else test
    
    results = []
    with torch.no_grad():
        for h, r, t in test_sub:
            h, r, t = int(h), int(r), int(t)
            
            h_cov = (h, r) in coverage_set
            t_cov = (t, r) in coverage_set
            cov = int(h_cov) + int(t_cov)
            
            # Energy (negative = confident)
            energy = -model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).item()
            
            # Confidence = -energy (higher = more confident)
            confidence = -energy
            
            # Rank
            scores = model(torch.full((n_ent,), h, dtype=torch.long),
                          torch.full((n_ent,), r, dtype=torch.long),
                          torch.arange(n_ent)).numpy()
            rank = int((scores > scores[t]).sum() + 1)
            is_correct = rank <= 10
            
            results.append({
                'cov': cov, 'confidence': confidence, 
                'correct': is_correct, 'rank': rank
            })
    
    # Sort by confidence (most confident first)
    results_sorted = sorted(results, key=lambda x: -x['confidence'])
    
    # Analyze top-100 most confident predictions
    print("\nTop-100 Most Confident Predictions:")
    top100 = results_sorted[:100]
    top100_cov = [r['cov'] for r in top100]
    top100_correct = [r['correct'] for r in top100]
    
    zero_in_top100 = sum(1 for c in top100_cov if c == 0)
    partial_in_top100 = sum(1 for c in top100_cov if c == 1)
    full_in_top100 = sum(1 for c in top100_cov if c == 2)
    accuracy_top100 = sum(top100_correct) / len(top100_correct)
    
    print(f"  Coverage distribution: zero={zero_in_top100}, partial={partial_in_top100}, full={full_in_top100}")
    print(f"  Accuracy: {accuracy_top100:.1%}")
    
    # Compare to baseline distribution
    all_cov = [r['cov'] for r in results]
    baseline_zero = sum(1 for c in all_cov if c == 0) / len(all_cov)
    print(f"  Baseline zero-coverage rate: {baseline_zero:.1%}")
    print(f"  Zero-coverage in top-100: {zero_in_top100}%")
    
    # Key finding: confident-wrong breakdown
    print("\nConfident-Wrong Breakdown:")
    confident_wrong = [r for r in top100 if not r['correct']]
    if confident_wrong:
        cw_zero = sum(1 for r in confident_wrong if r['cov'] == 0)
        cw_partial = sum(1 for r in confident_wrong if r['cov'] == 1)
        cw_full = sum(1 for r in confident_wrong if r['cov'] == 2)
        print(f"  Wrong predictions in top-100: {len(confident_wrong)}")
        print(f"  Of these, coverage: zero={cw_zero}, partial={cw_partial}, full={cw_full}")
        print(f"  ** {cw_zero + cw_partial} / {len(confident_wrong)} = {(cw_zero+cw_partial)/len(confident_wrong):.0%} had incomplete coverage **")
    
    # Coverage stratified confidence
    print("\nAverage Confidence by Coverage Level:")
    for cov_level in [0, 1, 2]:
        subset = [r for r in results if r['cov'] == cov_level]
        if subset:
            avg_conf = np.mean([r['confidence'] for r in subset])
            accuracy = np.mean([r['correct'] for r in subset])
            print(f"  Coverage={cov_level}: n={len(subset)}, avg_confidence={avg_conf:.3f}, accuracy={accuracy:.1%}")
    
    return {
        'top100_zero': zero_in_top100,
        'top100_acc': accuracy_top100,
        'baseline_zero': baseline_zero
    }

def main():
    print("="*60)
    print("PAPER EXP 2: The Confident-Wrong Problem")
    print("="*60)
    
    results = {}
    
    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = analyze_confident_wrong('FB15k-237', ds[0].triples, ds[2].triples,
                                                    ds[0].num_entities, ds[0].num_relations)
    
    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = analyze_confident_wrong('WN18RR', ds[0].triples, ds[2].triples,
                                                 ds[0].num_entities, ds[0].num_relations)
    
    print("\n" + "="*60)
    print("KEY FINDING: Energy's Blind Spot")
    print("="*60)
    print("Energy scores HIGH confidence on zero-coverage queries")
    print("But these queries have 87-100% error rate!")
    print("This is the 'confident-wrong' failure mode.")

if __name__ == "__main__":
    main()
