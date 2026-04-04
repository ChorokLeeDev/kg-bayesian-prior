#!/usr/bin/env python3
"""
Paper Experiment 1: How often is zero-coverage in practice?
Analyze real query distributions across datasets
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237, load_wn18rr

def load_icews14():
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/ICEWS14")
    entity2id, relation2id = {}, {}
    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entity2id: entity2id[h] = len(entity2id)
                    if r not in relation2id: relation2id[r] = len(relation2id)
                    if t not in entity2id: entity2id[t] = len(entity2id)
                    triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)
    train = load_triples("train.txt")
    test = load_triples("test.txt")
    return train, test, len(entity2id), len(relation2id)

def load_yago():
    data_dir = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw/YAGO3-10")
    entity2id, relation2id = {}, {}
    def load_triples(filename):
        triples = []
        with open(data_dir / filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h not in entity2id: entity2id[h] = len(entity2id)
                    if r not in relation2id: relation2id[r] = len(relation2id)
                    if t not in entity2id: entity2id[t] = len(entity2id)
                    triples.append([entity2id[h], relation2id[r], entity2id[t]])
        return np.array(triples)
    train = load_triples("train.txt")
    test = load_triples("test.txt")
    return train, test, len(entity2id), len(relation2id)

def analyze_coverage(name, train, test):
    print(f"\n{'='*60}")
    print(f"{name}: Coverage Analysis")
    print(f"{'='*60}")
    
    # Build coverage from train
    coverage_set = set()
    for h, r, t in train:
        coverage_set.add((int(h), int(r)))
        coverage_set.add((int(t), int(r)))
    
    # Analyze test queries
    zero_cov = 0
    partial_cov = 0
    full_cov = 0
    
    for h, r, t in test:
        h_cov = (int(h), int(r)) in coverage_set
        t_cov = (int(t), int(r)) in coverage_set
        
        if h_cov and t_cov:
            full_cov += 1
        elif h_cov or t_cov:
            partial_cov += 1
        else:
            zero_cov += 1
    
    total = len(test)
    print(f"Test queries: {total}")
    print(f"  Zero coverage:    {zero_cov:>6} ({zero_cov/total:>6.1%})")
    print(f"  Partial coverage: {partial_cov:>6} ({partial_cov/total:>6.1%})")
    print(f"  Full coverage:    {full_cov:>6} ({full_cov/total:>6.1%})")
    
    return {
        'total': total,
        'zero': zero_cov,
        'partial': partial_cov,
        'full': full_cov,
        'zero_pct': zero_cov/total,
        'partial_pct': partial_cov/total,
        'full_pct': full_cov/total
    }

def main():
    print("="*60)
    print("PAPER EXP 1: Real-World Coverage Distribution")
    print("="*60)
    
    results = {}
    
    # FB15k-237
    ds = load_fb15k237()
    results['FB15k-237'] = analyze_coverage('FB15k-237', ds[0].triples, ds[2].triples)
    
    # WN18RR
    ds = load_wn18rr()
    results['WN18RR'] = analyze_coverage('WN18RR', ds[0].triples, ds[2].triples)
    
    # ICEWS14
    train, test, _, _ = load_icews14()
    results['ICEWS14'] = analyze_coverage('ICEWS14', train, test)
    
    # YAGO3-10
    try:
        train, test, _, _ = load_yago()
        results['YAGO3-10'] = analyze_coverage('YAGO3-10', train, test)
    except:
        print("\nYAGO3-10 not available")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Coverage Distribution in Test Sets")
    print("="*60)
    print(f"{'Dataset':<12} {'Total':<8} {'Zero%':<8} {'Partial%':<10} {'Full%':<8}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<12} {r['total']:<8} {r['zero_pct']:<8.1%} {r['partial_pct']:<10.1%} {r['full_pct']:<8.1%}")
    
    avg_zero = np.mean([r['zero_pct'] for r in results.values()])
    print("-"*50)
    print(f"{'AVERAGE':<12} {'':<8} {avg_zero:<8.1%}")
    print("\n** This is the X% for the abstract **")

if __name__ == "__main__":
    main()
