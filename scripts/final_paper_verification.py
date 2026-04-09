#!/usr/bin/env python3
"""
Final verification: Paper claims vs experimental data.
Focus on NOVEL-CONTEXT detection (the paper's main claim).
"""

import json
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")

def load_json(filename):
    with open(OUTPUT_DIR / filename) as f:
        return json.load(f)

def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    return np.sqrt(sum((x - m)**2 for x in values) / len(values))

print("="*80)
print("PAPER CLAIM VERIFICATION: Novel-Context Detection")
print("="*80)

# Load all baselines
fb_missing = load_json("fb15k237_missing_baselines.json")
wn_missing = load_json("wn18rr_missing_baselines.json")
yago_missing = load_json("yago_missing_baselines.json")
icews_missing = load_json("icews14_missing_baselines.json")

# Load canonical results for Energy/Coverage
canonical = load_json("canonical_temporal_results_v2.json")
icews = load_json("icews14_temporal_results.json")

seeds = [42, 123, 456]

# =============================================================================
# NOVEL-CONTEXT AUROC (the paper's main claim)
# =============================================================================

print("\n" + "="*80)
print("NOVEL-CONTEXT AUROC (detecting cov=0 queries)")
print("Paper claims MC Dropout/DeepEns < 0.5 (worse than random)")
print("="*80)

datasets = {
    'FB15k-237': {
        'missing': fb_missing,
        'canonical': canonical.get('fb15k237', {}),
    },
    'WN18RR': {
        'missing': wn_missing,
        'canonical': canonical.get('wn18rr', {}),
    },
    'YAGO3-10': {
        'missing': yago_missing,
        'canonical': None,  # load separately
    },
    'ICEWS14': {
        'missing': icews_missing,
        'canonical': icews,
    }
}

# Collect novel_ctx_auroc for each method
results = {}

for ds_name, ds_data in datasets.items():
    results[ds_name] = {}
    missing = ds_data['missing']

    # MC Dropout
    mc_nc = [missing[f'seed_{s}']['MCDropout'].get('novel_ctx_auroc', None) for s in seeds]
    mc_nc = [x for x in mc_nc if x is not None]
    if mc_nc:
        results[ds_name]['MCDropout'] = {'mean': mean(mc_nc), 'std': std(mc_nc), 'values': mc_nc}

    # Deep Ensemble
    de_nc = [missing[f'seed_{s}']['DeepEnsemble'].get('novel_ctx_auroc', None) for s in seeds]
    de_nc = [x for x in de_nc if x is not None]
    if de_nc:
        results[ds_name]['DeepEnsemble'] = {'mean': mean(de_nc), 'std': std(de_nc), 'values': de_nc}

    # SNGP
    sngp_nc = [missing[f'seed_{s}']['SNGP'].get('novel_ctx_auroc', None) for s in seeds]
    sngp_nc = [x for x in sngp_nc if x is not None]
    if sngp_nc:
        results[ds_name]['SNGP'] = {'mean': mean(sngp_nc), 'std': std(sngp_nc), 'values': sngp_nc}

# Add Energy from canonical
for ds_name in ['FB15k-237', 'WN18RR']:
    key = ds_name.lower().replace('-', '')
    if key in canonical:
        summary = canonical[key].get('summary', {})
        energy_nc = summary.get('Energy', {}).get('novel_ctx_auroc_mean', None)
        if energy_nc:
            results[ds_name]['Energy'] = {'mean': energy_nc, 'std': summary.get('Energy', {}).get('novel_ctx_auroc_std', 0)}

# ICEWS14 Energy
if 'summary' in icews:
    results['ICEWS14']['Energy'] = {
        'mean': icews['summary']['Energy']['novel_ctx_auroc_mean'],
        'std': icews['summary']['Energy']['novel_ctx_auroc_std']
    }

# YAGO Energy - load separately
yago_energy_files = [f"yago_temporal_energy_seed{s}.json" for s in seeds]
yago_energy_nc = []
for f in yago_energy_files:
    try:
        data = load_json(f)
        yago_energy_nc.append(data['temporal']['novel_ctx_auroc'])
    except:
        pass
if yago_energy_nc:
    results['YAGO3-10']['Energy'] = {'mean': mean(yago_energy_nc), 'std': std(yago_energy_nc)}

# Print results
print("\n| Dataset    | MCDrop | DeepEns | SNGP  | Energy | Coverage |")
print("|------------|--------|---------|-------|--------|----------|")
for ds in ['FB15k-237', 'WN18RR', 'YAGO3-10', 'ICEWS14']:
    mc = results[ds].get('MCDropout', {}).get('mean', 'N/A')
    de = results[ds].get('DeepEnsemble', {}).get('mean', 'N/A')
    sngp = results[ds].get('SNGP', {}).get('mean', 'N/A')
    energy = results[ds].get('Energy', {}).get('mean', 'N/A')

    mc_str = f"{mc:.3f}" if isinstance(mc, float) else mc
    de_str = f"{de:.3f}" if isinstance(de, float) else de
    sngp_str = f"{sngp:.3f}" if isinstance(sngp, float) else sngp
    energy_str = f"{energy:.3f}" if isinstance(energy, float) else energy

    print(f"| {ds:<10} | {mc_str:>6} | {de_str:>7} | {sngp_str:>5} | {energy_str:>6} | 1.000    |")

# =============================================================================
# Check paper claims
# =============================================================================

print("\n" + "="*80)
print("VERIFICATION AGAINST PAPER CLAIMS")
print("="*80)

print("\nPaper Table 3 claims (Temporal task, overall AUROC):")
paper_claims = {
    'FB15k-237': {'MCDrop': 0.449, 'DeepEns': 0.476, 'Energy': 0.589, 'Coverage': 0.664},
    'WN18RR': {'MCDrop': 0.381, 'DeepEns': 0.514, 'Energy': 0.851, 'Coverage': 0.724},
    'YAGO3-10': {'MCDrop': 0.423, 'DeepEns': 0.498, 'Energy': 0.612, 'Coverage': 0.701},
    'ICEWS14': {'MCDrop': 0.379, 'DeepEns': 0.410, 'Energy': 0.590, 'Coverage': 0.993},
}

# Compare with OVERALL AUROC (not novel_ctx)
print("\nComparing with OVERALL temporal AUROC from data:")
for ds in ['FB15k-237', 'WN18RR', 'YAGO3-10', 'ICEWS14']:
    key = ds.lower().replace('-', '').replace('3-10', '310')
    missing = datasets[ds]['missing']

    mc_overall = [missing[f'seed_{s}']['MCDropout'].get('overall_auroc', None) for s in seeds]
    mc_overall = [x for x in mc_overall if x is not None]
    de_overall = [missing[f'seed_{s}']['DeepEnsemble'].get('overall_auroc', None) for s in seeds]
    de_overall = [x for x in de_overall if x is not None]

    print(f"\n{ds}:")
    if mc_overall:
        print(f"  MCDropout overall: {mean(mc_overall):.3f} (paper: {paper_claims[ds]['MCDrop']})")
    if de_overall:
        print(f"  DeepEns overall:   {mean(de_overall):.3f} (paper: {paper_claims[ds]['DeepEns']})")

# =============================================================================
# Final assessment
# =============================================================================

print("\n" + "="*80)
print("ASSESSMENT")
print("="*80)

print("""
Key Finding:
- The paper's Table 3 shows AUROC values that DON'T match experimental data
- MC Dropout FB15k-237: Paper says .449, data shows ~0.626 (overall) / ~0.606 (novel_ctx)
- Deep Ensemble FB15k-237: Paper says .476, data shows ~0.606 (overall) / ~0.588 (novel_ctx)

Possible explanations:
1. Different experimental setup (different model, different training)
2. Different task definition (Error prediction vs Temporal OOD)
3. Numbers from a different experiment run that wasn't saved
4. Typos in the paper

Recommendation:
- Re-run the experiments with a clean script to generate authoritative numbers
- Update paper Table 3 with verified results
- If MC Dropout/DeepEns are NOT < 0.5, the paper's main claim needs revision
""")

# Check which individual seeds show < 0.5
print("\n" + "="*80)
print("INDIVIDUAL SEED ANALYSIS: Which seeds show AUROC < 0.5?")
print("="*80)

for ds in ['FB15k-237', 'WN18RR', 'YAGO3-10', 'ICEWS14']:
    missing = datasets[ds]['missing']
    print(f"\n{ds}:")
    for method in ['MCDropout', 'DeepEnsemble', 'SNGP']:
        below_random = []
        for s in seeds:
            nc = missing.get(f'seed_{s}', {}).get(method, {}).get('novel_ctx_auroc', None)
            if nc is not None and nc < 0.5:
                below_random.append((s, nc))
        if below_random:
            print(f"  {method}: Seeds with novel_ctx_auroc < 0.5: {below_random}")
        else:
            all_nc = [missing.get(f'seed_{s}', {}).get(method, {}).get('novel_ctx_auroc', None) for s in seeds]
            all_nc = [x for x in all_nc if x is not None]
            if all_nc:
                print(f"  {method}: All seeds >= 0.5 (values: {[f'{x:.3f}' for x in all_nc]})")
