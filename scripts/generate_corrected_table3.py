#!/usr/bin/env python3
"""
Generate corrected Table 3 for NeurIPS Position Paper.
Based on actual experimental data.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")

def load_json(filename):
    with open(OUTPUT_DIR / filename) as f:
        return json.load(f)

def mean(values):
    return sum(values) / len(values)

print("="*80)
print("CORRECTED TABLE 3: Uncertainty Method Comparison (AUROC)")
print("Based on experimental data")
print("="*80)

# Load all data
canonical = load_json("canonical_temporal_results_v2.json")
fb_missing = load_json("fb15k237_missing_baselines.json")
wn_missing = load_json("wn18rr_missing_baselines.json")
yago_missing = load_json("yago_missing_baselines.json")
icews = load_json("icews14_temporal_results.json")
icews_missing = load_json("icews14_missing_baselines.json")

# Compute means for MC Dropout and Deep Ensemble
def get_mc_de_means(data):
    mc_vals = [data[f"seed_{s}"]["MCDropout"]["overall_auroc"] for s in [42, 123, 456]]
    de_vals = [data[f"seed_{s}"]["DeepEnsemble"]["overall_auroc"] for s in [42, 123, 456]]
    return mean(mc_vals), mean(de_vals)

fb_mc, fb_de = get_mc_de_means(fb_missing)
yago_mc, yago_de = get_mc_de_means(yago_missing)

# WN18RR missing baselines - check structure
wn_missing_data = load_json("wn18rr_missing_baselines.json")
print("\nWN18RR missing baselines structure:", list(wn_missing_data.keys())[:3])

# ICEWS14 missing baselines
icews_missing_data = load_json("icews14_missing_baselines.json")
print("ICEWS14 missing baselines structure:", list(icews_missing_data.keys())[:3])

# Get Energy and Coverage from canonical
fb_energy = canonical["fb15k237"]["summary"]["Energy"]["temporal_auroc_mean"]
fb_cov = canonical["fb15k237"]["summary"]["CoverageOnly"]["temporal_auroc_mean"]
wn_energy = canonical["wn18rr"]["summary"]["Energy"]["temporal_auroc_mean"]
wn_cov = canonical["wn18rr"]["summary"]["CoverageOnly"]["temporal_auroc_mean"]

# ICEWS14
icews_energy = icews["summary"]["Energy"]["overall_auroc_mean"]
icews_cov = icews["summary"]["CoverageOnly"]["overall_auroc_mean"]

# YAGO - need to load separately
yago_energy_files = [f"yago_temporal_energy_seed{s}.json" for s in [42, 123, 456]]
yago_cov_files = [f"yago_temporal_coverageonly_seed{s}.json" for s in [42, 123, 456]]

yago_energy_vals = []
yago_cov_vals = []
for f in yago_energy_files:
    data = load_json(f)
    yago_energy_vals.append(data["temporal"]["overall_auroc"])
for f in yago_cov_files:
    data = load_json(f)
    yago_cov_vals.append(data["temporal"]["overall_auroc"])

yago_energy = mean(yago_energy_vals)
yago_cov = mean(yago_cov_vals)

print("\n" + "="*80)
print("TEMPORAL TASK - OVERALL AUROC (distinguishing train from test)")
print("="*80)

print("\n| Dataset    | MC Drop | DeepEns | Energy | Coverage | E+C    |")
print("|------------|---------|---------|--------|----------|--------|")
print(f"| FB15k-237  | {fb_mc:.3f}   | {fb_de:.3f}   | {fb_energy:.3f}  | {fb_cov:.3f}    | {max(fb_energy, fb_cov):.3f}  |")
print(f"| WN18RR     | TBD     | TBD     | {wn_energy:.3f}  | {wn_cov:.3f}    | {max(wn_energy, wn_cov):.3f}  |")
print(f"| YAGO3-10   | {yago_mc:.3f}   | {yago_de:.3f}   | {yago_energy:.3f}  | {yago_cov:.3f}    | {max(yago_energy, yago_cov):.3f}  |")
print(f"| ICEWS14    | TBD     | TBD     | {icews_energy:.3f}  | {icews_cov:.3f}    | {icews_cov:.3f}  |")

# Now check novel-context specifically (what paper claims)
print("\n" + "="*80)
print("NOVEL-CONTEXT DETECTION AUROC (the blind spot)")
print("This is what the paper's argument focuses on")
print("="*80)

fb_mc_nc = mean([fb_missing[f"seed_{s}"]["MCDropout"]["novel_ctx_auroc"] for s in [42, 123, 456]])
fb_de_nc = mean([fb_missing[f"seed_{s}"]["DeepEnsemble"]["novel_ctx_auroc"] for s in [42, 123, 456]])
yago_mc_nc = mean([yago_missing[f"seed_{s}"]["MCDropout"]["novel_ctx_auroc"] for s in [42, 123, 456]])
yago_de_nc = mean([yago_missing[f"seed_{s}"]["DeepEnsemble"]["novel_ctx_auroc"] for s in [42, 123, 456]])

fb_energy_nc = canonical["fb15k237"]["summary"]["Energy"]["novel_ctx_auroc_mean"]
wn_energy_nc = canonical["wn18rr"]["summary"]["Energy"]["novel_ctx_auroc_mean"]
fb_cov_nc = canonical["fb15k237"]["summary"]["CoverageOnly"]["novel_ctx_auroc_mean"]
wn_cov_nc = canonical["wn18rr"]["summary"]["CoverageOnly"]["novel_ctx_auroc_mean"]

icews_energy_nc = icews["summary"]["Energy"]["novel_ctx_auroc_mean"]
icews_cov_nc = icews["summary"]["CoverageOnly"]["novel_ctx_auroc_mean"]

yago_energy_nc = mean([load_json(f)["temporal"]["novel_ctx_auroc"] for f in yago_energy_files])
yago_cov_nc = mean([load_json(f)["temporal"]["novel_ctx_auroc"] for f in yago_cov_files])

print("\n| Dataset    | MC Drop | DeepEns | Energy | Coverage |")
print("|------------|---------|---------|--------|----------|")
print(f"| FB15k-237  | {fb_mc_nc:.3f}   | {fb_de_nc:.3f}   | {fb_energy_nc:.3f}  | {fb_cov_nc:.3f}    |")
print(f"| WN18RR     | TBD     | TBD     | {wn_energy_nc:.3f}  | {wn_cov_nc:.3f}    |")
print(f"| YAGO3-10   | {yago_mc_nc:.3f}   | {yago_de_nc:.3f}   | {yago_energy_nc:.3f}  | {yago_cov_nc:.3f}    |")
print(f"| ICEWS14    | TBD     | TBD     | {icews_energy_nc:.3f}  | {icews_cov_nc:.3f}    |")

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)
print(f"""
For NOVEL-CONTEXT detection:
- MC Dropout: {fb_mc_nc:.3f} (FB15k) - ABOVE random (0.5), not below!
- Deep Ensemble: {fb_de_nc:.3f} (FB15k) - ABOVE random!
- Coverage: {fb_cov_nc:.3f} (FB15k) - Perfect (by definition)

The paper's claim that MC Dropout/DeepEns perform WORSE than random
on novel-context appears to be INCORRECT based on this data.

However, looking at YAGO:
- MC Dropout: {yago_mc_nc:.3f} - BELOW random!
- Deep Ensemble: {yago_de_nc:.3f} - ABOVE random

The picture is mixed across datasets.
""")
