#!/usr/bin/env python3
"""
Verify paper numbers against experimental data.
Check consistency of all tables in neurips_position paper.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")

def load_json(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

print("="*80)
print("PAPER NUMBER VERIFICATION")
print("="*80)

issues = []

# =============================================================================
# Table 1: Error rates by coverage level
# Paper claims: Zero-coverage 67-100%, Full varies
# =============================================================================
print("\n--- TABLE 1: Error Rates by Coverage ---")
print("Paper claims:")
print("  FB15k-237 DistMult: Zero 88.2%, Partial 40.7%, Full 67.1%")
print("  FB15k-237 ComplEx:  Zero 70.6%, Partial 41.2%, Full 66.5%")
print("  WN18RR DistMult:    Zero 100%, Partial 82.8%, Full 32.8%")

# We need to check if these numbers exist in outputs
# These are 1-Hits@10 by coverage level
print("\n[MANUAL CHECK NEEDED] Error rates by coverage from compute_actual_error_rate.py")
print("  Need to verify Table 1 numbers match experimental data")

# =============================================================================
# Table 2: Coverage distribution
# =============================================================================
print("\n--- TABLE 2: Coverage Distribution ---")
canonical = load_json("canonical_temporal_results_v2.json")
if canonical:
    fb_split = canonical["fb15k237"]["seed_42"]["UKGE"]["temporal"]
    wn_split = canonical["wn18rr"]["seed_42"]["UKGE"]["temporal"]

    fb_total = fb_split["n_emerging"] + fb_split["n_novel_ctx"] + fb_split["n_id"]
    fb_zero_pct = 100 * fb_split["n_novel_ctx"] / fb_total  # novel_ctx = zero coverage for one entity

    print(f"FB15k-237 from data:")
    print(f"  n_emerging={fb_split['n_emerging']}, n_novel_ctx={fb_split['n_novel_ctx']}, n_id={fb_split['n_id']}")
    print(f"  Total={fb_total}")

    wn_total = wn_split["n_emerging"] + wn_split["n_novel_ctx"] + wn_split["n_id"]
    print(f"WN18RR from data:")
    print(f"  n_emerging={wn_split['n_emerging']}, n_novel_ctx={wn_split['n_novel_ctx']}, n_id={wn_split['n_id']}")
    print(f"  Total={wn_total}")

print("\nPaper Table 2 claims:")
print("  FB15k-237: Zero 1.6%, Partial 30.0%, Full 68.5%")
print("  WN18RR:    Zero 2.0%, Partial 43.8%, Full 54.1%")
print("\n[CHECK] These percentages should be verified against actual test split")

# =============================================================================
# Table 3: AUROC comparison - MC Dropout, Deep Ensemble, Energy, Coverage
# =============================================================================
print("\n--- TABLE 3: AUROC Comparison ---")

# MC Dropout ablation
mc_data = load_json("mc_dropout_ablation.json")
if mc_data:
    fb_mc = mc_data["datasets"]["fb15k237"]["summary"]
    wn_mc = mc_data["datasets"]["wn18rr"]["summary"]
    print(f"\nMC Dropout (novel-context detection):")
    print(f"  FB15k-237: {fb_mc['20']['mean']:.3f} ± {fb_mc['20']['std']:.3f}")
    print(f"  WN18RR:    {wn_mc['20']['mean']:.3f} ± {wn_mc['20']['std']:.3f}")
    print("  Paper claims: FB15k-237 .449, WN18RR .381 (Temporal task)")

    if fb_mc['20']['mean'] > 0.5:
        issues.append(f"MC Dropout FB15k-237: Data shows {fb_mc['20']['mean']:.3f}, paper says .449")

# UKGE baseline
ukge_data = load_json("ukge_baseline_results.json")
if ukge_data:
    print(f"\nUKGE (novel-context AUROC):")
    fb_ukge = ukge_data["fb15k237"]["summary"]
    print(f"  FB15k-237 UKGE-Logi: {fb_ukge['UKGE_Logi']['novel_ctx_auroc_mean']:.3f}")
    print(f"  FB15k-237 UKGE-Rect: {fb_ukge['UKGE_Rect']['novel_ctx_auroc_mean']:.3f}")
    print(f"  FB15k-237 UKGE-PSL:  {fb_ukge['UKGE_PSL']['novel_ctx_auroc_mean']:.3f}")
    print("  Paper Appendix claims: UKGE-Logi .50, UKGE-Rect .33, UKGE-PSL .60")

# Canonical results
if canonical:
    print(f"\nCanonical temporal AUROC (overall):")
    fb_sum = canonical["fb15k237"]["summary"]
    wn_sum = canonical["wn18rr"]["summary"]

    print(f"  FB15k-237 Energy:   {fb_sum['Energy']['temporal_auroc_mean']:.3f}")
    print(f"  FB15k-237 Coverage: {fb_sum['CoverageOnly']['temporal_auroc_mean']:.3f}")
    print(f"  WN18RR Energy:      {wn_sum['Energy']['temporal_auroc_mean']:.3f}")
    print(f"  WN18RR Coverage:    {wn_sum['CoverageOnly']['temporal_auroc_mean']:.3f}")

    print("\nPaper Table 3 claims (Temporal task):")
    print("  FB15k-237: Energy .589, Coverage .664")
    print("  WN18RR:    Energy .851, Coverage .724")

    # Check discrepancies
    if abs(fb_sum['Energy']['temporal_auroc_mean'] - 0.589) > 0.05:
        issues.append(f"Energy FB15k-237: Data={fb_sum['Energy']['temporal_auroc_mean']:.3f}, Paper=.589")
    if abs(wn_sum['Energy']['temporal_auroc_mean'] - 0.851) > 0.05:
        issues.append(f"Energy WN18RR: Data={wn_sum['Energy']['temporal_auroc_mean']:.3f}, Paper=.851")

# =============================================================================
# Table 4: Coverage Paradox
# =============================================================================
print("\n--- TABLE 4: Coverage Paradox ---")
print("Paper claims for FB15k-237:")
print("  Full coverage:  14,010 samples, 32.3% Hits@10")
print("  Partial zero:   6,131 samples, 59.5% Hits@10")
print("  Full zero:      325 samples, 14.8% Hits@10")
print("\n[MANUAL CHECK NEEDED] Verify from analyze_partial_coverage.py output")

# =============================================================================
# Appendix Table: MC Dropout pass ablation
# =============================================================================
print("\n--- APPENDIX: MC Dropout Pass Ablation ---")
if mc_data:
    print("Paper claims (mean ± std over 3 seeds):")
    print("  FB15k-237: .586±.014 (10p), .587±.013 (20p), .586±.014 (30p), .586±.014 (50p)")
    print("  WN18RR:    .455±.054 (10p), .447±.066 (20p), .447±.073 (30p), .443±.064 (50p)")

    print("\nActual data:")
    for passes in [10, 20, 30, 50]:
        fb = mc_data["datasets"]["fb15k237"]["summary"][str(passes)]
        wn = mc_data["datasets"]["wn18rr"]["summary"][str(passes)]
        print(f"  {passes} passes: FB15k-237 {fb['mean']:.3f}±{fb['std']:.3f}, WN18RR {wn['mean']:.3f}±{wn['std']:.3f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("ISSUES FOUND")
print("="*80)
if issues:
    for issue in issues:
        print(f"  [!] {issue}")
else:
    print("  No major discrepancies detected in automated checks.")

print("\n[MANUAL VERIFICATION NEEDED]:")
print("  1. Table 1 error rates by coverage level")
print("  2. Table 2 coverage distribution percentages")
print("  3. Table 4 coverage paradox numbers")
print("  4. Line 66 says '3 datasets, 3 architectures' but Abstract says '4 datasets, 4 architectures'")
