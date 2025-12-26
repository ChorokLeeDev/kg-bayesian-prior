"""
Analyze SOTA Base Model Results

Generates:
1. Performance comparison table
2. Improvement analysis
3. Paper-ready LaTeX tables
4. Insights and recommendations
"""

import json
from pathlib import Path


def analyze_results(results_file):
    """Analyze SOTA base model comparison results."""

    with open(results_file) as f:
        results = json.load(f)

    print("="*80)
    print("SOTA BASE MODELS - ANALYSIS")
    print("="*80)

    # Performance table
    print("\n## Performance Comparison\n")
    print("Model        AUROC   AUPR    Separation  Learned α   GP Sep   Cov Sep")
    print("-" * 75)

    for r in results:
        gp_sep = r['ood_gp_mean'] - r['id_gp_mean']
        cov_sep = r['ood_coverage_mean'] - r['id_coverage_mean']

        print(f"{r['scoring_fn']:12s} {r['auroc']:.4f}  {r['aupr']:.4f}  "
              f"{r['separation']:10.3f}  {r['learned_alpha']:9.3f}  "
              f"{gp_sep:7.3f}  {cov_sep:7.3f}")

    # Best model
    print("\n## Best Performing Model\n")
    best = max(results, key=lambda x: x['auroc'])
    print(f"**{best['scoring_fn'].upper()}**: {best['auroc']:.4f} AUROC")

    # Relative improvements
    print("\n## Relative Improvements\n")
    distmult = next(r for r in results if r['scoring_fn'] == 'distmult')

    for r in results:
        if r['scoring_fn'] != 'distmult':
            improvement = (r['auroc'] - distmult['auroc']) / distmult['auroc'] * 100
            print(f"{r['scoring_fn']:10s}: {improvement:+.1f}% vs DistMult")

    # Component analysis
    print("\n## Component Analysis (Which Signal Dominates?)\n")

    for r in results:
        gp_sep = r['ood_gp_mean'] - r['id_gp_mean']
        cov_sep = r['ood_coverage_mean'] - r['id_coverage_mean']

        # Normalize to see relative contribution
        total_sep = r['separation']
        if total_sep > 0:
            gp_contrib = (gp_sep * r['learned_alpha']) / total_sep * 100
            cov_contrib = (cov_sep * (1 - r['learned_alpha'])) / total_sep * 100
        else:
            gp_contrib = cov_contrib = 0

        print(f"{r['scoring_fn']:10s}: GP={gp_contrib:5.1f}%, Coverage={cov_contrib:5.1f}%, α={r['learned_alpha']:.3f}")

    # Generate LaTeX table
    print("\n## LaTeX Table (for Paper)\n")
    print(generate_latex_table(results))

    # Insights
    print("\n## Key Insights\n")
    generate_insights(results)

    return results


def generate_latex_table(results):
    """Generate LaTeX table for paper."""

    latex = r"""\begin{table}[h]
\centering
\caption{CAGP Generalization Across Base Models (FB15k-237 Temporal OOD)}
\label{tab:sota_base_models}
\begin{tabular}{lccc}
\toprule
Base Model & AUROC & AUPR & Learned $\alpha$ \\
\midrule
"""

    for r in results:
        model_name = r['scoring_fn'].capitalize()
        latex += f"{model_name:12s} & {r['auroc']:.3f} & {r['aupr']:.3f} & {r['learned_alpha']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    return latex


def generate_insights(results):
    """Generate insights from results."""

    # Check if all models benefit
    all_strong = all(r['auroc'] > 0.85 for r in results)

    if all_strong:
        print("✅ **All base models achieve strong OOD detection (>0.85 AUROC)**")
        print("   → CAGP's uncertainty decomposition is architecture-agnostic")

    # Check learned α variation
    alphas = [r['learned_alpha'] for r in results]
    alpha_std = (max(alphas) - min(alphas))

    if alpha_std < 0.1:
        print(f"\n✅ **Learned α is consistent across models** (range: {min(alphas):.3f}-{max(alphas):.3f})")
        print("   → Decomposition strategy is stable across architectures")
    else:
        print(f"\n⚠️ **Learned α varies across models** (range: {min(alphas):.3f}-{max(alphas):.3f})")
        print("   → Different architectures learn different GP variance quality")

    # Check coverage dominance
    for r in results:
        cov_sep = r['ood_coverage_mean'] - r['id_coverage_mean']
        gp_sep = r['ood_gp_mean'] - r['id_gp_mean']

        if cov_sep > gp_sep * 2:
            print(f"\n📊 **{r['scoring_fn'].upper()}**: Coverage dominates (sep={cov_sep:.3f} vs GP={gp_sep:.3f})")

    # Comparison to paper baseline
    distmult = next(r for r in results if r['scoring_fn'] == 'distmult')

    print(f"\n## Paper Integration Recommendations\n")
    print(f"1. Add table to Appendix B showing architecture-agnostic improvement")
    print(f"2. Baseline (DistMult): {distmult['auroc']:.3f} AUROC")
    print(f"3. All models achieve {min(r['auroc'] for r in results):.3f}+ AUROC")
    print(f"4. This addresses \"weak baseline\" concern directly")


if __name__ == "__main__":
    results_file = Path(__file__).parent.parent / "outputs" / "sota_base_models.json"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run scripts/run_sota_base_models.py first")
        exit(1)

    analyze_results(results_file)
