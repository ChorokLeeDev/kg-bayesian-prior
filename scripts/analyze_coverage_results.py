"""
Analyze Continuous Coverage Results

Reads the JSON output from continuous coverage experiments and generates:
1. Comparison tables
2. Recommendation based on results
3. Draft text for paper integration
"""

import json
from pathlib import Path
import sys

def load_results(quick=True):
    """Load results from JSON file."""
    output_dir = Path(__file__).parent.parent / 'outputs'

    if quick:
        result_file = output_dir / 'continuous_coverage_quick.json'
    else:
        result_file = output_dir / 'continuous_coverage_ablation.json'

    if not result_file.exists():
        print(f"Error: Results file not found: {result_file}")
        print("Run the experiment first:")
        if quick:
            print("  python scripts/run_continuous_coverage_quick.py")
        else:
            print("  python scripts/run_continuous_coverage_ablation.py")
        return None

    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_quick_results(results):
    """Analyze quick test results."""
    print("="*70)
    print("CONTINUOUS COVERAGE QUICK TEST ANALYSIS")
    print("="*70)

    # Find binary baseline
    baseline = next((r for r in results if r['coverage_mode'] == 'binary'), None)
    if not baseline:
        print("Error: Binary baseline not found in results")
        return

    # Find best performer
    best = max(results, key=lambda x: x['auroc'])

    # Calculate improvements
    print("\nPERFORMANCE COMPARISON")
    print("-"*70)
    print(f"{'Mode':<12} {'AUROC':<10} {'Δ vs Binary':<15} {'Alpha':<10}")
    print("-"*70)

    for r in sorted(results, key=lambda x: x['auroc'], reverse=True):
        delta = r['auroc'] - baseline['auroc']
        delta_pct = (delta / baseline['auroc']) * 100
        marker = " ⭐" if r == best else ""
        print(f"{r['coverage_mode']:<12} "
              f"{r['auroc']:<10.4f} "
              f"{delta:+.4f} ({delta_pct:+.1f}%) "
              f"{r['learned_alpha']:<10.3f}"
              f"{marker}")

    # Statistical significance assessment
    improvement = best['auroc'] - baseline['auroc']

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if improvement > 0.02:
        print(f"\n✓ SIGNIFICANT IMPROVEMENT: {improvement:+.4f} ({improvement/baseline['auroc']*100:+.1f}%)")
        print(f"\n  Best mode: {best['coverage_mode']}")
        print(f"  Baseline: {baseline['auroc']:.4f}")
        print(f"  Best: {best['auroc']:.4f}")
        print("\n  RECOMMENDATION: Switch to continuous coverage")
        print(f"  → Use {best['coverage_mode']}-scaled coverage in final model")
        print("  → Update Section 4.2 in paper")
        print("  → Extend Theorem 1 to continuous case")

    elif improvement > -0.02:
        print(f"\n✓ EQUIVALENT PERFORMANCE: {improvement:+.4f} ({improvement/baseline['auroc']*100:+.1f}%)")
        print("\n  All modes perform similarly (|Δ| < 2%)")
        print("\n  RECOMMENDATION: Keep binary coverage (Occam's Razor)")
        print("  → Binary is simpler and equally effective")
        print("  → Add this ablation to Appendix B")
        print("  → Demonstrates presence/absence is dominant signal")

    else:
        print(f"\n✓ BINARY OUTPERFORMS: {improvement:.4f} ({improvement/baseline['auroc']*100:.1f}%)")
        print("\n  Binary coverage is actually better")
        print("\n  RECOMMENDATION: Keep binary coverage")
        print("  → Investigate: continuous may overfit to training frequencies")
        print("  → Possible confound with sampling artifacts")

    # Generate paper text
    print("\n" + "="*70)
    print("DRAFT TEXT FOR PAPER")
    print("="*70)

    if abs(improvement) < 0.02:
        print("\n[Add to Section 5.4 or Appendix B]:")
        print("\n" + "-"*70)
        print("""
\\textbf{Continuous Coverage Ablation.} We evaluated continuous
coverage formulations—raw counts, log-scaled, TF-IDF—to test
whether co-occurrence frequency improves upon binary presence/absence.
Results show minimal difference (AUROC Δ <0.02, see Table~\\ref{tab:coverage_ablation}),
indicating that the discrete signal of \\emph{whether} an entity-relation
pair was observed dominates over the continuous signal of \\emph{how
frequently} it was observed. This aligns with Theorem~1: novel contexts
are characterized by zero coverage, making finer frequency distinctions
irrelevant. For emerging entities, GP variance already captures frequency
information through learned embeddings, rendering explicit frequency
counts redundant.
""")
        print("-"*70)

        print("\n[Table for Appendix B]:")
        print("\n" + "-"*70)
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Continuous vs. Binary Coverage Ablation}")
        print("\\label{tab:coverage_ablation}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Coverage Mode & AUROC & $\\Delta$ vs Binary \\\\")
        print("\\midrule")
        for r in results:
            delta = r['auroc'] - baseline['auroc']
            print(f"{r['coverage_mode'].capitalize():<15} & {r['auroc']:.4f} & {delta:+.4f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print("-"*70)


def analyze_full_results(results):
    """Analyze full ablation results."""
    print("="*70)
    print("CONTINUOUS COVERAGE FULL ABLATION ANALYSIS")
    print("="*70)

    # Group by dataset
    datasets = {}
    for r in results:
        dataset = r['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(r)

    # Analyze each dataset
    for dataset, dataset_results in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"{'='*70}")

        baseline = next((r for r in dataset_results if r['coverage_mode'] == 'binary'), None)
        if not baseline:
            continue

        print(f"\n{'Mode':<12} {'Temporal AUROC':<15} {'Random AUROC':<15} {'Δ vs Binary':<12}")
        print("-"*70)

        for r in sorted(dataset_results, key=lambda x: x['temporal_auroc'], reverse=True):
            delta_temporal = r['temporal_auroc'] - baseline['temporal_auroc']
            marker = " ⭐" if delta_temporal > 0.02 else ""
            print(f"{r['coverage_mode']:<12} "
                  f"{r['temporal_auroc']:<15.4f} "
                  f"{r['random_auroc']:<15.4f} "
                  f"{delta_temporal:+.4f}{marker}")


def main():
    """Main analysis function."""
    # Try quick results first
    print("Checking for quick test results...")
    results = load_results(quick=True)

    if results:
        analyze_quick_results(results)
        print("\n\n" + "="*70)
        print("To run full ablation:")
        print("  python scripts/run_continuous_coverage_ablation.py")
        print("="*70)
    else:
        # Try full results
        print("\nChecking for full ablation results...")
        results = load_results(quick=False)
        if results:
            analyze_full_results(results)
        else:
            print("\nNo results found. Run experiments first:")
            print("  python scripts/run_continuous_coverage_quick.py  # Quick (15-20 min)")
            print("  python scripts/run_continuous_coverage_ablation.py  # Full (4-5 hours)")
            sys.exit(1)


if __name__ == "__main__":
    main()
