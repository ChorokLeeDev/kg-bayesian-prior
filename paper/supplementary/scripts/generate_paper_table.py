#!/usr/bin/env python3
"""Generate updated results table for paper."""
import functools
print = functools.partial(print, flush=True)

# Results from experiments
RESULTS = {
    # Encyclopedic - PARADOX
    'FB15k-237': {'type': 'Encyclopedic', 'full': 32.3, 'partial': 59.5, 'zero': 14.8, 'paradox': True},
    'FB15k': {'type': 'Encyclopedic', 'full': 28.0, 'partial': 55.2, 'zero': 12.1, 'paradox': True},
    'CoDEx-S': {'type': 'Encyclopedic', 'full': 52.0, 'partial': 87.0, 'zero': 8.0, 'paradox': True},
    'CoDEx-M': {'type': 'Encyclopedic', 'full': 32.0, 'partial': 59.0, 'zero': 7.0, 'paradox': True},
    'CoDEx-L': {'type': 'Encyclopedic', 'full': 27.0, 'partial': 49.0, 'zero': 5.0, 'paradox': True},
    'YAGO3-10': {'type': 'Encyclopedic', 'full': 12.0, 'partial': 44.2, 'zero': 8.1, 'paradox': True},
    'DRKG': {'type': 'Biomedical', 'full': 0.7, 'partial': 1.5, 'zero': 0.0, 'paradox': True},

    # Hierarchical - NO PARADOX
    'WN18RR': {'type': 'Hierarchical', 'full': 45.0, 'partial': 7.0, 'zero': 0.0, 'paradox': False},
    'WN18': {'type': 'Hierarchical', 'full': 48.0, 'partial': 12.0, 'zero': 0.0, 'paradox': False},

    # Temporal - NO PARADOX
    'ICEWS14': {'type': 'Temporal', 'full': 59.0, 'partial': 18.0, 'zero': 16.0, 'paradox': False},
    'ICEWS18': {'type': 'Temporal', 'full': 42.0, 'partial': 15.0, 'zero': 8.0, 'paradox': False},
    'GDELT': {'type': 'Temporal', 'full': 35.0, 'partial': 10.0, 'zero': 5.0, 'paradox': False},
    'WIKI': {'type': 'Temporal', 'full': 36.7, 'partial': 0.6, 'zero': 0.0, 'paradox': False},
    'YAGO-temp': {'type': 'Temporal', 'full': 35.8, 'partial': 3.0, 'zero': 0.0, 'paradox': False},

    # Domain
    'BKG-BPMN': {'type': 'Domain', 'full': 85.0, 'partial': 45.0, 'zero': 12.0, 'paradox': False},
}

def generate_latex_table():
    """Generate LaTeX table for paper."""
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Coverage Paradox across 15 Knowledge Graphs. Hits@10 (\\%) by coverage type.}")
    print("\\label{tab:main-results}")
    print("\\small")
    print("\\begin{tabular}{llcccl}")
    print("\\toprule")
    print("Dataset & Type & Full & Partial & Zero & Paradox \\\\")
    print("\\midrule")

    # Group by type
    for kg_type in ['Encyclopedic', 'Biomedical', 'Hierarchical', 'Temporal', 'Domain']:
        datasets = [(k, v) for k, v in RESULTS.items() if v['type'] == kg_type]
        if not datasets:
            continue

        for i, (name, r) in enumerate(sorted(datasets)):
            paradox_str = "\\cmark" if r['paradox'] else "\\xmark"
            full_str = f"\\textbf{{{r['full']:.1f}}}" if not r['paradox'] else f"{r['full']:.1f}"
            partial_str = f"\\textbf{{{r['partial']:.1f}}}" if r['paradox'] else f"{r['partial']:.1f}"

            print(f"{name} & {r['type']} & {full_str} & {partial_str} & {r['zero']:.1f} & {paradox_str} \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def generate_summary():
    """Generate summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    paradox_count = sum(1 for r in RESULTS.values() if r['paradox'])
    total = len(RESULTS)

    print(f"\nTotal datasets: {total}")
    print(f"Paradox observed: {paradox_count} ({100*paradox_count/total:.0f}%)")
    print(f"No paradox: {total - paradox_count} ({100*(total-paradox_count)/total:.0f}%)")

    # By type
    print("\nBy KG Type:")
    for kg_type in ['Encyclopedic', 'Biomedical', 'Hierarchical', 'Temporal', 'Domain']:
        datasets = [r for r in RESULTS.values() if r['type'] == kg_type]
        if datasets:
            paradox = sum(1 for r in datasets if r['paradox'])
            print(f"  {kg_type}: {paradox}/{len(datasets)} show paradox")

    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    print("""
The Coverage Paradox is STRUCTURE-DEPENDENT:

1. Multi-relational KGs (Encyclopedic, Biomedical): 7/7 show paradox
   - Many relations per entity
   - Full coverage = model overfits to seen patterns
   - Partial coverage = forces generalization

2. Temporal/Hierarchical KGs: 0/7 show paradox
   - Few, repetitive relations
   - Coverage is a meaningful signal
   - Full coverage = genuine familiarity

PRACTICAL IMPLICATION:
- For Freebase-like KGs: DO NOT trust coverage as reliability signal
- For temporal/hierarchical KGs: Coverage CAN indicate reliability
""")

def main():
    print("="*60)
    print("COVERAGE PARADOX: MULTI-DATASET RESULTS")
    print("="*60)

    generate_latex_table()
    generate_summary()

if __name__ == '__main__':
    main()
