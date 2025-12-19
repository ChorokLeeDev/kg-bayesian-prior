#!/usr/bin/env python3
"""
Quick script to parse experiment results and generate analysis.
Run after pasting results into results_template.md
"""

import re
import json
from pathlib import Path

# Expected format from Colab output:
# Model Results:
# Link Prediction: MRR=0.0832, H@1=0.0255, H@3=0.0720, H@10=0.2015
# Calibration:     ECE=0.3165, Brier=0.2875
# OOD Detection:   AUROC=0.1038

def parse_results(text):
    """Parse results from Colab output."""
    results = {}

    # Pattern to match model results
    model_pattern = r"(\w+(?:\+\w+)?(?:\s*\([^)]+\))?)\s+Results:"
    metrics_pattern = r"MRR=([0-9.]+).*H@1=([0-9.]+).*H@3=([0-9.]+).*H@10=([0-9.]+)"
    calibration_pattern = r"ECE=([0-9.]+).*Brier=([0-9.]+)"
    ood_pattern = r"AUROC=([0-9.]+)"

    current_model = None

    for line in text.split('\n'):
        # Check for model name
        model_match = re.search(model_pattern, line)
        if model_match:
            current_model = model_match.group(1).strip()
            results[current_model] = {}
            continue

        if current_model:
            # Check for metrics
            metrics_match = re.search(metrics_pattern, line)
            if metrics_match:
                results[current_model]['mrr'] = float(metrics_match.group(1))
                results[current_model]['hits@1'] = float(metrics_match.group(2))
                results[current_model]['hits@3'] = float(metrics_match.group(3))
                results[current_model]['hits@10'] = float(metrics_match.group(4))

            calibration_match = re.search(calibration_pattern, line)
            if calibration_match:
                results[current_model]['ece'] = float(calibration_match.group(1))
                results[current_model]['brier'] = float(calibration_match.group(2))

            ood_match = re.search(ood_pattern, line)
            if ood_match:
                results[current_model]['auroc'] = float(ood_match.group(1))

    return results


def generate_markdown_table(results):
    """Generate markdown table from results."""
    header = "| Model | MRR | H@1 | H@3 | H@10 | ECE | Brier | AUROC |"
    separator = "|-------|-----|-----|-----|------|-----|-------|-------|"

    rows = [header, separator]

    for model, metrics in results.items():
        row = f"| {model} | {metrics.get('mrr', '?'):.4f} | {metrics.get('hits@1', '?'):.4f} | " \
              f"{metrics.get('hits@3', '?'):.4f} | {metrics.get('hits@10', '?'):.4f} | " \
              f"{metrics.get('ece', '?'):.4f} | {metrics.get('brier', '?'):.4f} | " \
              f"{metrics.get('auroc', '?'):.4f} |"
        rows.append(row)

    return '\n'.join(rows)


def analyze_results(results):
    """Generate analysis of results."""
    analysis = []

    # Find best calibration
    eces = {m: r['ece'] for m, r in results.items() if 'ece' in r}
    if eces:
        best_ece_model = min(eces, key=eces.get)
        analysis.append(f"Best Calibration (ECE): {best_ece_model} ({eces[best_ece_model]:.4f})")

    # Compare GGPN vs GP-KGE
    ggpn_ece = None
    gpkge_ece = None

    for model, metrics in results.items():
        if 'GGPN' in model:
            ggpn_ece = metrics.get('ece')
        if 'GP-KGE' in model or 'Ours' in model:
            gpkge_ece = metrics.get('ece')

    if ggpn_ece and gpkge_ece:
        improvement = (ggpn_ece - gpkge_ece) / ggpn_ece * 100
        analysis.append(f"\nCALIBRATION IMPROVEMENT:")
        analysis.append(f"  GGPN ECE:   {ggpn_ece:.4f}")
        analysis.append(f"  GP-KGE ECE: {gpkge_ece:.4f}")
        analysis.append(f"  Improvement: {improvement:.1f}%")

    # Find best MRR
    mrrs = {m: r['mrr'] for m, r in results.items() if 'mrr' in r}
    if mrrs:
        best_mrr_model = max(mrrs, key=mrrs.get)
        analysis.append(f"\nBest Link Prediction (MRR): {best_mrr_model} ({mrrs[best_mrr_model]:.4f})")

    # Find best AUROC
    aurocs = {m: r['auroc'] for m, r in results.items() if 'auroc' in r}
    if aurocs:
        best_auroc_model = max(aurocs, key=aurocs.get)
        analysis.append(f"Best OOD Detection (AUROC): {best_auroc_model} ({aurocs[best_auroc_model]:.4f})")

    return '\n'.join(analysis)


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    latex = r"""\begin{table}[h]
\centering
\caption{Experimental results on FB15k-237.}
\begin{tabular}{lccccccc}
\toprule
Model & MRR $\uparrow$ & H@1 $\uparrow$ & H@10 $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$ \\
\midrule
"""

    for model, metrics in results.items():
        row = f"{model} & {metrics.get('mrr', 0):.4f} & {metrics.get('hits@1', 0):.4f} & " \
              f"{metrics.get('hits@10', 0):.4f} & {metrics.get('ece', 0):.4f} & " \
              f"{metrics.get('brier', 0):.4f} & {metrics.get('auroc', 0):.4f} \\\\\n"
        latex += row

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    return latex


if __name__ == "__main__":
    # Example usage
    sample_output = """
    DistMult Results:
    Link Prediction: MRR=0.0832, H@1=0.0255, H@3=0.0720, H@10=0.2015
    Calibration:     ECE=0.3165, Brier=0.2875
    OOD Detection:   AUROC=0.1038
    """

    results = parse_results(sample_output)
    print("Parsed Results:")
    print(json.dumps(results, indent=2))
    print("\nMarkdown Table:")
    print(generate_markdown_table(results))
    print("\nAnalysis:")
    print(analyze_results(results))
