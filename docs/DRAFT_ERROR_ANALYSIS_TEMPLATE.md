# Draft: Error Analysis Template

**Purpose**: Pre-written text with placeholders for actual experimental results
**Location**: Main text (experiments section) + Appendix table
**Fill in**: When error analysis experiment completes (~30 min remaining)

---

## PART 1: Main Text Addition

**Location**: `paper/sections/experiments_uai.tex`, after binary coverage discussion (line ~136)

### Template (Fill in [PLACEHOLDERS]):

```latex
\paragraph{Error analysis.}
To understand CAGP's failure modes, we analyzed [N_TOTAL] predictions on
FB15k-237 standard OOD (random tail corruption).
The model achieves [AUROC] AUROC with [ACCURACY]% accuracy at threshold
$\tau = [THRESHOLD]$.
False positives ([FP_COUNT], [FP_RATE]% of ID) primarily occur on low-degree
entities ([FP_LOW_DEGREE]% have degree below 25th percentile), where GP
variance is high despite coverage being observed.
False negatives ([FN_COUNT], [FN_RATE]% of OOD) occur when random corruptions
coincidentally involve observed (h,r) pairs ([FN_HAS_COVERAGE]% have coverage=1).
Component attribution shows [FP_GP_ONLY]% of false positives driven by high
GP variance alone, while [FN_LOW_COV]% of false negatives have low coverage
uncertainty.
This validates the complementarity thesis: both signals contribute distinct
information.
See Appendix~\ref{app:error_analysis} for detailed breakdown.
```

---

### Alternative: Concise Version (if space is tight):

```latex
\paragraph{Error analysis.}
CAGP achieves [AUROC] AUROC on FB15k-237 standard OOD with [FP_RATE]% false
positive rate (primarily low-degree entities) and [FN_RATE]% false negative
rate (corruptions with coincidental coverage).
Component attribution shows [FP_GP_ONLY]% of errors driven by GP variance
only, validating complementarity (Appendix~\ref{app:error_analysis}).
```

---

## PART 2: Appendix Table

**Location**: `paper/main_uai.tex`, add new appendix section

### Full Template:

```latex
\section{Error Analysis}
\label{app:error_analysis}

We analyzed CAGP's failure modes on FB15k-237 standard OOD detection
(random tail corruption, [N_ID] ID + [N_OOD] OOD samples).

\begin{table}[h]
\centering
\caption{Error analysis on FB15k-237 standard OOD. FP = false positive
(ID flagged as OOD), FN = false negative (OOD missed).}
\label{tab:error_analysis}
\small
\begin{tabular}{lrr}
\toprule
Metric & Value & Rate \\
\midrule
\multicolumn{3}{l}{\textit{Overall Performance}} \\
AUROC & [AUROC] & --- \\
Accuracy & [ACCURACY] & --- \\
Precision & [PRECISION] & --- \\
Recall & [RECALL] & --- \\
F1 Score & [F1] & --- \\
\midrule
\multicolumn{3}{l}{\textit{Confusion Matrix}} \\
True Negatives & [TN] & --- \\
False Positives & [FP] & [FP_RATE]\% \\
False Negatives & [FN] & [FN_RATE]\% \\
True Positives & [TP] & --- \\
\midrule
\multicolumn{3}{l}{\textit{False Positive Patterns}} \\
Low-degree entities & [FP_LOW_DEG] & [FP_LOW_DEG_PCT]\% \\
High GP variance only & [FP_GP_ONLY] & [FP_GP_ONLY_PCT]\% \\
High coverage only & [FP_COV_ONLY] & [FP_COV_ONLY_PCT]\% \\
Both high & [FP_BOTH] & [FP_BOTH_PCT]\% \\
\midrule
\multicolumn{3}{l}{\textit{False Negative Patterns}} \\
Has coverage (c=1) & [FN_HAS_COV] & [FN_HAS_COV_PCT]\% \\
Low GP variance only & [FN_GP_ONLY] & [FN_GP_ONLY_PCT]\% \\
Low coverage only & [FN_COV_ONLY] & [FN_COV_ONLY_PCT]\% \\
Both low & [FN_BOTH] & [FN_BOTH_PCT]\% \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key findings.}
\textbf{False positives} ([FP_RATE]\%) occur primarily on low-degree
ID entities where GP variance is elevated despite observed coverage.
[FP_LOW_DEG_PCT]\% of FPs involve entities below the 25th percentile
of training degree distribution.
Component analysis shows [FP_GP_ONLY_PCT]\% driven by GP variance alone,
[FP_COV_ONLY_PCT]\% by coverage alone, and [FP_BOTH_PCT]\% by both.

\textbf{False negatives} ([FN_RATE]\%) occur when random corruptions
coincidentally create observed (h,r) pairs.
[FN_HAS_COV_PCT]\% of missed OOD samples have coverage=1, indicating
the corrupted tail was previously observed with the head's relation.
[FN_LOW_COV_PCT]\% have low coverage uncertainty, confirming this pattern.

These patterns validate Theorem~\ref{thm:complementarity}: semantic
uncertainty alone misses low-degree ID (high variance); structural
uncertainty alone misses coincidental coverage (c=1).
The combination reduces both error types.
```

---

## PART 3: Python Script to Auto-Generate Text

Save this for when results are ready:

```python
def generate_error_analysis_text(results):
    """
    Generate LaTeX text from error analysis results.

    Args:
        results: dict from run_error_analysis_standard_ood.py
    """
    cm = results['confusion_matrix']
    metrics = results['metrics']

    # Extract values
    tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
    auroc = metrics['auroc']
    accuracy = metrics['accuracy'] * 100
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    fp_rate = results['fp_rate'] * 100
    fn_rate = results['fn_rate'] * 100
    threshold = results['threshold']

    # Main text paragraph
    main_text = f"""\\paragraph{{Error analysis.}}
CAGP achieves {auroc:.3f} AUROC on FB15k-237 standard OOD with {fp_rate:.1f}\\% false
positive rate (primarily low-degree entities) and {fn_rate:.1f}\\% false negative
rate (corruptions with coincidental coverage).
See Appendix~\\ref{{app:error_analysis}} for detailed breakdown."""

    # Appendix table (simplified - fill in pattern details manually)
    table_text = f"""\\begin{{table}}[h]
\\centering
\\caption{{Error analysis on FB15k-237 standard OOD.}}
\\label{{tab:error_analysis}}
\\small
\\begin{{tabular}}{{lrr}}
\\toprule
Metric & Value & Rate \\\\
\\midrule
AUROC & {auroc:.3f} & --- \\\\
Accuracy & {accuracy:.1f}\\% & --- \\\\
Precision & {precision:.3f} & --- \\\\
Recall & {recall:.3f} & --- \\\\
F1 Score & {f1:.3f} & --- \\\\
\\midrule
True Negatives & {tn:,} & --- \\\\
False Positives & {fp:,} & {fp_rate:.1f}\\% \\\\
False Negatives & {fn:,} & {fn_rate:.1f}\\% \\\\
True Positives & {tp:,} & --- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    return main_text, table_text

# Usage (when results available):
# with open('outputs/error_analysis_standard_ood.json') as f:
#     results = json.load(f)
# main, table = generate_error_analysis_text(results)
# print(main)
# print(table)
```

---

## PART 4: Expected Values (for planning)

Based on paper's reported ~0.96 AUROC on standard OOD:

**Likely results**:
- AUROC: 0.95-0.97
- Accuracy: 90-95%
- FP rate: 5-10% (ID flagged as OOD)
- FN rate: 5-10% (OOD missed)
- FP pattern: Low-degree entities (60-70%)
- FN pattern: Coincidental coverage (50-60%)

**If AUROC < 0.90**:
- Check if experiment used proper OOD split
- May need to retrain with more epochs
- Coverage matrix may not be initialized correctly

**If AUROC > 0.97**:
- Excellent! Close to perfect detection
- FP/FN will be minimal
- Still report patterns for completeness

---

## PART 5: Minimal Version (If Results Are Weak)

If experiment gives poor results (AUROC < 0.85):

```latex
\paragraph{Error analysis.}
We analyzed failure modes on FB15k-237 standard OOD.
While CAGP achieves [AUROC] AUROC, error analysis reveals
[KEY_PATTERN] (Appendix~\ref{app:error_analysis}).
This suggests [INSIGHT] as a direction for future improvement.
```

**Strategy**: Acknowledge limitation honestly, focus on insights

---

## PART 6: Integration Checklist

When experiment completes:

- [ ] Load results from `outputs/error_analysis_standard_ood.json`
- [ ] Fill in all [PLACEHOLDERS] in main text
- [ ] Fill in all [PLACEHOLDERS] in appendix table
- [ ] Add pattern analysis from experiment console output
- [ ] Verify numbers add up (TP+TN+FP+FN = total)
- [ ] Add to `paper/sections/experiments_uai.tex` after line 136
- [ ] Add appendix section to `paper/main_uai.tex`
- [ ] Update label references
- [ ] Compile and check
- [ ] Verify fits in page limit

---

## PART 7: Quick-Fill Template (Copy-Paste Ready)

**When results available, fill this in first:**

```
N_TOTAL = [20000 or 40000]
N_ID = [10000 or 20000]
N_OOD = [10000 or 20000]
AUROC = [0.XXX]
ACCURACY = [XX.X]
PRECISION = [0.XXX]
RECALL = [0.XXX]
F1 = [0.XXX]
THRESHOLD = [0.XXX]

TN = [XXXXX]
FP = [XXXXX]
FN = [XXXXX]
TP = [XXXXX]
FP_RATE = [FP/(FP+TN)*100]
FN_RATE = [FN/(FN+TP)*100]

FP_LOW_DEGREE = [from console output]
FP_LOW_DEG_PCT = [%]
FP_GP_ONLY = [from console]
FP_GP_ONLY_PCT = [%]
FP_COV_ONLY = [from console]
FP_COV_ONLY_PCT = [%]
FP_BOTH = [from console]
FP_BOTH_PCT = [%]

FN_HAS_COV = [from console]
FN_HAS_COV_PCT = [%]
FN_GP_ONLY = [from console]
FN_GP_ONLY_PCT = [%]
FN_COV_ONLY = [from console]
FN_COV_ONLY_PCT = [%]
FN_BOTH = [from console]
FN_BOTH_PCT = [%]
```

Then search-replace in template files.

---

## Current Experiment Status

Check status:
```bash
ps aux | grep run_error_analysis_standard_ood
tail -100 /tmp/claude/-Users-i767700-Github-kg-bayesian-prior/tasks/b1cdc42.output
```

Expected completion: ~30 minutes from now

---

**Status**: Template ready
**Next**: Wait for experiment, then fill in placeholders
**Time needed**: 15 minutes to fill + integrate after results available
