# Error Analysis - Paper Integration Plan

**Goal**: Address Reviewer Concern #6 - "No systematic error analysis"

**Status**: Experiment running (~15 min)

## What We're Analyzing

Systematic failure mode identification:

### 1. False Positives (ID → OOD)
- ID triples incorrectly flagged as OOD
- **Causes**: High GP variance on rare entities? Incomplete coverage?
- **Pattern**: Low-degree entities? Rare relations?

### 2. False Negatives (OOD → ID)
- OOD triples that slipped through
- **Causes**: Entity seen in training? Similar relation?
- **Pattern**: Common entities with novel contexts?

### 3. Component Attribution
- Which component fails: Coverage, GP variance, or both?
- When does coverage dominate? When does variance help?

### 4. Entity/Relation Patterns
- Entity degree correlation with failures
- Relation frequency impact
- Edge cases (emerging entities, rare relations)

## Expected Findings

Based on theory and continuous coverage results:

### Scenario A: Binary Coverage is Perfect (1.0 AUROC)
**If we see this:**
- Zero false positives (all ID correctly identified)
- Zero false negatives (all OOD correctly detected)

**Insight for paper:**
> Perfect detection on temporal OOD validates Theorem 1. Novel contexts
> are precisely characterized by zero coverage, enabling perfect separation.

**Discussion point:**
- Limitation: Only works when OOD = novel contexts
- Random corruption OOD may have different characteristics

### Scenario B: Some Errors Exist (~0.85-0.95 AUROC)
**If we see this:**
- Small number of failures (~5-15%)

**Likely patterns:**
1. **FP: Rare ID entities**
   - Entities with very few training observations
   - High GP variance despite being "in-distribution"
   - Coverage = 1 but variance is high

2. **FN: Common OOD entities**
   - Entities frequently seen in training
   - Novel relation context but low GP variance
   - Coverage = 0 but model is confident

**Insights for paper:**
- CAGP excels when coverage aligns with distribution shift
- Struggles when entity frequency ≠ context novelty
- Trade-off controlled by learned α

### Scenario C: Specific Failure Modes

**Pattern 1: Low-degree entities (degree < 10)**
- If FP concentrated here → GP variance too conservative
- Solution: Degree-aware thresholding

**Pattern 2: Rare relations (frequency < 100)**
- If FN concentrated here → Coverage insufficient
- Solution: Relation-specific thresholds

**Pattern 3: Emerging entities (first appeared late in training)**
- Edge case: Partial coverage, moderate variance
- Interesting for discussion

## Paper Integration Points

### 1. Section 5 (Experiments) - Add Subsection

After Section 5.4, add:

```latex
\subsection{Error Analysis}

To identify systematic failure modes, we analyzed CAGP's predictions on
FB15k-237 temporal OOD at a balanced threshold (0.XX).

\textbf{Results.} [One of three scenarios]

SCENARIO A (Perfect):
CAGP achieves perfect classification with zero false positives and zero
false negatives. This validates Theorem~1: novel contexts are precisely
characterized by zero coverage, enabling perfect separation.

SCENARIO B (Near-perfect with patterns):
CAGP achieves XX.X\% accuracy with balanced error rates (FP: XX\%, FN: XX\%).
False positives concentrate on low-degree entities (mean degree: XX vs XX
for correct predictions), indicating high GP variance on rare entities.
False negatives occur when [pattern from analysis].

SCENARIO C (Trade-offs):
We observe a trade-off between false positives and false negatives controlled
by threshold selection. At the median uncertainty threshold (0.XX), CAGP
achieves XX\% precision and XX\% recall. Component analysis reveals that
[XX\%] of errors stem from [coverage/variance], suggesting [insight].

\textbf{Implications.} [Based on findings]
- CAGP excels when distribution shift aligns with novel contexts (temporal)
- Performance degrades when [failure pattern]
- Future work: [mitigation strategy]
```

### 2. Discussion Section - Add Limitations Paragraph

```latex
\paragraph{Limitations.}
Error analysis reveals that CAGP's performance depends on the alignment
between distribution shift and novel contexts. On temporal OOD where entities
encounter new relation types, CAGP achieves [XX.X\%] accuracy. However,
[failure pattern]. This suggests that CAGP is best suited for scenarios
where OOD is characterized by novel contexts rather than [alternative].

Additionally, [any systematic biases from analysis]. Future work could
address this by [proposed solution].
```

### 3. Appendix - Detailed Error Analysis

```latex
\subsection{Detailed Error Analysis}

We provide a comprehensive breakdown of CAGP's prediction errors on FB15k-237
temporal OOD (Table~\ref{tab:error_analysis}).

\begin{table}[h]
\centering
\caption{Error Analysis on FB15k-237 Temporal OOD}
\label{tab:error_analysis}
\begin{tabular}{lccc}
\toprule
Error Type & Count & \% of Total & Main Pattern \\
\midrule
True Negative  & XXXX & XX.X\% & - \\
False Positive & XXX  & X.X\%  & [Pattern] \\
False Negative & XXX  & X.X\%  & [Pattern] \\
True Positive  & XXXX & XX.X\% & - \\
\midrule
Accuracy       & -    & XX.X\% & - \\
\bottomrule
\end{tabular}
\end{table}

\textbf{False Positive Analysis.}
[Detailed findings about FP cases]

\textbf{False Negative Analysis.}
[Detailed findings about FN cases]

\textbf{Component Attribution.}
[Which component fails in which cases]
```

## Reviewer Response

**Concern #6**: "No systematic error analysis - what causes failures?"

**Response** (will be customized based on results):

### If Perfect (1.0 AUROC at threshold):
> We conducted comprehensive error analysis (Section 5.5, Appendix B.3).
> On temporal OOD, CAGP achieves perfect classification (zero errors at
> median threshold). This occurs because novel contexts are precisely
> characterized by zero coverage, validating Theorem 1.
>
> We acknowledge this result is specific to temporal distribution shift.
> On random corruption OOD (Table 3), score-based methods excel (0.99 AUROC)
> because they detect implausibility, while CAGP achieves 0.87-0.97 by
> detecting novel patterns.

### If Near-Perfect with Patterns:
> We conducted systematic error analysis (Section 5.5). CAGP achieves XX.X%
> accuracy on temporal OOD. Errors exhibit clear patterns:
>
> **False Positives (X.X%)**: Concentrate on low-degree entities (mean
> degree XX vs XX), where GP variance is high despite being in-distribution.
> This suggests variance-based uncertainty is conservative for rare entities.
>
> **False Negatives (X.X%)**: Occur when [pattern]. This indicates [insight].
>
> These patterns suggest future improvements: [specific mitigation strategy].

### If Trade-offs Exist:
> We conducted error analysis across multiple thresholds (Appendix B.3).
> CAGP exhibits precision-recall trade-offs typical of uncertainty-based
> detection. At balanced threshold (0.XX), we achieve XX% precision and XX%
> recall. Errors stem primarily from [dominant component failure], suggesting
> that [insight for improvement].

## Acceptance Impact

**Before**: "No error analysis" → Unknown failure modes → Weakness

**After**: "Systematic error analysis" → **Understood limitations** → Strength

**Value**:
- Demonstrates thoroughness (reviewers love this!)
- Shows honest assessment of limitations
- Identifies future work directions
- Pre-empts "but what about X?" questions

**Estimated Impact**: 75% → 80% (+5%)

## Analysis Outputs

The script generates:

1. **Confusion Matrix**
   - TP, TN, FP, FN counts
   - Accuracy, Precision, Recall, F1

2. **Error Patterns**
   - Entity degree statistics for FP/FN
   - Relation frequency patterns
   - Component attribution (which fails)

3. **Insights**
   - Dominant failure mode
   - Systematic patterns
   - Recommendations

4. **JSON Results**
   - `outputs/error_analysis.json`
   - Ready for paper table generation

## Next Steps When Results Arrive

1. Run analysis script:
   ```bash
   python3 scripts/run_error_analysis.py
   ```

2. Review output and categorize findings:
   - Perfect / Near-perfect / Trade-offs?
   - Which scenario matches?

3. Generate paper text:
   - Use appropriate template above
   - Fill in numbers from analysis
   - Add to Section 5.5

4. Add table to Appendix B

5. Update discussion section with limitations

## Timeline

1. ⏳ Error analysis running (~15 min)
2. ⏳ Review results (5 min)
3. ⏳ Write paper text (15 min)
4. ⏳ Add to paper and compile (5 min)

**Total**: ~40 minutes from start to paper integration

## Success Criteria

✅ Clear understanding of when CAGP fails
✅ Systematic patterns identified (not random failures)
✅ Component attribution (coverage vs variance)
✅ Honest assessment in paper
✅ Addresses reviewer concern #6 directly

**Result**: Shows thoroughness and maturity of analysis
