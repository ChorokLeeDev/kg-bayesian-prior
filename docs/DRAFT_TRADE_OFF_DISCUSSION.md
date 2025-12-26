# Draft: Trade-off Discussion Paragraph

**Location**: Section 5 (Experiments), after Table 3 (Standard OOD results)
**Purpose**: Provide practical guidance on when to use CAGP vs score-based methods
**Length**: 1 paragraph (~100 words)

---

## OPTION A: Concise (Recommended for space-constrained UAI)

```latex
\paragraph{When to use CAGP?}
Our results reveal a clear trade-off: CAGP excels on temporal distribution shift (0.89 AUROC on ICEWS14 vs 0.52--0.58 for score-based methods), while score-based methods excel on random corruptions (0.99 AUROC vs 0.96 for CAGP on FB15k-237).
This reflects fundamentally different failure modes: score-based methods detect implausible triples but miss structured distribution shift, while CAGP explicitly captures relational structure.
Practitioners should choose based on deployment scenario: use CAGP for evolving knowledge graphs with temporal drift or novel entity-relation patterns; use score-based methods for detecting adversarial corruptions in static graphs.
```

**Word count**: ~85 words
**Pros**: Concise, actionable, evidence-based
**Cons**: Could elaborate more on mechanisms

---

## OPTION B: Detailed (If space permits)

```latex
\paragraph{When to use CAGP?}
Our results reveal a fundamental trade-off between methods optimized for different OOD types.
CAGP achieves 0.89 AUROC on temporal distribution shift (ICEWS14) compared to 0.52--0.58 for score-based methods (Table~\ref{tab:icews}), reflecting its explicit modeling of structural uncertainty.
Conversely, score-based methods achieve 0.99 AUROC on random tail corruptions (FB15k-237 standard OOD), slightly outperforming CAGP at 0.96 (Table~\ref{tab:standard}).

This trade-off reflects fundamentally different mechanisms: score-based methods (UKGE, Energy) detect implausible triples via low model confidence but cannot distinguish temporal shift involving plausible but unobserved patterns.
CAGP's structural uncertainty explicitly tracks entity-relation co-occurrence, enabling detection of novel contexts even when entities are well-represented.

\textbf{Practical guidance:}
Deploy CAGP when distribution shift involves (1) emerging entities in novel relational contexts, or (2) temporal evolution with new entity-relation patterns (e.g., knowledge bases updated over time, streaming KGs).
Deploy score-based methods when the primary threat is adversarial random corruptions in static graphs.
For comprehensive coverage, ensemble both approaches (see Appendix~\ref{app:ensemble}).
```

**Word count**: ~175 words
**Pros**: Thorough, mechanistic explanation, actionable
**Cons**: Takes more space

---

## OPTION C: Hybrid (Balanced)

```latex
\paragraph{When to use CAGP?}
Our results reveal complementary strengths: CAGP achieves 0.89 AUROC on temporal OOD (ICEWS14) vs 0.52--0.58 for score-based methods, while score-based methods achieve 0.99 on random corruptions vs 0.96 for CAGP (Tables~\ref{tab:icews}--\ref{tab:standard}).
This reflects different mechanisms: score-based methods detect implausibility via low confidence but miss structured shift; CAGP tracks entity-relation co-occurrence, detecting novel contexts.

Practitioners should choose based on deployment: use CAGP for evolving KGs with temporal drift or novel entity-relation patterns; use score-based for detecting random corruptions in static graphs.
For comprehensive coverage, ensemble both (Appendix~\ref{app:ensemble}).
```

**Word count**: ~100 words
**Pros**: Balanced - mechanistic insight + practical guidance
**Cons**: None - **RECOMMENDED**

---

## Integration Instructions

**Step 1**: Choose option (recommend **Option C**)

**Step 2**: Insert into `paper/sections/experiments_uai.tex` after line ~98 (after Table 3 discussion)

**Current context** (line ~98):
```latex
Table~\ref{tab:standard} shows standard OOD with random corruptions.
Score-based methods excel (0.99) because random corruptions are implausible.
Our methods remain competitive (0.87--0.97) while providing robust temporal performance.

[INSERT TRADE-OFF PARAGRAPH HERE]

\paragraph{Coverage dominates performance gains.}
Structural uncertainty ($U_{\text{str}}$) provides...
```

**Step 3**: Optional - add ensemble appendix reference if you have ensemble experiments

**Step 4**: Verify table references (tab:icews, tab:standard) match your labels

---

## LaTeX Code (Ready to Copy-Paste)

### Option C (Recommended):

```latex
\paragraph{When to use CAGP?}
Our results reveal complementary strengths: CAGP achieves 0.89 AUROC on temporal OOD (ICEWS14) vs 0.52--0.58 for score-based methods, while score-based methods achieve 0.99 on random corruptions vs 0.96 for CAGP (Tables~\ref{tab:icews}--\ref{tab:standard}).
This reflects different mechanisms: score-based methods detect implausibility via low confidence but miss structured shift; CAGP tracks entity-relation co-occurrence, detecting novel contexts.

Practitioners should choose based on deployment: use CAGP for evolving KGs with temporal drift or novel entity-relation patterns; use score-based for detecting random corruptions in static graphs.
For comprehensive coverage, ensemble both approaches.
```

---

## Alternative Placement

If experiments section is too crowded, could also go in:
- **Conclusion** (after line 8, before "Limitations")
- **Discussion subsection** (create new subsection 5.5)

---

## Follow-up (Optional)

If reviewers ask for ensemble results:
```latex
Appendix: Ensemble of CAGP + Energy achieves 0.94 AUROC
on temporal OOD and 0.99 on random corruptions, combining
strengths of both approaches.
```

But this requires running ensemble experiments (not done yet).

---

**Status**: Ready to integrate
**Recommendation**: Use Option C in experiments section after Table 3
