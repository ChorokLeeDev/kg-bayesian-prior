# Paper Update: Coverage Dominance Acknowledged

**Date**: 2025-12-25
**Status**: ✅ **COMPLETE & COMPILED**

## Summary

Added explicit acknowledgment throughout the paper that structural uncertainty (coverage) provides approximately 83% of performance gains on temporal OOD, addressing reviewer concern #15: *"Acknowledge directly that coverage provides ~80% of gains"*.

## Changes Made

### 1. Introduction - Contribution Paragraph (MOST IMPORTANT)

**Location**: `paper/sections/introduction_uai.tex`, lines 19-23

**Before**:
```latex
We decompose KG uncertainty into semantic and structural components.
[...] Theorem formalizes when each signal is necessary...
```

**After**:
```latex
We decompose KG uncertainty into semantic and structural components.
[...] Our contribution is not coverage itself---which provides
approximately 83% of performance gains on temporal shift---but rather:
(1) the formalization of why coverage is necessary through Theorem 1,
which proves semantic and structural uncertainties are non-redundant;
(2) demonstrating that learned embeddings cannot recover this signal;
and (3) identifying when to combine coverage with variance.
```

**Impact**:
- **Completely transparent from the start**
- Immediately frames contribution as theoretical formalization, not empirical engineering
- Pre-empts reviewer criticism by being upfront

### 2. Section 5.4 - Dedicated Paragraph

**Location**: `paper/sections/experiments_uai.tex`, lines 100-101

**Before**: "Relative contributions."

**After**: "Coverage dominates performance gains."

**Content Enhanced**:
```latex
Structural uncertainty (U_str) provides the majority of performance
improvement on temporal OOD. On ICEWS14, coverage alone achieves
0.824 AUROC compared to 0.687 for semantic variance---accounting
for approximately 83% of the total gain over random baseline (0.5).
The combination (CAGP) adds an additional 8% improvement (0.891 AUROC).
This asymmetry reflects that novel contexts---which coverage detects
perfectly---dominate realistic temporal distribution shift, while
emerging entities (detected by semantic uncertainty) are less prevalent.
Our contribution is not coverage itself, but the formalization of
why it is necessary (Theorem 1) and when to combine it with learned
variance.
```

**Key Improvements**:
- Explicit percentage calculation: 83%
- Clear separation: coverage = 83%, combination = additional 8%
- Explains WHY this asymmetry exists (novel contexts dominate)
- Reiterates what our actual contribution is

## Calculation Details

From Table 1 (ICEWS14 results):
```
Random baseline: ~0.5 (implicit)
U_str alone:      0.824
U_sem alone:      0.687
CAGP:             0.891

Total improvement (random → CAGP): 0.891 - 0.5 = 0.391
Coverage contribution (random → U_str): 0.824 - 0.5 = 0.324
Percentage: 0.324 / 0.391 = 82.9% ≈ 83%
```

## Why This Strengthens the Paper

### 1. **Demonstrates Intellectual Honesty**
Being upfront about where performance comes from shows confidence in the theoretical contribution.

### 2. **Reframes the Contribution**
Instead of claiming "we get good results," we're saying:
- Coverage provides most gains (acknowledged fact)
- But learned embeddings can't discover it (surprising finding)
- We formalize WHY it's necessary (theoretical contribution)
- We prove WHEN to combine it (practical contribution)

### 3. **Pre-empts Criticism**
By acknowledging this ourselves, we control the narrative rather than appearing defensive when reviewers point it out.

### 4. **Clarifies the Value Proposition**
The value is NOT:
- ❌ "We invented coverage" (it's a simple lookup)
- ❌ "Coverage works well" (that's obvious)

The value IS:
- ✅ Formalized theoretical framework (Theorem 1)
- ✅ Demonstrated learned methods fail to discover it
- ✅ Characterized when each signal is necessary
- ✅ Proved combination strictly dominates

## Files Modified

1. `paper/sections/introduction_uai.tex` (line 22)
2. `paper/sections/experiments_uai.tex` (line 100-101)

## Compilation Status

✅ **Paper compiles successfully**
- Output: `main_uai.pdf` (250 KB)
- No errors or new warnings

## Reviewer Response Template

**Concern**: "You should acknowledge that coverage provides most of the gains"

**Our Response**:

> We explicitly acknowledge this in two places:
>
> 1. **Introduction (line 22)**: "Our contribution is not coverage
>    itself---which provides approximately 83% of performance gains on
>    temporal shift---but rather..."
>
> 2. **Section 5.4**: Dedicated paragraph titled "Coverage dominates
>    performance gains" with detailed breakdown.
>
> We are transparent that structural uncertainty provides the majority
> of improvement. Our contribution is the formalization (Theorem 1) of
> why this is necessary and when to combine it with learned variance.
> This theoretical understanding is non-obvious: learned probabilistic
> embeddings have access to the same training data but fail to recover
> the coverage signal.

## Before vs After Framing

### Before (Potentially Misleading)
"Our method achieves 0.891 AUROC"
→ Implies we invented something novel

### After (Honest and Clear)
"Coverage provides 83% of gains (0.824 AUROC). Our contribution is formalizing why it's necessary and when to combine it with learned variance."
→ Clear about sources of improvement and our actual contribution

## Impact on Paper Narrative

The paper now has a clear three-part story:

1. **Problem**: Learned embeddings are relation-agnostic (0.52-0.58 AUROC)
2. **Empirical Observation**: Coverage helps a lot (83% of gains)
3. **Our Contribution**: Theoretical framework explaining WHY and WHEN

This is much stronger than claiming novelty for coverage itself.

## Related Updates

This complements other recent updates:
- ✅ Continuous coverage ablation (shows binary is optimal)
- ✅ Coverage dominance acknowledgment (this update)
- ⏳ Still needed: Error analysis, SOTA base models

## Next Steps

- [x] Acknowledge coverage dominance in introduction
- [x] Acknowledge coverage dominance in experiments
- [x] Compile and verify
- [ ] Consider adding similar acknowledgment to EMNLP version
- [ ] Update presentation slides to match this framing

---

**Conclusion**: The paper now honestly and prominently acknowledges that coverage provides ~83% of gains, while clearly articulating that our contribution is the theoretical formalization and understanding of when/why to use it. This transparency strengthens rather than weakens the paper.
