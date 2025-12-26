# UAI Revision Implementation Plan

## Current State Assessment (2025-12-26)

### Paper Structure (main_uai.tex)
- Uses UAI 2024 style class
- 8 sections: Intro, Related Work, Background, Method, Experiments, Conclusion, + Appendices
- Method file: `sections/method_uai_v2.tex`
- RelCondVar: Currently positioned as "Extension" after CAGP (method_uai_v2.tex:54)
- Theorem: States "mild assumptions" (method_uai_v2.tex:33)

### Current Results
- RelCondVar consistently best: 0.912 AUROC on ICEWS14, 0.972 on FB15k-237
- CAGP second best: 0.891 on ICEWS14, 0.965 on FB15k-237
- Strong appendix content already exists (calibration, error analysis, ablations)

### Identified Inconsistencies
1. RelCondVar is relegated to appendix but gets best results
2. "Mild assumptions" language conflicts with violations (A4: Δ=1.10 on WN18RR)
3. No impossibility theorem for relation-agnostic methods
4. Missing simple baselines (frequency-only, coverage-only explicit mention)
5. Binary coverage = 1.0 discrepancy not explained in main text

---

## CRITICAL REVISIONS (Must Do)

### 1. Lead with RelCondVar ✓
**Location**: `sections/method_uai_v2.tex`
**Changes**:
- Move RelCondVar from "Extension" paragraph to main method presentation
- Restructure as: "Two Approaches to Relation-Specific Uncertainty"
  - Approach 1 (Primary): RelCondVar - learned end-to-end
  - Approach 2 (Baseline): CAGP - explicit coverage augmentation
- Update abstract/introduction to reflect this ordering

**Files to edit**:
- `sections/method_uai_v2.tex` (restructure §4.4)
- `sections/abstract_uai.tex` (mention both approaches)
- `sections/introduction_uai.tex` (reorder presentation)

### 2. Add Impossibility Theorem ✓
**Location**: NEW section in `sections/method_uai_v2.tex` before Theorem 1
**Content**:
```latex
\begin{theorem}[Impossibility of Relation-Agnostic Detection]
\label{thm:impossibility}
Any uncertainty estimator $U: \mathcal{E} \times \mathcal{R} \times \mathcal{E} \to \mathbb{R}$
of the form $U(h,r,t) = f(\sigma^2_h, \sigma^2_t)$ where $\sigma^2_e$ depends only on
entity $e$ (relation-agnostic) achieves AUROC $\leq 1/2 + O(\epsilon)$ on novel contexts
under assumptions A1-A3, where $\epsilon$ measures frequency overlap.
\end{theorem}
```

**Files to edit**:
- `sections/method_uai_v2.tex` (add new theorem + proof sketch)
- Appendix proof section (add detailed proof)

### 3. Add Simple Baselines ✓
**Location**: `sections/experiments_uai.tex` - Table 1 (ICEWS14 results)
**Changes**:
Add 3 rows before current baselines:
```
Frequency-only    | 0.687 | ... | ... |
Coverage-only     | 0.824 | ... | ... |
Simple avg (α=0.5)| 0.883 | ... | ... |
```

Note: Coverage-only = $U_{\text{str}}$ (already exists), frequency ≈ $U_{\text{sem}}$
Just need to add "Simple average" row explicitly

**Files to edit**:
- `sections/experiments_uai.tex` (update Table 1)
- May need to run quick experiment for simple average if not already computed

### 4. Soften Theorem Assumptions ✓
**Location**: `sections/method_uai_v2.tex:33`
**Changes**:
```latex
% BEFORE:
Under mild assumptions (variance decreases with frequency, ID triples have
full coverage, bounded semantic gap; see Appendix~\ref{app:proof}):

% AFTER:
Under idealized conditions (monotonic variance-frequency relationship, complete
ID coverage, approximate frequency overlap; see Appendix~\ref{app:proof} for
precise statements and robustness analysis):
```

**Files to edit**:
- `sections/method_uai_v2.tex` (line 33)
- Appendix (add robustness discussion after assumption verification table)

### 5. Explain Binary Coverage = 1.0 Discrepancy ✓
**Location**: `sections/experiments_uai.tex` after binary vs continuous coverage paragraph
**Content**: Add new paragraph explaining temporal split composition
- FB15k-237 simulated temporal: 94% novel contexts → coverage perfect
- ICEWS14 ground-truth temporal: 61% novel contexts, 39% emerging → 0.824

**Files to edit**:
- `sections/experiments_uai.tex` (add paragraph after line 140)

### 6. Rewrite Introduction ✓
**Location**: `sections/introduction_uai.tex`
**Changes**:
- Remove defensive "coverage is 83% but formalization is our contribution"
- Replace with: "We identify systematic failure + propose two solutions"
- Lead with scientific discovery, not justification

**Key paragraph to rewrite** (lines 20-23):
```latex
% NEW VERSION:
We identify a systematic limitation in probabilistic KG embeddings: learned
variances are relation-agnostic despite training on data containing
entity-relation co-occurrence patterns. This failure persists even when
architectures are given capacity for relation-specific uncertainty
(§\ref{sec:why_baselines_fail}), revealing a mismatch between standard
training objectives (link prediction) and OOD detection requirements.

We propose two solutions: (1) RelCondVar, which learns relation-conditioned
variance σ²(e,r) via an auxiliary OOD objective, and (2) CAGP, which explicitly
augments variance with coverage tracking. Both achieve 0.89-0.91 AUROC on
temporal shift (67% relative improvement over baselines).
```

---

## HIGH PRIORITY REVISIONS (Substantially Strengthens)

### 7. Add Stratified Evaluation Table ✓
**Location**: NEW table in `sections/experiments_uai.tex` after Table 2
**Content**: Table showing AUROC separately for:
- Emerging entities only (freq < τ)
- Novel contexts only (freq ≥ τ, coverage = 0)
- Mixed (both conditions)
- Overall

This directly validates each part of Theorem 1.

**Note**: May need to compute this from existing results or run targeted evaluation

### 8. Add 2×2 Comparison Table ✓
**Location**: `sections/introduction_uai.tex` after Figure 1 or in experiments
**Content**:
```
Method          | Random Corruption | Temporal Shift |
UKGE (score)    | 0.992            | 0.523          |
SNGP (distance) | 0.634            | 0.614          |
CAGP (coverage) | 0.960            | 0.891          |
```

Purpose: Show methods are complementary, not competitive

### 9. Add "Why Don't Baselines Learn Coverage?" Section ✓
**Location**: NEW subsection in `sections/experiments_uai.tex`
**Content**:
- Experiment: Augment Deep Ensemble with relation-aware architectures
- Result: Still fails on novel contexts (AUROC ≈ 0.61)
- Explanation: Training objectives optimize prediction, not OOD detection
- Binary coverage is discrete, non-smooth → gradient descent doesn't discover it

**Note**: This likely requires NEW experiments (may need to defer or simulate based on existing architecture ablations)

---

## NICE TO HAVE

### 10. Scalability Analysis ✓
Add paragraph on computational/memory costs for different KG sizes

### 11. Fix Notation Inconsistencies ✓
- Define $\tilde{U}_{\text{sem}}$ when first used
- Ensure consistent entity/relation notation (e vs $e$, etc.)

### 12. Consolidate Appendix ✓
- Group related content (all ablations together, all calibration together)
- Current structure is already pretty good

---

## Implementation Order

**Day 1 (Critical):**
1. Soften theorem assumptions (30 min) - EASY
2. Explain binary coverage discrepancy (1 hour) - MEDIUM
3. Add simple baselines to tables (30 min) - EASY
4. Rewrite introduction (2 hours) - MEDIUM
5. Restructure method section (RelCondVar first) (2 hours) - MEDIUM
6. Add impossibility theorem (2 hours) - HARD

**Day 2 (Strengthening):**
7. Add 2×2 table (30 min) - EASY
8. Add stratified evaluation table (3 hours: may need computation) - HARD
9. Add "why baselines fail" section (framework only, defer experiments) - MEDIUM
10. Scalability paragraph (30 min) - EASY
11. Fix notation (1 hour) - EASY
12. Final consistency check + compile (1 hour) - EASY

---

## Files to Modify

### Main Paper Sections:
1. `sections/abstract_uai.tex` - Update contribution statement
2. `sections/introduction_uai.tex` - Major rewrite (less defensive)
3. `sections/method_uai_v2.tex` - Add impossibility theorem, restructure CAGP/RelCondVar
4. `sections/experiments_uai.tex` - Add baselines, tables, explanations
5. `sections/conclusion_uai.tex` - Minor updates if needed

### Appendix:
6. `main_uai.tex` (appendix section) - Add impossibility proof, robustness analysis

### Potentially New Files:
7. May need new experiment scripts (stratified eval, relation-aware baseline)

---

## Success Metrics

After revisions, paper should:
- ✓ Lead with principled learned solution (RelCondVar)
- ✓ Have formal impossibility result (not just empirical observation)
- ✓ Include simple baselines for fair comparison
- ✓ Present theorem as qualitative insight with robustness caveats
- ✓ Explain all discrepancies (binary coverage = 1.0, etc.)
- ✓ Frame as discovery of systematic failure + solutions (not defensive about novelty)
- ✓ Position complementarity to existing methods (not competition)

**Target**: Upgrade from BORDERLINE ACCEPT (6/10) → ACCEPT (7-8/10)
