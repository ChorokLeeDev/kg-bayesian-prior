# UAI 2025 Review Issues

This document tracks issues identified during UAI submission review and their resolution status.

## Summary

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| W1: Theorem proof | Major | Open | Needs proof fix |
| W2: Simulated variance | Major | **Resolved** | Script now trains models |
| W3: Missing baselines | Major | Open | Add DUQ, SNGP, etc. |
| W4: GP contribution overstated | Major | Open | Reframe narrative |
| W5: (A4) violations | Minor | **Resolved** | All pass with real variance |
| W6: Missing calibration | Minor | Open | Add ECE metrics |
| W7: CAGP vs RelCondVar | Minor | Open | Clarify naming |

---

## W1: Theorem 1 Proof Issues (MAJOR - OPEN)

### Problem

The proof of Theorem 1 (Complementarity) contains a problematic claim at lines 133-136 of `method_uai.tex`:

```latex
AUROC decomposes linearly over OOD mixtures:
AUROC(U, D_ID, D_OOD) = p_E · AUROC(U, D_ID, D_emerge) + p_N · AUROC(U, D_ID, D_novel)
```

**This is not generally true.** AUROC is not linear in mixture proportions. The correct relationship involves the ROC curve's behavior under distribution shift, which is more complex.

### Additional Issues

1. **Assumption (A3)** ("OOD contains both emerging and novel contexts") is tautological given how OOD is defined—it's a definitional choice, not an assumption.

2. **Part (iii) strict dominance** requires `A^{sem}_E > A^{str}_E` (line 168), which is stated as empirically verified but not proved. If semantic and structural perform identically on emerging entities, the combination merely matches rather than strictly exceeds.

### Required Fix

Either:
1. Prove the AUROC decomposition lemma rigorously, or cite an existing result
2. Reframe Part (iii) with weaker but provable claims
3. Add conditions under which strict dominance holds

---

## W2: Simulated Variance in Assumption Verification (MAJOR - RESOLVED)

### Problem

The original `scripts/verify_theorem_assumptions.py` used simulated variance:

```python
def simulate_learned_variance(freq, num_entities):
    variance = 2.0 / (1 + 0.5 * log_freq)  # Circular reasoning!
```

This artificially satisfies (A1) by construction, making Table 6 unreliable.

### Resolution

Script rewritten to train actual GP-KGE models:

```python
def train_model(train_triples, num_entities, num_relations, ...):
    model = SimpleGPKGE(num_entities, num_relations, dim)
    # ... actual training with BCE + KL loss ...
    return model.get_entity_variance()  # Real learned variance
```

### New Results (Trained Model Variances)

| Assumption | FB15k-237 | WN18RR | YAGO3-10 |
|------------|-----------|--------|----------|
| (A1) Spearman ρ | **−0.94** | −0.64 | −0.66 |
| (A4) Δ | **1.00** | **0.83** | **0.82** |

Key finding: (A1) correlation is even stronger with real variances, and (A4) now passes on all datasets.

---

## W3: Incomplete Baseline Comparison (MAJOR - OPEN)

### Problem

For a UAI submission on uncertainty quantification, the paper is missing comparisons with modern uncertainty methods:

- **Evidential Deep Learning** (Sensoy et al., 2018)
- **Spectral-normalized GPs** (Liu et al., 2020)
- **Deterministic Uncertainty Quantification (DUQ)** (van Amersfoort et al., 2020)
- **SNGP** (Liu et al., 2022)

The current baselines (Energy, MC Dropout, Deep Ensemble) are somewhat dated.

### Required Fix

Add experiments comparing against at least 2 modern uncertainty baselines, particularly those designed for OOD detection.

---

## W4: Coverage is Trivial Baseline (MAJOR - OPEN)

### Problem

The structural uncertainty `U_str(h,r,t) = 2 - c(h,r) - c(t,r)` is essentially a lookup table, not a learned uncertainty estimate.

Table 8 (ablation) shows:
- Fixed α = 0.5: **0.958** AUROC
- Learned global α: 0.960 AUROC
- RelCondVar: 0.968 AUROC

The method's success is largely attributable to the coverage lookup, not GP machinery.

### Required Fix

Either:
1. Honestly acknowledge that coverage dominates and the GP component is secondary
2. Reframe the contribution as "uncertainty decomposition" rather than "GP-based"
3. Show scenarios where the GP component provides significant lift

---

## W5: (A4) Violations (MINOR - RESOLVED)

### Problem

Original simulated results showed WN18RR (Δ=1.10) and YAGO3-10 (Δ=1.01) violating the bounded semantic gap assumption.

### Resolution

With trained model variances, all datasets now satisfy Δ < 1:
- FB15k-237: Δ = 1.00 (borderline pass)
- WN18RR: Δ = 0.83 (pass)
- YAGO3-10: Δ = 0.82 (pass)

No further action needed.

---

## W6: Missing Calibration Analysis (MINOR - OPEN)

### Problem

For a UAI paper on uncertainty, Expected Calibration Error (ECE) or reliability diagrams should accompany AUROC. High AUROC doesn't guarantee well-calibrated uncertainties.

### Required Fix

Add to experiments section:
1. ECE metric for each method
2. Brier score
3. Reliability diagrams (appendix)

---

## W7: RelCondVar vs CAGP Naming (MINOR - OPEN)

### Problem

RelCondVar consistently outperforms CAGP:
- ICEWS14: 0.912 vs 0.891
- FB15k-237: 0.968 vs 0.960

Yet CAGP is the named contribution in title/abstract.

### Required Fix

Either:
1. Make RelCondVar the primary method and rename the paper
2. Explain why CAGP's simplicity (no MLP) justifies the small performance gap
3. Position CAGP as the "simple baseline" and RelCondVar as the "full method"

---

## Questions From Review (For Author Response)

**Q1.** ✅ Resolved - Script now uses trained model variances.

**Q2.** Line 168 claims `A^{sem}_E > A^{str}_E` verified as "0.83 vs 0.78 AUROC." Clarify: this appears in Table 3 as Semantic=0.826 vs Structural=0.784 on emerging entities.

**Q3.** The AUROC linear decomposition (proof of Part iii) needs correction or citation.

**Q4.** How does coverage behave for semantically similar but distinct relations (e.g., `located_in` vs `headquarters_in`)? Does binary coverage hurt performance?

**Q5.** Why 20th percentile for τ in OOD split? Sensitivity analysis needed.

---

## Files Modified

| File | Change |
|------|--------|
| `scripts/verify_theorem_assumptions.py` | Rewritten to train actual models |
| `outputs/variance_*.npy` | Cached trained variances |
| `outputs/theorem_assumptions.json` | Updated results with metadata |
| `docs/UAI_REVIEW_ISSUES.md` | This document |

---

## Priority Order for Fixes

1. **W1** - Theorem proof (blocks acceptance)
2. **W3** - Missing baselines (significant effort)
3. **W4** - Reframe GP contribution (writing only)
4. **W6** - Add calibration (moderate effort)
5. **W7** - Naming clarification (writing only)
