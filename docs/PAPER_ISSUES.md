# Paper Issues & Fixes for NeurIPS 2026

## Critical Issues (Must Fix)

### 1. Abstract vs Experiments Error Mismatch
- **Abstract:** "validated within 3% error"
- **Experiments:** WN18RR shows 3.6% error
- **Fix:** Change abstract to "within 4% error" or recalculate

### 2. Complementarity Theorem Case 2 is Flawed
**Current:**
> Entity e is rare (high σ²_e) but seen once with relation r. For a truly OOD triple, GP correctly gives high uncertainty; coverage incorrectly gives low uncertainty.

**Problem:** This case is confusing. If e is seen with r, coverage gives low uncertainty for BOTH ID and OOD involving e with r. The case should be clearer.

**Fix:** Rewrite as:
> Case 2: Consider an OOD triple (h, r, t') where the randomly sampled tail t' is a rare entity (high σ²_{t'}) that happens to have been seen once with relation r. Coverage gives low uncertainty (incorrectly suggests ID). GP correctly flags the rare entity with high uncertainty.

### 3. No Figures Referenced in Paper
The paper doesn't reference any of the 5 figures we created:
- fig1_main_results.pdf
- fig2_theorem_validation.pdf
- fig3_decomposition.pdf
- fig4_synergy_breakdown.pdf
- fig5_gp_limitation.pdf

**Fix:** Add figure references in appropriate sections.

### 4. Missing YAGO3-10 in Theorem Validation
Table 2 only validates theorem on WN18RR and FB15k-237. YAGO3-10 is missing.

**Fix:** Either add YAGO3-10 validation or explain why it's not included.

### 5. α "Learnable" but Stays at 0.5
Paper says α is learnable but reports it stays at initialization (0.5).

**Reviewer attack:** "Did α actually train? This seems like just α=0.5 was hardcoded."

**Fix options:**
1. Show α training curve (did it move then return to 0.5?)
2. Report learned α values with confidence intervals
3. Reframe: "α initialized to 0.5 and empirically optimal at this value"
4. Show ablation: α ∈ {0.25, 0.5, 0.75} to prove 0.5 is optimal

### 6. Appendix Proof is Hand-Wavy
Line: "P(U_OOD ≥ 1) ≈ s_r (simplified)"

**Problem:** This approximation isn't justified and will be questioned.

**Fix:** Either provide full derivation or clearly state assumptions.

---

## Medium Issues (Should Fix)

### 7. No Link Prediction Results
Reviewers will ask: "Does adding coverage hurt the actual task?"

**Fix:** Add MRR/Hits@10 comparison showing CAGP maintains performance.

### 8. Synergy Not Defined
Paper uses "synergy" (e.g., "+32%") without defining the formula.

**Fix:** Add definition:
> Synergy = (AUROC_CAGP - max(AUROC_GP, AUROC_Cov)) / max(AUROC_GP, AUROC_Cov)

### 9. No Standard Deviations in Main Table
Main results table shows only means, std only in appendix.

**Fix:** Add ± std to main table, or at least reference appendix.

### 10. Chen2022 Citation is Vague
The reference is:
```
@article{Chen2022,
  title={Probabilistic Entity Representation...},
  author={Chen, Zhangjie and others},
  journal={arXiv preprint},
  year={2022}
}
```

**Fix:** Find proper citation or clarify this is a placeholder.

### 11. No Computational Cost Discussion
How much overhead does coverage add?

**Fix:** Add timing comparison or complexity analysis.

---

## Minor Issues (Nice to Fix)

### 12. "I prove" is Too Strong
Abstract: "I prove that coverage provides a sufficient statistic..."

**Problem:** The theorem is empirically validated, not proved from first principles.

**Fix:** Change to "I show" or "I demonstrate"

### 13. Proposition 1 is Trivial
"GP-KGE uncertainty is relation-agnostic... Proof: Immediate from equation."

**Problem:** Reviewers might say this doesn't need to be a formal proposition.

**Fix:** Could demote to observation, or keep for emphasis.

### 14. OOD Protocol Citation
Background cites safavi2020codex for OOD protocol. Need to verify this is correct.

---

## Potential Reviewer Attacks

| Attack | Current Defense | Strength |
|--------|-----------------|----------|
| "This is just feature engineering" | Theorems explain why | Medium |
| "α doesn't learn anything" | Reports α≈0.5 | Weak |
| "No link prediction results" | Not reported | Missing |
| "Simple linear combination" | "Insight > method" framing | Medium |
| "Only random corruption OOD" | Acknowledged in limitations | OK |
| "Coverage is not novel" | "Repurposed for uncertainty" | Weak |

---

## Action Items

### Must Fix Before Submission
1. [ ] Fix abstract error (3% → 4%)
2. [ ] Rewrite Complementarity Case 2
3. [ ] Add figure references
4. [ ] Add YAGO to theorem validation OR explain omission
5. [ ] Address α=0.5 concern (ablation or reframing)
6. [ ] Fix appendix proof approximation

### Should Fix
7. [ ] Add link prediction results (MRR/Hits@10)
8. [ ] Define synergy formula
9. [ ] Add std to main table
10. [ ] Fix Chen2022 citation
11. [ ] Add computational cost

### Optional
12. [ ] Change "prove" to "show"
13. [ ] Demote Proposition 1
14. [ ] Verify OOD citation
