# CAGP Paper — Consolidated Multi-Agent Review Report

**Paper:** "Why Relation-Agnostic Uncertainty Fails: Decomposing OOD Detection in Knowledge Graphs"
**Target Venue:** UAI 2026
**Review Date:** March 4, 2026
**Review Method:** 4 parallel specialized agents (Technical, Experimental, Writing, Novelty/Impact)

---

## Overall Assessment

This paper makes a genuine theoretical contribution through its impossibility theorem demonstrating that variance-based uncertainty methods fail on novel relational contexts in knowledge graphs. The decomposition into semantic and structural uncertainty is clean and the coverage-based solution provides dramatic practical improvements at zero computational cost. However, the paper suffers from a critical gap between theoretical claims and empirical reality: the key assumption (A4) underlying Proposition 2 is violated on all benchmarks, the semantic component adds negligible value on temporally-split (non-circular) evaluations, and statistical rigor is absent. The work is above the acceptance threshold but needs targeted revisions to align claims with evidence.

---

## Aggregate Score: 6.75/10 (Borderline Accept)

| Reviewer | Focus | Score |
|----------|-------|-------|
| 1 | Technical Methodology | 7.0/10 |
| 2 | Experimental Validation | 6.5/10 |
| 3 | Writing Clarity | 7.0/10 |
| 4 | Novelty & Impact | 6.5/10 |
| **Average** | | **6.75/10** |

**Interpretation**: Solidly above rejection threshold (6.0), below strong accept threshold (7.5). Revision required.

---

## Consensus Strengths

- **Sound impossibility theorem**: All reviewers acknowledged Theorem 1 as a legitimate theoretical contribution with correct logical structure
- **Practical value**: Zero-cost coverage signal produces dramatic OOD detection improvements (0.38-0.59 → 0.99 AUROC on ICEWS14)
- **Honest scope disclosure**: Paper transparently limits theorem scope to variance-based methods
- **Reproducibility**: Strong experimental documentation with hyperparameters, per-seed logs, and code availability
- **Clean problem decomposition**: Emerging entities vs. novel contexts framing is well-motivated and clearly articulated
- **Transparent about limitations**: Circularity concerns, oracle caveats, and assumption violations all acknowledged

---

## Critical Weaknesses (Must Address)

### 1. A4 Assumption Violated Everywhere
- Delta >= 1.0 on **all benchmarks** (WN18RR: 1.0, FB15k-237: 1.36, YAGO: 1.0)
- This invalidates Proposition 2's theoretical grounding
- Paper says "directional theory" in appendix but caveat must appear in main text

**Required action**: Add explicit caveat after Proposition 1: "We note that A4 is violated on all benchmarks (Delta >= 1.0). The proposition therefore provides directional predictions rather than formal bounds; we validate empirically that predictions hold within 0.005 AUROC."

### 2. No Statistical Tests
- No p-values, confidence intervals, or significance tests despite multi-seed evaluation
- Claims like "+8-11pp" lack error bars
- Unacceptable for a quantitative contribution

**Required action**: Add 95% confidence intervals and paired significance tests (Wilcoxon or paired t-test) for all main comparisons.

### 3. Semantic Component Adds Near-Zero Value on Non-Circular Benchmarks
- ICEWS18: CAGP = U_str exactly (both 0.987)
- ICEWS14: Improvement marginal (0.99 vs 0.98 on emerging entities only)
- This undermines the "complementarity" narrative

**Required action**: Explicitly state in results: "On current temporal benchmarks, coverage alone is sufficient. The semantic component's value is demonstrated only on static diagnostics."

### 4. Circularity Framing Inverts Evidence Hierarchy
- Headline claim "+8-11pp" comes from circular static benchmarks
- Non-circular temporal results show semantic provides zero/negative value
- Abstract/intro lead with static gains as primary evidence

**Required action**: Lead with ICEWS14 strict-split results (non-circular) as primary evidence. Relegate static benchmark gains to "diagnostic validation."

---

## Major Concerns (Should Address)

### 5. Limited Scale
- Only 4 small benchmarks (7K-123K entities)
- No evaluation on large-scale KGs (Wikidata, OGB-WikiKG2)
- Coverage matrix is |E| x |R| — scalability unvalidated

**Recommendation**: Add at least one benchmark with >500K entities.

### 6. Held-Out Relation Experiment Missing
- Repository contains `run_held_out_relations.py` but results not in paper
- This directly addresses circularity critique
- If it shows non-circular semantic gains, this is critical omission

**Recommendation**: Include held-out relation results in experiments or appendix.

### 7. RelCondVar Positioning
- RelCondVar outperforms CAGP on FB15k-237 (+1pp overall, +4pp emerging)
- Yet CAGP is the main contribution
- Unclear why CAGP is preferred as universal default

**Recommendation**: Explain CAGP preference or promote RelCondVar as co-contribution for dense KGs.

---

## Moderate Concerns (Nice to Address)

### 8. Presentation Issues
- **Abstract too dense**: 8 lines packed with metrics, buries core insight
- **Notation inconsistent**: GP-KGE vs. semantic uncertainty, GPOnly vs. U_sem
- **Tables use tiny fonts**: scriptsize/tiny nearly unreadable

**Recommendations**:
- Rewrite abstract to lead with insight, not metrics
- Add notation table after Section 3
- Increase table font sizes

### 9. Related Work Structure
- Reads as citation list rather than synthesis
- "Positioning" paragraph should come first to orient reader

### 10. Theorem 1 Scope vs. Baselines
- Theorem formally covers only variance-based methods
- But ensemble/energy results claimed as "compatible"
- Deep Ensembles achieve 0.80 on ICEWS14 — above 0.50 impossibility bound

**Recommendation**: Clearly separate formal coverage (variance-based) from empirical patterns (ensemble/energy).

---

## Prioritized Action Items

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **1** | Add statistical significance tests | Medium | High |
| **2** | Add A4 violation caveat to main text | Low | High |
| **3** | Reframe synergy claims for temporal KGs | Low | High |
| **4** | Lead with non-circular ICEWS evidence | Low | Medium |
| **5** | Add held-out relation experiment results | Medium | Medium |
| **6** | Add one large-scale benchmark | High | Medium |
| **7** | Fix notation consistency | Low | Low |
| **8** | Rewrite abstract | Low | Low |
| **9** | Restructure related work | Medium | Low |

---

## Final Recommendation

### **BORDERLINE ACCEPT** — Conditional on Revisions

The impossibility theorem and practical coverage-based solution represent real contributions that advance understanding of OOD detection in knowledge graphs.

**Acceptance conditions:**

| Condition | Priority |
|-----------|----------|
| Add statistical significance tests | **Required** |
| Add A4 violation caveat to main text | **Required** |
| Reframe synergy claims honestly | **Required** |
| Lead with non-circular evidence | **Required** |
| Add large-scale benchmark | Recommended |
| Include held-out relation experiment | Recommended |

**Conditional on addressing the 4 required items, this paper merits acceptance.**

---

## Data Consistency Verification

All numbers cited in text match tables:

| Claim | Verified |
|-------|----------|
| "+8-11pp emerging-entity gains" | Yes |
| "Coverage alone drives 0.99 AUROC" (ICEWS14) | Yes |
| "CAGP maintains 1.00 while Energy collapses to 0.50" (strict split) | Yes |
| "Mixture-AUROC identity within 0.005" | Yes |
| "Spearman rho <= -0.68" | Yes |
| "A4 violated: Delta >= 1.0" | Yes |
| ">=98% frequency-matched" | Yes |

---

## Detailed Reviewer Reports

### Technical Methodology (7/10)

**Strengths:**
- Clean problem decomposition into emerging entities vs novel contexts
- Well-constructed impossibility theorem with sound logical structure
- Honest scope disclosure about theorem applying only to variance-based methods
- All assumptions empirically verified with quantitative metrics

**Weaknesses:**
- A4 violation undermines Proposition 2
- Circular definition concern not fully resolved
- Theorem scope narrower than abstract implies
- Bayesian formulation shallow (standard N(0,I) prior)
- RelCondVar underdeveloped despite interesting results

---

### Experimental Validation (6.5/10)

**Strengths:**
- Comprehensive 5 baselines, multi-seed evaluation (5-10 seeds)
- Addresses circularity with ICEWS temporal splits and strict split experiment
- Detailed ablations (margin loss, architecture, coverage modes)
- Strong reproducibility

**Weaknesses:**
- No formal statistical tests
- Ceiling effect on ICEWS18 (CAGP = U_str)
- ICEWS14 improvement marginal
- Only 4 small datasets
- Selective prediction uses oracle threshold

---

### Writing Clarity (7/10)

**Strengths:**
- Concrete opening example clarifies problem
- Logical flow: problem → theory → solution → validation
- Honest about limitations

**Weaknesses:**
- Abstract too dense
- Notation inconsistencies
- Tables use tiny fonts
- Related work as citation list
- "What we do not claim" placement defensive

---

### Novelty & Impact (6.5/10)

**Core Contributions:**
1. Impossibility theorem for variance-based methods
2. Semantic + structural decomposition framework
3. Mixture-AUROC identity (validated within 0.005)
4. Non-circular temporal evaluation

**Novelty:** MEDIUM-HIGH
- Theorem is genuine contribution
- But coverage signal trivially simple
- "Track what you've seen" intuitive once stated

**Impact:** MEDIUM-HIGH
- Zero-cost practical recommendation
- But semantic component provides marginal gains on temporal benchmarks
- On non-circular evals, coverage alone suffices

---

*Report generated by 4-agent parallel review system*
