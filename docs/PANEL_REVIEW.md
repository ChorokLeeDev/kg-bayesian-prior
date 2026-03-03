# Simulated NeurIPS 2026 Panel Review

**Paper:** Why Relation-Agnostic Uncertainty Fails: Decomposing OOD Detection in Knowledge Graphs
**Date:** 2026-02-28
**Purpose:** Pre-submission review to identify weaknesses before NeurIPS deadline (~May 2026)

---

## Reviewer 1: Theory & Uncertainty Quantification Expert

**Overall Score:** 6/10
**Confidence:** 4/5

### Summary

This paper diagnoses failures of relation-agnostic probabilistic KG embeddings on temporal OOD detection by decomposing uncertainty into semantic (entity variance) and structural (coverage) components. The main contribution is Theorem 1, which proves that any relation-agnostic uncertainty estimator achieves near-random AUROC on novel-context OOD under assumptions A1--A3. While the decomposition insight is valuable, the theoretical results rest on questionable assumptions, and the gap between formal guarantees and empirical reality significantly weakens the contribution's novelty.

### Strengths

1. **Clear problem motivation and diagnostic value.** The paper precisely identifies a real limitation of existing probabilistic KG embeddings: relation-agnostic variance fails to detect when well-observed entities appear in novel relations. Well-articulated through concrete examples.

2. **Empirically validated decomposition.** The mixture AUROC formula (Eq. 3) predicts overall performance to within 0.005 on static benchmarks and 0.001 on some datasets. Stratified analysis clearly demonstrates complementary failure modes.

3. **Ground-truth temporal validation on ICEWS14/18.** Chronological train/test splits break the definitional coupling, providing independent evidence. CAGP achieves 0.99 AUROC on ICEWS14.

4. **Reparameterization sampling diagnostic.** Useful practical finding: KL regularization collapses entity variances without reparameterization sampling (AUROC 0.92 vs 0.72 on WN18RR).

5. **Honest discussion of limitations.** Transparent about circularity, A4 violations, and RelCondVar limitations.

### Weaknesses

1. **Theorem 1's novelty is limited.** The insight is largely intuitive: if variance is determined by frequency (A1), and novel-context entities are frequency-matched to ID (A3), then frequency-based signals cannot distinguish them. The proof is straightforward (3 sentences). Assumption A1 is informal and clearly false in practice. Only asymptotic bounds (1/2 + O(epsilon)) without finite-sample rates.

2. **Proposition 2 relies on violated assumptions.** A4 (Delta < 1) is violated on all three benchmarks (Delta >= 1.0). Justification relies on empirical fit rather than theoretical guarantee. The proposition's value is diagnostic rather than predictive.

3. **Static benchmarks exhibit definitional circularity.** Novel-context OOD is defined by coverage, and the detector uses the same coverage indicator (AUROC=1.0 by construction). Non-circular validation (ICEWS14/18) is valuable but CAGP and U_str often match (ICEWS18: identical at 0.987).

4. **CAGP model itself lacks novelty.** Fixed 50/50 linear combination. Gains almost entirely from structural uncertainty. On ICEWS18, identical to U_str.

5. **Limited scope and missing baselines.** No GNN encoders (R-GCN, CompGCN). MC Dropout and Deep Ensembles perform surprisingly poorly, raising implementation quality questions. No comparison with conformal prediction methods.

### Questions for Authors

1. If variance depends on factors beyond frequency, does the impossibility result still hold? Can you provide finite-sample AUROC bounds?

2. Under what conditions does Eq. 3 remain approximately valid when A4 is violated?

3. On ICEWS14, semantic uncertainty adds only +2pp on emerging entities (0.98 to 1.0). What is the evidence for complementarity beyond static benchmarks?

### Verdict

**Borderline** -- Useful diagnostic contribution but theoretical results provide limited novelty (intuitive insight, violated assumptions, asymptotic bounds). The method is simple (50/50 mixture) with gains driven by coverage. Needs tighter theoretical results under realistic assumptions or deeper investigation of when the decomposition breaks.

---

## Reviewer 2: Knowledge Graph & GNN Expert

**Overall Score:** 5/10
**Confidence:** 4/5

### Summary

This paper diagnoses a fundamental limitation in probabilistic KG embeddings (relation-agnostic variance) and proposes CAGP combining entity variance with a coverage matrix. While the diagnostic insight has merit, the method is primarily a static coverage lookup, evaluation on static benchmarks is circular, and comparison with modern KG methods (ULTRA, GNN encoders) is absent.

### Strengths

1. **Clear diagnostic contribution.** Theorem 1 formally captures the relation-agnostic limitation. Identifying that existing methods achieve only 0.34--0.87 AUROC on temporal OOD is valuable.

2. **Honest treatment of circularity.** Transparent acknowledgment that static results are diagnostic, not independent detection evidence.

3. **Solid theoretical framing.** Mixture AUROC decomposition predicts within 0.005 across datasets.

4. **Reparameterization sampling diagnostic.** Practical guidance for variational KGE practitioners.

### Weaknesses

1. **Coverage is a lookup table dressed up as a model.** $U_{\text{str}}(h,r,t) = 2 - c(h,r) - c(t,r)$ is deterministic. CAGP's gains come from a normalization trick, not a learned model component.

2. **Missing comparison with modern KG methods.** No ULTRA, NodePiece, inductive approaches. DistMult/TransE/ComplEx are dated baselines for 2026. No R-GCN or CompGCN. These are critical gaps.

3. **Static benchmark results are diagnostic, not evidence.** Novel-context AUROC=1.0 by construction. CAGP's only genuine contribution is +8--12pp on emerging entities. ICEWS18 shows zero complementarity.

4. **Limited scope to transductive, simple embedding models.** Modern architectures integrating uncertainty into scoring may break the theorem's assumptions. Restriction to transductive settings is limiting in 2026.

5. **Scalability concerns inadequately addressed.** No runtime benchmarks, memory profiling, or cost comparison. Coverage matrix $O(|E| \times |R|)$ for Wikidata-scale KGs is questionable.

6. **Arbitrary OOD splits.** Threshold tau = 25th percentile is not validated against real-world temporal KG evolution. Not compared to TGB 2.0 protocols.

### Questions for Authors

1. Why not include ULTRA and GNN encoders? If GNNs learn relation-aware uncertainty via message passing, this validates or refutes the core claim.

2. Can you quantify computational cost of coverage tracking on all datasets and extrapolate to 1M+ entity KGs?

3. How sensitive is CAGP to the OOD partition threshold tau across [10th, 50th] percentile?

### Verdict

**Borderline / Weak Reject** -- Clear theoretical contribution but method is coverage lookup + rescaling trick. Missing modern KG baselines (ULTRA, GNNs) is a critical gap for NeurIPS 2026. Needs stronger baselines and honest positioning as diagnostic/engineering rather than learned model innovation.

---

## Reviewer 3: OOD Detection & Robustness Expert

**Overall Score:** 6/10
**Confidence:** 5/5

### Summary

This paper decomposes KG OOD detection into semantic (entity-variance) and structural (coverage) components, proving relation-agnostic estimators fail on novel contexts. CAGP achieves 0.90--0.97 AUROC. While the impossibility result is elegant and the complementarity insight valuable, circularity on static benchmarks, over-reliance on ICEWS14/18, and the method being "use a feature" rather than algorithmic novelty limit the contribution.

### Strengths

1. **Formal impossibility theorem with real-world validation.** Theorem 1 elegantly formalizes relation-agnostic failure. Anti-predictive direction (AUROC < 0.5) is non-trivially predicted.

2. **Two distinct OOD types with predicted complementarity.** Clearly identified, empirically validated (Table 2). Mixture AUROC prediction holds to within 0.005.

3. **Ground-truth temporal benchmarks.** ICEWS14 strict-split results (0.9945 AUROC, improving after removing leakage) provide strong non-circular validation.

4. **Reparameterization sampling diagnostic.** Practical, actionable insight.

5. **Simple, implementable method.** No architectural overhead. Practical for real KGs.

### Weaknesses

1. **Fundamental circularity on static benchmarks.** Novel-context AUROC=1.0 by construction on 3 of 5 datasets. The 0.90--0.97 range in the introduction conflates circular and non-circular results. This is misleading.

2. **ICEWS14/18 alone are insufficient.** ICEWS18 shows zero complementarity (CAGP = U_str everywhere). ICEWS14 emerging-entity gain is only +2pp. Two temporal datasets, one showing no complementarity, do not robustly validate the core claim.

3. **The contribution is "use a feature the baselines don't."** No learned architecture, no new loss function. NeurIPS will ask: why not just test Energy + coverage or Deep Ensembles + coverage? If those match CAGP, novelty collapses.

4. **Training signal asymmetry.** Uncertainty margin loss (w_unc=0.1) directly optimizes the test objective for learned-variance methods. Without clear ablation showing margin loss is not the primary source of gains, comparison is confounded.

5. **Novel contexts are inherently easy to detect.** By definition c(e,r)=0 vs c(e,r)>0 -- a deterministic rule. The harder problem (emerging entities) gets +8--12pp. The decomposition may be KG-specific rather than transferable.

6. **RelCondVar is underexplored.** Beats CAGP on FB15k-237, matches U_str elsewhere. Why not investigate deeper? What does it learn? Can it be improved?

7. **Narrow evaluation scope.** Only DistMult-based, non-standard OOD splits, no standard KG OOD benchmarks.

### Questions for Authors

1. Does CAGP without margin loss still exceed U_sem? Full ablation needed.

2. Did you test Energy + coverage or Deep Ensembles + coverage? If competitive with CAGP, novelty shifts to "coverage is useful."

3. Why does RelCondVar underperform fixed alpha=0.5 on sparse KGs?

### Verdict

**Weak Accept** -- Valuable theoretical contribution (Theorem 1) and principled decomposition, but held back by circularity, marginal non-circular validation, training asymmetry, and method being "use a feature" not algorithmic novelty. Better suited if: (1) static results repositioned as diagnostics only, (2) GDELT or additional temporal KGs added, (3) fair baseline comparisons with coverage, (4) deeper RelCondVar investigation.

---

## Panel Summary

| | R1 (Theory) | R2 (KG/GNN) | R3 (OOD) | Average |
|---|---|---|---|---|
| Score | 6/10 | 5/10 | 6/10 | **5.7/10** |
| Verdict | Borderline | Borderline/WR | Weak Accept | **Borderline** |

### Consensus Weaknesses (all 3 reviewers agree)

1. **Circularity on static benchmarks** is the #1 concern. All reviewers flag that novel-context AUROC=1.0 by construction inflates reported numbers.
2. **ICEWS14/18 alone insufficient for complementarity claim.** ICEWS18 shows zero lift; ICEWS14 shows only +2pp on emerging entities.
3. **Missing modern baselines.** R-GCN, CompGCN, ULTRA not evaluated. Dated baseline set for 2026.
4. **Method novelty is low.** Fixed 50/50 mixture of variance + coverage lookup. No learned component.

### Consensus Strengths

1. **Theorem 1 is a useful formalization** even if the insight is intuitive.
2. **Honest self-assessment** of limitations and circularity.
3. **Reparameterization sampling diagnostic** has independent value.
4. **ICEWS14 strict-split result** is compelling evidence against transductive artifacts.

### Actionable Recommendations for NeurIPS Resubmission

1. **Reframe static results as purely diagnostic.** Do not cite 0.90--0.97 in the abstract/intro as performance claims. Lead with ICEWS14 strict-split (0.9945) as the headline result.
2. **Add GDELT or another temporal KG** to diversify non-circular validation beyond ICEWS14/18.
3. **Implement R-GCN/CompGCN baselines** to show decomposition applies to GNN architectures.
4. **Test "baseline + coverage" ablations** (Energy + coverage, Deep Ensembles + coverage) to demonstrate CAGP's value beyond just "use coverage."
5. **Include margin-loss ablation in main text** to address training signal asymmetry concern.
6. **Investigate RelCondVar** more deeply -- why it fails on sparse KGs and what it learns on dense ones.
7. **Position honestly:** The paper is primarily a diagnostic/analytical contribution with a simple but effective method, not a novel architecture. Own this framing.
