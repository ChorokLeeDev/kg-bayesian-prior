# Rebuttal Templates (UAI 2026)

Prepared Q→E (Concern → Evidence) responses for likely reviewer concerns.

---

## Rapid Rebuttal Card (Low Character Budget)

Use this when a reviewer box is short. Keep each response in the pattern:
**Concern → 1 number → 1 reference**.

1. **Leakage concern** → strict split removes 7,736/13,222 test triples (58.5%) and CAGP improves 0.99→0.995 while Energy drops 0.59→0.50 → Table 9, Appendix C.6
2. **"Novel-context AUROC=1 is definitional"** → ICEWS14 uses timestamp labels (not coverage labels) and CAGP still gets 0.99 → Table 1, Threats to Validity
4. **Robustness/seeds** → All results 3-seed mean±std; coverage-based methods yield std=0.000 (deterministic) → Table 1
5. **Density artifact concern** → shuffled coverage 0.551, random coverage 0.502 (near chance) → Table 8

Canonical framing sentence (reuse verbatim for consistency):
`Structural (coverage) signal is the primary driver on temporal OOD; semantic signal provides complementary value on emerging-entity slices.`

---

## Q_UNIFIED: "Coverage is trivial / AUROC is definitional / Why not just use coverage?"

**Core Response (unified Q1+Q2+Q3):**

We agree that binary coverage is simple—that is the point. Our contribution is not the feature but three things reviewers cannot dismiss:

1. **Impossibility theorem (Theorem 1):** Any entity-level uncertainty method fails on novel contexts, regardless of model complexity. This is architecture-agnostic and definition-independent.
2. **ICEWS14 ground-truth validation:** OOD is defined by chronological timestamps (not coverage labels), yet CAGP achieves 0.99 AUROC (3 seeds, std=0.000). Strict split removing 58.5% of test triples *improves* to 0.995.
3. **Ablation destroys performance:** Shuffled coverage → 0.55; random coverage at matched density → 0.50 (chance). The specific entity-relation structure drives detection, not density statistics.
4. **Matched-coverage analysis [NEW]:** Among triples sharing the same coverage value (coverage controlled), GP semantic uncertainty still provides 0.79 AUROC (WN18RR) and 0.86 AUROC (FB15k-237) separation on emerging entities—coverage alone is not doing all the work.

**Why coverage-only nearly matches CAGP (Q2):** Coverage is the dominant signal for temporal OOD. We state this explicitly (§5.1). The contribution is the *decomposition framework* (Theorem 1 + 2), not the combination weight. Semantic signal is complementary on emerging entities where coverage coupling is weakest (ρ=0.13–0.68). On YAGO3-10, CAGP achieves the largest gain (+0.18 over U_str) because both signals contribute roughly equally (0.82 vs 0.76).

**Why novel-context AUROC=1.0 is not vacuous (Q3):** We acknowledge circularity (Remark 3). But novel contexts constitute 23–32% of test triples across 4 datasets, and all existing probabilistic methods score 0.38–0.60 on them. The practical question is: "Can any existing method detect these?" The answer is no (Theorem 1).

---

## Q1: "AUROC 0.99–1.00 is suspiciously high. Is there data leakage?"

**Response:**
See Q_UNIFIED for the comprehensive defense. Additional leakage-specific evidence:

1. **Temporal integrity:** Train/test timestamps have zero overlap (train: IDs 0–6264, test: 7536–8736).
2. **Inverse relations:** 53% of test triples have inverse counterparts in training—expected for political event KGs. Our structural signal tracks *entity-relation* pairs c(e,r), not entity pairs, so inverse relations for different r do not inflate detection.
3. **Strict split:** Removing all exact duplicates + inverse overlaps (58.5% of test) *improves* CAGP from 0.99 to 0.995 (Table 9), ruling out transductive artifacts.
4. **Ablation:** Shuffling coverage rows (same density/statistics) drops AUROC to 0.55; random coverage at matched density yields 0.50 (Table 8). The *specific* entity-relation structure drives detection.

---

## Q4: "The base model (DistMult) underperforms SOTA. Would results hold with better models?"

**Response:**
Yes. Three pieces of evidence:

1. **Architecture generalization** (Appendix B.5): CAGP achieves 0.959–0.963 AUROC across DistMult, TransE, and ComplEx on FB15k-237—near-identical performance.
2. **Theorem 1 is architecture-agnostic:** The impossibility of entity-level variance detecting novel contexts holds for *any* model where σ²(e) depends only on entity e, regardless of scoring function.
3. **RelCondVar ablation** (§5): Replacing entity-level variance with learned σ²(e,r) = MLP([e;r]) matches CAGP, confirming the *framework* (structural + semantic decomposition) drives gains, not the specific base model.

We acknowledge R-GCN/CompGCN comparison as future work (§6 Limitations).

---

## Q5: "Only 3 seeds with std=0.000 — is this robust?"

**Response:**
The zero variance is itself informative:

1. **Coverage-based methods** (CoverageOnly, CAGP, RelCondVar) yield std=0.000 because the coverage matrix c(e,r) is a deterministic function of training data—it does not depend on random initialization.
2. **Score-based baselines** show non-zero variance (UKGE: ±0.02, Energy: ±0.01), confirming seeds do affect stochastic methods while our structural signal is invariant.
3. **ICEWS14 strict split** (Table 9): CAGP 0.995±0.000 over 3 seeds after removing 58.5% of test triples.
4. We can run 5+ seeds if requested, but additional seeds will produce identical results for the same reason.

---

## Q6: "ICEWS14 is only one temporal KG. How do you know this generalizes?"

**Response:**
We acknowledge this limitation (§6). However:

1. **Four benchmarks** already tested: WN18RR (11 rel), FB15k-237 (237 rel), YAGO3-10 (37 rel), ICEWS14 (230 rel)—spanning sparse-to-dense relation structures.
2. **Consistent pattern across all 4**: Coverage-based detection achieves 0.76–0.99 structural AUROC; baselines achieve 0.38–0.60.
3. **ICEWS14 is the hardest test**: It breaks definitional coupling, has 53% inverse overlap, and uses ground-truth timestamps. CAGP performs *better* after strict decontamination.
4. **The theoretical contribution** (Theorems 1–2) is dataset-independent.

GDELT and Wikidata-temporal evaluation would strengthen empirical breadth and is planned for the camera-ready.

---

## Q7: "Why not compare with R-GCN or other GNN-based relation-aware models?"

**Response:**
This is a fair gap we acknowledge in §6 Limitations. Three arguments why R-GCN comparison, while desirable, does not undermine our claims:

1. **Theorem 1 applies to GNN-derived uncertainty:** R-GCN produces entity-level embeddings via relation-specific message-passing. Uncertainty derived from these embeddings remains *entity-level*—it cannot signal the *absence* of an observation for a specific (e,r) pair that was never seen. Message-passing aggregates existing neighborhood information; it does not create a signal for missing observations.

2. **RelCondVar serves as a relation-aware proxy:** Our RelCondVar ablation learns σ²(e,r) = MLP([e;r])—a function that has explicit access to relation-specific variance, similar to what R-GCN's relation-specific transformations could provide. RelCondVar matches U_str across all datasets (WN18RR: both 0.86; FB15k-237: both 0.94; ICEWS14: 0.99), confirming that even with learned relation-conditioned variance, explicit coverage remains necessary.

3. **The contribution is diagnostic, not competitive:** We do not claim CAGP outperforms all possible architectures. We identify that *any* entity-level uncertainty method—including GNN-based ones—cannot detect novel contexts (Theorem 1). Full R-GCN validation remains future work, but the theoretical prediction is clear.

Our current comparison set spans score-based (UKGE, Energy), ensemble (MC Dropout, Deep Ensemble), distance-aware (SNGP), and relation-conditioned (RelCondVar) methods.
