# Information-Theoretic Decomposition of KG Uncertainty

## Overview

This document formalizes the decomposition of uncertainty for OOD detection in knowledge graphs using information theory. The key insight is that the mutual information between uncertainty and OOD status can be decomposed into semantic and structural components.

## Notation

- $U$: Combined uncertainty signal
- $U_{\text{sem}}$: Semantic uncertainty (entity variance)
- $U_{\text{str}}$: Structural uncertainty (coverage)
- $Y$: OOD indicator ($Y=1$ for OOD, $Y=0$ for ID)
- $Y_e$: Emerging-entity indicator
- $Y_n$: Novel-context indicator
- $\rho$: Coverage overlap (fraction of emerging entities with coverage for query relation)

## Main Result: Information Decomposition

**Theorem (Information-Theoretic Decomposition):**

The mutual information between combined uncertainty and OOD status decomposes as:

$$I(U; Y) = I(U_{\text{str}}; Y) + I(U_{\text{sem}}; Y | U_{\text{str}}) + \epsilon_{\text{interaction}}$$

where:
- $I(U_{\text{str}}; Y)$ is the structural information (captures novel contexts perfectly)
- $I(U_{\text{sem}}; Y | U_{\text{str}})$ is the conditional semantic information (captures emerging entities within coverage strata)
- $\epsilon_{\text{interaction}}$ is the interaction term (typically negligible)

## Proof Sketch

### Step 1: Chain Rule for Mutual Information

By the chain rule:
$$I(U_{\text{sem}}, U_{\text{str}}; Y) = I(U_{\text{str}}; Y) + I(U_{\text{sem}}; Y | U_{\text{str}})$$

### Step 2: Structural Information

For novel contexts, $U_{\text{str}} \geq 1$ (by definition of coverage), while ID has $U_{\text{str}} = 0$.
Therefore:
$$I(U_{\text{str}}; Y_n) = H(Y_n) \quad \text{(perfect separation)}$$

For emerging entities with partial coverage ($\rho > 0$):
$$I(U_{\text{str}}; Y_e) < H(Y_e) \quad \text{(imperfect separation)}$$

### Step 3: Conditional Semantic Information

Given $U_{\text{str}} = 0$ (entities have coverage), semantic uncertainty provides additional discrimination:
$$I(U_{\text{sem}}; Y_e | U_{\text{str}} = 0) > 0 \quad \text{iff } \rho > 0$$

When $\rho = 0$ (no coverage overlap), all emerging entities have $U_{\text{str}} > 0$, so:
$$I(U_{\text{sem}}; Y_e | U_{\text{str}}) = 0$$

This explains why semantic provides no gain on ICEWS14/18: $\rho \approx 0$ on temporal KGs.

## Key Prediction: When Semantic Matters

**Corollary (Semantic Necessity Condition):**

The semantic component provides positive information gain iff:
$$\rho \cdot \Delta_{\text{sem}} > 0$$

where:
- $\rho = P(U_{\text{str}} = 0 | Y_e = 1)$ is the coverage overlap for emerging entities
- $\Delta_{\text{sem}} = \mathbb{E}[U_{\text{sem}} | Y_e = 1, U_{\text{str}} = 0] - \mathbb{E}[U_{\text{sem}} | Y = 0]$ is the semantic separation

**Interpretation:**
- When $\rho = 0$: All emerging entities have $U_{\text{str}} > 0$, structural signal suffices
- When $\rho > 0$: Some emerging entities have $U_{\text{str}} = 0$, need semantic tiebreaker
- When $\Delta_{\text{sem}} = 0$: Semantic doesn't separate within the $U_{\text{str}} = 0$ stratum

## Empirical Predictions

| Dataset | $\rho$ | Predicted Semantic Gain | Observed |
|---------|--------|-------------------------|----------|
| WN18RR | 0.34 | +ve | +8pp |
| FB15k-237 | 0.43 | +ve | +11pp |
| YAGO3-10 | 0.66 | +ve | +12pp |
| ICEWS14 | ~0 | ~0 | +2pp (ceiling) |
| ICEWS18 | ~0 | ~0 | 0pp |

The theory correctly predicts that semantic gain is concentrated on static benchmarks (high $\rho$) and negligible on temporal benchmarks (low $\rho$).

## Connection to AUROC

The mutual information decomposition directly implies the mixture-AUROC identity (Eq. 6 in paper):

$$\text{AUROC}(U) = \pi_e \cdot A_e(U) + \pi_n \cdot A_n(U) + \delta_{\text{tie}}$$

where:
- $A_n(U_{\text{str}}) = 1$ (perfect structural separation of novel contexts)
- $A_e(U_{\text{comb}}) > A_e(U_{\text{str}})$ when $\rho > 0$ and $\Delta_{\text{sem}} > 0$

## Theorem Statement for Paper

**Theorem 2 (Information-Theoretic Characterization):**

Let $\rho = P(U_{\text{str}} = 0 | Y_e = 1)$ be the coverage overlap. Then:

(i) $I(U_{\text{str}}; Y_n) = H(Y_n)$ (structural captures all novel-context information)

(ii) $I(U_{\text{sem}}; Y_n | U_{\text{str}}) = 0$ (semantic adds no information for novel contexts)

(iii) $I(U_{\text{sem}}; Y_e | U_{\text{str}}) > 0$ iff $\rho > 0$ and $\Delta_{\text{sem}} > 0$

**Implication:** Semantic uncertainty is necessary iff emerging entities have non-trivial coverage overlap ($\rho > 0$). On temporal KGs where $\rho \approx 0$, structural uncertainty alone is sufficient.

## Proof of Theorem 2

### Part (i): Structural captures novel contexts

By definition, novel contexts have $c(h,r) = 0$ or $c(t,r) = 0$, so $U_{\text{str}} \geq 1$.
ID has $U_{\text{str}} = 0$ by construction.
Therefore $Y_n$ is a deterministic function of $U_{\text{str}}$:
$$Y_n = \mathbf{1}[U_{\text{str}} \geq 1 \land \min(\text{freq}(h), \text{freq}(t)) > \tau]$$
This implies $H(Y_n | U_{\text{str}}) = 0$, so $I(U_{\text{str}}; Y_n) = H(Y_n)$.

### Part (ii): Semantic adds nothing for novel contexts

Given $U_{\text{str}} \geq 1$, we already know $Y_n = 1$ (for high-frequency entities).
Therefore $H(Y_n | U_{\text{str}}, U_{\text{sem}}) = H(Y_n | U_{\text{str}}) = 0$.
By the data processing inequality: $I(U_{\text{sem}}; Y_n | U_{\text{str}}) = 0$.

### Part (iii): Semantic helps on emerging entities iff $\rho > 0$

When $\rho = 0$: All emerging entities have $U_{\text{str}} > 0$, so $H(Y_e | U_{\text{str}}) = 0$.
When $\rho > 0$: Some emerging entities have $U_{\text{str}} = 0$ (same as ID).
In this stratum, $U_{\text{sem}}$ provides discrimination if $\Delta_{\text{sem}} > 0$.

The conditional mutual information is:
$$I(U_{\text{sem}}; Y_e | U_{\text{str}} = 0) = H(Y_e | U_{\text{str}} = 0) - H(Y_e | U_{\text{str}} = 0, U_{\text{sem}})$$

This is positive when $U_{\text{sem}}$ reduces uncertainty about $Y_e$ within the $U_{\text{str}} = 0$ stratum, which happens iff emerging entities have systematically higher $U_{\text{sem}}$ than ID (i.e., $\Delta_{\text{sem}} > 0$).

## Generalization to Other Domains

The decomposition framework generalizes beyond KGs:

### Retrieval-Augmented Generation (RAG)
- $U_{\text{str}}$: Has this document been retrieved for similar queries? (co-occurrence)
- $U_{\text{sem}}$: How well does the query embedding match the document? (relevance)
- OOD: Query-document pairs never seen in training

### Recommendation Systems
- $U_{\text{str}}$: Has this user interacted with this item type? (coverage)
- $U_{\text{sem}}$: User-item embedding similarity (preference)
- OOD: User-item pairs in novel categories

The impossibility theorem applies in these domains too: any user/query-level uncertainty is blind to item/document-specific novelty.

## Conclusion

The information-theoretic decomposition provides:
1. **Formal justification** for why structural uncertainty is necessary
2. **Quantitative prediction** of when semantic uncertainty helps ($\rho > 0$)
3. **Unified framework** applicable beyond KGs to RAG and recommendations

This strengthens the theoretical contribution from "impossibility + empirical validation" to "impossibility + information-theoretic characterization + cross-domain generalization."
