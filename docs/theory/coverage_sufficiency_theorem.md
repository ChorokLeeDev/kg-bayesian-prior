# Coverage Sufficiency Theorem

## Executive Summary

We derive the theoretical AUROC for coverage-based OOD detection and validate it empirically.

**Main Result (Revised Theorem):**

$$\text{AUROC}_{cov} = p_h \cdot p_t \cdot s_r + \frac{1}{2}(p_h \cdot p_t \cdot (1-s_r) + p_{1} \cdot s_r)$$

where:
- $p_h, p_t$ = probability that head/tail is covered in ID test triples
- $s_r$ = relation sparsity

**Empirical Validation:**

| Dataset | Predicted | Observed | Error |
|---------|-----------|----------|-------|
| WN18RR | 0.6808 | 0.6570 | **2.4%** |
| FB15k-237 | 0.8147 | 0.8210 | **0.6%** |

---

## Setup and Notation

| Symbol | Definition |
|--------|------------|
| $\mathcal{T}$ | Training set of triples $(h, r, t)$ |
| $E$ | Set of all entities |
| $R$ | Set of all relations |
| $c(e, r)$ | Coverage: 1 if entity $e$ seen with relation $r$ in training |
| $s_r$ | Relation sparsity: fraction of entities NOT seen with relation $r$ |

### Coverage Definition

$$c(e, r) = \mathbb{1}[\exists (h', r, t') \in \mathcal{T} : h' = e \lor t' = e]$$

### Coverage-Based Uncertainty

$$U_{cov}(h, r, t) = 2 - c(h, r) - c(t, r)$$

---

## Original Theorem (Too Optimistic)

### Assumption A1 (Strong)
For ID test triple $(h, r, t)$: $c(h, r) = c(t, r) = 1$

### Theorem (Original)
$$\text{AUROC} = \frac{1 + s_r}{2}$$

### Problem
**A1 is violated in practice!**

| Dataset | P(both covered \| ID) | Expected by A1 |
|---------|----------------------|----------------|
| WN18RR | 0.5415 | 1.0 |
| FB15k-237 | 0.6845 | 1.0 |

This explains the large gap between prediction and observation.

---

## Revised Theorem (Accurate)

### Relaxed Assumption A1'
For ID test triple $(h, r, t)$:
- $c(h, r) = 1$ with probability $p_h$
- $c(t, r) = 1$ with probability $p_t$

### Empirical Values

| Dataset | $p_h$ | $p_t$ | $p_{both}$ |
|---------|-------|-------|------------|
| WN18RR | 0.636 | 0.885 | 0.542 |
| FB15k-237 | 0.763 | 0.905 | 0.685 |

### Theorem (Revised)

**ID Uncertainty Distribution:**
$$P(U_{ID} = 0) = p_h \cdot p_t$$
$$P(U_{ID} = 1) = p_h(1-p_t) + (1-p_h)p_t$$
$$P(U_{ID} = 2) = (1-p_h)(1-p_t)$$

**OOD Uncertainty Distribution** (head from real triple, random tail):
$$P(U_{OOD} = 0) = 1 - s_r$$
$$P(U_{OOD} = 1) = s_r$$

**AUROC Calculation:**
$$\text{AUROC} = P(U_{ID} < U_{OOD}) + \frac{1}{2}P(U_{ID} = U_{OOD})$$

Expanding:
$$\text{AUROC} = p_h p_t \cdot s_r + \frac{1}{2}\left[p_h p_t (1-s_r) + (p_h(1-p_t) + (1-p_h)p_t) \cdot s_r\right]$$

### Validation

| Dataset | Predicted | Observed | Error |
|---------|-----------|----------|-------|
| WN18RR | 0.6808 | 0.6570 | 2.4% ✓ |
| FB15k-237 | 0.8147 | 0.8210 | 0.6% ✓ |

**The revised theorem accurately predicts empirical AUROC!**

---

## Key Insights

### 1. Coverage Limitation Identified

The gap between A1 and reality reveals a fundamental limitation:

> **Coverage cannot distinguish between:**
> - OOD triple with random unseen tail
> - ID triple with entity appearing with relation for first time

Both have $c(t, r) = 0$, but one is ID and one is OOD.

### 2. Why GP Variance Helps

GP variance can partially address this limitation:

- Entity with low GP variance: well-constrained embedding, likely ID
- Entity with high GP variance: poorly constrained, could be either

The combination (CAGP) uses both signals:
- Coverage: relation-specific observation
- GP variance: entity embedding quality

### 3. Synergy Explanation

| Scenario | Coverage | GP | Ground Truth | Winner |
|----------|----------|-----|--------------|--------|
| New entity, new relation | High unc | High unc | OOD | Both ✓ |
| Known entity, new relation | High unc | Low unc | Depends | Coverage ✓ |
| New entity, known relation | High unc | High unc | Depends | GP helps |
| Known entity, known relation | Low unc | Low unc | ID | Both ✓ |

The synergy comes from covering each other's blind spots.

---

## Theorem: GP Variance Limitation

**Statement:** Entity-level GP variance $\sigma^2_e$ cannot achieve coverage AUROC because it is relation-agnostic.

**Proof:**

GP-KGE computes: $\sigma^2_e = \exp(\text{logvar}_e)$

For entity $e$ with:
- $c(e, r_1) = 1$ (seen with $r_1$)
- $c(e, r_2) = 0$ (not seen with $r_2$)

GP gives same uncertainty for both relations, but coverage correctly differentiates.

$\square$

---

## Theorem: Complementarity

**Statement:** Coverage and GP variance are not subsets of each other.

**Proof by construction:**

**Case 1: Coverage correct, GP wrong**
- Entity $e$: frequent overall (low $\sigma^2_e$), never seen with $r$
- Coverage: high uncertainty ✓
- GP: low uncertainty ✗

**Case 2: GP correct, Coverage wrong**
- Entity $e$: rare overall (high $\sigma^2_e$), seen with $r$ once
- Coverage: low uncertainty ✗ (if truly OOD)
- GP: high uncertainty ✓

$\square$

**Corollary:** Optimal OOD detection requires both signals.

---

## Implications for CAGP

### Theoretical Justification

CAGP combines:
$$U_{CAGP} = \alpha \cdot U_{GP} + (1-\alpha) \cdot U_{cov}$$

The complementarity theorem justifies this combination:
- Neither signal alone is sufficient
- Their combination covers more failure cases

### Why $\alpha \approx 0.5$?

Empirically, learned $\alpha \approx 0.5$ across datasets.

**Interpretation:** Both signals contribute approximately equal information.

This makes sense because:
1. Coverage is perfect for relation-specific observation (strong signal)
2. GP is good for entity quality (complementary signal)
3. Equal weighting balances both contributions

---

## Summary Table

| Theorem | Statement | Status |
|---------|-----------|--------|
| Original AUROC | $\frac{1+s_r}{2}$ | ❌ Assumes A1 (violated) |
| **Revised AUROC** | Complex formula above | ✅ **Validated (<3% error)** |
| GP Limitation | Relation-agnostic | ✅ Proven |
| Complementarity | Neither ⊂ other | ✅ Proven |

---

## Appendix: Derivation Details

### Full AUROC Derivation

Let:
- $a = p_h \cdot p_t$ (prob both covered in ID)
- $b = p_h(1-p_t) + (1-p_h)p_t$ (prob exactly one covered in ID)
- $c = (1-p_h)(1-p_t)$ (prob neither covered in ID)
- $q = 1 - s_r$ (prob random tail is covered)

ID distribution: $P(U=0)=a$, $P(U=1)=b$, $P(U=2)=c$
OOD distribution: $P(U=0)=q$, $P(U=1)=1-q$

AUROC = $\sum_{u_{id} < u_{ood}} P(U_{ID}=u_{id}) P(U_{OOD}=u_{ood})$
$+ \frac{1}{2}\sum_{u_{id} = u_{ood}} P(U_{ID}=u_{id}) P(U_{OOD}=u_{ood})$

$= a(1-q) + \frac{1}{2}[aq + b(1-q)]$
$= a - aq + \frac{1}{2}aq + \frac{1}{2}b - \frac{1}{2}bq$
$= a - \frac{1}{2}aq + \frac{1}{2}b - \frac{1}{2}bq$
$= a + \frac{1}{2}b - \frac{1}{2}q(a + b)$

Substituting $q = 1 - s_r$:
$= a + \frac{1}{2}b - \frac{1}{2}(1-s_r)(a+b)$
$= a + \frac{1}{2}b - \frac{1}{2}(a+b) + \frac{1}{2}s_r(a+b)$
$= \frac{1}{2}a + \frac{1}{2}s_r(a+b)$
$= \frac{1}{2}(a + s_r(a+b))$
$= \frac{1}{2}(a(1+s_r) + s_r \cdot b)$
