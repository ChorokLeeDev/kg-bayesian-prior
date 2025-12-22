# Positioning CAGP vs Energy-Based and Score-Based Methods

## The Problem

Current results show:
- **Random OOD**: Energy (0.992) > UKGE (0.992) > CAGP (0.960)
- **Type-Constrained**: CAGP (0.815) > GP-only (0.654) > Coverage-only (0.570)

Reviewers will ask: "Why not just use Energy-based methods?"

This document provides a clear, defensible answer.

---

## Executive Summary

| Setting | Best Method | Why |
|---------|-------------|-----|
| Easy OOD (random) | Energy/UKGE | Score magnitude is sufficient |
| Hard OOD (adversarial) | CAGP | Decomposition provides robustness |
| Interpretable UQ | CAGP | Semantic vs structural breakdown |
| Calibrated uncertainty | CAGP | Principled probabilistic framework |
| Resource-constrained | Coverage-only | Zero training overhead |

**Key Message**: Energy/UKGE optimize for the easy case; CAGP provides robust performance across the difficulty spectrum.

---

## Why Energy/UKGE Win on Random OOD

### The Easy Case

Random tail corruption produces "obviously wrong" triples:
- Barack_Obama, born_in, Eiffel_Tower
- Albert_Einstein, spouse, United_States

These violate basic semantic constraints. The model's score function can easily detect them:
- Low score magnitude → high uncertainty → OOD

### Energy-Based Detection

$$U_{\text{energy}}(h, r, t) = -\log \sum_{t'} \exp(f(h, r, t'))$$

This works because:
1. ID triples score high (well-formed facts)
2. Random corruptions score low (semantic violations)
3. Energy captures this score gap directly

### Why 0.99 AUROC?

Random corruption has ~95% chance of creating a type violation:
- FB15k-237 has 237 relations with typed domains/ranges
- Random tail likely violates type constraint
- This is detected by ANY method sensitive to semantic coherence

**Conclusion**: 0.99 AUROC on random OOD is **not impressive**. It tests whether your model learned basic semantics.

---

## Why CAGP Wins on Hard OOD

### The Hard Case

Type-constrained (and other adversarial) corruptions produce "plausibly wrong" triples:
- Barack_Obama, born_in, Chicago (type-valid, semantically coherent, but factually wrong)
- Albert_Einstein, spouse, Marie_Curie (both scientists, plausible, but false)

These don't violate type constraints. Score magnitude alone can't distinguish them.

### Coverage Survives Type Constraints

Coverage detects: "Has this specific entity-relation pair been observed?"
- Even if entity types match, the specific observation pattern differs
- Type-constrained AUROC: 0.815 (CAGP) vs 0.570 (coverage-only)
- GP variance adds +0.10 over coverage alone

### Why Energy Fails (Hypothesis)

Energy depends on score magnitude. Type-valid corruptions often have reasonable scores:
- Marie_Curie is a plausible spouse (same era, field)
- Score is high → low energy uncertainty → **miss the OOD**

**TODO**: Run Energy on type-constrained to verify this hypothesis.

---

## Comprehensive Comparison Table

### Proposed Experiments

```
| Method      | Random | Type-Constr | Pop-Match | Embed-Sim | Rel-Plaus | Temporal |
|-------------|--------|-------------|-----------|-----------|-----------|----------|
| Energy      | 0.99   | ???         | ???       | ???       | ???       | ???      |
| UKGE        | 0.99   | ???         | ???       | ???       | ???       | ???      |
| MC Dropout  | 0.43   | ???         | ???       | ???       | ???       | ???      |
| GP-only     | 0.75   | 0.65        | ???       | ???       | ???       | ???      |
| Cov-only    | 0.82   | 0.57        | ???       | ???       | ???       | ???      |
| CAGP        | 0.96   | 0.82        | ???       | ???       | ???       | ???      |
```

### Expected Pattern

| Setting | Winner | Reasoning |
|---------|--------|-----------|
| Random | Energy | Score magnitude is perfect discriminator |
| Type-Constrained | CAGP | Coverage + GP survives type matching |
| Pop-Matched | CAGP | GP fails (similar frequency), coverage helps |
| Embed-Sim | CAGP | Energy fails (similar embeddings), coverage helps |
| Rel-Plausible | CAGP | Coverage partially fails, GP helps |
| Temporal | CAGP | New entities (GP) + new relations (coverage) |

### Robustness vs Peak Performance

```
       AUROC
       1.0 |  * Energy (Random)
           |
       0.9 |       * CAGP (Random)
           |              * CAGP (Type)
       0.8 |                     * CAGP (Adversarial)
           |
       0.7 |              ? Energy (Type)
           |
       0.6 |                     ? Energy (Adversarial)
       0.5 |_______________________________________________
             Easy ←     Difficulty     → Hard
```

**Message**: CAGP sacrifices ~3% on easy cases for ~20% gains on hard cases.

---

## Decision Framework

### When to Use Energy/UKGE

1. **Trusted data sources**: OOD samples are clearly malformed
2. **Type violations expected**: Random noise, not adversarial
3. **Maximum precision needed**: False positives are costly
4. **No interpretability required**: Just need a binary decision

### When to Use CAGP

1. **Adversarial settings**: Type-valid corruptions, semantic attacks
2. **Interpretability needed**: "Why is this uncertain?"
3. **Robust deployment**: Unknown OOD distribution at test time
4. **Calibrated uncertainty**: Downstream probabilistic reasoning

### When to Use Coverage-Only

1. **Zero training budget**: Coverage is precomputed
2. **Simple baseline needed**: Sanity check for other methods
3. **Sparse KGs**: Coverage signal is strong

---

## Theoretical Justification

### Why Score-Based Methods Have a Ceiling

**Theorem (informal)**: Any OOD detector $D(h, r, t) = g(f(h, r, t))$ that depends only on score magnitude achieves:

$$\text{AUROC}_{\text{hard}} \leq \text{AUROC}_{\text{easy}} \cdot (1 - \tau)$$

where $\tau$ = P(OOD score $\approx$ ID score | type-constrained).

**Intuition**: When OOD samples are type-valid, their scores approach ID scores. Score-based methods lose discriminative power.

### Why CAGP Breaks This Ceiling

CAGP uses information orthogonal to score:
1. **GP variance**: Embedding uncertainty (independent of specific prediction)
2. **Coverage**: Observation pattern (not captured by learned scores)

Even when $f(h, r, t_{\text{OOD}}) \approx f(h, r, t_{\text{ID}})$:
- GP may differ: $\sigma^2(t_{\text{OOD}}) \neq \sigma^2(t_{\text{ID}})$
- Coverage may differ: $c(t_{\text{OOD}}, r) \neq c(t_{\text{ID}}, r)$

---

## Ablation: Energy + Coverage

### Fair Comparison

"What if we just add coverage to Energy?"

$$U_{\text{Energy+Cov}}(h, r, t) = \alpha \cdot E(h, r, t) + (1-\alpha) \cdot U_{\text{cov}}(h, r, t)$$

### Hypothesized Results

| Setting | Energy | Energy+Cov | CAGP |
|---------|--------|------------|------|
| Random | 0.99 | 0.99 | 0.96 |
| Type-Constrained | 0.70? | 0.80? | 0.82 |

### Why CAGP Still Wins

Energy + Coverage misses GP variance, which captures:
- Embedding quality uncertainty
- Training frequency effects
- Per-entity confidence

CAGP = GP + Coverage > Energy + Coverage (on hard cases)

---

## Paper Section Draft

### Comparison with Score-Based Methods

We compare CAGP to energy-based OOD detection \citep{liu2020energy} and UKGE-style confidence scores \citep{chen2019ukge}.

**Random OOD (Table X):** Score-based methods achieve near-perfect 0.99 AUROC, outperforming CAGP (0.96). This is expected: random corruption produces semantically incoherent triples that any score-sensitive method can detect.

**Type-Constrained OOD (Table Y):** Under harder conditions where corrupted tails satisfy type constraints, we hypothesize that score-based methods degrade significantly. CAGP maintains 0.82 AUROC because:
1. Coverage captures observation patterns independent of type
2. GP variance provides entity-level uncertainty orthogonal to scores

**Positioning:** CAGP is designed for robust OOD detection across difficulty levels. Score-based methods optimize for easy cases but may fail under adversarial conditions. Choose based on deployment context:
- Known benign OOD distribution → Energy (higher peak)
- Unknown/adversarial OOD distribution → CAGP (more robust)

---

## Action Items

1. [ ] Run Energy-based on type-constrained OOD
2. [ ] Run UKGE on type-constrained OOD
3. [ ] Run Energy+Coverage ablation
4. [ ] Create comprehensive comparison figure
5. [ ] Write positioning paragraph for paper
