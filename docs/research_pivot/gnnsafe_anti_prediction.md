# GNNSafe Anti-Prediction Investigation

## Status: HYPOTHESIS REFINED

**Date**: 2026-03-06
**Purpose**: Validate whether GNNSafe anti-prediction is a real phenomenon before pivoting thesis

---

## Key Finding (2026-03-06)

**The anti-prediction is NOT about relation count or freq-cov correlation.**

**It's about whether novel-context entities are higher or lower frequency than ID entities!**

| Dataset | Novel ctx freq vs ID | GNNSafe AUROC | Prediction |
|---------|---------------------|---------------|------------|
| WN18RR | 2.80x HIGHER | 0.79 | Should anti-predict but doesn't! |
| FB15k-237 | 4.33x HIGHER | 0.43 | Anti-predicts (as expected) |
| ICEWS14 | 0.59x LOWER | 0.73 | Works (as expected) |

**Puzzle**: WN18RR has novel-ctx freq 2.8x higher than ID, yet GNNSafe works (0.79).
This contradicts our simple "high freq → low energy → wrong prediction" theory.

### Possible Explanations for WN18RR Exception:

1. **Sparse relations (11)**: With few relations, even high-freq entities can't be too confident
2. **Small novel context set (340)**: Statistical noise in small sample
3. **Different graph structure**: WordNet has specific semantic patterns

---

## What We've Found So Far

### Empirical Results

| Dataset | Relations | Novel-ctx Freq/ID Freq | GNNSafe Novel-ctx AUROC |
|---------|-----------|------------------------|-------------------------|
| FB15k-237 | 237 | 4.33x | **0.43** (anti-predictive) |
| WN18RR | 11 | 2.80x | **0.79** (works - ANOMALY) |
| ICEWS14 | 222 | 0.59x | **0.73** (works) |
| ICEWS18 | 251 | ~0.6x* | **0.72** (works) |

*estimated from similar structure

### Hypothesized Mechanism

```
Dense-relation KG (many relations):
  - Entity can be high-frequency but low-coverage
  - Frequency ≠ Coverage (decoupled)
  - GNNSafe uses frequency signal → wrong for novel contexts

Sparse-relation KG (few relations):
  - High-frequency entities see most relations
  - Frequency ≈ Coverage (coupled by accident)
  - GNNSafe works by accident
```

### Key Statistics (FB15k-237)

- Novel context entities: mean neighbors = 166.2
- ID entities: mean neighbors = 71.8
- Novel context has HIGHER connectivity → GNNSafe predicts as ID → WRONG

---

## What Needs Validation

### 1. More Datasets

Need to check:
- [x] FB15k-237 (237 relations) - DONE: 0.43 anti-predictive
- [x] WN18RR (11 relations) - DONE: 0.79 works
- [x] ICEWS14 (222 relations) - DONE: **0.725** (WORKS, not anti-predictive!)
- [ ] YAGO3-10 (37 relations) - TODO
- [ ] Nell-995 (200 relations) - TODO (if available)

**IMPORTANT UPDATE**: ICEWS14 has 222 relations (similar to FB15k-237's 237) but GNNSafe achieves 0.725 AUROC on novel contexts - NOT anti-predictive! This challenges our hypothesis.

### 2. Correlation Analysis

For each dataset, compute:
- Spearman(frequency, coverage_rate)
- Spearman(neighbors, coverage_rate)
- GNNSafe AUROC on novel contexts

**Prediction**: Anti-prediction when frequency-coverage correlation is low

### 3. Controlled Experiment

Create synthetic KGs with varying relation counts:
- Same entity count, same triple count
- Vary only relation count: 10, 50, 100, 200
- Measure GNNSafe AUROC at each point

**Prediction**: Phase transition at some critical relation density

---

## FINAL VERDICT (2026-03-06)

**GNNSafe anti-prediction is REAL but too DATASET-SPECIFIC for thesis pivot.**

### Summary Table

| Dataset | Relations | NC freq/ID freq | GNNSafe Novel-ctx | Explanation |
|---------|-----------|-----------------|-------------------|-------------|
| FB15k-237 | 237 | 4.33x | **0.43** | Anti-predicts (high freq NC) |
| ICEWS14 | 222 | 0.59x | **0.73** | Works (low freq NC) |
| WN18RR | 11 | 2.80x | **0.79** | Special case (uniform low coverage) |

### Why NOT Pivot

1. **Too narrow**: Only reliably anti-predicts on FB15k-237
2. **WN18RR anomaly**: Same freq pattern but different result
3. **Not generalizable**: Can't predict a priori which datasets will anti-predict
4. **Complicated story**: Multi-variable interaction (relations, freq ratio, coverage uniformity)

### What To Do With This Finding

- Keep as supplementary observation in paper (already in Table 2)
- NOT a main contribution
- Could mention in discussion: "GNNSafe anti-predicts on FB15k-237 due to..."

---

## Next Steps

Return to finding other novel angles in the original research:

1. **Information-theoretic decomposition** - Still potentially novel
2. **Evaluation protocol critique** - Strongest angle per AC
3. **α-insensitivity** - Minor finding
4. **Neural coverage unlearning** - Needs more work

**Decision**: The GNNSafe pivot is NOT viable. Continue searching for genuine novelty in original work or pivot to completely different direction.

The original paper focused on:
1. Impossibility theorem for relation-agnostic methods
2. Coverage tracking as solution
3. Semantic + structural decomposition

**Problem**: Reviewers found this "obvious" and "trivial solution"

## New Direction (Under Investigation)

Potential new thesis:
> "Graph-based OOD detection exhibits relation-density phase transition"

**Why this might be novel**:
1. GNNSafe is SOTA for graph OOD - showing it anti-predicts is significant
2. Relation density as critical variable - not previously studied
3. "Works by accident" pattern - explains conflicting prior results
4. Generalizes to other methods - broader implications

**Risks**:
1. Might be specific to our implementation
2. Might not generalize to true GNN (we use MLP proxy)
3. Might be artifact of OOD definition

---

## Next Steps

1. Run GNNSafe on ICEWS14, YAGO3-10
2. Compute frequency-coverage correlations
3. If pattern holds, design controlled synthetic experiment
4. If pattern doesn't hold, investigate why

---

## Decision Point

After sanity checks, decide:
- A) Pivot to "phase transition" thesis if pattern is robust
- B) Stay with original thesis if pattern is weak
- C) Abandon both if neither is strong enough
