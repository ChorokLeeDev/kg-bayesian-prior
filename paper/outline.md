# The Diversity Trap: Why Models Fail on What They've Seen in Many Contexts

## Paper Outline

**Target**: ICML/NeurIPS 2026 Main Track

---

## Title Options
1. **"The Diversity Trap: Why Models Fail on What They've Seen in Many Contexts"** (Primary)
2. "Less Context, More Accuracy: Diversity-Induced Embedding Dilution"
3. "When Familiarity Breeds Confusion: A Cross-Domain Analysis"

---

## Abstract

Neural embeddings are assumed to improve with more training exposure. We show this assumption fails when exposure is *diverse*: entities seen across many contexts develop diluted representations that lose specificity. In knowledge graphs, entities with full relational coverage achieve only 32.3% accuracy vs 59.5% for partial coverage—a 27 percentage point gap. We find the same pattern in BERT (25pp gap on factual recall) but not in single-relation settings like collaborative filtering, where diversity of *preferences* (not frequency) causes dilution. We formalize this as Diversity-Induced Dilution, prove embedding geometry bounds, and provide practical guidelines: track context diversity, not just frequency.

---

## 1. Introduction

**Hook**: The assumption that "more training data leads to better representations" is foundational to deep learning. We show this assumption fails in a specific, predictable way.

**Paradox**: In knowledge graphs, queries where both entities have been extensively trained (full coverage) achieve only 32.3% accuracy, while queries with partial coverage achieve 59.5%—nearly double.

**Thesis**: The culprit is not frequency, but *diversity*. Entities seen in many different contexts develop averaged, diluted embeddings that lose context-specific information.

**Contributions**:
1. Empirical discovery of the Diversity Trap across three domains
2. Theoretical formalization via embedding geometry
3. Practical guidelines for diversity-aware uncertainty estimation

---

## 2. The Diversity Trap: Empirical Evidence

### 2.1 Knowledge Graphs (FB15k-237)

**Setup**: Link prediction task, measuring Hits@10 by coverage type

**Results**:
| Coverage Type | Definition | Hits@10 |
|---------------|------------|---------|
| Full Coverage | Both entities seen with relation | 32.3% |
| Partial Zero | One entity seen, one not | **59.5%** |
| Full Zero | Neither entity seen | 14.8% |

**Controls**:
- Frequency matching: +9.4pp effect persists (95% CI: [8.0, 10.8])
- Relation type: Effect strongest on common relations (+30pp)

**Mechanism**: Entities connected to many relations have diluted embeddings

### 2.2 Language Models (BERT/LAMA)

**Setup**: Factual recall probes (e.g., "The capital of [X] is [MASK]")

**Results**:
| Entity Frequency | Accuracy |
|------------------|----------|
| High (Tier 1-2) | 75.0% |
| Low (Tier 3-5) | **100.0%** |
| Gap | **25pp** |

**Example**: 
- Germany (high-freq) → "Bonn" (wrong, historical confusion)
- Latvia (low-freq) → "Riga" (correct, high confidence)

**Mechanism**: High-frequency entities appear in diverse contexts, diluting factual associations

### 2.3 Recommender Systems (MovieLens)

**Setup**: Rating prediction, stratified by user activity and item popularity

**Initial finding**: Opposite pattern! Heavy users have LOWER error (MAE 0.71 vs 0.95)

**Key insight**: Single-relation setting (only ratings). Test with *diversity* instead:

| Rating Diversity | MAE |
|------------------|-----|
| Low (std < 0.8) | **0.54** |
| High (std > 1.2) | **0.98** |

**Conclusion**: Frequency alone doesn't cause dilution. Diversity does.

---

## 3. Theory: Diversity-Induced Dilution

### 3.1 Embedding Geometry Analysis

**Intuition**: An entity embedding is pulled toward the centroid of all its training contexts.

**Definition** (Context Diversity): For entity e, diversity D(e) = number of distinct contexts (relations, sentence types, etc.) in which e appears.

**Theorem 1** (Dilution Bound): For an entity e with diversity D(e) = k, the distance from its embedding to any context-specific optimum e_r* satisfies:
```
||e - e_r*|| >= ε · sqrt(k)
```
where ε depends on context dissimilarity.

**Proof sketch**: Each context pulls the embedding in a different direction. With k contexts, the embedding converges to a weighted centroid, distance from any specific optimum grows with sqrt(k).

### 3.2 Why Diversity Hurts

1. **Multi-context averaging**: Embedding represents "average" behavior, not specific
2. **Information loss**: Context-specific signals cancel out
3. **Overconfidence**: Model has seen entity often, assumes it "knows" it

### 3.3 Calibration Failure

**Observation**: High-diversity entities have:
- High confidence (seen often)
- Low accuracy (diluted embedding)

This is the Diversity Trap: **confidence scales with frequency, but accuracy scales with specificity**.

---

## 4. Practical Implications

### 4.1 Uncertainty Estimation

**Current practice**: Flag low-frequency/low-coverage as uncertain

**Our recommendation**: Flag high-diversity as uncertain

**Separation of concerns**:
- Coverage/frequency → OOD detection (has model seen this?)
- Diversity → Confidence calibration (can model be specific?)

### 4.2 Guidelines

1. **Track context diversity**, not just frequency
2. **Stratify evaluation** by diversity level
3. **Consider context-specific embeddings** for high-diversity entities
4. **Calibrate differently** for high vs low diversity

---

## 5. Related Work

- **Popularity bias** (RecSys): Known that popular items are over-recommended, but mechanism differs
- **Long-tail recognition** (CV): Head classes confused, tail classes distinct—same pattern
- **Calibration**: Guo et al. (2017) on modern neural network overconfidence
- **KG uncertainty**: UKGE, BEUrRE—entity-level variance misses diversity

---

## 6. Conclusion

We identified the Diversity Trap: a counter-intuitive phenomenon where models fail on what they've seen most—when that exposure is diverse. This is not a bug but a structural property of shared embeddings.

**Key insight**: More context ≠ better embedding. Diversity dilutes.

**Practical impact**: Uncertainty estimation should track diversity, not just frequency. The confident predictions on high-diversity entities are the ones most likely to be wrong.

---

## Figures

1. **Figure 1**: Coverage Paradox bar chart (KG)
2. **Figure 2**: BERT frequency vs accuracy scatter
3. **Figure 3**: MovieLens: frequency vs diversity 2D heatmap
4. **Figure 4**: Embedding geometry illustration (centroid convergence)
5. **Table 1**: Cross-domain comparison

---

## Appendix

- A. Full experimental details
- B. Theorem proof
- C. Additional datasets (WN18RR, YAGO3-10)
- D. Calibration analysis

---

## TODO

- [ ] Formalize Theorem 1 with full proof
- [ ] Create figures
- [ ] Write full paper draft
- [ ] Additional datasets for robustness
- [ ] ImageNet-LT validation (optional)
