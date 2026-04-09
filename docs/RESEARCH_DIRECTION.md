# Research Direction: Familiarity Trap Paper

**Date**: 2026-04-09
**Decision**: Pursue Option 2 (Broad "Familiarity Trap" Paper)

---

## Executive Summary

We discovered a counter-intuitive phenomenon in Knowledge Graphs that appears to generalize across ML domains: **The Familiarity Trap** - models become overconfident on frequently-seen data while their actual accuracy degrades due to embedding dilution.

**Primary Direction**: Broad cross-domain paper targeting ICML/NeurIPS main track

---

## Three Options Considered

### Option 1: KG-focused Paper (Safe)
- **Scope**: Coverage Paradox + Embedding Geometry Theory + Cascading Uncertainty
- **Pros**: All experiments done, lower risk
- **Cons**: Limited impact, "just another KG paper"
- **Target**: Workshop → Main track
- **Status**: Backup plan

### Option 2: Broad "Familiarity Trap" Paper (Chosen)
- **Scope**: KG + MovieLens + BERT verification, universal theory
- **Pros**: High impact, novel cross-domain insight
- **Cons**: More experiments needed, higher risk
- **Target**: ICML/NeurIPS main track
- **Status**: **Primary direction**

### Option 3: Position + Method Split
- **Scope**: Two separate papers
- **Pros**: Hedges bets, two publications
- **Cons**: Dilutes the story, more work
- **Target**: Position track + Main track
- **Status**: Fallback if Option 2 reviewers want split

---

## Exploration Results Summary

### Theory Exploration (Option A)
**File**: `docs/theory_exploration.md`

**Selected Direction**: Embedding Geometry
- Dilution-Specificity Trade-off: High-degree entities → center-of-mass embedding → loss of relation-specific info
- Proposed Theorem: `||e - e_r*|| >= ε * sqrt(k)` for degree-k entity
- Calibration gap formally derivable

**Cross-insights**:
- Method needs relation-specific embedding layers
- Broader impact: embedding sharing is structural limitation

### Method Exploration (Option B)
**File**: `docs/method_exploration.md`

**Key Findings**:
| Direction | OOD AUROC | Sel. MRR | Verdict |
|-----------|-----------|----------|---------|
| Coverage-aware Calibration | 0.36 | +48.1% | Calibration only |
| Anchor-based Prediction | 0.47 | +57.2% | Best for selective pred |
| Disentangled Embeddings | 0.98 | -51.8% | Failed |
| Cascading Uncertainty | 1.00 | -32.0% | Best for OOD |

**RCUE Failure Analysis**: Mixing coverage (between-class) with energy (within-class) pollutes both signals.

**Recommended Approach**: Separation of Concerns - use coverage for OOD detection, energy for confidence ranking.

**Surprising Discovery**: OOD triples can have HIGHER MRR than ID triples! Reframe: OOD detection is about verifiability, not correctness.

### Broader Impact Exploration (Option C)
**File**: `docs/broader_impact_exploration.md`

**Cross-Domain Evidence**:
| Domain | Phenomenon | Literature |
|--------|------------|------------|
| RecSys | Popularity Bias, heavy user degradation | Abdollahpouri (2019) |
| NLP | High-freq entity type confusion | Onoe & Durrett (2020) |
| CV | Long-tail: head class confusion | Liu (2019) |
| Psychology | Illusion of Knowledge | Rozenblit & Keil (2002) |

**Proposed Framework**: "Familiarity Trap"
- Confidence scales with frequency
- Accuracy peaks at medium frequency, then degrades
- Mechanism: Embedding Dilution

**Validation Plan**:
1. MovieLens Popularity Paradox (1-2 days)
2. BERT Entity Frequency on LAMA (1 day)
3. ImageNet-LT analysis (optional)

---

## Familiarity Trap Paper Outline

### Title Options
1. "The Familiarity Trap: Why Models Fail on What They've Seen Most"
2. "Embedding Dilution: A Cross-Domain Analysis of the Familiarity-Accuracy Paradox"
3. "Less is More: How Partial Exposure Beats Full Coverage in Neural Embeddings"

### Abstract (Draft)
> We identify the Familiarity Trap: a counter-intuitive phenomenon where neural models become overconfident on frequently-encountered data while their accuracy degrades. In knowledge graphs, queries with partial coverage (59.5%) outperform those with full coverage (32.3%). We trace this to embedding dilution—entities seen in many contexts develop averaged representations that lose specificity. We prove this formally via an embedding geometry analysis and demonstrate the phenomenon generalizes to recommender systems (MovieLens) and language models (BERT entity linking). Our findings challenge the assumption that "more data = better predictions" and provide practical guidelines for uncertainty estimation.

### Contributions
1. **Empirical Discovery**: The Familiarity Trap across 3 domains (KG, RecSys, NLP)
2. **Theoretical Explanation**: Dilution-Specificity Trade-off with formal bounds
3. **Practical Guidelines**: Separation of Concerns for uncertainty estimation
4. **Cross-Domain Validation**: Unified framework explaining disparate phenomena

### Required Experiments
- [x] KG Coverage Paradox (FB15k-237) - DONE
- [ ] MovieLens Popularity Paradox - TODO (1-2 days)
- [ ] BERT Entity Frequency (LAMA) - TODO (1 day)
- [ ] Embedding geometry visualization - TODO
- [ ] Formal theorem proof - TODO

---

## Scripts Created

### Hypothesis Testing
- `scripts/analyze_anchor_hypothesis.py`
- `scripts/analyze_overfitting_hypothesis.py`
- `scripts/analyze_information_leakage.py`
- `scripts/analyze_full_coverage_difficulty.py`

### Method Exploration
- `scripts/method_exploration_coverage_paradox.py`
- `scripts/cascading_v2_test.py`
- `scripts/investigate_ood_mrr.py`

### Utility
- `scripts/rcue_viability_analysis.py`

---

## Next Steps

1. **MovieLens Experiment**: Validate Familiarity Trap in RecSys
2. **BERT/LAMA Experiment**: Validate in NLP entity linking
3. **Formal Theorem**: Write up embedding geometry proof
4. **Paper Draft**: Start writing unified narrative

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Cross-domain doesn't replicate | Fallback to Option 1 (KG-only) |
| Theory too weak | Focus on empirical + practical |
| Scope too broad for reviewers | Option 3 (split papers) |
| Scooped | Check arXiv regularly, move fast |
