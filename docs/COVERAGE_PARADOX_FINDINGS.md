# Coverage Paradox: Research Findings Summary

**Date**: 2026-04-09
**Status**: Empirical findings complete, theoretical explanation needed

---

## The Paradox

On FB15k-237, we observe a counter-intuitive phenomenon:

| Coverage Type | Definition | Hits@10 |
|---------------|------------|---------|
| Full Coverage | Both h and t seen with relation r | 32.3% |
| Partial Zero | Only one entity seen with relation r | **59.5%** |
| Full Zero | Neither entity seen with relation r | 14.8% |

**Conventional wisdom**: "More coverage = better predictions"  
**Reality**: Partial > Full (frequency-controlled, p<0.001)

---

## Hypothesis Testing Results

### ✅ REJECTED: Test Set Bias
- **Hypothesis**: Partial coverage entities are higher-frequency (easier)
- **Result**: After frequency matching, paradox persists (+9.4pp, 95% CI: [8.0, 10.8])
- **Conclusion**: Not a data artifact

### ✅ CONFIRMED: Relation Semantics
- **Finding**: Paradox is stronger for common relations
- Rare relations: Coverage helps (conventional wisdom holds)
- Common relations: Partial > Full (paradox dominates)
- **Interpretation**: Compositional generalization on frequent patterns

### ⚠️ PARTIALLY SUPPORTED: Anchor Effect
- **Finding**: Covered entity contributes 1.36-1.48x more to scores
- **Discovery**: Asymmetric effect based on which entity is covered
  - tail_covered: 100% accuracy, margin 17.8
  - head_covered: lower accuracy, margin 7.3
- **Interpretation**: Covered entity = context anchor, Uncovered = prediction target

### ❌ REJECTED: Overfitting Hypothesis
- **Hypothesis**: Full coverage = trained more = overfitting
- **Result**: OPPOSITE! Partial entities trained 4x more (821 vs 190 exposures)
- **New interpretation**: More diverse relation experience → robust embeddings

### ✅ CONFIRMED: Information Dilution
- **Finding**: High-degree entities have diluted embeddings
- Coverage-Degree correlation: Spearman 0.636
- High coverage entities paradoxically have MORE novel contexts (23.9% vs 17.0%)
- **Interpretation**: Many relations → embedding averages over diverse contexts → less specific

### ❌ REJECTED: Full = Difficult Facts
- **Hypothesis**: Full coverage triples are inherently harder
- **Result**: OPPOSITE! Full coverage triples are EASIER (difficulty -3.407 vs -2.938)
- **Interpretation**: Full coverage = common patterns, but model is overconfident

---

## Integrated Explanation

### Why Full Coverage Underperforms:
1. Contains easier, more common patterns
2. Model learns them well
3. BUT: Overconfidence (calibration failure)
4. Embeddings diluted by many relation contexts
5. Result: Confidently wrong

### Why Partial Zero Outperforms:
1. One entity provides anchor (context constraint)
2. More training exposure (4x)
3. More diverse relation experience → robust embedding
4. Clear signal about which entity is unknown → appropriate uncertainty
5. Result: Humbly correct

### Why Full Zero Fails:
1. No anchor
2. Pure extrapolation
3. 14.8% accuracy ≈ random guessing

---

## Key Insight

> **Coverage Paradox = Calibration Problem**
>
> "Seen before" creates false confidence. The model knows it has seen 
> the entity-relation pair, but the embedding is diluted across many 
> contexts. Partial coverage provides clearer signal: one anchor 
> constrains the prediction while the model correctly recognizes 
> uncertainty about the other entity.

---

## Current Paper Value (Honest Assessment)

| Aspect | Status | Value |
|--------|--------|-------|
| Empirical finding | ✅ Strong | Surprising, rigorous validation |
| Mechanistic explanation | ⚠️ Partial | Multiple confirmed factors, no unified theory |
| Theoretical foundation | ❌ Missing | No formal proof of why partial > full |
| Method contribution | ❌ Missing | RCUE attempt failed (MLP +4.4pp marginal) |
| Practical guidelines | ✅ Useful | "Flag full-zero only" reduces false alarms 95% |

**Current level**: Workshop paper / Short paper  
**For main venue**: Need Option A (Theory) or Option B (Method)

---

## Output Files

- `outputs/anchor_hypothesis_results.txt` - Anchor effect analysis
- `outputs/overfitting_hypothesis_results.txt` - Training frequency analysis  
- `outputs/information_leakage_results.txt` - Embedding dilution analysis
- `outputs/full_coverage_difficulty_results.txt` - Difficulty comparison

## Scripts

- `scripts/analyze_anchor_hypothesis.py`
- `scripts/analyze_overfitting_hypothesis.py`
- `scripts/analyze_information_leakage.py`
- `scripts/analyze_full_coverage_difficulty.py`
