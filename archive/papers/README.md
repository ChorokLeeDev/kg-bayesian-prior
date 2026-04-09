# Archived Paper Drafts

This folder contains previous paper drafts that are no longer the main focus.

## Active Paper

The active paper is now at `/paper/` (Coverage Paradox).

## Archived Papers

### paper_neurips_position/
**Title**: "Stop Trusting Embedding-Based Uncertainty for Knowledge Graphs"

**Why archived**: 
- Core contribution (impossibility theorem) is valid but feels "obvious" to reviewers
- "Coverage tracking solves it" = essentially a lookup table, not a method
- Position paper framing is weaker than the empirical Coverage Paradox finding

**Salvageable parts**:
- Theorem 1 (relation-agnostic impossibility) - could be appendix material
- 83% confident-wrong statistic - good hook
- Baseline comparisons (MC Dropout, Deep Ensemble near-random)

---

### paper_rcue/
**Title**: "Relation-Conditioned Uncertainty Estimation for Knowledge Graph Embedding"

**Why archived**:
- MLP contribution is marginal (+4.4pp within-class, not significant)
- RCUE performs worse than pure Energy on selective prediction
- "Just use Coverage + Energy separately" is a valid reviewer critique
- Method doesn't justify the complexity over simple lookup table

**Salvageable parts**:
- Cross-dataset experiments (YAGO3-10, ICEWS14 results)
- Ablation on multiplicative vs additive boost
- Calibration analysis framework

---

### paper_blindspot/
**Title**: "Semantic vs. Structural Uncertainty: Why Standard Methods Fail on Knowledge Graphs"

**Why archived**:
- Overlaps heavily with paper_neurips_position
- Less focused than Coverage Paradox framing
- "Semantic vs Structural" distinction is useful but not novel enough alone

**Salvageable parts**:
- Definition framework (semantic vs structural uncertainty)
- Coverage-energy ensemble results

---

### paper_empty/
Empty folder from old structure. Can be deleted.

---

## Decision Log

**2026-04-09**: Decided to focus on Coverage Paradox paper after RCUE viability analysis showed:
1. MLP within-class advantage only +4.4pp (below 5pp threshold)
2. RCUE worse than Energy on selective prediction
3. Coverage Paradox is genuinely surprising and empirically rigorous
