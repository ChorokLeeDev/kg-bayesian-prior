# TODO: NeurIPS Paper Preparation

## Current Status: 2024-12-21

### 🎯 NEW DIRECTION: Coverage-Augmented GP-KGE (CAGP)

**Key Insight:** Relation-specific coverage is the core mechanism for OOD detection.

**Problem:** Vanilla GP-KGE fails on low-diversity KGs (WN18RR: 11 relations)

**Solution:** CAGP = α × GP_variance + (1-α) × Coverage_uncertainty
- α is learned adaptively
- Low-diversity KGs: α → 0 (rely on coverage)
- High-diversity KGs: α → 1 (leverage learned variance)

See `docs/neurips_directions.md` for full research plan.

---

## 🔬 Experiments Status

### ✅ Completed
- [x] FB15k-237 main experiments - GP-KGE AUROC 0.854
- [x] WN18RR validation - GP-KGE fails (AUROC 0.629 vs DistMult 0.860)
- [x] YAGO3-10 - GP-KGE wins (AUROC 0.830 vs DistMult 0.619)
- [x] Relation threshold ablation - Coverage works universally
- [x] Identified core mechanism: relation-specific coverage

### 🔄 In Progress
- [ ] **CAGP on WN18RR** - Fix the failure mode
  - Notebook: `notebooks/cagp_wn18rr_fix.ipynb`
  - Expected: AUROC 0.629 → 0.85+

### 📋 TODO
- [ ] CAGP on FB15k-237 - Maintain performance
- [ ] CAGP on YAGO3-10 - Validate on medium-diversity
- [ ] Ablation: α fixed vs learned
- [ ] Ablation: α global vs per-relation
- [ ] Theoretical: Coverage sufficiency theorem (Path C)

---

## Results Summary

### Current Baselines

| Dataset | Relations | DistMult | GP-KGE | Winner |
|---------|-----------|----------|--------|--------|
| WN18RR | 11 | **0.860** | 0.629 | DistMult |
| YAGO3-10 | 37 | 0.619 | **0.830** | GP-KGE |
| FB15k-237 | 237 | 0.550 | **0.854** | GP-KGE |

### Relation Ablation (Simplified GP-KGE with explicit coverage)

| Relations | DistMult | GP-KGE | Delta |
|-----------|----------|--------|-------|
| 5 | 0.187 | 0.883 | +0.70 |
| 10 | 0.140 | 0.899 | +0.76 |
| 15 | 0.169 | 0.921 | +0.75 |
| 30 | 0.186 | 0.909 | +0.72 |
| 237 | 0.173 | 0.894 | +0.72 |

**Finding:** Explicit coverage works universally (no threshold needed).

---

## Paper Contributions (NeurIPS-level)

### 1. Algorithmic: CAGP
- Novel architecture combining learned + explicit uncertainty
- Fixes known failure mode (WN18RR)
- Adaptive weighting learns optimal combination

### 2. Theoretical: Coverage Sufficiency
- Prove relation-specific coverage is sufficient statistic for OOD detection
- Explains why GP-KGE works (implicitly learns coverage)
- Explains why baselines fail (don't track coverage)

### 3. Empirical: Comprehensive Evaluation
- Multiple datasets (WN18RR, YAGO3-10, FB15k-237)
- Multiple metrics (AUROC, ECE, MRR)
- Ablation studies

---

## Paper Structure

1. **Introduction**: OOD detection in KGs, existing methods fail on some datasets
2. **Analysis**: Identify relation-specific coverage as key mechanism
3. **Theory**: Coverage sufficiency theorem (Path C)
4. **Method**: Coverage-Augmented GP-KGE (Path B)
5. **Experiments**:
   - CAGP fixes WN18RR
   - Maintains FB15k-237/YAGO3-10 performance
   - Ablation studies
6. **Conclusion**: Universal OOD detection via coverage augmentation

---

## Timeline

- **Week 1**: CAGP experiments on all datasets
- **Week 2**: Sufficiency theorem + ablations
- **Week 3**: Paper writing
- **Week 4**: Revision and submission prep
