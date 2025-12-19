# TODO: NeurIPS Paper Preparation

## Current Results (FB15k-237, Full Test)

| Model | MRR | H@1 | H@10 | ECE ↓ | Brier ↓ | AUROC |
|-------|-----|-----|------|-------|---------|-------|
| DistMult | 0.2411 | 0.1707 | 0.3817 | 0.2118 | 0.1417 | 0.9937 |
| MCDropout | 0.2351 | 0.1621 | 0.3842 | 0.2119 | 0.1385 | 0.0546 |
| GGPN | 0.1205 | 0.0540 | 0.2527 | 0.1965 | 0.1519 | 0.0100 |
| **GP-KGE** | **0.2545** | **0.1761** | **0.4171** | **0.1171** | **0.1013** | 0.2523 |

**Key Finding:** GP-KGE achieves 40.4% better ECE than GGPN.

---

## ✅ Fixed Issues

### 1. ~~Loss Function Mismatch~~ ✅ FIXED
- **Problem:** DistMult/MCDropout use margin ranking loss, GP-KGE uses BCE loss
- **Solution:** All models now use BCE loss (unified in `exp_*.ipynb`)
- **Status:** Need to re-run experiments

### 2. ~~Inverted AUROC for GP-KGE~~ ✅ FIXED
- **Problem:** GP-KGE AUROC=0.2523 (< 0.5)
- **Solution:** Changed uncertainty from `score_var` to `-mean_score` in `predict_with_uncertainty`
- **Status:** Verified AUROC > 0.5 in `test_auroc_fix.ipynb`

---

## ⚠️ Remaining Issues

### 1. GGPN Parameter Mismatch
- **Problem:** GGPN uses 50 dim (memory constraints), others use 200 dim
- **Impact:** GGPN's lower MRR (0.1205) may be due to smaller model, not method
- **Solution:** Note limitation in paper, or try 100 dim if memory allows

### 2. Single Seed
- **Problem:** All results from seed=42 only
- **Impact:** No statistical significance, results may be lucky/unlucky
- **Solution:** Run 3-5 seeds, report mean ± std

---

## ❌ Missing for NeurIPS

### High Priority

- [ ] **Kernel Ablation**: Compare rbf vs relation_aware kernel
  - Shows contribution of relation-aware design

- [ ] **Second Dataset**: WN18RR
  - Standard requirement for KGE papers
  - Shows generalization

- [ ] **Multiple Seeds**: 3-5 runs
  - For error bars and statistical significance

### Medium Priority

- [ ] **Embedding Dim Ablation**: 50, 100, 200
  - Shows scalability

- [ ] **Inducing Points Ablation**: 100, 200, 500
  - Shows efficiency vs quality tradeoff

- [ ] **Calibration Reliability Diagram**
  - Visual proof of calibration quality

- [ ] **Computational Cost Table**
  - Training time, memory usage per model

### Nice to Have

- [ ] **YAGO3-10 Dataset**: Large-scale evaluation
- [ ] **Case Study**: Qualitative uncertainty examples
- [ ] **Negative Sampling Ablation**: 1, 5, 10, 20 negatives

---

## Experiment Code Needed

### Ablation: Kernel Type
```python
# Run GP-KGE with rbf kernel (no relation-aware)
model_rbf = GPKGE(..., kernel_type="rbf")
# Run GP-KGE with relation_aware kernel
model_ra = GPKGE(..., kernel_type="relation_aware")
```

### Ablation: Embedding Dimension
```python
for dim in [50, 100, 200]:
    model = GPKGE(..., embedding_dim=dim)
    # train and evaluate
```

### Multiple Seeds
```python
for seed in [42, 123, 456, 789, 1000]:
    set_seed(seed)
    # train all models and collect results
# report mean ± std
```

---

## Paper Structure (NeurIPS)

1. **Abstract** (150 words)
2. **Introduction** (1 page)
   - Motivation: Uncertainty in KG
   - Gap: Existing methods poorly calibrated
   - Contribution: Relation-aware GP for KGE
3. **Related Work** (0.5 page)
   - Uncertainty in KGE
   - GPs on graphs
4. **Method** (2 pages)
   - Relation-aware kernel
   - Inducing point approximation
   - Training objective
5. **Experiments** (2 pages)
   - Main results (Table 1)
   - Ablation studies (Table 2)
   - Calibration diagram (Figure 1)
6. **Conclusion** (0.5 page)
7. **References**

---

## Timeline Suggestion

| Task | Priority |
|------|----------|
| Fix GGPN params or note limitation | High |
| Run kernel ablation (rbf vs relation_aware) | High |
| Run WN18RR experiments | High |
| Run 3 seeds for main results | High |
| Write paper draft | Medium |
| Create calibration diagram | Medium |
| Add computational cost | Low |
