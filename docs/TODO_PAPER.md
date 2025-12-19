# TODO: NeurIPS Paper Preparation

## Baselines (Final)

| Model | Role | Status |
|-------|------|--------|
| DistMult | Deterministic baseline | Running |
| GGPN | GP-based baseline (prior work) | Running |
| **GP-KGE** | **Ours** | Running |

> MC Dropout removed: AUROC < 0.5 indicates inability to detect OOD samples.

---

## Current Experiments (FB15k-237)

**Settings:**
- Loss: BCE (unified)
- Seeds: 42, 123, 456
- Embedding dim: 200 (DistMult, GP-KGE), 100 (GGPN)

**Notebooks:**
- `exp_distmult.ipynb` - DistMult baseline
- `exp_ggpn.ipynb` - GGPN baseline
- `exp_gpkge.ipynb` - GP-KGE (ours)
- `kernel_ablation.ipynb` - RBF vs Relation-Aware kernel

**Results:** Saved to `results/` folder and auto-pushed to GitHub.

---

## ✅ Fixed Issues

### 1. Loss Function Mismatch ✅
- All models now use BCE loss

### 2. Inverted AUROC ✅
- GP-KGE uncertainty changed from `score_var` to `-mean_score`

### 3. GGPN Parameter Mismatch ✅
- Increased to 100 dim

### 4. Single Seed ✅
- All experiments run 3 seeds (42, 123, 456)

### 5. MC Dropout Removed ✅
- AUROC=0.4373 (< 0.5) → Not suitable for OOD detection in KGE

---

## ❌ Remaining for NeurIPS

### High Priority

- [ ] **Collect Current Results**: Wait for running experiments
- [ ] **Kernel Ablation**: RBF vs Relation-Aware
- [ ] **Second Dataset**: WN18RR

### Medium Priority

- [ ] **Calibration Reliability Diagram**
- [ ] **Computational Cost Table**

### Nice to Have

- [ ] **YAGO3-10 Dataset**
- [ ] **Case Study**: Qualitative examples

---

## Paper Structure (NeurIPS)

1. **Abstract** (150 words)
2. **Introduction** (1 page)
3. **Related Work** (0.5 page)
4. **Method** (2 pages)
   - Relation-aware kernel
   - Inducing point approximation
5. **Experiments** (2 pages)
   - Main results: DistMult, GGPN, GP-KGE
   - Ablation: Kernel type
   - Calibration diagram
6. **Conclusion** (0.5 page)
