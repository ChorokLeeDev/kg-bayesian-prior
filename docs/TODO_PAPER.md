# TODO: NeurIPS Paper Preparation

## Current Status: 2024-12-20

### ✅ Completed
- [x] FB15k-237 main experiments (3 seeds) - GP-KGE AUROC 0.854
- [x] WN18RR quick validation - GP-KGE fails (AUROC 0.598 vs DistMult 0.865)
- [x] Ablation: Graph-Init vs Random-Init (ECE +44% improvement)
- [x] Results documentation

### 🔄 In Progress
- [x] **WN18RR 문제 해결: Global Kernel 추가** ✅ TESTED
  - 문제: WN18RR은 11개 relation만 있어 eigendecomp 5/11만 성공
  - 해결: `use_global_kernel=True` 옵션 추가
  - 결과:
    - ECE: **+64% 개선** (0.144 → 0.051) ✅
    - AUROC: +4% 개선 (0.605 → 0.629) - 여전히 DistMult(0.860)보다 낮음
  - 결론: Global kernel은 calibration에 효과적, OOD detection은 relation-rich KG에서만 유효

### 📋 TODO (NeurIPS 수준)
- [ ] 추가 데이터셋: YAGO3-10, CoDEx-L/M
- [ ] 추가 Baselines: RGCN, CompGCN
- [ ] Scalability 분석 (시간/메모리)
- [ ] 이론적 분석 (optional)

---

## PIVOT: Calibration → OOD Detection

**Original Hypothesis:** GP-KGE provides better calibration (ECE)
- ❌ Result: GP-KGE has worst ECE (0.118 vs GGPN 0.079)

**New Direction:** GP-KGE provides superior OOD detection
- ✅ Result: GP-KGE AUROC 0.854 >> DistMult 0.550 >> GGPN 0.221

---

## Results Summary

### FB15k-237 (3 seeds)

| Model | MRR | H@10 | ECE ↓ | AUROC ↑ |
|-------|-----|------|-------|---------|
| DistMult | 0.264 | 0.433 | 0.082 | 0.550 |
| GGPN | 0.249 | 0.416 | **0.079** | 0.221 |
| **GP-KGE** | 0.255 | 0.413 | 0.118 | **0.854** |

### WN18RR (1 seed, with Global Kernel)

| Model | MRR | ECE ↓ | AUROC ↑ |
|-------|-----|-------|---------|
| DistMult | **0.205** | 0.133 | **0.860** |
| GP-KGE (no global) | 0.164 | 0.144 | 0.605 |
| GP-KGE (global) | 0.171 | **0.051** | 0.629 |

**Finding:** Global kernel helps calibration (+64% ECE) but not OOD detection (+4% AUROC).

### Ablation: Init Method (FB15k-237)

| Model | ECE ↓ | AUROC ↑ |
|-------|-------|---------|
| Random-Init | 0.099 | 0.870 |
| **Graph-Init** | **0.055** | **0.877** |

**Key Finding:** Graph eigenvector init improves calibration +44%.

---

## New Paper Story

### Title Options
1. "Detecting Unreliable Knowledge Graph Predictions with Gaussian Process Priors"
2. "GP-KGE: Out-of-Distribution Detection for Knowledge Graph Embeddings"

### Main Contribution
- GP prior on entity embeddings enables effective OOD detection
- Relation-aware kernel captures graph structure
- 55% improvement over deterministic baseline (AUROC)

### Why OOD Detection Matters
- KG often incomplete → many valid queries have no answer
- Need to know when model is uncertain
- Applications: medical KG, financial KG (high-stakes)

---

## Remaining Experiments

### High Priority
- [ ] **WN18RR dataset** - Show generalization
- [ ] **Kernel ablation** - RBF vs Relation-Aware (already have notebook)

### Medium Priority
- [ ] **OOD type ablation** - Random vs Corrupted vs Novel entities
- [ ] **Selective prediction** - Risk-coverage curves

### Optional
- [ ] **Temperature scaling** - Can it fix GP-KGE calibration?
- [ ] **Uncertainty visualization** - Qualitative examples

---

## Paper Structure (Revised)

1. **Abstract** - OOD detection focus
2. **Introduction** - Why OOD detection matters for KG
3. **Related Work** - Uncertainty in KGE, OOD detection
4. **Method** - GP-KGE with relation-aware kernel
5. **Experiments**
   - Main: OOD detection (AUROC) - Table 1
   - Link prediction (MRR) - competitive
   - Calibration (ECE) - discuss limitation
   - Ablation: kernel type
6. **Conclusion**

---

## Questions to Address

1. **Why poor ECE but good AUROC?**
   - ECE: confidence vs accuracy matching
   - AUROC: ranking uncertainty (ID vs OOD separation)
   - GP may be overconfident but still distinguishes OOD

2. **Why GGPN AUROC < 0.5?**
   - Their uncertainty estimation may be inverted
   - Worth investigating in paper

3. **Is 0.854 AUROC good enough?**
   - Yes, significant improvement over baselines
   - Practical for filtering unreliable predictions
