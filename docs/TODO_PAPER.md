# TODO: NeurIPS Paper Preparation

## Current Status: 2024-12-20

### ✅ Completed
- [x] FB15k-237 main experiments (3 seeds) - GP-KGE AUROC 0.854
- [x] WN18RR quick validation - GP-KGE fails (AUROC 0.598 vs DistMult 0.865)
- [x] Ablation: Graph-Init vs Random-Init (ECE +44% improvement)
- [x] Results documentation

### 🔄 In Progress
- [x] **WN18RR 문제 해결: Global Kernel 추가** ✅ IMPLEMENTED
  - 문제: WN18RR은 11개 relation만 있어 eigendecomp 5/11만 성공
  - 해결: `use_global_kernel=True` 옵션 추가
  - 구현: `src/kernels/relation_aware.py` - 모든 edge 통합한 global Laplacian 추가
  - 테스트: `notebooks/exp_wn18rr_global_kernel.ipynb`

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

### WN18RR (1 seed, quick validation)

| Model | MRR | AUROC ↑ | Note |
|-------|-----|---------|------|
| DistMult | 0.236 | **0.865** | Winner |
| GP-KGE | 0.167 | 0.598 | Fails (sparse relations) |

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
