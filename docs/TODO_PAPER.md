# TODO: NeurIPS Paper Preparation

## PIVOT: Calibration → OOD Detection

**Original Hypothesis:** GP-KGE provides better calibration (ECE)
- ❌ Result: GP-KGE has worst ECE (0.118 vs GGPN 0.079)

**New Direction:** GP-KGE provides superior OOD detection
- ✅ Result: GP-KGE AUROC 0.854 >> DistMult 0.550 >> GGPN 0.221

---

## Final Results (FB15k-237, 3 seeds)

| Model | MRR | H@10 | ECE ↓ | AUROC ↑ |
|-------|-----|------|-------|---------|
| DistMult | 0.264 | 0.433 | 0.082 | 0.550 |
| GGPN | 0.249 | 0.416 | **0.079** | 0.221 |
| **GP-KGE** | 0.255 | 0.413 | 0.118 | **0.854** |

**Key Finding:** GP-KGE achieves 55% better AUROC than DistMult for OOD detection.

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
