# Experiment Results

## Summary

| Dataset | Best OOD Model | AUROC | GP-KGE Advantage |
|---------|----------------|-------|------------------|
| FB15k-237 | **GP-KGE** | 0.854 | ✅ +55% vs DistMult |
| WN18RR | DistMult | 0.865 | ❌ -31% vs DistMult |

**Key Finding:** GP-KGE excels on relation-rich KGs (FB15k-237: 237 relations), but struggles on relation-sparse KGs (WN18RR: 11 relations).

---

# FB15k-237 Results

**Date:** 2024-12-20
**GPU:** NVIDIA A100-SXM4-40GB (Colab Pro Education)
**Seeds:** 42, 123, 456 (mean ± std)
**Loss:** BCE (unified for all models)

---

## Final Results

| Model | MRR | H@1 | H@10 | ECE ↓ | Brier ↓ | AUROC ↑ |
|-------|-----|-----|------|-------|---------|---------|
| DistMult | 0.2644 ± 0.0003 | 0.1835 ± 0.0008 | 0.4329 ± 0.0021 | 0.0820 ± 0.0014 | 0.0710 ± 0.0007 | 0.5503 ± 0.0086 |
| GGPN | 0.2494 ± 0.0020 | 0.1694 ± 0.0034 | 0.4156 ± 0.0004 | **0.0788 ± 0.0035** | **0.0711 ± 0.0026** | 0.2206 ± 0.0090 |
| **GP-KGE** | 0.2550 ± 0.0011 | 0.1777 ± 0.0022 | 0.4130 ± 0.0005 | 0.1181 ± 0.0008 | 0.1025 ± 0.0006 | **0.8542 ± 0.0025** |

---

## Key Findings

### 1. OOD Detection (AUROC) - GP-KGE Wins
```
GP-KGE:   0.8542  ████████████████████████████████████████
DistMult: 0.5503  ██████████████████████
GGPN:     0.2206  █████████
```
- **GP-KGE achieves 55% better AUROC than DistMult**
- GGPN AUROC < 0.5 indicates inverted uncertainty

### 2. Calibration (ECE) - GGPN Wins
```
GGPN:     0.0788  ████████
DistMult: 0.0820  ████████
GP-KGE:   0.1181  ████████████
```
- GP-KGE has worst calibration
- Original hypothesis (GP-KGE best calibration) rejected

### 3. Link Prediction (MRR) - DistMult Wins
```
DistMult: 0.2644  ██████████████████████████
GP-KGE:   0.2550  █████████████████████████
GGPN:     0.2494  █████████████████████████
```
- All models competitive (~0.25-0.26)
- GP-KGE maintains reasonable accuracy

---

## Pivot: OOD Detection Focus

**Original Story:** "GP-KGE provides better calibration"
- ❌ ECE: GP-KGE worst (0.118 vs 0.079)

**New Story:** "GP-KGE provides superior OOD detection"
- ✅ AUROC: GP-KGE best (0.854 vs 0.550)
- GP prior captures epistemic uncertainty effectively
- Useful for detecting unreliable predictions

---

## Model Configurations

| Model | Embedding Dim | Other Params |
|-------|---------------|--------------|
| DistMult | 200 | BCE loss |
| GGPN | 100 | hidden=100, layers=1, rff=20 |
| GP-KGE | 200 | kernel=relation_aware, inducing=500 |

---

## For Paper (LaTeX)

```latex
\begin{table}[h]
\centering
\caption{Results on FB15k-237 (mean $\pm$ std over 3 seeds). Best in \textbf{bold}.}
\begin{tabular}{lccccc}
\toprule
Model & MRR $\uparrow$ & H@10 $\uparrow$ & ECE $\downarrow$ & AUROC $\uparrow$ \\
\midrule
DistMult & 0.264 & 0.433 & 0.082 & 0.550 \\
GGPN & 0.249 & 0.416 & \textbf{0.079} & 0.221 \\
GP-KGE (Ours) & 0.255 & 0.413 & 0.118 & \textbf{0.854} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Analysis Notes

### Why GP-KGE has poor ECE but good AUROC?

- **ECE** measures: "confidence matches accuracy"
- **AUROC** measures: "can distinguish ID vs OOD"

GP-KGE may be:
- Overconfident on correct predictions (hurts ECE)
- But correctly assigns higher uncertainty to OOD samples (good AUROC)

### GGPN AUROC < 0.5

- Indicates uncertainty is inverted
- High uncertainty on ID, low on OOD
- Possible issue with their uncertainty estimation

---

# WN18RR Results (Quick Validation)

**Date:** 2024-12-20
**Settings:** Reduced for quick validation (epochs=30, dim=100, sample=2000)
**Seed:** 42 (single run)

## Results

| Model | MRR | H@1 | H@10 | ECE ↓ | AUROC ↑ | Time |
|-------|-----|-----|------|-------|---------|------|
| **DistMult** | **0.2363** | **0.1385** | **0.4495** | 0.1513 | **0.8649** | 39s |
| GGPN | - | - | - | - | - | Kernel crash |
| GP-KGE | 0.1665 | 0.0825 | 0.3555 | **0.1453** | 0.5984 | 264s |

## Dataset Comparison

| Metric | FB15k-237 | WN18RR |
|--------|-----------|--------|
| Entities | 14,541 | 40,943 |
| Relations | 237 | **11** |
| Train triples | 272,115 | 86,835 |
| Eigendecomp success | 223/237 | **5/11** |

## Analysis: Why GP-KGE Fails on WN18RR

### 1. Relation-Aware Kernel Limitation
```
FB15k-237: 223/237 relations → eigendecomp works
WN18RR:    5/11 relations   → most relations skipped
```
- GP-KGE의 핵심인 relation-aware kernel이 제대로 작동하지 않음
- 대부분의 relation에서 edge가 너무 적어 eigendecomp 실패

### 2. Graph Structure Difference
- **FB15k-237** (Freebase): 다양한 관계 타입, 밀집된 subgraph
- **WN18RR** (WordNet): 계층적 구조, 희소한 연결

### 3. OOD Detection Baseline Already Strong
- WN18RR에서 DistMult AUROC = 0.865 (이미 매우 높음)
- 단순한 entropy 기반 uncertainty로도 OOD 잘 구분됨
- GP의 추가 복잡성이 오히려 방해

## Key Insight

| Dataset | Relations | GP-KGE AUROC | DistMult AUROC | Winner |
|---------|-----------|--------------|----------------|--------|
| FB15k-237 | 237 (many) | **0.854** | 0.550 | GP-KGE ✅ |
| WN18RR | 11 (few) | 0.598 | **0.865** | DistMult ✅ |

**Conclusion:** GP-KGE's relation-aware kernel provides significant OOD detection advantage only when:
1. Many relation types exist (enables meaningful per-relation kernels)
2. Each relation has sufficient edges (enables eigendecomposition)

For relation-sparse KGs like WN18RR, simpler baselines are more effective.
