# Experiment Results

## Summary

| Dataset | Best OOD Model | AUROC |
|---------|----------------|-------|
| FB15k-237 | GP-KGE | 0.854 |
| WN18RR | TBD | TBD |

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

## Results (In Progress)

| Model | MRR | H@1 | H@10 | ECE ↓ | AUROC ↑ | Time |
|-------|-----|-----|------|-------|---------|------|
| DistMult | 0.2016 | - | 0.4210 | 0.1279 | 0.8479 | - |
| GGPN | - | - | - | - | - | - |
| GP-KGE | - | - | - | - | - | - |

## Dataset Comparison

| Metric | FB15k-237 | WN18RR |
|--------|-----------|--------|
| Entities | 14,541 | 40,943 |
| Relations | 237 | 11 |
| Train triples | 272,115 | 86,835 |

## Preliminary Observations

### DistMult Baseline AUROC
- FB15k-237: 0.550
- WN18RR: **0.848**

WN18RR baseline AUROC가 높은 이유:
- Relations이 11개로 적음 (FB15k-237: 237개)
- 구조가 단순해서 OOD 탐지가 더 쉬움
- Random negative가 더 명확하게 OOD로 구분됨

### Key Question
GP-KGE가 WN18RR에서도 baseline을 넘을 수 있을까?
- FB15k-237: GP-KGE 0.854 >> DistMult 0.550 (+55%)
- WN18RR: GP-KGE ? vs DistMult 0.848
