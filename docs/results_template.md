# Experiment Results - FB15k-237

**Date:** 2024-12-19
**GPU:** Tesla T4 (Colab Free Tier)

---

## Results Table

| Model | MRR | H@1 | H@3 | H@10 | ECE | Brier | AUROC |
|-------|-----|-----|-----|------|-----|-------|-------|
| DistMult | 0.0832 | 0.0255 | 0.0720 | 0.2015 | 0.3165 | 0.2875 | 0.1038 |
| DistMult+MCDropout | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |
| GGPN | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |
| GP-KGE (Ours) | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |

---

## Key Comparisons

### Calibration (ECE) - Lower is Better
```
DistMult:           0.3165  ████████████████████████████████
DistMult+MCDropout: ?.????
GGPN:               ?.????
GP-KGE (Ours):      ?.????
```

### ECE Improvement over GGPN
```
Improvement = (GGPN_ECE - GPKGE_ECE) / GGPN_ECE × 100%
            = (?.???? - ?.????) / ?.???? × 100%
            = ?.?%
```

---

## Analysis (To Fill)

### 1. Link Prediction
- Best MRR: _MODEL_ (?.????)
- GP-KGE vs DistMult: _COMPARISON_

### 2. Calibration (Main Result)
- Best ECE: _MODEL_ (?.????)
- GGPN ECE: ?.???? (confirms poor calibration)
- GP-KGE ECE: ?.???? (confirms good calibration)
- **Improvement: ?.?%**

### 3. OOD Detection
- Best AUROC: _MODEL_ (?.????)
- GP-KGE provides meaningful uncertainty: _YES/NO_

---

## For Paper

### Table 2: Calibration Results (Main Table)
```latex
\begin{table}[h]
\centering
\caption{Calibration comparison on FB15k-237. Lower ECE is better.}
\begin{tabular}{lcccc}
\toprule
Model & ECE $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$ \\
\midrule
DistMult & 0.3165 & 0.2875 & 0.1038 \\
DistMult+MCDropout & ?.???? & ?.???? & ?.???? \\
GGPN & ?.???? & ?.???? & ?.???? \\
\textbf{GP-KGE (Ours)} & \textbf{?.????} & \textbf{?.????} & \textbf{?.????} \\
\bottomrule
\end{tabular}
\end{table}
```

### Key Finding Statement
> GP-KGE achieves **?.?%** better calibration (ECE) compared to GGPN,
> validating that relation-aware kernels with proper Bayesian treatment
> yield well-calibrated uncertainty estimates.

---

## Runtime

| Model | Training Time | Eval Time | Total |
|-------|--------------|-----------|-------|
| DistMult | ~12 min | ~1 min | ~13 min |
| DistMult+MCDropout | ~?? min | ~?? min | ~?? min |
| GGPN | ~?? min | ~?? min | ~?? min |
| GP-KGE (Ours) | ~?? min | ~?? min | ~?? min |

**Total Experiment Time:** ~?? minutes

---

## Raw Output (Paste Here)

```
[PASTE COLAB OUTPUT HERE]
```
