# Experiment Results - FB15k-237

**Date:** 2024-12-20
**GPU:** NVIDIA A100-SXM4-40GB (Colab Pro Education)
**Test Set:** Full (20,466 triples)
**Seeds:** 42, 123, 456 (mean ± std)

---

## Baselines

| Model | Role | Embedding Dim |
|-------|------|---------------|
| DistMult | Deterministic baseline | 200 |
| GGPN | GP-based baseline | 100 |
| **GP-KGE** | **Ours** | 200 |

> MC Dropout removed: AUROC=0.4373 (< 0.5) indicates inability to detect OOD samples.

---

## Results Table (Pending)

| Model | MRR | H@1 | H@10 | ECE ↓ | Brier ↓ | AUROC |
|-------|-----|-----|------|-------|---------|-------|
| DistMult | - | - | - | - | - | - |
| GGPN | - | - | - | - | - | - |
| **GP-KGE** | - | - | - | - | - | - |

*Results will be populated from `results/*.json` after experiments complete.*

---

## Model Configurations

| Model | Embedding Dim | Other Params |
|-------|---------------|--------------|
| DistMult | 200 | BCE loss |
| GGPN | 100 | hidden=100, layers=1, rff=20 |
| GP-KGE | 200 | kernel=relation_aware, inducing=500 |

---

## For Paper

### Table 1: Main Results on FB15k-237

```latex
\begin{table}[h]
\centering
\caption{Experimental results on FB15k-237 (mean $\pm$ std over 3 seeds). Best results in \textbf{bold}.}
\begin{tabular}{lcccccc}
\toprule
Model & MRR $\uparrow$ & H@1 $\uparrow$ & H@10 $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$ \\
\midrule
DistMult & - & - & - & - & - & - \\
GGPN & - & - & - & - & - & - \\
\textbf{GP-KGE (Ours)} & - & - & - & - & - & - \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Notes

- All models use BCE loss for fair calibration comparison
- GGPN uses 100 dim (vs 200) due to memory constraints
- Results auto-saved to `results/` folder via Colab
