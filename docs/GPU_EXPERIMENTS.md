# GPU Experiments Guide

## Overview

This guide explains how to run the GPU-required experiments for the NeurIPS submission.

---

## 1. YAGO3-10 Full Experiment ✅ COMPLETE

### Notebook
`notebooks/colab_yago_full.ipynb`

### Actual Results

| Method | AUROC | std |
|--------|-------|-----|
| Coverage-only | 0.760 | 0.002 |
| VanillaGPKGE | 0.824 | 0.004 |
| **CAGP** | **0.942** | 0.0001 |

**Synergy: +14.3%** over best single component (GP)

### How it was run
- Google Colab with T4 GPU
- 50 epochs, 3 seeds (42, 123, 456)
- ~30 minutes total runtime

---

## 2. Additional Baselines 🟡 READY TO RUN

### Notebook
`notebooks/colab_baselines.ipynb`

### Baselines implemented

| Baseline | Description | Uncertainty Method |
|----------|-------------|-------------------|
| MC Dropout | DistMult with dropout | Predictive entropy over 20 samples |
| Deep Ensemble | 5x DistMult models | Variance of ensemble predictions |
| Coverage-only | Our simple baseline | Relation-specific lookup |

### Expected results (FB15k-237)

| Method | Expected AUROC | vs CAGP (0.96) |
|--------|----------------|----------------|
| Coverage-only | 0.82 | -0.14 |
| MC Dropout | ~0.70-0.75 | -0.21 to -0.26 |
| Deep Ensemble | ~0.75-0.80 | -0.16 to -0.21 |
| **CAGP** | **0.96** | -- |

### How to run
1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChorokLeeDev/kg-bayesian-prior/blob/main/notebooks/colab_baselines.ipynb)
2. Set runtime to GPU
3. Run all cells
4. Download `baseline_results.json`

### Time estimate
~1 hour on T4 GPU (5 ensemble models x 3 seeds)

---

## 3. Why These Baselines?

### MC Dropout
- Standard uncertainty quantification technique
- Easy to implement on any model
- Expected to underperform because it's relation-agnostic (like GP variance)

### Deep Ensemble
- State-of-the-art for uncertainty in deep learning
- More expensive (5x training cost)
- Expected to underperform because ensemble variance != coverage

### Coverage-only
- Our proposed simple baseline
- No training required
- Strong performance due to relation-specific signal

---

## 4. Current Results Table

| Method | WN18RR | FB15k-237 | YAGO3-10 |
|--------|--------|-----------|----------|
| Coverage-only | 0.657 | 0.821 | 0.760 |
| VanillaGPKGE | 0.647 | 0.749 | 0.824 |
| **CAGP (ours)** | **0.871** | **0.960** | **0.942** |
| MC Dropout | -- | pending | -- |
| Deep Ensemble | -- | pending | -- |

**Key message:** CAGP significantly outperforms all components with consistent synergy.

---

## 5. Quick Validation (CPU)

If you want to quickly validate without GPU:

```bash
# Coverage-only on all datasets (instant)
python scripts/run_coverage_only_ablation.py

# Verify theorem
python scripts/verify_theorem.py
```

---

## 6. After Running Baselines

1. Download `baseline_results.json`
2. Place in `outputs/` directory
3. Update `docs/FINDINGS.md` with actual results
4. Run `git add -A && git commit -m "Add baseline results"`

---

## Troubleshooting

If experiments fail or results differ significantly from expected, check:
1. CUDA availability (`torch.cuda.is_available()`)
2. Dataset download success
3. Random seed consistency
