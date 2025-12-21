# GPU Experiments Guide

## Overview

This guide explains how to run the GPU-required experiments for the NeurIPS submission.

---

## 1. YAGO3-10 Full Experiment

### Notebook
`notebooks/colab_yago_full.ipynb`

### What it runs
- Coverage-only (instant, no training)
- VanillaGPKGE (50 epochs)
- CAGP (50 epochs)

### Expected results
Based on WN18RR/FB15k-237 pattern, we expect:

| Method | Expected AUROC |
|--------|----------------|
| Coverage-only | ~0.76 (already verified) |
| VanillaGPKGE | ~0.70 |
| CAGP | ~0.85-0.90 |

### How to run
1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChorokLeeDev/kg-bayesian-prior/blob/main/notebooks/colab_yago_full.ipynb)
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Run all cells
4. Download `yago_full_results.json`

### Time estimate
~30 minutes on T4 GPU

---

## 2. Additional Baselines

### Notebook
`notebooks/colab_baselines.ipynb`

### Baselines implemented

| Baseline | Description | Uncertainty Method |
|----------|-------------|-------------------|
| MC Dropout | DistMult with dropout | Predictive entropy over 20 samples |
| Deep Ensemble | 5× DistMult models | Variance of ensemble predictions |
| Coverage-only | Our simple baseline | Relation-specific lookup |

### Expected results (FB15k-237)

| Method | Expected AUROC | vs CAGP (0.96) |
|--------|----------------|----------------|
| Coverage-only | 0.82 | -0.14 |
| MC Dropout | ~0.70-0.75 | -0.21 to -0.26 |
| Deep Ensemble | ~0.75-0.80 | -0.16 to -0.21 |
| **CAGP** | **0.96** | — |

### How to run
1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChorokLeeDev/kg-bayesian-prior/blob/main/notebooks/colab_baselines.ipynb)
2. Set runtime to GPU
3. Run all cells
4. Download `baseline_results.json`

### Time estimate
~1 hour on T4 GPU (5 ensemble models × 3 seeds)

---

## 3. Why These Baselines?

### MC Dropout
- Standard uncertainty quantification technique
- Easy to implement on any model
- Expected to underperform because it's relation-agnostic (like GP variance)

### Deep Ensemble
- State-of-the-art for uncertainty in deep learning
- More expensive (5× training cost)
- Expected to underperform because ensemble variance ≠ coverage

### Coverage-only
- Our proposed simple baseline
- No training required
- Strong performance due to relation-specific signal

---

## 4. Expected Paper Table

After running all experiments:

| Method | WN18RR | FB15k-237 | YAGO3-10 |
|--------|--------|-----------|----------|
| MC Dropout | ~0.65 | ~0.72 | ~0.68 |
| Deep Ensemble | ~0.67 | ~0.78 | ~0.72 |
| Coverage-only | 0.66 | 0.82 | 0.76 |
| VanillaGPKGE | 0.65 | 0.75 | ~0.70 |
| **CAGP (ours)** | **0.87** | **0.96** | **~0.88** |

**Key message:** CAGP significantly outperforms all baselines including Deep Ensembles.

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

## 6. After Running

1. Download result JSONs
2. Place in `outputs/` directory
3. Update `docs/FINDINGS.md` with actual results
4. Run `git add -A && git commit -m "Add GPU experiment results"`

---

## Contact

If experiments fail or results differ significantly from expected, check:
1. CUDA availability (`torch.cuda.is_available()`)
2. Dataset download success
3. Random seed consistency
