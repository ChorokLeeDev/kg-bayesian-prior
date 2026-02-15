# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on out-of-distribution (OOD) detection in Knowledge Graphs using Gaussian Processes. The core contribution is **CAGP (Coverage-Augmented GP-KGE)**, which combines:
- **Semantic uncertainty** (GP variance): How well-constrained is the entity embedding?
- **Structural uncertainty** (Coverage): Has the entity been observed with this relation?

Key result (v3, with reparameterization sampling fix): CAGP achieves 0.90--0.97 temporal OOD AUROC across WN18RR, FB15k-237, YAGO3-10, and ICEWS14. Target venue: UAI 2026 (deadline Feb 25).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .  # Development mode

# Run CPU experiments (quick validation)
python scripts/run_coverage_only_ablation.py
python scripts/verify_theorem.py

# Held-out relation experiment (breaks circularity critique)
python scripts/run_held_out_relations.py  # FB15k-237 + YAGO3-10, 3 seeds

# Full GPU experiments require Colab - see notebooks/colab_yago_full.ipynb

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Format code
black src/ scripts/
isort src/ scripts/
```

No automated tests exist yet. The `tests/` directory is empty.

## Folder Structure Update (2026-02-07)

The paper folder was cleaned up to reduce confusion for active writing:

- Active manuscript source of truth: `paper/main.tex`
- Active section files remain under `paper/sections/`:
  - `abstract_uai.tex`
  - `introduction_uai.tex`
  - `related_work_uai.tex`
  - `background.tex`
  - `method_uai_v2.tex`
  - `experiments_uai.tex`
  - `conclusion_uai.tex`
- Legacy section drafts moved to `archive/retired_ideas/paper/sections/`
- Non-LaTeX PNG figure copies moved to `archive/retired_ideas/paper/figures_png/`

Before paper edits, check `docs/FOLDER_STRUCTURE_UPDATE.md`.

## Architecture

### Model Hierarchy
All models inherit from `BaseKGEModel(ABC, nn.Module)` in `src/models/base.py`:
- `score_triple()`, `score_heads()`, `score_tails()` for scoring
- Supports DistMult, ComplEx, TransE scoring functions

Key model: `CoverageAugmentedGPKGE` in `src/models/coverage_augmented_gpkge.py`:
- Entity embeddings: `entity_mean` + `entity_logvar` (variational)
- Coverage matrix: `[num_entities, num_relations]` binary buffer
- Adaptive `α` learned via logit parameterization: `U = α * U_gp + (1-α) * U_coverage`

### Graph Kernels (`src/kernels/`)
Relation-aware kernel computes: `K(i,j) = Σ_r σ_r² · exp(-L_r / ℓ_r²)`
- Per-relation Laplacian from relation-specific subgraphs
- Global fallback kernel for sparse-relation KGs (WN18RR has only 11 relations)

### Evaluation (`src/evaluation/`)
- `ood_detection.py`: AUROC, AUPR, FPR@95TPR
- `link_prediction.py`: MRR, Hits@10
- `calibration.py`: Expected calibration error

### Data Flow
1. Load KG triples (`src/data/loaders.py`)
2. Build coverage matrix via `model.precompute_coverage(train_triples)`
3. Train with BCE loss + KL regularization (beta=0.001) + uncertainty margin loss (weight=0.1)
4. Reparameterization sampling: `h_emb = entity_mean[h] + exp(0.5*logvar[h]) * randn` during training
5. Evaluate: compute uncertainties -> OOD metrics

## Key Files

- `src/models/coverage_augmented_gpkge.py` - Main CAGP model
- `src/models/gp_kge.py` - Vanilla GP-KGE with relation-aware kernel
- `src/kernels/relation_aware.py` - Per-relation kernel implementation
- `scripts/run_wn18rr_temporal.py` - Canonical experiment script (fixed CAGP+GPOnly with reparameterization sampling)
- `scripts/test_cagp_fix_multiseed.py` - 3-seed x 3-dataset CAGP validation
- `scripts/run_held_out_relations.py` - Held-out relation experiment (breaks circularity critique)
- `configs/default.yaml` - Hydra config for experiments
- `docs/FINDINGS.md` - Detailed research findings (updated to v3 results)
- `docs/HELD_OUT_RELATIONS_EXPERIMENT.md` - Documentation for held-out relations experiment
- `docs/theory/` - Theorem proofs

## Configuration

Uses Hydra for experiment config (`configs/default.yaml`):
- Dataset: fb15k237, wn18rr (data in `data/raw/`)
- Model: embedding_dim=100, kernel_type=relation_aware
- Training: batch_size=128, lr=0.001, kl_weight=0.001

## Datasets

Located in `data/raw/`:
- **WN18RR**: 40K entities, 11 relations (sparse - needs coverage augmentation)
- **FB15k-237**: 14K entities, 237 relations (dense - GP works well)
- **YAGO3-10**: Run via Colab notebooks

## Dependencies

Core: `torch>=2.0.0`, `gpytorch>=1.10.0`, `torch-geometric>=2.3.0`, `pykeen>=1.10.0`
