# Semantic-Structural Decomposition for KG Uncertainty

**OOD Detection in Knowledge Graphs via Coverage-Augmented GP-KGE (CAGP)**

## Key Finding

Effective OOD detection in knowledge graphs requires **two complementary signals**:

| Signal | Type | What it captures |
|--------|------|------------------|
| **GP Variance** | Semantic | How well-constrained is the entity embedding? |
| **Coverage** | Structural | Has the entity been observed with this relation? |

**Neither signal alone is sufficient.** Their combination (CAGP) achieves 17-32% improvement over the best single component.

## Results

| Dataset | GP-only | Coverage-only | CAGP | Synergy |
|---------|---------|---------------|------|---------|
| WN18RR | 0.647 | 0.657 | **0.871** | +32% |
| FB15k-237 | 0.749 | 0.821 | **0.960** | +17% |
| YAGO3-10 | 0.824 | 0.760 | **0.942** | +14% |

## The CAGP Algorithm

```python
U_cagp = α × U_gp + (1-α) × U_coverage

where:
  U_gp = (σ²_head + σ²_tail) / 2      # GP variance (semantic)
  U_cov = 2 - c(h,r) - c(t,r)          # Coverage uncertainty (structural)
  α ≈ 0.5                              # Learned mixing coefficient
```

## Theoretical Contributions

| Theorem | Statement | Validation |
|---------|-----------|------------|
| **Coverage AUROC** | Closed-form AUROC from coverage statistics | <3% error |
| **GP Limitation** | GP variance is relation-agnostic | Proven |
| **Complementarity** | Coverage ⊥ GP variance | Proven by construction |

## Project Structure

```
kg-bayesian-prior/
├── paper/               # Active manuscript (source of truth)
│   ├── main.tex
│   ├── sections/        # Active UAI sections only
│   └── figures/         # Active PDF figures
├── archive/retired_ideas/
│   └── paper/           # Archived legacy drafts and PNG figure copies
├── src/                 # Model/data/evaluation code
├── scripts/             # Experiment scripts
├── notebooks/           # Colab notebooks for GPU experiments
├── outputs/             # Experiment results (JSON)
└── docs/
    ├── FINDINGS.md                 # Main findings document
    ├── STATUS.md                   # Project status
    ├── FOLDER_STRUCTURE_UPDATE.md  # Folder structure change guide
    └── theory/                     # Theorem proofs
```

## Quick Start

```bash
# Coverage-only baseline (CPU, instant)
python scripts/run_coverage_only_ablation.py

# Verify Coverage AUROC theorem
python scripts/verify_theorem.py

# Full experiments (GPU required - use Colab)
# notebooks/colab_yago_full.ipynb
```

## Documentation

- **[FINDINGS.md](docs/FINDINGS.md)** - Complete research findings
- **[STATUS.md](docs/STATUS.md)** - Current project status
- **[GPU_EXPERIMENTS.md](docs/GPU_EXPERIMENTS.md)** - Colab instructions
- **[FOLDER_STRUCTURE_UPDATE.md](docs/FOLDER_STRUCTURE_UPDATE.md)** - Agent-facing folder migration guide

## Citation

```bibtex
@article{cagp2025,
  title={The Semantic-Structural Decomposition: Understanding Uncertainty in Knowledge Graph Embeddings},
  author={...},
  journal={NeurIPS 2026 (under review)},
  year={2025}
}
```
