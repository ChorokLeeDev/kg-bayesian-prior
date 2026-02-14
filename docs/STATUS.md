# Project Status

> **STALE (2025-12-21):** This file is a historical snapshot. Numbers below are pre-v3 and do not
> reflect the reparameterization sampling fix (2026-02-12). For current canonical results, see
> `docs/FINDINGS.md` or the MEMORY.md file. For re-entry workflow, see `docs/RETURNING_AFTER_BREAK.md`.

**Last Updated:** 2025-12-21 (historical -- do not treat numbers as current)

## Executive Summary

Research pivot complete: from GP-KGE → **Semantic-Structural Decomposition for KG OOD Detection**.

**Key Finding:** Effective OOD detection requires combining:
1. **Semantic Uncertainty** (GP variance) - How well-constrained is the embedding?
2. **Structural Uncertainty** (Coverage) - Has entity been seen with this relation?

**CAGP achieves 0.87-0.96 AUROC** with 14-32% synergy over single components.

---

## Experiment Status

### Core Experiments ✅ Complete

| Dataset | Entities | Relations | GP-only | Coverage-only | CAGP | Synergy |
|---------|----------|-----------|---------|---------------|------|---------|
| WN18RR | 40,943 | 11 | 0.647 | 0.657 | **0.871** | +32% |
| FB15k-237 | 14,541 | 237 | 0.749 | 0.821 | **0.960** | +17% |
| YAGO3-10 | 123,161 | 37 | 0.824 | 0.760 | **0.942** | +14% |

### Theoretical Contributions ✅ Complete

| Theorem | Statement | Status |
|---------|-----------|--------|
| Coverage AUROC | AUROC = f(p_h, p_t, s_r) | ✅ Validated (<3% error) |
| GP Limitation | GP variance is relation-agnostic | ✅ Proven |
| Complementarity | Coverage ⊥ GP variance | ✅ Proven by construction |

### Additional Baselines 🟡 Prepared

| Baseline | Notebook | Status | Expected AUROC |
|----------|----------|--------|----------------|
| MC Dropout | `colab_baselines.ipynb` | Ready to run | ~0.72 |
| Deep Ensembles | `colab_baselines.ipynb` | Ready to run | ~0.78 |

---

## File Inventory

### Key Results
| File | Description |
|------|-------------|
| `outputs/coverage_only_results.json` | WN18RR, FB15k-237 ablation |
| `outputs/yago_coverage_only.json` | YAGO3-10 coverage-only |
| `outputs/yago_full_results.json` | YAGO3-10 full comparison |
| `outputs/final_results.json` | Consolidated results |

### Documentation
| File | Description |
|------|-------------|
| `docs/FINDINGS.md` | **Main findings document** |
| `docs/neurips_honest_assessment.md` | Submission strategy |
| `docs/theory/coverage_sufficiency_theorem.md` | Theorem with proof |
| `docs/GPU_EXPERIMENTS.md` | Colab instructions |

### Notebooks
| File | Description | Requires GPU |
|------|-------------|--------------|
| `notebooks/colab_yago_full.ipynb` | YAGO3-10 experiments | ✅ Complete |
| `notebooks/colab_baselines.ipynb` | MC Dropout, Deep Ensembles | 🟡 Ready |

### Scripts
| File | Description |
|------|-------------|
| `scripts/run_coverage_only_ablation.py` | Coverage-only experiments |
| `scripts/verify_theorem.py` | Theorem validation |
| `scripts/analyze_theorem_gap.py` | Gap analysis |

---

## NeurIPS 2026 Submission Plan

### Contributions
| Type | Contribution | Strength |
|------|--------------|----------|
| Conceptual | Semantic-Structural Decomposition | Strong |
| Theoretical | 3 Theorems (validated) | Medium-Strong |
| Method | CAGP Algorithm | Weak (simple) |
| Experimental | 3 datasets, consistent synergy | Strong |

### Recommended Framing
**Analysis/Insight Paper** - The value is in understanding WHY decomposition helps, not in the simple algorithm.

### Estimated Acceptance
60-70% (competitive but not guaranteed)

---

## Remaining Tasks

### Required for Submission
- [ ] Run baselines notebook (MC Dropout, Deep Ensembles)
- [ ] Draft paper (8 pages + appendix)
- [ ] Create figures (synergy visualization, theorem validation)

### Nice to Have
- [ ] Additional datasets (NELL, ConceptNet)
- [ ] Per-relation α analysis
- [ ] Temporal KG extension

---

## Git History (Recent)

```
151efaa Update findings with complete YAGO3-10 GPU results
c1150a2 Colab을 통해 생성됨
e482070 Add GPU experiment notebooks and documentation
bea609d Add comprehensive FINDINGS.md documentation
72a1662 Add theoretical foundation and YAGO coverage results
```

---

## Quick Commands

```bash
# Run coverage-only (CPU, instant)
python scripts/run_coverage_only_ablation.py

# Verify theorem
python scripts/verify_theorem.py

# Full experiments (GPU required)
# Open notebooks/colab_yago_full.ipynb in Colab
# Open notebooks/colab_baselines.ipynb in Colab
```
