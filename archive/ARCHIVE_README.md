# Archived Files

This directory contains files that were archived during the cleanup for the final paper submission.
These files are kept for reference but are no longer actively used in the project.

**Archive Date:** 2025-12-21

---

## Archived Notebooks (`notebooks/`)

These exploratory and development notebooks have been superseded by the final Colab notebooks.

| File | Reason for Archiving |
|------|---------------------|
| `cagp_wn18rr_fix.ipynb` | Early WN18RR experiment; results incorporated into final paper |
| `cagp_fb15k237.ipynb` | Early FB15k-237 experiment; results incorporated into final paper |
| `cagp_yago310.ipynb` | Superseded by `colab_yago_full.ipynb` |
| `exp_distmult.ipynb` | Exploratory baseline experiment; methodology finalized in final notebooks |
| `exp_ggpn.ipynb` | Exploratory GGPN baseline; not used in final paper |
| `exp_gpkge.ipynb` | Exploratory GP-KGE experiment; incorporated into main method |
| `exp_wn18rr_global_kernel.ipynb` | Kernel exploration; results summarized in paper |
| `exp_wn18rr_quick.ipynb` | Quick test notebook; duplicate functionality |
| `exp_yago310.ipynb` | Early YAGO exploration; superseded by `colab_yago_full.ipynb` |
| `kernel_ablation.ipynb` | Kernel ablation study; results in paper, not needed for reproduction |
| `relation_threshold_ablation.ipynb` | Large ablation notebook; results summarized in paper |
| `test_auroc_fix.ipynb` | Bug fix testing; issue resolved |
| `colab_baselines_all.ipynb` | Extended baseline version; `colab_baselines.ipynb` is the canonical version |

**Kept in active notebooks:**
- `colab_yago_full.ipynb` - Main GPU experiments for YAGO3-10
- `colab_baselines.ipynb` - MC Dropout & Deep Ensemble baselines
- `coverage_only_ablation.ipynb` - Coverage-only proof of concept (key ablation)
- `demo.ipynb` - Simple demonstration for reviewers

---

## Archived Experiments Directory (`experiments/`)

The entire `experiments/` directory was archived because its functionality has been superseded by the `scripts/` directory.

| File | Reason for Archiving |
|------|---------------------|
| `ablation_study.py` | Superseded by `scripts/run_coverage_only_ablation.py` |
| `calibration_comparison.py` | Results already in outputs; not needed for reproduction |
| `full_experiment.py` | Superseded by `scripts/run_full_experiment.py` |
| `gpu_experiment.py` | Functionality in Colab notebooks |
| `minimal_experiment.py` | Development/testing file |
| `quick_experiment.py` | Development/testing file |
| `relation_threshold_ablation.py` | Results in paper; not needed for reproduction |
| `run_all_experiments.py` | Batch runner; individual scripts are clearer |
| `train.py` | Training functionality integrated into main scripts |

---

## Archived Documentation (`docs/`)

These documentation files represent earlier versions or have been superseded by consolidated documents.

| File | Reason for Archiving |
|------|---------------------|
| `paper_planning.md` | Superseded by `PAPER_WRITING_PLAN.md` |
| `neurips_directions.md` | Early planning; incorporated into final strategy |
| `neurips2026_assessment.md` | Superseded by `neurips_honest_assessment.md` |
| `research_gap_analysis.md` | Older version; superseded by `NEURIPS_GAP_ANALYSIS.md` |
| `theorem_proofs.md` | Content moved to `theory/coverage_sufficiency_theorem.md` |
| `theoretical_contribution.md` | Superseded by `FINDINGS.md` |
| `results_template.md` | Template file; not needed for final paper |
| `decomposition_insight.md` | Content incorporated into `FINDINGS.md` |
| `findings_summary.md` | Superseded by `FINDINGS.md` (more comprehensive) |
| `literature_review.md` | Background research; incorporated into paper |
| `mathematical_formalization.md` | Early formalization; finalized in paper |

**Kept in active docs:**
- `FINDINGS.md` - Main research findings
- `STATUS.md` - Current project status
- `PAPER_WRITING_PLAN.md` - Paper structure guide
- `GPU_EXPERIMENTS.md` - Colab experiment instructions
- `PAPER_ISSUES.md` - Known issues to fix
- `BASELINE_TODO.md` - Baseline experiments status
- `NEURIPS_GAP_ANALYSIS.md` - Gap analysis for submission
- `neurips_honest_assessment.md` - Self-critical assessment
- `theory/coverage_sufficiency_theorem.md` - Full theorem proof

---

## Archived Figures Directory (`figures/`)

The root-level `figures/` directory was archived because it is a complete duplicate of `paper/figures/`.

All publication figures are maintained in `paper/figures/` which is the canonical location.

---

## Archived Backup (`project.zip`)

Old project backup archive. Not needed since project is under version control (git).

---

## Notes

- Build artifacts (`.log`, `.aux`, `.blg`, `.out` files) were deleted, not archived
- `.DS_Store` files were deleted, not archived
- `texput.log` was deleted, not archived

To restore any archived file, simply move it back to its original location.
