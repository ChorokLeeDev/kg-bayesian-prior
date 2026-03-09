# KG Uncertainty Paper - Strong Accept Campaign

## Current Phase: Reviewer Response
## Target: NeurIPS 2026 Strong Accept (8+/10)
## Current Score: 7/10 (Accept, not Strong)

---

## Reviewer Panel Summary (2026-03-09)

| Reviewer | Focus | Score | Key Concern |
|----------|-------|-------|-------------|
| R1 | Theory | 7/10 | GNN boundary imprecise |
| R2 | Empirical | 7/10 | ULTRA validation missing |
| R3 | Applications | 7/10 | Hetionet OOD experiment missing |

**Consensus**: Accept, but not Strong Accept

---

## Experiments Tracker

### Completed
| Experiment | Result | Status | Paper Section |
|------------|--------|--------|---------------|
| ULTRA validation | **0.29 AUROC** (anti-predictive) | ✅ Done | Section 5.10 |
| GNN boundary analysis | γ* ∈ [5, 12] | ✅ Done | docs/theory/ |
| Hetionet prevalence | 98.9% OOD (inductive) | ✅ Done | Section 5.9 |
| **Hetionet OOD detection** | **0.503 AUROC** (random), CtD=0.431 | ✅ Done | Section 5.9 |
| BioKG split analysis | Random vs structural = 100x diff | ✅ Done | Section 5.9 |
| RAG framework | 382-line theory doc | ✅ Done | docs/theory/ |

### In Progress
| Experiment | Status | ETA | Notes |
|------------|--------|-----|-------|
| - | All critical done | - | Ready for next reviewer panel |

### Pending (for Strong Accept)
| Experiment | Priority | Estimated Impact |
|------------|----------|------------------|
| YAGO3-10 GNN validation | HIGH | Confirms γ heuristic |
| Mahalanobis baseline | MEDIUM | Addresses reviewer Q |
| Conformal prediction comparison | LOW | Nice to have |
| TGB 2.0 temporal benchmarks | LOW | Additional validation |

---

## What's Working (Strengths)

1. ✅ **83% confident-wrong** - striking finding
2. ✅ **Impossibility theorems** - clean theoretical contribution
3. ✅ **ULTRA fails** (0.29 AUROC) - foundation models inherit blind spot
4. ✅ **Biomedical validation** - 99% OOD in drug repurposing
5. ✅ **Trivial fix** - hash table coverage tracking
6. ✅ **Honest limitations** - circularity disclosed

---

## Missing for Strong Accept

### Must Have
- [x] Hetionet actual AUROC (not just prevalence) - **DONE (0.503, CtD=0.431)**
- [x] ULTRA empirical validation - **DONE (0.29)**
- [x] GNN boundary theoretical justification - **DONE**

### Should Have
- [ ] YAGO3-10 GNNSafe test (confirm γ prediction)
- [ ] Statistical significance on all comparisons

### Nice to Have
- [ ] Mahalanobis distance comparison
- [ ] Conformal prediction baseline
- [ ] Wall-clock timing measurements

---

## Key Results Table

| Finding | Metric | Impact |
|---------|--------|--------|
| Confident-wrong | 83% of top-100 | Core finding |
| ULTRA blind spot | 0.29 AUROC | Foundation model failure |
| Hetionet inductive | 98.9% OOD | Biomedical safety |
| Coverage fix | 0.99 AUROC | Trivial solution works |
| GNN boundary | γ* ∈ [5, 12] | Theoretical insight |

---

## Paper Stats

- **Pages**: 37 (8 main + 29 appendix)
- **Warnings**: 0
- **Last commit**: c3255f5
- **Branch**: main

---

## Next Actions

1. Wait for Hetionet OOD experiment to complete
2. Add Hetionet AUROC results to paper
3. Run YAGO3-10 GNNSafe validation
4. Final reviewer panel check
5. Push and prepare for submission

---

## Session Log

### 2026-03-09 Session
- 07:30 - Started reviewer panel
- 08:00 - Panel complete: 7/7/7 (Accept)
- 08:15 - Launched 3 parallel agents (ULTRA, Hetionet, GNN theory)
- 08:45 - GNN boundary analysis complete
- 09:00 - ULTRA validation complete (0.29 AUROC!)
- 09:15 - Added ULTRA section to paper
- 09:20 - Hetionet OOD experiment still running...

---

*Last updated: 2026-03-09 09:20*
