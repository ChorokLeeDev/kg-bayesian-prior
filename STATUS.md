# KG Uncertainty Paper - NeurIPS 2026

## Current Status: ACCEPT ACHIEVED ✅
## Final Score: 8/10 (Accept)

---

## Reviewer Panel History

| Round | Score | Key Changes |
|-------|-------|-------------|
| 1 | 7.0 | Initial submission |
| 2 | 7.0 | Circularity concerns |
| 3 | 7.0 | Missing baselines |
| 4 | 7.0 | Definition 2.3 validation |
| **5** | **8.0** | **ULTRA cross-dataset + all fixes** |

---

## All Reviewer Concerns Resolved

| Issue | Resolution |
|-------|------------|
| 83%/78% discrepancy | 78%±3% consistently |
| Definition 2.3 CMI | "Sufficient condition" framing |
| GNN boundary quantification | Coverage correlation (0.46/0.08/0.29) |
| R-GCN baseline | 0.46 AUROC (confirms theory) |
| Mahalanobis baseline | 0.685 AUROC |
| GDELT limitation | Explicit in abstract (0.57 AUROC) |
| ULTRA single-run | WN18RR validation (0.69 AUROC) |

---

## Key Contributions (Final)

1. **78%±3% confident-wrong** - striking core finding
2. **Limitation theorems** - rigorous theory
3. **ULTRA fails** (0.29 on FB15k-237, 0.69 on WN18RR)
4. **GNN boundary analysis** - coverage correlation predicts success
5. **GDELT negative result** - honest limitation
6. **Trivial fix** - hash table solves the problem

---

## Paper Statistics

- **Pages**: 40 (8 main + 32 appendix)
- **Datasets**: 7 (WN18RR, FB15k-237, ICEWS14/18, GDELT, Hetionet, WikiKG2)
- **Baselines**: 10+ (Energy, MC Dropout, Deep Ensembles, SNGP, R-GCN, Mahalanobis, ULTRA, etc.)
- **Seeds**: 5-10 per experiment

---

## Commits This Session

| Commit | Change |
|--------|--------|
| 7ee1e3e | Add ULTRA cross-dataset validation on WN18RR |
| 66cc38a | Refine Definition 2.3 as sufficient condition |
| 63e73e0 | Update Definition 2.3 with direct CMI test results |
| e2074f9 | Fix 83%→78%, add coverage correlation |
| 82cf7da | Add R-GCN baseline (0.46 AUROC) |

---

*Last updated: 2026-03-10 02:05*
*Target: NeurIPS 2026*
