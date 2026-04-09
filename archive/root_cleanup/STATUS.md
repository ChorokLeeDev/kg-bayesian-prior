# KG Uncertainty Paper - NeurIPS 2026

## Current Status: STRONG ACCEPT ✅
## Final Score: 9/10 (Strong Accept)

---

## Reviewer Panel History

| Round | Score | Key Changes |
|-------|-------|-------------|
| 1-4 | 7.0-8.0 | Initial submission + fixes |
| 5 | 8.0 | Accept achieved |
| 6 | 8.5 | Minor issues identified |
| **7** | **9.0** | **Strong Accept** |

---

## Key Improvement: Frequency-Controlled Analysis

The reviewer's major concern was: "The 78% may reflect frequency-coverage correlation."

**Resolution:**
- Frequency-matched baseline: 44%±5%
- Energy top-100: 80%
- Elevation: **1.83x** (z=7.78, p<10⁻¹²)
- Within-quintile elevation: 1.55-1.96x across ALL quintiles

**Conclusion:** The 78% phenomenon is NOT a frequency artifact.

---

## All Issues Resolved

| Issue | Status |
|-------|--------|
| 78% frequency confound | ✅ 1.83x elevation (z=7.78) |
| Definition 2.3 CMI | ✅ "Sufficient condition" |
| GNN boundary | ✅ Coverage correlation |
| R-GCN baseline | ✅ 0.46 AUROC |
| ULTRA validation | ✅ FB15k: 0.29, WN18RR: 0.69 |
| GDELT limitation | ✅ Explicit (0.57 AUROC) |
| Bloom filter claim | ✅ ≤1pp (corrected) |

---

## Paper Statistics

- **Pages**: 40 (8 main + 32 appendix)
- **Datasets**: 7
- **Baselines**: 10+
- **Final Score**: 9/10 (Strong Accept)

---

## Latest Commits

```
036ae06 Add frequency-controlled analysis for 78% finding
7ee1e3e Add ULTRA cross-dataset validation on WN18RR
66cc38a Refine Definition 2.3 as sufficient condition
```

---

*Last updated: 2026-03-10*
*Target: NeurIPS 2026*
