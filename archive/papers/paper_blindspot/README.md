# Paper: Semantic vs. Structural Uncertainty

**Status:** Active (main version for NeurIPS 2026)

## Key Claims (Verified)

1. **Coverage blind spot**: 67-100% error rate on zero-coverage queries
2. **MC Dropout/Deep Ensemble**: "near random" (0.51-0.63 AUROC), NOT "worse than random"
3. **Coverage-energy ensemble**: 0.70-0.95 AUROC
4. **Neural variant**: up to 28% improvement (p<0.001)

## Data Sources

All numbers verified against:
- `outputs/fb15k237_missing_baselines.json`
- `outputs/wn18rr_missing_baselines.json`
- `outputs/canonical_temporal_results_v2.json`

## Build

```bash
cd paper_blindspot
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## History

- Commit `7b43196`: Fixed numerical inconsistencies with verified data
- Previous version (`paper_neurips_position`) archived due to incorrect claims
