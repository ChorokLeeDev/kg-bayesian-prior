# Submission Freeze — UAI 2026

## Freeze Metadata
- Workspace base HEAD: `246957265dc6dc3f50ebb0059c25c1974ba29795`
- Freeze timestamp (local): `2026-02-08`
- Seed policy: all multi-seed runs use `{42, 123, 456}`

## Final PDF
- File: `paper/main.pdf` (current compile: 19 pages total)
- Compile: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## Key Result Files
| File | Description |
|------|-------------|
| `outputs/wn18rr_temporal_results.json` | WN18RR 3-seed results (Table 1) |
| `outputs/icews14_temporal_results.json` | ICEWS14 3-seed results (Table 1) |
| `outputs/icews14_strict_split_results.json` | ICEWS14 strict split (Table 9) |
| `outputs/multiseed_run_log.txt` | WN18RR+FB15k-237 multi-seed raw log |

## Raw Seed Payload Paths
- `outputs/wn18rr_temporal_results.json`
  - WN18RR per-seed: `.wn18rr.seed_42`, `.wn18rr.seed_123`, `.wn18rr.seed_456`
  - FB15k-237 per-seed: `.fb15k237.seed_42`, `.fb15k237.seed_123`, `.fb15k237.seed_456`
- `outputs/icews14_temporal_results.json`
  - Per-seed: `.seed_42`, `.seed_123`, `.seed_456`
  - Aggregate: `.summary`
- `outputs/icews14_strict_split_results.json`
  - Original split per-seed: `.all_results.original.seed_42|seed_123|seed_456`
  - Strict split per-seed: `.all_results.strict.seed_42|seed_123|seed_456`
  - Split composition: `.split_stats` (removed exact-only 715, inverse-only 2555, both 4466)

## Experiment Scripts → Table Mapping
| Script | Tables |
|--------|--------|
| `scripts/run_wn18rr_temporal.py` | Table 1 (WN18RR), Table 2 (WN18RR) |
| `scripts/run_icews14_temporal.py` | Table 1 (ICEWS14) |
| `scripts/icews14_ablation.py` | Table 8 (ICEWS14 ablation) |
| `scripts/icews14_strict_split.py` | Table 9 (strict split) |
| `scripts/run_focused_experiments.py` | Table 5 (method comparison) |
| `notebooks/exp_temporal_ood.ipynb` | Table 1 (FB15k-237), Table 2 (FB15k-237) |
| `notebooks/colab_yago_full.ipynb` | Table 1 (YAGO3-10) |

## Reproduction Commands
```bash
# ICEWS14 (main result)
PYTHONUNBUFFERED=1 python scripts/run_icews14_temporal.py

# ICEWS14 ablation (Defense 4)
PYTHONUNBUFFERED=1 python scripts/icews14_ablation.py

# ICEWS14 strict split (Defense 1)
PYTHONUNBUFFERED=1 python scripts/icews14_strict_split.py

# WN18RR multi-seed
PYTHONUNBUFFERED=1 python scripts/run_wn18rr_temporal.py
```

## Environment
- Python 3.11, PyTorch 2.x, Apple Silicon MPS
- Dependencies: `pip install -r requirements.txt && pip install -e .`
- No GPU required (all experiments run on MPS/CPU)

## Seeds
All multi-seed experiments use seeds: 42, 123, 456
