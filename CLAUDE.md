# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating **uncertainty quantification blind spots in Knowledge Graph Embeddings**.

**Current Focus**: The Coverage Paradox paper
- Surprising finding: Partial zero-coverage queries (59.5% Hits@10) outperform full coverage queries (32.3%)
- Full zero-coverage queries fail catastrophically (14.8%)
- This contradicts the assumption "more coverage = more reliable"

**Active Paper**: `paper/main.tex` (Coverage Paradox)

**Archived Papers** (see `archive/papers/README.md` for why):
- `paper_neurips_position/` - Impossibility theorem approach (contribution felt "obvious")
- `paper_rcue/` - RCUE method (MLP contribution marginal, +4.4pp)
- `paper_blindspot/` - Semantic vs Structural framing (overlaps with above)

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Key experiment: Coverage Paradox analysis
python scripts/analyze_anchor_hypothesis.py
python scripts/analyze_overfitting_hypothesis.py

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Project Structure

```
kg-bayesian-prior/
├── paper/                 # Active paper (Coverage Paradox)
├── src/
│   ├── data/loaders.py    # Dataset loading (FB15k-237, WN18RR, etc.)
│   ├── models/            # KGE models
│   │   ├── base.py        # BaseKGEModel
│   │   └── relation_conditioned/rcue.py  # RCUE (archived but code remains)
│   └── evaluation/        # Metrics (AUROC, MRR, calibration)
├── scripts/               # Experiment scripts
├── outputs/               # Experiment logs (gitignored)
├── archive/
│   ├── papers/            # Archived paper drafts with explanations
│   └── root_cleanup/      # Old root files (reviews, logs, etc.)
└── data/raw/              # Datasets
```

## Key Concepts

### Coverage Types
```python
# For query (h, r, t):
cov(e, r) = 1 if entity e seen with relation r in training

Full coverage:    cov(h,r)=1 AND cov(t,r)=1  → 32.3% Hits@10
Partial zero:     cov(h,r) ≠ cov(t,r)        → 59.5% Hits@10 (BEST!)
Full zero:        cov(h,r)=0 AND cov(t,r)=0  → 14.8% Hits@10 (worst)
```

### The Paradox
- Conventional wisdom: "More coverage = better predictions"
- Reality: Partial coverage > Full coverage (frequency-controlled, p<0.001)
- Hypothesis under investigation: Anchor effect, overfitting, information leakage

## Datasets

Located in `data/raw/`:
- **FB15k-237**: 14,541 entities, 237 relations (main benchmark)
- **WN18RR**: 40,943 entities, 11 relations
- **YAGO3-10**: 123,182 entities, 37 relations
- **ICEWS14**: 7,128 entities, 230 relations (temporal)

## Current Research Questions

1. **Why does partial > full?** (Anchor hypothesis, overfitting, information leakage)
2. **Is this dataset-specific?** (Need cross-dataset validation)
3. **Practical implications?** (When to trust predictions)

## Dependencies

Core: `torch>=2.0.0`, `numpy`, `scikit-learn`, `pykeen>=1.10.0`
