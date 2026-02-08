# Returning After a Long Break

**Last updated:** 2026-02-08

## Source of Truth

- Active manuscript: `paper/main.tex`
- Submitted camera-ready snapshot: `paper/UAI2026_ChorokLee.pdf`
- Active paper sections: `paper/sections/`
- Archived legacy drafts: `archive/retired_ideas/paper/`

## Project Snapshot

- Branch used for active work: `main`
- The repository includes historical planning docs in `docs/`; treat dates as authoritative.
- This file is the quickest way back in before reading older notes.

## Recommended Re-entry Order

1. Check repo state: `git status --short --branch`
2. Read this file, then `docs/FINDINGS.md`
3. Rebuild manuscript:
   - `cd paper`
   - `pdflatex main.tex`
   - `pdflatex main.tex`
4. Run targeted experiment scripts only as needed:
   - `scripts/run_focused_experiments.py`
   - `scripts/run_wn18rr_temporal.py`
   - `scripts/run_wn18rr_missing_baselines.py`
   - `scripts/verify_assumption_a3.py`

## Housekeeping Rules

- Do not commit temporary LaTeX artifacts (`*.aux`, `*.log`, etc.).
- Do not commit intermediate submission files (for example `*_submission_ready.zip`).
- Keep only intentional release artifacts (for example finalized PDFs) under version control.

## Before New Work

1. Confirm the next milestone in one sentence.
2. Update this file's date and add a short note if priorities changed.
3. Keep commits small and named by intent (paper, experiments, cleanup, docs).
