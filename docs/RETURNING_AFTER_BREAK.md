# Returning After a Long Break

**Last updated:** 2026-02-13

## Source of Truth

- Active manuscript: `paper/main.tex`
- Active paper sections: `paper/sections/`
- Archived legacy drafts: `archive/retired_ideas/paper/`
- Canonical v3 results: see MEMORY.md or `docs/FINDINGS.md`

## Project Snapshot

- Branch used for active work: `main`
- **UAI 2026 deadline: Feb 25.** Paper is ~18 pages, compiles cleanly.
- All tables and text updated with v3 numbers (reparameterization sampling fix, 2026-02-12).
- The repository includes many historical planning docs in `docs/`; most are stale. Trust `FINDINGS.md` and MEMORY.md.

## Recommended Re-entry Order

1. Check repo state: `git status --short --branch`
2. Read this file, then `docs/FINDINGS.md` for canonical results
3. Rebuild manuscript:
   - `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
4. Key experiment scripts:
   - `scripts/run_wn18rr_temporal.py` -- canonical experiment (CAGP + GPOnly with reparam fix)
   - `scripts/test_cagp_fix_multiseed.py` -- 3-seed x 3-dataset CAGP validation
   - `scripts/test_gponly_fix_multiseed.py` -- 3-seed x 3-dataset GPOnly validation
   - `scripts/test_standard_ood_fixed.py` -- standard OOD + AUPR with fixed models

## Housekeeping Rules

- Do not commit temporary LaTeX artifacts (`*.aux`, `*.log`, etc.).
- Do not commit intermediate submission files (for example `*_submission_ready.zip`).
- Keep only intentional release artifacts (for example finalized PDFs) under version control.

## Before New Work

1. Confirm the next milestone in one sentence.
2. Update this file's date and add a short note if priorities changed.
3. Keep commits small and named by intent (paper, experiments, cleanup, docs).
