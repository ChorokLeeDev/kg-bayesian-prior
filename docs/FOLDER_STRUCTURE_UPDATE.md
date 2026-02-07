# Folder Structure Update (2026-02-07)

This note documents the paper-folder cleanup so agents do not edit retired files by mistake.

## Why this changed

The repository had multiple historical paper drafts (`EMNLP`, legacy `UAI`, generic section variants) mixed with the active UAI manuscript. This increased edit mistakes and agent confusion.

## Source of Truth

- Active manuscript entrypoint: `paper/main.tex`
- Active bibliography: `paper/references.bib`
- Active section files:
  - `paper/sections/abstract_uai.tex`
  - `paper/sections/introduction_uai.tex`
  - `paper/sections/related_work_uai.tex`
  - `paper/sections/background.tex`
  - `paper/sections/method_uai_v2.tex`
  - `paper/sections/experiments_uai.tex`
  - `paper/sections/conclusion_uai.tex`

## What was moved

No files were deleted. Retired assets were moved to archive paths:

- Legacy paper section drafts:
  - `archive/retired_ideas/paper/sections/`
- PNG duplicates not used by LaTeX build:
  - `archive/retired_ideas/paper/figures_png/`

## Agent Rules

1. For paper edits, start from `paper/main.tex` includes only.
2. Do not reintroduce files from `archive/retired_ideas/paper/` into active paths unless explicitly requested.
3. If unsure whether a section is active, verify with:
   - `rg -n -F "\\input{" paper/main.tex`
4. Build check command:
   - `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## Current High-Level Layout

```text
paper/
  main.tex
  references.bib
  sections/            # active UAI files only
  figures/             # active PDF figures

archive/retired_ideas/paper/
  sections/            # retired drafts
  figures_png/         # retired PNG copies
```
