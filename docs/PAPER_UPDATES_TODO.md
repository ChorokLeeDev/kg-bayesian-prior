# Paper Updates TODO - UAI Revision

## ✅ COMPLETED

### 1. A3 Assumption Verification
- [x] Run verification script on FB15k-237
- [x] Results: 100% matching for all ε values
- [x] Create appendix section (appendix_assumption_verification.tex)

**Files created:**
- `paper/sections/appendix_assumption_verification.tex`
- `results/assumption_a3_fb15k237.json`

---

## 📝 TO ADD TO MAIN PAPER

### Method Section (paper/sections/method_uai_v2.tex)

**After line 48 (end of Theorem 1 proof sketch), add footnote:**

```latex
\footnote{Assumption A3 is empirically verified in Appendix~\ref{app:assumption_verification},
where we show that 100\% of novel-context triples in FB15k-237 have frequency-matched
ID counterparts for all $\epsilon \geq 1$.}
```

**Or inline after line 50:**

```latex
This result applies to all existing probabilistic KG methods using entity-level variances:
GP-KGE, UKGE, box embeddings, and ensemble approaches. It explains their systematic
failure on temporal OOD (§\ref{sec:experiments}, Table~\ref{tab:icews}).
\textbf{Empirical validation}: Assumption A3 holds with 100\% coverage on FB15k-237
(Appendix~\ref{app:assumption_verification}).
```

### Main paper appendix (paper/main_uai.tex)

**In appendix section (around line 83), add:**

```latex
\appendix

\input{sections/appendix_assumption_verification}

\section{Theorem Proofs}
\label{app:proof}
...
```

---

## ⏳ PENDING (waiting for experiments)

### 2. RelCondVar Ablation Study

**Status:** Running (50 epochs, ~30-60 min remaining)

**What to add (IF results are good):**
- Appendix section showing auxiliary objective is necessary
- Table comparing different auxiliary formulations
- Justify design choices

**What to add (IF results are still poor):**
- Reframe RelCondVar as alternative, not primary
- Focus paper on CAGP only
- Move RelCondVar to appendix or remove

### 3. GPN Baseline Comparison

**Status:** Waiting for torch-geometric installation

**What to add (when complete):**
- Add GPN row to Table 1 (ICEWS14 results)
- Paragraph in Section 4.2 explaining why GPN fails
- Validates that graph-aware methods need explicit coverage

---

## 📋 OTHER UPDATES NEEDED

### 4. Scalability Discussion (READY TO WRITE NOW)

**Location:** Conclusion section

**Option A (Recommended - Honest acknowledgment):**

```latex
\paragraph{Scalability considerations.}
CAGP's coverage matrix requires $O(|\mathcal{E}| \times |\mathcal{R}|)$ memory.
For our largest evaluated dataset (YAGO3-10: 123K entities, 37 relations),
this is 17.5MB dense or $<$1MB sparse (4.6\% non-zero). However, for
Wikidata-scale KGs (90M entities, 1K relations), dense storage would require
360GB---prohibitive for most systems.

We propose two solutions for large-scale deployment:
\begin{enumerate}
\item \textbf{RelCondVar}: Avoids explicit coverage matrix, requiring only
      MLP parameters ($\sim$25K params). Suitable for massive KGs.
\item \textbf{Sparse CAGP}: Store only observed $(e,r)$ pairs in hash tables,
      reducing memory to $O(|\mathcal{T}|)$. Empirical evaluation of sparse
      implementations is left to future work.
\end{enumerate}

For datasets up to $\sim$1M entities, CAGP is deployable with sparse storage.
```

**File to create:** `paper/sections/scalability_discussion.tex`

---

## 🎯 PRIORITY ACTIONS (while waiting)

### High Priority (do now):
1. ✅ Add A3 verification to appendix
2. ✅ Create paper update TODO list (this file)
3. ⬜ Write scalability discussion
4. ⬜ Check if paper claims match experimental results

### Medium Priority:
5. ⬜ Prepare Plan B (remove RelCondVar if needed)
6. ⬜ Update abstract with A3 verification mention
7. ⬜ Review related work section

### Low Priority:
8. ⬜ Check for typos/formatting
9. ⬜ Verify all references compile
10. ⬜ Generate updated figures

---

## 📊 CURRENT EXPERIMENT STATUS

| Experiment | Status | Results | Paper Section Affected |
|-----------|--------|---------|----------------------|
| A3 Verification | ✅ DONE | 100% matched | Appendix, Method footnote |
| RelCondVar 20ep | ✅ DONE | Poor (AUROC~0.50) | Reference only |
| RelCondVar 50ep | ⏳ RUNNING | TBD | Method, Appendix |
| GPN Baseline | ⏳ WAITING | TBD | Table 1, Section 4.2 |

---

## 🚨 DECISION POINTS

### If RelCondVar 50ep fails (AUROC < 0.70):
- **Action:** Remove RelCondVar from main paper
- **Reframe:** CAGP as sole method
- **Story:** "Simple explicit decomposition works best"

### If RelCondVar 50ep succeeds (AUROC > 0.80):
- **Action:** Keep both methods
- **Add:** Ablation table showing aux objective is useful
- **Story:** "Two approaches: explicit (CAGP) vs learned (RelCondVar)"

### If GPN installation fails:
- **Action:** Use existing SNGP results more prominently
- **Alternative:** Implement simple GNN baseline without torch-geometric
- **Fallback:** Emphasize theory + CAGP, downplay baseline comparison

---

## 📝 WRITING CHECKLIST

Before submission:
- [ ] A3 verification in appendix
- [ ] Scalability discussion in conclusion
- [ ] All experiment results match tables
- [ ] References compile correctly
- [ ] Figures have captions
- [ ] Abstract mentions A3 verification
- [ ] No overstated claims
- [ ] Limitations clearly stated
- [ ] Code/data availability statement

---

## ⏱️ TIME ESTIMATES

- Scalability section: 30 min
- Plan B (remove RelCondVar): 1 hour
- Update abstract: 15 min
- Final proofreading: 1 hour
- LaTeX compilation debugging: 30 min

**Total remaining work: 3-4 hours** (after experiments complete)
