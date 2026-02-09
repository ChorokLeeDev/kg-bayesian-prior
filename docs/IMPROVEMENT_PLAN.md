# UAI 2026 Paper Improvement Plan

**Paper**: Decomposing Uncertainty in Probabilistic KG Embeddings
**Submission**: #42, OpenReview (submitted Feb 8)
**Deadline**: Feb 25 (update allowed until deadline)
**Device**: Apple Silicon MPS (no CUDA)
**Estimated total time**: 3-5 days

---

## Current Weaknesses (Reviewer Perspective)

| # | Weakness | Severity | Fixable? |
|---|----------|----------|----------|
| W1 | Simulated temporal splits only — no ground-truth temporal KG | **Critical** | ✅ ICEWS14 |
| W2 | Novel-context AUROC=1 is by construction (definitional coupling) | High | ⚠️ Mitigated |
| W3 | Single-seed on FB15k-237 and YAGO3-10 | Medium | ✅ Easy |
| W4 | Base model weak (DistMult MRR 0.28/0.32) | Medium | ✅ ComplEx |
| W5 | CAGP is trivially simple (binary coverage + α) | Medium | ⚠️ Framing |
| W6 | No relation-aware architecture baseline (R-GCN) | Low-Med | ⚠️ Time-risky |

---

## Priority Tasks

### P0: ICEWS14 Experiment (Critical — eliminates W1)

**Why**: Single highest-impact addition. Converts "simulated only" into "validated on real temporal KG". Reviewer's most natural request.

**What**: ICEWS14 is a temporal KG with ground-truth timestamps.
- ~7K entities, ~230 relations, ~90K triples
- Natural temporal split: train on early timestamps, test on later ones
- No synthetic coverage-based partitioning needed — OOD is defined by time

**Implementation plan**:

1. **Add ICEWS14 data loader** (`src/data/loaders.py`)
   - Download from PyKEEN or direct URL (tab-separated: subject, relation, object, timestamp)
   - Parse timestamps, sort chronologically
   - Split: first 80% timestamps → train, last 20% → test
   - Entity remapping to contiguous IDs
   - Return standard `KGDataset` objects

2. **Create experiment script** (`scripts/run_icews14_temporal.py`)
   - Reuse model definitions from `run_wn18rr_temporal.py`
   - Key difference: temporal split is **real** (timestamp-based), not simulated
   - OOD definition: test triples involving entity-relation pairs unseen in train
   - ID definition: test triples where both (h,r) and (t,r) were seen in train
   - Run all 6 models × 3 seeds

3. **Evaluation**:
   - Same `evaluate_temporal()` logic but using timestamp-derived splits
   - Also compute: % emerging, % novel-context, % ID in real temporal test set
   - This validates that our simulated splits approximate real temporal patterns

4. **Paper integration**:
   - Add ICEWS14 column to Table 1 (temporal OOD) or create new Table
   - Update abstract/conclusion if results are strong
   - Remove "future work" language about ICEWS14 from conclusion

**Expected results** (from EMNLP draft notes): CAGP ~0.89, RelCondVar ~0.91. Even if lower than simulated, it proves the framework works on real temporal data.

**Estimated time**: 1-2 days
**Risk**: Low (small dataset, proven pipeline)

---

### P1: Multi-seed FB15k-237 and YAGO3-10 (Eliminates W3)

**Why**: Removes "single-seed" limitation. Cheap to run.

**What**: Run existing `run_wn18rr_temporal.py` pipeline but ensure FB15k-237 uses the **same split** as the paper (25th percentile threshold producing 2,223/5,193/13,050).

**Implementation plan**:

1. **Verify split consistency**: The existing 3-seed FB15k-237 data in `wn18rr_temporal_results.json` has different splits (1,902/5,369/13,195) than the paper (2,223/5,193/13,050). This is because the threshold computation depends on the random seed affecting training. Need to ensure deterministic splits.

2. **Fix**: Compute entity frequency threshold from the **full dataset** (before any random operations), then split. This ensures identical categorization across seeds.

3. **Run**: 3 seeds × 6 models × 2 datasets. FB15k-237 ~30min/seed, YAGO ~1hr/seed on MPS.

4. **Paper update**: Replace single-seed entries with mean±std. Update Table 1 + Table 2 captions.

**Estimated time**: 4-6 hours (mostly compute)
**Risk**: Very low

---

### P2: ComplEx Base Model (Addresses W4)

**Why**: Shows OOD results are robust to base model quality. ComplEx MRR ~0.34 on FB15k-237 vs DistMult's 0.32.

**What**: Swap DistMult scoring `(h * r * t).sum()` with ComplEx scoring `Re(h * r * conj(t)).sum()`.

**Implementation plan**:

1. **Add ComplEx variants** of CAGP and baselines:
   - Complex-valued embeddings (2× dim, or split real/imag)
   - Scoring: `Re(sum(h * r * conj(t)))`
   - Uncertainty: same decomposition (entity logvar + coverage)

2. **Run on FB15k-237** (most sensitive to base model quality):
   - ComplEx-CAGP vs ComplEx-baselines, 1 seed
   - Compare temporal OOD AUROC: should be similar to DistMult results

3. **Paper integration**:
   - Add to Appendix (architecture ablation already exists)
   - One sentence in main text: "ComplEx base model achieves MRR 0.34 with identical OOD detection patterns (Appendix X)"
   - Strengthens "architecture-agnostic" claim

**Estimated time**: Half day
**Risk**: Low

---

### P3: R-GCN Baseline (Addresses W6, partially W2)

**Why**: Tests whether relation-aware **architectures** automatically learn relation-specific uncertainty. If R-GCN still fails at temporal OOD → proves explicit decomposition is needed even with relation-aware scoring.

**What**: Train R-GCN (relation-specific weight matrices) and evaluate its uncertainty for temporal OOD.

**Implementation plan**:

1. **Implement simplified R-GCN**:
   - Per-relation weight matrices: `W_r ∈ R^{d×d}`
   - Scoring: `h^T W_r t`
   - Uncertainty: entity variance (same as GPOnly but with R-GCN scoring)
   - Basis decomposition for parameter efficiency: `W_r = Σ_b a_{rb} V_b`

2. **Challenge**: R-GCN is a GNN — needs graph structure during forward pass.
   - Simplified version: just use relation-specific bilinear scoring (no message passing)
   - This tests whether relation-conditioned **scoring** helps uncertainty, without full GNN overhead

3. **Run on FB15k-237**: 1 seed, temporal OOD evaluation

4. **Expected result**: R-GCN temporal OOD AUROC should be similar to GPOnly (~0.54) because:
   - Relation-aware scoring helps **predictions** but not **uncertainty calibration**
   - The model still produces entity-level variance, not (entity, relation)-level

5. **Paper integration**:
   - Add to Table 1 or create "R-GCN baseline" row
   - Key sentence: "Even relation-aware architectures produce relation-agnostic uncertainty"
   - Removes "comparison with R-GCN remains future work" from limitations

**Estimated time**: 2-3 days
**Risk**: Medium (implementation complexity, may need debugging)

---

## Execution Order

```
Week 1 (Feb 8-14):
  Day 1-2: P0 (ICEWS14) — highest impact, do first
  Day 2-3: P1 (multi-seed) — run in background while writing ICEWS14 results
  Day 3:   P2 (ComplEx) — quick addition

Week 2 (Feb 15-21):
  Day 1-3: P3 (R-GCN) — only if time permits
  Day 3-4: Paper updates + recompile
  Day 4-5: Buffer for debugging

Feb 22-25: Final review + resubmit PDF
```

---

## Paper Changes Required

### If P0 (ICEWS14) succeeds:

**experiments_uai.tex**:
- Add ICEWS14 column to Table 1 or create new Table (Table 3 currently unused space)
- Add paragraph: "Ground-truth temporal validation"
- Update text: "across three benchmarks" → "across four benchmarks, including one with ground-truth timestamps"

**abstract_uai.tex**:
- "across three benchmarks" → "across three static and one temporal benchmark"

**conclusion_uai.tex**:
- Remove: "Evaluation on temporal KGs with ground-truth timestamps... remains important future work"
- Add ICEWS14 result summary

**related_work_uai.tex**:
- Strengthen temporal KG paragraph with ICEWS14 connection

### If P1 (multi-seed) succeeds:

**experiments_uai.tex**:
- Table 1: Add ±std to FB15k-237 and YAGO columns
- Table 2: Update caption to "3-seed mean" for all datasets
- Remove "(single seed)" annotations

**conclusion_uai.tex**:
- Remove: "FB15k-237 and YAGO3-10 results are from single seeds"

### If P2 (ComplEx) succeeds:

**experiments_uai.tex**:
- One sentence in main text referencing appendix
- Appendix: Add ComplEx results table

### If P3 (R-GCN) succeeds:

**experiments_uai.tex**:
- Add R-GCN row to Table 1
- Paragraph explaining result

**conclusion_uai.tex**:
- Remove: "comparison with architecturally relation-aware models (e.g., R-GCN) remains future work"

---

## Impact Assessment

| Scenario | Tasks Done | Accept Probability |
|----------|-----------|-------------------|
| Current (no changes) | — | 25-35% |
| +ICEWS14 only | P0 | 40-50% |
| +ICEWS14 +multi-seed | P0+P1 | 45-55% |
| +ICEWS14 +multi-seed +ComplEx | P0+P1+P2 | 50-55% |
| All four tasks | P0+P1+P2+P3 | 55-65% |

**The biggest single jump is P0 (ICEWS14).** Everything else is incremental.

---

## Technical Notes

### ICEWS14 Data Format
```
# Tab-separated: subject_id  relation_id  object_id  timestamp
0    1    2    2014-01-01
3    4    5    2014-01-01
...
```
- Source: https://github.com/dair-iitd/tkbi or PyKEEN
- ~7,371 entities, 230 relations, ~90K total triples
- Timestamps: 2014-01-01 to 2014-12-31

### Temporal Split Strategy for ICEWS14
```python
# Sort by timestamp
sorted_triples = triples[triples[:, 3].argsort()]

# 80/20 temporal split
split_idx = int(0.8 * len(sorted_triples))
train = sorted_triples[:split_idx, :3]  # drop timestamp column
test = sorted_triples[split_idx:, :3]

# Build coverage from train
coverage[h, r] = 1 for all (h, r, t) in train
coverage[t, r] = 1 for all (h, r, t) in train

# Categorize test triples using SAME logic as simulated:
# - emerging: entity freq ≤ 25th percentile in train
# - novel_ctx: both entities frequent but (entity, relation) unseen in train
# - ID: all entity-relation pairs seen in train
```

### Key Difference from Simulated Splits
- Simulated: OOD defined by coverage → **definitional coupling with detector**
- ICEWS14: OOD defined by **time** → novel contexts emerge naturally from temporal evolution
- If CAGP still works → validates that coverage captures real temporal patterns, not just a tautology

### Device/Performance
- MPS (Apple Silicon): ~30min per model per dataset (FB15k-237 scale)
- ICEWS14 (~7K entities): should be faster than FB15k-237 (~14K)
- 3 seeds × 6 models × ~20min = ~6 hours total for ICEWS14
