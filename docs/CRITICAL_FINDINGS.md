# CRITICAL FINDINGS - Paper Claims Verification

**Date:** 2024-12-26
**Status:** 🚨 URGENT - Major discrepancies found

---

## 🚨 CRITICAL ISSUE #1: ICEWS14 Results Don't Exist

### What the paper claims:

**Table 1 (experiments_uai.tex, lines 42-43):**
```
CAGP (learned α)              0.891  0.847  0.781
RelCondVar (learned σ²(e,r))  0.912  0.873  0.805
```

**Abstract (abstract_uai.tex, line 14):**
```
"across four benchmarks, including ICEWS14 with ground-truth temporal splits"
```

**Experiments section mentions:**
- "ICEWS14 (7,128 entities, 230 relations with timestamps)"
- "Train on Jan--Sep 2014, evaluate on Oct--Dec 2014"
- Multiple paragraphs discussing ICEWS14 results

### What actually exists:

**Git commit 388ecb3 (Dec 23, 2025):**
```
"Switch temporal OOD to FB15k-237 frequency-based split (ICEWS14 unavailable)"
```

**Notebook exp_temporal_ood.ipynb:**
- Title: "Temporal OOD Experiments"
- Dataset: FB15k-237 (NOT ICEWS14)
- Method: Frequency-based temporal simulation
- Comment: "ICEWS14 unavailable, using FB15k-237 proxy"

**Search results:**
- No ICEWS14 data files found
- No ICEWS14 result files found
- No scripts that use ICEWS14
- Git history confirms ICEWS14 was never successfully used

**Notebook results (FB15k-237 simulated temporal):**
- CAGP: 0.9856
- Coverage-only: 0.9353
- GP-only: 0.5421

### Conclusion:

**ICEWS14 experiments were NEVER run.** The numbers in Table 1 are either:
1. Hypothetical/aspirational targets
2. FB15k-237 results mislabeled as ICEWS14
3. Incorrectly left in from an earlier draft

**Impact:** 🚨 **SEVERE** - The paper claims results on a dataset that was never evaluated.

---

## 🚨 CRITICAL ISSUE #2: Table 2 Sample Sizes Don't Match

### What the paper claims:

**Table 2 caption (experiments_uai.tex, line 54):**
```
Sample sizes: Emerging n=2,134, Novel contexts n=17,896, Mixed n=531
```

**Table 2 results:**
```
                  Emerging  Novel Ctx  Mixed   Overall
U_sem             0.826     0.421      0.673   0.542
U_str             0.784     1.000      0.912   0.935
CAGP              0.923     0.979      0.962   0.965
RelCondVar        0.941     0.983      0.971   0.972
```

### What our A3 verification found:

**From results/assumption_a3_fb15k237.json:**
```
Emerging: 1,317 triples (6.4%)
Novel contexts: 5,678 triples (27.7%)
ID: 13,471 triples (65.8%)
Total: 20,466 triples
```

### Discrepancies:

|  | Paper Claims | A3 Verification | Difference |
|---|---|---|---|
| **Emerging** | 2,134 | 1,317 | **+817 (62% more)** |
| **Novel contexts** | 17,896 | 5,678 | **+12,218 (315% more!)** |
| **Mixed** | 531 | ??? | Not in our split |
| **Total OOD** | 20,561 | 6,995 | **+13,566** |

### Additional issues:

1. **"Mixed" category undefined** - Our A3 verification doesn't have this category
2. **Total doesn't add up** - Paper: 2,134 + 17,896 + 531 = 20,561, but test set has 20,466 triples
3. **No source file** - Cannot find any result file with these sample sizes

### Searched for:

- ❌ No file contains "17896" or "2134"
- ❌ `complementarity_analysis.json` doesn't have stratified AUROC
- ❌ `exp_temporal_ood.ipynb` uses different split (new_entity: 2,223, new_pair: 5,193)
- ❌ No stratified evaluation script found

### Possible explanations:

1. **Different OOD split definition** - Perhaps an older version with different thresholds
2. **Different test set** - Maybe validation set instead of test set?
3. **Hypothetical numbers** - Like ICEWS14, these may not be from actual experiments
4. **Notebook results mislabeled** - The notebook has different sample sizes

### Conclusion:

**Table 2 sample sizes and results cannot be verified from existing files.**

**Impact:** ⚠️ **HIGH** - Key theorem validation table may contain unverifiable claims.

---

## ✅ VERIFIED RESULTS (Can use confidently)

### 1. A3 Assumption Verification - FB15k-237 ✅

**File:** `results/assumption_a3_fb15k237.json`

**Results:**
- 100% of novel-context triples have frequency-matched ID counterparts
- Tested for ε ∈ {1, 2, 5, 10, 20, 50, 100}
- All values: 100% matching

**OOD composition:**
- Emerging: 1,317 (6.4%)
- Novel contexts: 5,678 (27.7%)
- ID: 13,471 (65.8%)

**Already integrated into paper appendix** ✅

---

### 2. YAGO3-10 Full Results ✅

**File:** `outputs/yago_full_results.json`

**Results:**
```json
{
  "CoverageOnly": {"mean": 0.76, "std": 0.0024},
  "VanillaGPKGE": {"mean": 0.8242, "std": 0.0042},
  "CAGP": {"mean": 0.9424, "std": 0.0001}
}
```

**Details:**
- 3 seeds: [42, 123, 456]
- 50 epochs
- Embedding dim: 100
- Synergy: 0.1182 (14.3% over single signals)
- Learned α: 0.5

**Ready for paper use** ✅

---

### 3. Temporal OOD - FB15k-237 Simulated ✅

**File:** Notebook `exp_temporal_ood.ipynb`

**Results:**
- CAGP: 0.9856
- Coverage-only: 0.9353
- GP-only: 0.5421
- Synergy: 0.0503

**OOD breakdown:**
- New entity (emerging): n=2,223
  - GP-only: 0.8256
  - Coverage-only: 0.7841
  - CAGP: 0.9520
- New pair (novel contexts): n=5,193
  - GP-only: 0.4207
  - Coverage-only: 1.0000
  - CAGP: 1.0000

**Note:** This is FB15k-237 with **frequency-based temporal simulation**, NOT real ICEWS14.

---

## ⏳ PENDING VERIFICATION

### 1. RelCondVar 50 Epochs

**Status:** Running (PID 2903)
**ETA:** ~20-50 minutes
**Will determine:** Keep or remove RelCondVar from paper

---

### 2. GPN Baseline

**Status:** Waiting on torch-geometric installation
**File:** `scripts/run_gpn_baseline.py` (ready)
**Will provide:** Graph-aware baseline comparison

---

## ❓ UNVERIFIABLE CLAIMS

### Cannot verify from existing files:

1. **ICEWS14 all results** (Table 1: 0.891, 0.912, etc.)
2. **WN18RR temporal OOD** (mentioned in abstract "0.87-0.97 across four benchmarks")
3. **Table 2 stratified evaluation** (specific AUROC per OOD type)
4. **Table 2 sample sizes** (2,134 emerging, 17,896 novel contexts)
5. **Standard OOD results** (outputs/final_results.json is incomplete)

---

## 🎯 RECOMMENDED ACTIONS

### URGENT (Before submission):

#### Option 1: Conservative Approach (RECOMMENDED) ✅

**Remove all unverifiable claims:**

1. **Remove ICEWS14 entirely from paper**
   - Delete Table 1
   - Remove "four benchmarks" from abstract
   - Remove ICEWS14 from experiments section
   - Change to: "We evaluate on FB15k-237 and YAGO3-10"

2. **Verify or remove Table 2**
   - Run stratified evaluation to get correct numbers
   - OR use notebook breakdown (new_entity: 2,223, new_pair: 5,193)
   - OR remove table and describe qualitatively

3. **Update abstract**
   ```latex
   OLD: "0.87-0.97 AUROC across four benchmarks, including ICEWS14"
   NEW: "0.94-0.99 AUROC on FB15k-237 and YAGO3-10 temporal OOD,
         with 100% empirical validation of theoretical assumptions"
   ```

4. **Focus on strengths**
   - A3 verification (100%) is the star result
   - YAGO3-10 results are solid (0.9424)
   - Theory is rigorous and validated

**Timeline:** 2-3 hours of paper editing

**Benefits:**
- Every claim is defensible
- Can't be caught with unverifiable numbers
- UAI values honesty over hype
- A3 validation alone is strong enough for acceptance

---

#### Option 2: Aggressive Approach (RISKY) ⚠️

**Run missing experiments:**

1. **Obtain ICEWS14 dataset** (4-6 hours)
   - Download from alternative sources
   - Process temporal splits
   - Run full experiments

2. **Run stratified evaluation** (2-3 hours)
   - Implement evaluation script
   - Verify Table 2 numbers
   - Get correct sample sizes

3. **Run WN18RR temporal** (3-4 hours)
   - Create temporal-like split
   - Run experiments
   - Verify "four benchmarks" claim

**Timeline:** 10-15 hours total

**Risks:**
- Results may not match paper claims
- ICEWS14 may still be unavailable
- May discover more discrepancies

---

### MEDIUM PRIORITY:

#### Update documentation:

1. **Create VERIFIED_RESULTS_ONLY.md**
   - List only results with source files
   - Provide file paths for verification
   - Clear mapping: claim → evidence

2. **Create DISCREPANCIES_LOG.md**
   - Document all mismatches found
   - Explain likely causes
   - Track resolution status

---

## 💡 BOTTOM LINE

### Current situation:

**Verified:**
- ✅ A3 assumption: 100% validation (HUGE win)
- ✅ YAGO3-10: 0.9424 AUROC (solid)
- ✅ FB15k-237 temporal: 0.9856 AUROC (strong)

**Unverified:**
- ❌ ICEWS14: Entire table (NEVER RUN)
- ❌ Table 2: Sample sizes don't match
- ⏳ RelCondVar: Waiting on 50 epochs result

### Paper strength assessment:

**With conservative approach (verified only):**
- Current: Weak Accept (6/10)
- With A3 + verified results: **Accept (7/10)**
- Reason: Solid theory + empirical validation + honest claims

**With current claims (includes unverified):**
- Risk: **Reject or Major Revision**
- Reason: Reviewer asks for ICEWS14 verification → can't provide → credibility damaged

### Recommendation:

**USE CONSERVATIVE APPROACH**

1. Remove ICEWS14 (was never run)
2. Fix or remove Table 2 (sample sizes wrong)
3. Focus on A3 validation + YAGO3-10 (both verified)
4. Be honest about limitations

**The A3 verification (100%) is strong enough to carry the paper even without ICEWS14.**

UAI reviewers will respect:
- 1 strong verified claim
over
- 4 weak unverified claims

---

## 📋 NEXT STEPS

1. **Wait for RelCondVar 50ep** (~20-50 min)
2. **Decide:** Conservative or aggressive?
3. **If conservative:** Start paper edits (2-3 hours)
4. **If aggressive:** Start running missing experiments (10-15 hours)

**My recommendation: Go conservative. The A3 result is the paper's strength, not ICEWS14.**
