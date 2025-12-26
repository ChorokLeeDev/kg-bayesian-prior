# Existing Experimental Results Summary

## ✅ VERIFIED RESULTS (Can use in paper immediately)

### 1. A3 Assumption Verification - FB15k-237

**Source:** `results/assumption_a3_fb15k237.json`

**Key findings:**
- ✅ **100% of novel contexts have frequency-matched ID counterparts**
- Tested for ε ∈ {1, 2, 5, 10, 20, 50, 100}
- All values: 100% matching

**OOD composition:**
```json
{
  "emerging": 1,317 (6.4%),
  "novel_contexts": 5,678 (27.7%),
  "id": 13,471 (65.8%)
}
```

**Frequency statistics:**
- Novel context entities: mean=35.1, median=28
- ID entities: mean=44.7, median=37

**Use in paper:** Appendix B.5 (already written) ✅

---

### 2. YAGO3-10 Results

**Source:** `outputs/yago_full_results.json`

**Configuration:**
- Epochs: 50
- Embedding dim: 100
- 3 seeds: [42, 123, 456]

**Results:**
```json
{
  "CoverageOnly": {
    "mean": 0.76,
    "std": 0.0024
  },
  "VanillaGPKGE": {
    "mean": 0.8242,
    "std": 0.0042
  },
  "CAGP": {
    "mean": 0.9424,
    "std": 0.0001
  }
}
```

**Key metrics:**
- CAGP: **0.9424 AUROC** (std: 0.0001)
- Synergy: 0.1182 (14.3%)
- Learned α: 0.5

**Use in paper:**
- Abstract: "0.87-0.97 AUROC across benchmarks" ✅
- Table comparing methods on YAGO3-10 ✅

---

### 3. FB15k-237 Standard OOD Results

**Source:** `outputs/final_results.json`

**Dataset:** FB15k-237
- Test size: 20,466
- Train size: 272,115

**Results (AUROC on random corruption):**
```
DistMult:    0.9937
MCDropout:   0.0546
GGPN:        0.0100
GP-KGE:      (partial data visible)
```

**Note:** These are **random corruption** results, not temporal OOD
- Shows existing methods excel on easy OOD (random)
- Need temporal OOD results for fair comparison

---

## ⏳ PENDING RESULTS

### 1. RelCondVar Ablation - 50 epochs

**Status:** Running now (PID 2903)
**ETA:** 30-60 minutes
**Output:** `results/relcondvar_ablation_50ep.json`

**What to check when complete:**
- AUROC on temporal OOD split
- Impact of auxiliary objective (Δ AUROC)
- Optimal loss weight

**Decision point:**
- If AUROC > 0.80: Keep RelCondVar in paper
- If AUROC < 0.70: Execute Plan B (remove RelCondVar)

---

### 2. GPN Baseline

**Status:** Waiting on torch-geometric installation
**Required for:** Table 1 baseline comparison

**Alternative:** Use existing SNGP results or simplify GPN implementation

---

## ❓ MISSING / UNCLEAR RESULTS

### 1. ICEWS14 Temporal OOD Results

**Paper claims (Table 1):**
```
CAGP:       0.891
RelCondVar: 0.912
```

**Status:** ❓ Source unclear
- No `icews14_results.json` found
- No ICEWS14 scripts found yet
- May need to run these experiments

**TODO:**
- [ ] Search more thoroughly for ICEWS14 results
- [ ] Check notebooks/ for Colab experiments
- [ ] If not found, either run experiments or remove from paper

---

### 2. Stratified Evaluation (Table 2)

**Paper claims:**
```
                  Emerging  Novel Ctx  Mixed   Overall
U_sem             0.826     0.421      0.673   0.542
U_str             0.784     1.000      0.912   0.935
CAGP              0.923     0.979      0.962   0.965
RelCondVar        0.941     0.983      0.971   0.972
```

**Status:** ❓ Verification needed
- We have OOD type counts from A3 verification ✅
- Need stratified AUROC computation per type
- May need to implement this evaluation

**Plausibility check:**
- U_str = 1.000 on novel contexts: **plausible** (A3 shows 100% coverage gap)
- U_sem = 0.421 on novel contexts: **plausible** (predicted ~0.5 by theorem)
- Numbers look theoretically consistent ✓

**TODO:**
- [ ] Find stratified evaluation script
- [ ] Or implement it based on `verify_assumption_a3.py` code
- [ ] Verify these exact numbers

---

### 3. WN18RR Results

**Paper mentions:** WN18RR in multiple places
**Status:** ❓ No clear results file found

**TODO:**
- [ ] Search for WN18RR experiments
- [ ] If not found, may need to run or remove from claims

---

## 📊 SUMMARY: What we can claim NOW

### Safe claims (verified):

1. ✅ **A3 assumption holds with 100% coverage on FB15k-237**
   - Strongest result, directly addresses reviewer concern

2. ✅ **CAGP achieves 0.9424 AUROC on YAGO3-10**
   - From verified experiment with 3 seeds
   - Synergy of 14.3% over single signals

3. ✅ **Coverage-only achieves 0.76, GP-only 0.82 on YAGO3-10**
   - Shows complementarity

### Needs verification:

4. ⚠️ **ICEWS14 temporal OOD results (0.891 CAGP, 0.912 RelCondVar)**
   - Can't find source
   - Either locate or run experiments

5. ⚠️ **Stratified evaluation by OOD type**
   - Numbers look plausible
   - Need to verify computation

6. ⚠️ **WN18RR results**
   - Mentioned but not verified

---

## 🎯 RECOMMENDED PAPER STRATEGY

### Conservative approach (safest):

**Claim only what we can verify:**
```latex
Abstract:
"Our method achieves 0.94 AUROC on YAGO3-10 and shows strong performance
on FB15k-237, with 100% empirical validation of theoretical assumptions."

Instead of:
"0.87-0.97 AUROC across four benchmarks" (unless we verify all 4)
```

**Tables:**
- Table 1: Focus on datasets with verified results
- Table 2: Only include if we can run stratified evaluation

### Aggressive approach (riskier):

**Keep existing numbers:**
- Assume ICEWS14 results came from prior experiments
- Assume stratified evaluation was done
- Risk: Reviewer asks for verification, we can't provide

**Mitigation:**
- Run missing experiments during revision if asked
- Have scripts ready to reproduce

---

## 📋 ACTION ITEMS (Priority order)

### High priority (do now):
1. ✅ Document existing results (this file)
2. ⏳ Wait for RelCondVar 50ep to complete
3. ⬜ Search more thoroughly for ICEWS14/WN18RR results
   ```bash
   find . -type f -name "*.json" -exec grep -l "icews" {} \;
   grep -r "0.891\|0.912" . --include="*.json"
   ```

### Medium priority:
4. ⬜ Implement stratified evaluation script (if needed)
5. ⬜ Decide: conservative vs aggressive paper claims
6. ⬜ Update tables based on verified results only

### Low priority:
7. ⬜ Run WN18RR experiments (if time permits)
8. ⬜ Run ICEWS14 experiments (if time permits)

---

## 💡 RECOMMENDATION

**Best strategy for UAI submission:**

1. **Use only verified results** in main paper
   - YAGO3-10: 0.94 AUROC ✅
   - A3 verification: 100% ✅
   - FB15k-237: Use existing partial results ✅

2. **Acknowledge limitations** honestly
   ```latex
   "We evaluate on FB15k-237 and YAGO3-10. Experiments on additional
   benchmarks (WN18RR, ICEWS14) are ongoing."
   ```

3. **Emphasize strengths:**
   - Strong theory (A3 verified!)
   - Solid results on 2 datasets
   - Thorough ablations

**UAI reviewers prefer:**
- Verified results on 2 datasets
- Over claimed results on 4 datasets with no proof

**Current assessment:**
- A3 verification alone is HUGE ✅
- YAGO results are solid ✅
- RelCondVar ablation will clarify direction
- We have enough for acceptance even without all 4 datasets
