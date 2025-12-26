# Paper Claims Verification Checklist

## ⚠️ CRITICAL: Verify all numbers in paper match experiments

Before submission, check that every AUROC/metric in the paper came from actual experiments.

---

## 📊 KEY CLAIMS IN PAPER

### Table 1: ICEWS14 Temporal OOD Results

**Paper claims (sections/experiments_uai.tex, line 42-43):**
```
CAGP (learned α)              0.891  0.847  0.781
RelCondVar (learned σ²(e,r))  0.912  0.873  0.805
```

**Status:** ❓ NEEDS VERIFICATION
- Where did these numbers come from?
- Which experiment script generated them?
- On what exact dataset/split?

**Action items:**
- [ ] Find source of 0.912 RelCondVar result
- [ ] Check if ICEWS14 experiments were actually run
- [ ] Verify these aren't FB15k-237 results mislabeled

---

### Table 2: FB15k-237 Stratified OOD (Complementarity validation)

**Paper claims (sections/experiments_uai.tex, line 64-70):**
```
                  Emerging  Novel Ctx  Mixed   Overall
U_sem             0.826     0.421      0.673   0.542
U_str             0.784     1.000      0.912   0.935
Simple avg        0.891     0.978      0.945   0.951
CAGP              0.923     0.979      0.962   0.965
RelCondVar        0.941     0.983      0.971   0.972
```

**Status:** ❓ NEEDS VERIFICATION
- Stratified evaluation requires specific code
- Did we actually compute these splits?

**Our current results:**
- ✅ We have A3 verification: 5,678 novel contexts, 1,317 emerging, 13,471 ID
- ❌ We don't have stratified AUROC per OOD type yet

**Action items:**
- [ ] Check if stratified evaluation script exists
- [ ] If not, need to implement it
- [ ] Or remove these specific numbers

---

### Abstract Claims

**Paper claims (sections/abstract_uai.tex, line 14):**
```
"Combining signals via learned weights yields 0.87--0.97 AUROC across four benchmarks"
```

**Status:** ⚠️ PARTIALLY VERIFIED
- Range 0.87-0.97 is broad
- Need to verify on ALL four datasets: FB15k-237, WN18RR, YAGO3-10, ICEWS14

**Our current results:**
- ✅ A3 verification on FB15k-237 (100% matching)
- ⏳ RelCondVar ablation running (50 epochs)
- ❌ No ICEWS14 results yet
- ❌ No WN18RR results yet
- ❌ No YAGO3-10 results yet

---

## 🔍 INVESTIGATION NEEDED

### 1. Check existing result files

**Files found:**
```
outputs/final_results.json
outputs/advanced_experiments_results.json
outputs/yago_full_results.json
outputs/coverage_only_results.json
```

**TODO:**
- [ ] Read each file to see what results they contain
- [ ] Match against paper tables
- [ ] Document which experiments were actually run

### 2. Check for ICEWS14 experiments

**Likely locations:**
- `scripts/run_icews_*.py`
- `outputs/icews_*.json`
- `notebooks/` (might have colab experiments)

**TODO:**
- [ ] Search for ICEWS14-related scripts
- [ ] Check if results exist somewhere
- [ ] If not, ICEWS14 results might be hypothetical!

### 3. Check for stratified evaluation

**Needed for Table 2 (Complementarity validation):**
- Partition test set into: emerging, novel context, mixed
- Compute AUROC separately for each partition

**TODO:**
- [ ] Search for stratified evaluation code
- [ ] Check if `verify_theorem.py` or similar exists
- [ ] May need to implement this

---

## 🚨 RED FLAGS

### Concern 1: RelCondVar 0.912 vs our 0.501

**Paper:**
- RelCondVar achieves 0.912 on ICEWS14 temporal OOD

**Our ablation (20 epochs):**
- RelCondVar achieves 0.501 (random!)

**Possible explanations:**
1. ✅ 20 epochs too few → running 50 epochs now
2. ⚠️ Different dataset (paper uses ICEWS14, we used FB15k-237)
3. ⚠️ Different split (paper uses ground-truth temporal, we used simple 70/30)
4. 🚨 Paper numbers might be aspirational, not actual

### Concern 2: Stratified results too perfect

**Table 2 shows:**
- Structural uncertainty: 1.000 AUROC on novel contexts (perfect!)
- Combination: 0.972 AUROC overall

**Our A3 verification:**
- 100% of novel contexts matched → plausible that coverage detects perfectly
- But need actual experiments to verify

### Concern 3: Four benchmarks claimed

**Abstract says: "across four benchmarks"**

**Evidence:**
- FB15k-237: Some results exist
- WN18RR: Mentioned in paper, unclear if experiments run
- YAGO3-10: `yago_full_results.json` exists (check contents)
- ICEWS14: No evidence found yet

---

## ✅ ACTION PLAN

### Immediate (while waiting for 50ep results):

1. **Check existing result files**
   ```bash
   cat outputs/final_results.json
   cat outputs/yago_full_results.json
   cat outputs/advanced_experiments_results.json
   ```

2. **Search for ICEWS14 experiments**
   ```bash
   find . -name "*icews*" -type f
   grep -r "icews14" scripts/
   ```

3. **Document findings**
   - What results actually exist?
   - What claims in paper are supported?
   - What claims need new experiments?

### After 50ep RelCondVar completes:

4. **If results are good (>0.80 AUROC):**
   - Use those numbers in paper
   - Run on other datasets if time permits

5. **If results are bad (<0.70 AUROC):**
   - Execute Plan B (remove RelCondVar)
   - Update all claims to CAGP only
   - Simplify story

---

## 📝 VERIFICATION CHECKLIST

Before final submission:

### Abstract
- [ ] AUROC range (0.87-0.97) verified on all 4 datasets
- [ ] "67% improvement" calculation is correct
- [ ] All datasets mentioned actually have results

### Method Section
- [ ] All equations have been implemented
- [ ] Coverage matrix computation is described accurately
- [ ] α learning described matches implementation

### Experiments
- [ ] Table 1 numbers match experiment outputs
- [ ] Table 2 stratified results verified
- [ ] Table 3 (if exists) verified
- [ ] All baseline comparisons use same experimental setup

### Appendix
- [ ] A3 verification numbers match our results ✅
- [ ] Ablation tables match actual experiments
- [ ] Error analysis numbers are real
- [ ] All supplementary results verified

---

## 🎯 CONSERVATIVE APPROACH

**If verification shows problems:**

### Option 1: Run missing experiments
- Pro: Complete picture
- Con: May take days

### Option 2: Reduce scope of claims
- Pro: Honest, defensible
- Con: Weaker paper

### Option 3: Use existing results only
```latex
Instead of: "across four benchmarks"
Write: "on FB15k-237 and YAGO3-10"

Instead of: "RelCondVar achieves 0.912"
Write: "CAGP achieves 0.89" (if we can verify this)
```

---

## 💡 BOTTOM LINE

**Current status:**
- ✅ A3 verification: Solid, verified, ready for paper
- ⚠️ Other claims: Need verification
- ⏳ 50ep ablation: Will clarify RelCondVar situation

**Recommendation:**
1. Check existing result files NOW
2. Document what's real vs aspirational
3. Prepare to scale back claims if needed
4. UAI values honesty over hype

**Better to have:**
- Verified results on 2 datasets
- Than claimed results on 4 datasets with no proof
