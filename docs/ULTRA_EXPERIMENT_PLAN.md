# ULTRA Foundation Model: Coverage Blind Spot Validation

## Executive Summary

**Goal**: Empirically validate that ULTRA (a foundation KG model) inherits the coverage blind spot.

**Hypothesis**: ULTRA achieves ~0.5 AUROC on novel relational contexts because its NBFNet architecture does not track (entity, relation) co-occurrence.

**Status**: Ready to execute. Code exists, checkpoints available, GPU required (~5-10 min on Colab).

---

## 1. ULTRA Background

### 1.1 Architecture Overview

ULTRA (Universal Link-prediction Transformer for Relational graphs Architecture) uses a **two-level NBFNet** design:

```
Query: (h, r, ?)
    │
    ▼
┌─────────────────────────────────────────────┐
│ RelNBFNet (Relation-level message passing)  │
│ - 6-layer GNN on relation graph             │
│ - 4 edge types: h2h, t2t, h2t, t2h          │
│ - Output: relation representation for r     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ EntityNBFNet (Entity-level message passing) │
│ - 6-layer Bellman-Ford style propagation    │
│ - Initialize head h with relation embed r   │
│ - Propagate through ALL graph edges         │
│ - MLP scores each candidate tail t          │
└─────────────────────────────────────────────┘
    │
    ▼
Score: logit for each candidate tail
```

### 1.2 Key Insight: Why ULTRA Has the Blind Spot

ULTRA encodes:
- **Global graph structure** (connectivity patterns)
- **Relation semantics** (via relation graph)
- **Path-based reasoning** (multi-hop paths)

ULTRA does NOT encode:
- **Per-entity relation coverage** (which relations has entity e been seen with?)

**Example**: Entity "Barack Obama" has 1000+ training triples (nationality, birthplace, profession...). When ULTRA scores `(Barack_Obama, chemical_formula, ?)`:
- Obama has rich representations from extensive connectivity
- ULTRA will confidently score this despite zero training evidence for Obama + chemical_formula
- No mechanism to detect "Obama was never seen with chemical_formula"

### 1.3 Available Checkpoints

| Checkpoint | Training Data | Size | Parameters |
|------------|---------------|------|------------|
| `ultra_3g.pth` | FB15k237, WN18RR, CoDExMedium | 2 MB | 168,705 |
| `ultra_4g.pth` | Above + NELL995 | 2 MB | 168,705 |
| `ultra_50g.pth` | 50 diverse graphs | 2 MB | 168,705 |
| `ultraquery.pth` | For complex queries | 2 MB | ~170K |

All checkpoints are available locally at `/Users/i767700/Github/ultra_test/ckpts/`.

### 1.4 Inference Cost

- **CPU**: ~2+ hours for FB15k-237 test set (20K triples) - not practical
- **GPU (T4/V100)**: ~5-10 minutes - practical for Colab
- **GPU (A100)**: ~2-3 minutes

ULTRA is ~100-1000x slower than embedding lookup methods due to 6-layer message passing per query batch.

---

## 2. Experiment Design

### 2.1 Datasets

| Dataset | Entities | Relations | Test Triples | Expected Novel Ctx |
|---------|----------|-----------|--------------|-------------------|
| FB15k-237 | 14,541 | 237 | 20,466 | ~5,000 (25%) |
| WN18RR | 40,943 | 11 | 3,134 | ~800 (25%) |

FB15k-237 is primary (more relations = more novel contexts).

### 2.2 Test Triple Categorization

```python
for each test triple (h, r, t):
    if freq(h) <= threshold_25pct or freq(t) <= threshold_25pct:
        category = "emerging"  # Low-frequency entity
    elif coverage[h, r] == 0 or coverage[t, r] == 0:
        category = "novel_context"  # Entity seen, but not with this relation
    else:
        category = "in_distribution"  # Entity seen with this relation
```

### 2.3 Uncertainty Metric

ULTRA outputs logit scores (higher = more confident). We use:

```python
uncertainty = -score  # Negative score as uncertainty (energy-based)
```

### 2.4 Evaluation

1. **Emerging vs ID**: Can ULTRA detect low-frequency entities?
   - Expected: Moderate AUROC (0.6-0.8) because low connectivity = poor representations

2. **Novel Context vs ID**: Can ULTRA detect unseen (entity, relation) pairs?
   - **Expected: AUROC ~0.5** (random guessing = blind spot confirmed)

3. **Overall OOD**: Combined emerging + novel context vs ID
   - Expected: AUROC pulled down by novel context failures

---

## 3. Implementation

### 3.1 Existing Code

**Local script**: `/Users/i767700/Github/kg-bayesian-prior/scripts/run_ultra_experiment.py`
- Loads ULTRA model
- Loads FB15k-237/WN18RR
- Computes coverage matrix
- Categorizes test triples
- Computes AUROC for each category

**Colab notebook**: `/Users/i767700/Github/kg-bayesian-prior/notebooks/ultra_blind_spot_test.ipynb`
- Self-contained, installs dependencies
- Downloads checkpoint automatically
- Generates publication-ready figures

### 3.2 Running the Experiment

**Option A: Google Colab (Recommended)**
1. Upload `notebooks/ultra_blind_spot_test.ipynb` to Colab
2. Enable GPU runtime (T4 sufficient)
3. Run all cells (~10 minutes)
4. Download `ultra_ood_results.json` and `ultra_score_distributions.png`

**Option B: Local with GPU**
```bash
cd /Users/i767700/Github/kg-bayesian-prior
python scripts/run_ultra_experiment.py --dataset fb15k237
```

### 3.3 Expected Output

```
ULTRA OOD DETECTION RESULTS
============================================================

1. Overall OOD Detection (Emerging + Novel Context vs ID):
    AUROC: 0.58-0.65

2. Emerging Entity Detection (Emerging vs ID):
    AUROC: 0.65-0.75

3. Novel Relational Context Detection (Novel Context vs ID):
    >>> THIS IS THE KEY TEST <<<
    AUROC: 0.48-0.55  (near random = blind spot confirmed)
```

---

## 4. Expected Outcomes

### 4.1 Hypothesis Confirmation (Expected)

| Metric | Expected | Interpretation |
|--------|----------|----------------|
| Novel Ctx AUROC | ~0.50 | ULTRA cannot detect novel contexts |
| Emerging AUROC | ~0.70 | ULTRA partially detects low-connectivity |
| Overall AUROC | ~0.60 | Overall performance degraded by blind spot |

This would confirm: **The coverage blind spot is fundamental to embedding-based architectures, not fixable by foundation model scale.**

### 4.2 What Would Be Surprising?

| Result | Interpretation |
|--------|----------------|
| Novel Ctx AUROC > 0.65 | ULTRA somehow learned implicit coverage tracking |
| Novel Ctx AUROC > 0.80 | Our theoretical analysis has a gap - needs investigation |
| Emerging AUROC < 0.55 | ULTRA's connectivity-based representations don't help |

### 4.3 Comparison Table for Paper

| Method | Emerging | Novel Ctx | Overall | Source |
|--------|----------|-----------|---------|--------|
| ULTRA (foundation) | ~0.70 | ~0.50 | ~0.60 | This experiment |
| U_sem (GP-KGE) | 0.81 | ~0.50 | 0.59 | Our paper |
| U_str (Coverage) | 0.78 | 0.94 | 0.94 | Our paper |
| CAGP (Ours) | 0.89 | 0.97 | 0.97 | Our paper |

---

## 5. Literature Search: ULTRA + OOD

**Finding: No existing work on ULTRA OOD detection.**

Searched:
- GitHub issues: No OOD/uncertainty discussions
- OpenReview comments: Standard link prediction metrics only
- ULTRA paper (NeurIPS 2023): Reports MRR/Hits@k, no calibration/uncertainty analysis

This validates that our experiment would be a novel contribution.

---

## 6. Technical Details

### 6.1 ULTRA Score Mechanism

From `ultra/models.py`:

```python
# EntityNBFNet.forward():
# 1. Initialize head with relation embedding
# 2. 6 layers of Bellman-Ford message passing
# 3. Extract tail representations
# 4. MLP projects to scalar logit

score = self.mlp(feature).squeeze(-1)  # (batch_size, num_negative + 1)
```

**No uncertainty quantification**: ULTRA produces deterministic point estimates (logits) without confidence intervals, predictive variance, or Bayesian mechanisms.

### 6.2 Why NBFNet Cannot Track Coverage

NBFNet aggregates information via message passing:
- Each entity's representation is a function of its neighbors
- Relation-aware: uses different edge types in aggregation
- But: the representation encodes "connectivity structure", not "which specific relations were observed"

An entity with many neighbors will have rich representations regardless of which relations those neighbors connect through.

### 6.3 Dependencies

Already installed in `/Users/i767700/Github/ultra_test/`:
- PyTorch 2.1+ (compatible with 2.8.0)
- PyG 2.4+ (compatible with 2.6.1)
- torch-scatter, torch-sparse
- rspmm C++ extension (compiled for CPU/CUDA)

---

## 7. Timeline

| Task | Time |
|------|------|
| Upload notebook to Colab | 1 min |
| Run experiment (GPU) | 10 min |
| Analyze results | 5 min |
| Add to paper | 30 min |
| **Total** | **~45 min** |

---

## 8. Paper Integration

If results confirm hypothesis, add to Section 5 (Experiments):

> **Foundation Model Validation.** We evaluate ULTRA, a foundation model for KG reasoning pretrained on 3 graphs. Despite its sophisticated NBFNet architecture and zero-shot transfer capabilities, ULTRA achieves only X.XX AUROC on novel relational contexts (vs. 0.97 for CAGP). This confirms that the coverage blind spot is architectural, not a matter of scale.

And reference in Related Work:
> Foundation models like ULTRA (Galkin et al., 2023) achieve impressive zero-shot transfer but inherit the same blind spot: their entity representations encode connectivity patterns, not relation-specific coverage.
