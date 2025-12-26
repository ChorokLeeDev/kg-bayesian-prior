# Continuous Coverage Baseline Analysis

**Addresses Reviewer Concern**: "Binary coverage doesn't capture frequency - why not use continuous counts?"

## Motivation

The original CAGP uses binary coverage: `c(e,r) ∈ {0,1}` indicating whether entity `e` has been observed with relation `r` in training. A natural question is whether using actual co-occurrence counts provides richer information for uncertainty quantification.

## Continuous Coverage Variants

We implement and compare 6 coverage formulations:

### 1. **Binary** (Baseline)
```
c(e,r) = 1 if (e,r) observed, else 0
U_coverage = 2 - c(h,r) - c(t,r)
```
- **Rationale**: Simple presence/absence detection
- **Range**: U ∈ {0, 1, 2}

### 2. **Raw Counts**
```
c(e,r) = count of (e,r) co-occurrences
U_coverage = 2 - c(h,r)/max_count - c(t,r)/max_count
```
- **Rationale**: More frequent = better constrained
- **Range**: U ∈ [0, 2] (continuous)

### 3. **Log-Scaled**
```
c(e,r) = log(1 + count)
U_coverage = 2 - c(h,r)/max_log - c(t,r)/max_log
```
- **Rationale**: Diminishing returns from high frequencies
- **Theory**: Similar to GP variance-frequency relationship (decreases logarithmically)

### 4. **Normalized** (Per-Relation)
```
c(e,r) = count / max_count_for_relation_r
U_coverage = 2 - c(h,r) - c(t,r)
```
- **Rationale**: Normalize by relation popularity (rare relations → higher counts matter more)
- **Advantage**: Handles long-tail relation distributions

### 5. **Inverse Frequency**
```
c(e,r) = 1 - 1/(1 + count)
U_coverage = 2 - c(h,r) - c(t,r)
```
- **Rationale**: Explicitly models uncertainty as inverse of frequency
- **Connection**: IDF-like weighting

### 6. **TF-IDF Style**
```
TF(e,r) = count(e,r)
IDF(r) = log(N_entities / entities_with_r)
c(e,r) = TF(e,r) * IDF(r)  [normalized]
U_coverage = 2 - c(h,r) - c(t,r)
```
- **Rationale**: High count with rare relation → low uncertainty; High count with common relation → moderate uncertainty
- **Advantage**: Balances entity-relation specificity with relation rarity

## Experimental Design

### Datasets
- FB15k-237 (14.5K entities, 237 relations)
- WN18RR (40.9K entities, 11 relations)

### OOD Settings
1. **Random Corruption** (Easy): Replace tail with random entity
   - Tests: Can uncertainty detect implausible triples?

2. **Temporal OOD** (Realistic): Train on 70% of data, test on remaining 30%
   - Tests: Can uncertainty detect emerging entities + novel contexts?

### Metrics
- AUROC (primary)
- AUPR (precision-recall trade-off)
- Learned α (how much does model rely on coverage vs GP variance?)

## Expected Outcomes

### Hypothesis 1: Continuous Coverage Helps on Dense Relation Graphs
For FB15k-237 (237 relations), continuous counts may distinguish between:
- Entity seen with relation once (exploratory)
- Entity seen with relation 100 times (well-established)

### Hypothesis 2: Binary Sufficient for Sparse Relation Graphs
For WN18RR (11 relations), binary may suffice because:
- Each relation occurs frequently → counts less informative
- Novel contexts dominated by missing coverage (binary sufficient)

### Hypothesis 3: Log-Scaled Performs Best
Theory suggests GP variance scales logarithmically with frequency:
```
σ²(e) ≈ 1/log(1 + freq(e))
```
Log-scaled coverage mirrors this relationship.

## Implementation Details

**Model**: `ContinuousCoverageModel` (extends `CoverageAugmentedGPKGE`)
- All 6 coverage matrices precomputed once
- `coverage_mode` parameter selects which to use
- No additional computational cost at inference

**Training**: Same as CAGP
- 50 epochs, Adam optimizer (lr=0.001)
- KL weight = 0.01
- Batch size = 2048

## Interpretation Guide

### If Continuous > Binary:
- **Actionable**: Use log-scaled or TF-IDF coverage in production
- **Theory**: Theorem 1 should be extended to continuous coverage
- **Paper revision**: Add continuous coverage as main method

### If Binary ≈ Continuous:
- **Justification**: Binary coverage captures the essential signal (presence/absence)
- **Defense**: Simplicity preferred when performance equal (Occam's razor)
- **Paper revision**: Add this ablation to justify binary choice

### If Continuous < Binary:
- **Analysis**: Overfitting to training frequencies? Continuous coverage may conflate:
  - Actual entity-relation uncertainty
  - Training data sampling artifacts (popular entities overrepresented)

## Script Usage

```bash
# Full experiment (all datasets, all modes)
python scripts/run_continuous_coverage_ablation.py

# Expected runtime: ~30-45 minutes on CPU
# Outputs: outputs/continuous_coverage_ablation.json
```

## Integration with Paper

### If Results Support Continuous Coverage:

**Section 4.2 Update**:
```
Structural uncertainty from coverage:
U_str(h, r, t) = 2 - f(h,r) - f(t,r)

where f(e,r) = log(1 + count(e,r)) / max_r log(1 + count(·,r))
captures the logarithmic relationship between observation frequency
and epistemic uncertainty.
```

**Table 2 Addition**:
Add row comparing binary vs continuous on complementarity task.

**Appendix B**:
Full ablation table with all 6 variants.

### If Results Show Binary Sufficient:

**Section 5.4 Addition**:
```
We evaluated continuous coverage (raw counts, log-scaled, TF-IDF)
but found binary coverage captures the essential signal (AUROC
difference <0.02). This suggests the presence/absence of entity-
relation observation dominates over frequency information.
```

## Connection to Theorem 1

Current theorem assumes binary coverage: `c(e,r) ∈ {0,1}`

If continuous coverage helps, Theorem 1 proof requires adjustment:

**Part (iii)** currently states: "Novel contexts have c(h,r)=0 or c(t,r)=0 by definition, so U_str ≥ 1"

With continuous coverage:
- Novel contexts have `f(h,r)=0` or `f(t,r)=0` (never observed)
- But emerging entities may have `0 < f(e,r) < ε` (observed once)
- Need to relax perfect separation claim to "near-perfect" with threshold ε

## References

- IDF weighting: Sparck Jones (1972)
- Frequency-uncertainty relationship in GPs: Rasmussen & Williams (2006)
- Coverage in KG completion: Safavi & Koutra (2020)
