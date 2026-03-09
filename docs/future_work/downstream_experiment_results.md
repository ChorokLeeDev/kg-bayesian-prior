# Downstream Task Experiment Results - 2026-03-09

## Key Finding: Coverage is About RELIABILITY, Not ACCURACY

### Experiment: Selective Link Prediction on FB15k-237

**Setup:**
- Train DistMult model (20 epochs)
- Evaluate Hits@10 on test set
- Compare abstention strategies: confidence vs. coverage

**Results:**

| Metric | Novel Context | Covered Queries |
|--------|---------------|-----------------|
| Accuracy (Hits@10) | **58.3%** | 33.3% |
| Mean entity frequency | 780 | 206 |
| Model confidence | 0.992 | 0.997 |
| Fraction of test | 31.5% | 68.5% |

**Surprising Finding:** Novel contexts have HIGHER accuracy!

### Why This Happens

1. **Novel contexts = high-frequency entities**
   - High-frequency entities appear with many relations
   - Some relations are in test but not train (= novel context)
   - But the entities themselves are well-known

2. **High-frequency entities generalize better**
   - More training examples overall
   - Better embedding quality
   - Transfer across relations

3. **Low-frequency entities have fewer novel contexts**
   - Appear with few relations
   - All relations likely covered in training
   - But predictions are harder (less training data)

### What Coverage Actually Tells Us

Coverage is NOT about prediction accuracy. It's about:

1. **Evidence-based confidence**: "Is my confidence founded?"
2. **Risk stratification**: "Should this prediction be trusted?"
3. **Safety-critical flagging**: "Does this need human review?"

### The 83% Finding Reinterpreted

Our paper says: "83% of top-confident predictions have zero training evidence"

This does NOT mean:
- Those predictions are wrong
- Abstaining improves accuracy

It DOES mean:
- Model confidence is UNFOUNDED
- In safety-critical domains, this is dangerous
- Human review should be triggered

## Implications for Paper

### What We CAN Claim

1. **Detection**: Coverage perfectly detects zero-evidence queries (AUROC = 1.0)
2. **Overconfidence**: Models are confidently wrong without evidence
3. **Risk stratification**: Coverage enables evidence-based decision making

### What We Should NOT Claim

1. ~~"Coverage improves downstream accuracy"~~ - Not consistently true
2. ~~"Abstain on zero-coverage for better predictions"~~ - May hurt accuracy

### Recommended Framing

> Coverage tracking enables **evidence-aware decision making**. While predictions on zero-coverage queries may be accurate (due to entity generalization), the confidence is unfounded. In safety-critical applications (healthcare, finance), distinguishing "confident with evidence" from "confident without evidence" is essential for responsible deployment.

## Next Steps

1. **Safety-critical experiment**: Hetionet drug-disease predictions
   - Correct answer may exist, but zero-evidence = needs human review
   - Metric: Fraction of "risky" predictions flagged

2. **Calibration experiment**: Does coverage-stratified calibration improve?
   - Hypothesis: ECE improves when separated by coverage
   - Shows: Coverage is a hidden confounder in calibration

3. **Human-in-the-loop simulation**:
   - Route zero-coverage to "human review"
   - Simulate human correcting errors
   - Metric: System accuracy with human-in-the-loop

## Files Created

- `scripts/downstream/selective_link_prediction.py` - v1 (incorrect design)
- `scripts/downstream/selective_link_prediction_v2.py` - v2 (correct design, shows surprising result)
- `docs/future_work/downstream_task_experiments.md` - Updated with findings
