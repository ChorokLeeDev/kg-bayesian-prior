# Held-Out Relation Experiment

## Overview

This experiment breaks the circularity critique: "Coverage detector is circular because it uses the same training data to define both coverage and what's OOD."

The held-out relation experiment decouples the OOD definition from the coverage computation by:
1. Splitting relations into train_rels (80%) and held_out_rels (20%)
2. Training the model on ALL training triples (for good embeddings)
3. Building coverage matrix using ONLY train_rels triples
4. Defining OOD purely as "test triples using held-out relations"

## Key Insight

The detector has never seen held-out relation coverage patterns, but the OOD definition is purely "relation was held out", NOT defined by coverage itself. This breaks circularity.

## Running the Experiment

```bash
# Run on both FB15k-237 and YAGO3-10 with 3 seeds
python3 scripts/run_held_out_relations.py

# Results will be saved to outputs/held_out_relations_results.json
```

## Configuration

- **Datasets**: FB15k-237 (237 relations), YAGO3-10 (37 relations)
- **Holdout fraction**: 20% of relations
  - FB15k-237: ~47 relations held out
  - YAGO3-10: ~7 relations held out
- **Seeds**: 42, 123, 456
- **Training**: 30 epochs, lr=1e-3, batch=1024, beta_KL=0.001, unc_weight=0.1
- **Models evaluated**: GPOnly (U_sem), CoverageOnly (U_str), CAGP

## Test Split Categories

- **OOD**: Test triples where relation ∈ held_out_rels
- **ID**: Test triples where relation ∈ train_rels AND both entities covered

## Expected Results

If coverage detector is NOT circular:
- **CoverageOnly** should still detect held-out relations as OOD (high AUROC)
- **CAGP** should combine semantic + structural uncertainty effectively

If coverage detector IS circular:
- Performance would collapse to random (AUROC ~0.5)

## Output Format

```json
{
  "fb15k237": {
    "per_seed": [...],
    "summary": {
      "GPOnly": {"auroc_mean": 0.xxx, "auroc_std": 0.xxx, ...},
      "CoverageOnly": {"auroc_mean": 0.xxx, "auroc_std": 0.xxx, ...},
      "CAGP": {"auroc_mean": 0.xxx, "auroc_std": 0.xxx, ...}
    }
  },
  "yago310": {...}
}
```

## Implementation Details

### Coverage Building
```python
# CRITICAL: Coverage built ONLY from train_rels triples
train_rels_triples = train[[r in train_rels for r in train[:, 1]]]
model.precompute_coverage(train_rels_triples)

# But model trains on ALL triples (including held-out relations)
model = train_model(model, train, device, ...)
```

### Test Categorization
```python
for i in range(len(test)):
    h, r, t = test[i]
    if r in held_out_rels:
        test_holdout_idx.append(i)  # OOD
    elif r in train_rels and coverage[h, r] == 1.0 and coverage[t, r] == 1.0:
        test_id_idx.append(i)  # ID
```

## Relationship to Paper

This experiment directly addresses reviewer concerns about circularity. Include results in:
- **Main paper**: Brief discussion in experiments section
- **Appendix**: Full results table and detailed methodology

## Notes

- WN18RR has only 11 relations, making it unsuitable for this experiment (holding out 2 relations doesn't test generalization well)
- FB15k-237 and YAGO3-10 have sufficient relations for meaningful splits
- The experiment validates that coverage generalizes to unseen relation patterns
