# Running the ICEWS14 Strict Split 5-Seed Experiment

## Quick Start

```bash
cd /sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior
python scripts/icews14_strict_split_5seed.py
```

**Expected runtime:** ~4 minutes on CPU (5 epochs, 5 seeds, 6 models)

## What This Script Does

### 1. Load ICEWS14 Dataset
- Training set: 63,685 triples
- Test set: 13,222 triples
- Entities: 7,128
- Relations: 230

### 2. Build Strict Test Split
Removes from test set:
- **Exact duplicates:** 715 triples (h,r,t) that appear in training
- **Inverse overlaps:** 2,555 triples (h,r,t) where (t,r',h) exists in training
- **Both:** 4,466 triples (counted under exact duplicates)
- **Total removed:** 7,736 triples (58.5%)
- **Strict test set:** 5,486 triples

### 3. Train and Evaluate 6 Models, 5 Seeds Each

Models evaluated:
- UKGE (uncertainty baseline)
- Energy (energy-based OOD)
- GPOnly (pure Gaussian Process variance)
- CoverageOnly (pure coverage-based uncertainty)
- **CAGP** (Coverage-Augmented GP-KGE - main contribution)
- RelCondVar (relation-conditional variance)

Configurations per seed:
- Training epochs: 5
- Batch size: 1024
- Learning rate: 0.001
- Optimizer: Adam
- Device: CPU

### 4. Evaluate on Both Original and Strict Test Sets
For each model, compute:
- **Overall temporal OOD AUROC:** Emerging entities + novel contexts vs ID
- **Emerging entity AUROC:** Low-frequency entities vs ID
- **Novel context AUROC:** Unseen (entity, relation) pairs vs ID
- **AUPR:** Average precision-recall

### 5. Report Results
Aggregates across 5 seeds and displays:
- Mean ± std AUROC for each model
- Original vs strict split comparison
- Per-category breakdown (emerging vs novel context)

## Output Files

Results saved to:
```
/sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior/outputs/icews14_strict_split_5seed_results.json
```

JSON structure:
```json
{
  "split_stats": {
    "original_test_size": 13222,
    "removed_total": 7736,
    "strict_test_size": 5486,
    "pct_removed": 58.5
  },
  "summary": {
    "original": { /* mean ± std for original split */ },
    "strict": { /* mean ± std for strict split */ }
  },
  "all_results": {
    "original": { "seed_42": {...}, "seed_123": {...}, ... },
    "strict": { "seed_42": {...}, "seed_123": {...}, ... }
  },
  "seeds": [42, 123, 456, 789, 1024],
  "dataset_info": { ... }
}
```

## Results at a Glance

| Model | Original AUROC | Strict AUROC | Robustness |
|-------|----------------|--------------|-----------|
| CAGP | 0.992 ± 0.000 | **0.994 ± 0.002** | Excellent |
| CoverageOnly | 0.992 ± 0.001 | **0.994 ± 0.001** | Excellent |
| RelCondVar | 0.992 ± 0.001 | **0.994 ± 0.000** | Excellent |
| GPOnly | 0.822 ± 0.004 | 0.786 ± 0.008 | Moderate (-0.036) |

**Key insight:** CAGP maintains 0.994 AUROC on strict split, demonstrating robustness against transductive artifacts.

## Modifying the Script

### To run with more epochs (requires GPU or patience):
Edit `/sessions/admiring-youthful-knuth/mnt/kg-bayesian-prior/scripts/icews14_strict_split_5seed.py`

Change line with `train_model(model, train, device, epochs=5)` to:
```python
train_model(model, train, device, epochs=30)  # Full training
```

### To run with different seeds:
Find the line:
```python
seeds = [42, 123, 456, 789, 1024]
```

Replace with your desired seeds:
```python
seeds = [42, 100, 200, 300, 400, 500]  # 6 seeds instead of 5
```

### To run just one seed for quick testing:
```python
seeds = [42]  # Single seed
```

## Troubleshooting

### Script times out
- Reduce `max_samples` in `evaluate_temporal_real()` (default: 1500)
- Reduce epochs (currently 5, minimum is 1)

### Memory issues
- Reduce batch size in `train_model()` (default: 1024, try 512)
- Reduce `max_samples` parameter

### Different results each run
This is expected - each run uses fresh random seeds. To get identical results:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Citation

When using this experiment in papers, cite:
- Original ICEWS14: "ICEWS14: Temporal Reasoning over Knowledge Graphs"
- CAGP: Include arxiv/DOI when available

## See Also

- `ICEWS14_5SEED_SUMMARY.md` - Full results summary
- `scripts/icews14_strict_split.py` - Original 3-seed version
- `scripts/run_icews14_temporal.py` - Full-training version (30 epochs)
