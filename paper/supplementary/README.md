# Supplementary Material: Knowledge Graph Uncertainty Research Should Adopt Scope Conditions

This supplementary material contains code to reproduce the experiments in our position paper.

## Contents

```
supplementary/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── test_all_biomedical.py    # Test coverage paradox on biomedical KGs
│   ├── test_new_datasets.py      # Test on encyclopedic KGs (OpenBioLink, DRKG)
│   ├── test_new_temporal.py      # Test on temporal KGs (ICEWS, GDELT)
│   └── generate_paper_table.py   # Generate Table 1 from results
└── results/
    └── all_results.json          # Pre-computed results for all 22 datasets
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Results

### Quick Start (Pre-computed Results)

To regenerate Table 1 from pre-computed results:
```bash
python scripts/generate_paper_table.py
```

### Full Reproduction

To reproduce all experiments from scratch:

1. **Biomedical KGs** (DRKG, PrimeKG, Hetionet):
```bash
python scripts/test_all_biomedical.py
```

2. **Encyclopedic KGs** (FB15k-237, CoDEx, YAGO3-10, ConceptNet, NELL):
```bash
python scripts/test_new_datasets.py
```

3. **Temporal KGs** (ICEWS14, ICEWS18, GDELT):
```bash
python scripts/test_new_temporal.py
```

## Model

All experiments use ComplEx (Trouillon et al., 2016) with:
- Embedding dimension: 50
- Training epochs: 10-15
- Optimizer: Adam (lr=0.001)
- Loss: Margin ranking loss (margin=1.0)
- Negative samples: 5 per positive

## Coverage Definition

For a test triple (h, r, t):
- **Full coverage**: Both h and t have appeared with relation r in training
- **Partial coverage**: Exactly one of h or t has appeared with relation r
- **Zero coverage**: Neither h nor t has appeared with relation r

## Key Metric

**Coverage Paradox**: When partial coverage achieves higher Hits@10 than full coverage.
- Present in all 13 multi-relational KGs (AUROC < 0.5)
- Absent in all 9 structured KGs (AUROC > 0.6)

## Contact

For questions about reproducing results, please open an issue on the anonymous repository or contact the authors after the review period.
