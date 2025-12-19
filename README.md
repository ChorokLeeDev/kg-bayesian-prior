# Knowledge Graph as Bayesian Prior for Uncertainty Quantification

A research project exploring relation-aware Gaussian Process priors on Knowledge Graphs for entity-level uncertainty quantification.

## Core Idea

Combine the expressiveness of Knowledge Graph embeddings with Bayesian uncertainty quantification through Gaussian Process priors that respect the heterogeneous relational structure of KGs.

## Research Gap

| Existing Work | Limitation | Our Solution |
|---------------|------------|--------------|
| GP on Graphs (Borovitskiy et al.) | Homogeneous edges only | Relation-specific kernels |
| Uncertain KGE (UKGE/BEUrRE) | Triple-level confidence | Entity-level posterior |
| Graph Posterior Network | Classification, ignores relations | Embedding uncertainty + relation-aware |
| Bayesian KG→BN (BIKG) | Symbolic/discrete | Continuous embedding space |

## Key Contributions

1. **Relation-aware kernel**: Learnable lengthscale per relation type
2. **Entity-level posterior**: Full posterior distribution over embeddings
3. **GPN axioms for KG**: Extended uncertainty propagation axioms
4. **Scalable inference**: Inducing point methods for large KGs

## Project Structure

```
kg-bayesian-prior/
├── src/
│   ├── models/          # Model implementations
│   ├── data/            # Data loading and processing
│   ├── kernels/         # GP kernel implementations
│   ├── utils/           # Utility functions
│   └── evaluation/      # Metrics and evaluation
├── experiments/         # Experiment scripts
├── configs/             # Configuration files
├── tests/               # Unit tests
├── docs/                # Documentation and literature review
├── notebooks/           # Jupyter notebooks
├── data/                # Data storage
└── outputs/             # Results and saved models
```

## Installation

```bash
pip install -e .
```

## Datasets

- **FB15k-237**: Standard KG benchmark (14K entities, 237 relations)
- **CN15k**: ConceptNet with confidence scores
- **YAGO3-10**: Large-scale KG (123K entities)

## Results (FB15k-237)

| Model | MRR | H@1 | H@10 | ECE ↓ | Brier ↓ |
|-------|-----|-----|------|-------|---------|
| DistMult | 0.2411 | 0.1707 | 0.3817 | 0.2118 | 0.1417 |
| MCDropout | 0.2351 | 0.1621 | 0.3842 | 0.2119 | 0.1385 |
| GGPN | 0.1205 | 0.0540 | 0.2527 | 0.1965 | 0.1519 |
| **GP-KGE (Ours)** | **0.2545** | **0.1761** | **0.4171** | **0.1171** | **0.1013** |

**Key Finding:** GP-KGE achieves **40.4% better calibration (ECE)** compared to GGPN, validating that relation-aware kernels with proper Bayesian treatment yield well-calibrated uncertainty estimates.

## Evaluation Metrics

- **Link Prediction**: MRR, Hits@1/3/10
- **Calibration**: ECE, Brier Score
- **OOD Detection**: AUROC, AUPR
- **Selective Prediction**: Risk-Coverage curves

## References

- Borovitskiy et al. (2021) - Matérn GP on Graphs
- Stadler et al. (2021) - Graph Posterior Network
- Chen et al. (2019) - UKGE
- Kondor & Lafferty (2002) - Diffusion Kernels
