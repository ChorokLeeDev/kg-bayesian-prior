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
├── notebooks/           # Experiment notebooks
│   ├── exp_distmult.ipynb   # DistMult baseline
│   ├── exp_ggpn.ipynb       # GGPN baseline
│   ├── exp_gpkge.ipynb      # GP-KGE (ours)
│   └── kernel_ablation.ipynb
├── results/             # Experiment results (JSON)
├── docs/                # Documentation
└── data/                # Data storage
```

## Baselines

| Model | Role |
|-------|------|
| DistMult | Deterministic baseline |
| GGPN | GP-based baseline (prior work) |
| **GP-KGE** | **Ours** |

## Installation

```bash
pip install -e .
```

## Datasets

- **FB15k-237**: Standard KG benchmark (14K entities, 237 relations)
- **WN18RR**: WordNet subset (41K entities, 11 relations)

## Results (FB15k-237)

*Results pending - experiments running with 3 seeds (42, 123, 456)*

| Model | MRR | H@1 | H@10 | ECE ↓ | Brier ↓ | AUROC |
|-------|-----|-----|------|-------|---------|-------|
| DistMult | - | - | - | - | - | - |
| GGPN | - | - | - | - | - | - |
| **GP-KGE (Ours)** | - | - | - | - | - | - |

## Evaluation Metrics

- **Link Prediction**: MRR, Hits@1/10
- **Calibration**: ECE, Brier Score
- **OOD Detection**: AUROC

## References

- Borovitskiy et al. (2021) - Matérn GP on Graphs
- Stadler et al. (2021) - Graph Posterior Network
- Chen et al. (2019) - UKGE
- Kondor & Lafferty (2002) - Diffusion Kernels
