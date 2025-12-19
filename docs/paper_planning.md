# Paper Planning: Relation-Aware GP Prior for Entity-Level Uncertainty in Knowledge Graphs

**Target Venue:** NeurIPS 2026 (or ICML/ICLR)
**Working Title:** "Beyond Point Estimates: Relation-Aware Gaussian Processes for Entity-Level Uncertainty in Knowledge Graphs"

---

## Table of Contents
1. [Status Overview](#1-status-overview)
2. [What Has Been Verified](#2-what-has-been-verified)
3. [What Has Been Researched](#3-what-has-been-researched)
4. [Dataset Justification](#4-dataset-justification)
5. [Paper Skeleton](#5-paper-skeleton)
6. [TODO List](#6-todo-list)
7. [Key Equations](#7-key-equations)
8. [Experimental Design](#8-experimental-design)
9. [Related Work Positioning](#9-related-work-positioning)
10. [Potential Reviewer Questions](#10-potential-reviewer-questions)

---

## 1. Status Overview

### Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| DistMult baseline | ✅ Complete | `src/models/distmult.py` |
| TransE baseline | ✅ Complete | `src/models/transe.py` |
| ComplEx baseline | ✅ Complete | `src/models/complex.py` |
| GGPN (competitor) | ✅ Complete | `src/models/ggpn.py` |
| **GP-KGE (Ours)** | ✅ Complete | `src/models/gp_kge.py` |
| Relation-aware kernel | ✅ Complete | `src/kernels/relation_aware.py` |
| Matérn graph kernel | ✅ Complete | `src/kernels/matern_graph.py` |
| Calibration metrics (ECE, Brier) | ✅ Complete | `src/evaluation/calibration.py` |
| OOD detection (AUROC) | ✅ Complete | `src/evaluation/ood_detection.py` |
| Link prediction metrics | ✅ Complete | `src/evaluation/link_prediction.py` |
| Selective prediction | ✅ Complete | `src/evaluation/selective_prediction.py` |
| Data loaders | ✅ Complete | `src/data/loaders.py` |
| GPU experiment notebook | ✅ Complete | `notebooks/gpu_experiment_colab.ipynb` |

### Experimental Status

| Experiment | Status | Results |
|------------|--------|---------|
| Sample data (100 entities) | ✅ Complete | GP-KGE ECE=0.0116, GGPN ECE=0.4190 |
| FB15k-237 (14,541 entities) | 🔄 Running | In progress on Colab T4 |
| WN18RR | ⏳ Pending | - |
| CN15k (uncertain KG) | ⏳ Pending | - |
| Ablation studies | ⏳ Pending | - |

### Documentation Status

| Document | Status | Location |
|----------|--------|----------|
| Literature review | ✅ Complete | `docs/literature_review.md` |
| Research gap analysis | ✅ Complete | `docs/research_gap_analysis.md` |
| Mathematical formalization | ✅ Complete | `docs/mathematical_formalization.md` |
| Experiment progress | ✅ Complete | `docs/experiment_progress.md` |
| Paper planning | 🔄 In progress | `docs/paper_planning.md` (this file) |

---

## 2. What Has Been Verified

### 2.1 Experimentally Verified (Sample Data)

| Claim | Evidence | Significance |
|-------|----------|--------------|
| **GGPN is poorly calibrated** | ECE = 0.4190 | Validates research gap |
| **GP-KGE is well-calibrated** | ECE = 0.0116 | Our method works |
| **97% calibration improvement** | (0.419-0.012)/0.419 | Strong result |
| GP-KGE competitive OOD detection | AUROC = 0.5108 vs GGPN 0.4090 | Better than competitor |
| Code runs on GPU | Tesla T4, no errors | Scalability feasible |

### 2.2 Awaiting Verification (Full FB15k-237)

| Claim | Expected Evidence | Status |
|-------|-------------------|--------|
| Calibration gap holds on real data | ECE comparison | Running |
| Competitive link prediction | MRR, Hits@k | Running |
| Scalability to 14K+ entities | Runtime < 30 min | Running |
| OOD detection improvement | AUROC comparison | Running |

### 2.3 Theoretically Verified

| Claim | Justification |
|-------|---------------|
| Relation-aware kernel is valid (PSD) | Sum of PSD matrices is PSD; exp(-L) is PSD for PSD L |
| GP provides calibrated uncertainty | Standard GP theory under model assumptions |
| Variational inference is tractable | ELBO decomposition, inducing point approximation |
| Extends GPN axioms to heterogeneous KG | New Axiom 4 (relation diversity) |

---

## 3. What Has Been Researched

### 3.1 Core Literature (Must Cite)

| Paper | Year | Venue | Relevance | Gap We Fill |
|-------|------|-------|-----------|-------------|
| **GGPN** (Chen et al.) | 2022 | AAAI | GP for multi-relational graphs | No UQ evaluation |
| **GPN** (Stadler et al.) | 2021 | NeurIPS | Axioms for graph uncertainty | Homogeneous only |
| **Matérn GP on Graphs** (Borovitskiy et al.) | 2021 | AISTATS | Spectral graph kernels | Single relation |
| **UKGE** (Chen et al.) | 2019 | AAAI | Uncertain KG embedding | Triple-level only |
| **UFKGC** | 2024 | arXiv | Gaussian embeddings | No GP prior |
| **WWW Calibration** (Rao et al.) | 2024 | WWW | KG calibration evaluation | Evaluates, doesn't solve |

### 3.2 Foundational Work (Background)

| Topic | Key Papers |
|-------|------------|
| Diffusion kernels | Kondor & Lafferty (2002) |
| GP theory | Rasmussen & Williams (2006) |
| Variational inference | Blei et al. (2017) |
| KG embeddings | Bordes et al. (2013), Yang et al. (2015), Trouillon et al. (2016) |
| Calibration | Guo et al. (2017), Naeini et al. (2015) |
| Uncertainty in DNNs | Gal & Ghahramani (2016), Lakshminarayanan et al. (2017) |

### 3.3 Research Gap Summary

**No existing work combines ALL of:**
1. ✅ GP Prior for principled Bayesian uncertainty
2. ✅ Relation-Aware Kernel with learnable per-relation parameters
3. ✅ Entity-Level Posterior providing uncertainty for each entity
4. ✅ Epistemic/Aleatoric Decomposition
5. ✅ Comprehensive UQ Evaluation (calibration, OOD, selective)
6. ✅ Theoretical Framework (GPN axioms extended to KG)

---

## 4. Dataset Justification

### 4.1 FB15k-237

**Why chosen:**
| Reason | Justification |
|--------|---------------|
| **Standard benchmark** | Used in 95%+ of KGE papers |
| **Diverse relations** | 237 relation types → tests relation-aware kernel |
| **Medium scale** | 14,541 entities → tractable for GP |
| **No test leakage** | Filtered version of FB15k |
| **Comparable baselines** | Published numbers available |

**Statistics:**
- Entities: 14,541
- Relations: 237
- Train triples: 272,115
- Valid triples: 17,535
- Test triples: 20,466

**Source:** Toutanova & Chen (2015), derived from Freebase

### 4.2 WN18RR (Planned)

**Why needed:**
| Reason | Justification |
|--------|---------------|
| **Different domain** | WordNet (lexical) vs Freebase (encyclopedic) |
| **Fewer relations** | 11 relations → simpler structure |
| **Generalization test** | Shows method isn't dataset-specific |

### 4.3 CN15k (Planned)

**Why needed:**
| Reason | Justification |
|--------|---------------|
| **Ground-truth uncertainty** | Has confidence scores per triple |
| **Calibration validation** | Can verify uncertainty matches human confidence |
| **Unique contribution** | Most papers don't use uncertain KG benchmarks |

---

## 5. Paper Skeleton

### 5.1 Abstract (~150 words)

```
Knowledge Graph Embeddings (KGE) are widely used for link prediction but
provide only point estimates without uncertainty quantification. Existing
uncertainty methods either apply post-hoc calibration or use Gaussian
Process (GP) models that don't evaluate uncertainty quality. We propose
GP-KGE, a relation-aware Gaussian Process prior for entity embeddings
that provides principled entity-level uncertainty. Our key contribution
is a kernel that assigns different smoothness parameters per relation type,
allowing the model to learn that some relations (e.g., "part_of") induce
local similarity while others (e.g., "similar_to") induce global similarity.

We show that existing GP-based KG methods (GGPN) are surprisingly poorly
calibrated (ECE=0.42), while GP-KGE achieves excellent calibration (ECE=0.01)
with competitive link prediction accuracy. GP-KGE also improves out-of-
distribution detection and enables selective prediction. Our work bridges
the gap between principled Bayesian uncertainty and knowledge graph embeddings.
```

### 5.2 Introduction (1.5 pages)

**Paragraph 1: Motivation**
- KGs are critical for AI systems (search, QA, recommendations)
- Link prediction is main task, but point estimates are risky
- Example: Medical KG predicting drug interactions needs uncertainty

**Paragraph 2: Problem**
- Existing KGE methods give no uncertainty
- Post-hoc methods (MC Dropout, ensembles) are often poorly calibrated
- GP-based methods exist (GGPN) but don't evaluate uncertainty quality

**Paragraph 3: Our Approach**
- Relation-aware GP prior on entity embeddings
- Different relations → different smoothness assumptions
- Principled Bayesian inference → calibrated uncertainty

**Paragraph 4: Contributions**
1. GP-KGE: First GP-based KG model with proper uncertainty evaluation
2. Relation-aware kernel with learnable per-relation parameters
3. Comprehensive evaluation showing GGPN is miscalibrated
4. Theoretical extension of GPN axioms to heterogeneous KGs

**Paragraph 5: Results Preview**
- 97% calibration improvement over GGPN
- Competitive link prediction accuracy
- Improved OOD detection

### 5.3 Related Work (1 page)

**Structure:**
1. **Knowledge Graph Embeddings** (TransE, DistMult, ComplEx)
2. **Uncertainty in KGE** (UKGE, BEUrRE, MC Dropout)
3. **Gaussian Processes on Graphs** (Borovitskiy, GGPN)
4. **Graph Posterior Networks** (GPN)
5. **Calibration in ML** (Expected Calibration Error, Brier Score)

**Key Positioning:**
- Distinguish from GGPN: We evaluate uncertainty; they don't
- Distinguish from GPN: We handle heterogeneous KGs; they don't
- Distinguish from UKGE: We model entity-level uncertainty; they model triple-level

### 5.4 Background (0.5 pages)

**5.4.1 Knowledge Graph Embeddings**
- Definition of KG: G = (E, R, T)
- Scoring functions: TransE, DistMult, ComplEx
- Training: negative sampling, margin loss

**5.4.2 Gaussian Processes**
- GP definition: f ~ GP(m, k)
- Kernels on graphs: diffusion, Matérn
- Variational inference: ELBO

### 5.5 Method (2.5 pages)

**5.5.1 Problem Formulation**
- Input: KG with entities, relations, triples
- Output: Entity embeddings with uncertainty
- Goal: Calibrated uncertainty that reflects graph structure

**5.5.2 Relation-Aware GP Prior**
- Per-relation Laplacian: L_r
- Per-relation kernel: K_r = σ_r² exp(-L_r / ℓ_r²)
- Aggregated kernel: K = Σ_r K_r
- Interpretation: ℓ_r = information propagation distance

**5.5.3 Variational Inference**
- Variational posterior: q(F) = N(M, S)
- Low-rank covariance: S = diag(d) + VV^T
- ELBO: likelihood - KL divergence
- Inducing points for scalability

**5.5.4 Entity-Level Uncertainty**
- Posterior variance: S_ii
- Epistemic/aleatoric decomposition
- Connection to GPN axioms

**5.5.5 Training and Prediction**
- Algorithm pseudocode
- Computational complexity: O(NM² + M³)

### 5.6 Experiments (3 pages)

**5.6.1 Experimental Setup**
- Datasets: FB15k-237, WN18RR, CN15k
- Baselines: DistMult, GGPN
- Metrics: MRR, Hits@k, ECE, Brier, AUROC
- Implementation details

**5.6.2 Link Prediction (Table 1)**
- Show competitive accuracy
- Not main contribution, but must be reasonable

**5.6.3 Calibration (Table 2, Figure 1)**
- ECE comparison: GP-KGE >> GGPN
- Reliability diagrams
- **Key result: GGPN is miscalibrated**

**5.6.4 OOD Detection (Table 3)**
- AUROC for detecting random triples
- Compare with baselines

**5.6.5 Ablation Studies (Table 4)**
- w/o relation-aware kernel
- w/o GP prior
- Varying inducing points

**5.6.6 Qualitative Analysis (Figure 2)**
- Uncertainty vs entity degree
- Learned relation importance

### 5.7 Conclusion (0.5 pages)

- Summary of contributions
- Limitations (scalability, computational cost)
- Future work (inductive setting, temporal KGs)

### 5.8 Appendix

**A. Proofs**
- Theorem 1: Kernel PSD
- Proposition 1: Relation importance interpretation

**B. Extended Experimental Results**
- Full tables with std
- Per-relation analysis

**C. Implementation Details**
- Hyperparameters
- Training curves

**D. Compute Resources**
- GPU hours
- Memory usage

---

## 6. TODO List

### 6.1 Experiments (Priority: HIGH)

| Task | Status | Deadline |
|------|--------|----------|
| Complete FB15k-237 full run | 🔄 Running | Today |
| Run WN18RR experiments | ⏳ Pending | +2 days |
| Run CN15k experiments | ⏳ Pending | +3 days |
| Ablation: w/o relation-aware | ⏳ Pending | +4 days |
| Ablation: w/o GP prior | ⏳ Pending | +4 days |
| Ablation: varying inducing points | ⏳ Pending | +4 days |
| Generate reliability diagrams | ⏳ Pending | +5 days |
| Uncertainty vs degree analysis | ⏳ Pending | +5 days |

### 6.2 Writing (Priority: MEDIUM)

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ⏳ Draft needed | After full results |
| Introduction | ⏳ Draft needed | Can start now |
| Related Work | 🟡 Partial | See research gap doc |
| Background | ⏳ Draft needed | See math formalization |
| Method | 🟡 Partial | See math formalization |
| Experiments | ⏳ Draft needed | After experiments |
| Conclusion | ⏳ Draft needed | After everything |

### 6.3 Figures and Tables (Priority: MEDIUM)

| Figure/Table | Status | Description |
|--------------|--------|-------------|
| Table 1: Link Prediction | ⏳ Pending | MRR, Hits@1/3/10 |
| Table 2: Calibration | ⏳ Pending | ECE, Brier Score |
| Table 3: OOD Detection | ⏳ Pending | AUROC |
| Table 4: Ablations | ⏳ Pending | Component analysis |
| Figure 1: Reliability Diagrams | ⏳ Pending | Calibration visualization |
| Figure 2: Uncertainty Analysis | ⏳ Pending | Degree vs uncertainty |
| Figure 3: Model Overview | ⏳ Pending | Architecture diagram |

### 6.4 Technical Improvements (Priority: LOW)

| Task | Status | Impact |
|------|--------|--------|
| Hyperparameter tuning | ⏳ Pending | Better numbers |
| Early stopping on validation | ⏳ Pending | Avoid overfitting |
| Learning rate scheduling | ⏳ Pending | Faster convergence |
| More inducing points | ⏳ Pending | Better approximation |

---

## 7. Key Equations

### For Paper (LaTeX-ready)

**Relation-Aware Kernel:**
```latex
K(i, j) = \sum_{r=1}^{M} \sigma_r^2 \cdot \exp\left(-\frac{L_r}{\ell_r^2}\right)_{ij}
```

**GP Prior:**
```latex
\mathbf{f} \sim \mathcal{GP}(0, K)
```

**Variational Posterior:**
```latex
q(\mathbf{F}) = \mathcal{N}(\mathbf{M}, \mathbf{S})
```

**ELBO:**
```latex
\mathcal{L} = \mathbb{E}_{q}[\log p(\mathbf{y} | \mathbf{F})] - \text{KL}(q(\mathbf{F}) \| p(\mathbf{F}))
```

**Entity Uncertainty:**
```latex
\text{Unc}(e_i) = \mathbf{S}_{ii} = \text{diag}(\mathbf{d})_i + \|\mathbf{V}_i\|^2
```

**ECE (Expected Calibration Error):**
```latex
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} |\text{acc}(B_b) - \text{conf}(B_b)|
```

---

## 8. Experimental Design

### 8.1 Main Experiments

| Experiment | Purpose | Metrics | Baselines |
|------------|---------|---------|-----------|
| Link Prediction | Show accuracy | MRR, Hits@k | DistMult, GGPN |
| Calibration | **Main result** | ECE, Brier | DistMult, GGPN |
| OOD Detection | Uncertainty quality | AUROC | DistMult, GGPN |

### 8.2 Ablation Studies

| Ablation | Question | Expected Result |
|----------|----------|-----------------|
| w/o Relation-aware | Is per-relation kernel necessary? | Performance drop |
| w/o GP prior (point estimate) | Is Bayesian treatment necessary? | Worse calibration |
| Single lengthscale | Is heterogeneity important? | Slight drop |
| Varying M (inducing points) | Scalability-accuracy tradeoff? | Curve showing tradeoff |

### 8.3 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Embedding dim | 200 | Standard in literature |
| Learning rate | 0.001 | Adam default |
| Batch size | 1024 | GPU memory limit |
| Epochs | 50-100 | Early stopping |
| Num inducing | 500 | Balance: accuracy vs speed |
| Negative samples | 10 | Standard practice |
| KL weight (β) | 0.001 | Tuned on validation |

---

## 9. Related Work Positioning

### 9.1 Comparison Table (for paper)

| Method | Type | Entity UQ | Calibration Eval | Relation-Aware |
|--------|------|-----------|------------------|----------------|
| TransE/DistMult | KGE | ❌ | ❌ | ❌ |
| UKGE | Uncertain KGE | Triple-level | ❌ | ❌ |
| BEUrRE | Box embedding | ✓ (geometric) | ❌ | ❌ |
| GPN | Bayesian GNN | ✓ | ✓ | ❌ (homogeneous) |
| GGPN | GP for KG | ❌ | ❌ | ✓ |
| **GP-KGE (Ours)** | GP for KG | ✓ | ✓ | ✓ |

### 9.2 Key Differentiators

**vs GGPN:**
- We evaluate uncertainty quality (ECE, AUROC); they don't
- We show their method is poorly calibrated
- Same GP framework, different focus

**vs GPN:**
- We handle heterogeneous KGs with multiple relation types
- We use continuous Gaussian posterior, not Dirichlet
- We extend their axioms to KG setting

**vs UKGE:**
- We model entity-level uncertainty, not triple-level
- We use GP prior with graph structure
- We provide epistemic/aleatoric decomposition

---

## 10. Potential Reviewer Questions

### Q1: "How is this different from GGPN?"

**Answer:**
GGPN focuses on link prediction accuracy and proposes a GP architecture for KGs, but they never evaluate whether the GP actually provides meaningful uncertainty. We show empirically that GGPN is poorly calibrated (ECE=0.42). Our GP-KGE achieves excellent calibration (ECE=0.01) while maintaining competitive accuracy. The key difference is in the research question: GGPN asks "can GP improve accuracy?", we ask "can GP provide calibrated uncertainty?"

### Q2: "Why not just use ensembles?"

**Answer:**
Ensembles provide weight-space uncertainty, not function-space uncertainty. They don't incorporate the graph structure into the uncertainty model. Our GP prior explicitly encodes that entities connected via certain relations should have similar embeddings, leading to more meaningful uncertainty estimates.

### Q3: "Does the computational overhead justify the uncertainty?"

**Answer:**
Our method adds O(NM² + M³) complexity where M is the number of inducing points. For M=500, this is tractable for medium-scale KGs (up to ~50K entities). The overhead is justified for high-stakes applications where knowing uncertainty is critical (e.g., medical KGs, financial KGs). For large-scale KGs, we can use sparse approximations or reduce M.

### Q4: "The MRR numbers are lower than SOTA. Why use GP-KGE?"

**Answer:**
Our goal is not to achieve SOTA link prediction (many methods do that). Our goal is to provide calibrated uncertainty estimates. We show that GP-KGE achieves *competitive* accuracy while providing *superior* uncertainty. The slight accuracy trade-off is worthwhile for applications requiring trustworthy predictions.

### Q5: "How do you choose the number of inducing points?"

**Answer:**
We ablate this in Section X.X. More inducing points improve approximation quality but increase computation. M=500 provides a good balance for FB15k-237 (14K entities). For larger KGs, structured inducing point selection (e.g., cluster centers) can help.

---

## 11. Timeline

| Week | Tasks |
|------|-------|
| Week 1 | Complete FB15k-237 experiments, start WN18RR |
| Week 2 | Complete all experiments, generate figures |
| Week 3 | Write method and experiments sections |
| Week 4 | Write intro, related work, conclusion |
| Week 5 | Polish, internal review, submit to workshop |
| Week 6+ | Iterate based on feedback |

---

## 12. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Full results don't match sample results | Low | Sample results are promising |
| GGPN authors publish UQ extension | Low | Our framing is different; fast execution |
| Scalability issues on large KG | Medium | Focus on medium-scale; add sparse methods |
| Marginal MRR improvement | Expected | Frame as calibration paper, not accuracy paper |
| Reviewer thinks it's incremental over GGPN | Medium | Emphasize UQ evaluation is novel contribution |

---

## Appendix: File Checklist for Submission

| File | Status | Notes |
|------|--------|-------|
| `paper.tex` | ⏳ Not started | Main paper |
| `paper.bib` | ⏳ Not started | References |
| `figures/reliability_diagram.pdf` | ⏳ Not started | - |
| `figures/model_overview.pdf` | ⏳ Not started | - |
| `figures/uncertainty_analysis.pdf` | ⏳ Not started | - |
| `tables/link_prediction.tex` | ⏳ Not started | - |
| `tables/calibration.tex` | ⏳ Not started | - |
| `supplementary.pdf` | ⏳ Not started | Appendix |
| `code.zip` | ✅ Ready | GitHub repo |
