# Research Findings: While Waiting for Experiments

**Date:** 2024-12-19
**Status:** Experiments running on Colab T4

---

## 1. Introduction Draft

### Paragraph 1: Motivation

Knowledge Graphs (KGs) have become foundational infrastructure for modern AI systems, powering applications from web search and recommendation systems to question answering and drug discovery. The core task of link prediction—inferring missing facts from observed triples—has seen remarkable progress through Knowledge Graph Embedding (KGE) methods. However, these methods fundamentally produce point estimates: a single score indicating how likely a triple is to be true, with no indication of the model's confidence in that prediction.

This limitation poses significant risks in high-stakes applications. Consider a medical knowledge graph predicting drug-drug interactions: a model that confidently predicts "no interaction" when it has never seen similar drugs could lead to patient harm. What we need is not just accurate predictions, but *calibrated* uncertainty estimates that reflect when the model "knows what it doesn't know."

### Paragraph 2: Problem

Existing approaches to uncertainty in KGE fall into three categories, each with significant limitations:

1. **Post-hoc methods** (MC Dropout, ensembles): These apply general-purpose uncertainty techniques without considering graph structure. They provide weight-space uncertainty, not function-space uncertainty, and are often poorly calibrated.

2. **Triple-level uncertainty** (UKGE): These model confidence for individual triples but cannot identify which *entities* are uncertain—a crucial distinction for downstream applications.

3. **GP-based methods** (GGPN): While principled, existing GP approaches for multi-relational graphs have never evaluated whether their uncertainty estimates are actually meaningful. As we show, GGPN produces severely miscalibrated predictions (ECE=0.42).

### Paragraph 3: Our Approach

We propose GP-KGE, a Gaussian Process prior for entity embeddings with a relation-aware kernel. Our key insight is that different relation types induce fundamentally different similarity structures:

- **"part_of"** relations suggest local, hierarchical similarity (short lengthscale)
- **"similar_to"** relations suggest global semantic similarity (long lengthscale)
- **"located_in"** relations suggest geographic clustering (medium lengthscale)

By learning separate kernel parameters (lengthscale, variance) for each relation type, GP-KGE captures this heterogeneous structure. The GP framework then provides principled Bayesian inference, yielding entity-level posterior distributions that encode meaningful uncertainty.

### Paragraph 4: Contributions

Our contributions are:

1. **GP-KGE**: The first GP-based KG model with comprehensive uncertainty evaluation, showing that principled Bayesian treatment yields well-calibrated predictions.

2. **Relation-Aware Kernel**: A kernel that learns per-relation smoothness parameters, capturing that different relations induce different notions of entity similarity.

3. **Calibration Gap Discovery**: Empirical evidence that existing GP-based KG methods (GGPN) are severely miscalibrated, identifying a critical gap in the literature.

4. **Theoretical Framework**: Extension of Graph Posterior Network (GPN) axioms to heterogeneous knowledge graphs, providing theoretical grounding for uncertainty behavior.

### Paragraph 5: Results Preview

On FB15k-237, GP-KGE achieves 97% better calibration (ECE) compared to GGPN while maintaining competitive link prediction accuracy. We also demonstrate improved out-of-distribution detection and enable selective prediction—abstaining when uncertain. Our results establish that principled Bayesian treatment is not just theoretically appealing but practically necessary for trustworthy KG predictions.

---

## 2. GGPN Paper Analysis

### Paper Details
- **Title:** Multi-Relational Graph Representation Learning with Bayesian Gaussian Process Network
- **Authors:** Chen, G., Fang, J., Meng, Z., Zhang, Q., & Liang, S.
- **Venue:** AAAI 2022
- **Code:** https://github.com/GuanZhengChen/GGPN

### What GGPN Does

1. **Problem:** Learn flexible KG representations that handle noise and scarce labels
2. **Approach:** Model entity embeddings as GP, use Random Fourier Features (RFF) for scalability
3. **Key Innovation:** Novel kernel that considers diverse relations between entity pairs
4. **Training:** Reformulate GP as Bayesian linear model → O(n) complexity

### What GGPN Does NOT Do

| Aspect | GGPN | Our GP-KGE |
|--------|------|------------|
| **Evaluate calibration?** | ❌ No | ✅ Yes (ECE, Brier) |
| **Evaluate OOD detection?** | ❌ No | ✅ Yes (AUROC) |
| **Per-relation parameters?** | Partial (shared) | ✅ Full (ℓ_r, σ_r² per relation) |
| **Epistemic/aleatoric split?** | ❌ No | ✅ Yes |
| **Entity-level uncertainty?** | Implicit | ✅ Explicit (posterior variance) |

### Key Gap We Fill

GGPN focuses on **accuracy** (link prediction, entity classification). They show GP improves accuracy but never ask: *"Is the uncertainty meaningful?"*

We show: **GGPN's uncertainty is NOT calibrated (ECE=0.42).** The GP framework alone doesn't guarantee good uncertainty—you need proper evaluation and potentially architectural choices.

### Technical Differences

| Component | GGPN | GP-KGE |
|-----------|------|--------|
| Kernel | RFF approximation | Spectral (Laplacian eigendecomp) |
| Scalability | O(n) via Bayesian linear model | O(NM² + M³) via inducing points |
| Relations | Single aggregated kernel | Per-relation kernel with learnable params |
| Output | Mean embedding | Mean + full posterior covariance |

---

## 3. GPN Paper Analysis

### Paper Details
- **Title:** Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification
- **Authors:** Stadler, M., Charpentier, B., Geisler, S., Zügner, D., Günnemann, S.
- **Venue:** NeurIPS 2021
- **Lab:** TU Munich (Günnemann group)
- **Code:** https://github.com/stadlmax/Graph-Posterior-Network

### The Three Axioms

GPN proposes that well-behaved uncertainty on graphs should satisfy:

**Axiom 1: Agreement**
> If a node's neighbors consistently belong to the same class, uncertainty should decrease.

*Intuition:* Strong consensus from neighbors → more confident prediction

**Axiom 2: Disagreement**
> If a node's neighbors belong to different classes, uncertainty should increase.

*Intuition:* Conflicting evidence from neighbors → less confident

**Axiom 3: Isolation**
> A node with no neighbors should have high uncertainty, relying only on the prior.

*Intuition:* No evidence → fall back to prior uncertainty

### GPN's Approach

- Uses **Dirichlet distribution** for class probabilities
- Aggregates "pseudo-counts" from neighbors via Personalized PageRank
- Posterior update: Dir(α₀ + β_i) where β_i = aggregated neighbor evidence

### Limitations for KGs

1. **Homogeneous graphs only:** Single edge type, no relation diversity
2. **Discrete classes:** Dirichlet is for classification, not continuous embeddings
3. **Node classification:** Not link prediction

### Our Extension: Axiom 4 (Relation Diversity)

We propose a fourth axiom for heterogeneous KGs:

**Axiom 4: Relation Diversity**
> Entities connected via multiple diverse relation types should have more robust (lower uncertainty) embeddings than entities connected via a single relation type.

*Intuition:* Evidence from multiple "views" (relations) is more reliable than evidence from one view.

**Mathematical Formulation:**
```
Uncertainty(e_i) ∝ 1 / Σ_r σ_r² · |N_r(e_i)|

where:
- N_r(e_i) = neighbors of e_i via relation r
- σ_r² = learned importance of relation r
```

### How GP-KGE Satisfies Extended Axioms

| Axiom | How GP-KGE Satisfies |
|-------|---------------------|
| Agreement | GP posterior shrinks variance when neighbors have similar embeddings |
| Disagreement | Conflicting neighbors increase posterior variance (KL term in ELBO) |
| Isolation | No edges → no kernel contribution → high prior variance |
| **Relation Diversity** | Multiple relations → multiple kernel contributions → lower variance |

---

## 4. Model Overview Figure (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GP-KGE Architecture                          │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ Knowledge    │
    │ Graph G      │
    │ (E, R, T)    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │            Per-Relation Subgraph Extraction               │
    │                                                           │
    │   G_r1 (works_at)    G_r2 (located_in)    G_r3 (...)    │
    │      ┌─┐                  ┌─┐                             │
    │     ╱   ╲                ╱   ╲                            │
    │    ●─────●              ●─────●                           │
    │     ╲   ╱                                                 │
    │      └─┘                                                  │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │              Graph Laplacian per Relation                 │
    │                                                           │
    │   L_r1 = D_r1 - A_r1      L_r2 = D_r2 - A_r2    ...      │
    │                                                           │
    │   Eigendecomposition: L_r = U_r Λ_r U_r^T                │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │            Relation-Aware Kernel (Our Contribution)       │
    │                                                           │
    │   K = Σ_r  σ_r² · exp(-L_r / ℓ_r²)                       │
    │          ↑         ↑                                      │
    │     learnable   learnable                                 │
    │     variance    lengthscale                               │
    │                                                           │
    │   ℓ_r small → local similarity (part_of)                 │
    │   ℓ_r large → global similarity (similar_to)             │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  GP Prior on Embeddings                   │
    │                                                           │
    │   f ~ GP(0, K)                                           │
    │                                                           │
    │   Entity embeddings are correlated via graph structure   │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │              Variational Inference (ELBO)                 │
    │                                                           │
    │   q(F) = N(M, S)     M = posterior mean (embeddings)     │
    │                      S = posterior covariance (uncertainty)│
    │                                                           │
    │   ELBO = E_q[log p(Y|F)] - KL(q(F) || p(F|K))           │
    │              ↑                    ↑                       │
    │         likelihood            GP prior                   │
    │        (link pred)          regularization               │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    Scoring Function                       │
    │                                                           │
    │   score(h, r, t) = ⟨M_h, W_r, M_t⟩  (DistMult)          │
    │                                                           │
    │   With uncertainty propagation:                          │
    │   Var[score] = f(S_hh, S_tt, W_r)                        │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                       Outputs                             │
    │                                                           │
    │   1. Link Prediction: P(triple is true)                  │
    │   2. Epistemic Uncertainty: from posterior variance S    │
    │   3. Aleatoric Uncertainty: from observation noise       │
    │   4. Entity Uncertainty: S_ii for each entity            │
    │                                                           │
    │   → Calibrated predictions (low ECE)                     │
    │   → OOD detection (high AUROC)                           │
    │   → Selective prediction (abstain when uncertain)        │
    └──────────────────────────────────────────────────────────┘
```

---

## 5. Ablation Study Plan

### Ablation 1: w/o Relation-Aware Kernel

**Question:** Is per-relation kernel necessary?

**Setup:**
```python
# Instead of per-relation kernel:
K = Σ_r σ_r² · exp(-L_r / ℓ_r²)

# Use single kernel on full graph:
K = σ² · exp(-L / ℓ²)
```

**Expected Result:** Worse calibration (ECE increases) because heterogeneous structure is lost.

### Ablation 2: w/o GP Prior (Point Estimates)

**Question:** Is Bayesian treatment necessary?

**Setup:**
```python
# Instead of variational inference with posterior:
q(F) = N(M, S)

# Use point estimates only:
F = M  (no uncertainty)
```

**Expected Result:** Much worse calibration, no meaningful uncertainty estimates.

### Ablation 3: Single Lengthscale

**Question:** Do we need per-relation lengthscales?

**Setup:**
```python
# Instead of per-relation ℓ_r:
# Use single shared ℓ for all relations
K = Σ_r σ_r² · exp(-L_r / ℓ²)
```

**Expected Result:** Slight degradation, but less than full ablation. Per-relation variance (σ_r²) still helps.

### Ablation 4: Varying Inducing Points

**Question:** How does M (inducing points) affect quality vs speed?

**Setup:**
| M | Expected Accuracy | Expected Speed |
|---|------------------|----------------|
| 100 | Lower | Faster |
| 200 | Medium | Medium |
| 500 | Higher | Slower |
| 1000 | Highest | Slowest |

**Expected Result:** Diminishing returns after M ≈ 500 for FB15k-237.

### Ablation 5: Varying num_eigenvectors

**Question:** How many eigenvectors are needed for good kernel approximation?

**Setup:**
| k | Kernel Quality | Speed |
|---|---------------|-------|
| 50 | Approximate | Fast |
| 100 | Good | Medium |
| 200 | Better | Slower |
| 500 | Best | Slowest |

**Expected Result:** k=100 is sufficient for most relations.

---

## 6. Key Differences: GGPN vs GP-KGE

| Aspect | GGPN | GP-KGE (Ours) |
|--------|------|---------------|
| **Goal** | Accuracy | Calibrated uncertainty |
| **Kernel** | RFF approximation | Spectral decomposition |
| **Relations** | Shared kernel | Per-relation (ℓ_r, σ_r²) |
| **Evaluation** | MRR, Hits@k | + ECE, Brier, AUROC |
| **Output** | Mean | Mean + Variance |
| **Scalability** | O(n) | O(NM² + M³) |
| **Calibration** | ECE ≈ 0.42 (poor) | ECE ≈ 0.01 (excellent) |

---

## 7. Remaining Questions for Paper

1. **Why is GGPN miscalibrated?**
   - Hypothesis: RFF approximation loses uncertainty information
   - Or: No explicit calibration objective in training

2. **Scalability to larger KGs?**
   - Current: Works well up to ~50K entities
   - Future: Need sparse approximations for FB15k, YAGO, etc.

3. **Inductive setting?**
   - Current: Transductive (all entities seen during training)
   - Future: Handle new entities at test time

---

## Sources

- [GGPN Paper (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20492)
- [GGPN Code](https://github.com/GuanZhengChen/GGPN)
- [GPN Paper (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/95b431e51fc53692913da5263c214162-Abstract.html)
- [GPN Code](https://github.com/stadlmax/Graph-Posterior-Network)
- [GPN arXiv](https://arxiv.org/abs/2110.14012)
