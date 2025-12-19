# Literature Review: Knowledge Graph as Bayesian Prior for Uncertainty Quantification

## 1. Foundational Work

### 1.1 Diffusion Kernels on Graphs
**Kondor & Lafferty (2002)**

**Key Contribution:**
- Defined the diffusion kernel: `K = exp(-βL)` where L is the graph Laplacian
- Showed this is positive semi-definite (valid kernel)
- Interpretation: K_ij measures random walk transition probability from i to j

**Mathematical Details:**
```
L = D - A  (unnormalized)
L = I - D^{-1/2} A D^{-1/2}  (normalized)

K = exp(-βL) = U exp(-βΛ) U^T

where L = UΛU^T is eigendecomposition
```

**Relevance to Our Work:**
- Base kernel for GP on graphs
- Our extension: per-relation diffusion parameters β_r

---

### 1.2 Matérn Gaussian Processes on Graphs
**Borovitskiy et al. (2021) - NeurIPS**

**Key Contribution:**
- Extended Matérn covariance to graphs via spectral theory
- `K = σ² (2ν/κ² + L)^{-ν}`
- Showed equivalence to Euclidean Matérn in the graph limit

**Mathematical Details:**
```
Spectral density: S(λ) = (2ν/κ² + λ)^{-ν}
Kernel: K = U S(Λ) U^T

Special cases:
- ν → ∞: Squared exponential (very smooth)
- ν = 1/2: Exponential (once differentiable)
- ν = 3/2, 5/2: Common choices
```

**Limitations (Our Gap):**
- Homogeneous graphs only (single edge type)
- Doesn't handle multiple relation types in KGs
- Single lengthscale for entire graph

**Our Extension:**
- Per-relation lengthscales: ℓ_r for each relation r
- Aggregation of relation-specific kernels
- Learnable relation importance weights

---

## 2. Uncertainty in Knowledge Graphs

### 2.1 UKGE: Uncertain Knowledge Graph Embedding
**Chen et al. (2019) - AAAI**

**Key Contribution:**
- First to model confidence scores for KG triples
- Extended DistMult to predict confidence: `score(h,r,t) → conf ∈ [0,1]`
- Used probabilistic soft logic for constraint satisfaction

**Limitations:**
- Triple-level uncertainty only
- Cannot distinguish epistemic vs aleatoric uncertainty
- Point estimate, not full distribution

**Our Difference:**
- Entity-level uncertainty (embedding has variance)
- Full posterior distribution
- Epistemic/aleatoric decomposition

---

### 2.2 BEUrRE: Probabilistic Box Embeddings
**Chen et al. (2021) - NAACL**

**Key Contribution:**
- Represent entities as boxes (hyper-rectangles) not points
- Box volume represents uncertainty
- Containment relations for hierarchy modeling

**Limitations:**
- Geometric constraints on uncertainty shape
- Doesn't use graph structure for uncertainty
- No principled Bayesian treatment

**Our Difference:**
- GP prior uses graph structure
- Principled Bayesian inference
- Flexible uncertainty shape (Gaussian)

---

## 3. Graph Neural Networks + Uncertainty

### 3.1 Graph Posterior Network (GPN)
**Stadler et al. (2021) - NeurIPS** ⭐ (Key paper for Günnemann lab)

**Key Contribution:**
- Three axioms for uncertainty on graphs:
  1. Agreement: Same-class neighbors → decrease uncertainty
  2. Disagreement: Different-class neighbors → increase uncertainty
  3. Isolation: No neighbors → rely on prior

- Used Dirichlet distribution for class probabilities
- Posterior update based on neighbor evidence

**Mathematical Framework:**
```
Prior: p(π) = Dir(α_0)
Evidence from neighbors: β_i = Σ_{j∈N(i)} w_{ij} f_θ(x_j)
Posterior: p(π|x, G) = Dir(α_0 + β_i)
```

**Limitations:**
- Node classification only
- Homogeneous graphs
- Dirichlet (discrete classes), not continuous embeddings

**Our Extension for KG:**
- Extend axioms to heterogeneous KG:
  - Relation-type affects information propagation
  - Different relations → different smoothness
- Continuous embeddings (Gaussian) instead of Dirichlet
- Entity representation uncertainty, not class uncertainty

---

### 3.2 Bayesian Graph Neural Networks

**Various Authors (2019-2023)**

**Approaches:**
1. **MC Dropout on GNN:** Simple, often poorly calibrated
2. **Variational GNN:** Learn distribution over weights
3. **Ensemble GNN:** Multiple independent models

**Limitations:**
- Weight-space uncertainty, not function-space
- Doesn't capture graph structure in prior

---

## 4. Bayesian Knowledge Graph Methods

### 4.1 BIKG: Bayesian Inference with KG Evidence
**AAAI 2024**

**Key Contribution:**
- Convert KG to Markov Random Field
- Bayesian updating with complex FOL evidence
- Symbolic reasoning with uncertainty

**Limitations:**
- Discrete/symbolic, not continuous embeddings
- Not scalable to large KGs
- Hand-crafted rules

---

## 5. Theoretical Foundations

### 5.1 Kolmogorov Complexity and Simplicity Bias

**Connection to GP:**
- RKHS norm of GP functions measures "complexity"
- GP prior assigns higher probability to "simpler" functions
- This is continuous analog of Kolmogorov complexity

**For KGs:**
- Simpler = consistent with graph structure
- Prior prefers functions smooth along graph edges
- Complexity penalized by KL term in ELBO

---

### 5.2 Neural Tangent Kernel Connection

**Key Insight (Jacot et al., 2018; Lee et al., 2019):**
- Infinite-width neural networks = Gaussian Process
- Network architecture defines kernel

**For Our Work:**
- GP-KGE can be seen as infinite-width limit of certain GNNs
- Principled uncertainty without approximation

---

## 6. Research Gap Summary

| Aspect | Existing Work | Gap | Our Solution |
|--------|--------------|-----|--------------|
| Graph Type | Homogeneous | No relation types | Relation-aware kernel |
| Uncertainty Level | Triple-level | Can't identify uncertain entities | Entity-level posterior |
| Distribution | Point estimate or Dirichlet | Limited expressiveness | Full Gaussian posterior |
| Prior Structure | Ignore graph or single kernel | Miss heterogeneous structure | Per-relation parameters |
| Uncertainty Type | Mixed | No decomposition | Epistemic/aleatoric split |

---

## 7. Key Equations for Our Approach

### 7.1 Relation-Aware Kernel

```
K(i, j) = Σ_r σ_r² · exp(-L_r / ℓ_r²)

where:
- L_r = graph Laplacian for relation r subgraph
- σ_r² = variance (importance) of relation r
- ℓ_r = lengthscale for relation r
```

### 7.2 GP Prior

```
f ~ GP(0, K)

For entity embeddings:
e_1, ..., e_N ~ N(0, K ⊗ I_d)

where d is embedding dimension
```

### 7.3 Variational Posterior

```
q(f) = N(μ, Σ)

where:
- μ = posterior mean (point estimate)
- Σ = posterior covariance (uncertainty)
```

### 7.4 ELBO

```
ELBO = E_q[log p(y|f)] - β · KL(q(f) || p(f))

where:
- First term: data fit (link prediction)
- Second term: prior regularization (encourages smoothness)
- β: trade-off parameter
```

### 7.5 Entity Uncertainty

```
Uncertainty(entity i) = Σ_ii = diag(Σ)[i]

Interpretation:
- High Σ_ii: uncertain about entity i's embedding
- Low Σ_ii: confident about entity i's embedding
```

---

## 8. Proposed Axioms (Extending GPN to KG)

### Axiom 1: Relation-Weighted Agreement
If entity i is connected to entity j via relation r, and both have similar embeddings, then i's uncertainty should decrease. The decrease is weighted by relation r's importance.

### Axiom 2: Relation-Specific Disagreement
If neighbors have dissimilar embeddings, uncertainty increases. The increase depends on relation type (some relations expect similarity, others don't).

### Axiom 3: Connectivity-Based Prior
Entities with more connections have lower prior uncertainty. Isolated entities have high uncertainty.

### Axiom 4: Relation Diversity
Entities connected via diverse relation types have more robust embeddings (lower uncertainty) than those connected via single relation type.

---

## 9. References

1. Kondor, R., & Lafferty, J. (2002). Diffusion kernels on graphs and other discrete structures. ICML.

2. Borovitskiy, V., et al. (2021). Matérn Gaussian processes on graphs. NeurIPS.

3. Stadler, M., et al. (2021). Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification. NeurIPS.

4. Chen, X., et al. (2019). Embedding uncertain knowledge graphs. AAAI.

5. Chen, X., et al. (2021). Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning. NAACL.

6. Fortuin, V. (2022). Priors in Bayesian Deep Learning: A Review. IJNS.

7. Williams, C. K., & Rasmussen, C. E. (2006). Gaussian processes for machine learning. MIT Press.

8. Jacot, A., et al. (2018). Neural tangent kernel: Convergence and generalization in neural networks. NeurIPS.
