# Mathematical Formalization: Relation-Aware GP Prior for KG

## 1. Problem Setup

### Knowledge Graph Definition
A Knowledge Graph `G = (E, R, T)` consists of:
- `E = {e_1, ..., e_N}`: Set of N entities
- `R = {r_1, ..., r_M}`: Set of M relation types
- `T ⊆ E × R × E`: Set of observed triples (facts)

### Goal
Learn entity embeddings `f: E → ℝ^d` such that:
1. Embeddings are useful for downstream tasks (link prediction)
2. Embeddings come with uncertainty estimates
3. Uncertainty reflects the graph structure

---

## 2. Gaussian Process Prior

### Standard GP on Graphs
For a function `f: V → ℝ` on graph vertices V:

```
f ~ GP(m, k)
```

where:
- `m: V → ℝ` is the mean function (usually 0)
- `k: V × V → ℝ` is the covariance (kernel) function

Any finite collection of function values is jointly Gaussian:
```
[f(v_1), ..., f(v_n)]^T ~ N(μ, K)
```
where `K_ij = k(v_i, v_j)`

### Graph Laplacian
For adjacency matrix A:

**Unnormalized:**
```
L = D - A
```

**Normalized (symmetric):**
```
L_sym = I - D^{-1/2} A D^{-1/2}
```

**Normalized (random walk):**
```
L_rw = I - D^{-1} A
```

Properties:
- L is positive semi-definite
- Eigenvalues: 0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_N
- Smallest eigenvalue λ_1 = 0 corresponds to constant eigenvector

### Spectral Representation
Graph Laplacian eigendecomposition:
```
L = U Λ U^T
```
where U = [u_1, ..., u_N] are eigenvectors, Λ = diag(λ_1, ..., λ_N).

Kernel via spectral filtering:
```
K = U g(Λ) U^T = Σ_i g(λ_i) u_i u_i^T
```
where g: ℝ → ℝ is the spectral filter function.

---

## 3. Standard Graph Kernels

### Diffusion Kernel
```
K = exp(-β L) = U exp(-β Λ) U^T
```

Spectral filter: `g(λ) = exp(-β λ)`

- β > 0: diffusion parameter
- Small β: only immediate neighbors matter
- Large β: information spreads further

### Matérn Kernel on Graphs
```
K = σ² (2ν/κ² + L)^{-ν}
```

Spectral filter: `g(λ) = σ² (2ν/κ² + λ)^{-ν}`

Parameters:
- σ²: output variance
- κ: inverse lengthscale
- ν: smoothness (ν → ∞ gives squared exponential)

---

## 4. Relation-Aware Kernel (Our Contribution)

### Per-Relation Graph Structure
For each relation r ∈ R, define:
- `A_r`: Adjacency matrix for edges of type r
- `L_r`: Laplacian of relation-r subgraph

### Relation-Aware Kernel

**Definition 1 (Additive Relation-Aware Kernel):**
```
K(i, j) = Σ_{r=1}^{M} K_r(i, j)
```

where each `K_r` is computed from the relation-r subgraph.

**Definition 2 (Weighted Relation-Aware Kernel):**
```
K(i, j) = Σ_{r=1}^{M} w_r · K_r(i, j | θ_r)
```

where:
- `w_r ≥ 0`: importance weight for relation r
- `θ_r`: kernel parameters for relation r

**Definition 3 (Diffusion-Based Relation Kernel):**
```
K_r(i, j) = σ_r² · exp(-L_r / ℓ_r²)_{ij}
```

Parameters per relation:
- `σ_r²`: variance (how much relation r contributes)
- `ℓ_r`: lengthscale (how far information propagates along r)

**Definition 4 (Matérn Relation Kernel):**
```
K_r(i, j) = σ_r² · (2ν/ℓ_r² + L_r)^{-ν}_{ij}
```

### Full Relation-Aware Kernel
```
K = Σ_{r=1}^{M} σ_r² · exp(-L_r / ℓ_r²)
```

**Theorem 1:** The relation-aware kernel K is positive semi-definite.

*Proof:* Sum of PSD matrices is PSD. Each exp(-L_r / ℓ_r²) is PSD since L_r is PSD and matrix exponential of negative PSD is PSD.

---

## 5. GP Prior on Entity Embeddings

### Vector-Valued GP
For d-dimensional embeddings, we use independent GPs per dimension:
```
f = [f^{(1)}, ..., f^{(d)}]

f^{(j)} ~ GP(0, K)  for j = 1, ..., d
```

Joint prior:
```
vec(F) ~ N(0, K ⊗ I_d)
```
where F ∈ ℝ^{N×d} is the embedding matrix.

### Prior Covariance Structure
```
Cov(f^{(j)}(e_i), f^{(k)}(e_l)) = K(e_i, e_l) · δ_{jk}
```

- Same entity, different dims: independent
- Different entities: correlated via graph structure

---

## 6. Observation Model

### Link Prediction Likelihood
For triple (h, r, t):
```
y_{hrt} ~ Bernoulli(σ(score(h, r, t)))
```

where σ is sigmoid and score is a KGE scoring function:

**DistMult:**
```
score(h, r, t) = ⟨f(h), w_r, f(t)⟩ = Σ_i f_i(h) · w_{r,i} · f_i(t)
```

**TransE:**
```
score(h, r, t) = -||f(h) + w_r - f(t)||_p
```

**ComplEx:**
```
score(h, r, t) = Re(⟨f(h), w_r, conj(f(t))⟩)
```

### Full Likelihood
```
p(Y | F, W) = Π_{(h,r,t) ∈ T} σ(score(h,r,t)) · Π_{(h,r,t) ∉ T} (1 - σ(score(h,r,t)))
```

---

## 7. Variational Inference

### Variational Distribution
Approximate posterior:
```
q(F) = N(M, S)
```

where:
- M ∈ ℝ^{N×d}: posterior mean (entity embeddings)
- S: posterior covariance (captures uncertainty)

### Low-Rank + Diagonal Covariance
For scalability:
```
S = diag(d) + VV^T
```
where V ∈ ℝ^{N×k} with k << N.

### Evidence Lower Bound (ELBO)
```
ELBO = E_q[log p(Y | F, W)] - KL(q(F) || p(F | K))
```

**Likelihood term:**
```
E_q[log p(Y | F, W)] ≈ Σ_{(h,r,t) ∈ T} E_q[log σ(score(h,r,t))]
                     + Σ_{neg} E_q[log(1 - σ(score(h,r,t)))]
```

Approximated via sampling (reparameterization trick):
```
F = M + ε ⊙ √diag(S),  ε ~ N(0, I)
```

**KL term:**
```
KL(q(F) || p(F)) = (1/2)[tr(K^{-1}S) + M^T K^{-1} M - N·d + d·log|K| - log|S|]
```

For inducing point approximation, replace with:
```
KL(q(U) || p(U))
```
where U are inducing point embeddings.

---

## 8. Entity-Level Uncertainty

### Posterior Variance
For entity e_i:
```
Var(f(e_i)) = S_{ii} = diag(d)_i + ||V_i||²
```

### Uncertainty Interpretation
- High S_{ii}: uncertain about entity i's embedding
- Low S_{ii}: confident about entity i's embedding

### Uncertainty Decomposition

**Total Uncertainty:**
```
U_total(h, r, t) = Var_q[score(h, r, t)]
```

**Epistemic Uncertainty (from posterior):**
```
U_epistemic = Var_q[score(h, r, t)]
            ≈ Σ_i w_{r,i}² · (S_{hh,i} · μ_{t,i}² + S_{tt,i} · μ_{h,i}² + S_{hh,i} · S_{tt,i})
```
(For DistMult scoring)

**Aleatoric Uncertainty (irreducible):**
From observation noise, typically fixed or learned.

---

## 9. Scalable Inference

### Inducing Points
Select M << N inducing entities U:
```
p(F, U) = p(F | U) p(U)
p(U) = N(0, K_{UU})
p(F | U) = N(K_{FU} K_{UU}^{-1} U, K_{FF} - K_{FU} K_{UU}^{-1} K_{UF})
```

Variational distribution:
```
q(U) = N(m, S)
```

Predictive:
```
q(F) = ∫ p(F | U) q(U) dU
     = N(K_{FU} K_{UU}^{-1} m, K_{FF} - K_{FU} K_{UU}^{-1}(K_{UU} - S) K_{UU}^{-1} K_{UF})
```

### Computational Complexity
- Full GP: O(N³) for Cholesky
- Inducing points: O(NM² + M³)
- Per-relation kernel: O(M·N_r²) where N_r = edges of type r

---

## 10. Learning Kernel Parameters

### Parameters to Learn
- {σ_r², ℓ_r} for each relation r
- Relation importance weights w_r (if using attention)
- Inducing point locations (optional)
- Observation noise variance

### Optimization
Maximize ELBO w.r.t. all parameters jointly:
```
θ* = argmax_θ ELBO(θ)
```

Gradient-based optimization (Adam):
```
θ ← θ + α ∇_θ ELBO
```

### Regularization
Priors on kernel parameters:
```
log(ℓ_r) ~ N(0, τ²)  (prevents extreme lengthscales)
log(σ_r²) ~ N(0, τ²)
```

---

## 11. Theoretical Properties

### Theorem 2 (Consistency)
As data → ∞, the posterior contracts to the true function.

### Theorem 3 (Calibration)
Under model assumptions, the GP provides calibrated uncertainty estimates.

### Proposition 1 (Relation Importance)
The learned σ_r² reflects how informative relation r is for the prediction task.

### Proposition 2 (Lengthscale Interpretation)
- Large ℓ_r: information propagates far along relation r
- Small ℓ_r: only immediate neighbors matter for relation r

---

## 12. Connection to GPN Axioms

Our model satisfies extended versions of GPN axioms:

**Axiom 1 (Agreement):**
If neighbors connected via relation r have similar embeddings, uncertainty decreases proportionally to σ_r².

**Axiom 2 (Disagreement):**
If neighbors have dissimilar embeddings, the prior KL term penalizes this, but doesn't directly increase uncertainty (unlike GPN's Dirichlet).

**Axiom 3 (Isolation):**
Isolated entities have posterior = prior, which has high variance (controlled by σ_r² sum).

**Axiom 4 (New - Relation Diversity):**
Entities connected via multiple relation types aggregate evidence from multiple kernels, leading to lower uncertainty.
