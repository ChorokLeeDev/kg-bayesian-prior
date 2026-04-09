# Coverage Paradox: Theoretical Framework Exploration

**Date**: 2026-04-09  
**Status**: Theoretical exploration for formal explanation

---

## Background: The Empirical Phenomenon

On FB15k-237, we observe a counter-intuitive relationship between coverage and prediction accuracy:

| Coverage Type | Definition | Hits@10 |
|---------------|------------|---------|
| Full Coverage | Both h and t seen with relation r | 32.3% |
| **Partial Zero** | Only one entity seen with relation r | **59.5%** |
| Full Zero | Neither entity seen with relation r | 14.8% |

**Verified contributing factors** (from empirical analysis):
1. **Embedding dilution**: High-degree entities have diluted embeddings (coverage-degree Spearman r=0.636)
2. **Training exposure asymmetry**: Partial entities trained 4x more (821 vs 190 exposures)
3. **Calibration failure**: Full coverage leads to overconfidence
4. **Anchor effect**: Covered entity constrains prediction (contribution ratio 1.36-1.48x)

---

## Direction 1: Information-Theoretic Bound

### Formalization

**Setup**: Let $(h, r, t)$ be a test triple. Define:
- $C_h = c(h, r) \in \{0, 1\}$: head coverage indicator
- $C_t = c(t, r) \in \{0, 1\}$: tail coverage indicator
- $Y \in \{0, 1\}$: correct prediction indicator
- $A = (C_h, C_t)$: coverage configuration

**Key Insight**: Mutual information $I(Y; \hat{Y} | A)$ differs by configuration.

### Proposition 1.1 (Anchor Information Advantage)

Under the "anchor hypothesis", for partial coverage $(1, 0)$ or $(0, 1)$:

$$I(Y; \hat{Y} | C_h=1, C_t=0) > I(Y; \hat{Y} | C_h=1, C_t=1)$$

**Intuition**: The covered entity provides a "clean" conditioning signal. When both are covered, we have:

$$H(Y | C_h=1, C_t=1) = H(Y | \text{conflicting signals})$$

The conflict arises because:
- Both embeddings encode **multiple** relation contexts
- The model must disambiguate which context is relevant
- This increases conditional entropy

### Proposition 1.2 (Conditional Entropy Decomposition)

For a test triple with full coverage:

$$H(Y | \mathbf{e}_h, \mathbf{e}_t, r) = H(Y | \text{relevant context}) + D_{KL}(\text{contexts})$$

where $D_{KL}(\text{contexts})$ measures the divergence between relation contexts encoded in $\mathbf{e}_h$ and $\mathbf{e}_t$.

**Key claim**: High-degree entities have higher context divergence, leading to:
- $H(Y | \text{full coverage, high degree}) > H(Y | \text{partial coverage})$

### Proof Sketch

1. Let $\mathbf{e}_h = \sum_{r' \in R_h} \alpha_{r'} \mathbf{v}_{h,r'}$ (weighted sum of relation-specific components)
2. For relation $r$, only component $\mathbf{v}_{h,r}$ is relevant
3. The "noise" from other components: $\mathbf{n}_h = \sum_{r' \neq r} \alpha_{r'} \mathbf{v}_{h,r'}$
4. Score function: $f(h, r, t) = g(\mathbf{v}_{h,r} + \mathbf{n}_h, \mathbf{r}, \mathbf{v}_{t,r} + \mathbf{n}_t)$
5. With partial coverage (say $C_t = 0$):
   - $\mathbf{e}_t$ has **no** context for relation $r$
   - Model can leverage $\mathbf{v}_{h,r}$ without interference
   - The "anchor" signal is clean

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Formalizability | HIGH | Standard information theory tools |
| Empirical testability | HIGH | Can measure conditional entropies |
| Novelty | MEDIUM | Similar to existing MI decomposition |
| Generalizability | HIGH | Applies to any embedding-based model |

**Verdict**: PROMISING - extends existing info-theoretic framework with anchor-specific analysis.

---

## Direction 2: Embedding Geometry (Center-of-Mass Dilution)

### Formalization

**Setup**: Entity embedding $\mathbf{e} \in \mathbb{R}^d$ trained on relations $R_e = \{r_1, ..., r_k\}$.

**Definition 2.1 (Embedding Dilution)**

An entity embedding exhibits *dilution* if:

$$\mathbf{e} \approx \frac{1}{|R_e|} \sum_{r \in R_e} \mathbf{e}_r^*$$

where $\mathbf{e}_r^*$ is the optimal embedding for relation $r$ alone.

### Theorem 2.1 (Dilution-Specificity Trade-off)

Let $\mathbf{e}$ be trained on $k$ relations via gradient descent on:

$$\mathcal{L} = \sum_{r \in R_e} \sum_{(h,r,t) \in \mathcal{T}_r} \ell(f(h, r, t))$$

Then:
$$\|\mathbf{e} - \mathbf{e}_r^*\|^2 \propto \frac{k-1}{k} \cdot \text{Var}(\{\mathbf{e}_{r'}^*\}_{r' \neq r})$$

**Interpretation**: As $k$ increases, the embedding moves toward the center of mass, away from any specific optimal.

### Corollary 2.1 (Angular Discrimination Loss)

For scoring functions $f(h, r, t) = \mathbf{e}_h^\top \mathbf{R}_r \mathbf{e}_t$ (bilinear):

$$\text{Var}(\text{score} | C_h=1, C_t=1) > \text{Var}(\text{score} | C_h=1, C_t=0)$$

when entity degrees are heterogeneous.

### Proposition 2.2 (Anchor as Discriminative Direction)

In partial coverage $(1, 0)$:
- Covered entity $h$ provides direction $\mathbf{d} = \mathbf{R}_r^\top \mathbf{e}_h$
- Prediction reduces to: $\arg\max_t \langle \mathbf{d}, \mathbf{e}_t \rangle$
- $\mathbf{d}$ is **fixed** by the anchor, reducing search space

In full coverage $(1, 1)$:
- Both $\mathbf{e}_h$ and $\mathbf{e}_t$ are diluted
- The product $\mathbf{e}_h^\top \mathbf{R}_r \mathbf{e}_t$ compounds the noise
- Result: higher variance, lower discriminative power

### Geometric Visualization

```
Full Coverage:                    Partial Coverage:
     e_h (diluted)                    e_h (diluted)
       *---->  (r direction)             *---->
       |    confusion                    |     clear direction
       v                                 v
     e_t (diluted)                    e_t = ? (unknown)
                                      [search in d-direction]
```

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Formalizability | HIGH | Linear algebra, clear bounds |
| Empirical testability | HIGH | Can measure embedding norms, angles |
| Novelty | HIGH | Specific to KG embedding geometry |
| Generalizability | MEDIUM | Specific to bilinear scoring |

**Verdict**: HIGHLY PROMISING - provides concrete, testable geometric predictions.

---

## Direction 3: Generalization Theory (PAC-Bayes Style)

### Formalization

**Setup**: Consider learning entity embeddings under a PAC-Bayes framework.

**Definition 3.1 (Context Effective Sample Size)**

For entity $e$ with coverage configuration $C$:

$$n_{\text{eff}}(e, r) = \begin{cases}
n_{e,r} & \text{if } c(e, r) = 1 \\
0 & \text{if } c(e, r) = 0
\end{cases}$$

where $n_{e,r}$ is the number of training triples involving $(e, r)$.

### Theorem 3.1 (Generalization Bound by Coverage)

For a triple $(h, r, t)$ with coverage configuration $A = (C_h, C_t)$:

$$\mathbb{E}[\mathcal{L}_{\text{test}}(h, r, t) | A] \leq \mathcal{L}_{\text{train}} + \sqrt{\frac{D_{KL}(Q \| P)}{n_{\text{eff}}(A)}}$$

where:
- $n_{\text{eff}}(1, 1) = \min(n_{h,r}, n_{t,r})$ (both constrained by weaker)
- $n_{\text{eff}}(1, 0) = n_{h,r}$ (only anchor matters)
- $n_{\text{eff}}(0, 0) = 0$ (pure extrapolation)

### Key Insight: "Seen in One Context" vs "Seen in Many"

**Proposition 3.1 (Multi-Context Degradation)**

Let entity $e$ be trained on $k$ relations with $n_r$ samples each. Then:

$$\text{Var}(\mathbf{e}) \propto \frac{1}{n_{\text{total}}} + \frac{k-1}{k} \cdot \sigma^2_{\text{inter}}$$

where $\sigma^2_{\text{inter}}$ is the inter-relation variance of optimal embeddings.

**Interpretation**: More contexts (higher $k$) increases variance when contexts are inconsistent.

### Theorem 3.2 (Partial Coverage Advantage)

Under the assumption that entity embeddings are shared across relations:

$$\text{Gen-gap}(1, 0) < \text{Gen-gap}(1, 1)$$

when:
1. $\deg(h) \cdot \deg(t) > \text{threshold}$
2. Relation $r$ has low frequency relative to other relations for $h, t$

**Proof Sketch**:
- In full coverage, both embeddings carry "multi-context" noise
- In partial coverage, the anchor embedding's noise is offset by having **no expectation** for the uncovered entity
- The model correctly attributes uncertainty to the uncovered entity

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Formalizability | MEDIUM | PAC-Bayes requires careful assumptions |
| Empirical testability | MEDIUM | Bounds often loose |
| Novelty | HIGH | Novel application to KG setting |
| Generalizability | HIGH | Framework-agnostic |

**Verdict**: INTERESTING BUT CHALLENGING - provides theoretical backing but bounds may be loose.

---

## Recommendation: Combined Approach

### Primary Direction: Embedding Geometry (Direction 2)

**Rationale**:
1. Most directly testable against empirical observations
2. Clear connection to verified findings (dilution, anchor effect)
3. Provides concrete predictions about embedding norms and angles
4. Novel contribution to KG embedding theory

### Secondary Direction: Information-Theoretic (Direction 1)

**Rationale**:
1. Complements geometric view with probabilistic interpretation
2. Connects to existing information decomposition framework
3. Useful for explaining calibration failure

### Suggested Theorem Statement

**Theorem (Coverage Paradox Explanation)**

Let $\mathcal{G}$ be a knowledge graph with entity embeddings trained via bilinear scoring. For test triples with coverage configurations:

**(i) Dilution Effect**: For entities with degree $k \geq k_0$:
$$\|\mathbf{e} - \mathbf{e}_r^*\| \geq \epsilon \cdot \sqrt{k}$$

**(ii) Anchor Advantage**: For partial coverage $(1, 0)$:
$$\mathbb{E}[\text{Rank}(t) | C_h=1, C_t=0] < \mathbb{E}[\text{Rank}(t) | C_h=1, C_t=1]$$
when $\deg(h), \deg(t) > k_0$.

**(iii) Calibration**: The confidence-accuracy gap:
$$|\Pr(\text{correct} | \text{conf} > \theta) - \theta|$$
is larger for full coverage than partial coverage.

---

## Insights for Other Options

### For Method Development

The geometric analysis suggests:
1. **Relation-specific embedding layers**: Learn $\mathbf{e}_r$ per relation, not shared $\mathbf{e}$
2. **Attention-based context selection**: Weight relation contexts by relevance
3. **Anchor-aware scoring**: Different scoring for partial vs full coverage

### For Broader Impact Section

The theoretical framework reveals:
1. **Structural limitation**: Embedding sharing across relations is fundamentally limiting
2. **Practical recommendation**: Report metrics stratified by coverage configuration
3. **Benchmark design**: Coverage distribution affects reported numbers

---

## Next Steps

1. **Formalize Theorem 2.1** with rigorous assumptions and proof
2. **Empirical validation**: Measure embedding norms/angles by degree
3. **Connect to calibration**: Show ECE difference by coverage type
4. **Write theorem for paper**: Clean statement with proof sketch

---

## References (Internal)

- `docs/theory/info_theoretic_decomposition.md` - Existing MI framework
- `docs/theory/necessity_theorems.md` - Semantic/structural ceiling theorems
- `docs/theory/coverage_sufficiency_theorem.md` - AUROC prediction
- `docs/COVERAGE_PARADOX_FINDINGS.md` - Empirical validation
- `scripts/analyze_anchor_hypothesis.py` - Anchor effect experiments
- `scripts/analyze_information_leakage.py` - Dilution analysis
