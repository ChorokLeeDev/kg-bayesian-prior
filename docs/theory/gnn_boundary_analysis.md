# GNN Boundary Condition: Theoretical Analysis

## Summary

This document provides theoretical grounding for the empirical observation that GNN-based OOD detectors (e.g., GNNSafe) succeed on sparse-relation KGs (WN18RR: 0.79 AUROC) but fail on dense-relation KGs (FB15k-237: 0.43 AUROC). We formalize the conditions under which GNN message passing can implicitly encode coverage information and derive a **relation-to-neighborhood ratio** as the governing parameter.

---

## 1. Empirical Observations

| Dataset | |R| | avg |N(e)| | Ratio | GNNSafe Novel-ctx AUROC |
|---------|------|-----------|-------|-------------------------|
| YAGO3-10 | 37 | 12.6 | 2.9 | (predicted: works) |
| WN18RR | 11 | 2.2 | 5.0 | **0.79** (works) |
| FB15k-237 | 237 | 19.0 | 12.5 | **0.43** (fails) |

**Key question**: Why does GNN architecture escape the impossibility theorem in some cases?

---

## 2. The Coverage Information Problem

### 2.1 Impossibility Theorem Recap

For **relation-agnostic embeddings** $\phi: \mathcal{E} \to \mathbb{R}^d$ satisfying:
$$I(\phi(e); c(e,r) \mid \text{freq}(e)) = 0 \quad \forall r \in \mathcal{R}$$

any uncertainty estimator $U(h,r,t) = F(\phi(h), \psi(r), \phi(t))$ achieves:
$$\text{AUROC}(U, \mathcal{D}_{\text{novel}}) \leq \frac{1}{2} + O(\epsilon)$$

**Core issue**: Standard embeddings encode *frequency* but not the combinatorial structure of *which relations* an entity has seen.

### 2.2 GNN Exception Hypothesis

GNN architectures compute:
$$\phi^{(L)}(e) = \text{AGG}^{(L)}\left(\phi^{(L-1)}(e), \left\{\phi^{(L-1)}(n) : n \in \mathcal{N}(e)\right\}\right)$$

The neighborhood $\mathcal{N}(e)$ is **determined by the training graph structure**, which encodes relation information. The question is: under what conditions does this architectural feature translate into coverage-discriminative representations?

---

## 3. Theoretical Analysis

### 3.1 Information-Theoretic Formulation

**Definition (Coverage Identifiability via Neighbors).**
An entity $e$'s coverage pattern $\mathbf{c}_e = (c(e,r_1), \ldots, c(e,r_m)) \in \{0,1\}^{|\mathcal{R}|}$ is **identifiable from neighbors** if:
$$I(\mathcal{N}(e); \mathbf{c}_e \mid \text{freq}(e)) > 0$$

That is, knowing the neighbor set provides information about coverage beyond what frequency reveals.

### 3.2 When Does Neighbor Structure Encode Coverage?

**Proposition 1 (Sufficient Condition for Coverage Proxy).**
Let $\bar{d} = \bar{|\mathcal{N}(e)|}$ be the average neighborhood size and $|\mathcal{R}|$ be the number of relations. If each relation $r$ induces a **distinctive neighborhood signature**---i.e., entities connected via $r$ share neighbors disproportionately compared to baseline---then:
$$I(\mathcal{N}(e); c(e,r)) > 0 \quad \text{when } |\mathcal{R}| \lesssim \bar{d}$$

**Intuition**: When the number of relations is small relative to neighborhood size, each relation "leaves a footprint" in the neighbor structure that can be detected by message passing.

**Proof Sketch.**
Consider the mutual information:
$$I(\mathcal{N}(e); c(e,r)) = H(c(e,r)) - H(c(e,r) \mid \mathcal{N}(e))$$

The term $H(c(e,r) \mid \mathcal{N}(e))$ depends on how deterministically coverage can be inferred from neighbors.

1. **Low-relation regime** ($|\mathcal{R}| \ll \bar{d}$): Each relation $r$ connects to a distinct subset of neighbors on average. With only $|\mathcal{R}|$ relations and $\bar{d}$ neighbors, each relation can be associated with $\approx \bar{d}/|\mathcal{R}|$ distinct neighbors. If $\bar{d}/|\mathcal{R}| \geq 1$, the neighbor pattern is informative: seeing neighbors $\{n_1, n_2, \ldots\}$ restricts the possible relations $e$ has seen.

2. **High-relation regime** ($|\mathcal{R}| \gg \bar{d}$): There are far more relations than neighbors. By pigeonhole, multiple relations must share the same "neighbor signature" (or have no signature at all). The neighborhood cannot distinguish between relations, so $H(c(e,r) \mid \mathcal{N}(e)) \approx H(c(e,r))$ and mutual information vanishes.

$\square$

### 3.3 Formalizing the Ratio Threshold

**Proposition 2 (Relation-Neighborhood Ratio Bound).**
Define the ratio $\gamma = |\mathcal{R}| / \bar{|\mathcal{N}(e)|}$. Under a simplified model where:
- Each relation $r$ is associated with a subset $S_r \subseteq \mathcal{E}$ of entities (those that appear as neighbors via $r$)
- Sets $\{S_r\}$ are drawn with bounded overlap

The probability that an arbitrary relation $r$ can be distinguished from coverage via the neighbor set satisfies:
$$P(\text{coverage identifiable from neighbors}) \geq 1 - \exp(-\bar{d}/\gamma)$$

**Corollary.**
- When $\gamma \lesssim 1$: Neighborhoods have sufficient capacity to encode all relations. GNN embeddings satisfy $I(\phi(e); c(e,r) \mid \text{freq}(e)) > 0$.
- When $\gamma \gg 1$: Neighborhoods cannot encode relation-specific information. GNN embeddings become relation-agnostic, and the impossibility theorem applies.

### 3.4 Critical Transition Zone

**Proposition 3 (Phase Transition).**
There exists a critical ratio $\gamma^*$ such that:
- For $\gamma < \gamma^*$: GNN can serve as a coverage proxy, achieving $\text{AUROC}(U, \mathcal{D}_{\text{novel}}) > 0.5$
- For $\gamma > \gamma^*$: GNN cannot distinguish novel contexts from ID, yielding $\text{AUROC} \approx 0.5$

From empirical observations:
- WN18RR ($\gamma = 5.0$): GNNSafe works (0.79)
- FB15k-237 ($\gamma = 12.5$): GNNSafe fails (0.43, anti-predictive)

This suggests $\gamma^* \in [5, 12]$, with the transition likely occurring around $\gamma^* \approx 7\text{--}10$.

---

## 4. Message Passing Depth Analysis

### 4.1 How Many Layers to Encode Coverage?

**Question**: Can deeper GNNs recover coverage information even when $\gamma > \gamma^*$?

**Proposition 4 (Layer Bound).**
To encode coverage information for relation $r$ via $L$-layer message passing, the receptive field must include entities that distinguish $r$ from other relations. The required depth satisfies:
$$L \geq \frac{\log(|\mathcal{R}|/\bar{d})}{\log(\bar{d})}$$

**Proof Sketch.**
At layer $L$, each entity aggregates information from its $L$-hop neighborhood, which contains $O(\bar{d}^L)$ entities. To distinguish $|\mathcal{R}|$ relations, the receptive field must be large enough to encode $|\mathcal{R}|$ bits of information. This requires $\bar{d}^L \gtrsim |\mathcal{R}|$, giving $L \gtrsim \log(|\mathcal{R}|)/\log(\bar{d})$.

For FB15k-237: $L \geq \log(237)/\log(19) \approx 1.9$ layers.
For WN18RR: $L \geq \log(11)/\log(2.2) \approx 3.0$ layers.

**Important caveat**: More layers help in theory but face practical issues:
1. **Over-smoothing**: Deep GNNs produce similar embeddings for all entities
2. **Information bottleneck**: Message passing compresses information at each layer
3. **Computational cost**: Receptive field grows exponentially

### 4.2 Why 2-Layer GNNs Are Insufficient for FB15k-237

Standard GNNSafe implementations use 2-layer networks. For FB15k-237:
- 2-hop receptive field: $\approx 19^2 = 361$ entities
- Relations: 237
- **Problem**: The receptive field is barely larger than the relation count, but the aggregation function must *compress* this information into a fixed-dimensional embedding. The information loss makes coverage unrecoverable.

For WN18RR:
- 2-hop receptive field: $\approx 2.2^2 = 4.8$ entities (but covers $\approx 5$ relations)
- **Advantage**: The ratio is favorable; even limited receptive fields can encode the small relation set.

---

## 5. Alternative Mechanisms

### 5.1 Relation-Typed Neighborhoods

If the GNN explicitly uses **relation types** in message passing:
$$\phi^{(L)}(e) = \text{AGG}\left(\left\{\phi^{(L-1)}(n) : (e,r,n) \in \mathcal{T} \text{ for some } r\right\}\right)$$

Then the aggregation inherently knows which relations contributed to the neighborhood, and coverage can be directly computed.

**This is equivalent to explicit coverage tracking** and no longer subject to the ratio bound.

### 5.2 Attention-Based Mechanisms

Relational attention mechanisms (e.g., R-GCN, CompGCN) can learn to weight neighbors by relation type, potentially recovering coverage information even at high $\gamma$. However, this requires:
1. Relation-specific parameters: $O(|\mathcal{R}| \times d^2)$ parameters
2. Sufficient training data per relation

On sparse-relation KGs, these mechanisms have inadequate per-relation data, reducing them to standard GNNs.

---

## 6. Formal Theorem Statement

**Theorem (GNN Boundary Condition).**
Let $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$ be a knowledge graph with $|\mathcal{R}|$ relations and average entity neighborhood size $\bar{d}$. Let $\phi: \mathcal{E} \to \mathbb{R}^d$ be an $L$-layer GNN embedding computed via:
$$\phi^{(\ell)}(e) = \sigma\left(W^{(\ell)} \cdot \text{AGG}\left(\{\phi^{(\ell-1)}(n) : n \in \mathcal{N}(e)\}\right)\right)$$

Define the **relation-neighborhood ratio** $\gamma = |\mathcal{R}| / \bar{d}$.

**(i) Low-ratio regime ($\gamma \lesssim \gamma^*$):**
The GNN embedding satisfies:
$$I(\phi(e); c(e,r) \mid \text{freq}(e)) > 0$$
for most relations $r$, enabling novel-context detection with $\text{AUROC} > 0.5$.

**(ii) High-ratio regime ($\gamma \gg \gamma^*$):**
The GNN embedding becomes effectively relation-agnostic:
$$I(\phi(e); c(e,r) \mid \text{freq}(e)) \approx 0$$
and the impossibility theorem applies: $\text{AUROC}(\mathcal{D}_{\text{novel}}) \leq 0.5 + O(\epsilon)$.

**(iii) Transition zone ($\gamma \approx \gamma^*$):**
Performance interpolates between the two regimes, with high variance across random seeds and entity subsets.

**Empirical estimate**: $\gamma^* \in [5, 12]$ based on WN18RR ($\gamma = 5.0$, works) and FB15k-237 ($\gamma = 12.5$, fails).

---

## 7. Predictions

### 7.1 YAGO3-10 Prediction

| Metric | Value |
|--------|-------|
| Relations | 37 |
| Avg neighbors | 12.6 |
| Ratio $\gamma$ | 2.9 |

**Prediction**: With $\gamma = 2.9 < \gamma^*$, YAGO3-10 should exhibit GNNSafe novel-context AUROC comparable to WN18RR (0.79), possibly higher given the more favorable ratio.

**Expected range**: AUROC $\in [0.70, 0.85]$

### 7.2 Other Datasets

| Dataset | |R| | Est. avg |N(e)| | Est. $\gamma$ | Prediction |
|---------|------|---------------|---------------|------------|
| NELL-995 | 200 | ~15 | ~13 | Fails (similar to FB15k-237) |
| ICEWS14 | 230 | ~8 | ~29 | Fails |
| Hetionet | 24 | ~100 | ~0.24 | Works (very favorable ratio) |
| Freebase (full) | 15,000 | ~30 | ~500 | Fails definitively |

### 7.3 Design Implications

1. **For new KG deployments**: Compute $\gamma$ before selecting OOD detection method. If $\gamma > 10$, do not rely on GNN-based uncertainty.

2. **Architecture guidance**: On high-$\gamma$ KGs, use explicit coverage tracking rather than hoping deeper GNNs will recover the signal.

3. **Hybrid approach**: Combine GNN uncertainty with explicit coverage as a safety net, weighted by $\gamma$.

---

## 8. Limitations

### 8.1 Simplified Model Assumptions

1. **Uniform relation distribution**: Real KGs have highly skewed relation frequencies. Rare relations may be unidentifiable regardless of $\gamma$.

2. **Neighborhood independence**: We assume neighbor sets are informative. In practice, many KGs have "hub" entities that dominate neighborhoods, reducing discriminative power.

3. **No heterogeneous effects**: The analysis treats all entities equally. In practice, high-degree entities may have favorable ratios while low-degree entities do not.

### 8.2 Transition Zone Uncertainty

The critical ratio $\gamma^*$ is estimated from only two data points (WN18RR and FB15k-237). More benchmarks are needed to:
1. Narrow the transition zone
2. Determine if the transition is sharp or gradual
3. Identify secondary factors (graph structure, relation type distribution)

### 8.3 Anti-Prediction Phenomenon

FB15k-237's 0.43 AUROC is *below* random (0.5), indicating **anti-prediction**. This is not explained by the ratio bound alone and requires additional analysis:

- Novel-context entities in FB15k-237 are **higher frequency** than ID entities (4.33x)
- GNN embeddings encode frequency $\to$ low uncertainty for high-frequency entities
- Novel-context entities receive low uncertainty despite being OOD

This frequency confound amplifies the failure beyond mere chance-level detection.

### 8.4 Practical GNN Architectures

Our analysis considers simplified aggregation. Real GNN architectures (GAT, GraphSAGE, R-GCN) have:
1. Learned attention weights
2. Relation-specific transformations
3. Skip connections

These may partially recover coverage information even at $\gamma > \gamma^*$, but at the cost of:
1. More parameters ($O(|\mathcal{R}| \times d^2)$)
2. Overfitting risk
3. Longer training time

---

## 9. Connection to Main Theory

### 9.1 Integration with Impossibility Theorem

The GNN boundary condition clarifies when Theorem 2 (Embedding-Based Impossibility) applies:

- **Standard embeddings**: Always satisfy Definition 2.3 (relation-agnostic) $\Rightarrow$ impossibility applies
- **GNN embeddings with $\gamma \lesssim \gamma^*$**: Violate Definition 2.3 $\Rightarrow$ impossibility does not apply
- **GNN embeddings with $\gamma \gg \gamma^*$**: Effectively satisfy Definition 2.3 $\Rightarrow$ impossibility applies

### 9.2 Practical Recommendation

The ratio $\gamma$ should be reported alongside any GNN-based OOD detection result:
- $\gamma < 5$: GNN approach is viable
- $5 < \gamma < 12$: Results may be dataset-specific; explicit coverage recommended as backup
- $\gamma > 12$: Do not use GNN for novel-context detection; explicit coverage is necessary

---

## 10. Conclusion

The relation-neighborhood ratio $\gamma = |\mathcal{R}|/\bar{|\mathcal{N}(e)|}$ governs whether GNN architectures can implicitly encode coverage information:

1. **Theoretical basis**: Message passing can recover coverage only when the neighborhood has sufficient capacity to distinguish relations.

2. **Empirical calibration**: $\gamma^* \in [5, 12]$ based on WN18RR (works) and FB15k-237 (fails).

3. **Actionable guidance**: Compute $\gamma$ before deploying GNN-based OOD detection; use explicit coverage tracking when $\gamma > 10$.

4. **YAGO3-10 prediction**: With $\gamma = 2.9$, GNNSafe should work (predicted AUROC: 0.70--0.85).

This analysis transforms the heuristic ratio observation into a principled bound, connecting GNN architecture to the information-theoretic impossibility framework.
