# RAG Error Decomposition: From KG Uncertainty to LLM Reliability

## Abstract

This document formalizes the connection between our knowledge graph (KG) uncertainty decomposition framework and Retrieval-Augmented Generation (RAG) systems. The central insight is that RAG inherits the "coverage blind spot" we identify in KG systems: when retrieval returns confident but evidence-free results, the downstream LLM hallucinates with high confidence. We provide a theoretical framework for understanding this failure mode and practical recommendations for coverage-aware RAG design.

---

## 1. RAG Error Decomposition

### 1.1 Probabilistic Framework

Let $y$ denote the correct answer to query $q$, let $\hat{y}$ denote the RAG system's response. We decompose RAG error into retrieval and generation components:

$$P(\text{RAG error}) = P(\text{retrieval error}) + P(\text{generation error} \mid \text{correct retrieval}) \cdot P(\text{correct retrieval})$$

More precisely, using the law of total probability:

$$P(\hat{y} \neq y) = P(\hat{y} \neq y \mid D_{\text{rel}}) \cdot P(D_{\text{rel}}) + P(\hat{y} \neq y \mid D_{\neg\text{rel}}) \cdot P(D_{\neg\text{rel}})$$

where:
- $D_{\text{rel}}$ = retrieved documents are relevant to the query
- $D_{\neg\text{rel}}$ = retrieved documents are irrelevant or misleading

**Key observation**: Our KG uncertainty work directly addresses the $P(D_{\neg\text{rel}})$ term---the probability that retrieval returns irrelevant or misleading results with high confidence.

### 1.2 The Retrieval Error Decomposition

Following our KG framework, we further decompose retrieval errors:

$$P(\text{retrieval error}) = \underbrace{P(\text{emerging query})}_{\text{rare query type}} + \underbrace{P(\text{novel context} \mid \neg\text{emerging})}_{\text{coverage blind spot}}$$

| KG Concept | RAG Analog |
|------------|------------|
| Entity $e$ | Query $q$ or Document $d$ |
| Relation $r$ | Task type (QA, summarization, fact-check) |
| Triple $(h, r, t)$ | (Query, Task, Retrieved Document) |
| Coverage $c(e, r)$ | Has this document been retrieved for similar queries? |
| Novel context | Query-document pair never seen in training |
| Emerging entity | Rare query or rare document |

---

## 2. KG-RAG Pipeline Analysis

### 2.1 Standard KG-RAG Architecture

Modern KG-augmented RAG systems follow this pipeline:

```
User Query → Entity Linking → KG Retrieval → Context Assembly → LLM Generation
     q            (e)           (h,r,t)           C                  y_hat
```

**Example**: For query "What diseases does metformin treat?":
1. Entity linking: "metformin" → `dbpedia:Metformin`
2. KG retrieval: `(Metformin, treats, ?)`
3. Context: Retrieved triples + confidence scores
4. Generation: LLM synthesizes answer from context

### 2.2 Where the Coverage Blind Spot Manifests

Our 83% confident-wrong finding (Table 3 in main paper) directly transfers to RAG:

**In KG systems**: Energy's top-100 most confident predictions have 83% zero training evidence.

**In KG-RAG**: When the KG returns high-confidence triples for novel (entity, relation) pairs, the LLM receives authoritative-looking but potentially fabricated "facts."

**Failure cascade**:
```
Zero-coverage KG query → High confidence score → LLM trusts retrieval
                                                        ↓
                                              Confident hallucination
```

### 2.3 Empirical Evidence from RAG Literature

This failure mode is documented across RAG systems:

1. **KILT benchmark** (Petroni et al., 2021): Dense retrievers achieve high confidence on queries where relevant documents do not exist in the corpus.

2. **FreshQA** (Vu et al., 2023): RAG systems confidently answer questions about recent events using outdated documents---a temporal coverage gap.

3. **RARR** (Gao et al., 2023): Post-hoc retrieval reveals that 30-40% of LLM "facts" cannot be supported by retrieved evidence, yet the original generation was confident.

---

## 3. Uncertainty Propagation

### 3.1 The Missing Link: Retrieval Uncertainty to Generation Confidence

Current RAG architectures have a critical gap: **retrieval uncertainty does not propagate to generation confidence**.

**Current systems**:
- Retriever: Returns top-k documents with similarity scores
- Generator: Treats retrieved documents as ground truth
- Output: Single response with no uncertainty indication

**What should happen**:
$$U_{\text{RAG}}(q, \hat{y}) = f(U_{\text{retrieval}}(q, D), U_{\text{generation}}(\hat{y} \mid D))$$

### 3.2 Semantic vs. Structural Uncertainty in RAG

Applying our decomposition framework to RAG:

**Semantic uncertainty** $U_{\text{sem}}^{\text{RAG}}$:
$$U_{\text{sem}}^{\text{RAG}}(q, d) = 1 - \cos(\phi_q(q), \phi_d(d))$$

Captures embedding-space distance. Suffers from the same blind spot as KG: a query can be semantically similar to a document without the document being relevant for this specific query type.

**Structural uncertainty** $U_{\text{str}}^{\text{RAG}}$:
$$U_{\text{str}}^{\text{RAG}}(q, d) = \mathbf{1}[\text{cooccur}(q_{\text{cluster}}, d) = 0]$$

Binary indicator: has this document ever been retrieved for queries in the same cluster? This is the coverage check that semantic methods miss.

### 3.3 Impossibility Theorem (RAG Version)

**Theorem (adapted from Theorem 1)**: Let $U(q, d) = F(\phi(q), \phi(d))$ where $\phi$ is a relation-agnostic embedding. Under assumptions analogous to A1-A3:

$$\text{AUROC}(U, \mathcal{D}_{\text{novel}}) \leq \frac{1}{2} + O(\epsilon)$$

**Interpretation**: Any uncertainty estimator based solely on query/document embeddings cannot detect novel query-document combinations where the retrieval system has never seen this pairing, yet both the query type and document type are individually common.

**Proof sketch**: Same as KG case. Query frequency and document frequency determine embedding uncertainty. Novel combinations have frequency-matched in-distribution counterparts. QED.

### 3.4 Why Current Systems Fail

| System | Uncertainty Signal | Coverage-Aware? | Failure Mode |
|--------|-------------------|-----------------|--------------|
| DPR (Karpukhin et al., 2020) | Cosine similarity | No | High confidence on unseen query-doc pairs |
| Contriever (Izacard et al., 2022) | Embedding distance | No | Same blind spot |
| RETRO (Borgeaud et al., 2022) | Retrieval-augmented perplexity | Partial | Perplexity conflates coverage with semantic fit |
| REPLUG (Shi et al., 2023) | Ensemble disagreement | No | Ensembles share coverage blind spot (Theorem 2) |

---

## 4. Practical Implications

### 4.1 Decision Framework: Abstain vs. Retrieve vs. Generate

Based on our KG findings, we propose a three-way decision for RAG systems:

```
                    ┌─────────────────────────────────────────┐
                    │         Query q arrives                 │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │  Check coverage: c(q_cluster, r) = ?    │
                    │  (Has this query type been seen?)       │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
        c(q,r) = 0              c(q,r) = 1             c(q,r) = 1
     (novel context)         (known context)        (known context)
     High U_str              Low U_str              Low U_str
              │                       │                       │
              ▼                       ▼                       ▼
        ┌─────────┐           ┌─────────────┐         ┌─────────────┐
        │ ABSTAIN │           │   CHECK     │         │  GENERATE   │
        │   or    │           │  SEMANTIC   │         │    with     │
        │  FLAG   │           │ UNCERTAINTY │         │ CONFIDENCE  │
        └─────────┘           └──────┬──────┘         └─────────────┘
                                     │
                              ┌──────┴──────┐
                              │             │
                              ▼             ▼
                        High U_sem     Low U_sem
                              │             │
                              ▼             ▼
                        ┌─────────┐   ┌─────────────┐
                        │ ABSTAIN │   │  GENERATE   │
                        │   or    │   │    with     │
                        │ HEDGE   │   │ CONFIDENCE  │
                        └─────────┘   └─────────────┘
```

### 4.2 Coverage-Aware RAG Architecture

**Minimal modification to existing RAG**:

```python
class CoverageAwareRAG:
    def __init__(self, retriever, generator, coverage_table):
        self.retriever = retriever  # DPR, Contriever, etc.
        self.generator = generator  # GPT-4, Llama, etc.
        self.coverage = coverage_table  # Query cluster -> Document set

    def answer(self, query: str, task_type: str) -> Tuple[str, float]:
        # 1. Retrieve
        docs, sem_scores = self.retriever.retrieve(query, k=5)

        # 2. Check coverage (the critical addition)
        q_cluster = self.get_query_cluster(query)
        coverage_flags = [self.coverage.check(q_cluster, task_type, d)
                         for d in docs]

        # 3. Compute combined uncertainty
        u_sem = 1 - np.mean(sem_scores)
        u_str = 1 - np.mean(coverage_flags)  # Fraction with zero coverage
        u_combined = 0.5 * u_sem + 0.5 * u_str

        # 4. Decide: abstain, hedge, or generate
        if u_str > 0.8:  # Most retrieved docs have no coverage
            return "I don't have reliable information on this.", 0.0
        elif u_combined > 0.5:
            # Generate with hedge
            context = self.filter_by_coverage(docs, coverage_flags)
            response = self.generator.generate(query, context)
            return f"Based on limited evidence: {response}", 1 - u_combined
        else:
            # Generate with confidence
            response = self.generator.generate(query, docs)
            return response, 1 - u_combined

    def filter_by_coverage(self, docs, flags):
        """Keep only documents with coverage."""
        return [d for d, f in zip(docs, flags) if f > 0]
```

### 4.3 Scalability: Bloom Filters for RAG Coverage

For large-scale RAG systems, exact coverage tracking may be prohibitive. Our Bloom filter analysis (Table 2 in main paper) directly applies:

| Scale | Exact Storage | Bloom (1% FPR) | Recall Drop |
|-------|--------------|----------------|-------------|
| 1M query-doc pairs | 50 MB | 10 MB | 0.14pp |
| 100M query-doc pairs | 5 GB | 1 GB | 0.14pp |
| 10B query-doc pairs (web-scale) | 500 GB | 100 GB | 0.14pp |

**Key result**: The recall degradation is constant regardless of scale, making coverage tracking practical even for web-scale RAG.

### 4.4 Training Implications

Our coverage-aware training experiment (Appendix H in paper) shows that attempting to learn coverage from embeddings **fails**:

> AUROC decreases from 0.63 to 0.58 (-4.3pp) when training with coverage-aware regularization.

**Implication for RAG training**: Do not attempt to make retrievers "coverage-aware" through training. Keep retriever training focused on semantic relevance. Add coverage tracking as a post-hoc module.

This mirrors our recommendation for KG systems:
> "Keep training as-is for link prediction; add coverage tracking as a post-hoc module. Attempting to learn coverage from embeddings is provably futile (Theorem 2)."

---

## 5. Connections to RAG Literature

### 5.1 Foundational RAG Work

**RAG** (Lewis et al., 2020): Introduced retrieval-augmented generation but provides no uncertainty quantification. The retrieval confidence (dot-product score) is used only for weighting, not for abstention.

**REALM** (Guu et al., 2020): Pre-trains retriever jointly with masked language modeling. Same blind spot: embeddings cannot encode coverage.

**Atlas** (Izacard et al., 2023): Scales RAG with instruction tuning. Does not address coverage gaps.

### 5.2 Recent Uncertainty-Aware RAG

**Self-RAG** (Asai et al., 2023): Trains LLM to self-assess retrieval relevance. This is a semantic assessment, not coverage checking. Per our Theorem 2, self-assessment inherits the embedding blind spot.

**CRAG** (Yan et al., 2024): Corrective RAG with web search fallback. Uses confidence thresholds but these are semantic, not structural. Our framework suggests adding coverage checks before the correction step.

**RAG-Fusion** (Rackauckas, 2023): Reciprocal rank fusion of multiple retrievers. Ensemble methods share the blind spot (per our Theorem 2, applied to RAG).

### 5.3 Hallucination Detection

**FActScore** (Min et al., 2023): Post-hoc fact verification. This is complementary to our approach: FActScore catches hallucinations after generation; coverage checking prevents them at retrieval time.

**Chain-of-Note** (Yu et al., 2023): LLM generates notes about retrieval relevance. Same limitation as Self-RAG: semantic assessment, not coverage.

---

## 6. Theoretical Implications

### 6.1 Information-Theoretic Perspective

Extending Theorem 3 (Information-Theoretic Characterization) to RAG:

$$I(U_{\text{total}}; Y_{\text{retrieval\_error}}) = I(U_{\text{str}}; Y) + I(U_{\text{sem}}; Y \mid U_{\text{str}})$$

where:
- $I(U_{\text{str}}; Y)$ captures novel query-document combinations perfectly
- $I(U_{\text{sem}}; Y \mid U_{\text{str}})$ provides additional signal only when coverage overlap exists

**Prediction**: On knowledge-intensive QA (e.g., Natural Questions), where most test queries have coverage for relevant documents, semantic uncertainty should help. On time-sensitive QA (e.g., FreshQA), where queries often target information not in the corpus, coverage alone should dominate.

### 6.2 The Confidence-Evidence Gap

Our 83% confident-wrong finding explains a puzzling phenomenon in RAG: LLMs confidently hallucinate despite having access to retrieval.

**Mechanism**:
1. Retriever returns documents with high semantic similarity
2. High similarity → high confidence score
3. LLM conditions on "confident" retrieval
4. Output inherits spurious confidence

**The missing link**: Confidence scores measure semantic similarity, not evidentiary support. A document can be semantically similar (e.g., discusses the same topic) without providing evidence for the specific claim.

### 6.3 Foundation Model Implications

Our Theorem 2 (Embedding-Based Impossibility) has implications for foundation models used in RAG:

> "Any embedding-based uncertainty inherits the blind spot: embeddings encode semantic similarity, not which relations each entity has seen. More training data, larger models, or foundation architectures do not escape this---the limitation is structural, not statistical."

This applies directly to large retrieval models (e.g., E5, BGE, Instructor). Scaling embedding dimension, training data, or model size does not address the coverage blind spot.

---

## 7. Recommendations for RAG Deployment

Based on our theoretical framework and empirical findings:

1. **Always track query-document coverage.** A hash table lookup (or Bloom filter for web-scale) catches the failure mode invisible to all embedding-based methods.

2. **Stratify retrieval confidence.** Report semantic confidence and coverage separately. High semantic confidence with zero coverage should trigger abstention or hedging.

3. **Do not trust embedding similarity alone.** High cosine similarity does not mean high evidentiary support.

4. **Flag novel query types.** When a query falls outside the training distribution of query clusters, flag as uncertain regardless of retrieval scores.

5. **Propagate uncertainty to generation.** Pass retrieval uncertainty to the LLM through prompting:
   ```
   [HIGH CONFIDENCE RETRIEVAL] Document: ...
   [LOW COVERAGE WARNING] This query type has limited training data. Document: ...
   [ABSTAINING] No reliable documents found for this query type.
   ```

6. **Post-hoc coverage is cheap.** Unlike retraining, coverage tracking is a constant-time lookup. It can be added to any existing RAG system without modification to the retriever or generator.

---

## 8. Future Directions

1. **Query cluster discovery**: Automatically identifying query types for coverage tracking. Current approach requires manual taxonomy or clustering.

2. **Temporal coverage decay**: Documents become stale. Coverage should include a temporal component: $c(q, d, t) = c(q, d) \cdot \exp(-\lambda(t - t_d))$.

3. **Multi-hop coverage**: For multi-hop QA (e.g., HotpotQA), coverage should track the full reasoning chain, not just individual query-document pairs.

4. **Coverage-aware fine-tuning**: Can we train LLMs to recognize and respond to coverage signals? Initial experiments with coverage-aware prompting show promise.

5. **Conformal prediction for RAG**: Providing guaranteed coverage (in the statistical sense) for RAG outputs. Complementary to our structural coverage.

---

## References

- Asai, A., et al. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection.
- Borgeaud, S., et al. (2022). Improving language models by retrieving from trillions of tokens.
- Gao, L., et al. (2023). RARR: Researching and revising what language models say.
- Guu, K., et al. (2020). REALM: Retrieval-augmented language model pre-training.
- Izacard, G., et al. (2022). Unsupervised dense information retrieval with contrastive learning.
- Izacard, G., et al. (2023). Atlas: Few-shot learning with retrieval augmented language models.
- Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
- Min, S., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision.
- Petroni, F., et al. (2021). KILT: A benchmark for knowledge intensive language tasks.
- Rackauckas, C. (2023). RAG-Fusion: A new take on retrieval-augmented generation.
- Shi, W., et al. (2023). REPLUG: Retrieval-augmented black-box language models.
- Vu, T., et al. (2023). FreshQA: A dynamic QA benchmark for testing real-world knowledge.
- Yan, S., et al. (2024). Corrective retrieval augmented generation.
- Yu, W., et al. (2023). Chain-of-Note: Enhancing robustness in retrieval-augmented language models.

---

## Appendix: Mapping to Main Paper Results

| Paper Finding | RAG Implication |
|---------------|-----------------|
| 83% confident-wrong (Table 3) | RAG retrievers are most confident on zero-evidence queries |
| Theorem 1 (Impossibility) | Embedding-based retrieval uncertainty cannot detect novel query-doc pairs |
| Theorem 2 (Embedding Impossibility) | Foundation model retrievers inherit the blind spot |
| Coverage alone achieves 0.99 AUROC (ICEWS) | Simple coverage check dominates complex uncertainty methods |
| Bloom filter scalability (Table 2) | Coverage tracking is practical at web scale |
| Training cannot fix the blind spot | Do not try to make retrievers "coverage-aware" through training |

---

*Last updated: 2026-03-09*
*Connection to main paper: NeurIPS 2026 submission on KG uncertainty decomposition*
