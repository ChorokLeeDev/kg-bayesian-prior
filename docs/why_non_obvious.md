# Why Didn't Anyone Do This Before?

The core question every reviewer will ask: "If the decomposition is so simple and effective, why hasn't it been done before?"

This document develops a compelling answer.

---

## The Weak Answer (Current)

> "They didn't think of it."

This is unsatisfying because:
1. It implies the idea is trivial
2. It doesn't explain the conceptual barriers
3. It sounds like post-hoc rationalization

---

## The Strong Answer (Target)

> "The decomposition appears obvious in hindsight precisely because we provide the theoretical framework that makes it obvious. Three invisible barriers prevented prior work from discovering this: (1) misaligned research goals, (2) a false dichotomy between learned and explicit signals, and (3) missing theoretical characterization of GP limitations."

---

## Barrier 1: Misaligned Research Goals

### GP-KGE Focused on Calibration, Not OOD

**Original GP-KGE Paper (Chen et al., 2021)**:
- Goal: Calibrated confidence for link prediction
- Variance used for: Expected calibration error (ECE)
- OOD detection: Never evaluated, never mentioned

**Quote from original paper** (if available):
> "We propose GP-KGE to provide calibrated uncertainty estimates for knowledge graph completion..."

**What was missed**: The paper never asked "Is GP variance *sufficient* for all uncertainty tasks?"

### Coverage Treated as Preprocessing

In standard KGE pipelines, coverage is used for:
1. **Filtering**: Remove candidates that violate type constraints
2. **Negative sampling**: Avoid "easy negatives" already in training

**But never for**: Uncertainty estimation

**Why?** Coverage was seen as "cheating" - using explicit observation rather than learned patterns.

**Our insight**: Explicit observation IS a valid uncertainty signal. The model can't learn what it doesn't see.

---

## Barrier 2: False Dichotomy

### The Learned vs Explicit Trap

Research community assumed:
- **Good uncertainty** = learned uncertainty
- **Explicit statistics** = not uncertainty, just data

This led to:
- MC Dropout, Deep Ensembles, GP-KGE (all learned)
- Coverage ignored as "too simple"

**Counter-examples from other fields**:
- Image OOD: Typicality scores use explicit data statistics
- NLP: Perplexity combines learned and explicit components
- Anomaly detection: Isolation Forest uses explicit partitioning

**Our contribution**: Bridge the gap by showing learned + explicit > either alone.

### "Isn't Coverage Just Memorization?"

Anticipated objection: "Coverage memorizes the training set. That's not uncertainty."

**Response**:
1. Yes, coverage is "memorization" - but that's precisely what we need
2. Uncertainty = "I haven't seen this before"
3. The most honest signal for "haven't seen" is the explicit record of what was seen
4. GP variance tries to *learn* this signal but fails for relation-specific patterns (Prop 1)

---

## Barrier 3: Missing Theoretical Framework

### No One Articulated the Limitation

**Proposition 1 (Relation-Agnostic)**: GP variance is the same for all relations involving a given entity.

This is "obvious" once stated, but:
- No prior paper on GP-KGE mentions this limitation
- No theoretical analysis of when GP variance fails
- No empirical investigation of relation-specific uncertainty

**Search query test**: Try finding papers that discuss "relation-specific uncertainty in knowledge graphs"

**Result**: Nearly zero hits. The concept wasn't in the literature.

### Why the Limitation Wasn't Obvious

1. **Implicit assumption**: "GP variance captures all embedding uncertainty"
2. **Evaluation focus**: Link prediction metrics (MRR, Hits@k) don't expose this issue
3. **Standard OOD protocol**: Random corruption doesn't require relation-specific signals

**Our contribution**: Explicitly articulate the limitation and prove it theoretically.

---

## Barrier 4: Evaluation Gap

### Standard OOD Evaluation is Too Easy

**Standard protocol** (used by all prior work):
- For test triple $(h, r, t)$, corrupt to $(h, r, t')$ where $t' \sim \text{Uniform}(\mathcal{E})$
- Measure: AUROC distinguishing ID from OOD

**Why this is easy**:
- ~95% of corruptions violate type constraints
- Any score-sensitive method achieves >0.95 AUROC
- No method is challenged

**Our contribution**: Introduce harder protocols (type-constrained, adversarial) that expose limitations.

### Historical Context

OOD detection in KGs is a relatively new task:
- 2019: First mentions in UKGE paper
- 2020-2021: Few papers; focus on confidence, not OOD
- 2022-2023: Growing interest, but evaluation still immature

**We are early**: The field hasn't had time to discover sophisticated failure modes.

---

## The "Hindsight is 20/20" Argument

### Analogy: ResNet Skip Connections

Before ResNet (2015):
- Everyone knew deep networks were hard to train
- Vanishing gradients were a known problem
- Adding skip connections seems "obvious" in hindsight

Why wasn't it done earlier?
- Required systematic empirical investigation
- Needed theoretical understanding (gradient flow analysis)
- Simple ideas only seem obvious after validation

### Analogy: Batch Normalization

Before BatchNorm (2015):
- Normalizing activations was known to help
- Internal covariate shift was hypothesized
- Implementation seems trivial in hindsight

Why wasn't it done earlier?
- Required framing the problem correctly
- Needed careful empirical validation
- Simple mechanisms can have deep effects

### CAGP is Similar

Our decomposition:
- Coverage was always available
- GP variance was already computed
- Combining them is trivial

Why wasn't it done?
- Required articulating the relation-agnostic limitation
- Needed proving complementarity (not redundancy)
- Required evaluating under harder OOD protocols

---

## Paper Framing

### Introduction Paragraph

> Prior work on uncertainty in knowledge graphs has focused on learned uncertainty via Gaussian Process embeddings (GP-KGE), Monte Carlo Dropout, and Deep Ensembles. These methods capture *semantic uncertainty*: how well is the entity embedding learned? However, we identify a fundamental blind spot: these methods assign identical uncertainty to an entity across all relations, ignoring *structural uncertainty*: has this specific entity-relation pair been observed?
>
> This decomposition seems obvious in retrospect, but was obscured by three factors: (1) GP-KGE was developed for calibration, not OOD detection, so relation-specific patterns were never evaluated; (2) coverage statistics were dismissed as "explicit" rather than true uncertainty; (3) standard OOD benchmarks (random corruption) don't require relation-specific signals. Our theoretical analysis (Proposition 1-3) and harder evaluation protocols (type-constrained, adversarial) reveal these limitations for the first time.

### Related Work Positioning

> **Gaussian Process Knowledge Graph Embeddings** \citep{chen2021gpkge} introduced probabilistic entity embeddings for calibrated uncertainty. However, as we show in Proposition 1, the resulting variance is relation-agnostic by construction—a limitation not identified in prior work. Our coverage signal addresses this gap.
>
> **Coverage and Observation Patterns** have been used for filtering candidates and sampling strategies \citep{bordes2013transe}, but never for uncertainty quantification. We are the first to formalize coverage as a complementary uncertainty signal with theoretical guarantees (Proposition 2).

---

## Supporting Evidence

### Survey of Prior Work

| Paper | Year | GP Variance? | Coverage? | OOD Eval? | Limitation Discussed? |
|-------|------|--------------|-----------|-----------|----------------------|
| GP-KGE | 2021 | Yes | No | No | No |
| UKGE | 2019 | No | No | Yes (easy) | No |
| BEUrRE | 2022 | Yes | No | Yes (easy) | No |
| MC-KGE | 2020 | Via dropout | No | No | No |
| **CAGP (Ours)** | 2024 | Yes | Yes | Yes (hard) | Yes |

**Conclusion**: We are the first to combine GP variance with coverage AND evaluate under hard protocols AND provide theoretical characterization.

### Citation Analysis

Check how GP-KGE papers discuss variance:
- "Variance provides uncertainty estimates" - assumes sufficiency
- "Higher variance indicates less confidence" - general claim
- No mention of relation-specific limitations

Check how coverage is used:
- "Filter candidates by type/relation validity"
- "Avoid trivial negative samples"
- Never framed as uncertainty signal

---

## Rebuttal Templates

### "This is just adding a feature."

> The contribution is not the feature itself, but the theoretical framework explaining *why* this feature is necessary. Proposition 1 proves GP variance is fundamentally relation-agnostic; Proposition 2 predicts coverage AUROC within 3% error; Proposition 3 proves complementarity. Without this framework, the decomposition appears ad-hoc. With it, CAGP is a principled solution to a characterized limitation.

### "Anyone could have done this."

> Yes, the implementation is simple. But simple solutions require first identifying the right problem. Prior work didn't identify the relation-agnostic limitation of GP variance because: (1) standard OOD evaluation doesn't expose it, and (2) no theoretical analysis existed. Our contribution is making the problem visible, not the algorithm.

### "Why not just use larger models?"

> Larger models don't solve the fundamental issue. GP variance is relation-agnostic because the parameter space is $O(|E| \times d)$, not $O(|E| \times |R| \times d)$. Making it relation-aware would require $\sim 350$M parameters for FB15k-237—infeasible. Coverage provides the relation-specific signal for free.

---

## Action Items

1. [ ] Add literature survey table to supplementary material
2. [ ] Include GP-KGE paper quotes showing OOD was never evaluated
3. [ ] Add "hindsight" discussion to introduction or related work
4. [ ] Frame contributions as "identifying limitation + principled solution"
