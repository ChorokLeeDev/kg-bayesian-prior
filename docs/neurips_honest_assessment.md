# NeurIPS 2026: Honest Assessment & Action Plan

## Executive Summary

**Current State:** Interesting empirical finding, weak theoretical foundation.
**Novelty:** Method = Weak, Insight = Medium, Theory = None
**NeurIPS Probability:** 50% (current) → 65-70% (with improvements)

---

## Critical Self-Assessment

### What We Have

| Component | Status | Strength |
|-----------|--------|----------|
| CAGP algorithm | Done | ❌ Trivial (α × GP + (1-α) × Coverage) |
| Empirical results | Done | ✅ Strong (0.87-0.96 AUROC) |
| Synergy demonstration | Done | ✅ Novel finding |
| Decomposition framework | Named | ⚠️ Needs justification |
| Theoretical foundation | Missing | ❌ Critical gap |

### The Hard Questions

#### Q1: "GP Variance, relation별 구분 못한다?"

**Answer: Yes, by implementation.**

```python
# Current GP-KGE: entity-level variance only
self.entity_logvar = torch.zeros(num_entities, dim)

def get_uncertainty(heads, relations, tails):
    h_var = exp(self.entity_logvar[heads])  # relation NOT used
    t_var = exp(self.entity_logvar[tails])  # relation NOT used
    return (h_var + t_var) / 2
```

Why not relation-aware variance?
- Would need `entity_relation_logvar[num_entities, num_relations, dim]`
- Parameters: 40K × 237 × 100 = **950M parameters** (infeasible)
- No one does this in practice

**This is a valid observation, but not deeply novel.**

#### Q2: "둘 다 필요하다는게 Novel?"

**Honest answer: Weak novelty.**

| Claim | Reviewer Response |
|-------|-------------------|
| "GP + Coverage = better" | "Adding features improves performance. Obvious." |
| "Synergy exists" | "Two signals combined usually help. So what?" |
| "Semantic + Structural decomposition" | "Post-hoc naming. Where's the theory?" |

**Why hasn't anyone done this?**
- Too simple for a paper?
- Assumed GP learns coverage implicitly?
- Just didn't think of it?

**The real novelty (if any):**
1. Explaining WHY GP-KGE fails on low-diversity KGs
2. Showing GP doesn't learn relation-specific coverage
3. Demonstrating the signals are complementary, not redundant

---

## Gap Analysis for NeurIPS

### What NeurIPS Reviewers Will Ask

| Question | Our Current Answer | Strength |
|----------|-------------------|----------|
| "What's the theoretical contribution?" | None | ❌ Fatal |
| "Why this decomposition?" | "It works empirically" | ⚠️ Weak |
| "Is this just feature engineering?" | "...maybe?" | ❌ Bad |
| "How does this generalize?" | "2 datasets so far" | ⚠️ Weak |

### The Fatal Gap: No Theory

**Current state:**
> "We show empirically that combining GP variance and coverage works better."

**What we need:**
> "We prove that under conditions X, Y, Z, the optimal uncertainty estimator decomposes into semantic and structural components, with coverage being a sufficient statistic for structural uncertainty."

---

## Required Improvements

### Priority 1: Theoretical Foundation (CRITICAL)

**Proposed Theorem:**

> **Theorem (Coverage Sufficiency):** Under random tail corruption with relation sparsity $s_r$ for relation $r$, the relation-specific coverage $c(e,r) \in \{0,1\}$ is a sufficient statistic for distinguishing ID from OOD triples, achieving:
>
> $$\text{AUROC} \geq 1 - (1-s_r)^2$$
>
> where $s_r = 1 - \frac{|\{e : (e,r,\cdot) \in \mathcal{T}\}|}{|E|}$ is the sparsity of relation $r$.

**Proof sketch:**
1. ID triple (h, r, t): Both h and t observed with r → coverage = 2
2. OOD triple (h, r, t'): t' random → P(coverage[t',r]=1) = 1-s_r
3. Expected coverage gap = s_r → AUROC scales with sparsity

**Why this matters:** Explains why coverage works, gives theoretical bound, makes the paper "scientific" not just "empirical."

### Priority 2: Why GP Variance Helps

**Proposed Theorem:**

> **Theorem (Complementarity):** GP variance captures uncertainty about entity embedding quality, which is orthogonal to structural coverage. Specifically:
>
> $$I(\text{GP}_\sigma; \text{OOD}) \not\subseteq I(\text{Coverage}; \text{OOD})$$
>
> where $I(X;Y)$ denotes mutual information.

**Intuition:**
- Coverage: "Have I seen this entity with this relation?"
- GP variance: "How well do I know this entity's embedding?"

An entity can:
- Be seen with relation (coverage=1) but have high variance (inconsistent contexts)
- Not be seen with relation (coverage=0) but have low variance (very consistent elsewhere)

### Priority 3: More Experiments

| Experiment | Purpose | Status |
|------------|---------|--------|
| YAGO3-10 | Third dataset validation | ⏳ Pending |
| α ablation (0, 0.25, 0.5, 0.75, 1) | Show sensitivity | ⏳ Pending |
| RGCN baseline | GNN comparison | ⏳ Pending |
| CompGCN baseline | GNN comparison | ⏳ Pending |
| Deep Ensembles | Standard UQ baseline | ⏳ Pending |
| Different OOD types | Robustness | ⏳ Pending |

### Priority 4: Stronger Narrative

**Current framing (weak):**
> "We combine GP variance with coverage and it works better."

**Better framing:**
> "We discover that GP-KGE's uncertainty is fundamentally incomplete: it captures semantic uncertainty (embedding quality) but misses structural uncertainty (relation-specific observation). We prove coverage is sufficient for structural uncertainty and show the combination is synergistic."

**Best framing:**
> "We challenge the implicit assumption in probabilistic KGE that learned variance captures all uncertainty. Through theoretical analysis and comprehensive experiments, we show that structural uncertainty—whether an entity has been observed with a specific relation—is orthogonal to semantic uncertainty and equally important. This decomposition explains prior method failures and yields a simple, principled improvement."

---

## Action Plan

### Phase 1: Theory (Week 1-2) 🔴 CRITICAL

| Task | Owner | Deadline |
|------|-------|----------|
| Formalize Coverage Sufficiency theorem | - | Week 1 |
| Prove AUROC lower bound | - | Week 1 |
| Formalize Complementarity theorem | - | Week 2 |
| Write theory section draft | - | Week 2 |

### Phase 2: Experiments (Week 2-3)

| Task | Owner | Deadline |
|------|-------|----------|
| Run YAGO3-10 full experiments | - | Week 2 |
| α ablation study | - | Week 2 |
| Add RGCN baseline | - | Week 3 |
| Add Deep Ensembles baseline | - | Week 3 |
| Alternative OOD scenarios | - | Week 3 |

### Phase 3: Writing (Week 3-5)

| Task | Owner | Deadline |
|------|-------|----------|
| Introduction draft | - | Week 3 |
| Related work | - | Week 3 |
| Theory section | - | Week 4 |
| Experiments section | - | Week 4 |
| Full paper draft | - | Week 5 |

### Phase 4: Polish (Week 5-6)

| Task | Owner | Deadline |
|------|-------|----------|
| Internal review | - | Week 5 |
| Revisions | - | Week 6 |
| Final polish | - | Week 6 |
| Supplementary material | - | Week 6 |

---

## Risk Assessment

### Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Theory doesn't hold | 30% | Fatal | Test on more datasets first |
| Reviewers say "obvious" | 50% | High | Strong narrative + theory |
| Experiments don't generalize | 20% | High | Run YAGO early |
| Scooped | 10% | Fatal | Move fast |

### Reviewer Simulation

**Likely Accept (40%):**
> "Novel decomposition framework with theoretical justification and strong empirical results. The insight that GP-KGE misses structural uncertainty is valuable."

**Likely Reject (60%):**
> "The method is trivial. The 'decomposition' is just adding a feature. Theory is shallow. Limited to OOD detection on KGs."

---

## Decision Points

### Go/No-Go Criteria

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Theory proven? | Yes | ❌ No |
| 3+ datasets consistent? | Yes | ⚠️ 2/3 |
| Beats all baselines? | Yes | ⚠️ Need more |
| Compelling narrative? | Yes | ⚠️ Needs work |

**Decision deadline:** End of Week 2

If theory doesn't work out → pivot to ICLR/AAAI with empirical focus.

---

## Alternative Venues

| Venue | Deadline | Fit | Strategy |
|-------|----------|-----|----------|
| NeurIPS 2026 | May 2026 | Ambitious | Full theory + experiments |
| ICLR 2026 | Sep 2025 | Good | Empirical focus OK |
| AAAI 2026 | Aug 2025 | Safe | Practical contribution |
| KDD 2026 | Feb 2026 | Good | Application focus |

---

## Summary

### The Honest Truth

| Aspect | Assessment |
|--------|------------|
| Is CAGP novel? | **No** (trivial algorithm) |
| Is the insight novel? | **Somewhat** (no one wrote it down) |
| Is it NeurIPS-worthy? | **Not yet** (needs theory) |
| Can it become NeurIPS-worthy? | **Yes** (with work) |

### What Must Happen

1. **Theory** - Prove why the decomposition is right
2. **More experiments** - Show generalization
3. **Strong narrative** - Sell the insight, not the method

### Next Immediate Action

**Start with the theory.** If we can't prove the theorems, the NeurIPS path is dead. Pivot to ICLR/AAAI.

---

## Appendix: Key Equations

### Coverage Uncertainty
$$U_{\text{structural}}(h, r, t) = 2 - \mathbb{1}[(h,r,\cdot) \in \mathcal{T}] - \mathbb{1}[(t,r,\cdot) \in \mathcal{T}]$$

### GP Uncertainty
$$U_{\text{semantic}}(h, r, t) = \frac{1}{2}(\sigma^2_h + \sigma^2_t)$$

### CAGP
$$U_{\text{CAGP}}(h, r, t) = \alpha \cdot U_{\text{semantic}} + (1-\alpha) \cdot U_{\text{structural}}$$

### Theoretical AUROC Bound (to prove)
$$\text{AUROC} \geq 1 - (1-s)^2 \text{ where } s = \text{relation sparsity}$$
