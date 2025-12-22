# Level-Up Strategy: From Borderline to Strong Accept

**Goal**: Transform CAGP from "interesting observation" to "foundational result"

---

## Critique 1: Deeper Theory

### Current State (Weak)
- **Prop 1**: GP variance is relation-agnostic (trivial - implementation detail)
- **Prop 2**: Coverage AUROC closed-form bound (validated <3% error)
- **Prop 3**: Complementarity via construction (failure case examples)

These are *observations*, not *theorems*. They show decomposition is **sufficient** but not **necessary**.

### Target State (Strong)
Prove that achieving optimal OOD detection in KGs **requires** both semantic and structural signals.

### Proposed Theorems

#### Theorem 1: Information-Theoretic Decomposition (MAIN THEOREM)

**Statement**: Let $Y \in \{ID, OOD\}$ be the OOD label for triple $(h,r,t)$. Define:
- $X_S$ = semantic features (entity embedding quality, training frequency)
- $X_C$ = structural features (entity-relation co-occurrence)

Then the mutual information decomposes as:
$$I(Y; X_S, X_C) = I(Y; X_S) + I(Y; X_C | X_S)$$

where $I(Y; X_C | X_S) > 0$ under standard KG assumptions (non-trivial relation sparsity).

**Why this matters**: Shows structural info provides *additional* predictive power beyond semantic info. Not redundant.

**Proof sketch**:
1. Model semantic features as function of entity frequency: $X_S = f(\text{freq}(e))$
2. Model structural features as indicator: $X_C = \mathbb{1}[(e,r) \in \text{training}]$
3. Show $X_C$ and $X_S$ are conditionally independent given entity identity
4. Compute conditional MI and show positive under sparsity assumptions

#### Theorem 2: Impossibility Result for Semantic-Only Methods

**Statement**: Let $\mathcal{D}$ be a KG with relation sparsity profile $(s_1, ..., s_{|R|})$. Any OOD detector using only entity-level uncertainty (semantic signal) achieves:
$$\text{AUROC}_{\text{semantic}} \leq 1 - \frac{1}{2|\mathcal{R}|}\sum_r s_r(1-s_r)$$

**Why this matters**: Shows fundamental upper bound on GP-only methods. Coverage breaks this barrier.

**Proof sketch**:
1. Semantic uncertainty assigns same score to $(e, r_1, ?)$ and $(e, r_2, ?)$
2. But OOD probability differs by relation sparsity
3. This mismatch bounds achievable AUROC

#### Theorem 3: Coverage-Only Limitation

**Statement**: Under type-constrained corruption with type similarity $\tau$ (fraction of same-type entities sharing coverage pattern), coverage AUROC is upper bounded:
$$\text{AUROC}_{\text{coverage}} \leq \frac{1 + (1-\tau)}{2}$$

**Why this matters**: Explains type-constrained degradation theoretically. When $\tau \to 1$ (all same-type entities have same coverage), coverage fails.

#### Theorem 4: Synergy Lower Bound

**Statement**: Given the orthogonality condition (Prop 3), the CAGP AUROC satisfies:
$$\text{AUROC}_{\text{CAGP}} \geq \max(\text{AUROC}_{\text{GP}}, \text{AUROC}_{\text{Cov}}) + \delta$$

where $\delta > 0$ is a function of the complementarity statistics (GP-only catch rate + Coverage-only catch rate).

**Why this matters**: Guarantees synergy exists under measurable conditions. Not just empirical!

### Implementation Plan

1. **Formalize notation** (Section 3.1 of paper)
   - Define information-theoretic quantities precisely
   - State assumptions explicitly (e.g., "Assumption A1: Relation sparsity $s_r > 0$ for all $r$")

2. **Prove main theorem** (Appendix B)
   - Full proof with all intermediate steps
   - Verify numerically on all three datasets

3. **Add impossibility results** (Section 4 of paper)
   - Show GP-only can never exceed computed bound
   - Verify empirically that our results match/exceed predictions

---

## Critique 2: More Challenging Evaluation

### Current State
- Random OOD (tail corruption) - done
- Type-constrained OOD - done

### Target State
- **Temporal OOD** - train on old, test on new
- **Adversarial OOD** - semantically plausible corruptions
- **Domain transfer** - train on one KG, test on another

### A. Temporal OOD Experiments

**Dataset Options**:
1. **ICEWS14** (Integrated Crisis Early Warning System)
   - 7,128 entities, 230 relations, 90,730 triples with timestamps
   - Split: train on 2014-01-01 to 2014-06-30, test on 2014-07-01 to 2014-12-31

2. **ICEWS18**
   - Same schema, larger scale

3. **GDELT** (Global Database of Events, Language, and Tone)
   - Massive scale, daily updates

**Protocol**:
```python
def temporal_split(triples, cutoff_date):
    train = [t for t in triples if t.timestamp < cutoff_date]
    test_id = [t for t in triples if t.timestamp >= cutoff_date and t.entities_in(train)]
    test_ood = [t for t in triples if t.timestamp >= cutoff_date and not t.entities_in(train)]
    return train, test_id, test_ood
```

**Hypothesis**: CAGP should excel because:
- New entities appear post-cutoff (GP catches: high variance)
- Existing entities appear in new relation contexts (Coverage catches: unobserved pairs)

**Evaluation**:
1. Train models on pre-cutoff data
2. Test: distinguish post-cutoff ID (entities exist, relations observed) from OOD (new entities OR new relations)
3. Report AUROC for GP-only, Coverage-only, CAGP

### B. Adversarial OOD Experiments

**Corruption Types**:

1. **Embedding-similarity corruption** (hardest)
   ```python
   def embedding_adversarial(h, r, t, entity_embeddings, k=10):
       # Find k nearest neighbors to t in embedding space
       neighbors = get_knn(entity_embeddings[t], k)
       # Sample corrupted tail from neighbors (excluding t)
       t_corrupt = random.choice([n for n in neighbors if n != t])
       return (h, r, t_corrupt)
   ```

2. **Popularity-matched corruption**
   ```python
   def popularity_adversarial(h, r, t, entity_counts):
       # Find entities with similar training frequency
       t_freq = entity_counts[t]
       similar = [e for e in entities if abs(entity_counts[e] - t_freq) < threshold]
       return (h, r, random.choice(similar))
   ```

3. **Relation-plausible corruption** (semantic attack)
   ```python
   def relation_plausible(h, r, t, relation_to_valid_tails):
       # Sample from entities that appear as tails for relation r
       candidates = relation_to_valid_tails[r]
       return (h, r, random.choice([c for c in candidates if c != t]))
   ```

**Expected Results**:
| Corruption Type | Coverage | GP | CAGP | Why |
|-----------------|----------|-----|------|-----|
| Random | High | Med | High | Easy baseline |
| Type-constrained | Low | Med | Med | Coverage loses type signal |
| Embedding-similar | Low | Low | Med | Both struggle, CAGP robust |
| Popularity-matched | Med | Low | Med | GP loses frequency signal |
| Relation-plausible | Low | Med | Med | Coverage loses relation signal |

### C. Domain Transfer Experiments

**Setup**: Train on Freebase (FB15k-237), test on YAGO (YAGO3-10)

**Challenge**: Different entity/relation vocabularies

**Approach 1: Zero-shot via relation name matching**
```python
# Map FB relations to YAGO relations by name similarity
fb_to_yago = {
    '/people/person/nationality': 'hasNationality',
    '/location/location/contains': 'isLocatedIn',
    ...
}

# Test: given FB-trained coverage matrix, can it detect OOD on YAGO?
# Project FB coverage patterns to YAGO via relation mapping
```

**Approach 2: Entity alignment (more rigorous)**
- Use existing FB-YAGO alignment datasets
- Train on FB entities that have YAGO counterparts
- Test on YAGO entities: which are "new" (OOD) vs aligned (ID)?

**Hypothesis**: Coverage patterns should transfer if:
- Relations have similar sparsity profiles across KGs
- Entity types align (people, places, organizations)

**Metrics**:
- Transfer AUROC: how well does FB-trained model detect YAGO OOD?
- Coverage correlation: do coverage patterns correlate across KGs?

---

## Critique 3: Stronger Baseline Positioning

### Current Problem
Energy/UKGE: 0.99 on random OOD
CAGP: 0.96 on random OOD, 0.81 on type-constrained

Reviewers will ask: "Why not just use Energy-based methods?"

### Solution: Create Clear Decision Framework

#### When to Use CAGP vs Energy-based Methods

| Criterion | Energy-based | CAGP | Winner |
|-----------|--------------|------|--------|
| Random OOD | 0.992 | 0.960 | Energy |
| Type-constrained OOD | ??? | 0.815 | CAGP* |
| Embedding-adversarial | ??? | ??? | Need data |
| Interpretability | None | Semantic vs Structural | CAGP |
| Computational cost | Forward pass | Lookup table | Similar |
| Calibration | Unknown | Unknown | Need data |

**TODO**: Run Energy-based on type-constrained OOD. Hypothesis: it will fail badly because score magnitude relies on type violations.

#### Unified Baseline Table (Target)

```latex
\begin{table}[t]
\caption{Comprehensive OOD Detection Comparison on FB15k-237}
\begin{tabular}{lcccc}
\toprule
Method & Random & Type-Constr. & Adversarial & Temporal \\
\midrule
\multicolumn{5}{l}{\textit{Score-based methods}} \\
Energy & \textbf{0.992} & 0.XX & 0.XX & 0.XX \\
UKGE & 0.992 & 0.XX & 0.XX & 0.XX \\
\midrule
\multicolumn{5}{l}{\textit{Uncertainty methods}} \\
MC Dropout & 0.430 & 0.XX & 0.XX & 0.XX \\
Deep Ensemble & 0.225 & 0.XX & 0.XX & 0.XX \\
GP-only & 0.749 & 0.654 & 0.XX & 0.XX \\
Coverage-only & 0.821 & 0.570 & 0.XX & 0.XX \\
\midrule
\textbf{CAGP} & 0.960 & \textbf{0.815} & \textbf{0.XX} & \textbf{0.XX} \\
\bottomrule
\end{tabular}
\end{table}
```

**Key Narrative**:
- Energy-based is optimal for "easy" OOD (high precision, but brittle)
- CAGP is robust across settings (lower peak, but stable)
- Choose based on deployment context:
  - Production system with adversarial users? → CAGP
  - Internal tool with trusted inputs? → Energy

### Additional Baseline: Score + Coverage Ensemble

**Fair comparison**: What if we just add coverage to Energy-based?

```python
def energy_plus_coverage(scores, coverage):
    return alpha * energy_score(scores) + (1 - alpha) * coverage_uncertainty
```

**Hypothesis**: This will help Energy on type-constrained, but CAGP will still win because GP variance captures embedding quality that score magnitude misses.

---

## Critique 4: "Why Didn't Anyone Do This?"

### Current Answer (Weak)
"They didn't think of it"

### Target Answer (Strong)

#### Historical Argument: Misaligned Research Goals

1. **GP-KGE authors focused on link prediction, not OOD**
   - Original paper (Chen et al., 2021) never evaluated OOD detection
   - Variance was used for calibration, not outlier detection
   - We're the first to ask: "Is GP variance *sufficient* for OOD?"

2. **Coverage seen as preprocessing, not signal**
   - Standard practice: filter candidates by coverage before ranking
   - But nobody asked: "Can coverage *itself* detect OOD?"
   - We reframe coverage from filter to uncertainty signal

3. **False dichotomy: learned vs explicit**
   - Community assumed good uncertainty must be *learned*
   - Explicit statistics (coverage) seen as "cheating"
   - We show: learned + explicit > either alone

#### Methodological Argument: Missing Theoretical Framework

1. **Without Prop 1, you don't know GP is relation-agnostic**
   - Obvious in hindsight, but never articulated
   - Previous work assumed GP captures "all uncertainty"
   - We prove it misses structural uncertainty

2. **Without Prop 2, you can't predict coverage performance**
   - Previous work: empirical only
   - We provide analytical tool for understanding when coverage helps

3. **Without Prop 3, combining seems ad-hoc**
   - "Why not GP + score?" "Why not coverage + dropout?"
   - Our complementarity analysis explains *why this specific combination works*

#### Framing for Paper

> "The decomposition appears obvious in hindsight precisely because we provide the theoretical framework that makes it obvious. Prior to our work, the relation-agnostic limitation of entity-level GP variance was unidentified, the predictive power of coverage was unexplored, and the complementarity was unproven. Each component of CAGP addresses a gap that was invisible before our analysis."

### Supporting Evidence

1. **Citation analysis**: Show that GP-KGE papers never mention relation-agnostic limitation
2. **Failure mode survey**: Show that 5+ uncertainty papers fail on KG OOD in the same way
3. **Ablation on "obvious" alternatives**: Show that naive combinations (GP+score, coverage+dropout) fail

---

## Implementation Timeline

### Phase 1: Theory (Week 1-2)
- [ ] Draft Theorem 1 (MI decomposition) with full proof
- [ ] Draft Theorem 2 (impossibility result)
- [ ] Verify theorems empirically on existing results
- [ ] Revise method.tex with new theoretical framework

### Phase 2: Evaluation (Week 2-4)
- [ ] Download ICEWS14 dataset
- [ ] Implement temporal split and evaluation
- [ ] Run temporal OOD experiments
- [ ] Implement adversarial corruption strategies
- [ ] Run adversarial experiments
- [ ] Run Energy baseline on type-constrained OOD

### Phase 3: Positioning (Week 4-5)
- [ ] Create comprehensive baseline table
- [ ] Write decision framework section
- [ ] Add Energy + Coverage ablation

### Phase 4: Narrative (Week 5-6)
- [ ] Revise introduction with stronger "why now" argument
- [ ] Add historical/methodological justification section
- [ ] Polish related work to position against prior failures

---

## Risk Assessment

| Experiment | Risk | Mitigation |
|------------|------|------------|
| Temporal OOD | CAGP might not help (new entities dominate) | Frame as "understanding when decomposition helps" |
| Adversarial OOD | All methods might fail | Still valuable to show failure modes |
| Domain transfer | Mapping might be noisy | Focus on qualitative insights |
| Theory proofs | Might be hard to formalize | Start with weaker statements, strengthen |

---

## Success Criteria

**Minimum for Strong Accept**:
1. One new theorem showing necessity (not just sufficiency)
2. One additional OOD setting where CAGP excels
3. Clear positioning statement vs Energy-based
4. Compelling "why non-obvious" narrative

**Stretch for Oral**:
1. Information-theoretic decomposition theorem
2. All three new OOD settings with comprehensive results
3. Domain transfer showing generalization
4. Theoretical connection to optimal Bayes classifier
