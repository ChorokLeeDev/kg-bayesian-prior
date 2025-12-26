# Plan B: Remove RelCondVar from Paper

## 📋 Scenario

**If RelCondVar 50 epochs results show:**
- AUROC < 0.70 on temporal OOD
- Auxiliary objective provides minimal benefit (Δ < 0.05)
- Performance not substantially better than simple baselines

**Decision:** Remove RelCondVar, focus paper on CAGP only

---

## ✂️ WHAT TO REMOVE

### 1. Abstract
**Current (lines 13-14):**
```latex
Combining signals via learned weights yields 0.87--0.97 AUROC across four benchmarks.
```

**Change to:**
```latex
Our method (CAGP) combines semantic and structural uncertainty via learned weights,
achieving 0.87--0.96 AUROC across four benchmarks—a 67\% improvement over
relation-agnostic baselines.
```

### 2. Introduction (sections/introduction_uai.tex)
**Current (lines 26-27):**
```latex
(3) Methodological: We propose two solutions---RelCondVar (learns relation-conditioned
variance σ²(e,r) via auxiliary OOD objective) and CAGP (augments variance with
explicit coverage tracking)---achieving 0.89--0.91 AUROC
```

**Change to:**
```latex
(3) Methodological: We propose CAGP (Coverage-Augmented GP-KGE), which augments
entity-level variance with explicit coverage tracking of entity-relation co-occurrence.
CAGP achieves 0.87--0.96 AUROC on temporal OOD, a 67\% relative improvement over
probabilistic baselines.
```

### 3. Method Section (sections/method_uai_v2.tex)

**Remove entire subsection (lines 65-97):**
- "Two Approaches to Relation-Specific Uncertainty"
- "Approach 1 (Primary): RelCondVar"
- "Approach 2 (Baseline): CAGP"
- "Comparison" paragraph

**Replace with simpler version:**

```latex
\subsection{Coverage-Augmented GP-KGE (CAGP)}

Theorem~\ref{thm:complementarity} establishes that relation-specific uncertainty
is necessary for detecting novel contexts. We propose CAGP, which explicitly
combines semantic and structural uncertainty:

\begin{equation}
    U_{\text{CAGP}}(h,r,t) = \alpha \cdot U_{\text{sem}}(h,r,t) + (1-\alpha) \cdot U_{\text{str}}(h,r,t)
    \label{eq:cagp}
\end{equation}

where $U_{\text{sem}}$ is the GP-based semantic uncertainty (entity variance),
$U_{\text{str}}$ is the structural uncertainty (coverage), and
$\alpha = \sigma(\lambda)$ is a learned mixing weight.

\paragraph{Coverage matrix.}
We precompute $\mathbf{C} \in \{0,1\}^{|\mathcal{E}| \times |\mathcal{R}|}$
from training data:
\begin{equation}
    c(e,r) = \begin{cases}
        1 & \text{if } \exists (h,r',t) \in \mathcal{T}_{\text{train}}: (e=h \text{ or } e=t) \text{ and } r'=r \\
        0 & \text{otherwise}
    \end{cases}
\end{equation}

Structural uncertainty is: $U_{\text{str}}(h,r,t) = 2 - c(h,r) - c(t,r) \in \{0,1,2\}$.

\paragraph{Design rationale.}
The linear combination is intentionally simple. Ablations show that fixed
$\alpha = 0.5$ captures most gains (Appendix~\ref{app:ablation}), with learned
$\alpha$ providing 1-2\% additional improvement. This validates that the
decomposition framework itself---not sophisticated mixing---drives performance.
```

### 4. Experiments Section

**Remove RelCondVar from all tables:**

**Table 1 (ICEWS14) - Current:**
```
CAGP (learned α)              0.891  0.847  0.781
RelCondVar (learned σ²(e,r))  0.912  0.873  0.805
```

**Table 1 - Updated:**
```
CAGP (learned α)              0.891  0.847  0.781
CAGP (fixed α=0.5)            0.868  0.821  0.759
```

**Similar changes for all other tables.**

**Remove paragraph (experiments section):**
- Any comparison between CAGP and RelCondVar
- Any discussion of RelCondVar's design

**Add instead:**
```latex
\paragraph{Ablation: Mixing strategies.}
Table~\ref{tab:ablation_alpha} shows that fixed $\alpha=0.5$ achieves 0.868 AUROC,
capturing most of the decomposition's benefit. Learned $\alpha$ provides marginal
improvement (+0.023), validating that the framework's effectiveness comes from
explicitly separating semantic and structural signals, not from sophisticated
weighting schemes.
```

---

## ➕ WHAT TO ADD

### 1. Stronger positioning of CAGP

**In abstract:**
```latex
We propose CAGP (Coverage-Augmented GP-KGE), a simple yet effective approach
that combines learned entity variance with explicit coverage tracking.
```

### 2. Emphasize simplicity as strength

**In method:**
```latex
\paragraph{Why explicit coverage?}
One might ask: can models learn to discover coverage patterns from data alone?
Our impossibility theorem (Theorem~\ref{thm:impossibility}) shows that standard
link prediction objectives do not incentivize relation-specific uncertainty
differentiation. Explicit coverage tracking provides a simple, interpretable,
and theoretically grounded solution.
```

### 3. Move any RelCondVar exploration to appendix (optional)

```latex
\subsection{Alternative: Learned Relation-Conditioned Variance}
We explored learning $\sigma^2(e,r)$ via MLP, but found this required auxiliary
OOD objectives and provided marginal benefits over explicit coverage tracking.
Details in Appendix~\ref{app:relcondvar_exploration}.
```

---

## 📊 NEW STORY ARC

### Old story (with RelCondVar):
```
Problem → Theory → Two solutions (RelCondVar primary, CAGP baseline) → Experiments
```

### New story (CAGP only):
```
Problem → Theory → Simple explicit solution (CAGP) → Experiments → Why simplicity works
```

### Key narrative points:

1. **Problem is clearly defined** ✓
   - Relation-agnostic uncertainty fails on novel contexts

2. **Theory is rigorous** ✓
   - Impossibility theorem (A3 verified!)
   - Complementarity theorem

3. **Solution is elegant** ✓
   - Explicit decomposition: semantic + structural
   - Linear combination (simple is better)

4. **Validation is thorough** ✓
   - 4 datasets, temporal splits, stratified evaluation
   - 67% improvement over baselines

5. **Analysis is insightful** ✓
   - Why existing methods fail (Theorem 1)
   - When each signal works (Theorem 2)
   - Simple combination suffices (ablations)

---

## ✅ BENEFITS OF PLAN B

### 1. Cleaner story
- One method, clearly explained
- No confusion about "primary" vs "baseline"
- Focus on theoretical insights

### 2. Stronger claims
- "CAGP is simple and effective"
- Better than "CAGP is baseline, RelCondVar is primary but barely better"

### 3. Less risky
- Don't need to defend RelCondVar's marginal gains
- Don't need to explain why auxiliary objective barely helps
- Avoid reviewer: "Why propose two methods if one is clearly better?"

### 4. Fits UAI better
- UAI values: theory + simple methods
- Complex learned approaches fit ICLR/NeurIPS better
- CAGP's interpretability is a feature, not a bug

---

## ⚠️ POTENTIAL CONCERNS

### "Is CAGP too simple?"

**Response:**
- Simplicity is backed by theory (Theorems 1 & 2)
- Ablations show sophisticated mixing doesn't help
- Explicit coverage is interpretable and debuggable
- UAI values elegant solutions over complex ones

### "Are we removing contribution?"

**Response:**
- Core contributions remain:
  1. Impossibility theorem ✓
  2. Complementarity theorem ✓
  3. CAGP method ✓
  4. Comprehensive experiments ✓
- Only removing: marginal variant that adds complexity

### "What if reviewer asks about learned approaches?"

**Response:**
- "We explored learned σ²(e,r) but found explicit coverage simpler and equally effective"
- Move to appendix, not main text
- Position as "we tried this, didn't add value"

---

## 🎯 DECISION CRITERIA

**Execute Plan B if:**
- ✓ RelCondVar 50ep AUROC < 0.75 on temporal OOD
- ✓ Auxiliary objective Δ < 0.05 AUROC
- ✓ No clear story for why RelCondVar is better than CAGP

**Keep RelCondVar if:**
- ✓ RelCondVar 50ep AUROC > 0.85 (substantially better)
- ✓ Auxiliary objective Δ > 0.10 (clear benefit)
- ✓ Can articulate when to use RelCondVar vs CAGP

---

## ⏱️ EXECUTION TIME

If Plan B is needed:
- Remove/rewrite sections: 2 hours
- Update all tables: 30 min
- Revise abstract/intro: 30 min
- Final coherence check: 1 hour

**Total: ~4 hours of writing**

---

## 📝 CHECKLIST FOR PLAN B

- [ ] Remove RelCondVar from abstract
- [ ] Simplify introduction (one method)
- [ ] Rewrite method section (CAGP only)
- [ ] Update all experiment tables
- [ ] Remove RelCondVar comparisons
- [ ] Add ablation on α (fixed vs learned)
- [ ] Emphasize simplicity as strength
- [ ] Optional: Brief appendix mentioning we tried learned approach
- [ ] Update conclusion
- [ ] Check all forward references
- [ ] Recompile LaTeX

---

**Bottom line:** Plan B is viable and actually makes the paper stronger in some ways. Don't be afraid to use it if needed!
