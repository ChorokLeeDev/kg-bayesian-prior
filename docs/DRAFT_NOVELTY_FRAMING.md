# Draft: Novelty Framing Enhancement

**Location**: Introduction, in the "Contribution" paragraph (after line ~22 in introduction_uai.tex)
**Purpose**: Emphasize that learned methods fail to discover coverage despite having access to same data
**Length**: 1-2 sentences

---

## Current Text (lines 19-24 in introduction_uai.tex):

```latex
\paragraph{Contribution.}
We decompose KG uncertainty into \textbf{semantic} (embedding variance)
and \textbf{structural} (entity-relation coverage) components.
While coverage trivially detects novel contexts by construction, the
non-obvious insight is that \emph{learned probabilistic embeddings fail
to discover this signal}---despite having access to the same training data.
Our contribution is not coverage itself---which provides approximately 83\%
of performance gains on temporal shift---but rather: (1) the \emph{formalization}
of why coverage is necessary through Theorem~\ref{thm:complementarity}, which
proves semantic and structural uncertainties are non-redundant; (2) demonstrating
that learned embeddings cannot recover this signal; and (3) identifying
\emph{when} to combine coverage with variance (emerging entities vs novel contexts).
```

**Issue**: This is already quite good! But we can strengthen the "learned methods fail" point.

---

## OPTION A: Minimal Addition (Add 1 sentence)

**Insert after "despite having access to the same training data":**

```latex
\paragraph{Contribution.}
We decompose KG uncertainty into \textbf{semantic} (embedding variance)
and \textbf{structural} (entity-relation coverage) components.
While coverage trivially detects novel contexts by construction, the
non-obvious insight is that \emph{learned probabilistic embeddings fail
to discover this signal}---despite having access to the same training data.
This failure persists across diverse uncertainty methods (UKGE, GP-KGE,
BEUrRE, SNGP), none of which learn relation-specific uncertainty patterns
autonomously.
Our contribution is not coverage itself---which provides approximately 83\%
of performance gains on temporal shift---but rather: (1) the \emph{formalization}
of why coverage is necessary through Theorem~\ref{thm:complementarity}...
```

**Change**: +1 sentence (25 words)
**Impact**: Concrete examples strengthen the claim

---

## OPTION B: Replace existing sentence (No length change)

**Replace "learned probabilistic embeddings fail to discover this signal" with:**

```latex
While coverage trivially detects novel contexts by construction, the
non-obvious insight is that \emph{existing uncertainty methods (UKGE, GP-KGE,
BEUrRE, SNGP) fail to discover relation-specific patterns autonomously}---despite
having access to the same training data and being designed for uncertainty estimation.
```

**Change**: More specific claim with examples
**Impact**: Stronger, more concrete

---

## OPTION C: Strengthen with empirical evidence

**Insert after the contribution list (line ~24):**

```latex
Empirically, our method achieves 0.87--0.97 AUROC on temporal OOD across
four benchmarks, vs 0.52--0.58 for baselines (including baselines augmented
with coverage, Table~\ref{tab:baseline_coverage}).
```

**Change**: +1 sentence connecting theory to results
**Impact**: Immediate empirical validation

---

## OPTION D: Minimal Edit - Single Word Addition (RECOMMENDED)

**Current:**
```latex
...learned probabilistic embeddings fail to discover this signal...
```

**Enhanced:**
```latex
...learned probabilistic embeddings systematically fail to discover this signal...
```

**Change**: +1 word ("systematically")
**Impact**: Emphasizes this is not a coincidence but a structural limitation

---

## Alternative: Enhance Related Work Instead

If introduction is too crowded, strengthen the Related Work section:

**Current Related Work (related_work_uai.tex, line 1-5):**
```latex
\paragraph{Probabilistic KG Embeddings.}
UKGE~\citep{chen2019embedding} associates confidence scores with triples;
BEUrRE~\citep{chen2021probabilistic} uses box embeddings where volume
indicates uncertainty; GP-KGE~\citep{Chen2021PERM} learns Gaussian
distributions over entities.
All share the limitation we identify: learned variances are relation-agnostic.
```

**Enhanced:**
```latex
\paragraph{Probabilistic KG Embeddings.}
UKGE~\citep{chen2019embedding} associates confidence scores with triples;
BEUrRE~\citep{chen2021probabilistic} uses box embeddings where volume
indicates uncertainty; GP-KGE~\citep{Chen2021PERM} learns Gaussian
distributions over entities.
All share the limitation we identify: learned variances are relation-agnostic.
Despite being trained on data containing entity-relation co-occurrence patterns,
these methods do not autonomously learn relation-specific uncertainty---motivating
our explicit decomposition.
```

**Change**: +1 sentence in Related Work
**Impact**: Positions contribution clearly without crowding introduction

---

## Recommended Approach

**BEST: Combination of Option D (intro) + Related Work enhancement**

### In Introduction (line ~20):
```latex
...the non-obvious insight is that \emph{learned probabilistic embeddings
systematically fail to discover this signal}---despite having access to
the same training data.
```

### In Related Work (line ~6):
```latex
All share the limitation we identify: learned variances are relation-agnostic.
Despite being trained on data containing entity-relation co-occurrence patterns,
these methods do not autonomously learn relation-specific uncertainty,
necessitating our explicit decomposition framework.
```

**Total change**: 2 words (intro) + 1 sentence (related work)
**Impact**: ⭐⭐⭐⭐ Strengthens novelty framing without cluttering

---

## LaTeX Code (Ready to Copy-Paste)

### For Introduction (introduction_uai.tex, line ~20):

```latex
While coverage trivially detects novel contexts by construction, the
non-obvious insight is that \emph{learned probabilistic embeddings
systematically fail to discover this signal}---despite having access
to the same training data.
```

### For Related Work (related_work_uai.tex, line ~6):

```latex
\paragraph{Probabilistic KG Embeddings.}
UKGE~\citep{chen2019embedding} associates confidence scores with triples;
BEUrRE~\citep{chen2021probabilistic} uses box embeddings where volume
indicates uncertainty; GP-KGE~\citep{Chen2021PERM} learns Gaussian
distributions over entities.
All share the limitation we identify: learned variances are relation-agnostic.
Despite being trained on data containing entity-relation co-occurrence patterns,
these methods do not autonomously learn relation-specific uncertainty,
necessitating our explicit decomposition framework.
```

---

## Integration Instructions

**Step 1**: Add "systematically" to introduction (1 word change)

**Step 2**: Add 1 sentence to Related Work section (after "learned variances are relation-agnostic")

**Step 3**: Compile and verify it fits within page limit

**Step 4**: Optional - if still feels weak, add Option A's sentence to intro

---

## Reviewer Impact

**Before**: "They just added coverage, which is obvious"
**After**: "OK, interesting that learned methods don't discover this despite having the data. The formalization makes sense."

**Key message**: It's not about *inventing* coverage (obvious), it's about *why learned methods fail* (non-obvious) and *formalizing when to use it* (contribution).

---

**Status**: Ready to integrate
**Recommendation**: Use minimal edits (Option D + Related Work) for maximum impact with minimal space
