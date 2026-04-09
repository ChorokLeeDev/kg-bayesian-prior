# NeurIPS 2026 Position Paper Track: Submission Rationale

**Paper Title**: Stop Trusting Embedding-Based Uncertainty for Knowledge Graphs: Bayesian Methods Miss Structural Blind Spots

## Why This Submission is Suited for the Position Paper Track (250 words max)

This paper argues for a viewpoint about what practitioners *should do* when deploying knowledge graph embedding systems—specifically, that they should not rely solely on standard Bayesian uncertainty methods (MC Dropout, Deep Ensembles) for reliability assessment.

**Why position track, not main track:**
1. The contribution is *diagnostic*, not methodological. We identify why existing methods fail rather than proposing a novel algorithm. The "fix" (coverage tracking via hash table) is intentionally trivial—the value lies in understanding *why* sophisticated methods miss what simple bookkeeping catches.

2. The paper argues *against* current practice. Standard deployment patterns use MC Dropout or energy-based uncertainty without structural coverage. We provide evidence that this practice is actively harmful (worse than random), advocating for a change in community norms.

3. The empirical findings support a *perspective* about uncertainty decomposition (semantic vs. structural) rather than advancing a method. The coverage paradox finding (partial > full coverage accuracy) challenges assumptions but does not constitute a new technique.

**Not suited for main track because:**
- No novel model architecture or training procedure
- No formal theoretical contribution (proofs, bounds)
- The methodological prescription (track coverage) is engineering best practice, not research innovation

**Contemporary interest to NeurIPS community:**
- KGE systems are increasingly deployed in production (drug discovery, fraud detection)
- Uncertainty quantification is active NeurIPS area
- The failure mode affects standard benchmarks and multiple architectures
- Practical recommendations have immediate applicability
