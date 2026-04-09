# Motivation Letter Draft — UvA CANES PhD Position

**To:** Prof. Sara Magliacane  
**Position:** PhD Position on Learning Causally Grounded Concepts (Topic 2: Weak Supervision)  
**Applicant:** Chorok Lee

---

Dear Prof. Magliacane,

I am writing to apply for the PhD position in the CANES project, specifically for **Topic 2: Weak Supervision**. My research has consistently addressed a core question: *how can we quantify and diagnose model reliability when direct supervision is unavailable?* This aligns directly with your goal of learning causally grounded concepts from weak supervision signals.

## Research Background

My work spans three interconnected areas:

**1. Diagnosing Model Failures via Indirect Signals**

In "Diagnosing Conformal Prediction Failures Under Distribution Shift" (UAI 2026, under review), I show that SHAP feature concentration—computed solely on validation data—predicts conformal prediction failure severity under distribution shift (ρ=0.853, p<0.001 across 16 tasks). The key insight is that *structural properties of the learned model* serve as a diagnostic for downstream reliability, even without access to test labels. This is precisely the weak supervision paradigm: using indirect signals (feature importance structure) to infer guarantees about model behavior.

**2. Uncertainty Quantification Under Distribution Shift**

My work on volatility-adaptive conformal prediction for financial returns demonstrates that simple structural adjustments (volatility scaling) can restore coverage guarantees that standard methods fail to provide during high-volatility regimes. I proved explicit bounds showing how coverage degrades with heteroskedasticity and how scaling restores uniform conditional coverage.

**3. Causal Structure Discovery**

In "When Does Neural Granger Causality Work?" I systematically benchmark when neural methods outperform classical causal discovery approaches. The finding—that discontinuities (margin calls, circuit breakers) differentiate neural from linear methods—informs when structural assumptions in causal models hold.

## Fit with Topic 2: Weak Supervision

Topic 2 asks: *Can we learn causally grounded concepts with theoretical guarantees when we only have downstream task labels or logical constraints?*

My research directly prepares me for this:

- **From downstream signals to guarantees**: My conformal prediction work establishes that structural properties of learned representations can predict downstream behavior—the same inferential pattern required for learning concepts from weak supervision.

- **Formal guarantees under limited information**: I have experience deriving theoretical bounds (coverage guarantees, score inflation bounds) that hold despite not having direct access to the quantity of interest.

- **Bridging theory and empirics**: Each of my papers combines formal theorems with extensive empirical validation across multiple domains—the methodology CANES requires.

## Why This Position

The CANES project addresses a fundamental gap: current neuro-symbolic and XAI methods learn concepts without formal guarantees about their causal grounding. Your prior work on causal representation learning and the CANES framework's focus on integrating background knowledge as constraints resonates with my research trajectory.

The dual affiliation with Saarland University and the research visit component are particularly attractive, as they provide exposure to the broader causal ML community centered around the MPI Tübingen network.

## Practical Considerations

- **Start date**: October 2026 aligns with my timeline
- **Background**: MS in AI (KAIST), BS in Computer Science (Yonsei University)
- **Programming**: Python, PyTorch, statistical modeling, causal inference libraries

I would welcome the opportunity to discuss how my background in reliability diagnostics and uncertainty quantification could contribute to CANES's goals.

Sincerely,  
Chorok Lee

---

## Notes for Revision

- [ ] Add specific CANES paper citations (Magliacane's prior work)
- [ ] Mention specific collaboration ideas
- [ ] Adjust length (currently ~500 words, target ~600-800)
- [ ] Add reference to KG uncertainty work if relevant
- [ ] Check if neural causal discovery paper should be mentioned
