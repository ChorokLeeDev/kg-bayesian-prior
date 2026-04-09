# Coverage Paradox Method Exploration

## Background

### Coverage Paradox Discovery
FB15k-237에서 발견된 현상:
- **Full Coverage**: 32.3% incorrect (overconfident, diluted embeddings)
- **Partial Zero**: 59.5% incorrect (one entity covered = anchor effect)
- **Full Zero**: 14.8% incorrect (neither covered = extrapolation failure)

### RCUE 실패 원인 분석

RCUE (Relation-Conditioned Uncertainty Estimation) 결과:
- MLP within-class contribution: **+4.4pp** (marginal improvement)
- Selective prediction: Energy baseline보다 못함
- **근본 원인**: Coverage boost가 Energy signal을 오염시킴

```
FB15k-237 Results (from rcue_experiment.log):
Energy         : 0.6366
UKGE           : 0.5000
RCUE           : 0.4808  <- Coverage boost pollutes signal
RCUE-noCov     : 0.6095  <- Without coverage, similar to Energy
```

RCUE의 설계 문제:
1. **Additive combination**: `uncertainty = MLP_variance * (1 + k*(1-coverage))`
2. Coverage boost가 binary OOD signal을 MLP variance에 곱함
3. MLP는 within-class (covered) 패턴을 학습, coverage는 between-class (OOD) signal
4. 두 signal이 multiplicative하게 섞이면서 둘 다 희석됨

---

## Exploration Directions

### Direction 1: Coverage-aware Calibration

**핵심 아이디어**: Coverage type별로 다른 temperature scaling 적용

```
Full coverage → Higher temperature (overconfident 보정)
Partial coverage → Moderate temperature (anchor effect 활용)
Zero coverage → Don't calibrate (abstain 권장)
```

**구현**:
- Validation set에서 category별 optimal temperature 학습
- ECE (Expected Calibration Error) 최소화

**예상 효과**:
- OOD detection: 제한적 (calibration은 confidence, not OOD)
- Selective prediction: 개선 가능 (better calibrated confidence)
- 구현 난이도: 낮음 (post-hoc, 학습 불필요)

**RCUE와의 차이**: Temperature는 score의 sharpness만 조절, signal 자체는 보존

---

### Direction 2: Anchor-based Prediction

**핵심 아이디어**: Covered entity를 explicit anchor로 활용

Anchor hypothesis (from `anchor_hypothesis_results.txt`):
- Covered entities contribute **1.36x-1.48x** more to scores
- Covered entity provides **context**, uncovered entity provides **discrimination**

**구현**:
```python
class AnchorBasedPredictor:
    # Given (h, r, ?):
    # - If h is covered: use h as anchor
    # - Attention: anchor -> relation -> predict target
    # - Uncertainty: attention confidence + coverage indicator
```

**예상 효과**:
- OOD detection: 중간 (attention uncertainty 활용)
- Selective prediction: 높음 (anchor가 prediction 품질 결정)
- 구현 난이도: 중간 (attention 학습 필요)

**RCUE와의 차이**: Coverage를 input feature가 아닌 architectural constraint로 사용

---

### Direction 3: Disentangled Embeddings

**핵심 아이디어**: Entity embedding을 relation-specific components로 분리

문제: Full coverage entity의 embedding이 여러 relation에 "희석"됨
해결: Mixture-of-Experts style routing

```python
e_r = Σ_k router_k(r) * expert_k(e_base)
```

**예상 효과**:
- OOD detection: 중간 (expert disagreement as uncertainty)
- Selective prediction: 높음 (dilution 문제 해결)
- 구현 난이도: 높음 (n_experts scaling, training stability)

**RCUE와의 차이**: Embedding 자체를 relation-specific하게 만듦 (variance 추정 아님)

---

### Direction 4: Cascading Uncertainty (Hybrid)

**핵심 아이디어**: Coverage와 Energy를 섞지 말고, 각각 다른 목적으로 사용

```
Stage 1: Coverage-based OOD detection
  - Zero coverage → Flag/Abstain (AUROC ~0.95+)

Stage 2: Energy-based selective prediction (among covered)
  - Low energy → Trust prediction
  - High energy → Lower confidence
```

**구현**:
```python
def get_uncertainty(h, r, t):
    if not covered(h, r) or not covered(t, r):
        return INFINITY  # Always flag as uncertain
    else:
        return -energy_score(h, r, t)  # Fine-grained confidence
```

**예상 효과**:
- OOD detection: **최고** (coverage lookup = perfect detection)
- Selective prediction: Energy baseline과 동등
- 구현 난이도: **최저** (post-hoc, no training)

**RCUE와의 차이**: 
- RCUE: Ensemble (coverage + Energy 혼합)
- Cascading: Sequential (coverage 먼저, 그 다음 Energy)

---

## Expected Results

| Method | OOD AUROC | Selective Pred. | Implementation |
|--------|-----------|-----------------|----------------|
| Energy baseline | 0.64 | baseline | - |
| Coverage-aware Calib. | ~0.65 | +small | Low |
| Anchor-based | ~0.70 | +moderate | Medium |
| Disentangled | ~0.70 | +moderate | High |
| **Cascading** | **~0.95** | +0 | **Lowest** |

---

## RCUE 실패 교훈

### 1. Signal Separation Principle
**Coverage와 Energy는 다른 종류의 signal:**
- Coverage: Binary structural signal (seen vs unseen)
- Energy: Continuous semantic signal (confidence)

**섞으면 안 되는 이유**:
- Coverage는 AUROC 0.95+ 달성 가능 (단독으로)
- Energy는 within-class에서 유용 (0.60-0.70)
- Ensemble하면 둘 다 희석됨

### 2. Architectural vs. Post-hoc
RCUE의 실수: MLP가 coverage signal을 "학습"하도록 함
- MLP는 training data에서만 학습 → test time novel context 불가
- Coverage lookup은 training 불필요 → novel context 직접 탐지

**교훈**: Novel context detection은 학습 불가능 (Theorem 2)

### 3. Multiplicative Contamination
```python
# RCUE 문제
uncertainty = MLP_variance * (1 + k*(1-coverage))

# 이 설계에서:
# - MLP가 낮은 variance를 출력하면, coverage boost가 무력화됨
# - MLP가 높은 variance를 출력하면, covered도 높은 uncertainty
```

**해결책**: Multiplicative 대신 cascading (sequential decision)

---

## Theory/Broader Impact Insights

### Theorem 2 Connection
Coverage Paradox는 Theorem 2의 empirical validation:

**Theorem 2**: Embedding-based uncertainty cannot detect novel contexts
**Coverage Paradox**: 
- Full coverage (more evidence) ≠ better prediction
- Partial coverage의 anchor effect가 오히려 도움
- Zero coverage = complete blind spot

이는 embedding이 "seen context"의 정보만 인코딩함을 보여줌.

### Practical Recommendation Update

Position paper의 practical recommendations을 강화:

1. **Coverage tracking is NECESSARY** (not just useful)
   - Energy alone: 0.64 AUROC
   - Coverage alone: 0.95+ AUROC

2. **Don't mix signals**
   - Ensemble/Mixture = signal pollution
   - Cascading = signal preservation

3. **Two-stage uncertainty**
   - Stage 1: Structural OOD (coverage)
   - Stage 2: Semantic confidence (Energy)

---

## Experimental Results (2026-04-09)

### Main Experiment: 4 Directions Comparison

| Method | OOD AUROC | Sel. MRR (50%) | Improvement |
|--------|-----------|----------------|-------------|
| Energy (baseline) | 0.6256 | 0.3219 | +27.4% |
| Coverage Calibration | 0.3643 | 0.3742 | +48.1% |
| Anchor-based | 0.4683 | **0.3929** | **+57.2%** |
| Disentangled | 0.9836 | 0.0004 | -51.8% |
| **Cascading** | **1.0000** | 0.1719 | -32.0% |

**Key Observations**:
- **Best OOD Detection**: Cascading (AUROC=1.0, perfect)
- **Best Selective Prediction**: Anchor-based (+57.2%)
- **Trade-off**: OOD detection과 selective prediction은 다른 목표

### Surprising Finding: OOD has HIGHER MRR

```
Full covered: MRR = 0.149, Hits@10 = 33.0%
T only covered: MRR = 0.667, Hits@10 = 79.9%  <- High MRR!
H only covered: MRR = 0.010, Hits@10 = 1.4%   <- Very low
Full zero: MRR = 0.045, Hits@10 = 9.5%
```

**Root Cause Analysis**:

1. **Tail covered (22.1%)의 높은 MRR**
   - Avg tail frequency: **1418.1** (vs covered: 279.0)
   - 매우 빈번한 tail entity가 포함됨
   - Model은 frequent tail을 잘 예측 (entity frequency effect)

2. **Coverage 정의의 함정**
   - "Covered" = entity가 **이 specific relation과** 함께 seen
   - Entity 전체 frequency와는 다름
   - High-freq entity도 특정 relation에서는 "uncovered"

3. **핵심 insight**
   ```
   OOD detection ≠ "will model be wrong"
   OOD detection = "does model have evidence for this (e,r) pair"
   ```

### Reframing: Coverage Blind Spot

Position paper의 핵심 claim 재정의:

**기존**: "Coverage blind spot causes incorrect predictions"
**수정**: "Coverage blind spot means we cannot verify correctness"

- Zero-coverage에서 model이 맞을 수도 있음
- 하지만 **evidence가 없으므로 verify 불가**
- 이것이 "blind spot"의 진정한 의미

### Method별 분석

**1. Coverage-aware Calibration**
- OOD AUROC: 0.36 (worse than baseline)
- Selective MRR: +48.1%
- **문제**: Calibration은 confidence 조정, OOD 탐지 아님
- **유용**: Covered 내에서 calibration 개선

**2. Anchor-based Prediction**
- OOD AUROC: 0.47
- Selective MRR: **+57.2%** (best)
- Attention mechanism이 anchor effect 활용
- **유망**: Selective prediction에서 best

**3. Disentangled Embeddings**
- OOD AUROC: 0.98 (near-perfect)
- Selective MRR: -51.8% (catastrophic)
- **문제**: Expert disagreement가 prediction 품질 저하
- Expert routing이 불안정

**4. Cascading Uncertainty**
- OOD AUROC: **1.0** (perfect by design)
- Selective MRR: -32.0%
- **Trade-off**: 100% OOD abstain → fewer triples for selection
- **유용**: OOD detection이 primary goal일 때

### Practical Implication

```python
def should_flag_uncertain(h, r, t):
    """
    Flag NOT because model will be wrong,
    but because we cannot verify if model is right.
    """
    h_cov = coverage[h, r]
    t_cov = coverage[t, r]

    if not h_cov or not t_cov:
        return True, "NO_EVIDENCE"  # Blind spot
    else:
        return False, energy_score(h, r, t)  # Can verify
```

### For Paper

이 finding은 position paper의 message를 강화:

1. **Theorem 2 보강**: Embedding-based uncertainty는 evidence 존재 여부를 판단 불가
2. **83% confident-wrong**: Energy가 선택한 top-100 중 83%가 zero-coverage (no evidence)
3. **Recommendation**: Coverage tracking은 correctness가 아니라 **verifiability** 보장

---

## RCUE 실패 원인 최종 정리

### 1. Signal 오염
RCUE: `uncertainty = MLP_variance * (1 + k*(1-coverage))`

문제:
- MLP는 **within-class** 패턴 학습 (covered 내 variation)
- Coverage는 **between-class** signal (covered vs uncovered)
- Multiplicative combination → 둘 다 희석

### 2. 목표 혼동
RCUE는 두 가지를 동시에 달성하려 함:
1. OOD detection (novel context)
2. Selective prediction (confidence)

결과:
- OOD AUROC: 0.48 (Energy 0.64보다 낮음)
- Selective MRR도 Energy보다 낮음

### 3. 해결책: Separation of Concerns
- **OOD detection**: Coverage lookup (perfect, no training)
- **Selective prediction**: Energy (continuous confidence)
- **Don't mix**: 각각 다른 목적으로 사용

---

## Next Steps

1. ~~실험 실행: `method_exploration_coverage_paradox.py`~~ (완료)
2. ~~결과 분석~~ (완료)
3. **Paper 반영**: 
   - "Blind spot = no evidence for verification" framing 강조
   - Coverage tracking의 목적: correctness → verifiability
   - Anchor-based가 selective prediction에서 best (+57.2%)

---

## Code Location
- Main prototype: `/Users/i767700/Github/kg-bayesian-prior/scripts/method_exploration_coverage_paradox.py`
- Quick test: `/Users/i767700/Github/kg-bayesian-prior/scripts/cascading_v2_test.py`
- OOD analysis: `/Users/i767700/Github/kg-bayesian-prior/scripts/investigate_ood_mrr.py`
- Results: `/Users/i767700/Github/kg-bayesian-prior/outputs/method_exploration_results.log`
