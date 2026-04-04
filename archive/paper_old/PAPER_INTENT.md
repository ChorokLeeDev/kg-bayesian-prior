# Paper Intent Document

## 논문의 핵심 의도 (Our Intent)

### 논문 구조
```
1. Empirical Discovery (78% confident-wrong)
   → 기존 방법들의 top-confident predictions 중 78%가 zero-evidence
   
2. Theoretical Explanation (Theorem 1)
   → "왜" 이런 일이 발생하나? Entity-level uncertainty는 relation-specific novelty 구분 불가
   → AUROC ≤ 0.5 on novel contexts (예측 + 검증됨)
   
3. Practical Implication
   → 해결책은 간단: coverage tracking (hash table)
   → 하지만 아무도 안 하고 있었음 (11-25% queries affected)
```

### 핵심 contribution 순서
1. **78% finding** = Empirical discovery (theorem과 별개의 독립적 기여)
2. **Theorem 1** = "왜" 실패하는지 설명 (AUROC ≤ 0.5 예측, 실제 0.34-0.50 관측)
3. **Coverage tracking** = 해결책 (당연해 보이지만 아무도 안 함)
4. **11-25% prevalence** = 이게 edge case가 아니라 substantial failure

---

## Reviewer 오해 vs 실제 의도

### 오해 1: "Theorem이 78%를 예측 못함"
- **Reviewer 해석**: Theorem의 claim이 78%인데 설명 못함
- **실제 의도**: 78%는 theorem의 claim이 아님. 별도의 empirical finding.
  - Theorem 1: AUROC ≤ 0.5 (novel context detection 성능)
  - 78%: "어디서" 가장 심하게 실패하나 (top-confident predictions)
  - 둘은 같은 현상의 다른 측면

### 오해 2: "Coverage tracking은 당연한 것"
- **Reviewer 해석**: 누구나 아는 trivial solution
- **실제 의도**: 당연한데 **아무도 안 하고 있었음**
  - UKGE, Energy, Deep Ensembles 등 모든 주요 방법이 이걸 안 함
  - 11-25%의 queries가 이 blind spot에 빠짐
  - "당연한 걸 왜 안 했나"를 밝히는 게 contribution

### 오해 3: "CAGP = U_str on ICEWS (semantic 기여 없음)"
- **Reviewer 해석**: Semantic component가 쓸모없음
- **실제 의도**: 이건 정확히 Theorem 3(iii)가 예측한 것
  - ρ ≈ 0 (coverage overlap 거의 없음) → semantic 불필요
  - ρ > 0 (static benchmarks) → semantic +8-11pp 기여
  - ICEWS에서 CAGP=U_str인 건 "bug가 아니라 feature"

### 오해 4: "Impossibility theorem은 과대포장"
- **Reviewer 해석**: Obvious한 걸 formal하게 포장
- **실제 의도**: Theorem의 가치는 universality
  - "어떤" combining function f(σ²_h, σ²_t)도 안 됨
  - 미래의 sophisticated learned combination도 안 됨
  - 이건 "coverage 안 보면 무조건 실패"의 formal guarantee

### 오해 5: "Definition 2 violated → theorem 무효"
- **Reviewer 해석**: 가정이 틀렸으니 theorem 적용 안 됨
- **실제 의도**: Definition 2는 **sufficient condition**
  - Real embeddings가 violate해도 limitation은 여전히 발생
  - 왜? Coverage info가 embedding에 있어도 **architecturally inaccessible**
  - Coverage-aware training 실패 (-4.3pp)가 이걸 증명

---

## 논문에서 명확히 해야 할 것들

### 1. 78%와 Theorem의 관계
**현재**: 둘 다 나열되어 있지만 연결 약함
**필요**: 명시적 연결
```
- Theorem 1: "왜" 실패하는가 (AUROC ≤ 0.5)
- 78% finding: "어디서" 가장 심하게 실패하는가 (top-confident)
- Anti-predictive (AUROC < 0.5): 왜 0.5보다 낮은가?
  → High-freq entities = low variance = high confidence
  → 근데 high-freq entities가 novel context에 많음
```

### 2. "당연한 solution"에 대한 방어
**현재**: Implicit
**필요**: Explicit statement
```
"Coverage tracking is obvious in hindsight, but:
(1) No major uncertainty method (UKGE, Energy, etc.) implements it
(2) This affects 11-25% of queries - not an edge case
(3) The contribution is exposing this gap, not inventing a complex solution"
```

### 3. ICEWS에서 CAGP=U_str 설명
**현재**: Theorem 3(iii) 언급
**필요**: 더 강조
```
"On ICEWS, ρ ≈ 0: nearly all emerging entities lack coverage.
Theorem 3(iii) predicts semantic adds nothing when ρ = 0.
CAGP correctly identifies coverage alone suffices here."
```

### 4. Sufficient condition 설명
**현재**: Definition 2에 "(Sufficient Condition)" 추가됨
**필요**: 왜 sufficient인데도 limitation 발생하는지 설명
```
"Real embeddings violate Definition 2 (CMI ≠ 0).
Yet novel-context AUROC remains 0.34-0.50. Why?
Coverage info exists but is architecturally inaccessible.
Coverage-aware training fails (-4.3pp): cannot learn to extract it."
```

---

## Commit History Context

| Commit | 의도 |
|--------|------|
| `9153faa` | "Impossibility" → "Limitation" reframe |
| `27d0a02` | Empirical discovery first, less defensive |
| `12120f5` | Theorem non-triviality, falsifiable conditions |
| `66cc38e` | Definition 2 as sufficient condition |
| `036ae06` | Frequency-controlled analysis for 78% |

---

## 이 문서의 용도

1. **Session 간 context 유지**: 새 session에서 논문 의도 파악
2. **Reviewer 오해 방지**: 논문 수정 시 체크리스트
3. **Agent review 비교**: Agent가 이해한 것 vs 실제 의도 gap 파악
