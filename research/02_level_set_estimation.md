# Level Set Estimation (LSE)

## 개요

Level Set Estimation은 어떤 함수 `f(x)`가 임계값 `h`를 넘는 영역 `{x : f(x) ≥ h}`를 최소한의 질의로 찾는 문제. 이진 blob 재구성은 `h = 0.5`인 LSE와 동일 구조.

---

## Straddle Heuristic (Gotovos et al., 2013)

### 핵심 논문
**Gotovos et al. (2013). *Active Learning for Level Set Estimation.* IJCAI 2013.**

### 아이디어

GP posterior를 사용해, 각 후보 지점 `x`에 대해:

```
straddle(x) = 1.96 · σ(x) - |μ(x) - h|
```

- `μ(x)`: GP posterior mean (예측값)
- `σ(x)`: GP posterior std (불확실성)
- `h`: 임계값 (blob에서는 0.5)
- `1.96`: 95% 신뢰구간 계수 (원논문 Bryan et al., 2005 기준)

**직관**: 불확실하면서(σ 높음) 임계값 근처(|μ-h| 낮음)인 지점을 우선 탐색.  
CI `[μ - 1.96σ, μ + 1.96σ]`가 임계값 h를 실제로 걸치는(straddle) 지점에 최고 점수.

### 기하학적 의미

straddle은 blob 경계 근처를 자동으로 우선 탐색. 내부(확실히 1)와 외부(확실히 0)는 점수가 낮아 자연스럽게 생략.

---

## 레거시 `lse` 구현의 문제

기존 codebase의 `lse`는 dense GP + straddle을 단일 레이어에 적용:
- **158초, 700 MiB** → 실용성 없음
- Dense GP의 `O(n³)` 복잡도가 50×200 = 10,000 픽셀에서 폭발

---

## Gotovos et al. (2013) LSE: Straddle의 개선

straddle에 ε-정확도 파라미터와 조기 분류 추가:
1. 각 미분류 픽셀의 CI 계산: `[μ - β·σ, μ + β·σ]`
2. **조기 분류**: CI 하한 > h → 양성 확정; CI 상한 < h → 음성 확정
3. 분류 확정된 픽셀을 후보 집합 `Q_t`에서 제거
4. `Q_t`가 비면 종료 → **이론적 보장이 있는 종료 기준** (straddle 원논문은 종료 기준 없음)

8-레이어 확장: **max-over-layers** 권고:
```python
score(row, col) = max_k ambiguity_k(row, col)
# 각 레이어별 Q_t 관리, 모든 Q_t^k가 빌 때 종료
```
이 방식이 "모든 레이어 ≥ 98%"라는 all-pass 요구사항에 가장 엄밀히 대응.

---

## 8-레이어 공유 좌표로의 확장

### 방법 1: Naive 합산 (권고 출발점)

```python
score(row, col) = sum_k straddle_k(row, col)
               = sum_k [w · σ_k(row, col) - |μ_k(row, col) - 0.5|]
```

각 레이어에서 GP를 독립적으로 유지하고, 점수만 합산.

**장점**: 구현 단순, 독립 레이어 가정에 정확히 부합  
**단점**: GP 8개를 병렬 업데이트해야 함, 여전히 메모리 부담

### 방법 2: Straddle without GP (더 가벼운 대안)

GP 없이 단순 관측 기반 경계 추정:
- 현재까지 관측된 1/0 픽셀에서 경계 근처 좌표 추정
- 관측되지 않은 픽셀 중 관측된 경계 근처를 우선 탐색

이 방식은 GP overhead 없이 straddle의 핵심 직관을 구현.

---

## GP 없는 Straddle 대안 구현 스케치

```python
def score(row, col, observations):
    # 각 레이어별: 이미 관측된 인접 픽셀에서 불확실성 추정
    total = 0
    for k in range(8):
        neighbors = get_neighbors(row, col)
        observed_neighbors = [observations[k][r,c] for r,c in neighbors if observed[r,c]]
        if not observed_neighbors:
            total += 1.0  # 미관측 = 최대 불확실성
        else:
            frac_ones = mean(observed_neighbors)
            uncertainty = 1.0 - abs(frac_ones - 0.5) * 2  # 0.5 근처 = 불확실
            total += uncertainty
    return total
```

---

## Sparse GP: 복잡도 절감

Dense GP 대신 **Sparse GP (inducing points)** 사용 시:
- 복잡도: `O(m²n)` where `m << n` (inducing point 수)
- 레거시 `sparse_gp`가 이 방향이었지만 "여전히 복잡도 대비 성능 이득 불충분"으로 평가됨

---

## 결론 및 권고

| 방법 | 복잡도 | 권고 |
|---|---|---|
| Dense GP + straddle (per layer) | O(n³) × 8 | 사용 금지 (레거시 실패 재현) |
| Sparse GP + straddle | O(m²n) × 8 | 복잡도 정당화 어려움 |
| 합산 straddle (GP 없음, 인접 기반) | O(8n) | **권고: 빠른 기준선** |
| 합산 straddle (Sparse GP) | O(m²n) × 8 | 기준선 통과 후 시도 |

**핵심 교훈**: straddle의 직관(경계 탐색)은 유효하나, GP는 이 문제에 과중. GP를 버리고 경계 추정 로직만 살리는 것이 실용적.

---

## 관련 논문

- Gotovos et al. (2013). *Active Learning for Level Set Estimation.* IJCAI 2013
- Bryan et al. (2005). *Active Learning for Level Set Estimation.* (선행 연구)
- Bichon et al. (2008). *Efficient Global Reliability Analysis for Nonlinear Implicit Performance Functions.* (공학적 적용)
