# Information-Theoretic Acquisition Functions

## 개요

정보 이론 기반 acquisition function은 "다음 질의가 우리의 불확실성을 얼마나 줄이는가"를 정량화. 단순 uncertainty sampling보다 이론적 근거가 강함.

---

## Entropy Sampling

### 단일 레이어

```
H(y | x) = -p(x) log p(x) - (1-p(x)) log(1-p(x))
```

- `p(x)`: 현재 모델의 해당 좌표가 1일 예측 확률
- `H`는 `p = 0.5`에서 최대 → 경계 근처 탐색과 동일 효과

### 8개 독립 레이어로 확장

레이어가 독립적이면 joint entropy 분리:

```
H(y_1, ..., y_8 | x) = sum_k H_k(y_k | x)
```

따라서:
```python
score(row, col) = sum_k entropy(p_k(row, col))
```

**이론적 보장**: 독립성 가정이 맞으면 joint entropy와 합산 entropy가 동일. 이 프로젝트에서는 8개 레이어가 독립 생성이므로 합산이 최적.

---

## Mutual Information (BALD)

### BALD 공식

```
I(y; θ | x, D) = H(y | x, D) - E_θ[H(y | x, θ)]
```

- `H(y | x, D)`: 데이터 D가 주어진 현재 예측의 entropy
- `E_θ[H(y | x, θ)]`: 모델 파라미터 θ를 알면 남는 entropy (aleatoric uncertainty)
- 차이 = epistemic uncertainty = 모델이 실제로 배울 수 있는 정보량

### 이진 blob에서 BALD vs Entropy

이진 결정론적 oracle에서:
- Aleatoric uncertainty ≈ 0 (oracle은 노이즈 없음)
- 따라서 `E_θ[H(y | x, θ)] ≈ 0`
- BALD ≈ Entropy sampling

**귀결**: 이 프로젝트의 결정론적 oracle 환경에서는 BALD와 entropy sampling의 차이가 거의 없음. Entropy sampling으로 충분.

---

## 합산 vs Joint Entropy: 실제 차이

독립 레이어 가정 하에서:

| 상황 | 합산 entropy | Joint entropy |
|---|---|---|
| 레이어 독립 | = Joint entropy | = 합산 entropy |
| 레이어 상관 있음 | 과대 추정 | 정확 |
| 계산 비용 | O(8) per point | O(2^8) per point (완전 joint) |

이 프로젝트: 레이어 독립 생성 → **합산 entropy가 joint entropy와 동일하며 계산 비용도 낮음.**

---

## Expected Information Gain (EIG)

더 정교한 기준:

```
EIG(x) = H(θ | D) - E_y[H(θ | D ∪ {(x,y)})]
```

실제로 이 질의를 했을 때 모델 파라미터 불확실성이 얼마나 줄어드는지 예상.

**문제**: 이 계산은 일반적으로 intractable. 근사법(Monte Carlo 등) 필요.
이진 blob의 단순 구조에서는 과중한 방법.

---

## 실용적 구현 방향

### 방법 1: 관측 기반 확률 추정

GP 없이 간단한 확률 추정:
```python
# 각 미관측 픽셀에 대해 인접 관측값 기반 확률 추정
def estimate_prob(row, col, layer, observations):
    neighbors = get_8connected_neighbors(row, col)
    observed = [(r,c) for r,c in neighbors if is_observed(r,c)]
    if not observed:
        return 0.5  # 미관측 인접 없음 = 최대 불확실
    return mean(observations[layer][r,c] for r,c in observed)

def entropy_score(row, col, observations):
    return sum(binary_entropy(estimate_prob(row, col, k, observations)) for k in range(8))
```

### 방법 2: Boundary Distance 기반

경계 근처 픽셀 = entropy 높음을 이용:
```python
# 0과 1이 혼재하는 인접 픽셀을 가진 좌표 우선
def boundary_score(row, col, layer, observations):
    neighbors = get_observed_neighbors(row, col, layer)
    if not neighbors:
        return 1.0
    vals = [observations[layer][r,c] for r,c in neighbors]
    return 1.0 if (0 in vals and 1 in vals) else 0.0
```

---

## 관련 논문

- Houlsby et al. (2011). *Bayesian Active Learning for Classification and Preference Learning.* arXiv:1112.5745
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML 2016
- Kirsch et al. (2019). *BatchBALD.* NeurIPS 2019
- MacKay (1992). *Information-Based Objective Functions for Active Data Selection.* Neural Computation

---

## 이 프로젝트 결론

1. **Entropy 합산** = Joint entropy (독립 레이어 가정 하에) → 이론적으로 최적이면서 구현 단순
2. BALD = Entropy sampling (결정론적 oracle) → BALD 구현 불필요
3. GP 불필요 → 인접 관측값 기반 확률 추정으로 충분한 근사 가능
