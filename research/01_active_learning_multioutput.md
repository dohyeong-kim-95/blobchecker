# Active Learning — Multi-Output / Sequential Experimental Design

## 핵심 질문

레이어별로 다른 좌표를 선택할 수 없고, 하나의 좌표가 8개 레이어 모두에 동시에 적용될 때 — "다음 질의 좌표"를 어떻게 고르는가?

---

## 기본 구조: Acquisition Function

Active learning의 핵심은 **acquisition function** — 각 후보 좌표 `(row, col)`에 점수를 매겨 가장 유익한 좌표를 선택하는 함수.

단일 출력의 경우 흔히 쓰이는 기준:
- **Uncertainty sampling**: 예측 확률이 0.5에 가장 가까운 지점 선택
- **Entropy sampling**: `H(y) = -p log p - (1-p) log(1-p)` 최대화
- **BALD (Bayesian Active Learning by Disagreement)**: 모델 파라미터에 대한 mutual information 최대화

---

## Multi-Output으로의 확장

### 8개 레이어가 독립적일 때: 합산 근사

레이어들이 독립적으로 생성되므로 joint entropy가 분리됨:

```
H(y_1, ..., y_8 | x) = sum_k H(y_k | x)    [독립 가정]
```

실용적 귀결: **각 레이어의 acquisition score를 단순 합산해도 됨.**

```python
score(row, col) = sum_k acquisition_k(row, col)
```

단순하고 계산 비용이 낮으며, 독립성 가정이 맞으면 이론적으로도 정당화됨.

### 독립 합산의 한계

- 레이어 간 공간적 상관관계가 있다면 (예: 두 레이어의 blob이 겹치는 경향) 합산이 비효율적
- 이 프로젝트에서는 8개 레이어가 독립 생성이므로 합산이 합리적인 출발점

---

## BALD (Bayesian Active Learning by Disagreement)

**Houlsby et al., 2011** — 가장 영향력 있는 multi-output active learning 논문 중 하나.

### 아이디어

단순히 현재 불확실한 지점이 아니라, **모델이 배울 수 있는** 지점을 선택:

```
BALD(x) = H(y | x, D) - E_θ[H(y | x, θ)]
         = I(y; θ | x, D)
```

- 첫 번째 항: 현재 예측의 entropy (불확실성)
- 두 번째 항: 모델 파라미터를 알았을 때의 잔여 entropy (노이즈)
- 차이: 모델 파라미터에 대한 mutual information → 실제로 학습 가능한 정보

### Multi-output BALD

레이어가 독립적이면:
```
BALD_joint(x) = sum_k BALD_k(x)
```

GP 기반 구현에서는 각 레이어별 GP posterior variance를 사용해 근사 가능.

---

## 실용적 권고: Uncertainty Sampling이 출발점으로 충분

BALD는 이론적으로 우수하지만, **이 프로젝트의 이진 blob 구조에서는 uncertainty sampling(경계 근처 탐색)이 BALD와 유사한 결과를 낼 가능성이 높음.**

이유:
- 이진 blob에서 모델 불확실성이 가장 높은 곳 = blob 경계 근처
- BALD와 entropy sampling의 차이는 모델 파라미터 불확실성이 클 때 두드러지는데, 충분히 쿼리된 후에는 차이 감소

**권고 순서:**
1. Uniform random → baseline
2. Uncertainty/entropy sampling (합산) → 빠른 기준선
3. BALD (합산) → 성능이 충분하지 않을 때

---

## 관련 논문

- Houlsby et al. (2011). *Bayesian Active Learning for Classification and Preference Learning.* arXiv:1112.5745
- Gal et al. (2017). *Deep Bayesian Active Learning with Image Data.* ICML 2017
- Kirsch et al. (2019). *BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning.* NeurIPS 2019
- Settles (2010). *Active Learning Literature Survey.* UW Technical Report

---

## 이 프로젝트에 대한 함의

| 전략 | 구현 복잡도 | 예상 iteration 절감 | 권고 |
|---|---|---|---|
| Uniform random | 최저 | 없음 | Baseline |
| 합산 Entropy sampling | 낮음 | 중간 | 첫 번째 시도 |
| 합산 BALD (GP) | 중간 | 중간~높음 | 두 번째 시도 |
| Joint BALD (multi-output GP) | 높음 | 높음 (이론) | 복잡도 정당화 필요 |
