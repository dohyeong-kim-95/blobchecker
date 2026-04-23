# Sensor Placement & Submodular Maximization

## 개요

"어디에 센서를 놓을 것인가"는 "어느 좌표를 질의할 것인가"와 동일 구조. Submodular maximization은 이 문제에 이론적 보장을 제공.

---

## Submodularity 정의

집합 함수 `F: 2^V → R`이 submodular하면:

```
F(A ∪ {e}) - F(A) ≥ F(B ∪ {e}) - F(B)    for all A ⊆ B
```

직관: 집합이 클수록 새 원소의 한계 이득(marginal gain)이 감소 — **수확 체감 법칙**.

---

## Greedy 알고리즘의 보장

**Nemhauser et al. (1978)** — 가장 중요한 결과 중 하나:

> Submodular monotone 함수를 greedy로 최대화하면, 최적해의 `(1 - 1/e) ≈ 63%` 이상을 보장.

알고리즘:
```python
selected = []
for i in range(budget):
    best = argmax_{x not in selected} F(selected ∪ {x}) - F(selected)
    selected.append(best)
```

---

## 센서 배치에서의 적용: Krause et al.

**Krause et al. (2008). *Near-Optimal Sensor Placements in Gaussian Processes.* JMLR.**

GP mutual information을 objective로 사용:
```
F(A) = I(y_A; y_{V\A})  [관측 지점 A와 미관측 지점의 mutual information]
```

이 함수가 submodular임을 증명 → greedy가 63% 보장.

---

## 이 프로젝트에서의 적용

### Submodular Objective 후보

**옵션 1: Coverage-based (가장 단순)**
```
F(A) = |{pixels covered within distance d of some point in A}|
```
단순 coverage는 submodular → greedy pre-planned 전략으로 사용 가능.

**옵션 2: Information gain (GP 기반)**
```
F(A) = sum_k I_k(y_A; y_{V\A})   [레이어별 합산]
```
각 레이어의 GP MI는 submodular → 합산도 submodular.

**옵션 3: Uncertainty reduction**
```
F(A) = sum_k (H_k(initial) - H_k(after observing A))
```

### Pre-planned vs Adaptive

Submodularity는 특히 **pre-planned (offline)** 전략에 유용:
- 실행 전에 최적 좌표 set을 미리 계산
- Greedy로 1,500개 좌표 선택 → 최적의 63% 보장

Adaptive (online) 전략에서도 greedy를 매 iteration 적용 가능하지만, 계산 비용 증가.

---

## 실용적 구현: Greedy Coverage

가장 단순한 submodular 전략 — **균등 분포 + 가중 coverage:**

```python
def greedy_preplan(H, W, budget):
    grid = np.zeros((H, W))
    selected = []
    coverage = np.zeros((H, W))
    
    for _ in range(budget):
        # 아직 커버되지 않은 픽셀이 가장 많은 곳 선택
        scores = np.zeros((H, W))
        for r in range(H):
            for c in range(W):
                if (r, c) not in selected:
                    scores[r, c] = count_uncovered_neighbors(r, c, coverage)
        best = argmax(scores)
        selected.append(best)
        update_coverage(coverage, best)
    
    return selected
```

더 실용적으로는 **stratified sampling** (격자 기반 균등 샘플링)이 coverage submodularity의 근사.

---

## Lazy Greedy (속도 최적화)

Greedy의 병목: 매 iteration마다 모든 후보의 marginal gain 재계산.

**Lazy evaluation (Minoux 1978)**:
- Submodularity의 수확 체감 특성 활용
- 이전 iteration의 gain 상한으로 후보 정렬
- 상위 후보만 정확히 재계산

실제로 10~100배 속도 향상, 동일 결과 보장.

---

## 관련 논문

- Nemhauser et al. (1978). *An Analysis of Approximations for Maximizing Submodular Set Functions.* Mathematical Programming
- Krause & Guestrin (2005). *Near-Optimal Nonmyopic Value of Information in Graphical Models.* UAI 2005
- Krause et al. (2008). *Near-Optimal Sensor Placements in Gaussian Processes.* JMLR
- Minoux (1978). *Accelerated Greedy Algorithms for Maximizing Submodular Set Functions.* Optimization Techniques

---

## 이 프로젝트 결론

| 전략 | 보장 | 복잡도 | 권고 |
|---|---|---|---|
| Random | 없음 | O(1) | Baseline |
| Greedy coverage (pre-planned) | 63% of optimal | O(budget × H × W) | Pre-planned 기준선 |
| Greedy GP MI (pre-planned) | 63% of optimal | O(budget × n³) | 과중 |
| Adaptive greedy entropy (online) | 없음 (근사) | O(8 × H × W) per iter | 실용적 |

**핵심 통찰**: Pre-planned 전략에서는 greedy coverage가 빠르고 이론 보장 있음. Adaptive 전략에서는 매 iteration greedy entropy 합산이 실용적 출발점.
