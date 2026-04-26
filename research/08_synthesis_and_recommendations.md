# 종합 분석 및 구현 권고

## 7개 분야의 공통 결론

Active Learning, LSE, 정보 이론, 센서 배치, CS, MRI, 지진파 탐사 — 7개 분야가 독립적으로 같은 결론에 도달:

> **경계 정보가 가장 고가치 획득 대상. Coarse-to-fine 적응형 전략이 균등 샘플링보다 40-70% iteration 절감.**

---

## 핵심 통찰 정리

### 1. 독립 레이어 → 합산 = Joint Entropy

8개 레이어가 독립적으로 생성되므로:
```
H(y_1,...,y_8 | x) = Σ_k H(y_k | x)   [수학적으로 정확, 근사 아님]
```

**귀결**: 각 레이어 entropy를 합산하는 것이 이론적으로 최적이면서 계산도 단순.

### 2. GP는 과중: 기하 추론으로 충분

| 방법 | 비용 | 결론 |
|---|---|---|
| Dense GP | O(n³) | 메모리 예산 초과, 레거시에서 이미 실패 확인 |
| Sparse GP | O(m²n) | 복잡도 대비 이득 불충분 |
| 기하 + Occupancy model | O(8×H×W) | 충분한 성능, 예산 내 |

### 3. 결정론적 oracle에서 BALD ≈ Entropy Sampling

Aleatoric uncertainty ≈ 0 (노이즈 없는 oracle) → BALD 두 번째 항 소멸 → BALD = Entropy sampling.
BALD 구현 불필요.

### 4. Pre-planned: Submodular Greedy가 63% 보장

Pre-planned 전략을 쓸 경우, greedy coverage maximization이 최적해의 (1-1/e) ≈ 63% 보장. Jittered stratified sampling이 실용적 근사.

### 5. 내부 구멍 탐지는 필수

경계만 탐색하는 방법은 내부 구멍(0-3개, ~7×7px)을 놓침. 구멍 3개 × 49px = 1.47% 오차 → 98% 기준 실패. 별도 내부 스캔 필수.

---

## 권고 알고리즘 구조

### Adaptive 전략 (리서치 기반 초안)

아래 구조는 리서치에서 도출한 출발점입니다. 현재 구현 실험에서는 15%
budget에서 boundary binary search의 이득이 작고 runtime/복잡도 비용이
있었으므로, binary search는 즉시 고정할 권고가 아니라 Phase A의 넓은
budget에서 다시 검증할 후보로 취급합니다.

```
Phase 1 — Discovery (200 iterations, 13%)
  목적: 8개 레이어 모두의 Blob 중심과 대략적 범위 파악
  방법: Jittered stratified sampling (50×200 격자를 10×10 셀로 분할, 셀당 1개)
  
Phase 2 — Boundary Localization (900 iterations, 60%)
  목적: 각 행의 왼쪽/오른쪽 경계를 픽셀 수준으로 정밀 탐색
  방법: 합산 entropy 최대화 + boundary refinement
  acquisition: score(r,c) = Σ_k H(p̂_k(r,c)), unobserved cell 중 argmax
  
Phase 3 — Interior Hole Detection (300 iterations, 20%)
  목적: Blob 내부의 구멍 탐지
  방법: 식별된 Blob 영역 내 3픽셀 간격 구조적 격자 스캔
  
Phase 4 — Targeted Repair (100 iterations, 7%)
  목적: 8-연결성 위반, 경계 모순 수정
  방법: 기하 검증 후 모순 픽셀 인근 재탐색
```

### Pre-planned 전략 (대안)

```
Greedy Coverage Maximization (Submodular)
  1. 초기 belief: p̂_k(r,c) = 0.4 (coverage prior)
  2. 1,500번 greedy 반복:
     score(r,c) = Σ_k H(p̂_k(r,c)) [미관측 셀만]
     best = argmax(score)
     selected.append(best)
  3. 실행 전 모든 좌표 확정 → 실행 시 단순 순회
  
  Lazy greedy로 10-100배 속도 최적화 가능.
```

---

## Acquisition Function 구현

### Occupancy Model (권고)

GP 없이 경량 belief 유지:

```python
# 초기화
p = np.full((8, H, W), 0.4)   # coverage prior
observed = np.zeros((H, W), dtype=bool)

def update(row, col, labels):
    """Oracle 결과로 belief 업데이트"""
    p[:, row, col] = labels      # 직접 관측 → 확실성 1
    observed[row, col] = True
    # 공간 전파: 인접 미관측 픽셀 belief 업데이트 (선택적)
    propagate_to_neighbors(row, col, labels)

def score_all():
    """모든 미관측 픽셀의 합산 entropy"""
    h = binary_entropy(p)        # shape (8, H, W)
    joint = h.sum(axis=0)        # shape (H, W)
    joint[observed] = 0          # 관측된 픽셀은 0
    return joint

def binary_entropy(p):
    eps = 1e-10
    p = np.clip(p, eps, 1-eps)
    return -p * np.log2(p) - (1-p) * np.log2(1-p)
```

### 공간 전파 (중요 efficiency lever)

공간 전파 없이: 각 query가 1픽셀 정보만 감소  
공간 전파 있이: 각 query가 인접 5-20픽셀에 정보 전파 → 실질 iteration 효율 증가

```python
def propagate_to_neighbors(row, col, labels, decay=0.7):
    """관측된 레이블을 인접 픽셀 belief에 전파"""
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r, c = row+dr, col+dc
        if 0 <= r < H and 0 <= c < W and not observed[r,c]:
            # 인접 픽셀 belief를 관측값 방향으로 당김
            p[:, r, c] = p[:, r, c] * (1-decay) + labels * decay
```

---

## Outlier 처리

Oracle이 반환하는 1이 Blob 픽셀인지 Outlier인지 직접 구분 불가. 기하 추론으로 분리:

```python
def classify_observed_positives(observations, predicted_blob):
    """
    관측된 양성 픽셀 중 8-연결 컴포넌트 분석으로 Outlier 식별
    - 연결 크기 < threshold → Outlier
    - 메인 Blob에서 공간적으로 고립 → Outlier
    """
    for k in range(8):
        components = label_connected_components(observations[k])
        main_component = largest_component(components)
        outlier_mask[k] = (observations[k] == 1) & (component_id != main_component_id)
```

---

## 예산 충분성 검증 (CS 이론)

Phase 0: 50×200 = 10,000 픽셀, 1,500 iteration budget.

CS 이론 하한: `M ≥ C · |boundary| · log(N)`
- |boundary| ≈ 200 (전형적 40% coverage blob의 경계 길이)
- log(10,000) ≈ 9.2
- 필요량 ≈ 200 × 9.2 ≈ 1,840

→ 1,500 iteration은 **이론 하한보다 약간 낮음** → 경계가 매우 매끄럽고 구멍이 없는 경우에는 충분. 구멍이 있거나 outlier가 많은 경우 타이트.

**귀결**: 예산 내에서 통과하려면 적응형 전략이 필수. 균등 랜덤 샘플링으로는 보장 어려움.

---

## 방법론 선택 트리

```
시작
 │
 ├─ Pre-planned 속도 중요? ──Yes──► Greedy coverage (submodular, jittered)
 │
 └─ Adaptive 선택
      │
      ├─ 기준선 먼저 ──► Geometry-first (scanline + BFS, geo 계승)
      │                   합산 entropy로 다음 좌표 선택
      │
      └─ 성능 부족 시 ──► Occupancy model + 공간 전파
                          + Worst-layer scalarization
                          + GP (Sparse, 기준선 통과 후만)
```

---

## 미결 질문 (리서치로 해소되지 않은 것)

1. **공간 전파 decay 파라미터**: blob 경계 smoothness에 따라 튜닝 필요
2. **Budget ladder별 Phase 분할 비율**: 50%/30%/20%/15% 예산에서 discovery,
   boundary, hole, repair 비율이 어떻게 달라져야 하는지 실험 필요
3. **Worst-layer scalarization의 α값**: 순수 합산 vs 순수 worst-layer 비율
4. **내부 구멍 탐지 격자 간격**: 7×7 구멍을 3픽셀 간격으로 탐지 가능한지 검증
5. **8-연결성 기반 outlier 분리 임계값**: 어느 크기까지 outlier로 분류할지
