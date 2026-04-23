# Multi-Objective Bayesian Optimization

## 개요

MOBO는 여러 경쟁하는 블랙박스 목적함수의 Pareto frontier를 최소한의 평가로 탐색하는 방법. 표면적으로는 8개 레이어를 동시에 최적화하는 이 프로젝트와 유사해 보이지만, 실제로는 구조적 불일치가 크다.

---

## 구조적 불일치: 왜 MOBO가 과중한가

| MOBO 가정 | 이 프로젝트의 실제 구조 |
|---|---|
| 연속적 노이즈 있는 목적함수 | 이진 결정론적 oracle |
| Pareto frontier 탐색 | 8개 레이어 모두 ≥ 98% (trade-off 없음) |
| 소규모 예산 (50~200 평가) | 1,500 iteration |
| 느린 평가 (분~시간) | 빠른 oracle (microseconds) |
| GP 적합 비용이 평가 비용에 비해 낮음 | GP 적합이 전체 예산 초과 |

**핵심**: 8개 레이어가 "경쟁하는" 목적함수가 아니라 **모두 동시에 통과**해야 함 → Pareto frontier 개념이 적용되지 않음.

---

## 주요 MOBO 방법론 검토

### PESMO (Hernández-Lobato et al., 2016)

Pareto frontier에 대한 불확실성을 최대화하는 acquisition function:
```
PESMO(x) = H(Π* | D) - E_{y_x}[H(Π* | D ∪ {(x,y_x)})]
```

**이 프로젝트에서의 문제**:
- GP 적합 비용: O(n³) — n=1,500에서 64 MiB 메모리 예산 초과
- 이진 출력에 GP regression 사용 → Laplace/EP 근사 필요 → 추가 복잡도
- Pareto sampling 비용: O(S × n³)

### ParEGO (Knowles, 2006)

매 iteration마다 랜덤 가중치 벡터로 스칼라화:
```
f_w(x) = max_k(w_k · f_k(x))  [Augmented Chebyshev]
```

**이 프로젝트에서의 가치**: 스칼라화 아이디어 자체는 유용. 실용적 추출:
```python
score(x) = α · max_k H(p̂_k(x)) + (1-α) · sum_k H(p̂_k(x))
```
- 첫 번째 항: 가장 불확실한 레이어 집중 (worst-layer focus)
- 두 번째 항: 전체 커버리지 유지
- GP 없이 O(8 × H × W)로 계산 가능

### EHVI (Expected Hypervolume Improvement)

Pareto 지배 hypervolume의 기대 증가량 최대화.

**이 프로젝트에서의 변환**:
- "모든 레이어 ≥ 0.98"이 되는 확률 최대화와 동치
- 레이어 독립성에 의해: `P(all pass | x) ≈ Π_k P(accuracy_k ≥ 0.98 | x)`
- 이 계산은 간단한 occupancy model로 가능 → EHVI machinery 불필요

---

## 공유 좌표 제약과 MOBO

모든 MOBO 방법은 **모든 목적함수를 동일 지점에서 평가**한다고 가정 → 이 프로젝트의 공유 좌표 제약과 정확히 일치.

그러나 이 우연한 일치는 잘못된 방향: MOBO는 비싼 평가를 아끼려 설계되었으나, 이 프로젝트의 oracle은 빠르고 비용은 iteration count이다. MOBO의 내부 GP machinery가 오히려 bottleneck이 됨.

---

## 실용적 추출: MOBO 없이도 쓸 수 있는 아이디어

### 1. Worst-layer focus (ParEGO의 Chebyshev 스칼라화에서)

```python
score(x) = max_k H(p̂_k(x))
```
가장 불확실한 레이어에 집중 → 모든 레이어 동시 통과 요구사항에 직접 대응.

### 2. All-pass 확률 추적

매 iteration 후 각 레이어별 예상 정확도를 추적하여, 아직 98%에 도달하지 못한 레이어를 식별하고 해당 레이어의 불확실한 픽셀에 예산 집중.

### 3. 독립 목적함수 decomposition (qNEHVI에서)

독립 목적함수에 대해 최적 batch acquisition은 per-layer 독립 최적 선택의 합으로 분해 → 합산 entropy가 joint acquisition의 정확한 구현.

---

## 계산 비용 비교

| 방법 | Per-iteration 비용 | 메모리 | 권고 |
|---|---|---|---|
| Random | O(1) | O(1) | Baseline |
| Sum-entropy (occupancy) | O(8×H×W) | O(8×H×W) | 권고 |
| Worst-layer scalarization | O(8×H×W) | O(8×H×W) | 권고 |
| Sparse GP + BALD | O(m²n) | O(mn) | 기준선 통과 후 |
| Dense GP (per layer) | O(n³) × 8 | O(n²) × 8 | 사용 금지 |
| PESMO/EHVI | O(S × n³) | > 64 MiB | 사용 금지 |

Phase 0 메모리 예산 64 MiB: n=1,500에서 dense GP 8개 = 1,500² × 8 × 8B ≈ 144 MiB → **예산 초과**.

---

## 관련 논문

- Knowles (2006). *ParEGO.* IEEE Trans. Evolutionary Computation
- Hernández-Lobato et al. (2016). *PESMO.* NeurIPS
- Daulton et al. (2020). *qNEHVI.* NeurIPS — BoTorch 구현 포함
- Krause & Guestrin (2007). *Nonmyopic Active Learning of Gaussian Processes.* ICML
