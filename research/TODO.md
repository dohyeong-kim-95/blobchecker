# Research TODO: 공유 좌표 다중 Blob 재구성 — 유사 문제 조사

## 배경

이 프로젝트의 핵심 과제: **하나의 좌표를 선택하면 8개 레이어 모두에 동시에 질의된다.**
레이어마다 독립적인 Blob이 있고, 각 레이어의 정보 필요도가 다르기 때문에
"어느 좌표가 전체적으로 가장 가치 있는가"를 결정하는 전략이 필요하다.

이와 유사한 구조의 문제를 다른 분야에서 어떻게 풀고 있는지 조사한다.

---

## 조사 항목

### 1. Active Learning / 순차적 실험 설계 (Sequential Experimental Design)
- **핵심 질문**: 여러 타깃(multi-output)이 있을 때 다음 질의 지점을 어떻게 고르는가?
- **관련 키워드**: batch active learning, multi-output active learning, query-by-committee
- **참고 분야**: 머신러닝, 통계
- **조사 포인트**:
  - joint uncertainty를 단일 acquisition function으로 집약하는 방법
  - 레이어 간 상관관계가 없을 때(독립) vs 있을 때 전략 차이

### 2. 멀티암드 밴딧 / 정보 이득 최대화 (Information-Theoretic Acquisition)
- **핵심 질문**: 여러 목적함수를 동시에 최적화할 때 탐색 지점을 어떻게 정하는가?
- **관련 키워드**: mutual information, entropy search, joint entropy, BALD (Bayesian Active Learning by Disagreement)
- **조사 포인트**:
  - 8개 레이어의 joint entropy를 실용적으로 근사하는 방법
  - entropy 합산 vs joint entropy의 차이와 trade-off

### 3. 압축 센싱 / 희소 복원 (Compressed Sensing)
- **핵심 질문**: 제한된 측정으로 고차원 신호를 복원할 때 어디를 측정하면 좋은가?
- **관련 키워드**: compressed sensing, sparse recovery, RIP (restricted isometry property), adaptive sampling
- **조사 포인트**:
  - 공간적으로 구조화된 신호(blob)에 CS가 적용 가능한지
  - 랜덤 샘플링 vs 적응형 샘플링의 이론적 보장

### 4. 베이지안 최적화 (Bayesian Optimization) — 다중 목적
- **핵심 질문**: 여러 블랙박스 함수를 동시에 최적화할 때 어떻게 샘플 효율을 높이는가?
- **관련 키워드**: multi-objective Bayesian optimization, PESMO, ParEGO, scalarization
- **조사 포인트**:
  - acquisition function을 8개 레이어에 대해 어떻게 집약하는지
  - 독립 GP 8개 vs 멀티태스크 GP 비교

### 5. 센서 배치 최적화 (Sensor Placement / Coverage)
- **핵심 질문**: 공간적 불확실성을 줄이기 위한 최적 측정 위치를 어떻게 정하는가?
- **관련 키워드**: optimal sensor placement, submodular maximization, greedy coverage
- **조사 포인트**:
  - 부분 모듈성(submodularity)을 이용한 greedy 근사의 성능 보장
  - 기하학적 커버리지 방법(scanline, grid sampling)과 비교

### 6. 지구물리 / 의료 영상 — 적응형 스캔 (Adaptive Scanning)
- **핵심 질문**: MRI, 지진파 탐사 등에서 적은 측정으로 구조를 복원하는 기법은?
- **관련 키워드**: adaptive MRI sampling, seismic survey design, k-space sampling
- **조사 포인트**:
  - 경계(boundary)를 우선 탐색하는 전략이 Blob 복원에 얼마나 전이되는지
  - 도메인 특화 기하 지식을 활용하는 방식

### 7. 레벨셋 추정 (Level Set Estimation)
- **핵심 질문**: 이진 함수의 경계(0/1 전환 지점)를 효율적으로 찾는 방법은?
- **관련 키워드**: LSE (level set estimation), straddle heuristic, LSEMTS
- **조사 포인트**:
  - 기존 `lse` 구현(레거시)의 straddle acquisition이 8-layer 공유 좌표에 어떻게 확장 가능한지
  - 8개 레이어 각각의 straddle score를 합산하는 naive 방법의 한계

---

## 우선순위 권고

| 순위 | 항목 | 이유 |
|---|---|---|
| 1 | Active Learning — multi-output | 문제 구조와 가장 직접적으로 대응 |
| 2 | 레벨셋 추정 | 레거시 코드에서 이미 사용, 확장성 검토 필요 |
| 3 | 센서 배치 (submodular) | greedy 근사가 단순하면서도 이론 보장 있음 |
| 4 | 압축 센싱 | 이론적 배경 탄탄하나 blob 구조에 직접 적용은 불명확 |
| 5 | 베이지안 최적화 (다중 목적) | 무거울 수 있음, 복잡도 정당화 필요 |

---

## 조사 후 정리할 내용

- [ ] 각 방법론의 "좌표 선택 비용"이 우리 문제의 iteration 예산과 부합하는지
- [ ] 8개 레이어 독립성을 가정할 때 joint score를 단순 합산해도 되는지 이론적 확인
- [ ] 기하학적 방법(scanline + BFS, 레거시 `geo`)과 확률적 방법의 hybrid 가능성
- [ ] 각 방법의 구현 복잡도 vs iteration 절감 효과 trade-off 정리
