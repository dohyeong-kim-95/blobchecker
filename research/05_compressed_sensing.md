# Compressed Sensing — 구조적 이진 신호에 대한 적용

## 개요

Compressed Sensing(CS)은 신호가 어떤 기저에서 희소(sparse)할 때, Nyquist 기준보다 훨씬 적은 측정으로 완전 복원이 가능함을 보장. 이진 Blob 맵은 고전적 CS의 적용 대상이 아니지만, CS 이론의 핵심 통찰은 이 문제에 직접 전이된다.

---

## 이진 Blob은 CS 대상인가?

### 픽셀 도메인에서는 희소하지 않음

Coverage 30~70%이면 픽셀 도메인에서 희소하지 않음 — 고전 CS 적용 불가.

### 그러나 다른 의미에서 압축 가능

**Total Variation (TV) 희소성**: Blob은 부드러운 경계를 갖는 이진 마스크 → 공간 그래디언트가 경계 픽셀에만 집중 → TV 노름이 작음. TV-최소화 CS 복원이 이 구조에 이론적 보장 제공.

**경계 복잡도 기반 분석**:
- 50×200 그리드에서 전형적 blob의 경계 길이 ≈ O(200) 픽셀
- CS 샘플 수 하한: `M ≥ C · |boundary| · log(N)` ≈ `200 · log(10000)` ≈ 1,840
- 1,500 iteration budget은 이 경계를 아슬아슬하게 넘음 → **이론적으로 복원 가능**

단, 이 분석은 랜덤 측정을 가정. 적응형 측정은 추가로 절감 가능.

---

## 비적응형 vs 적응형 CS 샘플링

### 비적응형 (랜덤 균등 샘플링)

- 이론 보장: RIP 만족 → TV-최소화로 완전 복원 가능
- 실용 알고리즘: 랜덤 샘플링 후 ADMM/FISTA로 TV 최소화 풀기
- **이 프로젝트에서의 문제**: 이진 제약이 있어 연속 복원 후 임계값 처리 필요 → 경계 오차 발생. 또한 1.0s / 64 MiB 예산 내에 ADMM 실행 불가.

### 적응형 CS (Castro & Nowak, 2012)

이론적 결과: 적응형 감지는 `M = O(s^{1/2} · log N)` 측정으로 충분 (비적응형의 `O(s · log N)` 대비).

이 프로젝트에서:
- s ≈ 200 (경계 픽셀 수)
- 비적응형 필요량: 200 · log(10000) ≈ 1,840
- 적응형 필요량: 200^{0.5} · log(10000) ≈ 260

→ **이론상 7배 절감**, 적응형이 압도적으로 유리.

---

## Blob 구조가 CS 분석에 미치는 영향

### 긍정적

1. **이진 + Piecewise-constant**: TV 최소화 보장이 가장 강한 신호 클래스
2. **8-연결성 제약**: 두 개의 분리된 양성 구성 요소를 가진 복원은 오답 → 기하 추론으로 검증 가능한 free prior
3. **국소 연속 경계**: 경계 픽셀이 공간적으로 군집 → 한 경계 픽셀 발견 시 인접 픽셀도 경계일 가능성 높음 → 적응형 정제 효율화

### CS가 도움이 안 되는 부분

- **Blob vs Outlier 구분**: CS는 단일 ground truth를 가정, Outlier 분리는 기하 추론 필요
- **내부 구멍**: TV 최소화는 구멍을 채우려는 경향 → 별도 내부 탐색 필수

---

## 실용적 결론: Convex Solver 대신 기하 통찰 활용

CS 이론에서 가져올 것:
- 랜덤 15% 샘플링이 이론적으로 충분함을 확인 → budget 검증
- 적응형 경계 정제가 CS 적응형 감지의 실용적 구현

CS 이론에서 버릴 것:
- ADMM/FISTA convex solver → 시간/메모리 예산 초과
- Full CS pipeline 구현 → 불필요한 복잡도

**핵심 전이**: 경계 근처에 샘플을 집중하는 2단계 coarse-to-fine 전략이 CS 적응형 감지의 기하 구현이며, 이것이 실용적이고 충분.

---

## 예산 배분 권고 (CS 분석 기반)

| Phase | 목적 | Iteration 수 | 방법 |
|---|---|---|---|
| Discovery | Blob 중심 위치 파악 | ~200 (13%) | Jittered 균등 격자 |
| Boundary | 경계 정밀 탐색 | ~900 (60%) | 행별 이진 탐색 |
| Interior | 구멍 탐지 | ~300 (20%) | 내부 격자 스캔 |
| Repair | 모순 수정 | ~100 (7%) | 8-연결성 위반 패치 |

---

## 관련 논문

- Candès & Tao (2006). *Near-Optimal Signal Recovery from Random Projections.* IEEE Trans. IT
- Donoho (2006). *Compressed Sensing.* IEEE Trans. IT
- Lustig et al. (2007). *Sparse MRI.* Magnetic Resonance in Medicine
- Castro & Nowak (2012). *Adaptive Sensing for Sparse Recovery.* IEEE Trans. SP
- Haupt et al. (2011). *Adaptive Compressed Sensing.* IEEE Trans. IT
