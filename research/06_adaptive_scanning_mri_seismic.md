# 적응형 스캔 — MRI와 지구물리 탐사에서의 교훈

## 개요

MRI k-space 샘플링과 지진파 탐사 설계는 "제한된 측정으로 2D/3D 구조를 복원"하는 가장 성숙한 응용 분야. 이진 Blob 재구성 문제와 수학적 구조는 다르지만 알고리즘 통찰은 직접 전이된다.

---

## MRI k-Space 적응형 샘플링

### Variable-Density Random Sampling (Lustig et al., 2007)

**핵심 아이디어**: k-space 중심(저주파) → 밀집 샘플링, 주변부(고주파) → 희소 샘플링.

- 대부분의 신호 에너지가 저주파에 집중
- 고주파(경계 정보)는 진단에 중요하므로 완전히 생략 불가
- 랜덤 언더샘플링의 aliasing 아티팩트는 비간섭적 → TV/wavelet 최소화로 억제

**Blob 재구성으로의 전이**:
- MRI 저주파 = Blob 내부 (공간 자기상관 높음, 넓은 영역 확인)
- MRI 고주파 = Blob 경계 (공간 "고주파", 정밀 탐색 필요)
- → 내부 샘플은 넓은 영역을 확인, 경계 샘플은 정밀도에 기여

### CAIPIRINHA: 다중 슬라이스 동시 획득 (2002-2014)

**아이디어**: 여러 슬라이스를 동시에 획득하되, aliasing 패턴이 슬라이스 간에 비간섭적이 되도록 획득 궤적을 설계.

**이 프로젝트와의 직접 유사성**:
- 8개 레이어 = 8개 동시 획득 슬라이스
- 공유 좌표 = 하나의 공간 주파수를 8개 슬라이스 모두에 동시 적용
- CAIPIRINHA의 교훈: 공유 비용을 최대화하려면 **모든 레이어에서 동시에 불확실한 좌표**를 선택

### 경계 vs 내부: MRI 문헌의 결론

- 진단 품질(SSIM, edge-sharpness)은 경계 fidelity에 의해 지배됨
- Bounding-box 복원 정확도 ← 경계 샘플링 품질에 직결
- → 이 프로젝트의 blob 높이/너비 복원 요구사항과 직접 대응

---

## 지진파 탐사 설계

### 전통적 사전계획 방식

- 균등 격자 배치 (2D 탄성파)
- Nyquist 이론 기반 공간 샘플링 보장
- 장점: 단순, 이론 보장
- 단점: 흥미로운 지질구조 위치와 무관하게 균등 투자

### 적응형/순차적 탐사 (Moldoveanu et al., 2008)

- 초기 탐사 후 고불확실성 지역 식별 → 추가 탐사 집중
- 4D (time-lapse) 탐사: 변화 예상 지역만 재탐사
- 결과: 동등 품질에 shot 수 30-50% 절감

**Blob 재구성으로의 전이**:
- Phase 1 coarse scan → 고불확실성 영역 (경계) 식별
- Phase 2 adaptive → 경계 집중 탐색
- 순차 적응형 설계가 Pre-planned 균등 설계 대비 30-50% iteration 절감 가능성

### D-Optimal 탐사 설계 (Haber et al., 2010)

Fisher information matrix 행렬식 최대화:
```
max det(Fisher information matrix)
```
= 파라미터 불확실성 타원체 부피 최소화

**Blob 재구성 연결**:
- Fisher information 행렬 행렬식 최대화 ≈ 그리디 entropy 최대화 (근사)
- 8개 독립 레이어 합산 entropy가 D-optimal 설계의 실용적 근사

### Jittered Sampling (Herrmann et al., 2012)

**Jittered sampling**: 규칙 격자에서 랜덤 오프셋 → 순수 랜덤보다 나은 공간 커버리지 + CS 이론 보장 유지.

Blob 재구성 권고:
- Discovery phase에서 순수 uniform random 대신 jittered stratified 사용
- 50×200 격자를 10×10 셀로 분할 → 각 셀에서 1개 랜덤 샘플 → 200개 샘플로 전체 커버리지 보장

---

## 두 분야의 공통 결론

MRI와 지진파 탐사 문헌이 독립적으로 같은 결론에 도달:

> **경계 정보가 가장 고가치 획득 대상. 단, 내부 검증은 필수.**

구체적으로:
1. Coarse-to-fine 2단계 구조가 균등 단일 패스보다 40-70% 적은 측정으로 동등 품질 달성
2. 경계에 샘플 집중 → bounding-box 복원 정확도 극대화
3. 내부 구멍 탐지를 위한 별도 내부 스캔 필수 (경계만 보면 구멍 놓침)
4. 적응형이 사전계획보다 명확히 우수 (단, 구현 복잡도 증가)

---

## 이 프로젝트에 전이되는 것

| MRI/Seismic 기법 | Blob 재구성 대응 | 우선순위 |
|---|---|---|
| Variable-density sampling | 경계 집중 + 내부 희소 샘플링 | 높음 |
| CAIPIRINHA multi-slice | 8개 레이어 공유 좌표에서 joint entropy 최대화 | 높음 |
| Sequential adaptive design | Coarse-to-fine 2단계 알고리즘 구조 | 높음 |
| Jittered stratified sampling | Discovery phase coarse scan | 중간 |
| D-optimal design | 합산 entropy 최대화 (근사로 충분) | 중간 |
| Convex CS solver (ADMM) | — | 해당 없음 (시간/메모리 초과) |

---

## 관련 논문

- Lustig et al. (2007). *Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging.* MRM
- Breuer et al. (2006). *Controlled Aliasing in Parallel Imaging (CAIPIRINHA).* MRM
- Moldoveanu et al. (2008). *Randomized Marine Acquisition with Compressive Sampling.* SEG
- Haber et al. (2010). *Adaptive and Sequential Survey Design.* Geophysics
- Herrmann et al. (2012). *Randomized Sampling and Sparsity.* Geophysics
