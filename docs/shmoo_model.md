# Shmoo 데이터 생성기 정본

blob 데이터셋을 만드는 물리 모델의 정본 문서입니다. 결정은
[../README.md](../README.md), 평가 계약은 [spec.md](spec.md)에서 시작합니다.
원본 요구사항 프롬프트는 [../todo.md](../todo.md)에 보존되어 있고, 이
문서는 그것을 구현 가능한 형태로 정제한 것입니다.

## 핵심 발상

DRAM I/O 특성화의 **Shmoo Plot**에서 PASS 영역이 곧 blob입니다.

- 가로축(열, `W`): 샘플링 타이밍 오프셋, `[-0.5, +0.5] UI`
- 세로축(행, `H`): Vref 전압 레벨, `[200, 800] mV`
- 각 `(timing, vref)` 격자점은 여러 데이터 패턴으로 테스트되고, **모든
  패턴이 모든 Monte Carlo 반복에서 통과**할 때만 PASS입니다.

PASS 픽셀의 연결 영역이 eye 모양 blob을 형성합니다. 가우시안 스무딩
블롭과 달리 ISI/DCD/RJ가 만드는 물리적 경계라, 도메인에 맞는 모양이
나옵니다.

## 출력 계약 (현재)

| 항목 | 값 |
|---|---|
| 출력 shape | `(16, 150, 200)` = `(layers, H=Vref행, W=timing열)` |
| 레이어 수 | 16 (독립 생성) |
| 커버리지 목표 | 50% (커버리지 래더 C1, 아래 참조) |
| 커버리지 | 레이어별 20%~55% 균등 분포 |
| 구멍 | 없음 (eye 내부는 단조 PASS) |
| 이상치 | 없음 (eye 외부는 단조 FAIL) |
| `truth_outlier_mask` | 전부 0 (텐서 계약 유지용) |
| `truth_full_mask` | `== truth_blob_mask` |

구멍/이상치가 없으므로 Phase 1에서 oracle은 사실상 blob을 그대로
반환합니다. 텐서 3종 시그니처는 하위 호환을 위해 유지합니다.

## 신호 체인 (1 패턴당)

```text
3-bit 패턴 → 35-bit 시퀀스 → 이산 ISI FIR(인과 탭)
          → soft_saturate(tanh) → RC 에지 파형 재구성(DCD 포함)
          → (center + timing_offset + RJ)에서 샘플링
```

### 1) 비트 시퀀스와 패턴

- 8개 3-bit 패턴 전부: `000,001,010,011,100,101,110,111`
- **가운데 비트(index 1)** 가 샘플링 대상(target bit)
- 16 preamble(0) + 3-bit 패턴 + 16 postamble(0) = **35 bit**
- 35-bit 배열에서 target bit = **index 17**, 기준 시각 = `17.5 UI`

### 2) 이산 ISI (FIR 탭) — 정렬 주의

- 탭: `[1.0, -0.15, -0.08, -0.04]` (메인 커서 + post-cursor 3개)
- 인과 컨볼루션이어야 함:

```text
received[i] = bits[i]·1.0 + bits[i-1]·(-0.15) + bits[i-2]·(-0.08) + bits[i-3]·(-0.04)
```

- **함정**: `np.convolve(bits, taps, mode='same')`는 짝수 길이 커널에서
  메인 커서를 `bits[i+1]`에 놓아 1비트 밀린다. `mode='full'` 후 슬라이싱
  하거나 인과식으로 직접 구현하고, `received[17]`이 위 식과 맞는지 검증.

### 3) Soft saturation (하드 클리핑 금지)

```python
def soft_saturate(v, v_low=0.0, v_high=1.0, k=10.0):
    mid = (v_low + v_high) / 2
    span = (v_high - v_low) / 2
    return mid + span * np.tanh(k * (v - mid) / span)
```

- ISI 후 전압이 범위를 벗어날 수 있어(언더슈트/오버슈트) C∞ 포화 적용
- `k`↑ = 하드 클립에 근접(MOSFET rail), `k`↓ = 선형에 근접
- 파형 재구성 **전에** ISI 결과 전압 레벨에 적용

### 4) RC 에지 파형 재구성

- UI당 64 샘플
- 비트 경계마다 RC 지수 전이: 상승 `τ_rise=0.15`, 하강 `τ_fall=0.15 UI`
- `V(t) = V_cur + (V_next - V_cur)(1 - exp(-t_local / τ))`
- 동일 레벨이면 평탄

### 5) DCD (결정성 지터 — 듀티 왜곡)

- **에지 시작 위치**를 이동(목표 레벨이 아니라):
  - `0→1` 에지: 오른쪽 `+dcd_offset`
  - `1→0` 에지: 왼쪽 `-dcd_offset`
  - 전이 없으면 적용 안 함
- 기본 `dcd_offset = 0.01 UI`. 아이를 좌우 비대칭으로 만듦

### 6) RJ (랜덤 지터)와 Pass/Fail

- `RJ = σ_RJ · randn()`, `σ_RJ = 0.015 UI`, 샘플링 시점에만 적용
- 샘플링: `t_sample = 17.5 + timing_offset + RJ`
  - **선형 보간 필수**: 파형 해상도 1/64 ≈ 0.0156 UI ≈ σ_RJ 이므로
    nearest 샘플은 RJ를 격자에 먹혀 무력화함. `np.interp` 사용
- 패턴 판정: target=1이면 `V > Vref` PASS, target=0이면 `V < Vref` PASS
- 점 PASS = 8패턴 모두 × MC 모두 통과

### MC 붕괴 (효율 + 깔끔한 경계)

`V_sampled`는 Vref와 무관하다. "MC 전부 통과"는 패턴별 극단값으로 환원된다:

- target=1: `∀mc, V>Vref` ⟺ `min_mc(V) > Vref`
- target=0: `∀mc, V<Vref` ⟺ `max_mc(V) < Vref`

따라서 `V_eff[pattern, timing] = min/max over MC`를 한 번 구하고 Vref 축에
브로드캐스트로 비교한 뒤 8패턴 AND 한다. `10201×8×100` 평가가
`8×100×200` 보간 + 브로드캐스트로 줄어든다. 경계는 worst-of-N 극단값이
정하므로 약 `2.5·σ_RJ` 마진을 요구한다.

## 레이어 다양성 (16장)

| 메커니즘 | 역할 | 비고 |
|---|---|---|
| 패턴별·레이어별 수평 skew | 주 다양성 원천 | 8패턴 × 16레이어 독립 오프셋. 포락선 active 제약이 바뀌어 폭·비대칭·facet이 달라짐 |
| 레이어별 ISI 탭 크기 스케일 | 세로(높이) 다양성 | 수평 skew만으로는 아이 높이가 비슷해지는 문제 보정 |
| 샘플별 RJ (MC) | pass/fail 마진 + 경계 질감 | 위 MC 붕괴로 처리 |

전 레이어 동일 양으로 흔들면 아이가 평행이동만 하므로, skew는 반드시
패턴별·레이어별로 **독립**이어야 한다.

## 커버리지

레이어마다 독립적인 커버리지 목표를 `[20%, 55%]`에서 균등 추출하고,
레이어별 guard band를 이분 탐색으로 보정해 그 목표를 맞춘다. 작은 eye가
섞여 다양성이 커지고, 좌우 비대칭도 덜 두드러진다. `guard_v`를 직접 주면
모든 레이어에 고정 guard가 적용된다.

향후: 범위를 더 낮춰 난도를 올리는 것은 TODO.

## `generate_shmoo` 파라미터 (예정)

```python
def generate_shmoo(
    tau_rise=0.15, tau_fall=0.15,            # UI
    isi_taps=[1.0, -0.15, -0.08, -0.04],
    sigma_RJ=0.015, dcd_offset=0.01,         # UI
    sat_k=10.0, n_mc=100,
    n_timing=200, n_vref=150,                # W, H
    samples_per_ui=64,
) -> np.ndarray:                              # shmoo[n_vref, n_timing], bool
```

파라미터 영향: ISI 탭↑ → 아이↓(+facet), RJ↑ → 경계 흐려짐, DCD↑ → 좌우
비대칭, `sat_k`↑ → rail에 근접.

## 모듈 (구현 완료 — `src/shmoo/`)

| 모듈 | 책임 |
|---|---|
| `signal_gen.py` | RC 에지 파형(`reconstruct_waveform`), `soft_saturate` |
| `channel_model.py` | 이산 ISI FIR 컨볼루션(인과, `apply_isi`) |
| `jitter_model.py` | RJ 샘플(`sample_rj`), 패턴별 skew(`sample_skew`) |
| `shmoo_eval.py` | `generate_shmoo`, `generate_dataset`, MC 붕괴, 커버리지 보정 |
| `plotting.py` | Shmoo 맵 + eye diagram PNG |

실행: `python tools/visualize_shmoo.py --selftest --seed 0` (셀프체크 +
데이터 생성 + PNG → `docs/images/`).

## 구현 노트 / 관측 (2026-06-18)

- 셀프체크 통과: `received[17]` 인과 ISI 식 일치, MC 붕괴가 명시적
  전수 AND와 비트 단위로 동일.
- seed 0 기준 16레이어 생성 0.08s. 커버리지 범위 `[20%,55%]`에서
  레이어별 21~53%로 분포. eye 높이 270~600 mV, 폭 0.59~0.76 UI로
  수직·수평 모두 다양해짐(커버리지 고정 50% 때보다 개선).
- `sat_k=10`은 여전히 중심 전압을 거의 0/1로 포화시키지만, 레이어별
  커버리지 차이(guard band)가 eye 크기를 크게 벌려 수직 다양성을 만든다.
  더 큰 변화를 원하면 `sat_k`를 낮춘다.
- **eye는 우측으로 비대칭**: 타겟 비트 leading 에지의 상승시간이 창의
  좌측을 잠식하고 trailing 에지는 +0.5 경계에서 시작하므로 개구부가
  대략 `[-0.2, +0.5] UI`로 치우친다. 물리적으로 정상이며 다양성에 기여.

## 구현 시 검증 체크리스트

1. `received[17]`이 인과 ISI 식과 일치 (convolve `same` 함정 회피)
2. 샘플링에 선형 보간 사용 (RJ가 살아있는지)
3. MC를 min/max로 붕괴해도 "전부 통과"와 동치인지
4. target bit center = 17.5 UI, 아이가 timing=0 근처에 오는지(DCD 시 약간 이동 정상)
5. 16레이어가 시각적으로 서로 다른지(평행이동 복제가 아닌지)
6. 평균 커버리지 ≈ 50%
