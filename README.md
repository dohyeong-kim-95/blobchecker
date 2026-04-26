# Blobchecker 의사결정 브리프

프로젝트 의사결정을 하기 전에 이 문서를 먼저 봅니다.

## 현재 목표

하나의 2D 좌표 그리드 위에 있는 8개 이진 blob 레이어를 가능한 적은
좌표 질의로 재구성합니다.

한 iteration은 정확히 하나의 좌표 `(row, col)`를 선택합니다. Oracle은 그
좌표에서 8개 레이어의 이진 레이블을 동시에 반환합니다. 알고리즘은 질의
전에 그리드 크기만 알고 시작합니다.

## 정본 계약

상세한 문제/평가 계약은 [docs/spec.md](docs/spec.md)에 있습니다. 핵심만
요약하면 다음과 같습니다.

| 항목 | 현재 값 |
|---|---|
| 레이어 수 | 8 |
| Phase 0 그리드 | 50 x 200 |
| Iteration cap | `int(0.15 * H * W)` = Phase 0 기준 1,500 |
| Public seed suite | `[0..9]`, 개발/비교용 |
| Validation seed suite | `[100..199]`, 과최적화 방지용 |
| Oracle 출력 | `truth_full_mask[:, row, col]` |
| 채점 대상 | `truth_blob_mask`, outlier 제외 |
| 필수 출력 | `predicted_blob_mask.shape == (8, H, W)` |
| 정확도 기준 | 모든 레이어 `>= 98%` |
| Shape 기준, Phase 0 | 높이/너비가 `max(1, floor(5% * truth_dim))` 이내 |
| Shape 기준, Phase 1 | 높이/너비 정확히 일치 |
| 계산 비용 기준 | BOPs(blobchecker normalized operations) + native elapsed time |
| 전체 pass | 모든 레이어가 모든 활성 기준을 통과해야 함 |

Outlier는 oracle을 통해 관측되지만 핵심 복원 대상은 아닙니다. Outlier를
blob으로 예측하면 `truth_blob_mask` 기준 false positive로 자연스럽게
패널티를 받습니다.

## 현재 구현

| 컴포넌트 | 파일 |
|---|---|
| 데이터 생성기 | `src/generator.py` |
| Oracle wrapper | `src/oracle.py` |
| Evaluator와 결과 객체 | `src/evaluator.py` |
| 알고리즘 추상 API | `src/algorithms/base.py` |
| Pre-planned baseline | `src/algorithms/preplanned_greedy.py` |
| 현재 adaptive 알고리즘 | `src/algorithms/geometry_first.py` |
| 벤치마크 runner | `benchmark.py` |

현재 벤치마크 실행:

```bash
python benchmark.py --algo both
python benchmark.py --suite validation --algo geometry_first
python benchmark.py --suite both --algo geometry_first
```

## 최신 결과

Phase 0, 공개 placeholder seed 10개 `[0..9]`, grid 50 x 200 기준:

| 알고리즘 | Pass | Mean min accuracy | Mean time |
|---|---:|---:|---:|
| `preplanned` | 0/10 | 0.9279 | 7.0s/seed |
| `geometry_first` boundary trim 이전 | 0/10 | 0.9648 | 1.4s/seed |
| `geometry_first` boundary trim | 0/10 | 0.9695 | 1.5s/seed |
| `geometry_first` four-side trim | 0/10 | 0.9693 | 1.5s/seed |

현재 병목은 shared-coordinate API가 아니라 정확도입니다. Confirmed-zero
boundary trimming 이후 높이 복원은 개선됐지만, 아직 모든 레이어 98%
정확도 기준에는 도달하지 못했습니다.

새 알고리즘이나 파라미터 변경은 public seed `[0..9]` 결과만으로 판단하지
않습니다. 최소한 validation seed `[100..199]`에서도 같은 방향의 개선이
나오는지 확인합니다.

## 의사결정 원칙

다음 변경을 고를 때 이 원칙을 따릅니다.

1. Shared-coordinate 제약을 보존합니다. 레이어별 좌표 스케줄은 전역
   one-coordinate-per-iteration planner로 감싸지 않는 한 도입하지 않습니다.
2. Pass/fail의 정본은 evaluator입니다. 알고리즘은 `truth_blob_mask`나
   `truth_outlier_mask`에 접근하면 안 됩니다.
3. GP류 모델보다 geometry-first 개선을 먼저 시도합니다. 과거 작업에서
   dense/sparse GP는 복잡도 대비 이득을 정당화하기 어려웠습니다.
4. 평균 정확도가 아니라 모든 레이어 통과를 최적화합니다. 레이어 하나가
   실패하면 전체 run이 실패합니다.
5. Outlier 복원은 supplemental로 취급합니다. 핵심 기준은
   `truth_blob_mask` 대비 blob accuracy입니다.
6. 계산 비용은 Python 실행 시간만 보지 않습니다. C/Fortran/Nastran 계열
   저수준 구현을 가정한 BOPs와 native elapsed time을 함께 봅니다.
7. Public seed 개선만 보고 파라미터를 고정하지 않습니다. Validation seed
   suite에서 성능이 유지되지 않으면 과최적화로 봅니다.
8. 결과는 seed set, grid shape, 알고리즘 파라미터, 실행 command와 함께
   기록합니다.

## 현재 전략

15% 제출 예산 안에서 바로 최적화하지 않습니다. 아직 pass 가능한 알고리즘
구조가 확인되지 않았기 때문에, 먼저 넓은 query budget에서 성공 구조를
찾고 그 구조를 단계적으로 압축합니다.

| 단계 | Query budget | 목표 | 판단 기준 |
|---|---:|---|---|
| Phase A | 50% | Pass 가능한 알고리즘 구조 찾기 | 정확도/shape gate를 먼저 통과하는지 확인 |
| Phase B | 30% | 불필요한 query 제거와 예산 배분 조정 | Phase A 구조가 유지되면서 낭비가 줄어드는지 확인 |
| Phase C | 20% | Weakest-layer, boundary, hole 우선순위 압축 | 어떤 실패 위험에 query를 먼저 써야 하는지 정책화 |
| Phase D | 15% | 제출 조건 맞추기 | 공식 cap에서 public/validation 모두 유지되는지 확인 |

이 ladder는 실험 전략입니다. 정본 제출 기준은 여전히 `docs/spec.md`의
`int(0.15 * H * W)` cap입니다.

## 다음 결정 후보

가장 유용한 다음 작업은 다음 순서입니다.

| 우선순위 | 결정 | 이유 |
|---:|---|---|
| 1 | `budget_ratio`를 benchmark와 evaluator에서 설정 가능하게 만들기 | 50/30/20/15%를 같은 알고리즘으로 비교해야 구조 문제와 예산 문제를 분리할 수 있습니다. |
| 2 | Phase A 50%에서 현재 `geometry_first` 실패 유형 진단 | 15% 실패가 알고리즘 구조 실패인지 예산 부족인지 먼저 분리합니다. |
| 3 | Weakest-layer weighted acquisition 비교 | 한 레이어 실패가 전체 실패이므로 평균 uncertainty만으로는 부족합니다. |
| 4 | 내부 hole targeted probing 추가 | 최대 3개의 약 7 x 7 hole만으로도 2% error budget 대부분을 소모할 수 있습니다. |
| 5 | Boundary budget 재배분 | 더 많은 boundary scan이 항상 이득은 아니며 entropy acquisition budget을 빼앗을 수 있습니다. |

## 문서 지도

| 문서 | 역할 |
|---|---|
| [README.md](README.md) | 의사결정 브리프. 프로젝트 판단은 여기서 시작합니다. |
| [docs/spec.md](docs/spec.md) | 상세 문제 정의와 평가 계약의 정본입니다. |
| [journal.md](journal.md) | 시간순 실험, 실패, 시행착오, 폐기한 접근을 기록합니다. |
| [research/08_synthesis_and_recommendations.md](research/08_synthesis_and_recommendations.md) | 리서치 종합과 방법론 선택 근거입니다. |
| [docs/archive/legacy_wisdom.md](docs/archive/legacy_wisdom.md) | 과거 단일-blob 프로젝트에서 얻은 레거시 교훈입니다. |

삭제한 root-level 중복 문서는 다시 만들지 않습니다. 이 브리프보다 더
자세한 내용이 필요한 결정은 `docs/spec.md` 또는 `journal.md`에 반영하고,
여기에서 링크합니다.
