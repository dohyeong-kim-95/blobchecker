# 코드베이스 탐색 노트

## 전체 구조 요약

이 저장소는 **2D binary map 추정기 비교 실험**용 코드베이스입니다.

- `raw/synth/`: 합성 맵(ground truth) 생성기
- `lse/`: sklearn GP 기반 baseline
- `sparse_gp/`: sparse GP 기반 개선안
- `geo/`: 기하학(스캔라인 + BFS) 기반 추정기
- `contour/`: boundary search 기반 추정기
- `compare.py`: 각 방법의 결과 이미지/지표를 한 장으로 합성

핵심 문제 정의는 `problem_definition.md`에 있으며,
- 목표 정확도: 95%+
- 시간/메모리 제약: 50x200 기준 10초/100MB
로 요약됩니다.

## geo/ 상태

`geo/` 디렉토리는 이미 생성되어 있고, 다음 파일이 준비되어 있습니다.

- `geo/geo.py`: `GeoEstimator` 구현
- `geo/evaluate.py`: 평가 스크립트
- `geo/README.md`: 알고리즘/지표 설명
- `geo/output/`: truth/predicted/curve 이미지

`GeoEstimator`는 GP 없이 아래 5단계로 동작합니다.
1. 행별 스캔라인 이진 탐색으로 blob seed 탐지
2. 좌/우 경계 이진 탐색으로 row extent 추정
3. extent 기반 내부 채움
4. 희소 그리드 샘플링으로 hole seed 탐지
5. BFS flood-fill로 hole 복원

## 지금 바로 확인 가능한 실행 포인트

- 빠른 비교(캐시된 PNG 사용):
  - `python compare.py`
- 전체 재실행(느림):
  - `python compare.py --run-all`
- geo 단독 평가:
  - `python geo/evaluate.py`

## 권장 다음 작업

1. `compare.py --run-all --skip lse`로 빠른 회귀 확인
2. `geo/evaluate.py`에 budget/홀 파라미터 sweep 옵션 추가
3. `geo/README.md` 지표를 최신 실행 결과로 주기적 업데이트
