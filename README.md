# ATM ROI-pair SC / tractogram

T1 강조영상 하나로 ROI 쌍별 streamline 을 생성하고 structural connectivity (SC) 를 예측한다.
upstream [ATM](https://github.com/) 의 구성(번들 30개 ConvVAE + 번들별 KDE 잠재 prior)을
Desikan-PD25 82 ROI 의 **3,321 쌍 단일 모델**로 확장한 것이다.

> **현재 상태: 목표 미달이고 그 사실이 정량화되어 있다.**
> 훈련 144명 GT SC 평균(상수, T1 을 아예 안 봄)이 test 31명에서 r **0.945** 인데
> T1 조건부 생성은 **0.788** 이고 **31명 전원(0/31)** 이 이 상수를 못 넘는다.
> 남의 tractogram 을 쓰는 cross-subject 기준선(SC 0.841, 전뇌 dice 0.574)보다도 낮다.
> 이 저장소는 그 실패를 재현 가능한 측정으로 남기는 것이 목적이다.

## 먼저 읽을 문서 두 개

| 문서 | 내용 |
|---|---|
| [`docs/PIPELINE_13_CHANGE_LOG.md`](docs/PIPELINE_13_CHANGE_LOG.md) | **무엇을 어떻게 고쳤고 결과 숫자가 어떻게 변했나** (M1~M12, 순효과표, 체크포인트 D0~D3 진행) |
| [`docs/PIPELINE_12_PROBLEM_SUMMARY.md`](docs/PIPELINE_12_PROBLEM_SUMMARY.md) | **남은 문제와 실험 결과** (기준선 대조표, 실험 원장, 개인차 3x3 전수표, 확정/미확정 구분) |

나머지 `docs/` 는 시간순 이력이다.
[`docs/T1_subject_specific_SC_TRK_collapse_solution_strategy.md`](docs/T1_subject_specific_SC_TRK_collapse_solution_strategy.md)
가 현재 진행 중인 해결 전략이다.

## 핵심 측정치 (test 31명, `d3_joint_step3000`)

| 방법 | 개인 정보 | SC r | 전뇌 dice |
|---|---|---|---|
| 자기 tractogram 재샘플 | 완전 | 0.936 | 0.783 |
| **훈련 144명 GT SC 평균 (상수)** | **없음** | **0.945** | — |
| 그룹 tractogram | 없음 | 0.914 | 0.614 |
| 남의 tractogram 아무거나 | 틀린 사람 것 | 0.841 | 0.574 |
| **이 모델 (T1 입력)** | 있어야 함 | **0.788** | **0.553** |

- 예측 SC 의 subject 간 상관 **0.99914** (GT 는 0.896) — 사실상 전원 같은 출력
- 템플릿 제거 후 잔차 상관 `residual_r` **0.023 ± 0.162** — 0 과 구분되지 않음
- 디코더는 병목이 아니다: 오라클 latent 로는 pair dice 0.524, 생성 경로는 0.068

## 구조

```
src/atm_sc/
  models/     ROIPairATM, EdgeCountHead, EdgeHead, latent prior, EndpointAssigner
  losses/     geometry / edge_count / sc_corr / sc_magnitude / route / endpoint / tract_length
  training/   trainer(step 단위 손실 조립) · run(phase 루프) · config(yaml -> dataclass)
  data/       ROIPairSubject, sc_template(train 전용 통계), balanced sampler
  evaluation/ 재현성·기하 지표
scripts/      전처리 -> 학습 -> 진단 -> 평가 드라이버 (번호순 실행)
configs/      phase 별 학습 설정 (retrain p0~p4, d1/d3, a1/a2, route s1~s5)
tests/        pytest 144개
```

## 실행

```bash
pip install torch numpy scipy nibabel pyyaml joblib
export PYTHONPATH=src

pytest -q                                              # 144개
python scripts/49_eval_frozen.py --help                # 평가 (규약 고정)
python scripts/54_a2_residual.py --selfcheck-only      # 잔차 학습 자기검증
```

학습·평가는 전처리된 `outputs/roi_pairs/{subject}/` 와 T1 캐시를 요구한다.
**데이터는 포함되어 있지 않다** (PPMI 는 통제 접근이고 로컬 산출물이 100GB 를 넘는다).

## 평가 규약 (중요)

dice 는 **규약 없이 인용하면 안 된다**. `wb@N` 은 `n_pred == n_gt == N`,
`pair@64` 는 `pred = gtA = gtB = 64` 다. 표본 크기가 다르면 저울이 달라 비율 해석이 성립하지 않는다.
자세한 내용은 `PIPELINE_12` §3 축 C.

추론에 그룹 정보(latent bank, 템플릿 pair 배분)를 넣는 구성은 폐기했다.
보고 지표는 T1 단독 `generated` 경로 하나다.
