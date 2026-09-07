# 인코더·디코더 수정 파이프라인 (2026-09-07)

목표: **subject-specific** SC/tractogram. 이 문서는 그 목표를 향해 인코더와 디코더를 어떤 순서로
고치는지, 각 단계의 게이트가 무엇인지, 지금 어디까지 왔는지를 적는다.

관련 문서: [`PIPELINE_13_CHANGE_LOG.md`](PIPELINE_13_CHANGE_LOG.md) (수정 이력),
[`PIPELINE_12_PROBLEM_SUMMARY.md`](PIPELINE_12_PROBLEM_SUMMARY.md) (문제·실험),
[`T1_subject_specific_SC_TRK_collapse_solution_strategy.md`](T1_subject_specific_SC_TRK_collapse_solution_strategy.md) (전략).

---

## 0. 두 축은 다른 문제다 — 섞지 않는다

| 축 | 증상 | 담당 | 개인차에 직접 기여 |
|---|---|---|---|
| **E 개인차** | 예측 SC 가 전원 동일 (inter_subj 0.99914, GT 0.896) | **인코더 / 조건화** | **예** |
| **D 기하·prior** | 오라클 pair dice 0.524 vs 생성 0.068 | **디코더 / 좌표계** | 아니오 |

pair 앵커(D축)는 개인차를 개선하지 않는다. 생성 tractogram 품질을 고치는 작업이다.
두 축의 지표를 하나의 점수로 합치지 않는다 (전략 문서 §2.4).

---

## 1. 진단 — 병목을 어떻게 특정했나 (완료)

### E 축: 병목은 feature 도 손실도 아니고 **pooling** 이다

175명 실측 (`outputs/cache/s1b_feats/`):

| feature | subject 간 코사인 | subject 성분 비중 |
|---|---|---|
| `a512` global avg pool — **head 가 받던 것** | 0.9994 | **2.1%** |
| `g3` global stage3 | 0.9945 | 3.3% |
| `f_roi` ROI 국소 pooling | 0.8917 | **13.4%** |

GT SC 카운트의 subject 성분은 **18.5%** 다. 국소 feature 는 같은 자릿수, 전역 벡터는 한 자릿수 부족.

결정적 정황: 전역 feature 로 학습한 A2 arm 의 **예측 잔차 분산이 GT 의 2.7%** 인데, 이는
`a512` 의 subject 성분 **2.1%** 와 거의 같다. **모델은 입력에 있는 개인 정보를 이미 거의 다 쓰고
있었다.** 손실을 바꿔도 (A1 arm R/C, A2 arm D) 0 근처를 벗어나지 못한 이유다.

upstream 구조상 `global_avg_pool` 은 `rigid_UNet.forward` 안에 있다 (`model.py:222`). 번들 1개에는
맞는 설계지만 3,321 쌍에 같은 벡터를 주면 파탄난다.

### D 축: 병목은 디코더 용량이 아니라 **좌표계**다

| | 크기 mm | 부피 mm³ |
|---|---|---|
| 이 프로젝트 전역 뇌 박스 | 152 × 189 × 163 | 4.68e6 |
| upstream 번들 박스 (중앙, `supp/*_coords_*.npy`) | 83 × 121 × 100 | 1.17e6 |
| **실제 ROI 쌍의 GT 범위 (중앙)** | **43 × 71 × 61** | **1.67e5** |
| 앵커 박스 (valid 1,569개) | — | **전역 대비 39배 축소** |

디코더 복원은 3.08 mm 로 GT 잡음 바닥(pair 천장 MDF 3.92 mm)에 이미 닿아 있다 → **용량 문제 아님**
(W1-d 결론 유효). 피해자는 **prior** 다: 좌표가 전뇌 절대값이라 64-d `z` 가 "3,321 쌍 중 어느 것 +
뇌 어디쯤 + 모양"을 전부 떠안는다. upstream 은 번들 정체성이 30개 모델 가중치에 있어서 `z` 는
번들 *내부* 변이만 담으면 됐다. prior NLL 71.6 vs `arch_additive` 15.6 의 구조적 이유다.

---

## 2. 인코더 수정 (E)

| 단계 | 내용 | 파일 | 상태 |
|---|---|---|---|
| **E1** | ROI 국소 feature 캐시 (stage3 → 아틀라스 ROI 풀링, `[82,256]`) | `data/local_feats.py`, `scripts/42_space_probe.py` | **완료** 206명 |
| **E2** | **count head** 에 국소 가지 (0-init `Linear(512→hidden)` **가산**) | `models/edge_count_head.py` | **완료 + 학습·평가** |
| **E3** | **조건화(FiLM)** 에 국소 가지 (0-init `MLP(512→cond_dim)` 가산) | `models/roi_pair_embedding.py` | 배선·검증 완료, **학습 대기** |
| **E4** | UNet stage3/4 해동 + 잔차 supervision | `configs`, `unet_level` | 미착수 (E3 결과에 따라) |
| **E5** | 티어 1 명시적 해부 feature (ROI 조직량·거리·대비) | `data/anat_tier1.py`, `scripts/55_anat_tier1.py` | 추출 중, 프로브 대기 |

### E2 결과 (val 31명, 3,000 step, 같은 손실·데이터·step, **입력만 다름**)

| | resid_r | self−shuffled | diff_corr | var ratio | 식별 | inter_subj |
|---|---|---|---|---|---|---|
| 전역 `a512` | +0.001 | **−0.005** | −0.003 | 0.027 | 0.032 = chance | 0.99997 |
| **ROI 국소** | **+0.034** | **+0.031** | **+0.033** | 0.044 | 0.065 = 2× chance | 0.99972 |

국소 arm 은 250→2250 step 단조 상승(0.008→0.039)하고 **모든 지표가 같은 방향**으로 움직인다.
전략 문서 §8.4 의 1번 조건("올바른 T1 일 때만 잔차가 복원된다")이 처음 성립했다.
다만 0.034 는 go/no-go 기준 0.10 에 못 미친다 → E3·E4·E5 로 더 밀어야 한다.

### 왜 concat 이 아니라 가산인가
`net[0]` 의 입력 shape 이 바뀌면 기존 checkpoint 를 못 싣는다. 0-init 가산 가지는
**시작이 bit-exact** 라 d3_joint 를 그대로 이어받아 효과만 잰다.

---

## 3. 디코더 수정 (D)

| 단계 | 내용 | 파일 | 상태 |
|---|---|---|---|
| **D-a** | `decode_raw` 분리 (좌표 역정규화를 디코더에서 떼어냄) | `models/atm_adapter.py` | 완료 |
| **D-b** | 앵커 상수 생성 (train 144명 그룹 평균 경로 + 반범위) | `scripts/56_pair_anchors.py` | 완료 |
| **D-c** | `PairAnchor` + alpha 램프업 학습 | `models/pair_anchor.py`, `configs/retrain/d4_anchor.yaml`, `scripts/57_d4_anchor.py` | **실행 중** |
| **D-d** | invalid pair(53%) 처리 — pair 별 alpha 또는 앵커 개선 | — | D-c 결과에 따라 |
| **D-e** | prior 재구조화 (`arch_additive`) — 좁아진 좌표계 위에서 | — | D-c 이후 |

### 재매개화
```
mm = (1-α)·[(raw+1)·coord_scale + coord_min]  +  α·[anchor[pair] + half_range[pair]·raw]
```
`α=0` 이면 기존 경로와 **bit-exact**. 1,000 step 에 걸쳐 0→1 로 올린다.

### 왜 램프업이 필수인가 (실측)
```
α=0  복원 3.72 mm   (기존과 동일)
α=1  복원 25.24 mm  ← 한 번에 켜면 디코더가 무너진다
```
재매개화는 출력의 **의미**를 바꾼다. 같은 `raw` 가 57 mm 다른 곳을 가리킨다.

### 왜 recon 단독인가
이 단계의 질문은 "재매개화한 좌표에서도 같은 정확도로 복원하는가" **하나**다. 생성/SC 제약을 섞으면
무엇이 바뀌었는지 알 수 없다 (D1 이 같은 이유로 recon 단독이었다). 판정은 **eval 모드** RMSE 다 —
train 모드 값은 BN 때문에 추론에서 성립하지 않는다 (M6 참조).

---

## 4. 게이트

| 단계 | 통과 기준 | 실패 시 |
|---|---|---|
| E2 | `resid_r` 가 shuffled 를 유의하게 넘는다 | 국소 feature 도 아니라는 뜻 → 데이터 축 조사 |
| **E3** | E2(0.034) **대비 상승**, self−shuffled 유지 | 조건화 경로는 개인차와 무관 → E4 로 |
| E4 | `resid_r` ≥ 0.10 (전략 문서 §8.1 go/no-go) | 인코더 해동으로도 안 되면 GT 신뢰도·정합 조사 |
| E5 | 티어 1 단독 프로브가 전역 anatomy(0.101)를 넘는다 | 명시적 해부 feature도 무력 |
| **D-c** | eval 복원 RMSE ≤ 4.0 mm (W1-b 의 dice 역표 기준) | α 를 1 미만에서 멈추거나 D-d 로 |
| D-c 추가 | valid/invalid 앵커를 **나눠서** 본다 | invalid 만 나빠지면 D-d |
| D-e | prior NLL 하락 **그리고** 생성 pair dice 상승 | NLL 만 좋아지면 채택 안 함 |

---

## 5. 공통 규약 (모든 단계가 지킨다)

1. **새 통로는 0-init 가산** → 켜기 전 `max|diff| = 0.0` 을 assert 로 확인한다.
2. **통로가 죽지 않았는지도 확인** → 가중치를 교란해 출력이 실제로 바뀌는지 잰다.
3. **자기검증이 학습 전에 죽는다** (`scripts/5x_*.py` 의 `selfcheck`). 테스트 프레임워크는 안 쓴다.
4. **통계는 train split 에서만** (템플릿·앵커·정규화 상수). val/test 가 섞이면 누수다.
5. **평가 규약 고정** — dice 는 `wb@N`/`pair@64` 없이 인용 금지 (`outputs/eval/w3b_protocol.json`).
6. **한 번에 한 변수** — E2 의 두 arm 은 손실·데이터·step 이 같고 입력만 달랐다.

---

## 6. 이 과정에서 조용히 틀릴 뻔한 것 3건

| 문제 | 어떻게 드러났나 | 조치 |
|---|---|---|
| **val 국소 feature 부재** | `s1b_feats` 캐시가 train+test 175명뿐이라 첫 국소 arm 이 eval 훅에서 죽었다 | val 31명 추출 후 재실행. assert 가 잡아준 경우 |
| **T1 강도 미정규화** | 티어 1 의 `t1_mean` subject CV 2.17 이 이상해 조사 → 뇌 내부 중앙값이 **163~63,824 (390배, CV 1.635)**. 스캐너 스케일이지 해부가 아니다 | 중앙값 정규화, 스케일은 `t1_scale` 교란변수로 분리, 206명 재추출 |
| **`half` buffer 이름 충돌** | `nn.Module.half()` 와 겹쳐 `register_buffer` 가 `KeyError` | `half_range` 로 개명 |

두 번째가 가장 위험했다 — 정규화 없이 뒀으면 ridge 가 해부 대신 **획득 조건**을 학습하고,
사이트 효과가 SC 와 상관되면 **가짜 양성**이 나왔을 것이다.

---

## 7. 현재 상태

| 수정 | 배선 | bit-exact | pytest | 학습·평가 |
|---|---|---|---|---|
| E2 count head 국소 | ✅ | ✅ 0.0 | ✅ | ✅ **완료** resid_r 0.034 |
| E3 조건화(FiLM) 국소 | ✅ | ✅ 0.0 | ✅ | ⬜ 대기 |
| E5 티어 1 해부 feature | ✅ | — | — | ⏳ 추출 중 |
| D-c pair 앵커 | ✅ | ✅ 0.0 | ✅ | ⏳ **실행 중** |

**"배선했다"와 "효과를 확인했다"는 다르다.** 지금 성능으로 검증된 것은 E2 하나뿐이다.

## 8. 실행 순서와 의존성

```
E1 국소 캐시 ──> E2 count head ──> E3 조건화 ──> E4 UNet 해동
   (완료)         (완료 0.034)      (대기)        (E3 결과 후)
E5 티어 1 ────────────────────────> E3/E4 에 추가 입력으로 합류
   (추출 중)

D-a decode_raw ──> D-b 앵커 상수 ──> D-c α 램프업 ──> D-d invalid 처리 ──> D-e prior 재구조화
   (완료)           (완료)            (실행 중)         (조건부)            (D-c 이후)
```

E 축과 D 축은 **독립**이라 병렬로 간다. 마지막에 결합할 때는 두 축의 지표를 따로 기록하고
Pareto trade-off 를 확인한다 — 한쪽이 좋아지고 다른 쪽이 나빠지면 합산 점수로 숨기지 않는다.
