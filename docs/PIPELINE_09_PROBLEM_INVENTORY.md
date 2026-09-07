# 문제 목록 — 단계별 진단 재정리 (2026-09-05)

기준 체크포인트: `outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt`
근거: `outputs/eval/final_p4_joint_step3000.json` (test 31명), `outputs/checkpoints/retrain/{pipeline_state,val_metrics}.jsonl`,
`scripts/39_diag_conditioning.py`. **추론에 그룹 정보를 넣는 구성은 폐기했으므로 모든 수치는 T1 단독 `generated` 경로다.**

문제는 세 축으로 갈린다. 서로 독립이라 하나를 고쳐도 다른 축은 그대로다.

- **축 A 개인차** — 예측 SC 가 subject 마다 같다 (상관 0.99997, GT 0.882)
- **축 B 기하·스케일** — 생성 tractogram 이 GT 기하를 못 맞춘다
- **축 C 지표·절차** — 판단을 잘못하게 만드는 지표와 게이트

---

## 축 A — 개인차 (예측이 전부 같은 원인)

| ID | 단계 | 문제 | 측정 근거 | 심각도 | 선행 조건 |
|---|---|---|---|---|---|
| **A1** | anatomy 인코딩 | 전뇌 `global_avg_pool` 이 개인차를 지운다 | stage3 공간맵 subject 간 corr **0.575** → pool 직후 **0.9996** → fc 후 0.9996. 버리는 공간 편차 rel_spread 0.83 vs 남기는 채널 평균 0.13 | **치명** | A3 |
| **A2** | 조건 주입 | anatomy 512-d 가 3,321 쌍 **전부에 같은 상수**로 들어가 전역 배율만 바꾼다 (상관은 배율 불변) | `f=log_count−log_template` 의 pair 간 std 0.824 vs 같은 pair 의 subject 간 std **0.0277** (전체의 **0.96%**) | **치명** | A1 |
| **A3** | T1 전처리 | rigid T1 이 QSDR 아틀라스/streamline 과 **복셀 대응이 없다** → ROI 별 풀링이 불가능 | `PIPELINE_07` §2. 선형 probe 잔차 r: SyN **0.175** > rigid 0.104 | 大 (A1 의 전제) | — |
| **A4** | count head | 사실상 `(i,j) → train 평균` 조회표 | `count.all.r` 0.905 ≈ 템플릿, subject 성분 0.96%, `abl_pred_own_vs_zero_r` **0.998** | 치명 (A1·A2 의 결과) | A1, A2 |
| **A5** | edge head | "어느 연결이 존재하는가" 에 남은 개인 정보를 못 쓴다 | GT pair 목록을 주면 잔차 r 0.026 → **0.129** (`FINDINGS` §B-4). 미탐색 통로 | 大 | A1 |
| **A6** | 학습 p3·p4 | magnitude/scale/rmse 손실이 남은 개인 신호를 **깎는다** | `resid_r` 0.124 (p2) → 0.009 (p3) → **−0.031** (p4) | 大 | — |

크기 문제는 재학습으로 해결됐다: ‖a‖ 0.086 → **0.785**, 1층 기여비 ‖W_a·a‖/‖W_e·e‖ 1/22 → **0.57**.
남은 것은 **분산**이다 — ‖W_a(a_i−ā)‖ 0.097 vs ‖W_e·e‖ 2.25 (여전히 23배).

---

## 축 B — 기하·스케일 (생성 품질)

| ID | 단계 | 문제 | 측정 근거 | 심각도 |
|---|---|---|---|---|
| **B1** | latent z | prior 가 실제 다발 분포 밖 | pair 단위 dice **0.095** (실제 표본 천장 0.700 → 천장의 14%) | 大 |
| **B2** | 디코더 | 재구성 정확도가 기하의 병목. 학습 내내 평탄 | recon RMSE 4.40 → 4.59 → 4.00 → 3.91 → **3.73 mm** (아틀라스 복셀 2mm = 1.9칸). 오라클 latent 로도 dice 0.167 (천장의 28%) | 大 |
| **B3** | SC 추출 | 생성 가닥의 2/3 가 목표 ROI 쌍을 못 잇는다 | valid_conn **0.367** / partial 0.409 / invalid 0.224, endpoint_in_roi 0.745 | 大 |
| **B4** | 디코더·길이 | 길이를 재현하지 못한다 | 길이 r **0.182**, 생성 97 mm vs GT edge 158 mm, KS 0.349, Wasserstein 23.8 mm | 中 |
| **B5** | pair 선택 | GT 양성 pair 의 62% 만 고른다 | 1,812 선택 / GT 양성 **2,931** (전체 3,321). weak_edge_recall 0.866 | 中 |
| **B6** | 생성 예산 | 29k 가닥 vs GT 1,000,000 → 절대 스케일이 구조적으로 안 맞는다 | 총합비 **0.027**, CCC 0.058. (r 에는 영향 0 으로 이미 기각: 1.5만→10만에서 0.680→0.678) | 中 |
| **B7** | weight head | 마지막 층 weight rms **정확히 0**, 출력 상수 **1.0** (gradient 를 한 번도 못 받음) — 그런데 평가가 이 경로(`sc_w`)를 쓴다 | `FINDINGS` §B-1. 학습/평가 불일치 | 中 |

---

## 축 C — 지표·절차 (잘못된 판단을 만든 것)

| ID | 대상 | 문제 | 측정 근거 | 심각도 |
|---|---|---|---|---|
| **C1** | SC r 해석 | 전체 r 0.713 은 **tier 간 스케일 차이**가 만든 값이다. tier 안에서는 거의 못 맞춘다 | small **0.164** / mid **0.189** / large 0.624 vs all 0.713 | 大 |
| **C2** | p2 게이트 | val 8명 LOO 중심화는 잡음이 커서 **위양성**을 냈다 | 게이트 `resid_r ≥ 0.10` 을 0.124 로 통과 → test 31명에서 **−0.011** | 大 |
| **C3** | p4 게이트 | 기준 `SC r ≥ 0.88` 이 **폐기한 그룹 정보 구성**의 값이다 | latent bank + 템플릿 배분 구성에서 나온 0.88. T1 단독 기준으로 재정의 필요 | 中 |
| **C4** | p1 게이트 | `route_f1 ≥ 0.50` 미달인데 다음 phase 로 진행됐다 | 0.496 (p1) / 0.494 (p4) | 中 |
| **C5** | CCC·총합비 | B6 때문에 성능 지표로 읽을 수 없다 | CCC 0.058, 총합비 0.027 | 中 |
| **C6** | 기준선 | 유일하게 의미 있는 비교 대상은 그룹 템플릿이다 | 템플릿 r **0.945** / CCC 0.936, `beats_template` **0/31** | — |

---

## 인과 구조와 순서

```
A3 공간 대응 없음
   └─▶ A1 pooling 이 개인차 소멸 ──┬─▶ A4 count head = 조회표 ─┐
        └─▶ A2 상수 조건 주입 ─────┴─▶ A5 edge head 개인화 없음 ├─▶ 예측 SC 전부 동일
                                        A6 손실이 잔여 신호 삭제 ┘

B1 prior ┐
B2 디코더 ├─▶ B3 유효 연결 0.367 ──▶ 기하 품질 상한   (축 A 와 독립)
B5 pair  ┘   B4 길이 · B6 예산 · B7 weight head

C1~C5 ──▶ 위 두 축의 상태를 잘못 읽게 만든다
```

**순서 제약**: A3 → A1 → A2 는 사슬이다. A3 를 정하지 않으면 A1 을 고칠 수 없다
(stage3 feature 를 아틀라스 82 ROI 로 풀링하려면 둘이 같은 공간이어야 한다).
축 B 를 아무리 고쳐도 A1 이 남으면 예측 SC 는 계속 전부 같고, A1 을 고쳐도 B2(3.7mm)가
남으면 개인 신호가 SC 로 옮겨지지 않는다.

**C1·C2 는 코드 수정 없이 즉시 고칠 수 있다** — tier 내 상관을 주 지표로 올리고,
`resid_r` 게이트를 test 31명 기준으로 다시 정의한다. 이걸 먼저 해야 이후 실험의 판정이 성립한다.

---

## 추가 측정 (2026-09-06) — `outputs/checkpoints/retrain/val_metrics.jsonl` phase 궤적

| phase | pair_acc | pair_dice | route_f1 | sc r_log | length mm | resid_r | abl own / shuf / zero |
|---|---|---|---|---|---|---|---|
| p0 | 0.002 | 0.088 | 0.388 | 0.659 | 99.3 | 0.000 | 0.9481701 / 0.9481701 / 0.9481701 |
| p1 | 0.211 | 0.183 | 0.496 | 0.703 | 95.9 | 0.000 | 0.9481701 / 0.9481701 / 0.9481701 |
| p2 | 0.223 | 0.170 | 0.493 | 0.698 | 95.1 | 0.124 | 0.891387 / 0.891246 / 0.907328 |
| p3 | 0.282 | 0.184 | 0.492 | 0.743 | 91.9 | 0.009 | 0.902689 / 0.902639 / 0.915012 |
| p4 | 0.291 | 0.176 | 0.494 | 0.741 | 90.4 | -0.031 | 0.906504 / 0.906469 / 0.916852 |

`inter_subj_r` 는 다섯 phase 전부 1.0000 (GT 0.9021).

### A0 (신규, 치명) — anatomy 경로의 기여가 0 이거나 음수다
- **p0·p1 에서 own = shuf = zero 가 16자리까지 완전히 동일하다.** 6,000 step 동안 T1 입력이
  출력에 정확히 0 의 영향을 줬다. "영향이 작다" 가 아니라 경로가 끊겨 있었다는 뜻이다.
- p2~p4 에서도 own - shuf 격차는 1e-4 수준(+0.00014 / +0.00005 / +0.000035).
- **zero > own 이다** (0.9073 > 0.8914, 0.9150 > 0.9027, 0.9169 > 0.9065). anatomy 를 지우면
  SC 상관이 오히려 좋아진다 — 지금 anatomy 경로는 무시할 만한 게 아니라 약하게 해롭다.
- A1/A2 의 가장 강한 형태의 증거다. 상수 512-d 가 전역 배율만 흔들어 손해를 낸다.

### C7 (신규) — 개인차 지표를 phase 당 1점만 기록한다
- `resid_r` 이 전체 실행에서 5점뿐이라 "미학습" 과 "평탄역" 을 구분할 수 없다.
- 그런데 개인차 검증 비용은 **`indiv_sec` = 10.4 초**다. 500 step 마다 찍어도 phase 당 1분이다.
- 조치: 학습 중 500 step 간격으로 `resid_r` / `abl_gap` / `inter_subj_r` 을 기록한다.

### step 수에 대한 판정 (2026-09-06)
| 지표 | 궤적 | 판단 |
|---|---|---|
| route_f1 | p1 에서 0.496 후 9,000 step 미동 | 평탄역. step 증가로 0.50 못 넘는다 |
| pair_dice | p1 이후 0.17~0.18 고정 | 포화 |
| sc r_log | p3 이후 정지 | 포화 |
| length | 99.3 -> 90.4 단조 감소 (GT 158) | step 늘리면 악화 |
| resid_r | p2 0.124 -> p4 -0.031 | p3/p4 는 step 을 줄이거나 가중치를 낮춰야 한다 (A6) |
| pair_acc | 아직 상승 중 | 유일한 여지. 단 dice 가 평탄해 기하로 전환되지 않는다 |

**결론: step 은 A1/A2 를 고친 뒤에 조정할 변수다.** 표현력이 없는 모델을 오래 학습시키면
그룹 평균으로 더 단단히 수렴할 뿐이다. 늘릴 근거가 있는 곳은 S2 이후의 p2 하나이며,
그것도 3,000 step 파일럿에서 resid_r 이 상승 중일 때만이다.

---

## S0-a 재현 검증 결과 (2026-09-06) — `outputs/eval/s0a_metrics_recheck.json`

기존 보고 지표는 전부 정확히 재현됐다 (`reproduction_vs_reported` 전 항목 abs_diff = 0.0000).
`scripts/40_recheck_metrics.py --eval outputs/eval/final_p4_joint_step3000.json --template-subjects outputs/splits/train.txt`

### C6 강화 — 그룹 템플릿이 **모든 tier 에서** 모델을 이긴다
| | 전체 | small | mid | large | ctx-ctx | ctx-sub | sub-sub |
|---|---|---|---|---|---|---|---|
| 모델 (T1 단독 generated) | 0.713 | **0.164** | **0.189** | **0.624** | 0.736 | 0.603 | 0.672 |
| 그룹 템플릿 (train 144 GT 평균) | 0.945 | **0.222** | **0.323** | **0.926** | 0.947 | 0.863 | 0.888 |

이전에는 "전체 r 에서 템플릿에 진다(0.713 vs 0.945)" 였는데, 층화해 보니 **어느 한 구간에서도
이기지 못한다.** 모델이 잘하는 구간이 따로 있는 게 아니다.

### 사실 정정 1 — `n_ctx` 는 66 이다 (68 아님)
`roi_groups.N_CTX = 66`. C(66,2)=2145 · 66x16=1056 · C(16,2)=120 · 합 3321 로 저장된 모든
수치와 일치한다. 82 = 피질 66 + 피질하 16.

### 사실 정정 2 (중요) — 문서의 "test resid_r = -0.011" 은 **폐기된 alloc 경로** 값이었다
`summary["subject_specificity"].resid_r` = -0.0107 은 `--use-bank`/템플릿 배분(`alloc`) 벡터로
계산된 값이다. T1 단독 `generated` 경로의 값은 `summary["residual_r"]` = **+0.0040** 이다.

- 학습 중 검증의 `resid_r`(p2 = 0.124)은 `training/run.py:205` 의 `subject_specificity(P, G)` 로
  계산되며 **generated 경로**다 (run.py 에 alloc/use_bank 경로가 없다).
- 따라서 지금까지의 "val 0.124 vs test -0.011" 비교는 **경로가 서로 달라 성립하지 않았다.**
- C2(게이트 위양성) 의 결론 자체는 바뀌지 않는다 — 어느 쪽이든 test 값은 0 근처다. 다만
  정당한 비교를 위해 generated 경로의 LOO 중심화 값을 별도로 산출 중이다.
- 교훈은 A0 과 같다: **같은 이름의 지표라도 어느 경로에서 쟀는지 확인해야 한다.**
  (PIPELINE_08 §4 의 route_f1 관문 정정과 동일한 유형의 사고가 반복됐다.)

---

## 지표 프레이밍 정정 (2026-09-06, 사용자 지적)

**이 프로젝트의 산출물은 tractogram 이다.** T1 하나로 whole-brain streamline 을 생성하는
모델/알고리즘 개발이 목적이고, SC 는 생성물을 검증하는 하류 지표다.

그룹 평균 SC 행렬은 **streamline 을 하나도 생성하지 못한다.** 그것을 성능 기준선으로 놓고
"모델이 0.713 으로 템플릿 0.945 를 못 넘는다" 고 쓴 것은 과제를 수행하지 못하는 것과
수행하는 것을 비교한 것이다. **C6 을 아래로 대체한다.**

### C6 (재작성) — 템플릿의 역할은 목표가 아니라 지표 기각 근거다
그룹 템플릿이 층화 전 구간에서 0.222/0.323/0.926 을 찍는다는 사실이 말해주는 것은
"모델이 부족하다" 가 아니라 **"SC 상관은 개인차가 아니라 그룹 해부구조에 지배되므로
tractogram 생성 품질의 주지표가 될 수 없다"** 는 것이다. C1·C5 와 같은 결론이다.
템플릿 수치는 이 목적으로만 인용하고, 성능 목표로 인용하지 않는다.

### 주지표 재정의 — tractogram 수준
| 지표 | 현재값 | 기준선 |
|---|---|---|
| whole-brain trk dice | 0.546 (coverage 0.645) | S5-e 측정 중 |
| pair 단위 dice | 0.095 | 오라클 latent 천장 0.700 |
| valid_connection rate | 0.367 | S5-e 측정 중 |
| endpoint_in_roi | 0.745 | S5-e 측정 중 |
| 길이 평균 / 상관 | 97 mm (GT 158 mm) / r 0.182 | S5-e 측정 중 |
| streamline 수 | 29k (GT 1M) | - |
| **입력 의존성** own-shuf | **1.4e-4** (p2~p4: 1.4e-4/5.0e-5/3.5e-5) | 0 |
| **입력 의존성** abl_gap = own-zero | **-0.016/-0.012/-0.010** (전 phase <= 0) | 0 |

SC 상관은 **하류 검증 + 입력 의존성 진단**으로만 쓴다.

기준선 공백이 문제였다: dice 0.546 이 좋은지 판단할 근거가 없었다. S5-e 가
`cross_subject`(다른 subject 의 실제 GT) / `self`(GT 절반 분할 = 천장) /
`subsample`(1M->29k 개수 효과 분리) / `group_tractogram`(그룹 기준선의 tractogram 판)
네 가지를 실측 중이다. 전부 같은 QSDR 템플릿 공간이라 정합 없이 직접 비교된다.

### 계획 영향
- **축 B 는 대체안이 아니라 본선이다.** S1 미달 시 "한계 보고 후 종료" 가 아니라 축 B 에 자원 집중.
- S1 의 역할 변경: "SC 예측을 포기할지" 가 아니라 **"T1 이 조건화에 개인 정보를 실어 나를 수
  있는가"** 라는 전제 검사. 이게 안 되면 생성된 tractogram 이 subject 별로 같을 수밖에 없다.

---

## C2 확정 (2026-09-06) — p2 게이트 통과는 통계적으로 성립하지 않았다
`outputs/eval/s0b_gate_ci.json` · `python scripts/40_gate_ci.py`

| | resid_r | 95% CI | 폭 |
|---|---|---|---|
| val n=8 | +0.1235 | [+0.0194, +0.2276] | 0.208 |
| test n=31 | -0.0107 (LOO) / +0.0040 | [-0.0485, +0.0563] | 0.105 |

**두 CI 가 [+0.019, +0.056] 에서 겹친다.** 개인차가 test 수준(약 0)인 분포에서도 n=8 이면
**3.4% 확률로 0.10 을 넘는다.** 반폭 <= 0.05 에 n=34 가 필요해 `min_n: 31` +
`val_indiv_subjects: 8 -> 31` 로 상향했다 (추가 비용 약 30초/phase).

### 경로 혼동 (C2 의 실제 원인) — 세 숫자가 전부 다른 경로·다른 추정량이었다
| 값 | 경로 | 중심화 |
|---|---|---|
| val 0.124 (p2) | **count head** (`run.py:186-207` `count_sc()` = `edge_log_counts().exp()`, tractogram 미생성) | LOO |
| test -0.011 | **alloc (폐기된 그룹정보 경로)** | LOO |
| test +0.0040 | **generated** | 템플릿 |

generated 경로의 LOO 값은 계산 자체가 불가능했다 — subject 벡터가 `json.dumps` 직전에
pop 되어 어떤 산출물에도 없다. `scripts/29_final_evaluation.py` 가 이제
`final_*_vectors.npz` 로 저장하므로 다음 평가부터 재계산된다.

다만 게이트 위양성 결론은 test 값 없이도 증명된다: 평가한 체크포인트(p4_joint)의
**val 값 자체가 -0.031** 이다. p2 통과는 그 시점 한정이었다.

## C4 해소 (2026-09-06) — 게이트가 실제로 멈춘다
`src/atm_sc/evaluation/gates.py` 신규. 미달 시 `phase_idx` 를 올리지 않고 **exit 4**
(재시작해도 같은 지점에서 막힌다), state 에 `gate_halt.failed` 기록. 지표 키 없음/None/NaN
이면 즉시 assert (조용한 통과 없음). `min_n` 미달은 값이 기준을 넘어도 **통과가 아니라
"표본 부족"**. `--ignore-gates` 는 배너 경고 후 진행.

**실제 `val_metrics.jsonl` 재생 결과 — 새 게이트가 과거 사고 두 곳을 모두 잡는다**:
p1 `route_f1=0.4956 -> FAIL -> 중단`, p2 `resid_r=0.1235 n=8/min_n=31 -> 표본 부족 -> 중단`,
p4 `resid_r=-0.0307 -> 중단`.

p4 의 `SC r >= 0.88` 은 삭제했다 (폐기된 `--use-bank` 수치). 회귀 감지
`sc_pass_r_w_delta_prev >= -0.02` 로 대체.

## C7 해소 — `indiv_log_every: 500`
`indiv_trace.jsonl` 에 500 step 마다 개인차 지표를 append (no_grad + RNG/모드 복원,
학습 상태 불변). 기존 `val_metrics.jsonl` 스키마는 그대로.

### 정의 정정
`abl_gap` 은 `own - shuf` 가 아니라 **`abl_own_r - abl_zero_r`** 다 (`ablation_gap`).
- own - shuf: p2~p4 에서 1.4e-4 / 5.0e-5 / 3.5e-5
- abl_gap (own - zero): 0 / 0 / -0.0159 / -0.0123 / -0.0104 — **전 5 phase 에서 <= 0**,
  `abl_ordered` 전부 false. T1 을 0 으로 지우는 쪽이 자기 T1 보다 상관이 높다.

---

## S1-a 결과 (2026-09-06) — `outputs/eval/s1a_atlas_pool_check.json`
`python scripts/41_check_roi_pool.py --n 4`

### feature 격자 affine 확정
`stable/stable/model/model.py:148-166`: stage1 전부 k3/s1/p1, stage2·stage3 의 첫 conv 만 k3/**s2**/p1.
k3s2p1 은 출력 j <-> 입력 중심 2j 이므로 두 번이면 **4k**. k=0..48 -> 0..192 로 193 축을 정확히 덮는다.
**feature affine = W_AFFINE 3x3 x4, translation 불변 = [[4,0,0,-96],[0,4,0,-132],[0,0,4,-78]]**, shape (49,58,49).
autograd receptive-field probe 로 실측 확인 (내부 probe 중심이 정확히 4xindex, RF 반폭 18 voxel).

### 검증
- 82/82 ROI voxel >= 1 (labeled 24,905; min 1 `R_subthalamic_nucleus`, median 201, max 1180).
- **취약 ROI 3개**: R_STN 1 · L_STN 2 · L_red_nucleus 4 voxel. 통과하지만 사실상 단일 voxel 값이라
  잡음일 수 있다 -> S1-b 에서 포함/제외 두 결과를 낸다.
- 좌우 위반 0 (L 평균 x -23.4mm / R +26.7mm). **W 는 x=+1mm/voxel, 아틀라스는 -2mm/voxel 이라
  voxel index 직접 비교는 좌우가 뒤집힌다 — mm 경유가 필수.**
- `f[82,256]`: NaN/Inf 0, 빈 ROI 행 0. 51 s/subject (이 노드 nproc=1).

### 핵심 — ROI 풀링은 개인차를 보존한다
| 단계 | subject 간 상관 |
|---|---|
| stage3 공간맵 | 0.6206 |
| **ROI 풀링 f[82,256]** | **0.9321** |
| 전뇌 풀링 stage3[256] | 0.9991 |
| conv4 + GAP + fc [512] | 0.9998 (보고값 0.9996 재현) |

ROI 별 mean 0.969 / min 0.454 (R_STN, 1 voxel 이라 잡음 가능) / max 0.998.
**전뇌 풀링보다 확실히 낮다 = 개인차가 살아 있다.** 다만 이 잔여 분산이 SC 를 설명하는지는
S1-b 의 probe 가 답한다. 특징 수준 상관이 낮다는 것만으로 예측력을 주장할 수 없다.

### A3 정정 (중요) — "rigid 는 아틀라스와 대응이 없다" 는 과장이었다
`outputs/cache/{sub}_T1w__torigid.nii.gz` 는 **이미 템플릿 격자**다. rigid 정합은 subject T1 을
템플릿 공간으로 보내므로 **격자 대응은 있다.** rigid 가 못 맞추는 것은 개인별 형태 차이이지
격자 자체가 아니다. 따라서:
- A3 는 "선행 조건 blocker" 가 아니라 **"정합 정밀도 한계"** 로 격하된다.
- **결정 실험을 in-distribution 으로, 즉시 돌릴 수 있다** (rigid + ROI 풀링 vs rigid + 전뇌 풀링).
- 기존 근거였던 linear probe (SyN 0.175 > rigid 0.104) 는 "SyN 이 더 나은 정합" 을 뜻하지
  "rigid 는 대응이 없다" 를 뜻하지 않는다.

### 조건 (b) rigid->warp 는 현재 불가
206명 어디에도 ANTs 변환 파일이 없다. `src/atm_sc/data/prepare_t1.py:register_to_template` 이
`reg["warpedmovout"]` 만 저장하고 `fwdtransforms` 를 버린다. 재계산 비용 ~130 s/명
(`outputs/preprocess_logs/summary.csv` step 01 중앙값 129.5 s, n=204) x 206 = **7.4 core-hours**.
1차 판정이 통과하면 그때 결정한다.

---

## B7 해소 (2026-09-06) — weight head 가 죽은 원인 확정
`outputs/eval/s5a_weight_head.json`

### 원인: config/코드 불일치로 **손실에 연결된 적이 없다**
`src/atm_sc/training/trainer.py:204` 가 `w = m.weights(c, zc)` 를 계산하지만
`trainer.py:205-210` 이 `cfg.weight_mode != "head"` 이면 `w = nh` 로 **덮어쓴다**.
`configs/retrain/p0~p4.yaml` 과 `configs/route/s1~s5` 가 전부 `weight_mode: count` 다.
코드 기본값(`trainer.py:96`)은 `"head"` 라 배포 config 와 어긋나 있었다.

**결정적 증거**: 체크포인트 optimizer state 의 파라미터 110개 중 **정확히 idx 86-91
(weight_head 6개)만 AdamW state 가 없다.** AdamW 는 첫 gradient 에서 state 를 만드므로,
3,000 step 동안 gradient 가 0 이었다는 직접 증거다. net.0/net.2 도 PyTorch 기본 init 값 그대로.
출력은 12,288 샘플에서 bit-exact 1.0 (std 0). 옵티마이저 누락도 gradient 차단도 아니다.

### `sc_w` == unweighted `sc` (bit-exact)
`inference/generate_sc.py:57-75 _weighted_sc` 는 `tt_io.py:208-233 hard_sc` 의 pass/end 규칙을
그대로 복제하고 1 대신 `w[t]` 를 더한다. `w == 1` 이므로 동일. 실제 atlas + 합성 tractogram 으로
검증: pass/end 둘 다 `max_abs_diff = 0.0`. 두 번째 경로(`generate_sc.py:158`)는 `sc` 와 `sc_w` 에
**같은 ndarray 를 대입**해 정의상 동일 (aliasing).

**따라서 지금까지 보고된 `generated` 수치는 전부 가중 없는 count SC 이며 해석이 바뀌지 않는다.**

### 권고: (B) 제거
GT 에 streamline 별 가중치에 대응하는 값이 없다 (GT SC = 실제 가닥 수, 가닥마다 1). 절대 스케일은
`count_head_end` 가 이미 맡는다. `roi_atm.py:132-133` 주석에 (A) 시도 실패가 기록돼 있다
("총합 정규화된 magnitude loss 로는 절대 스케일을 못 배운다, 실측 CCC 0.02"). 수치가 동일하므로 무비용.

### 적용 대기 patch (다른 에이전트 작업 종료 후 일괄 적용)
- `scripts/29_final_evaluation.py` line 150·157·166·227(else)·231: `sc["pass"]["sc_w"]` -> `sc["pass"]["sc"]`
- `src/atm_sc/inference/generate_sc.py:158`: 두 키가 같은 배열을 공유 -> `W[mode].copy()`
- `src/atm_sc/training/trainer.py:96`: 기본값을 `"count"` 로, 또는 `weight_mode=="count"` 일 때
  line 204 의 `m.weights()` 호출을 건너뛰어 죽은 forward 제거

---

## S5-e 기하 기준선 실측 (2026-09-06) — `outputs/eval/s5e_geometry_baselines.json`
`scripts/43_geometry_baselines.py`. test 31명 중 시드 20250906 으로 10쌍(20명), group 은 train 8명 합본.
채점 규약은 모델 평가와 동일(`n_gen==n_gt==8000`, `voxel_mm=2.0`) — pair 단위 `self` 가 0.704 로
모델 JSON 의 ceiling 0.6997 과 일치해 규약 동일성을 확인했다. bbox IoU 0.94 로 동일 공간 확인.

### 결론: 모델은 "다른 사람의 실제 tractogram" 을 못 넘는다
| 지표 | 모델 | cross_subject | group_tractogram | self (천장) |
|---|---|---|---|---|
| whole-brain trk dice | **0.546** | **0.574** ±0.016 (0.545~0.600) | **0.614** | 0.783 |
| pair 단위 dice (16가닥) | **0.095** | **0.140** | 0.179 | 0.704 (자기 GT 16가닥 0.385) |
| edge 길이 상관 | **0.182** | **0.629** | - | 0.909 |

모델의 0.546 은 cross_subject 10쌍의 **최저값(0.545)과 같은 수준**이다. 즉 "무작위로 고른
다른 사람의 뇌" 분포의 바닥에 있다.

**그룹 비교가 이번에는 정당하다.** 그룹 평균 SC 행렬은 streamline 을 못 만들지만
**그룹 tractogram 은 만든다.** 같은 지표로 채점해 0.614 로 모델(0.546)을 이긴다.

### B6 (streamline 개수) 는 원인이 아니다 — 기각
pred 1k/4k/8k/16k/29k -> dice 0.506/0.740/0.791/0.802/0.788. 그리고 모델 평가가 양쪽을 8k 로
자르므로 "29k vs 1M" 격차는 채점 단계에서 이미 상쇄돼 있다. **모델의 0.546 은 개수 탓이 아니다.**

### B3 (valid_conn / endpoint_in_roi) 재검토 필요
`bundles.npz` 는 양 끝이 서로 다른 ROI 에 닿은 가닥만 담고 pair 라벨도 같은 아틀라스로 매겨서
**모든 기준선이 구조적으로 1.000** 이다 — 이 지표로는 기준선이 구분되지 않는다.
raw 1M whole-brain GT 의 실제 endpoint_in_roi 는 **0.597** 이고 모델은 0.745 로 오히려 높다.
raw 계열 dice 로 다시 재도 결론은 같다: cross 0.599 / self 0.771 / group 0.636.
**"endpoint_in_roi 0.745 가 나쁘다" 는 근거가 없었다.**

### B4 정정 — 모델은 짧은 게 아니라 **길다**
`final_p4_joint_step3000.json` 의 `len_mean_diff_mm = +13.65` (median +19.74, Wasserstein 23.8,
KS 0.349). 생성 97mm - GT 약 83mm. S5-e 의 `bundles.npz` 기준 GT 평균 81~82mm 와 일치한다.

- 기존 문서의 "GT 158mm" 은 **edge segment 길이**로, whole streamline 길이가 아니다. 비교 대상이 틀렸다.
- 따라서 학습 궤적 99 -> 90mm 는 악화가 아니라 **목표(약 83mm)로 수렴하는 개선**이었다.
- 실제 결함은 평균이 아니라 **edge 별 길이 상관 0.182 (cross_subject 0.629)** 다.
  어느 연결이 길고 짧은지를 못 맞춘다.

### 새로 생긴 목표 사다리 (이전에는 기준선이 없어 해석 불가였다)
| 단계 | trk dice | 의미 |
|---|---|---|
| 최소선 | > 0.574 | 다른 사람 뇌를 복사하는 것보다 낫다 |
| 유의미 | > 0.614 | 그룹 tractogram 보다 낫다 |
| 천장 | 0.783 | 같은 사람 GT 를 반으로 가른 값 |

---

## S1-b 공간 결정 실험 결과 (2026-09-06) — **A1/A2 중단**
`outputs/eval/s1b_space_probe.json` · `scripts/42_space_probe.py` · 특징 캐시 `outputs/cache/s1b_feats/`
사전등록 프로토콜(`PIPELINE_10_RESOLUTION_PLAN.md` "S1 분석 프로토콜") 그대로 실행. test 31명, 3321 pair.

### test 잔차 r (log, 중앙값 [95% CI])
| 조건 | all | small / mid / large |
|---|---|---|
| **R-ROI** (rigid, 82-ROI 풀링) | **+0.063** [+0.03,+0.09] | +0.027 / +0.076 / +0.065 |
| **R-GAP** (rigid, 전뇌 a[512]) | **+0.068** [-0.00,+0.13] | +0.050 / +0.073 / +0.054 |
| S-ROI (SyN, off-distribution) | +0.073 | +0.056 / +0.079 / +0.069 |
| S-GAP (SyN, off-distribution) | +0.097 | +0.075 / +0.098 / +0.099 |

### 대조군
- **셔플 = 깨끗하다**: R-ROI -0.0042 [-0.018,+0.018], R-GAP +0.0011 [-0.013,+0.042]. 둘 다 CI 가
  0 을 포함 -> **누수 없음, 판정 유효.**
- **머리 크기가 더 높다**: 뇌 부피 단일 예측자 **+0.102** [+0.06,+0.12] > R-ROI +0.063.
  짝지은 차 R-ROI - 머리크기 = **-0.012** [-0.051,+0.025]. (T1 총 강도는 +0.051.)
- 취약 ROI 3개 포함/제외 차이 없음 (+0.0629 vs +0.0650).
- SyN 은 off-distribution 확인: 175명 중 **21.7%** 가 ch0 max>1 (rigid 는 전원 [0,1]),
  ch0 평균 0.110 vs 0.165. WM 채널은 rigid T1 에서만 생성돼(scripts/36) syn 과 공간이 어긋난다
  (corr(T1,WM) 0.489->0.409). **2차는 참고용, 판정은 1차만으로 했다.**

### 판정
**ROI 국소 풀링이 전뇌 풀링보다 나쁘다** (R-ROI 0.063 < R-GAP 0.068; SyN 쪽도 S-ROI < S-GAP).
게이트가 요구한 "R-GAP 대비 +0.05" 를 못 넘고(-0.005), 필수 대조군 3(머리 크기 +0.102)도 못 넘는다.
사전등록 문언대로 이 잔여 신호는 개인 연결성이 아니라 **크기 효과**다.

**S1 이 물은 질문("ROI 국소 풀링이 전뇌 풀링이 지우는 신호를 살리는가")의 답은 아니오.**
**A1/A2 구조 변경(S2)과 파일럿(S3)을 하지 않는다.**

### 사전등록이 실제로 값을 했다
대조군 3(머리 크기) 없이 봤으면 +0.063 은 "약한 신호" 로 읽혀 S2+S3 에 며칠을 썼을 것이다.
전뇌 부피라는 스칼라 하나가 그보다 높다는 사실이 판정을 뒤집었다.

### 남은 개인차 통로 (미검증, 싸다)
rigid 는 6-DOF 라 개인 크기가 템플릿 공간에 보존된다. 따라서 **ROI 별 조직 부피**
(scripts/36 의 WM 확률맵을 82 ROI 로 합산)를 예측자로 쓰는 국소 형태계측 검정이 GPU 없이 가능하다.
전뇌 부피 +0.102 를 유의하게 넘지 못하면 개인차 축은 닫는다.

---

## W1-a 결과 (2026-09-06) — flip 은 무관, **BatchNorm train/eval 격차 발견**
`outputs/eval/w1a_flip_check.json` · `scripts/44_recon_flip_check.py`. 4 subject x 32,768 가닥, 오라클 posterior.

### flip(방향 모호성)은 원인이 아니다 — 기각
| | 값 |
|---|---|
| `rmse_noflip` (trainer.py:350 와 동일 식) | 3.700 mm |
| `rmse_flipaware` (손실과 동일 식) | 3.700 mm |
| `frac_flipped` | **0.0%** (32,768개 중 0개) |

네 조건(train/eval x mu/sample) 전부 뒤집힘 <= 0.01%, 차이 <= 6e-5 mm. self_test 로 flip 검출이
살아 있음을 보증(rec=GT.flip 강제 시 100% 검출). **코드 수정 없음.**

부수: 로그의 3.55 는 `L_recon`(가닥별 RMSE 평균), 무-flip 지표는 3.791. 차이는 flip 이 아니라
집계 방식(Jensen)이다.

### C8 (신규, 大) — 복원 오차 3.55mm 는 **학습 시점 값이고 추론에서 성립하지 않는다**
ConvVAE 에 **BatchNorm 5개**. 같은 복원이:
| 모드 | RMSE |
|---|---|
| train (batch 통계) | **3.70 mm** |
| **eval (running 통계)** | **7.96 ~ 8.14 mm** |

BN buffer 는 정상 저장·로드된다 (`num_batches_tracked` 132,531). 손상이 아니라 순수한
batch<->running 통계 격차다.

**실제 추론 복원 오차는 약 8 mm** — 복셀(2mm)의 4배, 점 간격(1.24mm)의 6.5배.
오라클 latent pair dice 0.167 이 이것으로 설명될 가능성이 크다.

**`scripts/38_decoder_capacity.py` 의 base arm 9.17mm 도 같은 원인이다** (`evaluate()` 가
`m.eval()` 을 쓴다). 3.55 vs 9.17 불일치는 이것으로 규명됐다.

### 원인 가설과 처방
학습 배치가 **pair 조건 다발**이라 배치 내 streamline 이 강하게 상관돼 있다. batch 통계가
배치마다 크게 달라지고, 망이 **배치 구성에 묶인 정규화에 의존**하도록 학습된다. 추론 시
전역 running 통계를 쓰면 off-distribution 이 된다. stale buffer 가 아니라 구조적 문제다.

처방 (D1 최우선, refiner 보다 먼저):
1. **BN 재보정** — gradient 없이 대표 배치로 running 통계 재축적. 몇 분. 파라미터 증가 0.
2. **BN 동결(eval 모드) 학습** — train 과 inference 를 일치시킨다. D1 기본 설정.
3. **배치 구성 재검토** — 순수 복원 단계에는 pair 조건이 불필요하니 여러 pair 를 섞는다.

이후 모든 RMSE 보고는 **train/eval 두 모드를 함께** 낸다. 한쪽만 보면 같은 착시가 반복된다.

---

## W1-b 보정 곡선 (2026-09-06) — **게이트 A 통과, 변위 가설 정량 확인**
`outputs/eval/w1b_dice_displacement.json` · `python scripts/45_dice_displacement.py --n-subjects 5`
seed 20250906, test 5명, subject 당 pair 96개(tier x block 층화 + top-50), 총 481 pair, 361초.
채점은 `bundle_geometry_metrics` 와 **bit-exact** 검증 (소수점 12자리 일치).

### 곡선
| sigma (= 실측 RMSE) | wb iid | wb smooth | wb shift | pair iid | pair smooth | pair shift |
|---|---|---|---|---|---|---|
| 0 | 0.778 | 0.778 | 0.778 | 0.498 | 0.498 | 0.498 |
| 2.0 | 0.703 | 0.726 | 0.731 | 0.525 | 0.411 | 0.409 |
| **3.55** | 0.631 | **0.670** | 0.677 | 0.444 | **0.325** | 0.317 |
| 5.0 | 0.589 | 0.629 | 0.637 | 0.375 | 0.266 | 0.256 |
| **8.0** | 0.540 | **0.577** | 0.585 | 0.275 | **0.191** | 0.169 |
| 14.0 | 0.481 | 0.516 | 0.529 | 0.164 | 0.118 | 0.093 |

### 검증: sigma=8 이 관측을 재현한다 (3.55 는 아니다)
| 관측값 | sigma=3.55 예측 | sigma=8.0 예측 |
|---|---|---|
| 오라클 pair dice **0.167** | 0.317~0.444 (**2배 과대**) | **0.169 / 0.191 — 명중** |
| 모델 whole-brain **0.546** | 0.631~0.677 | **0.540~0.585 — 명중** |

역산 실효 변위: wb 0.5457 -> 7.6~12.0mm, pair 0.167 -> 8.1~9.4mm. **W1-a 의 eval 모드
실측 7.96~8.14mm 와 수렴한다.** 즉 병목은 학습 시점 3.55mm 가 아니라 **추론 시점 BN 불일치 8mm** 다.

### 역표 — 목표 RMSE 는 4mm
| 목표 | iid | **smooth (디코더에 가까움)** | shift |
|---|---|---|---|
| whole-brain **0.614** (그룹 tractogram 초과) | 4.15 | **5.78** | 6.11 |
| pair **0.3** | 7.15 | **4.16** | 3.96 |
| wb 0.65 / 0.70 | 3.12 / 2.06 | 4.27 / 2.72 | 4.52 / 2.89 |

**두 목표의 교집합은 RMSE <= 약 4mm 이고, train 모드 오차 3.55~3.70mm 는 이미 그 안에 있다.**
따라서 **BN 격차만 없애면(8 -> 3.7mm) wb 0.670 · pair 0.325 가 예측되어 두 게이트를 모두
통과한다 — 디코더 용량 증설 없이.** D1 의 처방은 용량이 아니라 **BN 정합**이다.

### BN 을 고쳐도 남는 잔차가 있다
모델의 **생성** pair dice 0.0946 은 변위 **13.7~17.9mm** 에 해당해 8mm 를 초과한다
(`final_p4_joint_step3000.json` 의 per-pair MDF 16.75mm 와 일치). 오라클 0.167 은 BN 으로
설명되지만 생성 0.095 에는 **prior/조건화의 latent 오류가 추가로** 얹혀 있다 -> D2 의 대상.

### C9 (신규) — pair dice(n_pred=16) 는 흐릿한 생성기에 상을 준다
`pair_n16_vs_full` + iid 변위에서 dice 가 sigma 에 **단조 감소하지 않는다** (0.498 -> 0.545@sigma=1.0
-> 하락). 16 가닥이 256 가닥 GT 번들을 과소 커버(cov 0.35)하는 상태라 등방 잡음이 예측
footprint 를 GT 복셀 안으로 부풀려 coverage 가 overreach 보다 빨리 오른다.
**이 지표를 최적화 대상으로 삼으면 흐릿한 생성기가 유리해진다.** 상관(smooth) 변위에서는 단조다.
-> Wave 2 에서 pair dice 의 n_pred 를 올리거나 smooth 기준으로 해석한다.

### 부수 확인
두 변위의 방향이 반대다: whole-brain 에서는 상관 변위가 iid 보다 **유리**(3.55mm 에서
0.670 vs 0.631), pair 에서는 **불리**(0.325 vs 0.444). GT 다발 실측 평균 길이 74.5~83.6mm
("약 83mm" 가정 부합).

---

## W1-c dice -> SC r 대응 (2026-09-06) — `outputs/eval/w1c_baseline_sc.json`
`python scripts/43_geometry_baselines.py --sc` (18분, GPU 미사용). GT = `.mat` pass-SC, 상삼각 3321 pair,
test 10쌍 seed 20250906. 기하 실행과 **같은 seed/쌍/rng 로 같은 가닥**을 채점.

| 기준선 | dice | 전체 SC r | tier small/mid/large |
|---|---|---|---|
| 모델 | 0.546 | **0.713** | 0.164 / 0.189 / 0.624 |
| cross_subject (pairs / raw) | 0.574 / 0.599 | **0.841 / 0.891** | 0.157/0.195/0.793 |
| **group_tractogram** (pairs / raw) | 0.614 / 0.636 | **0.914 / 0.932** | 0.214/0.297/0.882 |
| group raw, 32명 pool, 29k | - | **0.949** | 0.244 / 0.324 / 0.932 |
| self (천장) | 0.783 / 0.771 | 0.936 / **0.993** | 0.551/0.654/0.903 · 0.396/0.694/0.991 |
| 그룹 평균 SC **행렬** 144명 | - | 0.953 | 0.261 / 0.340 / 0.936 |

### 판정: 0.85 통과. 그리고 tractogram = 행렬이다
같은 subject pool 로 맞추면 **tractogram − 행렬 = −0.002** (8명 0.939 vs 0.941 / 32명 0.949 vs 0.951).
남은 격차는 템플릿 subject 수뿐 (8명 0.941 -> 144명 0.953). 보고된 0.945(31명)도 재현된다.

### **전제의 절반이 틀렸다 — 전체 SC r 은 기하 품질의 증거가 아니다**
group tractogram 은 **개인 정보가 0** 인데도 0.93 을 낸다. tier 내부 r (0.19/0.28/0.91) 이
그룹 행렬(0.26/0.34/0.94)과 같고 self 천장(0.61/0.83/1.00)과는 멀다.
**즉 전체 SC r ~ 0.9 는 "그룹 수준이면 자동으로 나오는 값" 이다.**

추가 대조군이 이를 못박는다: 대상 본인 가닥을 쓰되 endpoint pair 당 개수를 균일화해 **망친**
tractogram (dice 0.732, 11.8k 가닥) 도 **SC r 0.945** 다.
-> **pass-SC r 은 개수 배분이 아니라 "어느 ROI 쌍을 지나갔나"(점유)에 지배된다.**

### C10 (신규, 大) — 모델은 기하 추세선 아래에 있다
기준선 8점 회귀 **`sc_r = 0.622 + 0.445 x dice`** (r=0.817) 는 dice 0.546 에서 **0.865** 를
예측하는데 실측은 **0.713** — 잔차 **−0.15**. 남의 진짜 tractogram(0.891)보다도 낮다.

기하만으로 설명되지 않는 별도 결함이 있다는 뜻이다. pass-SC 가 점유 지배적이라는 위 사실과
B5(edge head 가 GT 양성 2,931 중 1,812 만 선택) · valid_conn 0.367 을 함께 놓으면,
**"어느 ROI 쌍에 가닥을 놓는가" 가 틀렸다**는 가설이 선다. -> Wave 2 조사 대상.

### 주지표 재확정
전체 SC r 은 무정보 기준선이 이미 0.93~0.95 라 목표로 쓸 수 없다.
**주지표는 tier 내부 r 이고, 넘어야 할 선은 group 0.21/0.31/0.92, 천장은 0.61/0.83/1.00**
(29k 가닥 기준). 모델 현재값 0.164/0.189/0.624.

### 사용자 수용 기준(SC r >= 0.8)에 대한 함의
0.8 은 **남의 뇌를 그대로 쓴 값(0.841~0.891)보다 낮고**, 그룹 tractogram(0.914~0.949) 과
망친 tractogram(0.945) 보다도 낮다. 기준을 정할 때 이 기준선들이 아직 측정되지 않았다.
사용자에게 재검토를 제시한다 (결정은 사용자 몫).

---

## W1-d 결과 (2026-09-06) — **BN 재보정만으로 목표 달성. 용량 증설 불필요**
`outputs/eval/decoder_capacity.json` (8 arm, 3000 step)

### 9.17mm vs 3.55mm — 원인 두 개가 겹쳐 있었다
**(1) 스크립트가 잘못된 체크포인트를 잘못된 생성자로 로드했다.**
`scripts/38_decoder_capacity.py` 의 기본 `--ckpt` 가 `route2/s5_joint` 였다. 이 체크포인트는
`pair_emb.anatomy_norm`(LayerNorm) 이전이고, **1채널**이며, **syn** T1 으로 학습됐다.
그런데 스크립트가 `ROIPairATM(...)` 을 손으로 만들어(→ `in_channels=2` 기본, template 없음)
`--t1-mode rigid` 를 먹였다. 그 조합의 시작 recon 이 **117mm** 고, 1500 step 이 9.17 까지 끌어내린 것이다.
**9.17 은 애초에 디코더 성능이 아니었다.** (같은 수동 생성은 retrain 체크포인트에서는 template
buffer 때문에 `unexpected keys` 로 죽는다.)
-> `from_checkpoint()` 로 고치고 기본 ckpt 를 `retrain/p4_joint` 로, `--t1-mode` 가
`sd["t1_source"]` 와 일치하는지 assert 추가.

**(2) ConvVAE BatchNorm train/eval 격차** — W1-a 를 독립적으로 재확인. held-out eval 모드 **8.458mm**.
flip 은 무관 (전 arm 에서 flip-aware 와 1e-6 일치).

### BN 만으로 4mm 이하에 도달하는가 — **예**
| p4_joint, val 3명, z=mu | mm |
|---|---|
| eval 모드, 배포 상태 | **8.458** |
| **eval 모드, BN 재보정 (학습 0, 파라미터 0)** | **4.002** |
| train 모드 (batch 통계 — 착시의 원인) | 4.551 |

**재보정만으로 오차의 53% 가 사라진다.** recon 전용 3000 step 파인튜닝 후에는 전 arm 이
**2.8~3.1 mm**. 8.46mm 의 분해: **약 4.46mm 는 BN, 약 3.0mm 는 진짜 디코더 오차, 용량 문제는 0.**

### 원인 정정 — "pair 다발 배치" 가설은 틀렸다
recon 배치는 이미 **128 pair x 8 가닥** 이다 (`trainer.py:319-330`). 단일 pair 다발이 아니다.
실제 원인은 p4_joint 가 recon · segment(`mode=1`) · generation 을 **같은 BN 에 통과**시켜,
running 통계가 recon 시점에 맞지 않는 분포들의 평균이 된 것이다.

### 스윕 — refiner 는 쓰지 않는다
| arm | eval | extra params | sec |
|---|---|---|---|
| refine (h128 L3) | **2.815** | 582,019 | 196 |
| refine_deep (h128 L5) | 2.846 | 911,235 | 271 |
| bn_eval | 2.883 | 0 | 63 |
| lr3x | 2.890 | 0 | 61 |
| bn_recal | 2.920 | 0 | 60 |
| base | 3.076 | 0 | 65 |

refiner 는 **0.26mm 를 582k 파라미터로 산다.** 이미 목표(4mm)를 넘긴 지점에서다.
`d1_decoder.yaml` 은 `use_refiner: false`. 모듈은 0-init 스캐폴딩으로 남긴다.

### bit-exact 검증 통과 (tolerance 0)
`use_refiner=True` + `p4_joint_step3000.pt`: recon decode `max|diff| = 0` ([1024,128,3]),
`generate()` `max|diff| = 0` ([256,128,3]), NaN/Inf 0, refiner 35개 키가 `missing` 으로 0-init 유지,
`param_groups()` 키 불변. 30 step 후 delta 가 0 이 아님도 확인(gradient 흐름). pytest 21 passed.

### 적용 대기 patch (W1-d 가 보고만 함)
- **P1** `run.py` `PHASES`: `"d1_decoder": {"recon"},` 추가
- **P2** `trainer.py`: `TrainConfig.bn_mode: str = "train"`; `step()` 의 `m.train()`(158행) 뒤에
  `bn_mode != "train"` 이면 `m.atm.net.ae` 아래 모든 `BatchNorm1d` 를 `.eval()`
- **P3** `run.py`: `bn_mode == "recal_eval"` 일 때 루프 전 1회 BN 재보정
  (`scripts/38_decoder_capacity.py` 의 `bn_recalibrate()` 재사용)
- **P4** `run.py` `validate()`: `recon_rmse_eval_mm` (eval 모드, held-out) 방출.
  **현재 validate 는 recon 지표를 전혀 내지 않아 게이트가 발화할 수 없다.**
- **P5** refiner 를 켤 때만: `config.build()` 가 `use_refiner`/`refiner` 반환, `save()` meta 기록

### 지표 명명
`trainer.py:350` 의 `recon_rmse_mm` 은 **train 모드 값**이며 그렇게 라벨해야 한다.
이걸 "복원 오차" 로 보고한 것이 이 모든 혼선의 출발점이었다.

---

## W1-e prior 진단 (2026-09-06) — **다봉 확인, 그러나 최대 결함은 분산**
`outputs/eval/w1e_latent_modality.json` · `scripts/46_latent_modality.py` ·
`src/atm_sc/evaluation/prior_metrics.py` (`BASELINE` 상수에 기준값) ·
latent 캐시 `w1e_latent_cache.npz` · 검정 캐시 `w1e_modality_tests.json` (후보 채점 1분)

### 판정: 다봉이다
검정력 확인 구간(n>=200, 77 pair) **98.7%**, 전체 280 pair 97.9%. tier(0.96~0.99)·block(0.97~0.98)
어디서도 차이 없음 — 특정 부류가 아니라 **거의 모든 pair** 다.
성분 수: BIC 최소 K 중앙값 **7** (37%는 상한 8에 붙음), "BIC 개선 90% 달성 최소 K" 중앙값 **5**.
**2봉이 아니라 5봉 이상이다.**

**합성 대조군으로 검정력 보증**: 단봉 가우시안 거짓양성 **0/280**, t(df=3) 0.268,
PC1 방향 4시그마 2봉 0.987(n>=200), 2시그마 2봉 0.004. 실측 0.987 은 최악 거짓양성률(0.268)의 3.7배.
단 **무작위 64차원 방향** 4시그마 2봉은 검출력 0.014 (latent 유효차원 ~3) -> **98% 는 하한**이다.

### 최대 결함은 다봉성이 아니라 분산이다
| | 값 |
|---|---|
| pair 안 GT latent 표준편차 | **0.180** |
| prior 표준편차 | **1.0** (**5.6배 과대**) |
| `||mean_gt - mu_pair||` | **4.25** (`||z||` 8.0 대비) |
| 가닥별 posterior 표준편차 | 0.810 (mu 퍼짐 0.180 의 4.5배 -> **posterior 부분 붕괴**) |

### 오라클 사다리 (`precision_ratio`, 낮을수록 좋음. 동일 부분집합)
| 구성 | precision_ratio | MMD^2 |
|---|---|---|
| **현재 `N(mu_pair, I)`** | **38.5** | 1.825 (순열검정 p<=0.05 가 280/280) |
| pair별 대각 가우시안 (**분산 학습**) | **7.32** | 0.31 |
| mix2 | 5.08 | 0.145 |
| mix3 | 4.27 | 0.087 |
| **mix5** | **3.36** | 0.048 |
held-out 에서 K 를 늘릴수록 계속 개선 -> 과적합 아님. 지표 자기검증 19/19 통과.

### 권고 사다리
1. **분산 학습** — 38.5 -> 7.3 (**5.3배**, 최대 이득, 코드량 최소)
2. **pair 별 K~5 혼합** — 7.3 -> 3.4 (추가 2.2배)
3. flow/diffusion — K-혼합이 3.4 에서 막힐 때만
부수: 같은 pair 의 latent 평균 분산 중 **34% 가 subject 간 성분** (단 3명 이상 공유 pair 8개뿐, 예비값)
-> anatomy 조건화 통로는 만들되 0-init 으로 꺼둔다.

---

# 게이트 A 판정 (2026-09-06) — **통과. Wave 2 진행**

| 검증 | 결과 |
|---|---|
| W1-a flip | 무관 (뒤집힘 0.0%) — 대신 **BN eval 8.458mm** 발견 |
| W1-b 변위 가설 | **정량 확인** — sigma=8 이 오라클 0.167 과 모델 0.546 을 동시에 재현 |
| W1-c dice->SC | **0.85 통과** (group tractogram 0.914~0.949) — 단 전체 SC r 은 목표 지표로 부적합 |
| W1-d BN | **재보정만으로 8.458 -> 4.002mm** (파라미터 0). 파인튜닝 후 2.8~3.1. **refiner 불필요** |
| W1-e prior | **다봉 98.7%**, 그러나 최대 결함은 **분산 5.6배 과대** |

Wave 2 구성 (Wave 1 결과로 변경됨 — refiner 제외, 점유 조사 추가):
- **W2-a** (임계 경로): patch P1~P4 적용 + D1 recon 학습 + 오라클/생성 분리 측정
- **W2-b**: prior 사다리 (분산 학습 -> K~5 혼합), 캐시된 latent 로 채점(생성 안 함)
- **W2-c**: C10 점유 결함 조사 (-0.15 잔차의 점유 몫 vs 기하 몫 분리) + C9 지표 patch 보고

---

## W2-b prior 사다리 실측 (2026-09-06) — `outputs/eval/w2b_prior_ladder.json`
`scripts/48_prior_ladder.py` (약 20분, GPU 불필요). n>=200 77 pair, W1-e 와 **같은 held-out 절반**.

| prior | precision_ratio | MMD^2 | NLL | 오라클 대비 |
|---|---|---|---|---|
| `current` N(mu_pair, I) | **36.61** | 1.861 | 68.3 | 재현 (delta<1e-3) |
| `kl_fit_sigma` (KL 이 실제로 몰고 갈 분산) | **38.74** | 1.860 | 69.4 | - |
| `s1_var_fixed_mu` (분산만) | 22.47 | 1.969 | -13.5 | - |
| `s1_mean_only` (평균만) | 34.36 | 1.381 | 61.3 | - |
| **`s1_full`** (평균+분산) | **7.30** | 0.308 | -54.8 | 7.32 (0.997) |
| `s2_mix2/3/5` | 5.70 / 4.58 / 3.80 | 0.070 | -83.1 | 5.08/4.27/3.36 |
| **`arch_additive`** (**처음 보는 pair 로 일반화**) | **15.60** | 2.35 | 10.7 | - |
| `arch_additive_mix5` | 14.40 | 2.60 | -1.7 | - |

### 정정 1 — 오라클 사다리는 조건별 적합이라 학습으로 실현 불가
W1-e 의 7.32 -> 3.36 은 **pair 마다 따로 적합**한 값이다. 처음 보는 pair 로 일반화하면 **15.60**.
즉 현실적 이득은 36.61 -> 15.60 (**2.3배**) 이지 11배가 아니다.

### 정정 2 — 1단계 이득은 "분산" 이 아니라 평균과의 상호작용
분산만 = 1.63배(36.6->22.5), 평균만 = 1.07배, **둘을 같이 고쳐야 7.30**.
W1-e 의 "최대 결함은 분산" 서술은 절반만 맞다.

### 정정 3 — **2단계(K-혼합)에서 멈춘다**
일반화 수준에서 혼합은 거의 안 붙는다 (15.60 -> 14.40). **평균이 틀린 채로 모드를 얹어도 의미가 없다.**
3단계(flow/diffusion)는 명백히 시기상조.

### C11 (신규, 大) — **KL 로는 prior 분산을 학습할 수 없다 (실측)**
KL(q||p) 의 p 최적해는 aggregate posterior 모멘트 정합이다. 평균이 `mu_pair` 에 묶인 현 구조에서
**시그마* = 1.074** (중앙값) — **지금 값 1.0 이 이미 KL 최적점이다.** 실제로 그 시그마로 채점하면
36.61 -> **38.74 로 나빠진다**. 평균이 완벽해도 시그마* = 0.915 (E[var_q]=0.81^2 가 지배). 목표는 0.279.

-> prior 분산은 KL 이 아니라 **detach 한 posterior 평균에 대한 별도 적합항**으로 학습해야 한다:
`-lp.diag_log_prob(mu.detach(), mu_p, ls_p).mean()`, KL 에 넣는 prior 파라미터는 detach.

### C12 (신규, 大) — posterior 부분 붕괴가 정량적 상한이다
posterior 반경 **6.96** vs mu 퍼짐 2.24, **모드 간격 1.63 (간격/잡음 = 0.234)**.
2단계가 노리는 구조가 **decoder 가 학습 중 통과시킨 잡음보다 4배 작다.**

z_gt 를 posterior **샘플**(decoder 가 실제로 본 z)로 바꿔 같은 사다리를 재면 `current` 가 이미
**1.19**, `s1_full`/`mix5` 는 0.70~0.73 으로 **과소분산**이 된다.
-> **사다리 위쪽 이득이 생성으로 옮겨간다는 보장이 없다.** posterior 붕괴를 먼저 고쳐야 한다.

### anatomy — 근거는 강해졌으나 상한이 있다
pair 만 보는 prior 의 **원리적** 평균 오차 하한 **2.59** (subject 성분 0.169, 반복 pair 26개)
vs pair 안 GT 구름 반경 2.24. 현재 4.64, 가법 ROI 모형 held-out 4.26 (R^2=0.472,
현 `prior_mu` R^2=0.406 -> **이미 구조 한계 근처**). 데이터를 늘려도 4.26 -> 2.59 까지고
구름 안으로는 못 들어간다. `prior_use_anatomy=False`, 0-init 로 꺼 둠.

### bit-exact 검증 (`scripts/48_prior_ladder.py:selfcheck`, 매 실행)
`log_sigma` 0-init 시 `prior_params` 의 mu 가 기존 `prior_mean` 과 `torch.equal`,
`sample_prior` 가 `prior_mean+randn` 과 `torch.equal`, `PairMixturePrior(k=1)` 이 대각과 동일(<1e-4),
새 KL 식이 `logvar_prior=0` 에서 기존 식과 일치. 기존 테스트 16개 통과.

### 적용 대기 patch (W2-b 가 보고만 함)
- `src/atm_sc/losses/geometry.py:46` `kl_loss` 에 `logvar_prior=None` 인자 추가 (None 이면 기존 경로 bit-exact)
- `trainer.py:337,378` 호출부: `mu_p, ls_p = m.pair_emb.prior_params(canonical_pairs(P_gt))` 후
  `L.kl_loss(mu, logvar, mu_p, 2*ls_p)` + prior 파라미터 detach + 별도 적합항
- `roi_atm.py:330` `sample_z` -> `self.pair_emb.sample_prior(canonical_pairs(pairs_or_n), generator=generator)`

---

## W2-c 점유 결함 조사 (2026-09-06) — **C10 가설 기각. 잔차는 전부 기하다**
`scripts/47_occupancy_diag.py` · `outputs/eval/w2c_occupancy.json` (test 31명) ·
`outputs/eval/w2c_c9_pair_dice_npred.json`

### 2x2 — 기하와 점유를 교차 교체 (쌍당 16 가닥, pass-SC, GT=.mat, 31명 평균)
| | 모델 쌍 1812 | GT 쌍 1914 |
|---|---|---|
| **모델 기하** | **0.711** (보고값 0.713 재현) | 0.690 |
| **GT 기하** | **0.875** | 0.836 |

- **기하 교체 이득 +0.164 / +0.146**
- **점유 교체 이득 −0.021 / −0.038 (음수)**

GT 기하를 모델 자신의 쌍 집합에 꽂으면 **0.875** — dice 회귀 예측 0.865 와 일치한다.
**즉 −0.15 잔차는 전부 기하이고 점유 몫은 0 이다. C10 가설 기각.**

마스크 검증도 같다: `r(GT를 모델 점유로 가린 것, GT) = 0.9997` (점유 비용 0.0003),
`r(모델을 GT 지지로 가린 것) = 0.712` (FP 비용 0). 놓친 GT 총량 0.43%, FP 위 예측 질량 1.1%.

기전: 모델 가닥은 하나당 ROI **5.56개**(쌍 15.0)를 지나는데 GT 는 **6.87개**(쌍 26.6).
길이는 더 긴데 통과가 부족하다.

### B5 정정 — "2,931 중 1,812" 은 비교 대상이 틀렸다
recall **0.936** (TP 2699 / GT 2885), precision 0.897. tier recall small 0.861 · mid 0.949 · large **0.993**.
FN 185.8 = (a) 미선택 164.4 + (b) 선택했으나 미도달 21.4. 그런데 (a) 중 **107.5 는 edge head
학습 타깃(`bundles.npz` endpoint 쌍)에 애초에 없던 pass-only 쌍**이고 진짜 놓친 것은 **56.8** 뿐이다.
edge head 는 endpoint 양성(평균 1,914)으로 학습되므로 **1,812 는 자기 타깃의 95%** 다.
**B5 는 기각한다.** FN 은 약하고 길다 (GT 중앙값 25 vs TP 482, 201mm vs 156mm, sub-sub 0개).

### edge_thr 스윕 — 낮추면 나빠지고 올리면 좋아진다 (단 지표 놀이다)
| thr | 0.99 | **0.95** | 0.9 | 0.7 | **0.5(현재)** | 0.3 | 0.0 |
|---|---|---|---|---|---|---|---|
| 전체 SC r | 0.777 | **0.780** | 0.768 | 0.743 | **0.711** | 0.674 | 0.625 |

thr 0.95 면 **재학습 없이 +0.069** (0.711 -> 0.780), 쌍 1812 -> 617, 가닥 29k -> 9.9k (3배 저렴).
**그러나 이득은 large tier(0.622 -> 0.710)에서만 나고 주지표인 small tier 는 0.164 -> 0.120 으로 악화한다.**
전체 r 은 진단값이므로 이 설정 변경은 **"지표 올리기" 이지 성능 개선이 아니다.** 채택하지 않는다.

### 부수 검증 — W1-c 결론의 범위 한정
GT 가닥 + 자연 개수 배분(같은 30.6k) = **0.938** -> W1-c 의 pairs-family self 0.936 과 일치(구현 검증).
동시에 균일-16 규약 자체가 0.938 -> 0.836 으로 0.10 을 깎는다.
-> **"개수 배분은 무관" 이라는 W1-c 결론은 raw family 한정이다.**

### C9 해소안 — pair dice 규약이 두 군데서 어긋나 있다
실측 dice 최대 상승폭(iid 변위): n8 **+0.097** · n16 **+0.066@sigma=1.0** · n32 +0.029 ·
**n64 0.000(단조)** · n128 0.000. 크기를 맞춘 규약(pred=n vs 겹치지 않는 GT n)은 n16 부터 단조(천장 0.600),
n64 천장 0.742.

**추가 결함**: `scripts/29_final_evaluation.py` L278 의 모델 값은 **16 vs 전체(<=256)**, L286 의
"천장" 은 **128 vs 128** 로 **서로 다른 규약**이다.
-> **기존에 인용해 온 "pair dice 0.095 / 천장 0.700" 비율은 해석 불가다.**

권고: 새 인자 `--trk-pair-n` (기본 **64**) 를 두고 평가 대상 <=50 쌍만 따로 생성(+3,200 가닥, ~0.1s),
pred·gt·천장을 **같은 크기**로 맞춘다. **전역 `--n-per-pair` 는 건드리지 않는다** — 16->64 로 올리면
whole-brain 가닥이 29k->116k 가 되어 `generated` SC 전부가 과거 실행·W1-c 기준선과 비교 불가능해진다.
규약이 바뀌므로 **이전 pair dice 0.0946 과 직접 비교 금지**, 새 천장과 함께 재보고한다.

---

## W2-a D1 결과 (2026-09-07) — `outputs/eval/w2a_d1_result.json`
학습 6,000 step / 144명 / **510초**. 체크포인트 `outputs/checkpoints/retrain/d1_decoder/d1_decoder_step6000.pt`.
patch P1~P4 적용 (`run.py`, `trainer.py`, `19_train_pipeline.py`, `29_final_evaluation.py`), `pytest tests/ -q` **144 passed**.

### 학습 전 -> 후 (test 31명, 양쪽 동일 코드)
| 지표 | 전 | 후 |
|---|---|---|
| recon RMSE (eval, val 3) | 8.755 | **2.946** |
| recon RMSE (eval, test 31) | 7.721 | **2.240** |
| **오라클** pair dice (`pair_half`, 천장 0.711) | 0.328 | **0.590** |
| **오라클** wb dice (`wb_disjoint`, 천장 0.788) | 0.589 | **0.718** |
| **생성** pair dice | 0.0952 | **0.0727** |
| **생성** wb dice | 0.608 | **0.583** |
| **생성** SC pass r | 0.7126 | **0.7314** |
| SC tier r (small/mid/large) | 0.164/0.189/0.624 | 0.158/0.188/**0.647** |
| valid_conn / endpoint_in_roi | 0.367 / 0.745 | **0.042 / 0.555** |

### 게이트 B: recon **통과** (2.946 <= 4.0). 그리고 W1-b 보정 곡선이 정확히 맞았다
RMSE 2.240 에서 W1-b 가 예측한 `wb_disjoint` 0.718 (smooth) — **실측 0.718, 정확히 일치.**
`pair_half` 예측 0.552 vs 실측 0.590. 오라클 경로에서 보정 곡선은 검증됐다.

### C13 (신규, 치명) — **생성 경로는 오히려 나빠졌다**
`valid_conn` 0.367 -> **0.042**, `endpoint_in_roi` 0.745 -> 0.555, pair dice 0.095 -> 0.073.

원인 (`outputs/eval/w2a_latent_drift.json`): posterior 의 `prior_mean` 대비 오프셋은 오히려
**줄었다 (5.27 -> 3.51, sd 0.80 동일)** -> **posterior drift 가 아니다.**
recon 전용 학습이 생성 경로의 제약(endpoint/route/SC)을 **전부 떼어낸 채** 디코더만 날카롭게 만들어서,
**prior 가 뽑는 영역이 더 이상 디코더가 정확한 영역이 아니게 됐다.**

**교훈: 순차 단계 학습이 또 서로를 깎았다** (p2->p3->p4 에서 resid_r 이 0.124 -> -0.031 로 무너진 것과
같은 유형). **D1 을 단독으로 출하할 수 없다.** Wave 3 은 recon 을 생성 제약과 **함께** 도는
joint 학습이어야 한다.

### 재현되지 않는 기존 참조값 2건 (인용 금지)
1. **"오라클 pair dice 0.167"** — **기록된 규약이 없다.** 같은 p4_joint 체크포인트에서
   0.328 (`pair_half`) / 0.370 (`pair_self`) 로 측정된다.
2. **"생성 wb dice 0.546"** — `--n-per-pair 8` 로 잰 값이다. 기본값 16 에서는 **0.608**.
   -> **모델 0.546 vs cross_subject 0.574 비교가 규약 불일치였다.** 기준선을 맞춘 규약으로
   다시 재야 한다.

### 절차 사고 (에이전트가 자진 보고)
`outputs/eval/final_p4_joint_step3000.json` 을 초기에 2명 smoke 결과로 덮어썼다가 31명으로 재생성했다.
역사적 기준선을 재현함(SC r 0.7126, tier 0.164/0.189/0.624, pair dice 0.0952, valid_conn 0.367).
사본 `outputs/eval/w2a_before_p4_joint_step3000.json`.

---

## W3-b 규약 정리 + 기준선 재측정 (2026-09-07)
`outputs/eval/w3b_protocol.json` (규약 정의) · `w3b_baselines_matched.json` ·
`w3b_geom_n{8000,20000}.json` · `w3b_sc_n{8000,20000}.json` · 재현 `python scripts/51_w3b_protocol_report.py`

### C9 해소 — `scripts/29_final_evaluation.py`
새 인자 `--trk-pair-n` (기본 **64**) + `--trk-pair-seed`. 평가 대상 <=50 쌍을 따로 생성(+3,200 가닥),
**pred = gtA = gtB = n** (`n = min(trk_pair_n, generated, len(gt)//2)`), GT 를 겹치지 않는 두 조각으로
나눠 천장 = `gtB vs gtA`. 쌍마다 assert. **전역 `--n-per-pair` 는 불변.**
`summary.protocol` 블록에 모든 설정을 기록한다.

### 규약을 맞춘 대응표 (`wb@N` = n_pred == n_gt == N, `pair@64`)
| | wb@8000 | wb@20000 | pair@64 | 천장 | SC r |
|---|---|---|---|---|---|
| p4_joint n8 | **0.5465** | - | 0.135 | 0.630 | 0.711 |
| p4_joint n16 | **0.5457** | **0.6079** | 0.135 | 0.630 | 0.713 |
| d1_decoder n16 | 0.5244 | 0.5832 | 0.098 | 0.630 | 0.731 |
| **cross_subject** | **0.5738 ±0.016** | **0.6289 ±0.016** | **0.222** | 0.623 | 0.841 / 0.843 |
| group_tractogram | 0.6141 | 0.6697 | 0.283 | - | 0.914 / 0.918 |
| self (천장) | 0.7831 | 0.8437 | 0.619 | - | 0.936 |

### 판정: **모델은 cross_subject 를 넘지 못한다. 기존 판정 유지** (0/8 구성, 마진 −0.021 ~ −0.050)

**W2-a 의 진단 2건이 정정된다:**
1. `--n-per-pair` 8 vs 16 은 wb dice 를 **0.0008** 밖에 안 바꾼다 (0.5465 vs 0.5457).
   **0.546 과 0.608 을 가른 것은 n-per-pair 가 아니다.**
2. 진짜 변수는 **채점 표본 크기**다: 같은 체크포인트에서 wb@8000 -> 0.546, wb@20000 -> 0.608.
   그런데 **기준선도 같이 오른다** (0.574 -> 0.629). **격차는 그대로다.**
3. SC r 은 이 축에 거의 무감각하다 (cross_subject 0.8408@8k -> 0.8434@20k).
   **SC 판정은 애초에 규약 의존이 아니었다.**

새 pair dice: **0.135, 천장 0.630** (천장의 21%). 여전히 남의 뇌의 같은 쌍 다발(0.222)보다 낮다.
규약 일치는 독립 확인됨 (`29` 의 천장 0.630 vs `43` 의 0.623).

### W3-a 중간 체크포인트 `d3_joint_step3000` (W3-b 측정)
wb@8000 **0.4957** / wb@20000 0.5529 / pair@64 **0.0869** / **SC r 0.7881**
(tier small 0.171 · mid 0.196 · large 0.732).
**기하는 p4_joint 대비 더 나빠지고 SC r 은 올랐다.** W3-a 자체 보고 대기.

### 절차 사고 — 병렬 에이전트 간 산출물 경로 충돌
W3-b 의 `29_final_evaluation.py` 실행이 `outputs/eval/final_d3_joint_step3000.json` 을 덮어썼다
(W3-a 의 `49_eval_frozen.py` 와 **같은 출력 경로**). W3-a 의 `w3a_joint_result.json` 은 무사하나
원본 JSON 은 재생성 필요. 또 W3-a 의 전후 표는 **양쪽 다 옛 pair 규약**을 쓴다.
-> **파일 소유권 규칙이 소스 파일만 다루고 출력 경로를 다루지 않은 것이 원인.** 다음부터 산출물
경로도 에이전트별로 분리한다.

---

## W3-a joint 학습 결과 (2026-09-07) — **게이트 미달**
`outputs/eval/w3a_joint_result.json` · `configs/retrain/d3_joint.yaml` ·
`outputs/checkpoints/retrain/d3_joint/d3_joint_step3000.pt`
전후 비교는 `scripts/49_eval_frozen.py` (29번의 00:19 스냅샷, md5 2e8c5169) 로 양쪽 동일 규약.

### patch 적용
`kl_loss(logvar_prior)`, `sample_z -> sample_prior`, trainer 의 **detach 된 KL + 별도 prior 적합항**
`-log N(mu_q.detach(); mu_p, sigma_p)`. 0-init bit-exact 전부 통과, KL->prior gradient 없음 /
적합항->prior gradient 있음 확인.

**추가 발견**: `prior_scale` LR 그룹이 필요했다. Adam 보폭은 손실 가중치가 아니라 **LR** 로 정해져서
`lr_heads`(3.3e-5)로는 log sigma 가 1.25e-4/step, 목표까지 4,800 step 이 걸린다 -> `lr_prior=3e-4`.
기존 config 는 `lr_prior=None` -> `lr_heads` 로 동작 불변.

### 가중치 탐색 (각 500 step)
A(p4 그대로) recon 3.515 · vc 0.089 / B(recon 3.0) 3.066 · 0.055 / C(seg 0.25) 3.866 · 0.103.
**단조 상충**이라 중간인 A 선택.

### 결과
| 지표 | d1_decoder | **d3_joint 3k** | 6k |
|---|---|---|---|
| recon RMSE (eval, test) | 2.240 | 3.401 | 3.481 |
| 오라클 wb / pair dice | 0.718 / 0.590 | 0.680 / 0.524 | 0.678 / 0.519 |
| 생성 wb / pair dice (wb@20000) | 0.583 / 0.0727 | 0.553 / 0.0678 | 0.502 / 0.049 |
| **valid_conn / endpoint_in_roi** | 0.042 / 0.555 | **0.239 / 0.681** | 0.284 / 0.684 |
| 생성 SC pass r | 0.7314 | **0.7881** | 0.765 |
| SC tier r (s/m/l) | .158/.188/.647 | **.171/.196/.732** | .158/.177/.711 |
| prior precision_ratio | 45.09 | **32.97** | 40.68 |

### 판정: **미달**
`valid_conn` 은 0.042 -> 0.239 로 **5.7배 회복**했으나 기준 0.367 에 못 미치고, 생성 wb/pair dice 가
오히려 소폭 하락해 기준 두 개를 놓쳤다. recon 게이트(<=4.0)와 SC 는 통과·개선.
6k 는 valid_conn 만 오르고 기하가 무너져 **3k 가 최적점**이다.

### prior — 예측 15.60 미달 (실측 32.97)
같은 스크립트가 p4_joint 36.61 을 재현하므로 규약은 맞다. 이득은 **전부 sigma 에서** 왔고,
joint 학습이 posterior 평균을 `mu_pair` 에서 더 밀어내(offset 3.41 -> 5.8) mu 쪽 이득을 상쇄했다.

**새 사실: D1 이 prior 도 망가뜨렸다** (p4_joint 36.61 -> d1_decoder 45.09). C13 의 또 다른 얼굴이다.

### C14 (신규, 大) — **생성 경로의 pair 단위 기하를 겨냥하는 손실이 없다**
endpoint 는 맞아 가는데(endpoint_in_roi 0.555 -> 0.681) 경로가 GT 번들에서 벗어난다
(dice 정체, **mdf 18.3 -> 19.5**).

현재 손실 구성: recon(=**posterior** 경로, GT z 를 준 상태) + endpoint + route + SC + prior 적합항.
**어느 것도 "prior 에서 뽑아 만든 번들이 GT 번들과 겹치는가" 를 직접 겨냥하지 않는다.**
endpoint/route/SC 는 전부 거친 지표(어느 ROI 인가, 몇 개인가)다.

-> 다음 단계는 **생성(샘플) 경로 위의 번들 단위 기하 손실** (Chamfer/MDF 등, 재파라미터화로 미분 가능).

---

# Wave 1~3 종합 (2026-09-07)

## 규약 일치 최종 비교
| | wb@8000 | wb@20000 | pair@64 (천장 0.630) | valid_conn | SC r |
|---|---|---|---|---|---|
| **p4_joint (출발점)** | **0.5457** | **0.6079** | **0.135** | **0.367** | 0.713 |
| d1_decoder | 0.5244 | 0.5832 | 0.098 | 0.042 | 0.731 |
| d3_joint 3k | 0.4957 | 0.5529 | 0.0869 | 0.239 | **0.7881** |
| **cross_subject** | **0.5738** | **0.6289** | **0.222** | - | 0.841 |
| group_tractogram | 0.6141 | 0.6697 | 0.283 | - | 0.914 |
| self (천장) | 0.7831 | 0.8437 | 0.619 | - | 0.936 |

**생성 기하에서 출발점 p4_joint 를 넘은 구성이 없다.** SC r 만 0.713 -> 0.7881 로 올랐다.
모든 구성이 cross_subject 미달.

## 그래도 확보한 것
1. **디코더 복원**: eval 7.721 -> **2.240 mm**. 오라클 wb dice 0.589 -> **0.718** (천장 0.788).
   보정 곡선 예측(0.718)과 **정확히 일치**.
2. **SC r** 0.713 -> **0.7881**.
3. **RMSE -> dice 보정 곡선** — 이제 목표 RMSE 를 역산할 수 있다.
4. **규약 고정** (`w3b_protocol.json`) + **게이트 강제** + 500-step 개인차 로깅.
5. 무효 판명된 참조값 정리: 9.17mm · 3.55mm · 오라클 0.167 · wb dice 0.546 규약 · pair 천장 0.700 ·
   B5 분모 · C10 점유 가설.

## 남은 단일 병목
**C14** — 생성 경로 위 pair 단위 기하 손실의 부재. 오라클 경로는 천장의 91%(0.718/0.788)에
도달했으나 생성 경로는 88%(0.553/0.629 cross_subject 기준) 수준에 머문다. 그 격차는
"디코더가 못 그린다" 가 아니라 **"prior 에서 뽑은 z 가 GT 번들 위치가 아니고, 그걸 벌하는 항이 없다"** 다.
