# PIPELINE_11 — 코드 수정 파이프라인 (D0~D3)

작성 2026-09-06. `PIPELINE_09_PROBLEM_INVENTORY.md` 의 측정 결과 위에 세운다.

## 목적과 상한

**목적**: T1 단독 whole-brain tractogram 생성 품질을 **그룹 수준**까지 올린다.

개인차 축은 S1-b 사전등록 판정으로 닫혔다 — ROI 국소 풀링이 전뇌 풀링보다 나쁘고
(0.063 < 0.068), 뇌 부피 스칼라 하나(+0.102)를 못 넘는다. 잔여 신호는 **머리 크기**다.

**핵심 논리**: 그룹 tractogram 은 dice 0.614, 그 SC 는 개인 GT SC 와 r 0.945 다.
모델은 dice 0.546 / SC r 0.713 이다. 즉 **0.713 -> 0.9 대의 격차는 개인차 부재가 아니라
기하 품질 때문**이다. 개인차가 필요한 구간은 0.945 초과부터고 거긴 닫혔다.

| 지표 | 현재 | 목표 (기하 개선만) | 개인차 필요 |
|---|---|---|---|
| whole-brain trk dice | 0.546 | **> 0.614** | 0.783 (천장) |
| pair dice | 0.095 | ~0.18 | 0.704 |
| SC pass r | 0.713 | **>= 0.80** (수용 기준) | > 0.945 |
| SC tier small | 0.164 | > 0.222 | - |
| edge 길이 상관 | 0.182 | > 0.63 | 0.909 |

## 실행 환경 (2026-09-06 실측)

| 자원 | 상태 | 용도 |
|---|---|---|
| 로컬 A10 23GB | **비어 있음** (855 MiB) | 개발·짧은 검증·CPU 불가 작업 |
| `base_g` (H100) | walltime 24h, **대기 24개**. 잡 82900 은 10시간째 Q | 긴 학습만 |
| `base_8` (CPU 48h) | 사용 가능 | 전처리·대량 CPU |
| `std_q`/`base_32` | 권한 없음 | - |

**원칙: H100 을 블로킹하지 않는다.** 개발과 판정은 로컬 A10 에서 하고, 긴 학습만 제출한다.
24h walltime 이므로 모든 학습은 checkpoint/resume 가능해야 한다 (`pipeline_state.json` +
`*_latest.pt` 기존 기구 사용).

### 잡 82900 을 버리지 않고 재활용한다
`scripts/pbs_route.sh` 는 실행 시점에 `python scripts/19_train_pipeline.py --config
configs/pipeline_route.yaml` 을 돈다. **config 파일은 런타임에 읽힌다** — 즉 큐 순번(10시간치)을
잃지 않고 **내용만 바꿀 수 있다.** D1-d 가 준비되면 `configs/pipeline_route.yaml` 을 디코더
파인튜닝으로 재지정한다. 그 전에 잡이 잡히면 폐기된 route 파이프라인이 24h 를 낭비하므로,
**D1-d 를 최우선으로 만든다.**

---

## D0 — 목표 곡선 확정 (0.5일, CPU, GPU 불필요)

"디코더를 고치면 dice 가 오르는가" 를 추측이 아니라 곡선으로 만든다. **가장 싸고 가장 먼저.**

| 단계 | 파일 | 변경 |
|---|---|---|
| D0-a | 신규 `scripts/44_recon_flip_check.py` | flip 처리 유무별 recon RMSE 비교 |
| D0-b | 신규 `scripts/45_dice_displacement.py` | GT streamline 을 알려진 sigma 로 흔들어 dice 곡선 |
| D0-c | `scripts/43_geometry_baselines.py` 확장 | 그룹 tractogram 의 SC 계산 -> 개인 GT SC 와 r |

**D0-a 배경**: 손실 `stream_recon_loss` 는 방향 모호성을 처리한다 (`mm_gt.flip(1)` 중 작은 쪽).
그런데 보고 지표 `trainer.py:350` 에는 flip 처리가 없다. 손실이 뒤집힌 쪽을 학습시키면 지표만
큰 값을 보고한다. **3.55mm 가 과대평가일 수 있다.**
-> 차이가 0.5mm 초과면 `trainer.py:350` 을 flip-aware 로 고친다.

**D0-b 산출물**: "dice X 를 원하면 RMSE <= Y mm" 표. 이후 모든 디코더 작업의 목표가 여기서 나온다.

**D0-c 배경**: 위 "핵심 논리" 의 dice -> SC r 연결을 실증한다. 그룹 tractogram(dice 0.614)의
SC 가 실제로 r 0.9 대를 주는지 확인.

### 게이트 D0
- D0-b 에서 dice 가 변위에 지배되지 **않으면** -> D1 의 근거가 무너진다. **중단하고 재설계.**
- D0-c 에서 dice 0.614 짜리 tractogram 의 SC r 이 0.85 미만이면 -> SC 목표를 하향한다.

---

## D1 — 디코더 충실도 (2~3일, A10 개발 + H100 학습)

가장 큰 격차: 오라클 latent pair dice 0.167 vs 천장 0.704. 복원 오차 3.55mm, 점 간격 약 1.24mm
(128점, 길이 83mm), 아틀라스 복셀 2mm.

**조건화·개인차·SC 가 전혀 안 끼는 순수 autoencoding 문제다.** 정답을 넣고 정답을 복원한다.

| 단계 | 파일 | 변경 |
|---|---|---|
| D1-a | `scripts/38_decoder_capacity.py` | 스윕 완성. `--only base` 로 arm 하나만 돌고 끝났다 (arm 당 29초). refine/hidden/layers 축을 실제로 쓴다 |
| D1-b | 신규 `src/atm_sc/models/streamline_refiner.py` | 동결 디코더 출력에 얹는 refinement 망 |
| D1-c | `src/atm_sc/models/roi_atm.py` | refiner 를 decode 경로에 연결. **0-init -> 시작 시 기존 체크포인트와 bit-exact** |
| D1-d | 신규 `configs/retrain/d1_decoder.yaml` | 순수 복원 손실만. 조건화·SC 손실 없음 |
| D1-e | `scripts/29_final_evaluation.py` | 오라클 latent 경로 평가 추가 (GT posterior z -> decode -> dice) |

**upstream 무수정 제약과 충돌하지 않는다** — `38_decoder_capacity.py` 가 이미 "동결 디코더 위에
refinement 를 얹는" 구조(`refine`/`hidden`/`layers`/`extra_params`)로 설계돼 있다. 그 설계를 따른다.

### 게이트 D1
- recon RMSE <= D0-b 가 준 목표값 (flip-aware 기준)
- **오라클 latent pair dice 가 실제로 상승** (0.167 -> ?). 안 오르면 D0-b 해석이 틀린 것이니 중단.
- 기존 체크포인트 대비 bit-exact 시작 확인 (0-init)

---

## D2 — prior (3~5일, H100)

현재 `roi_pair_embedding.py:78`:
```python
def prior_mean(self, pairs, mode=0):
    return self.prior_mu(self.pair_vec(pairs)) + self.mode_prior(m)
```
`z ~ N(mu(pair), I)` — **단봉이고 anatomy 를 아예 안 본다.** 인자가 pairs 와 mode 뿐이다.

한 ROI 쌍의 GT 다발은 여러 갈래(다봉)인데 단봉 가우시안 평균에서 뽑으면 어느 갈래도 아닌
가운데가 나온다. prior latent dice 0.053 이 이것으로 설명된다.

| 단계 | 파일 | 변경 |
|---|---|---|
| D2-a | 신규 `scripts/46_latent_modality.py` | (subject, pair) GT 다발을 인코딩해 latent 분포 다봉성 검정 |
| D2-b | 신규 `src/atm_sc/models/latent_prior.py` | 조건부 flow 또는 소형 diffusion. z 가 저차원이라 싸다 |
| D2-c | `src/atm_sc/models/roi_pair_embedding.py`, `roi_atm.py` | prior 입력에 anatomy 추가. 0-init 로 기존 동작 보존 |
| D2-d | 신규 `configs/retrain/d2_prior.yaml` | prior 학습 phase (디코더 동결) |

### 게이트 D2
- **D2-a 가 단봉으로 나오면 이 처방은 폐기다.** D1 이득만 안고 D3 로 간다.
- pair dice 가 오라클 수준(D1 후 갱신값)에 접근하는가

---

## D3 — 통합 재학습 (1.5일, H100)

| 단계 | 파일 | 변경 |
|---|---|---|
| D3-a | 신규 `configs/pipeline_d3.yaml` | **phase 단순화.** 지금 5 phase 는 서로를 깎는다 (`resid_r` 0.124 -> -0.031, 길이 99 -> 90). 상충 손실을 순차로 걸지 않는다 |
| D3-b | 신규 `scripts/pbs_d3.sh` | base_g 제출 + preempt 핸드셰이크 (`pbs_route.sh` 패턴 재사용) + 24h walltime 재개 |

실측 2h10/phase (A10, 3000 step). H100 이면 대략 3~4배 빠를 것으로 보나 미측정 — D1 첫 학습에서 잰다.

### 게이트 D3 (`configs` 의 gates 블록에 건다)
| 지표 | 통과 |
|---|---|
| whole-brain trk dice | **> 0.614** (그룹 tractogram) |
| SC pass r (중앙값) | **>= 0.80** |
| SC tier small / mid | > 0.222 / > 0.323 |
| edge 길이 상관 | > 0.63 |

---

## 견적

| 단계 | GPU | 달력 | 확신도 |
|---|---|---|---|
| D0 | 0h | 0.5일 | 높음 (계측만) |
| D1 | ~10h | 2~3일 | **높음** — 목적함수가 깨끗하고 다발 478k 개 |
| D2 | ~10h | 3~5일 | 중간 — 생성 모델링 |
| D3 | ~12h | 1.5일 | 높음이 아니라 D1/D2 결과에 종속 |
| **합계** | **~32h** | **7~10 작업일** | |

H100 이면 GPU 시간은 줄지만 **대기 24개라 달력 시간은 크게 안 줄 수 있다.**
로컬 A10 개발 + H100 제출 병행이 실질적 최선이다.

## 하지 않을 것
| | 이유 |
|---|---|
| S2/S3 ROI 국소 조건화 | S1-b 사전등록 판정 음성 |
| step 증가 | 전 지표가 p1~p3 에 포화, route_f1 은 9,000 step 미동 |
| weight head 부활 | GT 에 대응 타깃 없음, 수치도 동일 |
| SC 전체 상관을 주지표로 | 그룹 구조에 지배됨 — tier 내부 r 로 본다 |
| SyN 재계산 (7.4 core-h) | S1-b 가 SyN 쪽도 ROI 풀링이 나쁘다고 확인 |

---

## 수용 기준 (2026-09-06 사용자 결정)

**subject 별 SC 상관 0.8 이상이면 수용한다. 개인 간 구별성은 요구하지 않는다.**

현재 위치 (test 31명, T1 단독 generated): 평균 0.713 · 중앙값 0.712 · IQR 0.697~0.734 ·
최소 0.663 · **최대 0.754** — 아직 0.8 에 도달한 subject 가 없다.

### D3 게이트 반영
| 지표 | 통과 | 함께 보고 |
|---|---|---|
| **SC pass r** | **중앙값 >= 0.80** | 최소값, 0.8 이상 subject 비율 |
| whole-brain trk dice | > 0.614 | cross_subject 0.574 대비 |
| SC tier small / mid | > 0.222 / > 0.323 | |
| edge 길이 상관 | > 0.63 | |

"전원 0.8 이상" 은 최저값 기준 +0.14 가 필요해 훨씬 어렵다. 중앙값 기준과 구분해 보고한다.

### D2 가 선택지가 된다
D1(디코더)만으로 중앙값 0.80 을 넘으면 **prior 작업(D2)을 건너뛴다.** Wave 2 끝에서 판정한다.
계획을 크게 줄일 수 있는 분기점이다.

### 단, tractogram 게이트는 유지한다
subject 별 SC r >= 0.8 은 **모든 사람에게 동일한 행렬을 내놔도 달성된다** (그룹 템플릿이 전원에게
0.945). SC 만 게이트로 두면 "아틀라스 재생" 과 "생성 모델" 이 구분되지 않는다. 산출물이
tractogram 이므로 dice 기준(cross_subject 0.574 / group_tractogram 0.614)을 품질 게이트로 두고,
SC 0.80 은 "생성된 tractogram 이 해부학적으로 타당한가" 의 검증으로 쓴다.
