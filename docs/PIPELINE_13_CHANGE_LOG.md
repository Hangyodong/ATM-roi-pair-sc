# 수정 이력과 그 결과 (2026-09-07)

**이 프로젝트 요약은 이 문서와 `PIPELINE_12_PROBLEM_SUMMARY.md` 두 개만 읽으면 된다.**
- **이 문서** = 무엇을 어떻게 고쳤고, 고친 결과 숫자가 어떻게 변했나.
- `PIPELINE_12` = 어떤 문제가 남았고, 어떤 실험을 해서 무엇이 나왔나.

근거: `outputs/eval/final_{p4_joint_step3000,d1_decoder_step6000,d3_joint_step3000,d3_joint_step6000}.json`,
`w2b_prior_ladder.json`, `w3a_joint_result.json`, `w4a_prior_anatomy.json`.
모든 성능 수치는 **T1 단독 `generated` 경로, test 31명**이다.

---

## 1. 한 눈에 — 수정이 성능을 어떻게 바꿨나

같은 스크립트(`scripts/49_eval_frozen.py`)·같은 규약으로 잰 네 체크포인트.

| 체크포인트 | 주요 수정 | SC r | CCC | 전뇌 dice | pair dice | 오라클 wb | 오라클 pair | resid_r |
|---|---|---|---|---|---|---|---|---|
| **p4_joint_3000** (D0) | M1~M5 (전처리 재정합 + prior 재구조 + 손실 분리) | 0.713 | 0.058 | **0.608** | **0.095** | 0.589 | 0.328 | 0.004 |
| **d1_decoder_6000** (D1) | **M6·M7** BN 재보정 + 복원 전용 phase | 0.731 | 0.054 | 0.583 | 0.073 | **0.718** | **0.590** | −0.007 |
| **d3_joint_3000** (D2) | **M8** joint 재학습 (복원+생성제약+prior 동시) | **0.788** | 0.090 | 0.553 | 0.068 | 0.680 | 0.524 | 0.023 |
| **d3_joint_6000** (D3) | 위를 6,000 step 까지 | 0.765 | 0.103 | 0.502 | 0.049 | 0.678 | 0.519 | 0.026 |

**읽는 법 — 이 표가 말하는 것은 세 줄이다.**
1. **BN 재보정(M6)이 이 프로젝트 최대 성과다.** 오라클 pair dice 0.328 → 0.590 (+80%). 파라미터 추가 0.
2. **joint 학습(M8)은 SC 를 올리고 기하를 깎는 맞교환이다.** SC 0.731→0.788 얻고 전뇌 dice 0.583→0.553,
   그리고 D1 이 벌어놓은 오라클을 도로 반납(0.718→0.680, 0.590→0.524). 6,000 step 은 더 나쁘다.
3. **어떤 수정도 `resid_r`(개인차)을 못 움직였다.** 0.004 → −0.007 → 0.023 → 0.026, 전부 0 근처.

**그리고 이 표 전체가 기준선 아래다** — 훈련 144명 GT SC 평균(상수) r **0.945**, 남의 tractogram
(`cross_subject`) SC 0.841 / 전뇌 dice 0.574. 자세한 대조는 `PIPELINE_12` §2.

---

## 2. 데이터·전처리 수정

### M1. rigid MNI 정합 + [0,1] 정규화 + WM 2채널 입력
- **파일**: `configs/pipeline_retrain.yaml`, `scripts/36_build_wm_maps.py`, `scripts/37_build_rigid_t1.py`
- **왜**: upstream ATM 은 번들마다 T1 정규화 상수와 좌표 박스를 따로 갖는데, 이 프로젝트는 3,321 쌍에
  단일 전역 박스를 쓴다. 그 결과 anatomy feature 크기가 죽어 있었다.
- **결과**: ‖a‖ **0.086 → 0.785**, 조건화 1층 기여비 **1/22 → 0.57**. **크기 문제는 해결.**
- **한계**: subject 간 cosine **0.9996** 그대로. 크기는 살았지만 **분산은 안 살았다.**
  → 이게 축 A(개인차)의 출발점이 된다.

---

## 3. 모델 코드 수정

### M2. `anatomy_norm` LayerNorm 추가
- **파일**: `src/atm_sc/models/roi_pair_embedding.py:48-50`
- **어떻게**: `nn.LayerNorm(cond_dim, eps=1e-8)`, weight 를 `cond_dim**-0.5` 로 초기화, bias 0.
- **결과**: M1 과 함께 ‖a‖ 회복에 기여.
- **후속**: "LayerNorm 이 ‖a‖ 를 지워서 개인차가 죽는다"는 가설은 나중에 **기각**됐다 —
  ‖a‖ 는 subject 간 CV 14.3% 로 변동이 큰데 공통 offset 예측력이 r=0.031 뿐이다. 신호는 크기가 아니라 방향.

### M3. prior 구조 변경 (upstream KDE → 단봉 가우시안 + mode 임베딩)
- **파일**: `roi_pair_embedding.py:69-72` (`mode_emb` / `mode_prior` / `mode_log_sigma`)
- **왜**: upstream 은 번들 30개마다 KDE 잠재 prior(`kde_models/{bundle}/kde_model.joblib`)를 따로 갖는다.
  3,321 쌍에 KDE 3,321개는 비현실적이라 `N(mu_pair, sigma_pair)` 로 대체했다.
- **결과 (W2-b prior 사다리, `w2b_prior_ladder.json`)**: 현재 구조 NLL **71.62**, 최고는
  `arch_additive` **15.60**. 즉 **prior 구조가 실제 병목의 하나**다.
  단 `arch_additive` 는 `mu = c_a + c_b + const` 로 **pair id 만 쓰는 ROI 가법 모형**이지
  anatomy 조건부가 아니다 (한 번 잘못 읽었다가 정정함).

### M4. `kl_loss` 에 `logvar_prior` 인자 추가
- **파일**: `src/atm_sc/losses/geometry.py:46-51`
- **어떻게**: `logvar_prior=None` 이면 단위분산 분기 → **기존 식과 bit-exact 동일**.
- **검증**: `w2b_prior_ladder.json` selfcheck — `kl_old == kl_new_logvar_prior0` = 68.5045 (일치),
  `bit_exact_zero_init: true`, `mix_k1_equals_diag: true`.

### M9. `prior_use_anatomy` 통로 배선 (켜지 않음)
- **파일**: `roi_pair_embedding.py:30,79-81` / `roi_atm.py` (`_prior_anat`, `prior_params`, `sample_z`,
  `generate`) / `training/run.py:461` (체크포인트 메타) / `training/trainer.py` 3개 호출부
- **어떻게**: `prior_anatomy = nn.Linear(cond_dim, 2*latent_dim)` 을 0-init 하고 생성자 인자로 on/off.
- **검증**: 0-init 상태에서 **bit-exact** (max|diff| 0.0), 가중치 교란 시 z 변화 확인, **pytest 144개 통과**.
- **결과**: **기본값 False 유지.** 켤 근거가 없다 — 이 Linear 의 출력은 **pair 와 무관**해서
  subject 잔차의 **3.8%** 밖에 표현 못 한다 (W4-a 측정). 나머지 96.2% 를 겨냥하려면
  `d = f(anatomy, pair_vec)` 로 바꿔야 하는데, 그 성분이 애초에 현재 feature 로 예측되지 않는다.

---

## 4. 손실·학습 수정

### M5. KL detach + 별도 prior 적합항 + `prior_scale` LR 그룹
- **파일**: `training/trainer.py:378-384` (손실), `:140` (LR 그룹)
- **어떻게**:
  ```
  l_kl    = kl_loss(mu, logvar, mu_p.detach(), 2.0 * ls_p.detach())   # posterior 만 민다
  l_prior = -prior_log_prob(mu.detach(), mu_p, ls_p).mean()           # prior 만 민다
  ```
  둘을 분리한 이유: 한 항으로 묶으면 prior 가 posterior 를 쫓아가며 같이 붕괴한다.
- **결과 (D3)**: prior `precision_ratio` **45.09 → 32.97** (같은 스크립트가 p4_joint 36.61 을 재현).
  예측 최고치 15.60 에는 못 미친다 — joint 학습이 posterior 평균을 `mu_pair` 에서 더 밀어내
  (offset 3.41 → 5.8) mu 쪽 이득을 상쇄했다.
- **부수 발견**: **D1 이 prior 를 망가뜨렸다** (p4_joint 36.61 → d1_decoder 45.09).
  복원만 학습시키면 잠재 분포가 prior 에서 멀어진다.

### M6. BatchNorm 재보정 (`bn_mode`) — **가장 큰 성과**
- **파일**: `training/trainer.py:100` (`bn_mode: str = "train"`), `:171` (assert), `:189` (BN 고정),
  `training/run.py:428-446` (학습 시작 시 1회 재보정, `scripts/38_decoder_capacity.py` 의 BN 유틸 재사용)
- **문제**: 같은 가중치인데 렌즈에 따라 복원 오차가 두 배 갈렸다 —
  **train 모드 3.55~3.79 mm (trainer 로그) vs eval 모드 8.46 mm (실제 추론)**.
  `p4_joint` 가 recon/segment/생성 경로를 같은 ConvVAE BN 에 통과시켜 running 통계가 여러 분포의 평균이 됨.
  → **3.55 mm 는 추론 시점에 성립한 적이 없다.**
- **결과**:
  - gradient 없이 running 통계만 다시 쌓기 → eval 8.458 → **4.002 mm** (오차의 53% 소멸, **파라미터 0 추가**)
  - recon 3,000 step 미세조정 → eval **3.08 mm** (LR 3배면 2.89 mm)
  - 오라클 dice: 전뇌 0.589 → **0.718**, pair 0.328 → **0.590**
- **판정**: W1-d 의 "디코더 용량 증설" 안건은 **불필요**로 종결. 렌즈 문제였지 용량 문제가 아니었다.

### M7. `d1_decoder` phase 신설
- **파일**: `training/run.py:63` (`"d1_decoder": {"recon"}`), `configs/retrain/d1_decoder.yaml`
- **어떻게**: 조건화·SC·route 없이 복원만. UNet 동결(`unet_level: none` → 디스크 캐시된 anatomy 사용,
  UNet forward 를 아예 안 돌아 학습 비용 대부분 절감). anatomy 경로 기여가 0 인 것은
  통제 실험으로 확인됨 (`abl_gap = own_r − zero_r` 가 p0/p1 에서 정확히 0).
- **결과**: 위 M6 수치. **단 대가가 있다** — 생성 전뇌 dice 0.608 → 0.583, pair 0.095 → 0.073,
  prior precision 36.61 → 45.09. 복원은 좋아졌는데 생성은 나빠졌다.

### M8. `d3_joint` 재학습
- **파일**: `configs/retrain/d3_joint.yaml`, `scripts/49_d3_joint.py`
- **어떻게**: 복원 + 생성 제약 + prior 를 동시에 학습.
- **결과 (`w3a_joint_result.json`, d1_decoder → d3_joint_step3000)**:
  - **회복**: `valid_conn` 0.042 → **0.239**, `endpoint_in_roi` 0.555 → **0.681**,
    recon 2.240 → 3.401 (게이트 4.0 통과)
  - **SC 개선**: pass r 0.7314 → **0.7881**, tier r small/mid/large 0.158/0.188/0.647 → 0.171/0.196/**0.732**
  - **기하 하락**: 생성 전뇌 dice 0.583 → **0.553**, pair dice 0.0727 → **0.0678** → **통과 기준 미달**
  - 6,000 step 까지 늘리면 `valid_conn` 은 0.284 로 더 오르지만 기하가 무너진다
    (전뇌 dice 0.502, pair 0.049, 길이 157 mm) → **3,000 step 이 최적점**

---

## 5. 평가·규약 수정 (성능은 안 바꾸지만 판단을 바꿈)

### M10. dice 규약 고정
- **파일**: `outputs/eval/w3b_protocol.json`, `scripts/51_w3b_protocol_report.py`, `scripts/49_eval_frozen.py`
- **어떻게**: `wb@N` 은 `n_pred == n_gt == N`, `pair@64` 는 `pred = gtA = gtB = 64`.
  **규약 없는 dice 는 인용 금지.**
- **결과**: 기존에 돌던 `pair dice 0.095`(pred 16 vs gt ≤256)와 그 "천장" `0.700`(128 vs 128)은
  **저울이 달라 비율 해석이 불가능** → 폐기.
- **재측정 후 판정 (변화 없음)**: 모델은 `cross_subject` 기준선을 **못 넘는다**.
  wb@8000 0.546 vs 0.574±0.016, wb@20000 0.608 vs 0.629±0.016, pair@64 0.135 vs 0.222(cross)/0.283(group)/0.623(천장).
  SC r 은 가닥 수에 거의 무관(Δ≤0.003)해 규약 영향 없음.

### M11. 추론 시 그룹 정보 주입 폐기
- **어떻게**: `--use-bank` latent bank 와 템플릿 pair 배분은 **보고 지표에서 제외**. `generated` 단독만 인용.
- **결과**: 보고 SC r 이 **0.85~0.88 → 0.713 으로 내려갔다.** 성능 하락이 아니라 **허수 제거**다 —
  그 0.85~0.88 은 개인 예측이 아니었다.

### M12. 그룹 템플릿 기준선 도입
- **파일**: `scripts/49_eval_frozen.py:47` (`group_template`), `--template-subjects outputs/splits/train.txt`
- **어떻게**: 훈련 144명 GT SC 평균. train/test 겹침 0 확인(144/31, `comm` 검증).
- **결과**: 템플릿 r **0.9450** / CCC 0.9364 vs 모델 0.7881 / 0.0897. **31명 전원(0/31) 미달.**
  잔차 상관 `residual_r` 0.023 ± 0.162 → 0과 구분 안 됨.
- **이게 지금 가장 중요한 숫자다.**

---

## 6. 만들었지만 안 켠 것 / 폐기한 것

| 항목 | 상태 | 이유 |
|---|---|---|
| `prior_use_anatomy` 통로 | **배선 완료, 기본 False** | 출력이 pair 무관 → 표현 상한 3.8% (M9) |
| `streamline_refiner.py` | 구현됐으나 미사용 | `use_refiner: false`. 켜려면 run/checkpoint 배선 추가 필요 |
| ROUTE 파이프라인 (`configs/pipeline_route.yaml`, r1~r8/s1~s6) | **폐기** | 방향 전환. **PBS 82900 이 아직 이 설정으로 큐에 있다 → 처리 필요** |
| latent bank / 템플릿 배분 추론 | 폐기 | 추론에 그룹 정보 금지 (M11) |
| 디코더 용량 증설 | 취소 | BN 재보정만으로 목표 달성 (M6) |
| 상류 ATM 후처리 (`tckedit` GM/WM 마스크, `minlength 20`, MATLAB `bundle_trimming`) | **미구현** | 로컬에 MRtrix/MATLAB 없음. overreach 1.407(GT 0.212)의 직접 원인 후보 |

---

## 7. 순효과 총정리

| 수정 | 무엇이 좋아졌나 | 무엇이 나빠졌나 | 순판정 |
|---|---|---|---|
| M1 전처리 재정합 | ‖a‖ 0.086→0.785, 기여비 1/22→0.57 | — (분산은 미해결) | **부분 성공** |
| M3 prior 재구조 | — | NLL 71.62, 최고 15.60 대비 열위 | **병목으로 확인** |
| M4 kl_loss 확장 | 배선 확보 | — (bit-exact) | 중립 |
| M5 KL detach + prior 항 | precision 45.09→32.97 | — | **성공** |
| **M6 BN 재보정** | **eval recon 8.46→4.00 mm, 오라클 pair 0.328→0.590** | — | **최대 성과** |
| M7 d1_decoder phase | 오라클 확보 | 생성 dice 0.608→0.583, prior 36.6→45.1 | 맞교환 |
| M8 d3_joint | SC 0.731→0.788, valid_conn 0.042→0.239 | 전뇌 dice 0.583→0.553, 오라클 반납 | **맞교환, 게이트 미달** |
| M9 anatomy prior 배선 | 실험 가능해짐 | — (안 켬) | 보류 |
| M10 규약 고정 | 비교 가능해짐 | 기존 수치 일부 무효 | **필수 정정** |
| M11 그룹정보 폐기 | 정직해짐 | 보고치 0.88→0.713 | **필수 정정** |
| M12 템플릿 기준선 | 문제의 크기가 드러남 | — | **결정적** |

**한 줄**: 디코더는 고쳤고(M6), 지표는 정직해졌고(M10·M11·M12), prior 는 병목으로 특정됐다(M3·M5).
**개인차는 어떤 수정으로도 안 움직였다.**

---

## 8. 다음 수정 후보 (아직 안 한 것)

| 우선 | 수정 | 겨냥하는 수치 | 비용 |
|---|---|---|---|
| **A** | **count head 를 `SC − 그룹템플릿` 잔차에 학습** | `resid_r` 0.023 (프로브 상한 0.101) | 학습 1회. **개인차 축의 데이터 가설을 검정하는 유일하게 안 해본 직접 실험** |
| B1 | prior 온도 스윕 `z = mu + s·sigma·eps` (s 0.2~1.0) | 오라클 0.524 vs 생성 0.068 격차. prior sigma 1.0 vs GT 조건 내 sd 0.18 | **학습 0** |
| B2 | 가닥 수 보정 상수 k=11.2 | CCC 0.090 → **0.746** (subject 최적 0.762) | 즉시 |
| B3 | 후처리 필터 (GM/WM 마스크 + minlength 20) | overreach **1.407** vs GT 0.212 | 마스크는 있음 |
| C | prior 를 `arch_additive` 구조로 | NLL 71.62 → 15.60 | 재학습 |
| — | DWI/FOD 입력 추가, GT native space | 정보를 실제로 더하는 유일한 길 | **막힘** (raw DWI/DSI Studio 없음) |

---

## 9. 2026-09-07 — E/D 재설계 (M13~M22)

전제가 바뀐 날이다. "feature 문제가 아니다"(사용자 지적)가 맞았고, 병목은 **풀링**과 **공유 가중치**였다.
전부 val 31명에서 판정했고 test 는 안 썼다 ([[test-split-used-once]]).

### M13. count head 에 ROI 국소 feature 통로 (`edge_count_head.py` local_proj)
- 왜: `a512` 의 subject 성분 2.1% (코사인 0.9994) vs ROI 국소 풀링 13.4%. global average pooling 이 개인차를 지운 뒤였다.
- 결과: `resid_r` +0.001 → **+0.034**, self−shuffled −0.005 → +0.031, 식별 0.032 → 0.065.

### M14. 티어 1 해부량 (`data/anat_tier1.py`, 9-d pair feature)
- 왜: 프로브 ridge 가 GT SC 잔차와 **r=0.1305** (206명, 순열 p=0.0, 귀무 sd 0.0132). 학습된 feature 전부를 이긴다.
- 함정: 원본 T1 밝기 중앙값이 163~63,824 (390배) — 스캐너 스케일. 중앙값 정규화하고 `t1_scale` 은 교란변수로만 남김.
- 공유 MLP 로 붙인 E5: `resid_r` 0.0354, shuffled +0.0158 — **프로브의 1/4, 실패**.

### M15. 티어 1 을 pair 인덱스 가중치로 (`tier1_w [3321,9]`, ridge 와 동형)
- 왜: 프로브가 이긴 자유도는 pair 마다 독립인 29,889 개였다. 공유 가중치는 그걸 통째로 버린다.
- 결과 (E6): `resid_r` 0.0354 → **0.0739**, self−shuffled +0.0692, 식별 **0.194 (p=0.00, 12회 전부 유의)**, `inter_subj` 0.9996 → 0.9879. wd 0.1 이 1.0 보다 낫다 (E7).

### M16. 인코더 stage3 해동 + M15 (E8)
- 결과: gap **0.0704 @2500**, 식별 **0.258 (8× chance)**. `unet_level=full` 은 t1_encoder 23.7M 으로 stage3(22.4M) 대비 +5.9% 뿐이고 20.8GB 로 OOM → 드롭.
- 함정: 마지막 checkpoint 만 남아 `max_steps` 가 곧 모델 선택이다. E8 은 최적(2500)이 아닌 3000 을 쓴다.

### M17. 추론 배분을 pair count head 잔차로 변조 (`allocate_counts(resid_log=...)`)
- 왜: `allocate_counts` 가 읽던 `count_head_end` 는 잔차 목적으로 학습된 적이 없다. E 축이 만든 개인차를 생성 경로가 아예 안 읽었다.
- 부수 버그 3개: `edge_count_matrix` 가 local/tier1 을 안 넘겨 **국소 통로 checkpoint 로는 추론이 죽던 것**, pair dtype, grad 누수.
- 결과 (val 12명, alloc 경로): gain 1 → `inter_subj` 0.9950 / `resid_r` 0.0295; **gain 8 → 0.8923 (GT 0.9023) / 0.0163**. 진폭을 키우면 상관이 내려간다 — MMSE 이론 그대로.

### M18. `inter_subj_r` 는 진폭 지표다 (시뮬레이션, val GT)
- var_ratio=1.0 이면 `resid_r` 0.074 든 1.0 이든 `inter_subj` 0.73~0.76. **0.8 은 gain 하나로 오늘 도달하지만 그때 퍼진 양의 99.5% 는 틀린 방향.** 진폭과 정확도(`resid_r`·식별)를 항상 같이 보고한다.

### M19. `edge_head` / `count_head_end` 에 국소+티어1 통로 (`a3_aux` phase, `count_end` 가중치 분리)
- 왜: val 10명 edge 선택 Jaccard **0.9896**, 확률 subject 성분 **1.6e-06** — "어떤 연결이 있는가"에서 개인차가 전부 지워진다. `count_end` 는 `w.count` 를 공유해 잔차 실험에서 **한 step 도 학습되지 않았다**.
- **본 학습 A3 (2000 step, val 10명)**: edge Jaccard **0.9951 → 0.8373**, 확률 subject 성분 **3.9e-06 → 7.0e-03**, 선택 pair 수 1821~1830 → 1768~1827. count_end r **0.867** / CCC 0.832 (이전 0.57, 균등 0.71). **통과.**

### M20. subject 조건부 prior — 두 번 실패, 세 번째 설계
- D-f (공유 MLP): prior mu subject 성분 **0.000154**. D-f' (pair 별 저랭크 [512→4→128], 1.7M): **0.000525**. 신호 상한은 posterior mu 기준 0.566 이고 split-half r=0.9956 으로 **재현되는 진짜 신호**다.
- 진단: 용량이 아니라 손실. pair 평균이 `L_prior` 를 압도해 subject 잔차 기울기가 묻힌다 — count head 에서 템플릿을 뺀 것과 같은 상황.
- D-f'' (`prior_res`): 본 학습 뒤 subject 성분 **0.000189 — 실패**. 스모크의 `prior_res_r` 0.21 은 EMA 지연(pair 오프셋 노름 ~5)을 맞춘 허수였다.
- 잡은 원인 4개 (각각 실측으로 확인): ① 가지 출력의 98% 가 pair 수준 → 입력 **중심화** (가지 subject 성분 0.02 → 0.12); ② EMA 가 pair 당 ~8회 갱신 → train 144명 pair 별 posterior mu **템플릿**으로 초기화 (타깃 노름 4.9 → 3.3, `prior_res_r` 0.2 → 0 = 이전 값이 허수였음을 확인); ③ 중심화 입력 크기 ~0.02/dim 이고 U 가 lr 3e-5 그룹 → **train std 표준화** + U 를 lr 1e-3 그룹으로 (가지 subject 성분 0.13 → 0.56); ④ step 당 64 pair 만 봄 → 256 으로.
- **D-f''' 본 학습 (2000 step)**: prior mu subject 성분 **0.00149** — 원래(0.000154)의 **10배**지만 게이트 0.005·목표 0.05 미달. 가지 크기 0.117 로 컸는데 subject 성분이 0.56 → 0.13 으로 희석 (공유 MLP 가지가 pair 수준 출력을 키운 것으로 추정). **부분 성공으로 채택**하고 D4 로 진행 — 생성 경로 개인차는 A3·잔차 배분·cond FiLM 이 주로 낸다.
- 그 전에 잡은 것: `prior: 0.0` 이라 prior_local 이 학습 자체가 안 되던 것, 학습된 prior 를 "0-init 이어야 한다"고 우기던 낡은 자기검증, `resid_stats` 누락, 잘못된 조상(d3_joint) 에서 resume.

### M21. 구조 상속 (`run.py`): resume checkpoint 의 가지를 config 대신 checkpoint 에서 읽는다
- 왜: 단계마다 새 가지가 생겨 뒤 config 가 전부 재선언해야 했고, 하나 빠지면 "checkpoint 에만 있는 키" 로 죽었다 (A3 20:20, D4 는 그 후폭풍으로 삭제된 경로를 집음). 파이프라인에 단계 실패 시 중단(`need_ok`/`need_ck`) 추가.

### M22. 국소 통로를 안 넘기던 옛 호출부 정리 (같은 원인, 10곳)
`_recon_rmse`, `generate` 3곳, `prior_log_std`, `select_pairs`(no_grad), `allocate_counts`, 59번 prior_params, selfcheck self-edge pair 3곳. 공용 헬퍼 `roi_feats_if_needed`, `aux_feats_for` 로 한 곳에 모음.

### 남은 것 (파이프라인 순서)
D-f'' → A3 → D4(좌표 상자, 게이트 ≤4.0mm 미달 상태) → J1 joint(낮은 LR) → A/B(val) → **test 1회**.
코드 미완: 최적 step checkpoint 보존, `weight_head` 제거.

### M23. D4 앵커 폐기, J1 no-anchor 채택 (2026-09-08 00:00~01:00)
- D4 (앵커 alpha 0→1 램프, 2500 step): alpha=1 recon **25.8 mm** (게이트 4.0, 이전 시도 12.4). 파이프라인이 이 게이트를 안 걸러 J1 이 그 위에서 돌아 recon 31.8→33.2 mm — **폐기**. `run_pipeline3.sh` 에 D4 게이트 부재 (TODO).
- J1 no-anchor (D-f''' 위, 전 LR 1/3, 1000 step): recon 4.53→**4.41 mm** (게이트 4.6 통과), valid_conn 0.18→0.22. **최종 checkpoint** = `j1_joint_noanchor/d3_joint_step1000.pt`.
- A/B (val 12명): temp 0.2 + WM 필터 0.3 이 Pareto (dice 0.512, overreach 1.02, retain 0.72). SC r 은 temp 에 둔감 (0.70~0.73).
- gain 재보정 (val 12명, 6만 가닥): gain 1 에서 alloc `inter_subj` 0.815 < GT 0.902 → **gain 1** 채택. alloc `resid_r` ≈ 0, count head `resid_r` 0.115.
- 국소 통로 누락 호출부 3곳 더 (57 selfcheck `condition`, `gen_chunk` 배분, `generate_by_count`).

### M24. test 1회 (31명, 46만 가닥/명, gain 1) — `outputs/eval/final_d3_joint_step1000.json`, 그림 `outputs/figs/test_sc_gt_vs_*.png`

| 경로 | r (절대) | `resid_r` | `inter_subj` (GT 0.896) |
|---|---|---|---|
| alloc (최종 추론) | 0.576 (0.51~0.67) | **−0.016** | 0.9715 |
| generated (균등) | 0.729 | 0.010 | 0.998 |
| count head | — | **0.084** | 0.991 |
| 그룹 템플릿 | **0.945** | — | — |

- **판정**: 생성 경로는 31명이 동일하진 않지만(0.9715 < 균등 0.998) 그 차이는 GT 잔차와 무관하다(`resid_r` ≈ 0). val 12명·6만 가닥에서 본 퍼짐(0.815)은 대부분 가닥 표본 잡음이었다 — 가닥을 46만으로 올리자 0.97 로 돌아왔다. 개인차 정보는 **count head 경로에만** 있다 (`resid_r` 0.084, val 0.07~0.115 와 일치). 절대 SC 는 여전히 템플릿(0.945)을 아무도 못 넘는다 (0/31).
- 오늘의 순효과: count head 개인차 0.034→0.08 (test), edge 선택이 subject 별로 갈라짐(Jaccard 0.995→0.837), prior 10배 — **그러나 tractogram 으로 가는 길에서 그 정보가 SC 로 전달되지 않는다.** 다음 병목은 "count head 가 아는 것을 생성 경로가 쓰게 하는 것" 이다 (배분은 count_end 가 정하고, 잔차 변조는 gain 1 에서 효과가 작다).

## §10 백질 제약 + 목표값 재설계 (2026-09-08)

### M25 개인차가 어디서 사라지는지 (단계별 측정, val 31명)

`scripts/64_alloc_to_sc.py` → `outputs/eval/alloc_to_sc_val.json`

| 단계 | resid_r |
|---|---|
| count head 예측 | 0.086 |
| 배분 라벨 | 0.018 |
| 생성 trk 끝점 SC | 0.014 |
| 생성 trk pass SC | 0.004 |

손실은 두 곳이다. (1) 배분을 지배하는 `count_head_end` 가 잔차 목적으로 학습된 적이 없다 → 79 %.
(2) 가닥 하나가 ROI 6.4 개를 지나 pair 20 개에 count 를 더한다 → 라벨→pass 전달률 0.27.

**"trk 에 개인차가 있는데 SC 에서 사라진다" 는 틀린 설명이었다.** SC 는 trk 의 결정적 함수다.
애초에 배분에 개인차가 거의 안 실렸고, 실린 것마저 희석됐다.

### M26 RL 역산 (`scripts/65_rl_alloc.py`)

pass SC 는 라벨의 선형 변환이다: `SC = A^T n`, `A[k,p]` = 라벨 k 가닥 1개가 pair p 에 더하는 기댓값.
A 를 subject 자신의 가닥 표본에서 재고 `A^T n = T` 를 Poisson EM(Richardson-Lucy)으로 푼다.
GT/그룹 정보를 쓰지 않는다.

전달률을 지배하는 두 인자 (`scripts/81_rl_diag.py`, 독립 표본 채점 = 실제 생성에서 나올 값):

| 표본/pair | alpha | 전달률 |
|---|---|---|
| 8 | 8 | 0.27 |
| 32 | 16 | 0.51 |
| 64 | 24 | 0.66 |

Â 의 추정 잡음이 역산에서 함께 증폭되므로 **표본 수가 전달률을 지배한다**. 같은 표본으로
채점하면 0.67 인데 독립 표본이면 0.27 이다 (표본 8 기준) — 절반이 과적합이었다.

### M27 선형 진폭 (정확도와 분산의 분리)

로그 공간에서 `alpha*r` 을 넣고 expm1 하면 지수가 큰 edge 를 폭발시킨다 (능선 alpha 8 에서
resid_r 0.147 → 0.018). **선형 공간 증폭** `템플릿 + alpha*(expm1(템플릿+r) - 템플릿)` 은
LOO 중심화 후 잔차의 배율일 뿐이라 resid_r 이 (거의) 불변이고 subject 간 상관만 내려간다.

| alpha | resid_r | inter (GT 0.904) |
|---|---|---|
| 1 | 0.1531 | 0.998 |
| 8 | 0.1521 | 0.902 |
| 16 | 0.1491 | 0.749 |

### M28 백질 점유 손실 (`losses/geometry.py::wm_occupancy_loss`, W1 단계)

WM 확률맵은 인코더 입력 채널로만 쓰고 있었고 **생성을 제약하지 않았다**. 실측(SyN 공간):
GT 중간구간 WM 점유 0.825 / 생성 0.393.

두 번 실패하고 고쳤다.
1. 원 WM 확률로 걸면 백질 밖에서 기울기가 정확히 0 이다 → 거리 램프 안내장 (`data/wm_guide.py`).
2. 램프 상한이 1 이면 5 mm 떨어진 점이 이미 target 을 넘긴다 (손실 0, 실제 점유 0.557)
   → 상한 0.5 로 낮춰 마지막 접근은 원 확률이 맡게 했다. 검증: 가닥 8.7 mm 이동, 점유 0.351 → 0.984.

스윕 판정 (`scripts/78_pick_wm_arm.py`): lr_dec 를 10 배 올리면 점유는 더 오르나 복원이
게이트(4.6)를 넘는다. wm_gen=10, lr_dec=1e-5 채택. 본 학습 1000 step: 점유 0.418 → 0.473,
복원 4.412 → 4.669.

### M29 공간 불일치 — 조직맵은 두 벌이 필요하다

GT 가닥은 QSDR/NLin6 (비선형) 공간이고 feature 용 조직맵은 rigid (개인 형태 보존) 다.
GT 점이 GM+WM 안에 들어가는 비율: rigid 0.820 / **syn 0.907**, GT WM 점유 중앙 0.573 / **0.801**.
→ `tissue_path(cache, sub, source)` 로 나눴다. rigid = feature, syn = filtering/trimming/손실.

### M30 새 T1 기반 추출 (전부 새 다운로드 없이)

| 데이터 | 산출물 | 비용 |
|---|---|---|
| CSF/GM/WM 확률 + subject 뇌 마스크 | `{sub}_tissue[_syn]_W.npz` | 8 코어 10 분 |
| corridor pair feature 7 개 | `anat_corridor/{sub}_{src}.npz` | 0.7 초/명 |
| ROI 별 WM 경계 면적 (marching cubes) | 같은 파일 | 포함 |

FreeSurfer/FSL/FastSurfer 는 미설치. WM 표면은 WM 확률 0.5 등고면의 marching cubes 로 대체.

### M31 `models/pair_ridge.py` — pair 별 선형 잔차 head

`count_head` 는 pair 별 tier1 가중치를 갖고도 val resid_r 0.086 인데, 같은 정보의 능선회귀가
0.147 이다. 신경망이 절대 SC 적합에 끌려가 잔차를 못 배웠다. 60 feature(tier1 9 + corridor 7 +
surf 3 + tissue 8 + 인코더 국소 PCA 32) 로 재적합 → **val 0.153**.

**표준화 폭발 버그**: pair 별로 표준화하는데 train 분산이 0 에 가까운 pair-feature 가 있으면
`sd + 1e-6` 으로 나뉜다. val 표준화 값 최대 36789, resid_r 0.147 → 0.0002. train CV 는 로그
상관이라 배율에 둔감해 0.0785 로 멀쩡해 보였다. 표준편차 하한을 feature 전체 규모의 5 % 로 두고
`abs(X) < 50` assert 추가.

### M32 W2 최종 (val 31명, alpha 24, 표본 64, iters 100, trimming 없음)

| 경로 | resid_r | inter | abs_r |
|---|---|---|---|
| 기존 배분 | 0.0016 | 0.9138 | 0.595 |
| **능선 + 역산** | **0.0813** | **0.8859** | **0.810** |
| GT 목표 (오라클) | 0.5387 | 0.9528 | 0.869 |
| GT 자체 | — | 0.9043 | — |

개인차 51 배, subject 간 상관이 GT(0.9043)보다 오히려 다양, 절대 정확도 0.595 → 0.810.

### M33 제거

- `weight_head` (213,761 파라미터): 마지막 층이 정확히 0 — 어느 phase 도 학습시킨 적이 없고
  by-count 배분에서 구조적으로 불필요. 옛 checkpoint 키는 `_drop_dead_keys` 가 버린다.
- `--use-bank`: 추론에 train latent bank 를 넣는 경로 (그룹 정보 금지 위반). 인자 자체를 삭제.

### M34 잡은 데이터 오류

- 한 train subject 의 기존 WM 맵이 틀렸다 (뇌 안 WM 비율 0.540, 코호트 0.372±0.020 = 8 sigma).
  재분할이 정상값(0.365)을 냈고 조직 대비도 더 좋다. 교체하고 `.bad.npy` 로 백업.
- 재현되는 극단값 5명 (비율 0.20~0.57)은 두 번 돌려도 같은 값 — 방법이 그 T1 에서 일관되게
  실패하는 것이다. 기록만 남김.
- **affine 불일치**: 아틀라스는 2 mm (91,109,91), 조직맵은 1 mm W 격자 (193,229,193).
  trimming 에 아틀라스 affine 을 쓰면 엉뚱한 위치를 읽고 숫자만 그럴듯하게 나온다. assert 추가.

## §11 SC 평가 분해와 인코더/prior 진단 (2026-09-09)

### M35 상관 하나로 판정하면 안 된다 (`evaluation/sc_profile.py`)

절대 r 0.810 / edge F1 0.93 / subject 간 상관 0.886 이 전부 좋아 보였는데 실제로는:

| 항목 | 생성 | GT |
|---|---|---|
| 0 인 edge 비율 | **0.002** | 0.159 |
| 0 특이도 (GT 0 에서 0 낼 확률) | **0.007** | — |
| 가닥당 pair 기여 | **28.9** | 7.50 |
| 작은 / 중간 / 큰 연결 r | **0.063 / 0.155** / 0.754 | — |
| 총합비 | 1.177 | 1.0 |
| **맞는 분산 몫** (resid_r²) | **0.66 %** | — |

절대 r 은 크기 가중이라 큰 연결이 만든 숫자고, subject 간 상관 0.886 은 **증폭된 예측 오차**다
(분산비 0.99 인데 resid_r 0.081 -> 생성 subject 차이의 99.3 % 가 틀린 차이).

8 게이트로 교체: 희소성(0 특이도 >= 0.5) · 밀도 · 총합 · 작은연결 · 중간연결 · 순위 ·
개인차(맞는 분산 >= 10 %) · 가닥당 pair.

### M36 SC 정의 재검증 — 우리 계산식은 맞다

원본 100 만 가닥 전체를 `hard_sc` pass 모드에 넣으면 `.mat` 이 그대로 나온다.

| | 재현 | `.mat` |
|---|---|---|
| 합 | 7.4999e6 | 7.5673e6 |
| 0 비율 | 0.2316 | 0.2298 |
| 비영 edge | 2552 | 2558 |

선형 r **0.9987**, 로그 r 0.9923, 0 일치 0.936. **성긴 구조는 만들 수 있다** — 문제는 우리
생성 가닥이 GT 가 연결하지 않는 pair 에 부스러기를 뿌리는 것이다 (가닥당 pair 28.9 vs 7.50).

주의: `bundles.npz` 는 ROI pair 에 배정된 15 만 개 **부분집합**이다. 이걸 100 만 개 기준의
`.mat` 과 비교하면 "1.8 배 과다 계수" 같은 잘못된 결론이 나온다 (실제로 한 번 그랬다).

### M37 trimming — 기하는 고치고 개인차는 못 올린다

자르기(백질 경계)와 버리기(뇌 마스크/길이)를 분리해야 한다. GT 를 죽이는 건 버리기다.

| 기준 | GT 유지율 |
|---|---|
| 뇌 마스크 밖 <= 0.05 | 0.28 |
| 길이 >= 20 mm (자른 뒤) | 0.84 ~ 0.86 |
| **자르기만 (둘 다 끔)** | **0.93 ~ 0.97** |

자르기만 켠 결과 (val 31명, J1): 가닥당 pair 25.7 -> **8.2** (GT 7.5), 0 특이도 0.011 -> **0.132**,
총합비 1.159 -> **0.949**. 8 게이트 중 0 -> 2 통과.

`--max-total` 상한이 60 만이라 31명 전원이 0.631 배로 눌려 총합비가 0.671 이었다 (역산이
95 만 가닥을 요구한다 — GT 100 만과 비슷하다). 상한을 150 만으로 올리고 눌리면 경고를 찍는다.

### M38 백질 점유 손실(W1)은 기여하지 않았다 -> 폐기

학습 중 점유는 0.418 -> 0.473 으로 올랐으나 SC 재현 지표는 J1 과 같거나 미세하게 나쁘다
(resid_r 0.081 vs 0.086, 가닥당 pair 28.9 vs 25.7, 복원 4.669 vs 4.412).

### M39 T1 은 문제가 없다 — 인코더 조건 벡터에서 죽는다

| 단계 | subject 간 상관 | 개인 성분 |
|---|---|---|
| T1 볼륨 (rigid) | 0.384 | **7.99 %** |
| WM 확률맵 | 0.397 | 34.37 % |
| T1 볼륨 (syn) | 0.621 | 2.52 % |
| ROI 국소 feature | 0.912 | 11.36 % |
| anatomy 전역 512d | 0.9994 | 1.94 % |
| **디코더 조건 c** | **0.9996** | **0.04 %** |

`rigid` 선택이 옳았다 (SyN 은 개인차를 1/3 로 줄인다). 죽는 지점은 `condition()` 의 덧셈이다.

```
c = anatomy_term + proj(pair_vec) + mode_emb + local_proj(local)
RMS   0.0307        0.0334                     0.0014      <- 24 배 차이
```

`local_proj` 는 0-init 이라 학습으로 자라야 하는데 J1 까지 와도 가중치 최대 0.0122 다.

**수정** (`cond_local_gain`): 두 항을 RMS 정규화하고 학습 가능한 배율로 크기를 구조로 잡는다.
재학습 전 조건 벡터만 봐도 gain 0.3 -> 0.69 %, 1.0 -> 4.44 %, **3.0 -> 11.14 %**.

두 번 실패하고 고쳤다.
1. gain 3.0 에서 기울기 NaN — 정규화 나눗셈이 미분 경로에 있어 RMS 0.0014 로 나뉘며 700 배
   증폭. 배율을 **detach** 했다.
2. 즉시 적용 시 게이트 미달 (복원 4.412 -> 4.937, valid_conn 0.222 -> 0.198) — 조건 분포가
   바뀌면 디코더가 못 보던 입력이 된다. `cond_local_gain_steps` 로 램프업.

### M40 prior 는 학습이 안 된 것이지 구조 한계가 아니다

train 6명 x 256 pair, posterior pair 평균 대비:

| 구성 | 거리 | 설명하는 분산 몫 |
|---|---|---|
| **현재 prior** | 7.91 | **-0.573** (상수보다 나쁘다) |
| 같은 구조(pair_vec 선형)의 최선 적합 | 2.45 | 0.633 |
| 자유 pair 표 | 0 | 1.0 |

`L_prior` 가중치가 0.1 인 항 하나뿐이고 그마저 가닥 단위 `mu` 를 맞추려 해 pair 내 잡음에
흔들린다. **pair 별 자유 평균표**(`prior_mu_table`, 3321x64 = 212,544 파라미터, 0-init 이라
켜도 bit-exact) 를 넣고 `init_prior_template` 의 train posterior pair 평균으로 직접 채운다.

그 과정에서 잠재 버그: `prior_mean` 과 `prior_params` 가 서로 다른 값을 낼 뻔했다
(`prior_res` 손실이 `prior_mean` 기준). `prior_table_term` 으로 통일.

### M41 개인차의 상한 — 데이터에 묶여 있다

```
실현 resid_r ≈ 목표 resid_r × 전달률(0.33 ~ 0.54)
개인차 게이트(맞는 분산 10 %) 통과에 필요한 실현 resid_r = 0.32
→ 필요한 목표 resid_r = 0.6 ~ 1.0   /   현재 목표 상한 0.153
```

4 ~ 6 배 부족하다. 이전 프로브가 이유를 이미 기록해뒀다: **개인차는 연결의 모양이 아니라
세기에 있다** (latent 조건평균 4.4 % vs GT SC 카운트 18.5 %). GT tractogram 이 DSI Studio
QSDR 템플릿 공간이라 모양이 정규화로 지워졌다. 정보를 더하는 길은 `docs/PIPELINE_14` 참조.
