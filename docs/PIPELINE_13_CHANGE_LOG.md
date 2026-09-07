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
