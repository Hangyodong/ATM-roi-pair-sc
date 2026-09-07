# PROJECT_STATUS

목표: T1w → ROI-pair 조건 ATM → whole-brain tractogram → SC weight / tract-length (TVB).
설계 기준: `docs/ATM_ROI_PAIR_SC_FINETUNING_PIPELINE.md`. 현황: `docs/ATM_ROI_PAIR_SMOKE_TEST_REPORT.md`.

## 좌표계 (실측 확정)
GT `.tt.gz` = DSI Studio QSDR 템플릿 공간 (전 subject 동일, ≈ MNI152NLin6). atlas DK-PD25 82 ROI NLin6 2mm.
T1 native → ANTs SyN → NLin6 → W (193,229,193)@1mm (ATM UNet 하드코딩). 모든 streamline 연산은 mm.
GT SC = **pass** count (`.mat`), endpoint 모드 target 은 재계산한 `sc_end`.

## 핵심 수치
pass-SC 재현 r=0.9986 · soft pass-SC r=0.9939 · 2-pass gradient 오차 0 · 인코더 전용 UNet bit-exact ·
sub-000001 양성 pair 1759 (endpoint) · decode 260k streamlines/s · **pretrained anatomy feature ‖a‖=0.086 (거의 0)**.

## 환경
A10 23GB. torch 2.6+cu124, cuDNN 9.1 (cu13→cu12 교체로 수리), dipy, antspyx (scipy 1.15.3 으로 다운그레이드됨), statsmodels 0.15.
MRtrix/FreeSurfer/MATLAB/DSI Studio/singularity 없음 (불필요). `/mnt/d` 의 `.fib.gz` 접근 불가.

## 데이터 위치
`stable/stable/` (upstream, 30 모델+KDE), `PPMI_QC263_tracto/PPMI_QC263_tracto/` (zip 에 225 명만 존재; manifest 263),
**학습/평가 subject = `.mat`(238) ∩ 폴더(225) = 206 명** (`paths.subjects()`, `scripts/check_subjects.py`).
`.mat` 에 있으나 zip 에 없는 32 명: `outputs/subjects_missing_in_tracto_zip.txt`. SC 없는 19 명 제외: `outputs/subjects_no_sc_excluded.txt`.
`outputs/roi_pairs/{sub}/{assignments,bundles}.npz`, `outputs/cache/` (T1_W, anatomy, dist_maps).

## 우선순위 1·2 (2026-09-02)
- T1 강도 스케일 230배 편차 → robust 정규화 필수 (구현됨). 그 후에도 pretrained feature 는 subject 간 cosine 0.97 → decoder 만으로는 subject-specific 불가. `trainable=vae+unet4` 구현. 판별력 여부는 SC loss 단계에서 held-out SC corr 로 판정.
- SC 정의 권장: bundle/endpoint 는 endpoint, SC loss/평가는 **pass** (`--sc-mode pass`). 사용자 확정 대기.
- Phase 2: 재구성 OK(RMSE 3.7mm), prior z 생성은 실패(pair acc 0) → Phase 3 endpoint loss 가 필수. endpoint loss gradient 소실 버그 수정.

## 최종 전략 반영 (2026-09-02 밤) — `docs/ATM_FINAL_FINETUNING_REPORT.md`
- T1 encoder unfreeze(`unet_level`), 그룹별 LR, UNet backward 1회(anatomy leaf), 조건부 prior, endpoint loss 수정, robust T1 정규화.
- smoke 30/30 PASS(SC→T1 encoder gradient 확인), pytest 43, sanity(3명·100 step·full) PASS: L_recon 20.8→7.9, VRAM 16.2 GB, 2.3 s/step.
- 파일럿(1명): prior z pair acc 0.63, whole-brain pass-SC weighted r 0.945 (학습 subject).
- split: train 144 / val 31 / test 31 (`outputs/splits/`). **로컬 A10 에서 9-phase 파이프라인 실행 중** (`scripts/19_train_pipeline.py`, 로그 `outputs/checkpoints/pipeline_local.log`). PBS job **82753.KITSM02** (`PIPELINE=configs/pipeline.yaml`) Q — 세션 종료 후 GPU 가 잡히면 `pipeline_state.json` + `*_latest.pt` 에서 이어받는다 (lock: 15 분 heartbeat). **ctx/sub block 별 log-Pearson SC loss + 소/중/대 tier 균등 pair 샘플링 + block×tier val** (`src/atm_sc/data/roi_groups.py`, 보고서 §14, phase 4 도중 적용). 모니터: `python scripts/20_monitor.py --watch 60` (전 phase 진행/ETA/loss·gradient 추이/val 표, `--json` 가능).
- 전처리: 로컬 배치(1 core) 는 중단하고 **CPU job 82756.KITSM02 (base_8, 4 subject 병렬)** 가 남은 subject 를 뒤에서부터 처리 중. `std_q`/`base_32` 는 권한 없음(Unauthorized).
- 로컬 학습 진행: phase 2 geometry 3000 step 완료(val 3명: pair 0.009 / pass-SC r 0.50 — endpoint 이전) → phase 3 t1_encoder(full UNet) 진행 중, 2.6 s/step, VRAM 18.6 GB.

## 결정 사항
- upstream 무수정. ROI-pair 조건은 FiLM 입력 자리에 합산 주입 (0-init → pretrained 보존).
- decoder 초기화는 AF_L 하나. latent prior N(0,I). 좌표 박스 = NLin6 brain mask bbox + 5mm.
- bf16 금지 (좌표 오차 57mm). AMP 기본 off.
- full training 은 아직 하지 않았다.
- `external/tractolearn/` (2026-09-03 clone, scil-vital 4ff8500): FINTA streamline autoencoder 소스. **EULA: 학술/비상업 사용만**, 수정 허용, 특허 시 SCIL 연락, 상업 이용은 Imeka. pip 설치 안 함(torch 1.13/numpy 1.23/dipy 1.7/scilpy 고정). 대신 `from atm_sc.compat import tractolearn_env` 후 `import tractolearn.*` (sys.path + dipy `downsample` shim + scilpy `streamlines_in_mask` 대체 + fury stub). 추가 설치: torchsummary·pathos·umap-learn·comet_ml (user site, 핵심 패키지 불변). 41개 모듈 전부 import OK, AE forward/backward 검증 (`tests/test_tractolearn_compat.py`). fury(렌더링) 는 설치 시 numpy 2.4 로 올라가므로 미설치.
- **Route loss + PASS-SC 세분화 + GESTA 증강** (2026-09-03, 보고서 §15): GT 통과 ROI 전처리 `outputs/roi_pairs/*/visit.npz` (206명, co-visitation SC vs .mat r=0.94), `losses/route.py`(log 공간 BCE/Dice, presence), `route_tau=1.0`, block 별 SC corr, λ_syn=0.5. 독립 파이프라인 `configs/pipeline_route.yaml` (r1~r8) 은 baseline 종료 후 `scripts/23_after_baseline.sh` 가 실행.
- **전환 계획 (2026-09-03 16:30)**: baseline 은 phase 6(sc_corr) 까지만 실행하고 `scripts/25_switch_to_route.sh` 가 driver 종료 → GESTA latent bank/synthetic cache → `configs/pipeline_route.yaml`(s1~s5, phase6 checkpoint 이어받기) 로 전환한다. phase 7~9(mag/length/joint)는 route 버전으로 대체된다.
- **GT SC 정의 확정 (2026-09-03)**: 전체 tractogram 실측으로 Case B(같은 streamline 의 모든 ROI 쌍) 확인 — r=0.9986 vs 인접 transition 0.781. edge segment 분해는 이 정의를 따른다 (`src/atm_sc/data/edge_segments.py`, `scripts/26`, 보고서 §16).
- **ROUTE 파이프라인 s1~s6 확정 (2026-09-03)**: s1 route → s2 presence → s3 segment+count → s4 magnitude → s5 length → s6 joint. `configs/pipeline_route.yaml`. A10 실행 중, `scripts/pbs_route.sh`(base_g) 가 잡히면 preempt 파일로 인계.
- **파이프라인 문서**:  (단계별 역할·학습 대상·사용 데이터·검증·운영). 최종 test 평가는  가 마지막 phase 뒤 자동 실행.
- **최종 파이프라인 문서**:  (데이터 사실·전처리·합성 QC·5단계 학습·판정 기준·운영·열린 문제). 이전  는 대체됨.
- **재학습 파이프라인 완료 (2026-09-05)**: `configs/pipeline_retrain.yaml` (rigid MNI 정합 + [0,1] 정규화 + WM 2채널) p0~p4 종료, `outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt`. test 31명 평가 `outputs/eval/final_p4_joint_step3000.json`.
  - **보고 지표는 T1 단독 `generated` 경로 하나로 고정한다**: pass-SC r **0.713** · r_log 0.678 · CCC 0.058 · edge_f1 0.918. 기준선은 그룹 템플릿 r 0.945.
  - **추론에 그룹 정보를 넣는 구성(`--use-bank` latent bank, 템플릿 pair 배분)은 폐기**한다 (2026-09-05 사용자 결정). 그 구성의 0.85~0.88 은 개인 예측이 아니다.
  - 기하 (**규약 필수. `outputs/eval/w3b_protocol.json` 참조 — 규약 없는 dice 는 인용 금지**):
    whole-brain dice `wb@8000` **0.546** / `wb@20000` **0.608** (같은 체크포인트·같은 n_per_pair 16. 차이는
    채점 표본 크기뿐이고 `--n-per-pair` 8 vs 16 은 wb dice 를 0.001 도 못 바꾼다), pair dice `pair@64`
    **0.135** (천장 0.630), valid_conn 0.367 / endpoint_in_roi 0.745, 길이 r 0.182 (생성 97mm vs GT edge 158mm).
    폐기: `0.095`(pred 16 vs gt ≤256) 와 그 "천장" `0.700`(128 vs 128) 은 서로 저울이 달라 비율 해석 불가.
  - **핵심 판정 (규약 일치 재측정, `outputs/eval/w3b_baselines_matched.json`)**: 모델은 `cross_subject`
    기준선을 **넘지 못한다 — 기존 판정 유지**. wb@8000 0.546 vs 0.574±0.016, wb@20000 0.608 vs 0.629±0.016
    (N 을 올리면 모델과 기준선이 함께 오른다). pair@64 0.135 vs cross_subject 0.222 / group 0.283 / 천장 0.623.
    SC r 0.713 vs cross_subject 0.841(8k)~0.843(20k) — SC r 은 가닥 수에 거의 무관(Δ≤0.003)해 규약 영향 없음.
- **예측 SC 가 subject 마다 동일한 원인 확정 (2026-09-05, `docs/PIPELINE_06_FINDINGS.md` §B-6)**: 예측 SC subject 간 상관 0.99997 (GT 0.882). 단계별 측정 — UNet stage3 공간맵 0.575 → **`global_avg_pool` 직후 0.9996** → fc 후 0.9996 → count head 0.99997. (1) `AtmAdapter.encode_anatomy` 의 전뇌 global average pooling 이 개인차를 지우고(버리는 공간 편차 rel_spread 0.83 vs 남기는 채널 평균 0.13), (2) 그 상수 512-d 가 3,321 쌍 전부에 동일하게 들어가 전역 배율만 바꾼다(상관은 배율 불변). 크기 문제는 해결됨(‖a‖ 0.086→0.785, 1층 기여비 1/22→0.57) — 남은 건 **분산**. 다음 수순은 §2⑥ ROI 별 국소 풀링.
- **W3-a joint 학습 (2026-09-07)**: prior patch 를 학습 경로에 연결하고 (`losses/geometry.py:kl_loss(logvar_prior)`,
  `roi_atm.sample_z/prior_params`, `trainer` 의 detach 된 KL + 별도 prior 적합항 + `prior_scale` LR 그룹),
  `configs/retrain/d3_joint.yaml` 로 복원·생성 제약·prior 를 동시에 학습했다 (`scripts/49_d3_joint.py`).
  결과 `outputs/eval/w3a_joint_result.json` (d1_decoder -> d3_joint_step3000, 전후 같은 규약 = 29번의
  00:19 스냅샷 `scripts/49_eval_frozen.py`).
  - **C13 은 부분 회복**: valid_conn 0.042 -> **0.239** (p4_joint 0.367 에는 못 미침), endpoint_in_roi 0.555 -> 0.681,
    recon (test 31) 2.240 -> 3.401 (게이트 4.0 통과). **생성 wb dice 0.583 -> 0.553, pair dice 0.0727 -> 0.0678 로
    오히려 소폭 하락** -> 통과 기준 미달.
  - SC 는 개선: pass r 0.7314 -> **0.7881**, tier r small/mid/large 0.158/0.188/0.647 -> 0.171/0.196/0.732.
  - 6,000 step 까지 늘리면 valid_conn 은 0.284 로 더 오르지만 생성 기하가 무너진다 (wb dice 0.502,
    pair dice 0.049, 길이 157mm). **3,000 step 이 최적점**.
  - prior: 학습된 sigma 로 precision_ratio 45.09 -> **32.97** (같은 스크립트가 p4_joint 36.61 을 재현).
    예측 15.60 에는 못 미친다 -- joint 학습이 posterior 평균을 mu_pair 에서 더 밀어내(offset 3.41 -> 5.8)
    mu 쪽 이득을 상쇄했다. **D1 이 prior 도 망가뜨렸다는 새 사실**: p4_joint 36.61 -> d1_decoder 45.09.
- **W4 개인차 축 종결 (2026-09-07)** — `scripts/52_prior_anatomy.py`, `outputs/eval/w4a_prior_anatomy.json`.
  subject 175명 x 공유 pair 48개, subject 단위 5-fold(pair 평균도 fold 내 계산), 순열검정 200회.
  **세 타깃 x 세 feature 위치를 전부 쟀고, 유의한 칸은 하나뿐이다.**

  | 타깃 | 전역 anatomy512 | 끝점 ROI 풀링 | 경로 풀링 |
  |---|---|---|---|
  | latent 조건평균(모양) | +0.075 | +0.011 | −0.014 (p=0.26) |
  | GT SC 카운트(세기) | **+0.101** | +0.053 | −0.007 (p=0.20) |
  | 경로 변위 mm(위치) | −0.007 | +0.002 | −0.017 (p=0.29) |

  - 유일한 신호 **전역 anatomy -> SC 0.101** 은 S1-b 의 머리 크기 0.102 재현이다. T1 이 개인에 대해
    아는 것은 사실상 **머리 크기 한 줄**이다.
  - **feature 가 부족한 게 아니다**: 경로 feature 는 subject 성분 **64.1%**, subject 잔차끼리 cosine
    −0.005 로 직교. 개인 정보는 넘치는데 연결의 세기·위치와 무관하다.
  - **개인차의 소재**: subject 성분 비중이 GT SC 카운트 **18.5%** vs latent 모양 **4.4%**. 개인차는
    모양이 아니라 세기에 있다. 개인 경로 변위는 평균 5.61 mm 로 실재하나 머리 크기와도 무상관이다
    -> GT 가 QSDR 템플릿 공간이라 해부학적 개인차가 정규화 단계에서 제거되고 잔차는 정합/tractography
    잡음에 가깝다는 가설과 일치한다 (재스캔 2명뿐이라 확정 불가).
  - **정정**: W2-b 의 subject 성분 16.9% 는 n=14 과대추정 (실제 4.4%). `arch_additive` 는 anatomy
    조건부가 아니라 ROI 가법 모형이다. LayerNorm 이 개인차를 지운다는 가설은 틀렸다 (신호는 ‖a‖ 가
    아니라 방향에 있다).
  - **배선**: `prior_use_anatomy` 를 생성자 인자로 전환하고 `from_checkpoint`/체크포인트 메타/
    `trainer` 3개 호출부/`sample_z`/`generate` 까지 anatomy 를 전달한다. 0-init bit-exact 검증,
    pytest 144개 통과. 기본값은 False 유지 (켤 근거가 없다).
  - **결론(범위 한정)**: 현재 모델·현재 frozen feature·선형 readout 에서 개인 신호는 0 이다.
    인코더·조건화 구조·풀링 위치 문제는 아니다. 다만 이 프로브들은 *streamline VAE 복원용으로 학습된*
    인코더에 *선형* readout 을 건 것이라 **"T1 에 정보가 없다"는 데이터 수준 주장은 아직 검정되지
    않았다**. 그 검정은 `SC - 그룹템플릿` 잔차를 타깃으로 end-to-end 지도학습을 돌리는 것이고,
    아직 한 번도 하지 않았다.
  - **결정적 대조 (2026-09-07)**: 훈련 144명 GT SC 평균(상수, T1 미사용)이 test 31명에서 r **0.945**,
    모델 generated 는 **0.788**, **0/31** 로 전원 미달. 남의 tractogram 을 쓰는 cross-subject
    기준선(SC 0.841 / dice 0.574)보다도 낮다. -> 개인차 결손과 **그룹 재현 열위**는 별개의 두 결손이다.
- **요약 문서 2개 — 먼저 읽을 것. 나머지 docs/ 는 이력이다.**
  - `docs/PIPELINE_13_CHANGE_LOG.md` — 무엇을 어떻게 고쳤고 그 결과 숫자가 어떻게 변했나 (M1~M12, 순효과표).
  - `docs/PIPELINE_14_ENCODER_DECODER.md` — 인코더·디코더 수정 파이프라인 (E1~E5 / D-a~D-e, 게이트, 현재 상태).
  - `docs/PIPELINE_12_PROBLEM_SUMMARY.md` — 남은 문제, 실험 원장, 기준선 대조표, 확정/미확정 구분.
- **문제 목록**: `docs/PIPELINE_09_PROBLEM_INVENTORY.md` — 단계별 문제를 축 A(개인차)/B(기하)/C(지표·절차) 로 재정리. 순서 제약: A3 공간 대응 → A1 pooling → A2 조건 주입. C1·C2 는 코드 수정 없이 즉시 수정 가능.
- **해결 계획**: `docs/PIPELINE_10_RESOLUTION_PLAN.md` — S0 계측 고정 → S1 공간 결정 실험(전체 게이트) → S2 ROI 국소 조건화 → S3 파일럿 → 분기. GPU 1장이라 GPU 구간은 직렬(lock), 나머지만 병렬.
