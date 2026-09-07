# ATM 최종 Fine-tuning 전략 구현 보고서

날짜: 2026-09-02 · 기준: `docs/ATM_FINAL_FINETUNING_STRATEGY.md`
결과: **pytest 43 passed · smoke test 30/30 PASS (SC loss → T1 encoder gradient 포함) · A10 sanity training PASS · PBS job 82650 제출(Q)**

---

## 1. 변경한 파일

| 파일 | 변경 |
|---|---|
| `src/atm_sc/models/atm_adapter.py` | `rigid_UNet` 인코더를 stage1~4 로 분리, `set_unet_trainable(level)`, `encode_anatomy_grad`(stage 별 gradient checkpointing), `unet_stage_parameters`, `BundleNorm.normalize_t1(robust=True)` |
| `src/atm_sc/models/roi_atm.py` | `trainable='full'`, `unet_level`, `param_groups()`(t1_encoder/vae_encoder/decoder/heads), `param_counts()`, `anatomy_forward()` |
| `src/atm_sc/models/roi_pair_embedding.py` | 조건부 prior `prior_mu` (z ~ N(μ_pair, I)), `anatomy_gain` |
| `src/atm_sc/models/endpoint_assigner.py` | `point_dist`, `point_log_probs`, `endpoint_log_probs` (gradient 소실 버그 수정) |
| `src/atm_sc/losses/{endpoint,geometry}.py` | `endpoint_loss(log_input)`, `stream_recon_loss`(mm 단위), `kl_loss(mu_prior)` |
| `src/atm_sc/training/trainer.py` | **재작성**: anatomy leaf(UNet forward/backward step 당 1회), 그룹별 LR, `LossWeights(recon,kl,geom,endpoint,edge,corr,mag,length)`, loss 별 dL/da·그룹별 gradient norm·VRAM 기록 |
| `src/atm_sc/training/run.py` | phase 이름을 최종 전략 §7 로, `unet_level`, T1 입력 로딩, checkpoint 에 `unet_level` |
| `src/atm_sc/training/config.py` (신규) | YAML/JSON config → run 인자. 전처리 완료 subject 만 필터 |
| `src/atm_sc/data/paths.py` | `subjects()` = `.mat ∩ 폴더`, `subject_meta()` (scanner/batch2 는 .mat 에만 있음) |
| `configs/phase2_geometry.yaml … phase9_joint.yaml`, `configs/sanity_a10.yaml` (신규) | 9 phase + sanity |
| `scripts/06_smoke_test.py` | [C] trainable T1 encoder 섹션 (11 항목) |
| `scripts/14_feature_qc.py`, `17_make_splits.py`, `16_train_config.py`, `pbs_train.sh`, `13_infer_sc.py` (신규) | feature QC, split, config 실행기, PBS, whole-brain SC 추론/평가 |
| `tests/test_trainable_encoder.py` (신규, 5 tests), `tests/test_roi_pair.py`(+2) | |

`external/atm_upstream` (= `stable/stable`) 무수정.

## 2. frozen → trainable 로 바꾼 것

| 모듈 | 이전 | 지금 | 초기값 |
|---|---|---|---|
| T1 Anatomy Encoder (`rigid_UNet` 인코더 conv1_x~conv4_x + fc) | 동결 | **`unet_level` 로 stage 별 선택 (`full` = 전부)** | ATM pretrained |
| UNet segmentation 가지 (`upconv*`, `final_conv`) | 동결 | 동결 (사용하지 않는 경로) | — |
| Streamline VAE Encoder (`ConvVAE.encode`) | 학습 | 학습 (`vae_encoder` 그룹) | ATM pretrained |
| Streamline Decoder (`ConvVAE.decode`) | 학습 | 학습 (`decoder` 그룹) | ATM pretrained |
| ROI-pair Embedding + Proj + prior_mu + anatomy_gain | 학습 | 학습 (`heads`) | 신규, 0-init (시작 = pretrained 동작) |
| Edge Head / Weight Head | 학습 | 학습 (`heads`) | 신규 |
| Soft endpoint assigner / SC builder / 재샘플 / atlas 변환 | 없음 | 없음 (deterministic) | — |

그룹별 LR 기본값 (config 에서 변경): t1_encoder 1e-5 · vae_encoder 3e-5 · decoder 3e-5 · heads 1e-4.

## 3. 파라미터 수

| 설정 | t1_encoder | vae_encoder | decoder | heads | **trainable** | frozen | 총 |
|---|---|---|---|---|---|---|---|
| `vae` (frozen encoder baseline) | 0 | 0.89 M | 0.63 M | 0.59 M | **2.10 M** | 49.86 M | 51.96 M |
| `vae+unet4` (stage4) | 17.96 M | 0.89 M | 0.63 M | 0.59 M | 20.06 M | 31.90 M | |
| **`full`** | **23.71 M** | 0.89 M | 0.63 M | 0.59 M | **25.81 M** | 26.15 M | |

frozen 26.15 M 은 UNet 의 segmentation 디코더 가지(사용 안 함).

## 4. Smoke test (`scripts/06_smoke_test.py --sc-mode pass`)

30 항목 전부 PASS. [A] synthetic 15 · [B] real frozen baseline 4 · [C] trainable encoder 11:

```
SC loss -> T1 encoder gradient         PASS   dL/da(SC)=1.79e+00, UNet grad conv1_1 1.15e-02 conv4_1 2.68e-02
loss finite (전체 objective)             PASS   L_recon 29.3 L_kl 95.0 L_geom 0.15 L_endpoint 17.9 L_edge 0.693 L_corr 0.76 L_mag 3.05 L_length 1.01
backward / gradient finite             PASS
T1 encoder gradient                    PASS   gnorm_t1_encoder 6.06e+01
VAE encoder gradient                   PASS   5.76e+01
decoder gradient                       PASS   1.67e+02
ROI-pair embedding gradient            PASS
edge/weight head gradient              PASS
T1 encoder 파라미터 실제 변경             PASS   conv1_1/conv2_1/conv4_1/fc Δ = 2.0e-05
VAE enc / decoder 파라미터 변경          PASS
UNet segmentation 가지 동결             PASS
```
peak VRAM 15.97 GB (full UNet, checkpointing) · step 2.8 s. frozen baseline [B] 도 그대로 PASS. `pytest -q`: **43 passed**.

## 5. T1 encoder gradient 확인

- anatomy feature 를 leaf 로 분리해 pass 별 dL/da 를 기록: SC pass **13.5**, recon pass 47→30, edge pass 0 (edge head 마지막 층 0-init → 초기에는 입력 gradient 0, 학습되면 생김).
- SC loss 만 켠 step 에서도 conv1_1 gradient 1.2e-2 → **SC → streamline → decoder → T1 encoder 경로 확인**.
- UNet forward/backward 는 step 당 정확히 1회 (chunk 마다 재실행 없음).

## 6. Feature QC (`scripts/14_feature_qc.py`, SyN 정합 10명, robust 정규화)

| encoder | ‖a‖ | cosine mean / min | Pearson | Euclid/‖a‖ | var>1e-6 dims | PCA top5 | 같은 scanner / 다른 scanner cosine |
|---|---|---|---|---|---|---|---|
| frozen pretrained | 0.089 | **0.954** / 0.858 | 0.954 | 0.29 | 103 / 512 | .52 .22 .13 .09 .02 | 0.958 / 0.953 |
| fine-tuned 100 step (sanity) | 0.086 | 0.952 / 0.858 | 0.952 | 0.30 | 101 | .47 .24 .18 .08 .02 | 0.957 / 0.950 |

해석: 완전 collapse 는 아니다(subject 간 변동이 4~5 개 주성분에 존재, Euclid 가 norm 의 29 %). 그러나 변동이 작고, scanner 구분력은 거의 없다(같은/다른 scanner cosine 차 0.005). 100 step·lr 1e-5 로는 feature 가 거의 안 바뀌었다 — 판별력 증가 여부는 full training 후 다시 재야 한다. 결과: `outputs/qc/feature_qc.json`.

## 7. A10 sanity training (`configs/sanity_a10.yaml`: 3명, 100 step, `full` unfreeze, phase `sc`)

| step | L_recon | L_kl | L_endpoint | L_edge | L_corr | L_mag | sc_r | step s |
|---|---|---|---|---|---|---|---|---|
| 1-20 | 20.82 | 63.3 | 18.05 | 0.6931 | 0.756 | 3.05 | 0.244 | 3.09 |
| 41-60 | 8.72 | 56.8 | 17.43 | 0.6928 | 0.703 | 2.91 | 0.297 | 1.82 |
| 81-100 | **7.92** | 53.7 | 16.09 | 0.6908 | 0.701 | 2.87 | 0.299 | 1.76 |

L_recon 감소 · 모든 값 유한 · NaN/OOM/divergence 없음 · T1 encoder 파라미터 변경(conv1_1 Δ 4.1e-4, conv2_1 5.6e-4, conv4_1 5.8e-4, fc 5.8e-4; `final_conv` 0) · decoder Δ 3.2e-3.

## 8. Loss / gradient norm (sanity, first20 → last20)

| | dL/da | gnorm |
|---|---|---|
| SC pass (G) | 13.5 → 13.4 | — |
| recon pass (R) | 47.5 → 30.3 | — |
| edge pass (E) | 0 → 1e-4 | — |
| t1_encoder | | 46.4 → 30.2 |
| vae_encoder | | 41.4 → 23.9 |
| decoder | | 127.0 → 83.6 |
| heads | | 58.4 → 54.9 |
| total (clip 50) | | 154 → 108 |

recon 을 mm 단위로 바꾼 뒤 항 간 gradient 가 같은 자릿수(10¹~10²)다. decoder 가 가장 크다.

## 9. VRAM / step time

full unfreeze: peak **16.16 GB** (23 GB A10), step **2.26 s** 평균(warm 1.75 s), nvidia-smi GPU util **97 %** (5 s 샘플 49개). frozen encoder: peak 0.6 GB, step < 0.1 s.
전처리: SyN ~2~3 min/subject(1 core), `02` ~95 s, `03` ~25 s.

## 10. GPU job

- 스케줄러: **PBS** (`qsub`/`qstat`). GPU 큐 `base_g` (walltime ≤ 24 h, **사용자당 running GPU 1개**).
- 스크립트: `scripts/pbs_train.sh` (`select=1:ncpus=8:ngpus=1`, walltime 24 h) + `configs/phase2_geometry.yaml` (seed 0, subjects = `outputs/splits/train.txt` 144명 중 전처리 완료분, out `outputs/checkpoints/phase2_geometry`).
- split: `outputs/splits/` train 144 / val 31 / test 31, group × scanner_proto stratify (`scripts/17_make_splits.py`).
- **제출: job `82650.KITSM02`, 상태 Q**. `comment = Not Running: Insufficient amount of resource: ngpus (R:1 A:0)` — 현재 interactive 세션(`82564`, queue remote)이 사용자 GPU 1개를 점유 중이라 그 세션이 끝나야 R 로 넘어간다. RUNNING 확인과 초기 로그 검증은 **아직 못 했다**. 확인 명령: `qstat -f 82650.KITSM02`, 로그 `outputs/pbs/atm_sc_train.o82650`, `outputs/checkpoints/phase2_geometry/log.jsonl`.

### 10b. 로컬 A10 → PBS 이어받기 (2026-09-03)

- `scripts/19_train_pipeline.py`: 9 phase 를 순서대로 실행, `outputs/checkpoints/pipeline_state.json` 에 진행 상태, phase 마다 `<phase>_latest.pt` (모델 + optimizer + step + numpy/torch RNG) 를 200 step 마다 저장. 재실행 시 같은 phase 의 latest 에서 정확히 이어받고(검증: step 4 에서 kill → 5~12 재개 → 최종 checkpoint), 완료된 phase 는 건너뛴다. phase 종료마다 val 3명 평가 → `val_metrics.jsonl`.
- lock (`outputs/checkpoints/pipeline.lock`, step 마다 heartbeat): 같은 호스트에 살아있는 프로세스가 있으면 두 번째 실행은 종료, 다른 호스트는 15 분 이상 heartbeat 가 없을 때만 인수.
- **로컬 A10 에서 실행 중** (`outputs/checkpoints/pipeline_local.log`, 시작 시 train 144명 중 전처리 완료 118명). 82650 은 취소하고 **82753.KITSM02** (`qsub -v PIPELINE=configs/pipeline.yaml scripts/pbs_train.sh`) 로 재제출 — 이 세션이 끝나 로컬이 죽으면 GPU 가 풀리고 job 이 latest 에서 이어받는다. 상태: Q (ngpus 사용자 한도).
- 전처리를 CPU job 으로 이관: `scripts/pbs_preprocess.sh` → **82756.KITSM02** (`base_8`, ncpus 8, 4 subject 병렬 × ITK 2 스레드, `--reverse` 로 로컬 배치와 반대 방향) — 제출 즉시 R. R 전환을 감시하는 watcher 가 로컬 배치를 자동 중단(충돌 방지). `std_q`, `base_32` 는 이 계정에 권한 없음.
- 로컬 A10 학습 현황: phase 2 geometry 3000 step 완료 (val 3명: pair_acc 0.009, pass-SC r_w 0.50, 길이 110 mm — endpoint loss 이전이라 예상 범위) → phase 3 t1_encoder 진행 중 (full UNet, 118 subject, 2.6 s/step, VRAM 18.6 GB).
- 모니터: `scripts/20_monitor.py` (`--watch 60`) — 전 phase 진행률·s/step·ETA(대기 phase 는 실측 + loss 별 추가 비용 추정), 설정(LR/λ), 현재 phase loss·gradient·dL/da 추이(sparkline), phase 별 val 표, lock/GPU/PBS/전처리 상태.
- 전처리 배치 실패 1명: sub-182427 (정합 후 streamline 60.8 % 만 뇌 안, assert) → 자동 제외.

## 11. 발견된 문제

1. **T1 강도 스케일**: PPMI native max 878~203,163 (230배). upstream 고정 상수로는 feature 가 40배 뛰는 subject 발생 → robust 정규화(뇌 p99.5→0.6) 로 해결.
2. **frozen feature 의 subject 판별력이 약함** (cosine 0.95). fine-tuning 이 이를 키우는지는 full training 후 §6 QC 로 판정.
3. **prior z 생성 실패 → 조건부 prior 로 해결**: recon 만으로는 decoder 가 pair 조건을 무시(pair acc 0). z ~ N(μ_pair, I) 도입 후 단일-subject 파일럿에서 prior z pair 정확도 0 → **0.63**, ROI 0.80/0.83, 끝점–GT 6.0 mm.
4. **endpoint loss gradient 소실**: 확률 clamp 로 먼 끝점에서 gradient 0 → log-softmax 경로로 수정 (회귀 테스트 추가).
5. **길이 과대**: 생성 streamline 96~108 mm vs GT 59 mm. L_length(Phase 8) 로만 간접 제약 — 파일럿에서 length MAE 32.6 mm. pair 별 길이 prior 추가를 검토.
6. **SC 절대 스케일**: 파일럿 whole-brain pass-SC weighted r=0.945 이지만 CCC 0.10 (sum 0.39 M vs GT 6.9 M). `sc_magnitude_loss` 가 총합 정규화라 절대량을 안 배움. weight head 의 스케일 학습(정규화 없는 magnitude 항) 필요.
7. **컴퓨트**: 이 세션은 CPU 1 core → 전처리 배치 ~5 min/subject 순차(13/206 완료, ~16 h 남음). GPU job 은 세션 종료 전까지 Q.
8. `.mat` 의 32명은 tracto zip 에 없음 → 206명으로 확정.

## 12. 다음 단계

1. 이 interactive 세션 종료 → `82650` 실행 확인 → `phase2_geometry` 로그로 L_recon 수렴 확인
2. 전처리 배치 완료(206명) 후 phase 3~9 config 순차 제출 (`configs/phase3_t1_encoder.yaml` …), 각 phase 후 val 31명 SC corr/CCC·pair 정확도 기록
3. Phase 3 후 feature QC 재실행 → frozen vs fine-tuned 판별력 비교 (§16 ablation A vs B)
4. §11-5/6 개선: pair 별 길이 prior, 절대 스케일 magnitude 항
5. val 기반 λ/LR 튜닝, test 31명 최종 1회 평가, CoRNN 대비 benchmark

## 13. 코드 대조 검증 (2026-09-03, phase 3 진행 중)

§1~§10 의 주장을 현재 코드·checkpoint 로 다시 확인한 결과 (CPU, 학습 프로세스와 병행).

| 항목 | 확인 방법 | 결과 |
|---|---|---|
| upstream 무수정 | `stable.zip` 내 코드 144개 파일 md5 vs `stable/stable/` | 144 identical / 0 differ |
| warm-start | `ROIPairATM` 초기 `atm.net.state_dict()` vs `atmvae_AF_L.pth` | bit-exact |
| trainable 범위 | `requires_grad` + `param_groups()` | UNet conv1_x~conv4_x+fc → `t1_encoder`, `upconv*/final_conv` 동결, 그룹 서로소 |
| 파라미터 수 | `param_counts()` | §3 과 동일 (23.71/0.89/0.63/0.59 M, trainable 25.81 M) |
| 신규 head 0-init | proj 마지막층·prior_mu·edge 마지막층 = 0, anatomy_gain = 1, weight head softplus = 1 | OK |
| BN | `model.train()` 후 UNet BN 5개 eval 유지, VAE BN train | OK |
| phase 2 checkpoint (`geometry_step3000.pt`) | pretrained 대비 max\|Δ\| | t1_encoder 0, seg 가지 0, vae_enc 3.5e-2, dec 5.6e-2, heads 5.1e-1; optimizer 그룹 3개 (3e-5/3e-5/1e-4) |
| phase 3 checkpoint (`t1_encoder_latest.pt`, step 600) | 동일 | t1_encoder 2.6e-3 (conv1_1 1.3e-3, fc 1.9e-3), seg 가지 0; optimizer 그룹 4개 (1e-5/3e-5/3e-5/1e-4); NaN 없음 |
| step 당 UNet 1회 | `trainer.py` `anatomy_forward` 1회, `a_full.backward(dLda)` 1회 | OK |
| log | phase 3 `gnorm_t1_encoder` 17.6→15.8, `dLda_R` 17.0→15.4 (모두 > 0) | gradient 가 T1 encoder 에 도달 |
| pytest | `CUDA_VISIBLE_DEVICES="" pytest -q tests/` | 43 passed (8.7 s) |
| phase config | `configs/phase2~9` | §7/§8 순서·LR 일치 (phase 9 는 1/2~1/3 LR) |

미확인 (GPU 가 학습 중이라 실행 불가): smoke test [C] 재실행, fine-tuned encoder feature QC (phase 3 완료 후). 미구현: §15 의 Spearman·length 행렬 지표(val 루프), §16 ablation A/B/C.

## 14. 소/중/대 · 피질/피질하 균형 (2026-09-03, phase 4 도중 적용)

**문제** (206명 GT pass-SC 분석): ctx-ctx 2145 pair 가 질량 87.4 %, ctx-sub 1056 pair 11.7 %, sub-sub 120 pair 0.9 %. edge 값은 heavy tail (상위 5 % edge 가 질량 50 %, 최대 5.7 만 vs 중앙값 84). 이전 `L_SC_corr` 는 raw count 의 whole-brain Pearson 이라 큰 ctx-ctx edge 몇 개가 결정했고, val 은 count 상위 64 pair (평균 ctx-ctx 61.7 / ctx-sub 2.2 / sub-sub 0.0) 만 평가했다. sub-sub 연결의 63 % 는 endpoint bundle 이 없는 통과 edge 다.

**변경** (`src/atm_sc/data/roi_groups.py` 신규):
| 항목 | 내용 |
|---|---|
| block | ROI 1~66 피질 / 67~82 피질하 (PD25: RN, SN, STN, caudate, putamen, GPe, GPi, thalamus) → ctx-ctx / ctx-sub / sub-sub mask |
| tier | GT edge 강도(.mat pass 값) ≤100 소 / ≤1000 중 / >1000 대 (206명 nonzero edge 33/67 백분위 92/1282 반올림, 고정 경계) |
| `L_SC_corr` | block 별 **log1p Pearson** 의 평균 (`sc_corr_group_loss`, `TrainConfig.sc_groups='block'`, `sc_log_corr=True`, `block_weights`) |
| `L_SC_mag`, `L_length` | block 별 masked 평균의 평균 (`masks=`) |
| pair 샘플링 | recon/edge 양성 pair 를 tier 마다 같은 개수 (`TrainConfig.pair_sampling='tier'`; 'log'/'uniform' 은 이전 방식). sub-100268: log 0.08/0.21/0.71 → tier 0.34/0.34/0.33 |
| 생성 pass | 변경 없음 (모든 pair × 4, pair 당 동일 가중) |
| val | count 상위 64 대신 block × tier 9 cell × 8 pair 층화; 전체·block·tier 별 pair_acc, SC r/r_log/CCC/log-MAE (`val_pairs_per_cell`). `19_train_pipeline.py --reval ckpt…` 로 기존 checkpoint 재평가 |
| 로그/모니터 | `corr_r_<block>`, `sc_rlog_<block>`, `recon_tier_<tier>`; 모니터에 block/tier 표 |

**검증**: pytest 49 passed (`tests/test_roi_groups.py` 6개: block 분할 2145/1056/120, tier 경계, 균등 샘플링, sub-sub 를 뒤섞어도 raw whole-brain corr 은 <0.02 인데 group loss 는 >0.2, group magnitude/length 가 희석되지 않음). GPU sanity 20 step (`sc` phase, full UNet): NaN 없음, VRAM 16.2 GB, 1.8 s/step, block r 기록 확인.
**적용**: phase 4 step 2250 에서 driver 중단 → `endpoint_latest.pt`(step 2200) 에서 재개. phase 4 는 recon 샘플링만 바뀌고, SC loss 변경은 phase 6 부터 효력.

## 15. Route loss + PASS-SC 세분화 + GESTA 증강 (2026-09-03)

기준 문서: `docs/ATM_ROUTE_LOSS_PASS_SC_GESTA_DETAILED_STRATEGY.md` (§12–25, §54–65), `docs/ATM_TRACTOLEARN_GESTA_BALANCED_TRAINING_STRATEGY.md`.

### 15.1 문제와 역할 분담

| 문제 | 장치 |
|---|---|
| endpoint 감독은 "출발·도착"만 본다. SUB-SUB 연결의 63 % 는 다른 bundle 이 **지나가며** 생기는 pass-edge | **Route loss** (streamline 별 통과 ROI) |
| whole-brain corr 하나면 ctx-ctx(질량 87 %) 가 지배 | **block 별 log-Pearson** + λ_global 항 |
| edge 존재 여부와 크기가 섞임 | **SUB-SUB presence loss** (1 − exp(−SC/scale) 의 BCE) |
| endpoint bundle 크기 불균형 (N<20 인 pair 가 46 %) | **GESTA latent 증강** + pair 균형 노출 |
| Weight Head 가 geometry 대신 weight 로 SC loss 를 줄이는 shortcut | magnitude 를 route/corr 뒤 단계로 미룸 (§24) |

### 15.2 추가·수정한 코드

| 파일 | 내용 |
|---|---|
| `src/atm_sc/losses/route.py` (신규) | `route_bce`/`route_dice`/`route_loss`(log 확률 입력), `route_metrics`, `pass_presence_loss`, `presence_metrics` |
| `models/endpoint_assigner.py` | `visit_log_probs` — 통과 확률을 **log 공간**에서 max 집계 (확률 공간이면 먼 ROI 에서 gradient 0) |
| `scripts/24_build_visitation.py` (신규) | GT streamline 별 통과 ROI 를 hard atlas 로 계산 → `outputs/roi_pairs/<sub>/visit.npz` (packed bits + pair marginal) |
| `data/dataset.py` | `visitation(k)`, `pair_marginal`, `has_visitation`, `synthetic_pair(k)` |
| `training/trainer.py` | 복원 pass 의 per-streamline route loss, 생성 pass 의 pair-marginal route loss, presence 항, `route_tau/route_mode/route_pos_weight/presence_*`, λ_syn 가중 recon |
| `losses/geometry.py` | `stream_recon_loss(weights=)` — synthetic streamline 은 λ_syn(기본 0.5) 로 감쇠 (§47) |
| `data/balanced_pair_sampler.py` | pair 균형 batch + real/synthetic 혼합 + 통과 ROI 동반 반환 |
| `generative/{latent_sampler,bundle_augmenter}.py`, `filtering/t1_streamline_filter.py`, `evaluation/balance_metrics.py` | GESTA 샘플러 / 증강 / T1-only 필터 / 균형 진단 지표 |
| `training/run.py` | phase `route`, `route_edge`, `route_sc_corr`, `route_sc_presence`, `route_sc_mag`, `route_full` (기존 phase 는 불변) |
| `scripts/19_train_pipeline.py` | val 에 route F1/recall, Spearman, weak-edge recall, **SUB-SUB endpoint-supported vs pass-only** 분리 |
| `configs/route/r1~r8.yaml`, `configs/pipeline_route.yaml`, `scripts/23_after_baseline.sh` | ROUTE §25 순서의 독립 파이프라인 + baseline 종료 후 자동 실행 |

### 15.3 실측·검증

- **GT 통과 정보가 GT pass-SC 와 일치**: sub-100001 에서 co-visitation 으로 만든 SC vs `.mat` pass-SC **r = 0.941, log r = 0.971**. 전 subject 206명 생성 완료(3 s/subject, 0.2 MB, 평균 통과 ROI 5.1).
- **route_tau 보정**: GT streamline 으로 잰 통과 ROI 확률 중앙값이 tau 0.5/1/2/5 에서 0.98/0.83/0.58/**0.32**. tau 가 크면 완벽한 경로도 target 1 에 닿지 못하므로 **기본값 1.0**(log 공간이라 gradient 는 유지). 지표 임계값 0.3.
- **끝점은 맞고 중간만 다른 streamline** (§18 도로 비유) 회귀 테스트: endpoint loss 변화 < 1e-4, route loss 4.3배 증가.
- 실제 subject CPU step: `L_route` 1.43, `L_route_gen` 1.82, `L_presence` 2.68 모두 유한, decoder/heads gradient > 0.
- pytest: route 7 + route 통합 5 + latent sampler 7 + filter/balance 10 + sampler 5 = 신규 34개 포함 전체 통과.

### 15.4 남은 것

증강 cache 생성(`scripts/22`)과 ROUTE 파이프라인 실행은 GPU 가 필요해 baseline(9 phase, 예상 9/4 00:00) 종료 후 `scripts/23_after_baseline.sh` 가 자동으로 이어받는다. 이후 §61–62 ablation (route 유무, GESTA 유무) 으로 채택 여부를 판정한다.

## 16. SC edge-aligned segment 분해 (2026-09-03)

기준: `docs/ATM_SC_EDGE_ALIGNED_BUNDLE_DUAL_REPRESENTATION_STRATEGY.md`.

### 16.1 문서의 전제 검증 (§26–30, §42) — 결론이 바뀌는 부분

전체 tractogram(sub-100001, 1,000,000 streamline)으로 두 정의를 계산해 GT `.mat` 와 비교했다.

| GT SC 정의 | r | log r | 합계비 | nonzero edge (GT 2785) | segment/streamline |
|---|---|---|---|---|---|
| Case A 인접 ROI transition | 0.781 | 0.569 | 0.66 | **700** | 4.70 |
| **Case B 같은 streamline 의 모든 ROI 쌍** | **0.9986** | **0.9912** | 0.98 | **2774** | 6.98 |

**이 데이터의 GT 는 Case B 다.** 따라서 문서 §4–§7 의 "A→C→D→B 를 A-C / C-D / D-B 로 분해하면
segment 수 = SC edge 값" 이라는 핵심 주장은 **그대로는 성립하지 않는다**. 인접 transition 만 쓰면
GT edge 의 25 % 만 생성되고 상관도 0.78 로 떨어진다. §30 의 두 번째 경우("all-pass-pair 방식")대로
**같은 streamline 안에서 ROI_i 구간과 ROI_j 구간을 잇는 부분경로**를 모든 방문 쌍에 대해 만들어야 한다.

### 16.2 구현한 것

| 파일 | 내용 |
|---|---|
| `src/atm_sc/data/edge_segments.py` (신규) | `dwell_intervals`(경계 jitter 제거, min_dwell), `streamline_segments`(모든 방문 ROI 쌍의 부분경로, 되돌아오면 가장 긴 것 1개), `decompose_bundle`, `segment_sc`(검증용) |
| `scripts/26_build_edge_segments.py` (신규) | subject 별 `outputs/roi_pairs/<sub>/edge_segments.npz` (pair_ids, offsets, segments fp16, lengths, count_full) + GT 대비 QC |
| `data/dataset.py` | `has_edge_segments`, `edge_pair_ids`, `edge_count_full`, `edge_segments(e)`, `edge_segment_lengths(e)` |
| `tests/test_edge_segments.py` | dwell/jitter, 인접이 아닌 전체 쌍(6개) 생성, 부분경로 구간, 짧은 segment 제거, SC 재구성 |

실측(sub-100001): 150,721 streamline → **1,713,357 segment (11.4/streamline), edge 2425개**, 32점 재샘플,
edge 당 cap 128 저장 시 **33 MB · 99 s/subject**. 분해 SC vs GT `.mat` **r 0.946 / log r 0.954, GT edge 재현 0.869**
(bundles.npz 가 endpoint-pair 당 256 로 잘려 있어 희귀 pass-edge 가 덜 잡힌다).

### 16.3 아직 하지 않은 것 (평가)

문서의 dual representation 전체(§13, §36)는 **segment 전용 생성 분기**가 필요하다. 현재 decoder 는
[128,3] full streamline 을 내도록 학습돼 있어 segment 생성에는 (a) 별도 decoder 또는 (b) 길이/모드 조건이 붙은
공유 decoder 가 필요하며, 이는 재학습을 수반하는 구조 변경이다. Edge Count Head(§20–22)는 상대적으로 작고
현재 미해결 문제(SC 절대 스케일 CCC≈0.02)를 직접 겨냥하므로 우선순위가 더 높다.

## 17. Segment 분기 + Edge Count Head 통합 (2026-09-03)

| 구성요소 | 파일 | 역할 |
|---|---|---|
| Edge Count Head | `models/edge_count_head.py`, `losses/edge_count.py` | (anatomy, ROI_i, ROI_j) → log count. SC edge 값을 직접 예측 (EDGE_ALIGNED §19–22) |
| segment 균형 표집 | `data/segment_sampler.py` | edge 크기의 제곱근 비례 노출(5000:20 → 5.4:1), 32점 → 128점 호길이 보간 |
| 모드 조건 | `models/roi_pair_embedding.py` | `mode_emb`/`mode_prior` (0 full / 1 segment). 0-init 이라 도입 시점 동작 불변 |
| 학습 배선 | `training/trainer.py` | `[S]` segment 복원·KL·기하·끝점, `[C]` block 균형 log-count 손실 |
| 단계 | `training/run.py` | `seg_count`, `seg_count_mag`, `seg_full` |
| checkpoint 호환 | `models/roi_atm.py` `load_checkpoint` | 새 head 는 초기값 사용, 없는 키만 허용 (구 checkpoint 에서 이어받기) |

검증: pytest **104 passed**. 실제 subject CPU 통합 step(seg_count) 정상, 모든 손실 유한.
**Edge Count Head 단독 60 step(1명, CPU)**: log-MAE 2.74 → 1.98, **CCC 0.000 → 0.167** — 총합 정규화 magnitude 로는
수천 step 뒤에도 CCC 0.02 였던 절대 스케일 문제를 직접 겨냥한다.

## 18. SC 절대 스케일 (2026-09-03)

**진단**: test subject 1명(phase6 checkpoint)에서 예측 SC 합 161,056 vs GT 7,673,899 — 정확히 **48배** 작다.
배율 하나만 곱하면 지표가 이렇게 바뀐다.

| 조건 | r | CCC | log-MAE | RMSE |
|---|---|---|---|---|
| 현재 | 0.831 | **0.024** | 2.78 | 5,856 |
| 전역 배율 ×48.6 | 0.831 | **0.817** | 1.86 | 3,044 |
| block 별 배율 | 0.858 | 0.848 | 1.78 | — |

즉 절대 스케일 문제의 대부분은 상수 하나다. 원인은 `sc_magnitude_loss` 의 총합 정규화로,
어떤 배율에도 손실이 0이 되어(테스트로 확인) weight head 가 배율을 배울 신호가 없었다.

**수정**: `losses/sc_magnitude.py` 에
- `sc_scale_loss` — |log(Σpred) − log(Σgt)|, block 별 평균. 배율을 직접 학습.
- `sc_rmse_loss` — GT 표준편차로 무차원화한 RMSE(또는 log/raw). 원시 RMSE 는 상위 1 % edge 가 오차의 32 %를
  차지해(실측) 큰 edge에 쏠리므로 무차원화 + block 평균을 기본값으로 둔다.
- `TrainConfig.sc_mag_normalize='none'` (s4 이후) — 크기 손실 자체를 절대값 비교로.
- 안전장치: 절대 항은 전체 pair 를 생성할 때만 허용(`max_pairs_per_step=None` assert). 일부만 생성하면
  분자만 작아져 배율을 과대 학습한다.

**검증**: pytest 신규 5개 포함 통과. 실제 subject 25 step(CPU, mag 만): 총합비 **0.0032 → 1.17**,
weight head 평균 1 → 1400. `L_scale` 5.04 → 0.45.

