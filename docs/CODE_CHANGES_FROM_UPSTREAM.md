# Upstream ATM → 이 프로젝트: 모듈별 코드 변경 정리

기준: `stable/stable/` (Zenodo 15792527, 공식 ATM) ↔ `src/atm_sc/` (이 프로젝트)
원칙: **upstream 파일은 한 줄도 수정하지 않았다.** 모든 변경은 import → wrapper → 대체 구현 순으로
바깥에서 이뤄졌다. 아래 표의 "upstream" 줄 번호는 `stable/stable/` 기준이다.

---

## 0. 한 장 요약

```
upstream (공식 ATM)                              이 프로젝트
────────────────────────────────────────────    ─────────────────────────────────────────────
T1 --rigid(ANTs)--> MNI-2009c 1mm               T1 --SyN(antspyx)--> MNI-NLin6 --> W 격자
30 anatomical bundle x (모델 + KDE + 상수)       decoder 1개 + ROI-pair embedding (82 ROI)
bundle 마다 UNet 실행 (subject 당 30회)          UNet 인코더만, subject 당 1회, 캐시
z ~ KDE(447k 학습 latent)                        z ~ N(0, I)  (학습 시 KL)
좌표 상수 = bundle 별 bbox                        좌표 상수 = whole-brain bbox
출력 -> numpy -> tckedit/MATLAB/scilpy 후처리     출력 -> torch 유지 -> soft SC -> loss
학습 코드 없음                                    trainer (재구성 + endpoint + edge + SC)
평가 없음                                        .tt.gz GT, pass/end SC, 82-ROI 지표
```

---

## 1. `infer.py` — 함수별 대응

| upstream 함수 (줄) | 하던 일 | 이 프로젝트 | 어떻게 바뀌었나 |
|---|---|---|---|
| `affine_registration` (21-83) | `antsRegistrationSyN.sh -t r` (**rigid**) 로 T1→MNI ICBM152-2009c, GM/WM 마스크 변환 | `data/prepare_t1.py: register_to_template, prepare_subject` | ANTs 바이너리 없음 → `antspyx`. **rigid → SyN** 으로 바꾸고 대상 템플릿을 **MNI152NLin6** 로 교체 (GT tractogram 이 QSDR 템플릿 공간 = NLin6 이기 때문. atlas 로 GT SC 를 r=0.9986 재현해 확인). 결과를 `spaces.W_SHAPE` 격자로 재샘플 (`resample_to_W`). `mode='rigid'` 로 원본 방식도 유지 (ablation 용). GM/WM 마스크는 쓰지 않는다 (필터링 단계 제거) |
| `get_t1w_input` (86-112) | `supp/{bundle}_{min,max}_vals_*.npy` 스칼라로 min-max 정규화 | `models/atm_adapter.py: BundleNorm.normalize_t1` | 식은 동일 (`(x-min)/(max-min+1e-6)`). 상수는 `BundleNorm.from_upstream(bundle)` 로 그대로 읽는다. ROI-pair 모델은 `init_bundle`(기본 AF_L) 의 T1 상수를 쓴다 |
| `gen_streamlines` (115-160) | 모델 로드 → `atm.unet(t1)` → `repeat(3000,1)` → `ae.decode` → `.cpu().numpy()` → 역정규화 | `ATMBundle.encode_anatomy` + `decode_mm` / `decode_chunks` / `generate` | 다섯 가지를 고쳤다. ① `.eval()` 강제 (upstream 은 미호출 → Dropout3d 활성, 비결정적; D2) ② `repeat(3000,1)` 하드코딩 → 실제 N (D3) ③ 좌표 상수를 `data/` 가 아니라 실제 위치인 `supp/` 에서 읽음 (D4) ④ 역정규화를 numpy 가 아니라 **torch 로** 수행해 gradient 유지 (D5) ⑤ UNet 은 **인코더 가지만** 실행 (`_unet_encoder_only`; segmentation 가지가 A10 23GB 에서 OOM. CPU 전체 forward 와 bit-exact 확인) |
| `main` 305-306 `kde.sample(n)` | bundle 별 KDE(tophat, bw=1, 447,000×64)에서 latent 샘플 | `sample_latents`, `load_kde` (재현용) / `ROIPairATM.sample_z` (학습·추론) | 재현 스크립트에서는 KDE 를 그대로 쓰되 `random_state` 를 **메서드 인자**로 넘겨 재현성 확보 (속성으로 넣으면 무효 — 실측). KDE 로드 30 s → 캐시. ROI-pair 모델은 KDE 가 bundle 별이라 의미가 없으므로 **N(0,I)** 로 교체 |
| `main` 315-325 (trk/tck 저장) | dipy `StatefulTractogram` + 참조 trk 헤더 | `scripts/00_reproduce_atm.py` | nibabel 만으로 저장 (dipy 의존 제거). 헤더는 같은 참조 trk 에서 |
| `filtering` (163-212) | MRtrix `tckedit` 로 GM/WM 마스크 필터 + 20 mm 최소 길이 | **없음** | MRtrix 없음. 학습 그래프에 들어갈 수 없는 hard 연산이고, 학습 목표(SC)는 soft endpoint 로 대체. 평가 시 필요하면 mask 필터를 numpy 로 추가할 수 있다 |
| `trimming` (215-229) + `matlab_post/bundle_trimming.m` | MATLAB 로 백질 표면 밖 끝점 절단 | **없음** | MATLAB/FreeSurfer 없음. 128점 등간격 표현을 깨뜨리고 미분 불가. endpoint loss 가 끝점을 ROI 로 끌어당기는 역할을 대신한다 |
| `to_native` (232-271) | scilpy 로 MNI→native 역변환 | **없음** | GT SC 가 템플릿 공간에서 정의되어 있으므로 native 로 돌아갈 필요가 없다. 필요 시 `register_to_template` 가 저장한 ANTs 변환으로 가능 |
| `main` 374-406 (30 bundle 루프) | bundle 마다 `main(args)` — **subject 당 UNet 30회** | `training/run.py: anatomy_feature` | ROI-pair 모델은 UNet 1개 → **subject 당 1회**, `outputs/cache/{sub}_anat_{bundle}.npy` 캐시. 30 bundle 이름은 `atm_adapter.BUNDLES` 로 보존 (재현·벤치마크용) |

## 2. `model/model.py` — 클래스별 대응

| upstream (줄) | 이 프로젝트 | 변경 |
|---|---|---|
| `FiLM` (9-39) | 그대로 import | 무수정. **ROI-pair 조건이 들어가는 자리**가 이것이다: `gamma_fc/beta_fc(conditioning_input)` 의 입력 [N,512] 에 `cond = gain·a + Proj(Emb(a)+Emb(b))` 를 넣는다 |
| `UNet` (42-141) | 사용 안 함 | `rigid=True` 경로만 쓴다 (체크포인트가 `rigid_UNet` 키) |
| `rigid_UNet.forward` (191-250) | `ATMBundle._unet_encoder_only` | 193-224 줄(인코더 → global_avg_pool → fc)을 **그대로 옮겨 적고** 227-248 줄(디코더/segmentation) 은 실행하지 않는다. `anatomical_condition` 은 디코더 가지의 영향을 받지 않으므로 수치 동일 (오차 0.0 확인). `Upsample(size=(49,58,49)/(97,115,97)/(193,229,193))` 하드코딩 때문에 입력 격자 W 는 (193,229,193) 고정 |
| `ATMVAE.forward` (267-274) | 사용 안 함 | upstream 은 `unet` 과 `ae` 를 한 forward 로 묶는데, 이러면 streamline 배치마다 UNet 이 돈다. 인코더 결과를 캐시하려고 `unet` 과 `ae` 를 **따로** 부른다 (`encode_anatomy` / `decode_mm`) |
| `ConvVAE.decode` (347-367) | `ATMBundle.decode_mm` 가 호출 | 무수정. 입력 `anatomical_info` 자리에 ROI-pair 가 합쳐진 `cond` 가 들어간다. 출력 `[N,3,128]` tanh → `permute` → `(s+1)/2·(max-min)+min` (torch) |
| `ConvVAE.encode` (323-345) | `ATMBundle.encode_streamline` | 무수정. **L_ATM 재구성**에 사용 (GT streamline → μ, logσ²). upstream 은 학습 코드가 없어 이 경로를 쓰는 곳이 없었다 |
| `ConvVAE.reparameterize` (375-378) | `ROIPairATM.reparameterize` | 동일 식 |
| `ATMVAE(512, 64, True)` (infer.py:128) | `ATMBundle.__init__` | `strict=True` 로드 검증 (missing/unexpected 모두 `[]`), `models_dir` 인자화 |

## 3. upstream 에 없어서 새로 만든 것

| 파이프라인 항목 | 파일 | 근거 |
|---|---|---|
| ROI-pair 조건 (§9) | `models/roi_pair_embedding.py` | `Emb(a)+Emb(b)` (순서 불변) → MLP → [N,512], 마지막 층 0-init → **시작 시 cond == a** (pretrained 보존). 학습 가능한 `anatomy_gain` (pretrained feature ‖a‖≈0.09 로 매우 작아서) |
| streamline weight (§16) | `models/streamline_weight_head.py` | `softplus(MLP(cond, z))`, init w=1 |
| edge 존재 (§10) | `models/edge_head.py`, `losses/edge.py` | `MLP(a, pair_vec)` → BCE, init p=0.5, pos_weight 지원 |
| 조립 | `models/roi_atm.py: ROIPairATM` | UNet 동결, 좌표 상수 = NLin6 brain-mask bbox+5mm (`brain_box_mm`), `trainable='vae'|'decoder'` |
| soft endpoint 할당 (§13) | `models/endpoint_assigner.py` | ROI 거리맵 + `grid_sample` + `softmax(-d/τ)`. `endpoint_probs`(양 끝점) / `visit_probs`(통과, GT pass 정의용) / 배경 클래스 `d_bg` |
| 미분 가능 SC (§14) | `models/sc_builder.py` | `endpoint_sc`: `0.5(Qsᵀ·diag(w)·Qe + Qeᵀ·diag(w)·Qs)` (matmul 2회), `pass_sc`, `ChunkedSC`(chunk 합산 후 loss 1회), `BundleAccumulator`(2-pass 정확 gradient) |
| loss (§12, §17-20) | `losses/{endpoint,sc_corr,sc_magnitude,tract_length,geometry,metrics}.py` | upstream 에 **loss 코드가 전혀 없다** → `L_ATM` = recon(정/역방향 min) + β·KL + 등간격 페널티로 정의 |
| 학습 (§11-21) | `training/trainer.py`, `training/run.py` | 2-pass 생성(subject-level SC) + 재구성 + edge, phase 별 활성 loss |
| 좌표계 | `spaces.py` | W 격자, mm↔voxel, `grid_sample` 축 순서 (import 시 자체 검사) |

## 4. 데이터 파이프라인 — upstream 과 완전히 다른 부분

| | upstream | 이 프로젝트 |
|---|---|---|
| GT streamline | bundle 별 `.trk` (예제 sub-1135) | whole-brain **DSI Studio `.tt.gz`** (gzip + MATLAB v4, 1/32-voxel int8 delta). `data/tt_io.py` 가 디코딩 (참조 구현과 대조, 버퍼 정확 소진) |
| bundle 정의 | 해부학 bundle 30개 (TractSeg 식) | **endpoint ROI pair** (`data/trk_to_roi_pairs.py`): 첫/끝점 → atlas ROI → canonical (a,b). 배경/같은 ROI 는 미할당 |
| 저장 | bundle 당 trk | subject 당 `assignments.npz` + `bundles.npz` (`docs/ROI_PAIR_DATA_FORMAT.md`) |
| 128점 재샘플 | (학습 전처리, 코드 미배포) | `data/resample_streamlines.py` (벡터화, 끝점 정확, 길이 오차 p99 0.28 %) |
| GT SC | 없음 (사후 평가) | `.mat` 의 `SC_weight/SC_length` (**pass** 정의) + 재계산 `sc_end` (endpoint 정의). 모드별로 target 을 분리 (`dataset.gt_for`) |
| subject 선택 | 예제 1명 | **`.mat`(GT SC) ∩ tracto 폴더(T1+tt.gz)** = 206명 (`paths.subjects`, `scripts/check_subjects.py`) |

## 5. 우회한 upstream 결함 (원본은 고치지 않음)

| # | upstream 위치 | 결함 | 우회 위치 |
|---|---|---|---|
| D1 | 배포본 전체 | 학습/loss 코드 없음 | `losses/`, `training/` 신규 |
| D2 | `infer.py:132` | `.eval()` 미호출 → Dropout3d 활성, 비결정적 | `ATMBundle.__init__` 에서 `net.eval()`; `train_mode_unet=True` 로 원본 재현 가능 |
| D3 | `infer.py:140` | `repeat(3000,1)` 하드코딩 | `decode_mm` 에서 `expand(n,-1)` |
| D4 | `infer.py:146` | 좌표 상수 경로 `data/` (파일은 `supp/`) | `BundleNorm.from_upstream` |
| D5 | `infer.py:144-158` | numpy 역정규화 → gradient 단절 | `decode_mm` (torch) |
| D6 | `infer.py:45` docstring | "affine" 이라 쓰고 `-t r`(rigid) 실행 | 문서화만 (`prepare_t1.py` 주석) |
| D7 | 예제 T1 강도 0-223 vs 상수 8330 | anatomy feature 입력이 [0, 0.026] | `run.anatomy_feature` 에 ‖a‖ assert, `anatomy_gain` |
| — | UNet 전체 forward | 23 GB 에서 OOM | 인코더 전용 경로 |
| — | `KernelDensity.sample` | `random_state` 속성 무시 | 메서드 인자로 전달 |

## 6. 의도적으로 바꾸지 않은 것

- `FiLM`, `ConvVAE.encode/decode`, `rigid_UNet` 인코더 연산: **가중치와 연산 모두 그대로**. 시작점이 pretrained 와 같아야 fine-tuning 이 의미를 갖는다.
- T1 정규화 상수 (`supp/*_vals_*`): 우리 데이터(GE, max≈5400)가 상수 범위 안에 들어와 [0, 0.65] 로 정규화되므로 유지. 단 D7 때문에 feature 크기는 계속 감시.
- 30 bundle 재현 경로 (`scripts/00_reproduce_atm.py`, `legacy/`): 원본 동작 확인용으로 보존.

## 7. 각 변경이 맞다는 근거 (실측)

| 변경 | 검증 |
|---|---|
| NLin6 로 정합 | atlas(NLin6) 로 GT `.tt.gz` → pass-SC 재계산 시 `.mat` 과 r=0.9986, edge F1 0.98 |
| 인코더 전용 UNet | CPU 전체 forward 대비 오차 **0.0** (`scripts/legacy/00_check_env.py`) |
| torch 역정규화 / 2-pass SC gradient | 단일 그래프 gradient 와 오차 **0.0** (`tests/test_sc_builder.py`) |
| soft endpoint/SC | soft pass-SC vs GT r=0.9939 (`scripts/legacy/03_validate_soft_sc.py`) |
| ROI-pair 주입이 pretrained 를 보존 | 0-init 에서 `cond == a` (`tests/test_roi_pair.py`) |
| 128점 재샘플 | 길이 오차 p99 0.28 %, 끝점 오차 0 (`scripts/04_resample_to_128.py`) |
| 전체 경로 | `scripts/06_smoke_test.py` synthetic + real FINAL PASS, `pytest` 35 passed |

---

## 8. 추가 변경 (최종 전략 `ATM_FINAL_FINETUNING_STRATEGY.md` 반영, 2026-09-02 후반)

| 항목 | 위치 | 내용 |
|---|---|---|
| **T1 encoder unfreeze** | `models/atm_adapter.py: set_unet_trainable, encode_anatomy_grad` | `rigid_UNet` 인코더를 stage1~4 로 나누고(`_stage1/2/3`, `_unet_stage4_fc` — model.py:193-224 와 연산 동일) level(`none/stage4/stage3/stage2/full`) 이상만 `requires_grad`. 학습 stage 는 `torch.utils.checkpoint` 로 감싸 193×229×193 활성을 저장하지 않음 (peak 16 GB). segmentation 가지(`upconv*`, `final_conv`)는 항상 동결 |
| **파라미터 그룹 LR** | `models/roi_atm.py: param_groups, param_counts`, `training/trainer.py: TrainConfig.lr_*` | `t1_encoder / vae_encoder / decoder / heads` 네 그룹, 기본 1e-5 / 3e-5 / 3e-5 / 1e-4 (§8). config 파일에서 변경 |
| **UNet backward 1회** | `training/trainer.py: step` | anatomy feature 를 leaf 로 분리 → 생성/재구성/edge pass 가 leaf 에 dL/da 누적 → 마지막에 `a.backward(dL/da)` 1회. T1 encoder 는 step 당 forward 1회·backward 1회 |
| **조건부 prior** | `models/roi_pair_embedding.py: prior_mu`, `losses/geometry.py: kl_loss(mu_prior)` | z ~ N(μ_pair, I). prior z 생성 pair 정확도 0 → 0.45 (파일럿) |
| **endpoint loss 수치** | `models/endpoint_assigner.py: point_log_probs`, `losses/endpoint.py(log_input)` | log-softmax 경로. 확률 clamp 로 gradient 가 0 이 되던 버그 수정 |
| **recon 을 mm 단위로** | `losses/geometry.py: stream_recon_loss` | mm² → RMSE. loss 간 gradient 스케일 정렬, clip 50 |
| **robust T1 정규화** | `models/atm_adapter.py: BundleNorm.normalize_t1(robust=True)` | subject 별 뇌 p99.5 → 0.6. PPMI 강도 스케일 230배 편차 대응 |
| **loss 별 gradient 기록** | `trainer.py` | `dLda_{G,R,E}`, `gnorm_{t1_encoder,vae_encoder,decoder,heads}` 를 log.jsonl 에 |
| config / phase | `training/config.py`, `configs/phase2_geometry … phase9_joint.yaml`, `scripts/16_train_config.py` | 최종 전략 §7 의 9 phase |
| feature QC | `scripts/14_feature_qc.py` | norm / cosine / Pearson / Euclid / 차원별 분산 / PCA / scanner 별 (§11) |
| split | `scripts/17_make_splits.py` → `outputs/splits/` | subject-level, group×scanner stratify (§12) |
| PBS | `scripts/pbs_train.sh` | `base_g`, ncpus 8, ngpus 1, walltime 24h |
