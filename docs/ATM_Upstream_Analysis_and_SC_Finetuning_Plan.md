# ATM Upstream 분석 및 SC-aware Fine-tuning 최소 수정 계획

작성일: 2026-09-02
근거: `external/atm_upstream/stable/` 실제 소스 + 실측 (torch forward, state_dict 로드)

---

## 0. 먼저: 저장소 상태 정정

지시받은 두 경로가 실제와 달랐다.

| 지시 | 실제 |
|---|---|
| `docs/ATM_SC_Aware_Finetuning_Framework.md` | `./ATM_SC_Aware_Finetuning_Framework.md` (repo root, `docs/` 없음) |
| `external/atm_upstream/` | 존재하지 않았음. 대신 `stable.zip` (11.7 GB) = Zenodo ATM stable release |

`stable.zip`을 확인한 결과 이것이 공식 ATM 배포본이었다 (`stable/model/model.py`, `stable/infer.py`,
`stable/models/<BUNDLE>/atmvae_<BUNDLE>.pth` × 30, `stable/kde_models/<BUNDLE>/kde_model.joblib` × 30).

분석을 위해 **코드/소형 파일만** `external/atm_upstream/`으로 압축 해제했다 (원본 수정 없음):

```
external/atm_upstream/stable/
├── infer.py                 (15,341 B)
├── model/model.py           (15,351 B)
├── model/__pycache__/*.pyc  (3개)
├── requirements.txt
├── matlab_post/*.m          (11개)
├── supp/*.npy               (120개)
└── models/AF_L/atmvae_AF_L.pth   ← 검증용 1개만
```

11 GB에 달하는 나머지 29개 `.pth`, 30개 `kde_model.joblib`, `data/sub-1135/`는 아직 압축 해제하지 않았다
(동시에 21 GB `PPMI_QC263_tracto.zip` 다운로드가 진행 중이라 I/O를 아꼈다).

---

## 1. Upstream 전체 구조 — 핵심 결론

> **공식 ATM 배포본에는 inference 코드만 있다. 학습 코드, loss 함수, dataset, optimizer가 전혀 없다.**

`.py` 파일은 전체 archive(546개 파일)에서 딱 2개다:

- `stable/infer.py` — 전처리 → 생성 → 필터 → 트리밍 → native 변환
- `stable/model/model.py` — 네트워크 정의만

`model/__pycache__/`의 세 `.pyc`를 strings로 조사한 결과 `sin_fr_layer`, `init_bases`, `phi_num`,
`high_freq_num`, `share_weights_layers`, `fc_mu`/`fc_rho`, `forward_with_activations` 등
**배포된 `model.py`에 없는 개발 중 클래스**의 흔적이 남아있으나, **loss 함수 이름은 어느 pyc에도 없다.**

→ 논문의 `L_recon + KL + adjacent-point loss`는 **우리가 직접 작성해야 한다.**
   이것이 계획 전체를 좌우하는 가장 중요한 사실이다.

---

## 2. 위치 지도 (요청 3번)

| 구성요소 | 파일:라인 | 클래스/함수 |
|---|---|---|
| **T1 encoder (anatomy encoder)** | `model/model.py:144-250` | `rigid_UNet` — 3D UNet, `latent_dim=512` |
| ↳ anatomy feature 추출 지점 | `model/model.py:222-224` | `global_avg_pool` → `view` → `fc` |
| ↳ segmentation head (부산물) | `model/model.py:248` | `final_conv` + sigmoid |
| **Streamline encoder** | `model/model.py:323-345` | `ConvVAE.encode` → `(mu, logvar)` |
| **Streamline decoder** | `model/model.py:347-367` | `ConvVAE.decode(z, anatomical_info)` |
| **조건부 결합 (FiLM)** | `model/model.py:9-39` | `FiLM` — encoder 3개, decoder 2개 |
| **VAE reparameterize** | `model/model.py:375-378` | `ConvVAE.reparameterize` |
| **최상위 모델** | `model/model.py:254-274` | `ATMVAE(anatomical_dim=512, latent_dim=64, rigid=True)` |
| **Inference 진입점** | `infer.py:274-353` | `main(args)` |
| **핵심 생성 함수** | `infer.py:115-160` | `gen_streamlines(samples, t1w, bundle)` |
| **latent 샘플링 (KDE)** | `infer.py:304-306` | `joblib.load(...)` → `kde.sample(N)` |
| **T1 전처리/정규화** | `infer.py:86-112` | `get_t1w_input` |
| **MNI 정합 (ANTs, rigid)** | `infer.py:21-83` | `affine_registration` — 실제 플래그는 `-t r` (rigid) |
| **TRK/TCK export** | `infer.py:315-325` | `StatefulTractogram` + `save_tractogram` |
| **해부학 필터링 (MRtrix)** | `infer.py:163-212` | `filtering` — `tckedit` |
| **끝점 트리밍 (MATLAB)** | `infer.py:215-229`, `matlab_post/bundle_trimming.m` | `trimming` |
| **native space 복원** | `infer.py:232-271` | `to_native` — scilpy, `--reverse_operation` |
| **좌표 정규화 상수** | `supp/{bundle}_{min,max}_coords_rigid_transformed_streamlines.npy` | shape `(3,)` float32 |
| **T1 정규화 상수** | `supp/{bundle}_{min,max}_vals_rigid_transformed_T1w.npy` | scalar float64 |

**파라미터 실측** (`atmvae_AF_L.pth`, `strict=True` 로드 성공, missing/unexpected 모두 `[]`):

```
total          51.38 M   (= 205.5 MB fp32, optimizer state 없음)
├─ unet        49.86 M   (97.0 %)
└─ ConvVAE      1.515 M
   └─ decoder만  0.627 M  (627,075 params)   ← fine-tuning 대상
```

---

## 3. 64-D latent → 128×3 streamline 추적 (요청 4번)

### 호출 흐름 (`infer.py:296` → `infer.py:160`)

```
main()
 ├─ get_t1w_input(sub, bundle)                              infer.py:301
 │    T1w(MNI 1mm) → (raw - min_val)/(max_val - min_val)     infer.py:110
 │    → ndarray (193, 229, 193)
 │
 ├─ kde = joblib.load("kde_models/{bundle}/kde_model.joblib") infer.py:305
 │  samples = kde.sample(N)                                   infer.py:306
 │    → ndarray [N, 64]      ※ N(0,I) prior가 아니라 학습 latent의 KDE 경험분포
 │
 └─ gen_streamlines(samples, t1w, bundle)                     infer.py:309
      ├─ atm = ATMVAE(512, 64, True); load_state_dict; .cuda()   infer.py:128-132
      ├─ t1w_input = tensor.reshape(1,1,193,229,193)             infer.py:138
      ├─ anatomical_condition, _ = atm.unet(t1w_input)           infer.py:139
      │    rigid_UNet.forward:
      │      conv1_x → output1  [1, 64, 193,229,193]
      │      conv2_x → output2  [1,128,  97,115, 97]
      │      conv3_x → output3  [1,256,  49, 58, 49]
      │      conv4_x → output4  [1,512,  25, 29, 25]
      │      global_avg_pool(output4)  → [1, 512, 1,1,1]         model.py:222
      │      .view(1, 512)                                       model.py:223
      │      fc: Linear(512→512)                                 model.py:224
      │    ⇒ anatomical_condition  a = [1, 512]
      │
      ├─ a_rep = a.repeat(3000, 1)   → [3000, 512]               infer.py:140
      │
      ├─ z = atm.ae.decode(samples, a_rep)                       infer.py:142
      │    ConvVAE.decode(z=[N,64], a=[N,512]):     model.py:347-367
      │      decoder_fc : Linear(64 → 2048)         → [N, 2048]
      │      view(N, 128, 16)                       → [N, C=128, L=16]
      │      up1  Upsample(×2, nearest)             → [N, 128,  32]
      │      deconv1 Conv1d(128→64, k=31, p=15)     → [N,  64,  32]
      │      dbn1 BatchNorm1d(64)
      │      dfilm1 FiLM(512→64):  γ(a)[N,64,1]*x + β(a)[N,64,1]
      │      up2  Upsample(×2)                      → [N,  64,  64]
      │      deconv2 Conv1d(64→32, k=63, p=31)      → [N,  32,  64]
      │      dbn2 BatchNorm1d(32)
      │      dfilm2 FiLM(512→32)
      │      up3  Upsample(×2)                      → [N,  32, 128]
      │      deconv3 Conv1d(32→3, k=127, p=63)      → [N,   3, 128]
      │      tanh                                   → 값 범위 [-1, 1]
      │
      ├─ infered_streamline = z.permute(0,2,1).cpu().numpy()     infer.py:144
      │    ⇒ [N, 128, 3]
      │
      └─ 역정규화 (numpy, list comprehension)                    infer.py:152-158
           s_mm = (s - (-1)) / (1 - (-1)) * (max_coords - min_coords) + min_coords
                = (s + 1)/2 * (max - min) + min
           min/max: shape (3,) — 축별 상수, bundle별 다름
           예 AF_L: min = [-73.418, -84.012, -43.798]
                    max = [ -3.692,  67.042,  66.949]
           ⇒ MNI(ICBM152 2009c asym) 1mm RASMM 좌표
```

### 실측 검증

```
decode(z[7,64], a[7,512]) -> (7, 3, 128)   range [-0.721, 0.688]
permute(0,2,1)            -> (7, 128, 3)
encode(x[7,3,128])        -> mu (7,64), logvar (7,64)
```

**요약:** 64-D → `Linear(64,2048)` → `(128ch, 16)` → 3×(upsample×2 + Conv1d) → `(3, 128)` → tanh →
permute → `[N,128,3]` → affine 역정규화 → mm. **recurrent propagation 없음. 순수 feed-forward.**

---

## 4. T1 encoder 호출 횟수 (요청 3/5번)

**streamline마다 호출되지 않는다.** `atm.unet(t1w_input)`은 `gen_streamlines` 안에서 **1회**만 실행되고,
결과 `[1,512]`를 `repeat(N,1)`로 복제해 FiLM 조건으로 넘긴다 (`infer.py:139-142`).

다만 **subject당 1회가 아니라 (subject × bundle)당 1회 = 30회**다.
`main(args)`가 bundle마다 호출되고 (`infer.py:409-411`), bundle마다 다른 `.pth`를 로드하며
(`infer.py:128-132`) 매번 UNet forward를 다시 돌린다. 게다가 `get_t1w_input`의 정규화 상수도
bundle마다 다르므로 (`supp/{bundle}_*_vals_*.npy`) 입력 T1 자체가 bundle마다 다르다.

→ **anatomy feature 캐시는 bundle별로 30개 벡터**로 잡아야 한다. 하나로 공유할 수 없다.
→ inference 최적화 여지: 30 × (205 MB 로드 + 193×229×193 UNet forward)가 반복된다.

---

## 5. Upstream에서 확인된 결함 — Stage 0 재현 전에 반드시 처리

| # | 위치 | 내용 | 영향 |
|---|---|---|---|
| **D1** | 배포본 전체 | **학습 코드·loss 함수 없음** | `L_stream`, `L_adj`를 직접 정의해야 함. 논문 수식과 일치 보장 불가 |
| **D2** | `infer.py:132-136` | `atm.eval()` **미호출**. `nn.Module` 기본이 train mode | UNet `Dropout3d(p=0.2)` 4개가 batch=1에서 활성 → **anatomy feature가 매 실행마다 달라짐**. BatchNorm1d도 batch 통계 사용. inference가 비결정적 |
| **D3** | `infer.py:140` | `anatomical_condition.repeat(3000, 1)` — **3000 하드코딩** | `--num_streamline 6000` 지정 시 FiLM에서 `RuntimeError: size of tensor a (3000) must match b (6000)`. 논문의 6000/9000 실험 재현 불가 |
| **D4** | `infer.py:146-147` | `data/{bundle}_..._streamlines.npy` 를 읽지만 파일은 `supp/`에만 존재 (archive 내 `data/`에 해당 npy 0개) | 실행 즉시 `FileNotFoundError`. `supp/*.npy` → `data/` 복사/심볼릭 필요 |
| **D5** | `infer.py:144,152-158` | `.cpu().numpy()` 후 역정규화 | **gradient 단절**. fine-tuning에서는 torch 연산으로 재작성 필수 |
| **D6** | `infer.py:39,45` | ref = `mni_icbm152_t1_tal_nlin_asym_09c_1mm`, `antsRegistrationSyN.sh -t r` = **rigid** (docstring은 "affine"이라 잘못 기술) | 우리 atlas `DesikanCortexPD25_space-**MNI152NLin6**_res-2x2x2`와 **템플릿·해상도가 모두 다름**. 부위에 따라 수 mm 차이 → endpoint→ROI 할당이 직접 망가짐 |
| **D7** | `matlab_post/bundle_trimming.m:6-13` | 트리밍이 `data/{sub}/mri/brainmask.nii` + FreeSurfer `surf/*.white.surf.gii` 사용. 그런데 `to_native`는 트리밍 **이후**에 실행 | 트리밍이 MNI 공간에서 수행됨을 전제 → FreeSurfer가 MNI-warped T1에서 돌아갔다는 뜻. 우리 데이터에 적용하려면 이 전제를 반드시 실측 확인 |
| **D8** | `infer.py:274-353` | 필터링(`tckedit`) + 트리밍(MATLAB)이 외부 CLI·비미분 | 학습 그래프에 넣을 수 없음. 128점 등간격도 깨짐 (트리밍이 끝점 제거) |
| **D9** | `requirements.txt` | `torch==2.5.1` 고정, 현재 환경 `torch 2.6.0+cu124` | `torch.load` 기본값이 `weights_only=True`로 바뀜. 순수 state_dict라 로드는 성공 확인함 |

D2가 특히 중요하다. **"공식 예제 output 재현" 통과 조건을 만족시키려면 `.eval()` 유무를 명시적으로 결정하고
두 조건 모두 기록해야 한다.** (논문 결과는 train mode 상태로 생성되었을 가능성이 높다.)

---

## 6. SC-aware fine-tuning의 진짜 병목: bundle-per-model 구조

SC 행렬은 **30개 bundle의 streamline 전체**에 대한 함수인데, ATM은 bundle마다 독립적인 51.4 M 모델이다.

```
30 bundles × 51.38 M params  = 1.54 B params  (6.2 GB fp32 weights)
30 × UNet forward on 193×229×193
30 × 3000 streamlines × 128 × 3 = 34.6 M coords
```

30개를 동시에 학습 그래프에 올리는 것은 불가능하다. 두 가지로 푼다.

### (a) UNet 동결 + anatomy feature 캐시

`unet`(49.86 M, 97 %)을 freeze하고 **ConvVAE decoder 627 K params만** 학습한다.
그러면 `a_b = unet_b(T1_b)` `[1,512]`를 subject당 30개 벡터(30×512 float = 60 KB)로 **미리 계산해
캐시**할 수 있고, 3D UNet이 학습 그래프에서 완전히 사라진다.

Framework 문서 §20-6("T1 encoder output은 subject별 재사용")과 정확히 일치하며,
파라미터가 627 K로 줄어 과적합 위험과 VRAM이 동시에 해결된다.

### (b) Bundle 간 정확한 gradient 누적 (2-pass)

`SC_pred = Σ_b SC_b` 는 bundle에 대해 **가산적**이지만 `L`(상관계수, log)은 비선형이라
bundle을 따로 backward할 수 없다. 다음이 수학적으로 정확하고 메모리가 bundle 수에 무관하다:

```python
# pass 1 : 전체 SC를 gradient 없이 누적
with torch.no_grad():
    SC = sum(sc_builder(decode(b)) for b in bundles)      # [R, R]

# 스칼라 loss의 SC에 대한 gradient G 만 얻는다
t = SC.detach().requires_grad_(True)
L = sc_loss(t, SC_gt)
L.backward()
G = t.grad                                                # ∂L/∂SC_total, [R, R]

# pass 2 : bundle 하나씩만 그래프에 올려 backward
for b in bundles:
    SC_b = sc_builder(decode(b))                          # grad 활성
    SC_b.backward(gradient=G)                             # ∂L/∂θ_b 정확히 누적
    del SC_b
optimizer.step()
```

`SC_total = Σ_b SC_b` 이므로 `∂L/∂SC_b = ∂L/∂SC_total = G`. 근사 없음. forward 2회 비용만 추가.

**주의:** tract length는 `Length(i,j) = Num(i,j) / Den(i,j)` 형태라 비율이다.
`Num = Σ_b Num_b`, `Den = Σ_b Den_b` 둘 다 가산적이므로,
누적 대상을 `(Num, Den)` 쌍으로 두고 `G_num`, `G_den` 두 개를 받아 동일한 트릭을 적용한다.

---

## 7. `L_stream`을 어떻게 확보할 것인가

D1 때문에 원본 reconstruction loss가 없고, 게다가 `L_stream`을 쓰려면
**bundle 라벨이 붙은 GT streamline**이 필요하다 (AF_L에 속하는 streamline만 AF_L 모델에 먹여야 함).
현재 PPMI 데이터에 30-bundle segmentation이 있는지 불확실하다.

두 경로:

- **B-1 (GT bundle 있음):** TractSeg/RecoBundles로 30 bundle 분리 → 128점 등간격 resample →
  bundle별 `supp` 상수로 [-1,1] 정규화 → `encode`→`reparameterize`→`decode` → MSE + KL.
- **B-2 (GT bundle 없음, 권장 기본값):** **anchor(distillation) loss**로 대체.
  동일한 `(z, a)`에 대해 **동결된 원본 decoder** 출력 `S_orig`를 기준으로

  ```
  L_anchor = mean( || S_pred(z,a) - S_orig(z,a) ||^2 )
  ```

  GT bundle 라벨이 전혀 필요 없고, "SC만 좋아지고 geometry가 무너지는 것"(§20-5)을 직접 막는다.
  Stage 4~6 내내 이것을 geometry 보호 항으로 쓰고, B-1이 가능해지면 교체한다.

`L_adj`도 배포본에 없으므로 우리가 정의한다. 128점이 등간격 resample이라는 전제를 이용해
**등간격 위반 페널티**로 잡는다:

```
d_t = || p_{t+1} - p_t ||         (t = 1..127)
L_adj = mean( (d_t - mean_t(d_t))^2 )        # 또는 2차 차분 ||p_{t+1} - 2p_t + p_{t-1}||^2
```

정의를 우리가 만든 것이므로 논문 값과 다르다는 점을 Methods에 명시해야 한다.

---

## 8. 최소 수정 파일 목록 (요청 6번)

### 원칙
`external/atm_upstream/`은 **한 줄도 수정하지 않는다.** D2/D3/D4/D5는 upstream을 고치지 않고
adapter 안에서 우회한다 (`.eval()` 명시 호출, `repeat(N,1)`, `supp/` 경로 사용, torch 역정규화).

### 신규 파일 — 6개

| 파일 | 역할 | 예상 규모 |
|---|---|---|
| `src/atm_sc/models/atm_adapter.py` | ① `ATMVAE` 로드 + `strict=True` 검증 ② UNet freeze ③ `(subject, bundle) → a[1,512]` 캐시 (D2: eval mode 명시) ④ **미분 가능** `decode_mm(z, a) -> [N,128,3] mm` (D5 대체: `(s+1)/2*(max-min)+min` 을 torch로) ⑤ `repeat(N,1)` (D3 대체) ⑥ `supp/` 경로 사용 (D4 대체) | ~150 줄 |
| `src/atm_sc/models/sc_builder.py` | 미분 가능 soft SC. atlas one-hot/거리맵 `[R,X,Y,Z]` → `grid_sample`로 endpoint 확률 → `SC = Pᵀ_start · P_end` 대칭화, 동시에 `Num/Den` 반환 (length용). mm→voxel→normalized 좌표 변환 포함 | ~180 줄 |
| `src/atm_sc/losses/sc_losses.py` | `L_SC_corr`, `L_SC_mag`, `L_length`, `L_adj`, `L_anchor` 를 한 파일에 (문서 §14의 3파일 분할은 아직 불필요) | ~120 줄 |
| `src/atm_sc/data/dataset.py` | subject 목록, MNI-rigid T1 (bundle별 정규화), atlas prob volume, `SC_weight_gt`/`SC_length_gt`, rigid affine (MNI↔native) | ~150 줄 |
| `src/atm_sc/training/finetune_sc.py` | §6(b) 2-pass bundle gradient 누적 학습 루프. stage 플래그로 loss 항 on/off | ~200 줄 |
| `src/atm_sc/inference/generate_tractogram.py` | 학습된 decoder로 30 bundle 생성 → 합쳐서 `.trk` export (`infer.py:315-325` 로직 재사용, 외부 CLI 없이) | ~100 줄 |

### 신규 스크립트 — 4개

| 스크립트 | 목적 |
|---|---|
| `scripts/00_check_upstream.py` | D1~D9 회귀 확인 + 30개 `.pth` 전부 `strict=True` 로드 검증 |
| `scripts/01_reproduce_atm.py` | Stage 0. `.eval()` on/off 양쪽 실행, runtime/VRAM 측정 |
| `scripts/02_build_gt_sc.py` | GT tractogram → MRtrix SC (weight/length) 저장 |
| `scripts/03_validate_soft_sc.py` | Stage 3. soft SC vs MRtrix SC (r, CCC, MAE, F1) — **여기서 실패하면 중단** |

### 수정하지 않는 것
`external/atm_upstream/**` 전부. 필요 시 `patches/` 에 diff만 기록.

---

## 9. 실행 순서 (Framework §8 대응, 실제 코드 기준으로 조정)

| Stage | 내용 | 통과 조건 |
|---|---|---|
| **0a** | `stable.zip`에서 나머지 29 `.pth` + 30 KDE + `data/sub-1135/` + MNI 템플릿 압축 해제 (약 11 GB) | 30개 모델 `strict=True` 로드 |
| **0b** | D4 우회(`supp/`→`data/` 링크) 후 `sub-1135`, `AF_L` inference 재현 | NaN 0, streamline 길이 분포 정상 |
| **0c** | D2 검증: `.eval()` on/off로 각 3회 생성, anatomy feature 분산 측정 | train mode 비결정성 정량화 |
| **1** | 좌표계 확정 (D6/D7). ATM MNI 1mm(NLin2009cAsym) ↔ 우리 atlas(NLin6 2mm) ↔ PPMI native | **좌우 반전 검증 필수.** GT tractogram을 ATM 그리드에 올려 육안+수치 확인 |
| **2** | anatomy feature 캐시 생성 (subject × 30 bundle × 512) | 캐시 재로드 시 bit-identical |
| **3** | `sc_builder` 단독 검증 (GT trk → soft SC vs MRtrix SC) | r, CCC 목표치 미달 시 **중단** |
| **4** | `L_anchor + L_adj` 만으로 decoder fine-tune (baseline) | geometry 유지 확인 |
| **5** | `+ λ_corr · L_SC_corr` (2-pass 누적) | SC r 개선, geometry 비열화 |
| **6** | `+ λ_mag · L_SC_mag` → `+ λ_len · L_length` | CCC/MAE/length 개선 |
| **7** | ablation + CoRNN 대비 benchmark (동일 streamline 수) | — |

Stage 4 이전에 **각 loss의 gradient norm을 기록**해서 §7 초기 가중치
(`λ_stream=1.0, λ_adj=1.0, λ_corr=0.1, λ_mag=0.1, λ_len=0.05`)를 재조정한다.

---

## 10. 반드시 넣을 assert (파이프라인 조용한 실패 방지)

```python
# 모델 로드
missing, unexpected = m.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)

# anatomy feature
assert a.shape == (1, 512) and torch.isfinite(a).all()
assert a.abs().max() > 0, "anatomy feature 전부 0 — UNet 입력 정규화 확인"

# 정규화 상수
assert min_coords.shape == (3,) and (max_coords > min_coords).all()
assert max_val > min_val, f"T1 정규화 상수 이상: {min_val}, {max_val}"

# decoder 출력
assert S.shape == (N, 128, 3)
assert torch.isfinite(S).all()
assert S.std(dim=1).min() > 0, "일직선/상수 streamline 생성됨"

# mm 좌표 범위 (MNI 1mm 193x229x193 밖이면 좌표계 오류)
assert (S_mm.amin(dim=(0,1)) > -110).all() and (S_mm.amax(dim=(0,1)) < 110).all()

# atlas
assert atlas_prob.shape[0] == R, f"ROI 수 불일치: {atlas_prob.shape[0]} vs {R}"
assert (atlas_prob.sum(0) > 0).any(), "atlas 전부 배경"
assert len(np.unique(atlas_labels)) - 1 == R, "해상도 변환 중 ROI 소실"

# SC
assert torch.isfinite(SC).all()
assert SC.sum() > 0, "SC 전부 0 — endpoint가 atlas 밖"
assert torch.allclose(SC, SC.T, atol=1e-5)
assert (SC.diagonal() >= 0).all()

# 2-pass gradient 정합성 (소규모에서 1회만)
#   전체를 한 그래프에 올린 grad와 2-pass 누적 grad가 일치하는지 확인
```

---

## 11. 아직 확인하지 못한 것 / 다음에 결정할 것

1. **PPMI 데이터에 30-bundle segmentation이 있는가** → `L_stream`(B-1) vs `L_anchor`(B-2) 선택을 좌우
2. **GT SC의 정의** (streamline count / SIFT2 / normalized) — 기존 TVB 파이프라인과 일치시켜야 함
3. **atlas 템플릿 불일치 해결 방향** (D6): `DesikanCortexPD25`를 NLin2009cAsym 1mm로 warp할지,
   ATM streamline을 native로 되돌려 native atlas를 쓸지. rigid 변환은 4×4 affine이라
   **torch 안에서 미분 가능하게 합성 가능** → native 공간에서 SC를 만드는 쪽이 더 깔끔할 수 있다
4. **`data/sub-1135/surf/`의 FreeSurfer가 native인지 MNI인지** (D7)
5. **30 bundle이 DK-82 SC의 몇 % edge를 설명하는가** (Framework §23-4) — Stage 3 직후 측정

---

## 부록: 30개 bundle

`AF_L/R, CC_Fr_1, CC_Fr_2, CC_Oc, CC_Pa, CC_Pr_Po, CG_L/R, FAT_L/R, FPT_L/R, IFOF_L/R,
ILF_L/R, MCP, MdLF_L/R, OR_ML_L/R, POPT_L/R, PYT_L/R, SLF_L/R, UF_L/R`
(`infer.py:375-406`)

`data/sub-1135/tractography/`에는 `FX_L/R`도 있으나 **모델은 없다** (30개 목록에 미포함).
