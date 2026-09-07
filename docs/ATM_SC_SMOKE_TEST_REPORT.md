# ATM → SC-aware Fine-tuning : 구현 및 Smoke Test 보고서

날짜: 2026-09-02 · 기준 문서: `docs/ATM_SC_Aware_Finetuning_Framework_v2.md`
결과: **FINAL: PASS** (synthetic + 실제 pretrained ATM integration)

---

## 1. ATM 원본 구조 (실제 source 기준)

`external/atm_upstream/stable` = `stable/stable` (Zenodo 15792527). `.py` 는 **두 개뿐**이다.

| 항목 | 위치 | 내용 |
|---|---|---|
| T1 anatomy encoder | `model/model.py:144` `rigid_UNet` | 3D UNet. `Upsample` size 가 `(49,58,49)/(97,115,97)/(193,229,193)` 로 **하드코딩** → 입력 격자 고정 |
| anatomy feature 추출 | `model/model.py:222-224` | `global_avg_pool` → `view` → `fc(512→512)` ⇒ `[1,512]` |
| streamline encoder | `model/model.py:323` `ConvVAE.encode` | Conv1d(k=127/63/31)+BN+AvgPool+FiLM → `(mu, logvar)` 각 `[N,64]` |
| streamline decoder | `model/model.py:347` `ConvVAE.decode` | `Linear(64→2048)` → `view(N,128,16)` → 3×(Upsample×2 + Conv1d) → `tanh` → `[N,3,128]` |
| conditioning | `model/model.py:9` `FiLM` | encoder 3개, decoder 2개. anatomy feature 를 γ/β 로 주입 |
| latent | `latent_dim=64` (`model.py:255`) | inference 시 **KDE 에서 샘플** (`infer.py:305`). `N(0,I)` prior 아님 |
| original loss | **없음** | 배포본에 학습 코드·loss 함수가 전혀 없다. `model/__pycache__/*.pyc` 3개에도 loss 이름 없음 |
| checkpoint load | `infer.py:131` | bundle 당 `atmvae_{bundle}.pth`. `strict=True` 로드 확인 (missing/unexpected 모두 `[]`) |
| inference entry | `infer.py:274` `main` → `infer.py:115` `gen_streamlines` | |
| bundle-specific | `infer.py:375-406` 30개 목록 | 모델·KDE·정규화 상수(`supp/*.npy`)가 **전부 bundle 별로 분리** |
| 출력 tensor shape | `[N, 3, 128]` → `permute(0,2,1)` → `[N, 128, 3]` | 실측 확인 |

**파라미터**: 총 51.38 M = UNet 49.86 M (97.0 %) + ConvVAE 1.515 M. **decoder 만 627,075**.

**T1 encoder 호출 횟수**: streamline 마다가 아니라 **(subject × bundle) 당 1회**.
`gen_streamlines` 안에서 `atm.unet(t1w_input)` 1회 → 결과 `[1,512]` 를 `repeat(N,1)` 로 복제
(`infer.py:139-142`). 다만 `main(args)` 가 bundle 마다 호출되므로 subject 당 총 30회다.
bundle 마다 T1 정규화 상수가 다르므로 anatomy feature 캐시는 **subject × 30 bundle** 로 잡아야 한다.

**batch decode 가능 구조인가**: 가능하다. decoder 는 `z [N,64]` 를 그대로 받는 순수
feed-forward 라 recurrent 의존이 없다. 단, upstream 은 `repeat(3000, 1)` 로 3000 을
하드코딩해 두어 `N != 3000` 이면 FiLM 에서 shape 오류가 난다 (아래 D3).

### upstream 결함 (adapter 에서 우회, 원본은 무수정)

| | 위치 | 내용 |
|---|---|---|
| D1 | 배포본 전체 | 학습 코드·loss 없음 → `L_ATM` 을 우리가 정의해야 함 |
| D2 | `infer.py:132` | `.eval()` 미호출. UNet `Dropout3d` 4개가 살아있어 **anatomy feature 가 실행마다 달라짐** |
| D3 | `infer.py:140` | `repeat(3000, 1)` 하드코딩 → 논문의 6000/9000 실험 재현 불가 |
| D4 | `infer.py:146` | 좌표 상수를 `data/` 에서 찾지만 파일은 `supp/` 에만 존재 → 즉시 `FileNotFoundError` |
| D5 | `infer.py:152` | 역정규화가 numpy → **gradient 단절** |
| D6 | `infer.py:45` | `-t r` = rigid (docstring 은 "affine" 이라 잘못 기술) |

---

## 2. 추가/수정한 파일

`external/atm_upstream/` 은 **한 줄도 수정하지 않았다.**

```
src/atm_sc/
  spaces.py               좌표계 정의 + grid_sample 축 순서 (import 시 자체 테스트)
  atm_adapter.py          ATMVAE 래퍼. D2~D5 우회, UNet 동결, chunk decode, AMP, inference_mode
  endpoint_assigner.py    ROI 거리맵 + grid_sample → soft ROI. endpoint_probs / visit_probs
  sc_builder.py           SCBuilder(endpoint|pass), ChunkedSC, BundleAccumulator
  losses/endpoint.py      L_endpoint (방향 대칭), endpoint_accuracy, L_roi_visit
  losses/sc_corr.py       L_SC_corr (분산 0 방지)
  losses/sc_magnitude.py  L_SC_mag (총합 정규화)
  losses/tract_length.py  L_length (GT edge mask)
  losses/geometry.py      L_ATM: adjacency / anchor / stream_recon / KL
  losses/metrics.py       sc_metrics (r, r_log, CCC, MAE, RMSE, edge F1, density)
  data/paths.py           subject 경로 해석
  data/tt_io.py           .tt.gz 디코더, 128점 재샘플, hard pass/end SC
  data/prepare_t1.py      native T1 → W 격자 (syn|rigid) + 정합 검증
scripts/
  00_check_env.py  01_verify_gt_sc.py  02_prepare_t1.py
  03_validate_soft_sc.py  04_integration_smoke.py  smoke_test_atm_sc.py
tests/
  conftest.py  test_endpoint.py  test_sc_builder.py  test_losses.py
```

---

## 3. Tensor flow

```
T1w native (192,256,256 등)
  │  ANTs SyN → MNI152NLin6Asym → W 격자로 재샘플            data/prepare_t1.py
  ▼
T1_W  [1, 1, 193, 229, 193]        ← rigid_UNet 하드코딩 때문에 격자 고정
  │  atm.encode_anatomy()  ── subject × bundle 당 **1회만**   atm_adapter.py
  ▼
anatomy feature  a [1, 512]        (캐시 대상)
  │
KDE (tophat, bw=1, 447,000×64) ──▶ z [N, 64]
  │
  │  atm.decode_chunks(z, a, chunk)  ── a 를 재사용, encoder 재호출 없음
  ▼
streamlines  [N, 128, 3]  (mm, MNI)      ← 역정규화까지 torch (D5 우회)
  │
  │  EndpointAssigner: 거리맵 grid_sample → softmax(-d/τ)
  ▼
q_start, q_end [N, 82]      또는     u [N, 82] (pass)
  │
  │  SCBuilder → matmul 2회.  chunk 별 부분 SC 를 ChunkedSC 로 **합산**
  ▼
SC_pred [82, 82], Num [82, 82]     ← 여기서 처음으로 loss 를 계산 (subject-level)
  │
  ▼
L = L_ATM + λ_end·L_endpoint + λ_corr·L_SC_corr + λ_mag·L_SC_mag + λ_len·L_length
  │  backward
  ▼
decoder 627,075 params            (UNet 49.86M 은 동결)
```

recurrent tracking 없음. T1 encoder 반복 호출 없음.

---

## 4. Smoke test 결과

### 4.1 pytest

```
23 passed in 2.64s
```

주요 항목: 거리맵 유효성 / ROI 소실 감지 / 확률 합 1 / 배경 클래스 효과 /
endpoint argmax 정확도 / visit_probs 3가지 집계 / SC shape·대칭·대각 0 /
chunk 합 == 전체 / 2-pass bundle gradient 정확성 / 상관 loss 스케일 불변 /
magnitude 가 상관이 놓치는 스케일을 잡는지 / 분산 0 거부 / endpoint 방향 대칭 /
geometry loss / 전체 objective backward.

### 4.2 `scripts/smoke_test_atm_sc.py --mode endpoint`

```
device cuda | GPU NVIDIA A10 | VRAM 23.0 GB | cudnn 90100 | torch 2.6.0+cu124

[A] synthetic
  streamline shape       (12, 128, 3)
  endpoint prob shape    (12, 6)
  SC shape               (6, 6)
  SC symmetry error      0.000e+00
  endpoint loss 0.001474 | SC corr 0.246 | SC mag 0.596 | length 0.172 | L_ATM 0.085
  total loss             0.106809
  streamline gradient    PASS  (max |grad| = 2.598e-03)

[B] ATM integration (AF_L pretrained, sub-100001 실제 T1/GT SC)
  anatomy feature        (1, 512)   0.50s, 1회 호출, 두 번 호출 차이 0.0
  streamline shape       (3000, 128, 3)  chunk=1500
  SC shape               (82, 82)   symmetry error 0.000e+00
  SC corr 0.995 | SC mag 4.514 | length 4.657 | total 0.369
  model gradient         decoder 20/20, UNet 0

  peak VRAM              9.00 GB
  elapsed                12.8 s
FINAL: PASS
```

`--mode pass`, `--amp fp16 --n-streamlines 12000 --chunk 6000` 도 모두 PASS.

### 4.3 실제 데이터로 확인한 수치 (smoke test 범위 밖, 참고)

| 항목 | 결과 |
|---|---|
| `.tt.gz` → hard **pass** SC vs `.mat` GT (sub-100001, 1e6 streamline) | **r=0.9986**, r_log=0.9912, edge F1=0.9815 |
| 같은 데이터, hard **end** SC vs GT | **r=0.6728** |
| 128점 재샘플 후 hard pass | r=0.9985 (ATM 의 128점 표현은 SC 병목이 아님) |
| soft pass SC (τ=0.5, max, d_bg=2.0) vs GT | **r=0.9939, ccc=0.9854**, length r=0.8991 / MAE 9.8 mm |
| 인코더 전용 UNet vs upstream 전체 forward | 차이 **0.0 (bit-exact)** |
| 2-pass bundle gradient vs 단일 그래프 | 차이 **0.0** |

---

## 5. 실행 명령

```bash
python scripts/00_check_env.py                    # 환경 + upstream 회귀
python -m pytest tests/ -q
python scripts/smoke_test_atm_sc.py               # endpoint 모드
python scripts/smoke_test_atm_sc.py --mode pass
python scripts/smoke_test_atm_sc.py --amp fp16 --n-streamlines 12000 --chunk 6000

python scripts/01_verify_gt_sc.py  --sub sub-100001    # GT SC 재현 검증
python scripts/02_prepare_t1.py    --sub sub-100001 --mode syn
python scripts/03_validate_soft_sc.py --sub sub-100001 # τ / 집계 / 배경 스윕
```

---

## 6. GPU / VRAM / runtime

- GPU: **NVIDIA A10 23 GB 1장**. torch 2.6.0+cu124, cuDNN 9.1.0.
- smoke test peak VRAM **9.00 GB**, 전체 12.8 s (거리맵 캐시 이후 3.3 s).
- 단계별: T1 encoder 0.50 s/subject·bundle · 3000 streamline decode+SC < 0.5 s ·
  `.tt.gz` 1e6 streamline 디코딩 약 65 s/subject · ANTs SyN 정합 약 6 분/subject.
- **AMP 실측** (AF_L, 3000 streamline, fp32 대비 좌표 오차):
  `fp16` mean 0.024 mm / p99 0.62 mm / max 6.35 mm — 사용 가능
  `bf16` mean 0.178 mm / p99 4.64 mm / max **57.2 mm** — 좌표에 쓰기 부적합
  decoder 가 큰 커널 Conv1d(k=127)+BatchNorm 이라 bf16 의 8비트 가수로는 부족하다.
  **기본값은 AMP off**. A100/H100 이라도 bf16 대신 fp16 을 쓸 것.

### 환경 문제 두 건을 고쳤다 (기존부터 있던 문제)

1. **cuDNN 이 깨져 있었다.** `nvidia-cudnn-cu13 9.19` 가 설치돼 있는데 torch 는 `+cu124` 라
   **모든 GPU conv 가** `CUDNN_STATUS_NOT_INITIALIZED` 로 실패했다 (32³ conv3d 조차).
   `nvidia-cudnn-cu12==9.1.0.70` 설치로 해결.
2. **`import ants` 가 깨져 있었다.** conda base 의 statsmodels 가 설치된 pandas 와 안 맞았다.
   statsmodels 0.15.0 으로 업그레이드해 해결.

설치한 것: `dipy 1.12.1`, `antspyx 0.6.3`, `nvidia-cudnn-cu12 9.1.0.70`, `statsmodels 0.15.0`.
설치하지 **않은** 것: MRtrix, FreeSurfer, MATLAB, DSI Studio, scilpy, singularity (불필요).

---

## 7. 현재 문제점

1. **GT SC 는 endpoint 가 아니라 `pass` 정의다.** `.mat` 의 info 에 명시되어 있고
   (`SC_weight=streamline count(pass)`), 실측으로도 endpoint r=0.673 / pass r=0.9986 이다.
   v2 §7·§8 의 endpoint 설계로는 GT SC 를 구조적으로 재현할 수 없다.
   → 사양대로 endpoint 를 **기본값**으로 구현하되 `--mode pass` 를 함께 제공했다.
   GT 재현이 목적이면 `pass` 를 써야 한다.

2. **공간 불일치.** GT tractogram 은 DSI Studio **QSDR 템플릿 공간**(전 subject 동일 격자)
   인데 ATM 원본은 subject T1 을 **rigid** 로만 MNI 에 올린다. QSDR 역변환이 든 `.fib.gz`
   는 이 서버에서 접근 불가(`/mnt/d` 없음)라, T1 을 GT 쪽 공간으로 SyN 정합하는 쪽으로
   진행했다(`mode='syn'`, `'rigid'` 로 ablation 가능). GT 가 이미 템플릿 공간이므로
   **subject-specific 한 "형상" 학습의 상한이 데이터에 의해 제한된다** — subject 차이는
   streamline 밀도/분포에 남는다. Methods 에 구분해 기술해야 한다.

3. **ATM UNet 전체 forward 는 23 GB 에서 OOM.** `infer.py` 가 버리는 segmentation 가지가
   193×229×193 에서 128채널 concat(4.4 GB)을 만든다. 인코더 가지만 실행하도록 했고
   CPU 전체 forward 와 **bit-exact** 임을 확인했다 (`scripts/00_check_env.py`).

4. **KDE `seed` 가 동작하지 않았다.** sklearn 의 `KernelDensity.sample` 은
   `self.random_state` 가 아니라 **메서드 인자**를 쓴다. 같은 seed 로 `max|z1-z2|=7.7` 이
   나왔다. `sample(n, random_state=seed)` 로 고쳐 재현성 0 확인. KDE 로드가 30 초라 캐시도 추가.

5. **AF_L 한 bundle 만으로는 82-ROI whole-brain GT SC 와 상관이 거의 없다**
   (endpoint r=0.005, pass r=0.239). 정상이다 — 30 bundle 중 1개다.
   **30-bundle coverage 측정 전에는 SC 성능을 해석하면 안 된다** (v2 §5.1).

6. **압축 해제 미완료**: PPMI 225/263명. `dwi_qc==pass` 이고 파일이 있는 subject 206명.

7. `antspyx` 가 scipy 를 1.17.1 → **1.15.3 으로 다운그레이드**했다. 다른 프로젝트 영향 확인 필요.

---

## 8. 다음 단계

1. real subject T1 / GT TRK / atlas alignment QC — 206명 SyN 정합 배치 + `check_alignment` 전원 통과
2. ATM original baseline — 30 bundle 전체 생성, `.trk` export, runtime/VRAM 측정
3. **30-bundle → GT SC coverage 측정** (v2 §5.1). 낮으면 SC loss 이전에 재검토
4. endpoint fine-tuning (+ `L_roi_visit` 병행 권장)
5. + SC corr
6. + SC magnitude
7. + tract length
8. GPU batch benchmark (streamline batch → subject batch → fp16)
9. 필요 시 whole-brain / ROI-pair conditioning 확장

부수 작업: bundle 별 정규화 상수를 우리 GT streamline 에서 재계산
(upstream 상수는 rigid-2009c 기준), subject split 은 `group`×`batch2` stratify.
