# ATM 기반 T1→TRK→SC 재현 Fine-tuning Framework

## 0. 프로젝트 목적

### 최종 목표
T1-weighted MRI만을 입력으로 사용하여:

```text
T1w MRI
  ↓
ATM-based streamline generator
  ↓
subject-specific tractogram (.trk/.tck)
  ↓
SC weight matrix
SC tract-length matrix
```

를 생성한다.

본 프로젝트의 핵심은 **단순히 streamline의 형태를 재현하는 것**이 아니라, 생성된 tractogram으로부터 계산한 structural connectivity (SC)가 ground-truth SC와 최대한 유사하도록 ATM을 fine-tuning하는 것이다.

### 현재 CoRNN 접근의 병목
현재 CoRNN 기반 접근은 streamline마다 point-to-point propagation을 수행한다.

예:

```text
8 subjects × 2048 streamlines × max 500 tracking steps
```

각 streamline의 다음 위치가 이전 위치에 의존하므로, GPU가 있어도 500-step sequential dependency가 남는다.

ATM은 complete streamline을 직접 생성하므로 이 recurrent propagation을 제거할 수 있다.

---

# 1. 왜 ATM을 사용하는가?

ATM (Anatomy-to-Tract Mapping)은 T1w MRI에서 white-matter streamline을 직접 생성한다.

논문:
- Tan et al., *Anatomy-to-tract mapping infers white matter pathways without diffusion streamline propagation*
- Nature Communications 17, Article 36 (2026)
- DOI: https://doi.org/10.1038/s41467-025-66615-w
- Official code + trained models: https://zenodo.org/records/15792527

ATM의 핵심:

```text
기존 tractography / CoRNN

seed
 ↓
point 1
 ↓
next direction
 ↓
point 2
 ↓
next direction
 ↓
...
 ↓
point N
```

ATM:

```text
T1
 ↓
Anatomy Encoder
 ↓
Anatomical feature
      +
Streamline latent vector
 ↓
Streamline Decoder
 ↓
Complete streamline
[128 × XYZ]
```

즉 streamline propagation loop가 없다.

ATM 원 논문에서는:
- streamline을 128개의 equidistant point로 resampling
- streamline latent dimension = 64
- bundle당 3000 streamlines를 training에 사용
- inference에서 3000 / 6000 / 9000 streamlines 생성 실험
- PyTorch 2.5.1 사용

---

# 2. 중요한 제한점

ATM 원본의 목적은 **30개 predefined white-matter bundle 생성**이다.

현재 프로젝트의 목적은:

```text
T1
 ↓
whole-brain tractogram
 ↓
atlas-based SC
```

이므로 ATM을 그대로 사용하는 것만으로는 부족할 수 있다.

따라서 두 단계로 진행한다.

## Phase A — ATM 재현

먼저 공식 ATM을 수정하지 않은 상태로 재현한다.

목표:
1. 공식 pretrained model inference 성공
2. T1 → streamline 생성 성공
3. 생성 결과를 `.trk` 또는 `.tck`로 저장
4. GT와 streamline/bundle metric 비교
5. 생성된 streamline으로 SC matrix 생성 가능 여부 확인
6. inference runtime / VRAM 측정

이 단계가 통과하기 전에는 SC-aware fine-tuning을 추가하지 않는다.

## Phase B — SC-aware ATM

ATM decoder를 기반으로 SC reconstruction objective를 추가한다.

최종 목표:

```text
T1
 ↓
ATM
 ↓
Predicted streamlines
 ↓
Differentiable SC Builder
 ↓
SC_pred
 ↓
SC_GT와 비교
 ↓
SC loss
 ↓
backpropagation
```

---

# 3. 권장 전체 Architecture

```text
                           ┌────────────────────┐
                           │      T1w MRI       │
                           └─────────┬──────────┘
                                     ↓
                           ┌────────────────────┐
                           │  Anatomy Encoder   │
                           │      E_A(T1)       │
                           └─────────┬──────────┘
                                     │
                              anatomy feature a
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
              latent z_1 ... z_N               optional condition
                    │                         bundle / ROI-pair
                    └────────────────┬────────────────┘
                                     ↓
                           ┌────────────────────┐
                           │ Streamline Decoder │
                           │       D_S          │
                           └─────────┬──────────┘
                                     ↓
                           [B, N, 128, 3]
                           complete streamlines
                                     │
                ┌────────────────────┼─────────────────────┐
                ↓                    ↓                     ↓
        streamline loss      differentiable SC      geometry / length
                                     │
                                     ↓
                              SC_weight_pred
                              SC_length_pred
                                     │
                                     ↓
                                  SC loss
                                     │
                                     ↓
                              Total objective
```

- B = subject batch
- N = streamline batch 또는 생성 streamline 수
- 128 = streamline당 resampled points
- 3 = x, y, z

---

# 4. Ground Truth

각 subject에 다음 자료를 준비하는 것을 권장한다.

```text
subject/
├── T1w.nii.gz
├── atlas.nii.gz
├── tractogram_gt.trk    # 또는 .tck
├── sc_weight_gt.npy
└── sc_length_gt.npy
```

SC matrix는 가능하면 training 전에 미리 계산해서 저장한다.

### SC weight
선택 가능한 정의:
1. streamline count
2. SIFT/SIFT2 weight
3. normalized streamline count
4. 기존 연구에서 사용 중인 SC weight 정의

현재 TVB 입력과 일치시키려면 **기존 pipeline에서 사용하는 동일한 SC weight 정의를 유지**한다.

### SC tract length
ROI i ↔ ROI j streamline들의:
- mean length
- 또는 현재 TVB pipeline에서 사용하는 정의

를 사용한다.

---

# 5. 가장 중요한 부분: Differentiable SC Builder

일반적인 MRtrix:

```text
TRK/TCK
 ↓
tck2connectome
 ↓
SC matrix
```

를 training graph 안에 직접 넣으면 gradient가 ATM까지 흐르지 않는다.

이유:
- atlas label lookup
- hard endpoint assignment
- 외부 CLI
- discrete operation

때문이다.

따라서 **training용 SC 계산기는 PyTorch 내부에서 differentiable하게 작성**한다.

최종 inference / evaluation에서는 다시 MRtrix/DSI Studio 결과와 비교한다.

---

## 5.1 Endpoint 기반 Soft ROI Assignment

streamline k:

```text
S_k = [p_1, p_2, ..., p_128]
```

endpoint:

```text
start_k = p_1
end_k   = p_128
```

각 endpoint가 ROI에 속할 확률을 hard label 대신 soft probability로 계산한다.

예:

```text
start:
ROI 1 = 0.01
ROI 2 = 0.91
ROI 3 = 0.08

end:
ROI 7 = 0.84
ROI 8 = 0.16
```

그 후 edge contribution:

```text
SC_k(i,j) =
P(start ∈ ROI_i) × P(end ∈ ROI_j)
```

전체 streamline에 대해:

```text
SC_pred(i,j) =
Σ_k SC_k(i,j)
```

이 계산을 PyTorch tensor operation으로 구현한다.

---

## 5.2 구현 후보

### Option A — ROI distance map 기반

각 ROI에 signed/unsigned distance map을 미리 생성한다.

endpoint 위치에서 `grid_sample()`로 distance를 interpolation한 뒤:

```text
P(ROI_i | p) = softmax(-distance_i(p) / τ)
```

로 assignment한다.

장점:
- differentiable
- 구현이 명확함
- endpoint 좌표에 gradient 가능

추천: **1순위**

---

### Option B — one-hot atlas probability sampling

atlas를 ROI one-hot volume으로 변환:

```text
[R, X, Y, Z]
```

endpoint에서 `torch.nn.functional.grid_sample()`로 probability를 sampling.

장점:
- 단순함

단점:
- hard one-hot atlas 경계에서는 gradient가 불안정할 수 있음

---

# 6. Loss Function

SC correlation만 단독 사용하지 않는다.

## 6.1 Streamline Reconstruction Loss

ATM 원본의 streamline reconstruction objective를 유지한다.

```text
L_stream
```

예:
- coordinate MSE
- VAE reconstruction loss
- KL loss

이는 TRK geometry 자체가 무너지지 않도록 한다.

---

## 6.2 Adjacent-point / Geometry Loss

ATM 원본의 geometric regularization 유지:

```text
L_adj
```

목적:
- unrealistic jumping 방지
- streamline smoothness 유지
- 좌표가 anatomical path를 유지하도록 도움

---

## 6.3 SC Correlation Loss

upper-triangle 또는 valid edge만 vectorization:

```text
v_pred = upper_triangle(SC_pred)
v_gt   = upper_triangle(SC_gt)
```

Pearson correlation:

```text
r_sc = corr(v_pred, v_gt)
```

loss:

```text
L_SC_corr = 1 - r_sc
```

목적:
- GT와 edge-wise pattern을 맞춤

---

## 6.4 SC Magnitude Loss

Correlation만 사용하면 scale이 틀려도 높은 correlation이 가능하다.

예:

```text
GT   = [1, 2, 5, 10]
Pred = [10, 20, 50, 100]
```

Pearson r = 1이 가능하다.

따라서 magnitude loss를 추가한다.

추천:

```text
L_SC_mag =
mean(
    |log(SC_pred + eps) - log(SC_gt + eps)|
)
```

또는 normalized MAE/MSE.

---

## 6.5 SC Tract-Length Loss

TVB에 tract length가 필요하므로 함께 학습한다.

각 edge:

```text
Length_pred(i,j)
```

를 differentiable weighted average로 계산:

```text
Length_pred(i,j) =
Σ_k w_k(i,j) * streamline_length_k
------------------------------------
Σ_k w_k(i,j) + eps
```

streamline length:

```text
length_k =
Σ_t ||p_(t+1) - p_t||_2
```

loss:

```text
L_length =
MAE(
    log(Length_pred + eps),
    log(Length_gt + eps)
)
```

단, GT edge가 존재하는 영역에 mask를 적용한다.

---

## 6.6 Optional Sparsity / Edge-existence Loss

SC topology까지 유지할 필요가 있다면:

```text
GT edge:
SC_gt(i,j) > threshold
```

에 대해 BCE 또는 focal loss를 사용할 수 있다.

```text
L_edge
```

초기 버전에서는 필수 아님.

---

# 7. 최종 추천 Objective

초기 fine-tuning:

```text
L_total =
λ_stream  * L_stream
+ λ_adj   * L_adj
+ λ_corr  * L_SC_corr
+ λ_mag   * L_SC_mag
+ λ_len   * L_length
```

권장 초기 가중치 탐색 시작점:

```text
λ_stream = 1.0
λ_adj    = 1.0
λ_corr   = 0.1
λ_mag    = 0.1
λ_len    = 0.05
```

주의:
- 위 값은 논문에서 검증된 최종값이 아니라 **초기 tuning용 시작점**
- 각 loss gradient magnitude를 기록한 뒤 재조정할 것
- SC loss를 처음부터 너무 크게 두면 streamline anatomy가 무너질 수 있음

---

# 8. 권장 Fine-tuning Strategy

## Stage 0 — Original ATM Reproduction

수정 없이:
- pretrained model load
- original inference
- sample subject에서 streamline 생성
- `.trk/.tck` export
- runtime 측정
- GPU utilization 측정

통과 조건:
- 공식 예제 output 재현
- NaN 없음
- streamline geometry 정상

---

## Stage 1 — Dataset Adapter

현재 데이터 구조로 변환:

```text
T1
GT tractogram
atlas
SC_weight
SC_length
```

ATM의 MNI-space assumptions와 현재 데이터 space를 확인한다.

반드시 좌표계 확인:
- voxel
- scanner RAS
- world RAS
- MNI

TRK 좌표 오류는 SC endpoint assignment를 완전히 망가뜨릴 수 있다.

---

## Stage 2 — Baseline without SC Loss

현재 데이터에서 ATM decoder를 fine-tuning하되:

```text
L = L_stream + L_adj
```

만 사용.

목적:
- ATM 자체가 데이터 domain에 적응 가능한지 확인
- SC loss를 넣기 전 baseline 확보

---

## Stage 3 — Differentiable SC Builder Validation

ATM과 연결하기 전에 SC Builder만 단독 검증한다.

GT tractogram을 입력하여:

```text
GT TRK
 ↓
Differentiable SC Builder
 ↓
SC_soft
```

를 생성.

그리고 동일 GT tractogram에 대해:

```text
GT TRK
 ↓
MRtrix tck2connectome
 ↓
SC_mrtrix
```

와 비교.

검증 지표:
- Pearson r
- CCC
- MAE
- RMSE
- edge existence F1
- degree correlation

목표:
`SC_soft`가 기존 SC 계산 결과와 충분히 유사해야 한다.

이 단계가 실패하면 ATM fine-tuning으로 넘어가지 않는다.

---

## Stage 4 — SC Correlation Fine-tuning

추가:

```text
L_SC_corr
```

즉:

```text
L =
L_stream
+ L_adj
+ λ_corr * L_SC_corr
```

validation에서:
- streamline metric
- SC correlation

둘 다 기록.

---

## Stage 5 — SC Magnitude Fine-tuning

추가:

```text
L_SC_mag
```

목적:
- correlation뿐 아니라 actual SC distribution 재현

---

## Stage 6 — Tract Length Fine-tuning

추가:

```text
L_length
```

최종:

```text
T1
 ↓
TRK
 ↓
SC_weight + SC_length
```

전체 pipeline tuning.

---

# 9. Whole-brain 확장 전략

ATM 원본은 30개 predefined bundles 중심이다.

따라서 바로 whole-brain SC를 완전하게 재현한다고 가정하지 않는다.

### 단계적 접근 추천

## Version 1
ATM 원본의 30 bundle로 SC를 구성하고:
- GT 전체 SC 중 해당 bundle이 설명하는 edge 평가
- SC correlation 개선 여부 검증

## Version 2
bundle set 확장 또는 whole-brain streamline decoder 개발

가능한 conditioning:

```text
T1 anatomy feature
+
ROI_start embedding
+
ROI_end embedding
+
streamline latent z
```

예:

```text
ROI 12 → ROI 37
```

조건에서 streamline distribution 생성.

개념:

```text
D_S(
    z,
    anatomy_feature,
    roi_i,
    roi_j
)
→ streamline
```

이 방식은 ATM 원본 이상의 방법론적 확장이므로 Stage 1부터 바로 하지 않는다.

---

# 10. GPU / Batch Strategy

ATM의 가장 큰 장점 중 하나는 streamline propagation dependency가 없다는 점이다.

## CoRNN

```text
2048 streamlines: step 1
 ↓
2048 streamlines: step 2
 ↓
...
 ↓
2048 streamlines: step 500
```

500 sequential stages가 남는다.

## ATM

```text
latent vectors [N, 64]
        ↓
Streamline Decoder
        ↓
streamlines [N, 128, 3]
```

N개의 streamline을 batch로 생성할 수 있다.

---

## 10.1 가장 우선할 batch

**Streamline batch**

한 subject T1을 encoder에 단 한 번 넣는다.

잘못된 구현:

```text
streamline 1 → T1 encoder → decoder
streamline 2 → T1 encoder → decoder
...
```

권장:

```text
T1
 ↓
Encoder 1회
 ↓
anatomy feature cache
 ↓
latent batch 1 [N1,64] → decoder
latent batch 2 [N2,64] → decoder
...
```

---

## 10.2 Subject Batch

VRAM이 충분하면:

```text
T1 batch:
[B, C, D, H, W]
```

streamline output:

```text
[B, N, 128, 3]
```

으로 처리 가능.

다만 3D T1 encoder가 VRAM을 많이 사용하므로 현실적으로:

```text
subject batch: 1~4
streamline batch: 가능한 크게
```

부터 시작한다.

---

## 10.3 GPU별 튜닝

GPU가 커질수록:
- streamline batch 증가
- subject batch 증가
- mixed precision 적용
- gradient accumulation 감소

가능.

권장:
- BF16: H100/A100 등 지원 GPU에서 우선 검토
- FP16: numerical stability 확인 후 적용
- AMP 사용
- `torch.compile()`은 baseline correctness 확보 후 benchmark
- DataLoader `pin_memory=True`
- non-blocking H2D transfer
- persistent workers
- precomputed SC/atlas maps

중요:
GPU별 실제 속도 향상은 반드시 동일한 데이터/streamline 수에서 benchmark한다.

---

# 11. Runtime Benchmark

각 pipeline에 대해 동일한 subject에서 측정:

```text
1. T1 preprocessing
2. T1 encoder
3. streamline generation
4. filtering
5. TRK/TCK export
6. SC computation
7. total inference
```

비교 대상:

```text
CoRNN baseline
ATM original
ATM fine-tuned
ATM + SC-aware
```

측정 항목:

```text
seconds / subject
streamlines / second
peak VRAM
average GPU utilization
SC corr
SC CCC
TRK geometry metrics
tract-length error
```

속도만 빠르고 SC 품질이 떨어지는 모델은 최종 후보로 사용하지 않는다.

---

# 12. Evaluation

## TRK / Streamline Level

가능한 metric:
- streamline coordinate error
- endpoint distance
- valid streamline ratio
- bundle overlap
- bundle coverage
- streamline length distribution
- Tractometer metrics

---

## SC Weight Level

필수:

```text
Pearson r
Spearman r
CCC
MAE
RMSE
```

추가:
- edge-wise error
- node degree correlation
- graph density
- nonzero edge F1

---

## SC Length Level

```text
Pearson r
CCC
MAE
RMSE
```

GT에 edge가 존재하는 mask 안에서 계산.

---

## Generalization

train / validation / test subject를 분리한다.

fine-tuning 과정에서 test SC를 절대 objective tuning에 사용하지 않는다.

---

# 13. Ablation Study

최종 연구에서는 최소 다음 비교를 권장.

| Model | Streamline loss | SC corr | SC magnitude | Length |
|---|---:|---:|---:|---:|
| ATM baseline | O | X | X | X |
| ATM + corr | O | O | X | X |
| ATM + corr + mag | O | O | O | X |
| ATM + full | O | O | O | O |

비교:
- TRK quality
- SC corr
- SC CCC
- length corr
- inference time

---

# 14. 권장 Repository 구조

```text
t1_atm_sc/
│
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
│
├── docs/
│   └── ATM_SC_Aware_Finetuning_Framework.md
│
├── external/
│   └── atm_upstream/
│       └── # Zenodo 공식 ATM 코드
│
├── src/
│   └── atm_sc/
│       ├── models/
│       │   ├── atm_adapter.py
│       │   └── sc_builder.py
│       │
│       ├── losses/
│       │   ├── sc_corr.py
│       │   ├── sc_magnitude.py
│       │   └── tract_length.py
│       │
│       ├── data/
│       │   ├── dataset.py
│       │   └── transforms.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   └── stages.py
│       │
│       ├── inference/
│       │   ├── generate_tractogram.py
│       │   └── export_trk.py
│       │
│       └── evaluation/
│           ├── evaluate_trk.py
│           └── evaluate_sc.py
│
├── scripts/
│   ├── 00_check_upstream.py
│   ├── 01_reproduce_atm.py
│   ├── 02_build_gt_sc.py
│   ├── 03_validate_soft_sc.py
│   ├── 04_train_baseline.py
│   ├── 05_train_sc_corr.py
│   ├── 06_train_sc_full.py
│   └── 07_benchmark.py
│
├── configs/
│   ├── atm_reproduce.yaml
│   ├── baseline.yaml
│   ├── sc_corr.yaml
│   └── sc_full.yaml
│
├── data/
│   └── # 실제 대용량 데이터는 git에 넣지 않음
│
└── outputs/
    ├── checkpoints/
    ├── tractograms/
    ├── sc/
    └── benchmarks/
```

---

# 15. 원본 ATM 코드는 어떻게 준비할 것인가?

## 결론

**Zenodo URL만 Claude Code 채팅창에 넣는 것보다, 공식 코드를 Claude Code가 접근 가능한 로컬/서버 작업 디렉터리에 실제로 다운로드하고 압축을 푸는 것을 강력히 권장한다.**

URL만 전달하면:
- 전체 archive 구조를 안정적으로 탐색하기 어려울 수 있음
- 모델 파일/설정/상대 경로를 실제로 실행할 수 없음
- 파일을 수정하고 test하기 어려움
- 다운로드 권한/웹 접근 여부에 의존
- 긴 코드를 웹 페이지에서 부분적으로 읽는 것보다 로컬 repo 검색이 훨씬 정확함

### 권장 방식

```text
t1_atm_sc/
└── external/
    └── atm_upstream/
        ├── [공식 ATM source]
        ├── [config]
        ├── [model]
        └── ...
```

ATM 공식 source:
https://zenodo.org/records/15792527

---

# 16. Claude Code 환경에 ATM 가져오기

## 방법 A — 가장 안전: 직접 다운로드

1. Zenodo record에서 source code / trained model 다운로드
2. 압축 해제
3. 프로젝트 아래에 배치

```bash
mkdir -p external/atm_upstream
# 다운로드한 archive를 external/atm_upstream 아래에 압축 해제
```

그 다음 Claude Code를 프로젝트 root에서 실행:

```bash
cd t1_atm_sc
claude
```

Claude가 다음을 모두 읽을 수 있게 한다:

```text
external/atm_upstream/
docs/ATM_SC_Aware_Finetuning_Framework.md
```

---

## 방법 B — Claude Code terminal에서 직접 다운로드

Claude Code 환경 자체에 인터넷 연결이 가능하다면 먼저 Zenodo metadata를 확인할 수 있다.

예:

```bash
curl -L \
  https://zenodo.org/api/records/15792527 \
  -o /tmp/atm_zenodo_record.json
```

그 후 record metadata에 들어 있는 실제 file download URL을 확인하여:

```text
external/atm_upstream/
```

에 다운로드/압축 해제한다.

**다운로드가 끝난 후에는 로컬 파일을 기준으로 분석/수정하도록 한다.**

즉 URL은 "원본 위치를 알려주는 용도"이고,
실제 구현은 "local source tree를 읽고 수정하는 방식"이 좋다.

---

# 17. 원본 코드를 직접 수정하지 않는 이유

`external/atm_upstream/`는 가능한 read-only reference처럼 취급한다.

새 기능은:

```text
src/atm_sc/
```

에 작성한다.

이유:
1. upstream 원본과 수정본 비교 가능
2. 재현성 유지
3. bug 발생 시 ATM 원본 동작 여부 확인 가능
4. 논문 Methods에서 변경점을 명확히 설명 가능
5. upstream 업데이트/재다운로드가 쉬움

필요한 class/function은:
- import
- wrapper
- subclass
- 최소 patch

순으로 활용한다.

원본을 수정해야 한다면 patch를 별도 기록한다.

---

# 18. Claude Code에 처음 줄 Prompt

아래와 같이 시작하는 것을 권장한다.

```text
이 프로젝트의 목표는 T1w MRI에서 tractogram을 생성하는 ATM을 기반으로,
생성된 tractogram의 geometry뿐 아니라 structural connectivity(SC) weight와
tract-length matrix도 ground truth와 잘 일치하도록 fine-tuning하는 것이다.

먼저 코드를 수정하지 말고 다음 작업만 수행해라.

1. docs/ATM_SC_Aware_Finetuning_Framework.md를 읽어라.
2. external/atm_upstream/의 ATM 공식 코드를 전체적으로 조사해라.
3. T1 encoder, streamline encoder/decoder, VAE loss, inference, streamline export와
   관련된 파일/class/function을 찾아라.
4. ATM에서 N개의 64-D latent vector가 어떻게 N개의 128-point streamline으로
   변환되는지 호출 흐름을 정리해라.
5. T1 encoder가 inference 중 streamline마다 반복 호출되는지, subject당 한 번
   호출되는지 확인해라.
6. SC-aware fine-tuning을 추가할 때 수정이 필요한 최소 파일 목록을 제안해라.
7. 원본 ATM 동작을 깨지 않도록 external/atm_upstream은 수정하지 않는 방향으로
   wrapper/adapter 설계를 제안해라.
8. 아직 코드를 구현하지 말고 ATM 원본 분석 결과와 구현 계획만 Markdown으로 작성해라.

중요:
- 추측하지 말고 실제 source code의 파일명과 함수명을 근거로 설명할 것.
- whole-brain SC 목적과 ATM의 30-bundle limitation을 구분할 것.
- 기존 point-to-point recurrent tracking을 새로 추가하지 말 것.
```

첫 단계에서 바로 코드 생성을 시키지 않는 것이 중요하다.

---

# 19. Claude Code 구현 순서

Claude에게 한 번에 전체 시스템을 만들라고 하지 않는다.

## Task 1
ATM 공식 코드 구조 분석

## Task 2
공식 pretrained inference 완전 재현

## Task 3
runtime benchmark

## Task 4
TRK/TCK export 확인

## Task 5
GT SC preprocessing script

## Task 6
Differentiable SC Builder 단독 구현

## Task 7
GT tractogram으로 Soft SC Builder 검증

## Task 8
ATM decoder와 SC Builder 연결

## Task 9
SC correlation loss 추가

## Task 10
SC magnitude loss 추가

## Task 11
tract-length loss 추가

## Task 12
GPU batching / AMP 최적화

## Task 13
ablation + benchmark

이 순서가 디버깅하기 가장 쉽다.

---

# 20. 구현에서 반드시 지켜야 할 원칙

### 1. 최초 목표는 "공식 ATM 재현"
처음부터 모델 구조를 바꾸지 않는다.

### 2. `tck2connectome`을 training loss graph 안에 넣지 않는다.
학습 중 SC는 PyTorch differentiable approximation을 사용.

### 3. 평가에서는 실제 MRtrix/DSI Studio SC와 다시 비교
soft SC가 좋은 것만으로 끝내지 않는다.

### 4. SC correlation만 최적화하지 않는다.
SC magnitude + tract length도 함께 평가/최적화.

### 5. TRK geometry를 희생해서 SC만 좋아지는 것을 방지
streamline reconstruction/geometric loss를 유지.

### 6. T1 encoder output은 subject별 재사용
streamline마다 T1 encoder를 다시 실행하지 않는다.

### 7. GPU 최적화는 correctness 이후
먼저 FP32 기준이 맞는지 확인한 후 BF16/AMP/batching 최적화.

### 8. 속도 benchmark는 동일한 streamline 수로 비교
CoRNN과 ATM 모두 동일한 조건에서:
- subject 수
- target streamline 수
- GPU
- preprocessing 범위

를 맞춘다.

---

# 21. 최종 성공 기준

최종 모델은 다음 세 축에서 동시에 평가한다.

## A. Speed

```text
ATM_SC runtime << CoRNN runtime
```

특히 streamline generation 부분.

## B. Tractogram fidelity

```text
TRK_pred ≈ TRK_GT
```

geometry / valid streamline / endpoint / length distribution 평가.

## C. Connectome fidelity

```text
SC_weight_pred ≈ SC_weight_GT
SC_length_pred ≈ SC_length_GT
```

최종적으로 TVB에 넣을 수 있는 수준인지 검증.

---

# 22. 프로젝트 핵심 Hypothesis

> Point-to-point recurrent streamline propagation 없이 complete streamline을 직접
> 생성하는 ATM을 SC-aware하게 fine-tuning하면, CoRNN보다 tractogram 생성 시간을
> 크게 줄이면서도 subject-specific SC weight 및 tract-length matrix의 재현도를
> 향상시킬 수 있다.

---

# 23. 가장 우선해서 검증할 질문

1. 공식 ATM pretrained inference를 현재 GPU에서 실행했을 때 subject당 몇 초가 걸리는가?
2. `.trk/.tck` 생성까지 포함하면 몇 초인가?
3. 동일 streamline 수에서 CoRNN 대비 얼마나 빠른가?
4. ATM 30 bundle만으로 현재 atlas SC edge의 몇 %를 설명할 수 있는가?
5. ATM baseline의 SC corr / CCC는 얼마인가?
6. differentiable soft SC가 MRtrix SC와 얼마나 유사한가?
7. `L_SC_corr` 추가 후 SC corr가 개선되는가?
8. SC 개선이 streamline geometry 저하를 일으키는가?
9. magnitude loss 추가 시 CCC/MAE가 개선되는가?
10. tract-length loss 추가 시 TVB용 length matrix 재현성이 개선되는가?
11. streamline batch를 512→1024→2048→4096로 늘릴 때 throughput이 얼마나 증가하는가?
12. A10/A100/H100 등 GPU 변경에 따른 실제 speedup은 얼마인가?

---

## Reference

Tan Y-F, Huynh KM, Liu S, et al.
**Anatomy-to-tract mapping infers white matter pathways without diffusion streamline propagation.**
Nature Communications. 2026;17:36.

Paper:
https://doi.org/10.1038/s41467-025-66615-w

Official source code and trained models:
https://zenodo.org/records/15792527
