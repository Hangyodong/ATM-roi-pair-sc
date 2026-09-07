# ATM 기반 T1→TRK→SC 재현 Fine-tuning Framework v2

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

본 프로젝트의 핵심은 단순히 streamline 좌표를 재현하는 것이 아니라:

1. GT tractography의 streamline geometry를 재현하고,
2. 각 streamline이 올바른 ROI pair를 연결하도록 만들고,
3. 생성된 tractogram으로부터 계산한 SC matrix가 GT SC와 유사하도록 하며,
4. 최종적으로 SC weight와 tract-length matrix를 TVB 등의 whole-brain model에 사용할 수 있도록 하는 것이다.

---

# 1. 현재 CoRNN 접근의 병목

현재 CoRNN 기반 접근은 streamline마다 point-to-point propagation을 수행한다.

예:

```text
8 subjects × 2048 streamlines × max 500 tracking steps
```

streamline의 다음 위치는 이전 위치에 의존한다.

```text
p1 → p2 → p3 → ... → p500
```

따라서 GPU로 여러 streamline을 batch 처리하더라도:

```text
2048 streamline: step 1
        ↓
2048 streamline: step 2
        ↓
...
        ↓
2048 streamline: step 500
```

과 같은 sequential dependency가 남는다.

즉 계산량뿐 아니라 sequential depth 자체가 병목이다.

---

# 2. 왜 ATM을 사용하는가?

ATM (Anatomy-to-Tract Mapping)은 T1w MRI에서 complete streamline을 직접 생성한다.

논문:
- Tan et al.
- *Anatomy-to-tract mapping infers white matter pathways without diffusion streamline propagation*
- Nature Communications, 2026
- DOI: https://doi.org/10.1038/s41467-025-66615-w
- Official source code + trained models:
  https://zenodo.org/records/15792527

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

즉 recurrent streamline propagation loop가 없다.

ATM 원 논문에서는:
- streamline을 128개의 equidistant point로 resampling
- streamline latent dimension = 64
- bundle당 3000 streamlines를 training에 사용
- inference에서 3000 / 6000 / 9000 streamline 생성 실험
- PyTorch 기반 구현

---

# 3. 현재 보유 데이터

현재 각 subject에 대해 다음 데이터를 보유한다고 가정한다.

```text
subject/
├── T1w.nii.gz
├── tractogram_gt.trk     # 또는 .tck
├── sc_weight_gt.npy
├── sc_length_gt.npy      # 없으면 GT tractography에서 생성 가능
└── atlas.nii.gz
```

필수:
- T1
- GT tractography
- GT SC matrix
- SC 계산에 사용한 동일 atlas

선택:
- GT tract-length matrix

DWI는 ATM fine-tuning 자체에는 반드시 필요하지 않다.
이미 GT tractography와 GT SC를 보유하고 있기 때문이다.

---

# 4. 프로젝트의 핵심 가설

> Point-to-point recurrent streamline propagation 없이 complete streamline을 직접 생성하는 ATM을 기반으로, streamline geometry뿐 아니라 endpoint connectivity와 SC-level objective를 함께 최적화하면, CoRNN보다 tractogram 생성 시간을 크게 줄이면서 subject-specific SC weight 및 tract-length matrix의 재현도를 향상시킬 수 있다.

---

# 5. ATM 원본의 중요한 제한

ATM 원본은 whole-brain unrestricted tractography가 아니라 30개의 predefined white-matter bundle 중심으로 학습되었다.

따라서 현재 목적:

```text
T1
 ↓
whole-brain tractogram
 ↓
whole-brain SC
```

과 완전히 동일하지 않을 수 있다.

이 때문에 SC loss를 추가하기 전에 반드시 다음을 확인한다.

## 5.1 ATM bundle coverage test

GT SC의 non-zero edge 중 ATM의 30 bundle이 설명할 수 있는 edge 비율을 계산한다.

```text
coverage =
ATM bundle로 설명 가능한 GT nonzero edge 수
-----------------------------------------
전체 GT nonzero edge 수
```

필수 평가:
- edge coverage %
- weighted SC coverage %
- 주요 long-range edge coverage
- 주요 subcortical edge coverage

coverage가 낮다면 SC_corr loss를 추가해도 모델 표현력의 한계 때문에 전체 SC를 재현하기 어렵다.

즉 낮은 SC 성능이 loss 문제인지 ATM의 30-bundle 표현력 문제인지 먼저 구분해야 한다.

---

# 6. 권장 전체 Architecture

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
                     ┌───────────────┴───────────────┐
                     │                               │
               latent z_1 ... z_N             optional condition
                     │                       bundle / ROI-pair
                     └───────────────┬───────────────┘
                                     ↓
                           ┌────────────────────┐
                           │ Streamline Decoder │
                           │       D_S          │
                           └─────────┬──────────┘
                                     ↓
                           [B, N, 128, 3]
                           complete streamlines
                                     │
         ┌───────────────────────────┼────────────────────────────┐
         ↓                           ↓                            ↓
  streamline geometry        endpoint connectivity        differentiable SC
         │                           │                            │
         ↓                           ↓                            ↓
    L_stream / L_adj            L_endpoint             SC_weight_pred
                                                             │
                                                             ├── L_SC_corr
                                                             ├── L_SC_mag
                                                             └── L_length
```

- B = subject batch
- N = streamline 수
- 128 = streamline당 resampled points
- 3 = x,y,z

---

# 7. 핵심 수정 1 — Endpoint Connectivity Loss

## 7.1 왜 필요한가?

SC_corr는 subject 전체 tractogram 수준의 global objective이다.

그러나 SC_corr만 사용하면 특정 streamline이 실제로 올바른 ROI pair를 연결하는지 직접 감독하지 않는다.

예:

```text
GT streamline:
ROI 12 → ROI 37

Pred streamline:
ROI 12 → ROI 41
```

개별 streamline geometry가 비슷해도 SC topology는 달라질 수 있다.

따라서 각 streamline이 GT와 동일한 endpoint ROI pair를 연결하도록 직접 학습시킨다.

## 7.2 GT endpoint label 생성

GT tractography의 각 streamline k에 대해:

```text
start_gt(k) → ROI_i
end_gt(k)   → ROI_j
```

를 atlas로 계산하여 저장한다.

방향성이 중요하지 않은 tractography라면 `(i,j) == (j,i)`가 되도록 canonical ordering을 사용한다.

## 7.3 Soft endpoint assignment

Pred streamline:

```text
S_k = [p_1, ..., p_128]
```

endpoint:

```text
p_start = p_1
p_end   = p_128
```

각 endpoint가 ROI에 속할 확률 `q_start`, `q_end`를 differentiable하게 계산한다.

### 추천 — ROI distance-map 기반

각 ROI별 distance map을 미리 생성하고 endpoint 위치에서 `grid_sample()`로 distance를 interpolation한다.

```text
q_i(p) = softmax(-d_i(p) / τ)
```

장점:
- endpoint 좌표에 gradient가 흐름
- ROI boundary에서도 smoother
- 구현이 명확함

## 7.4 Endpoint loss

GT start ROI = i, GT end ROI = j 라면:

```text
L_endpoint =
CE(q_start, i)
+
CE(q_end, j)
```

방향성이 없는 경우 direct/reverse 두 방향의 loss 중 더 작은 값을 사용하되, hard min 대신 soft-min도 검토한다.

이 loss는 **local streamline-level supervision**이다.

---

# 8. 핵심 수정 2 — Differentiable SC Builder

일반적인 MRtrix:

```text
TRK/TCK
 ↓
tck2connectome
 ↓
SC matrix
```

를 training graph 안에 직접 넣으면 gradient가 ATM decoder까지 흐르지 않는다.

따라서 training용 SC builder를 PyTorch 내부에 구현한다.

## 8.1 Soft SC 계산

streamline k의 start/end ROI probability:

```text
q_start(k)
q_end(k)
```

edge contribution:

```text
W_k(i,j) =
q_start(k,i) * q_end(k,j)
```

undirected SC라면:

```text
W_k =
0.5 * (
    q_start ⊗ q_end
    +
    q_end ⊗ q_start
)
```

전체 streamline:

```text
SC_pred = Σ_k W_k
```

필요하면 streamline-specific weight를 곱한다.

---

# 9. 왜 SC_corr를 넣는가?

ATM 원 논문은 SC_corr를 직접 loss로 사용하지 않았지만, 생성 tractography에서 계산한 SC correlation이 약 0.28~0.61 수준까지 기록되었다.

이는 streamline geometry 학습만으로 endpoint distribution이 어느 정도 재현되어 SC도 따라온다는 의미다.

따라서 SC를 직접 objective에 넣으면 connectome-level fidelity를 추가로 개선할 가능성이 있다.

하지만 **SC_corr loss를 넣는다고 반드시 validation/test SC correlation이 향상되는 것은 아니다.** 반드시 ablation으로 검증한다.

---

# 10. SC Correlation Loss

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

역할:
- SC edge pattern
- 상대적인 strong/weak connectivity structure

---

# 11. SC_corr만 사용하면 안 되는 이유

Pearson correlation은 scale에 민감하지 않다.

예:

```text
GT:   [1, 2, 5, 10]
Pred: [10, 20, 50, 100]
```

Pearson `r = 1`이 가능하다.

따라서 SC_corr만 최적화하면 실제 connection magnitude는 크게 틀릴 수 있다.

---

# 12. SC Magnitude Loss

추천:

```text
L_SC_mag =
mean(
    |log(SC_pred + eps) - log(SC_gt + eps)|
)
```

또는 MAE/MSE/Huber/normalized MSE를 검토할 수 있다.

로그 변환을 사용하는 이유:
- SC weight dynamic range 완화
- high-weight edge에 loss가 지나치게 지배되는 것 방지

---

# 13. Tract-Length Loss

TVB 입력에 tract-length matrix가 필요하다면 추가한다.

각 streamline:

```text
length_k =
Σ_t ||p_(t+1) - p_t||_2
```

soft edge weight `w_k(i,j)`를 사용하여:

```text
Length_pred(i,j) =
Σ_k w_k(i,j) * length_k
-------------------------
Σ_k w_k(i,j) + eps
```

GT edge가 존재하는 영역에 mask를 적용한다.

추천 loss:

```text
L_length =
MAE(
    log(Length_pred + eps),
    log(Length_gt + eps)
)
```

---

# 14. ATM 원본 Loss 유지

SC를 추가하더라도 ATM 원래 geometry objective를 제거하지 않는다.

이유: SC loss만 강하게 주면 endpoint만 맞추고 실제 경로가 비현실적으로 변할 수 있다.

따라서 streamline reconstruction, VAE objective, adjacency/geometry regularization 등 ATM 원래 loss를 유지한다.

---

# 15. 최종 추천 Objective

```text
L_total =
λ_atm        * L_ATM
+ λ_endpoint * L_endpoint
+ λ_corr     * L_SC_corr
+ λ_mag      * L_SC_mag
+ λ_len      * L_length
```

개념적으로:

```text
Local streamline supervision:
- L_ATM
- L_endpoint

Global tractogram supervision:
- L_SC_corr
- L_SC_mag
- L_length
```

이다.

---

# 16. Loss 역할 정리

| Loss | Level | 목적 |
|---|---|---|
| L_ATM | streamline | 경로/형태 재현 |
| L_endpoint | streamline | 올바른 ROI pair 연결 |
| L_SC_corr | subject/tractogram | SC 전체 pattern 재현 |
| L_SC_mag | subject/tractogram | 실제 edge strength scale 재현 |
| L_length | subject/tractogram | TVB용 tract-length matrix 재현 |

---

# 17. 권장 초기 Loss Weight

초기 실험용 예시:

```text
λ_atm      = 1.0
λ_endpoint = 0.1
λ_corr     = 0.05
λ_mag      = 0.05
λ_len      = 0.02
```

주의:
- 위 값은 검증된 최종값이 아니다.
- gradient magnitude를 기록한 후 조정한다.
- SC-related loss를 처음부터 크게 두지 않는다.

가능하면 각 loss의 gradient norm을 기록한다.

---

# 18. 중요한 변경 — SC Loss는 Subject-level로 계산

SC는 개별 streamline 하나의 특성이 아니라 streamline 집합 전체의 특성이다.

잘못된 방식:

```text
streamline 1 → SC loss
streamline 2 → SC loss
...
```

올바른 방식:

```text
Subject 1
 ├─ streamline 1
 ├─ streamline 2
 ├─ ...
 └─ streamline N
        ↓
Differentiable SC Builder
        ↓
SC_pred_subject1
        ↓
SC_gt_subject1
```

즉 `L_SC_corr`, `L_SC_mag`, `L_length`는 subject-level tractogram set에서 계산해야 한다.

---

# 19. Streamline batching과 Subject batching 구분

## 19.1 Streamline batch

한 subject의 T1을 encoder에 한 번 넣고 anatomy feature를 cache한다.

```text
T1
 ↓
Anatomy Encoder
 ↓
anatomy feature cache
 ↓
latent batch [N,64]
 ↓
decoder
 ↓
[N,128,3]
```

이게 가장 중요한 GPU 최적화다.

## 19.2 Subject batch

VRAM이 충분하면:

```text
[B, C, D, H, W]
```

T1 batch를 입력하고:

```text
[B, N, 128, 3]
```

streamline을 동시에 생성할 수 있다.

현실적으로는:
- subject batch = 1~4
- streamline batch = 가능한 크게

부터 benchmark한다.

---

# 20. 권장 Fine-tuning Strategy

## Stage 0 — Official ATM Reproduction
- pretrained model load
- official inference 성공
- streamline 생성
- `.trk/.tck` export
- runtime 측정
- peak VRAM 측정

## Stage 1 — Data Alignment / Coordinate QC
확인:
- T1 space
- GT tractography space
- atlas space
- SC atlas 정의
- voxel/world/RAS 좌표

## Stage 2 — ATM 30-bundle SC Coverage Test
GT tractography를 기준으로 ATM 30 bundle이 설명 가능한 SC edge와 설명 불가능한 edge를 분리한다.

## Stage 3 — ATM Baseline Fine-tuning

```text
L = L_ATM
```

SC 관련 loss 없는 baseline 확보.

## Stage 4 — Endpoint Connectivity Supervision

```text
L =
L_ATM
+ λ_endpoint L_endpoint
```

검증:
- endpoint ROI accuracy
- endpoint pair accuracy
- TRK geometry
- SC corr

## Stage 5 — Differentiable SC Builder 단독 검증

```text
GT TRK
 ↓
Differentiable SC Builder
 ↓
SC_soft
```

비교:

```text
GT TRK
 ↓
MRtrix / DSI Studio
 ↓
SC_reference
```

지표:
- Pearson r
- Spearman r
- CCC
- MAE
- RMSE
- edge F1

## Stage 6 — SC Correlation Fine-tuning

```text
L =
L_ATM
+ λ_endpoint L_endpoint
+ λ_corr L_SC_corr
```

## Stage 7 — SC Magnitude Fine-tuning

```text
+ λ_mag L_SC_mag
```

## Stage 8 — Tract-Length Fine-tuning

```text
+ λ_len L_length
```

최종:

```text
T1
 ↓
TRK
 ↓
SC_weight
+
SC_length
```

---

# 21. Ablation Study

| Model | ATM | Endpoint | SC Corr | SC Mag | Length |
|---|---:|---:|---:|---:|---:|
| ATM baseline | O | X | X | X | X |
| ATM + Endpoint | O | O | X | X | X |
| ATM + Endpoint + Corr | O | O | O | X | X |
| ATM + Endpoint + Corr + Mag | O | O | O | O | X |
| ATM + Full | O | O | O | O | O |

평가:
- TRK quality
- endpoint ROI accuracy
- endpoint pair accuracy
- SC corr
- SC CCC
- SC MAE
- SC RMSE
- tract length corr
- inference runtime

---

# 22. 성공 여부 해석

## Case A

```text
SC corr ↑
TRK quality 유지
```

→ 가장 이상적

## Case B

```text
SC corr ↑
TRK quality ↓
```

→ SC loss weight가 너무 크거나 geometry constraint가 약함

## Case C

```text
SC corr 변화 없음
TRK quality 유지
```

→ differentiable SC gradient가 약하거나 ATM 표현력/30-bundle coverage 한계 가능성

## Case D

```text
SC corr train ↑
SC corr validation/test ↓
```

→ SC overfitting 가능성

---

# 23. Whole-brain 확장 전략

ATM 30-bundle coverage가 충분하지 않다면 다음 확장을 검토한다.

## Option A — bundle set 확장
ATM 방식은 유지하되 bundle class 수 증가.

## Option B — ROI-pair conditioned streamline generation

조건:

```text
anatomy feature
+
ROI_start embedding
+
ROI_end embedding
+
latent z
```

Decoder:

```text
D(z, anatomy, ROI_i, ROI_j)
→ streamline
```

장점:
- SC 목적과 직접 연결
- arbitrary ROI pair generation 가능

단점:
- ATM 원본보다 구조 수정이 큼
- ROI pair imbalance 처리 필요

---

# 24. GPU / Speed Strategy

ATM의 장점:

```text
CoRNN:
2048 × 최대 500 sequential tracking steps

ATM:
2048 complete streamlines를 batch tensor로 생성
```

GPU가 커질수록:
- streamline batch 증가
- subject batch 증가
- BF16/FP16 mixed precision
- tensor core 활용
- throughput 개선

이 가능하다.

동일 subject, 동일 streamline 수에서 다음을 비교한다:

```text
CoRNN
ATM original
ATM fine-tuned
ATM SC-aware
```

측정:
- sec / subject
- streamlines / sec
- peak VRAM
- GPU utilization
- SC corr
- TRK quality

---

# 25. Runtime Breakdown

각 단계별 측정:

```text
1. T1 preprocessing
2. T1 encoder
3. streamline generation
4. post-processing / filtering
5. TRK export
6. SC computation
7. total inference
```

그래야 병목이 어디인지 명확히 알 수 있다.

---

# 26. Evaluation

## TRK Level
- coordinate error
- endpoint distance
- valid streamline ratio
- streamline length distribution
- bundle overlap
- bundle coverage
- Tractometer metric

## Endpoint Connectivity
- start ROI accuracy
- end ROI accuracy
- ROI-pair accuracy

## SC Weight
- Pearson r
- Spearman r
- CCC
- MAE
- RMSE
- edge F1
- density
- degree correlation

## SC Length
- Pearson r
- CCC
- MAE
- RMSE

---

# 27. Generalization / Overfitting

train / validation / test split을 subject-level로 분리한다.

절대 하지 말 것:
- 같은 subject의 streamline을 train/test에 나누기
- test SC를 loss weight tuning에 사용

특히 SC_corr를 직접 넣기 때문에 다음을 기록한다:

```text
Train SC corr
Validation SC corr
Test SC corr
```

Train만 상승하고 validation/test가 정체 또는 하락하면 SC pattern overfitting을 의심한다.

---

# 28. 권장 Repository 구조

```text
t1_atm_sc/
│
├── README.md
├── CLAUDE.md
├── docs/
│   └── ATM_SC_Aware_Finetuning_Framework_v2.md
│
├── external/
│   └── atm_upstream/
│       └── # Zenodo 공식 ATM 코드
│
├── src/
│   └── atm_sc/
│       ├── models/
│       │   ├── atm_adapter.py
│       │   ├── endpoint_assigner.py
│       │   └── sc_builder.py
│       ├── losses/
│       │   ├── endpoint.py
│       │   ├── sc_corr.py
│       │   ├── sc_magnitude.py
│       │   └── tract_length.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── endpoint_labels.py
│       │   └── transforms.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── stages.py
│       ├── inference/
│       │   ├── generate_tractogram.py
│       │   └── export_trk.py
│       └── evaluation/
│           ├── evaluate_trk.py
│           ├── evaluate_endpoint.py
│           └── evaluate_sc.py
│
├── scripts/
│   ├── 00_check_upstream.py
│   ├── 01_reproduce_atm.py
│   ├── 02_check_coordinate_space.py
│   ├── 03_check_bundle_sc_coverage.py
│   ├── 04_build_gt_endpoint_labels.py
│   ├── 05_build_gt_sc.py
│   ├── 06_validate_soft_sc.py
│   ├── 07_train_baseline.py
│   ├── 08_train_endpoint.py
│   ├── 09_train_sc_corr.py
│   ├── 10_train_sc_full.py
│   └── 11_benchmark.py
│
├── configs/
│   ├── atm_reproduce.yaml
│   ├── baseline.yaml
│   ├── endpoint.yaml
│   ├── sc_corr.yaml
│   └── sc_full.yaml
│
└── outputs/
    ├── checkpoints/
    ├── tractograms/
    ├── sc/
    ├── endpoint_metrics/
    └── benchmarks/
```

---

# 29. Claude Code 환경 준비

Zenodo URL만 Claude Code 채팅에 전달하는 것보다 ATM 공식 source code와 trained model을 실제 작업 환경에 다운로드하여 Claude Code가 직접 파일을 읽을 수 있도록 하는 것이 좋다.

권장:

```text
t1_atm_sc/
└── external/
    └── atm_upstream/
        ├── source code
        ├── configs
        ├── pretrained model
        └── ...
```

Official source:
https://zenodo.org/records/15792527

`external/atm_upstream/`는 reference로 보존하고 새 기능은 `src/atm_sc/`에 wrapper/adapter 형태로 작성한다.

---

# 30. Claude Code에 처음 줄 Prompt

```text
이 프로젝트의 목표는 T1w MRI에서 tractogram을 생성하는 ATM을 기반으로,
생성된 tractogram의 geometry뿐 아니라 각 streamline의 endpoint ROI pair,
structural connectivity(SC) weight 및 tract-length matrix도 ground truth와
잘 일치하도록 fine-tuning하는 것이다.

먼저 코드를 수정하지 말고 다음을 수행해라.

1. docs/ATM_SC_Aware_Finetuning_Framework_v2.md를 읽어라.
2. external/atm_upstream/의 ATM 공식 코드를 전체적으로 조사해라.
3. T1 encoder, streamline encoder/decoder, VAE loss, inference, streamline export와
   관련된 파일/class/function을 찾아라.
4. ATM에서 N개의 latent vector가 어떻게 N개의 128-point streamline으로
   변환되는지 실제 source call flow를 정리해라.
5. T1 encoder가 inference 중 subject당 한 번 호출되는지 확인해라.
6. ATM 원본의 30-bundle 구조가 코드에서 어디에 반영되어 있는지 찾아라.
7. endpoint ROI supervision을 추가할 때 수정이 필요한 최소 파일 목록을 제안해라.
8. differentiable soft SC builder를 ATM과 연결할 위치를 제안해라.
9. SC_corr, SC magnitude, tract-length loss를 subject-level로 계산하기 위한
   training batch 구조를 제안해라.
10. external/atm_upstream은 수정하지 않고 wrapper/adapter 방식으로 확장하는
    설계를 제안해라.
11. 아직 구현하지 말고 분석 결과와 구현 계획만 Markdown으로 작성해라.

중요:
- 추측하지 말고 실제 source file/function/class 이름을 근거로 설명할 것.
- whole-brain SC 목적과 ATM 30-bundle limitation을 명확히 구분할 것.
- point-to-point recurrent tracking을 새로 추가하지 말 것.
- SC loss는 individual streamline이 아니라 subject-level tractogram set에서
  계산해야 함을 지킬 것.
```

---

# 31. Claude Code 구현 순서

1. ATM 공식 코드 구조 분석
2. 공식 pretrained inference 재현
3. 좌표계 / atlas alignment QC
4. ATM 30-bundle → GT SC coverage 분석
5. GT streamline endpoint ROI label 생성
6. Differentiable endpoint assigner 구현
7. Endpoint loss 단독 검증
8. GT TRK → differentiable soft SC builder 구현
9. Soft SC vs MRtrix/DSI Studio SC 검증
10. ATM + endpoint loss fine-tuning
11. SC correlation loss 추가
12. SC magnitude loss 추가
13. tract-length loss 추가
14. GPU streamline batching / AMP 최적화
15. Ablation + runtime benchmark

---

# 32. 구현에서 반드시 지켜야 할 원칙

1. 최초 목표는 공식 ATM 재현.
2. ATM 30-bundle coverage를 먼저 계산.
3. Endpoint supervision을 SC_corr보다 먼저 검증.
4. `tck2connectome`을 training graph에 직접 넣지 않음.
5. 평가는 실제 MRtrix/DSI Studio SC와 다시 비교.
6. SC_corr만 사용하지 않고 magnitude loss 병행.
7. SC loss는 subject-level로 계산.
8. TRK geometry를 희생하지 않도록 ATM 원래 loss 유지.
9. T1 encoder output은 subject별 재사용.
10. GPU 최적화는 correctness 확보 이후 적용.

---

# 33. 최종 성공 기준

## A. Speed

```text
ATM_SC-aware runtime << CoRNN runtime
```

특히 streamline generation 단계.

## B. Tractogram fidelity

```text
TRK_pred ≈ TRK_GT
```

## C. Connectome fidelity

```text
SC_weight_pred ≈ SC_weight_GT
SC_length_pred ≈ SC_length_GT
```

최종적으로 TVB에 사용할 수 있는 수준인지 검증한다.

---

# 34. 최종 추천 Loss

```text
L_total =
L_ATM
+ λ_endpoint * L_endpoint
+ λ_corr     * (1 - corr(SC_pred, SC_gt))
+ λ_mag      * L_SC_mag
+ λ_len      * L_length
```

핵심 해석:

```text
L_ATM
    = 어디로 지나갈 것인가?

L_endpoint
    = 어느 ROI와 어느 ROI를 연결할 것인가?

L_SC_corr
    = 전체 connectome pattern이 맞는가?

L_SC_mag
    = 실제 edge strength가 맞는가?

L_length
    = TVB에 필요한 tract-length가 맞는가?
```

이 다섯 항은 서로 역할이 다르기 때문에 단순히 SC_corr 하나만 추가하는 것보다 안정적인 multi-level supervision이 가능하다.

---

## Reference

Tan Y-F, Huynh KM, Liu S, et al.

**Anatomy-to-tract mapping infers white matter pathways without diffusion streamline propagation.**

Nature Communications. 2026.

Paper:
https://doi.org/10.1038/s41467-025-66615-w

Official source code and trained models:
https://zenodo.org/records/15792527
