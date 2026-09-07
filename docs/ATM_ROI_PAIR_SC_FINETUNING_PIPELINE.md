# ATM 기반 ROI-pair Bundle Fine-tuning Pipeline

## 0. 최종 목표

새로운 subject의 **T1w MRI만 입력**하여:

```text
T1w MRI
  ↓
Fine-tuned ATM
  ↓
ROI-pair specific streamlines
  ↓
Whole-brain tractogram
  ↓
.trk / .tck
  ↓
SC weight matrix
SC tract-length matrix
```

를 생성한다.

학습 시에는 다음 GT를 사용한다.

```text
T1 + GT whole-brain tractography + Atlas + GT SC weight + GT SC length(가능하면)
```

Inference 시에는 GT tractography와 GT SC가 필요하지 않다.

---

# 1. ATM 원본과 현재 목적의 차이

ATM 원본:

```text
T1
→ predefined anatomical bundle
→ complete streamline generation
```

현재 프로젝트:

```text
T1
→ whole-brain tractogram
→ atlas-based SC matrix
```

| 항목 | ATM 원본 | 현재 프로젝트 |
|---|---|---|
| 입력 | T1 | T1 |
| tract grouping | anatomical bundle | ROI-pair bundle |
| GT streamline | predefined WM bundles | whole-brain tractography |
| bundle 의미 | CST, ILF, UF 등 | ROI_i ↔ ROI_j |
| SC | 사후 평가 | 핵심 학습 target |
| SC weight | 직접 학습 안 함 | 직접 재현 |
| tract length | 핵심 target 아님 | TVB용으로 재현 |
| inference | T1 → bundle streamlines | T1 → whole-brain TRK → SC |

핵심 아이디어는 **ATM의 bundle 개념을 SC edge에 해당하는 ROI-pair bundle로 바꾸는 것**이다.

---

# 2. 핵심 아이디어: Whole-brain TRK → ROI-pair bundle

Whole-brain tractography를 atlas 기준으로 나눈다.

```text
Whole-brain TRK
   ↓
Atlas endpoint assignment
   ↓
ROI 1 ↔ ROI 2 bundle
ROI 1 ↔ ROI 3 bundle
ROI 1 ↔ ROI 4 bundle
...
ROI 12 ↔ ROI 37 bundle
```

즉:

```text
SC matrix의 edge 하나
≈
ROI-pair streamline bundle 하나
```

로 정의한다.

---

# 3. 데이터 구조

각 subject:

```text
subject_x/
├── T1w.nii.gz
├── whole_brain_gt.trk
├── atlas.nii.gz
├── sc_weight_gt.npy
└── sc_length_gt.npy
```

가능하면 GT SC를 생성했던 동일 atlas와 동일 endpoint assignment rule을 사용한다.

---

# 4. Step 1 — 좌표계 QC

먼저 T1, GT TRK, atlas가 동일한 공간인지 확인한다.

확인:
- voxel/world space
- RAS orientation
- affine
- voxel size
- MNI vs subject space

반드시 streamline을 T1/atlas에 overlay해 시각적으로 확인한다.

좌표계 오류는 endpoint ROI assignment와 SC를 모두 망가뜨릴 수 있다.

---

# 5. Step 2 — TRK를 ROI-pair bundle로 분해

각 streamline:

```text
S_k = [p1, p2, ..., pN]
```

의 시작/끝 point를 atlas ROI에 할당한다.

예:

```text
streamline 0001 → ROI 12 ↔ ROI 37
streamline 0002 → ROI 5  ↔ ROI 21
streamline 0003 → ROI 12 ↔ ROI 37
```

MRtrix를 사용할 경우 예:

```bash
tck2connectome     whole_brain_gt.tck     atlas.mif     sc.csv     -out_assignments assignments.txt
```

`assignments.txt`의 ROI pair assignment를 사용해 edge별 bundle을 만든다.

중요:
**GT SC를 계산할 때 사용했던 node assignment 설정과 동일해야 한다.**

---

# 6. ROI-pair bundle 저장

권장:

```text
subject_x/
├── T1w.nii.gz
├── atlas.nii.gz
├── whole_brain_gt.trk
├── assignments.npy
├── bundles/
│   ├── roi_001_002.npz
│   ├── roi_001_004.npz
│   ├── roi_002_008.npz
│   └── roi_012_037.npz
├── sc_weight_gt.npy
└── sc_length_gt.npy
```

학습에서는 수천 개의 작은 TRK 파일보다 `.npz`, `.pt`, `.h5` 같은 묶음 형식이 효율적이다.

---

# 7. Step 3 — TRK streamline 좌표 추출

TRK/TCK 내부의 각 streamline은 이미 XYZ point sequence를 가진다.

```text
P1 = (x1,y1,z1)
P2 = (x2,y2,z2)
...
PN = (xN,yN,zN)
```

이 좌표가 ATM streamline reconstruction의 GT가 된다.

---

# 8. Step 4 — 128-point resampling

ATM 원본과 동일한 complete-streamline representation을 유지한다.

```text
variable point streamline
↓
equidistant resampling
↓
128 points
```

결과:

```text
one streamline = [128,3]
bundle = [N_edge_streamlines,128,3]
```

---

# 9. Step 5 — ROI-pair conditioning

ATM 원본의 anatomical bundle 조건을 ROI pair 조건으로 바꾼다.

원본 개념:

```text
T1 anatomy feature
+ bundle condition
+ latent z
→ streamline
```

변경:

```text
T1 anatomy feature
+ ROI_start embedding
+ ROI_end embedding
+ streamline latent z
→ complete streamline [128,3]
```

수식 개념:

```text
S_pred =
D(
  z,
  anatomy_feature,
  Emb(ROI_i),
  Emb(ROI_j)
)
```

Undirected SC라면:

```text
(i,j) == (j,i)
```

이므로 canonical ordering:

```text
roi_a = min(i,j)
roi_b = max(i,j)
```

을 사용한다.

---

# 10. Step 6 — Positive / Negative ROI pair

Positive:

```text
SC_GT(i,j) > threshold
```

또는 GT TRK에 실제 streamline이 존재하는 pair.

Negative:

```text
SC_GT(i,j) = 0
```

Positive edge만 학습하면 over-connectivity가 발생할 수 있다.

필요하면 edge-existence head:

```text
T1 feature + ROI pair
→ MLP
→ P(edge exists)
```

를 추가한다.

Loss:

```text
L_edge = BCE(pred_edge, GT_edge)
```

---

# 11. Step 7 — ROI-pair ATM baseline fine-tuning

처음부터 SC loss를 넣지 않는다.

```text
T1
+ ROI pair
+ latent z
↓
ATM decoder
↓
Pred streamline
```

먼저 ATM 원본 geometry/reconstruction objective만 사용한다.

목표:

```text
ROI_i ↔ ROI_j 조건에서
해당 edge의 streamline geometry를 생성 가능한가?
```

를 확인하는 것이다.

---

# 12. Step 8 — Endpoint Connectivity Loss

Pred streamline:

```text
p_start
p_end
```

가 GT ROI pair를 연결하도록 학습한다.

soft ROI probability:

```text
q_start
q_end
```

GT pair = `(ROI_i, ROI_j)`일 때:

```text
L_direct =
CE(q_start, ROI_i) + CE(q_end, ROI_j)

L_reverse =
CE(q_start, ROI_j) + CE(q_end, ROI_i)
```

Undirected tractography에서는 direct/reverse orientation을 동일하게 취급한다.

목적:
**경로가 비슷한 것뿐 아니라 실제 올바른 ROI pair를 연결하도록 하는 것.**

---

# 13. Step 9 — Differentiable Endpoint Assigner

hard atlas lookup은 gradient가 끊기므로 training에서는 soft assignment를 사용한다.

추천:

```text
ROI distance maps
↓
endpoint 위치에서 grid_sample
↓
distance_i
↓
softmax(-distance_i / temperature)
↓
ROI probability
```

예:

```text
ROI12 = 0.91
ROI13 = 0.06
ROI14 = 0.03
```

---

# 14. Step 10 — Differentiable SC Builder

streamline k의 endpoint probability:

```text
q_start(k)
q_end(k)
```

Undirected contribution:

```text
SC_k =
0.5 * (
 q_start ⊗ q_end
 +
 q_end ⊗ q_start
)
```

전체 subject SC:

```text
SC_pred = Σ_k SC_k
```

SC loss는 **개별 streamline이 아니라 subject-level 전체 streamline set**에서 계산한다.

---

# 15. 중요한 문제 — ROI-pair bundle 수와 SC weight

ATM 원본은 bundle당 일정 수의 streamline을 생성할 수 있다.

하지만 실제 SC에서는:

```text
ROI12 ↔ ROI37 = strong
ROI12 ↔ ROI40 = weak
```

처럼 edge strength가 다르다.

모든 ROI pair에서 같은 수의 streamline을 생성하면 absolute SC weight를 재현하기 어렵다.

---

# 16. 권장 해결 — Streamline Weight Head

각 generated streamline k에:

```text
w_k >= 0
```

를 예측한다.

```text
anatomy feature
+ ROI pair embedding
+ latent z
↓
MLP
↓
softplus
↓
w_k
```

SC contribution:

```text
SC_pred(i,j)
=
Σ_k w_k * P_i(start_k) * P_j(end_k)
```

이렇게 하면 streamline 생성 개수는 GPU 친화적으로 고정하면서도 subject-specific SC strength를 표현할 수 있다.

---

# 17. Step 11 — SC Correlation Loss

```text
L_SC_corr =
1 - corr(SC_pred, SC_gt)
```

upper triangle 또는 valid edge를 사용한다.

역할:
**전체 SC edge pattern**을 맞춘다.

---

# 18. Step 12 — SC Magnitude Loss

SC_corr만 사용하면 scale mismatch가 생길 수 있다.

예:

```text
GT   = [1,2,5,10]
Pred = [10,20,50,100]
```

도 Pearson r=1이 가능하다.

추천:

```text
L_SC_mag =
MAE(
 log(SC_pred + eps),
 log(SC_gt + eps)
)
```

역할:

```text
SC_corr → relative pattern
SC_mag  → actual strength
```

---

# 19. Step 13 — Tract-Length Matrix Loss

generated streamline:

```text
[p1,...,p128]
```

의 길이:

```text
length_k =
Σ_t ||p_(t+1)-p_t||
```

ROI pair별 weighted mean:

```text
Length_pred(i,j)
=
Σ_k edge_weight_k(i,j) * length_k
----------------------------------
Σ_k edge_weight_k(i,j) + eps
```

Loss:

```text
L_length =
MAE(
 log(Length_pred + eps),
 log(Length_gt + eps)
)
```

---

# 20. 최종 Loss

권장:

```text
L_total =
L_ATM
+ λ_endpoint * L_endpoint
+ λ_edge     * L_edge
+ λ_corr     * L_SC_corr
+ λ_mag      * L_SC_mag
+ λ_len      * L_length
```

edge-existence head를 사용하지 않으면 `L_edge` 제외.

| Loss | 목적 |
|---|---|
| L_ATM | streamline 경로/모양 |
| L_endpoint | 올바른 ROI pair 연결 |
| L_edge | edge 존재 여부 |
| L_SC_corr | 전체 SC pattern |
| L_SC_mag | 실제 edge strength |
| L_length | tract-length matrix |

---

# 21. Fine-tuning 순서

## Phase 0 — ATM 공식 재현
- pretrained load
- official inference
- streamline tensor shape
- runtime / VRAM 확인

## Phase 1 — Dataset preprocessing
```text
GT TRK
→ endpoint ROI assignment
→ ROI-pair bundles
→ 128-point resampling
```

## Phase 2 — ROI-pair ATM baseline
```text
T1 + ROI pair
→ Pred streamline
```
Loss: `L_ATM`

## Phase 3 — Endpoint supervision
추가: `L_endpoint`

평가:
- start ROI accuracy
- end ROI accuracy
- unordered pair accuracy

## Phase 4 — Edge existence
필요 시 `L_edge` 추가.

## Phase 5 — Differentiable SC Builder 검증
```text
GT TRK → Soft SC Builder → SC_soft
GT TRK → MRtrix/DSI Studio → SC_reference
```

비교:
- Pearson
- CCC
- MAE
- RMSE

## Phase 6 — SC correlation fine-tuning
추가: `L_SC_corr`

## Phase 7 — SC magnitude fine-tuning
추가: `L_SC_mag`

필요하면 streamline weight head 활성화.

## Phase 8 — tract-length fine-tuning
추가: `L_length`

## Phase 9 — Joint fine-tuning
작은 learning rate에서 전체 objective를 함께 최적화.

---

# 22. Inference Pipeline

학습이 끝난 뒤:

```text
New T1
↓
Anatomy Encoder
↓
ROI pair candidates
↓
Edge-existence prediction (사용 시)
↓
Positive ROI pairs
↓
각 ROI pair용 latent batch
↓
ATM decoder
↓
ROI-pair streamlines
↓
모든 bundle concatenate
↓
Whole-brain TRK
↓
.trk / .tck
↓
Atlas
↓
SC weight
SC length
```

Inference에는 GT TRK / GT SC가 필요하지 않다.

---

# 23. GPU 활용 전략

ATM의 장점은 complete-streamline generation을 batch 처리할 수 있다는 점이다.

CoRNN:

```text
step1
↓
step2
↓
...
↓
step500
```

ATM:

```text
large latent batch
↓
decoder
↓
[N,128,3]
```

## 우선순위

1. **T1 encoder는 subject당 1회**
2. **streamline batch 최대화**
3. **BF16 / AMP**
4. **그 다음 subject batch 증가**

잘못된 구조:

```text
streamline1 → T1 encoder → decoder
streamline2 → T1 encoder → decoder
```

권장:

```text
T1
↓
Encoder 1회
↓
feature cache
↓
large latent batch
↓
decoder
```

---

# 24. GPU Chunking

VRAM이 부족하면 streamline을 chunk로 decode한다.

예:

```text
2048 streamlines
→ 512 + 512 + 512 + 512
```

단:

```text
partial SC 1
+ partial SC 2
+ partial SC 3
+ partial SC 4
↓
full subject SC
↓
SC loss
```

로 해야 한다.

SC loss를 chunk마다 따로 계산하지 않는다.

---

# 25. Precision

추천:

```text
ATM encoder / decoder → BF16/AMP
SC correlation / log / normalization → FP32
```

A100/H100에서는 BF16 우선 검토.

---

# 26. Multi-GPU

권장:

```text
GPU0 → Subject A
GPU1 → Subject B
GPU2 → Subject C
GPU3 → Subject D
```

즉 subject-level DDP를 우선한다.

---

# 27. Smoke Test

Full training 전에 반드시 실행.

Synthetic 예:

```text
B = 1
ROI = 6
positive ROI pairs = 3
N = 8 streamlines / pair
P = 128
```

검증:

- ROI-pair embedding
- decoder output
- endpoint assignment
- SC builder
- SC corr loss
- SC magnitude loss
- backward
- gradient flow

필수 PASS:

```text
Generated streamline shape PASS
Endpoint probability PASS
SC shape PASS
SC symmetry PASS
Finite loss PASS
Backward PASS
Streamline gradient PASS
Model parameter gradient PASS
```

---

# 28. Real-data Smoke Test

subject 1명 + 작은 ROI-pair subset(5~10 edges)만 사용.

```text
T1
+ GT TRK
+ Atlas
+ GT SC
↓
ROI-pair ATM
↓
Pred streamlines
↓
Soft SC
↓
Loss
↓
Backward
```

까지 동작해야 한다.

---

# 29. Evaluation

## TRK / streamline
- coordinate error
- endpoint distance
- streamline length
- valid streamline ratio
- bundle overlap

## ROI pair
- start ROI accuracy
- end ROI accuracy
- unordered pair accuracy

## SC weight
- Pearson r
- Spearman r
- CCC
- MAE
- RMSE
- edge F1
- degree correlation
- graph density

## SC length
- Pearson r
- CCC
- MAE
- RMSE

---

# 30. Speed Benchmark

동일 subject / 동일 target streamline 수에서:

```text
CoRNN
ATM original
ROI-pair ATM
ROI-pair ATM + SC-aware fine-tuning
```

비교한다.

측정:

```text
T1 encoding sec
streamline generation sec
TRK write sec
SC computation sec
total sec
streamlines/sec
peak VRAM
GPU utilization
```

---

# 31. Ablation

| Model | ROI Pair | Endpoint | Edge | SC Corr | SC Mag | Length |
|---|---:|---:|---:|---:|---:|---:|
| ATM baseline | X | X | X | X | X | X |
| ROI-pair ATM | O | X | X | X | X | X |
| + Endpoint | O | O | X | X | X | X |
| + Edge | O | O | O | X | X | X |
| + SC Corr | O | O | O | O | X | X |
| + SC Mag | O | O | O | O | O | X |
| Full | O | O | O | O | O | O |

---

# 32. 권장 Repository 구조

```text
t1_roi_atm_sc/
├── README.md
├── CLAUDE.md
├── docs/
│   └── ATM_ROI_PAIR_SC_FINETUNING_PIPELINE.md
├── external/
│   └── atm_upstream/
├── src/
│   └── atm_sc/
│       ├── models/
│       │   ├── atm_adapter.py
│       │   ├── roi_pair_embedding.py
│       │   ├── endpoint_assigner.py
│       │   ├── edge_head.py
│       │   ├── streamline_weight_head.py
│       │   └── sc_builder.py
│       ├── losses/
│       │   ├── endpoint.py
│       │   ├── edge.py
│       │   ├── sc_corr.py
│       │   ├── sc_magnitude.py
│       │   └── tract_length.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── trk_to_roi_pairs.py
│       │   ├── resample_streamlines.py
│       │   └── build_distance_maps.py
│       ├── training/
│       ├── inference/
│       └── evaluation/
├── scripts/
│   ├── 00_reproduce_atm.py
│   ├── 01_qc_coordinate_space.py
│   ├── 02_assign_roi_pairs.py
│   ├── 03_build_roi_pair_bundles.py
│   ├── 04_resample_to_128.py
│   ├── 05_build_distance_maps.py
│   ├── 06_smoke_test.py
│   ├── 07_train_roi_atm.py
│   ├── 08_train_endpoint.py
│   ├── 09_train_sc.py
│   ├── 10_train_full.py
│   └── 11_benchmark.py
├── configs/
└── outputs/
```

---

# 33. Claude Code 구현 순서

```text
1. ATM upstream 분석
2. 공식 ATM inference 재현
3. TRK / atlas coordinate QC
4. TRK → ROI-pair assignment
5. ROI-pair bundle dataset 생성
6. 128-point resampling
7. ROI-pair embedding
8. ATM wrapper에 ROI-pair conditioning 연결
9. synthetic smoke test
10. real subject smoke test
11. endpoint loss
12. differentiable SC builder
13. SC_corr loss
14. SC magnitude loss
15. streamline weight head
16. tract-length loss
17. GPU batching
18. ablation
19. speed benchmark
```

---

# 34. 구현 원칙

- ATM upstream은 가능한 직접 수정하지 않는다.
- recurrent point-to-point tracking을 다시 추가하지 않는다.
- T1 encoder는 subject당 1회 실행한다.
- ROI-pair bundle은 기존 GT SC와 동일한 atlas assignment 기준을 사용한다.
- training SC builder는 differentiable PyTorch 구현을 사용한다.
- evaluation에서는 MRtrix/DSI Studio SC와 다시 비교한다.
- SC loss는 subject-level에서 계산한다.
- SC_corr만 단독으로 최적화하지 않는다.
- positive/negative edge imbalance를 확인한다.
- full training 전에 synthetic + real-data smoke test를 통과한다.

---

# 35. 최종 개념

ATM 원본:

```text
T1
→ anatomical bundle
→ streamline
```

현재 모델:

```text
T1
+
ROI_i ↔ ROI_j
→ edge-specific streamline distribution
→ ROI-pair bundles
→ whole-brain TRK
→ SC
```

즉:

> ATM의 “해부학적 bundle을 생성하는 모델”을  
> “atlas의 SC edge별 streamline bundle을 생성하는 모델”로 확장하고,  
> geometry + endpoint + edge existence + SC pattern + SC magnitude + tract length를 단계적으로 학습한다.

ATM의 핵심 장점인 **complete-streamline batch generation과 GPU 병렬화는 유지**하면서,
학습 목적을 **T1 → TRK → SC 재현**에 직접 맞춘다.

---

## Reference

Tan Y-F, Huynh KM, Liu S, et al.  
**Anatomy-to-tract mapping infers white matter pathways without diffusion streamline propagation.**  
Nature Communications, 2026.

Paper: https://doi.org/10.1038/s41467-025-66615-w  
Official code + trained models: https://zenodo.org/records/15792527
