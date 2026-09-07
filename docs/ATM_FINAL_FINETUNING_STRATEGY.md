# ATM 최종 Fine-tuning 전략
## T1 → TRK → SC 재현용 최종 권장안

## 0. 결론

최종 목표는 새로운 subject의 T1w MRI만 입력하여 다음을 생성하는 것이다.

```text
T1w MRI
  ↓
Trainable T1 Encoder
  ↓
Subject-specific anatomy feature
  ↓
ROI-pair conditioning + latent z
  ↓
ATM Streamline Decoder
  ↓
ROI-pair complete streamlines
  ↓
Whole-brain TRK/TCK
  ↓
SC weight + SC tract-length
```

핵심 원칙은 다음과 같다.

> ATM pretrained model을 그대로 고정해 쓰지 않는다.  
> ATM의 complete-streamline generation 구조와 pretrained streamline generator를 warm-start로 활용하고, 최종적으로 학습 가능한 모든 neural network module을 현재 데이터셋과 현재 목적함수에 맞게 fine-tuning한다.

---

## 1. 왜 pretrained ATM을 그대로 쓰면 안 되는가?

원 ATM과 현재 프로젝트는 학습 목적이 다르다.

### 원 ATM
```text
T1
→ predefined anatomical bundle
→ streamline/bundle reconstruction
```

### 현재 프로젝트
```text
T1
→ subject-specific whole-brain TRK
→ atlas-based SC weight
→ SC tract length
```

또한 MRI domain도 다를 수 있다.

- scanner / manufacturer
- acquisition protocol
- voxel resolution
- intensity distribution
- preprocessing
- registration/template
- population

따라서 원 ATM encoder가 현재 T1에서도 동일하게 유효한 subject-specific feature를 뽑는다고 가정하면 안 된다.

현재 구현에서도 pretrained anatomy feature가 매우 작고 T1=0 입력에서도 비슷한 크기의 feature가 관찰되어, pretrained encoder의 subject-specific conditioning이 충분한지 의심되는 상태다.

따라서 최종안에서는 T1 encoder를 frozen으로 두지 않는다.

---

## 2. 최종적으로 학습할 모듈

| Module | 최종 학습 여부 | 초기값 |
|---|---:|---|
| T1 Anatomy Encoder | O | ATM pretrained 또는 새 encoder |
| Streamline VAE Encoder | O | ATM pretrained |
| Streamline Decoder | O | ATM pretrained |
| ROI-pair Embedding | O | 신규 |
| Edge-existence Head | O | 신규 |
| Streamline Weight Head | O | 신규 |
| Conditioning/Projection Layers | O | 신규 |
| Soft Endpoint Assigner | X | deterministic |
| Differentiable SC Builder | X | deterministic |
| 128-point resampling | X | preprocessing |
| Atlas assignment / coordinate transform | X | preprocessing |

즉 **학습 가능한 neural network component는 최종적으로 모두 현재 데이터셋으로 update**한다.

---

## 3. Pretrained ATM은 무엇에 쓰나?

Pretrained ATM은 최종 정답이 아니라 **warm-start**다.

특히 streamline VAE/decoder는 이미:

```text
latent z + conditioning
→ complete streamline [128,3]
```

을 생성하는 능력을 학습했다.

따라서 random initialization보다 pretrained initialization에서 시작하는 것이 유리하다.

---

## 4. 데이터 전처리

각 subject:

```text
T1
GT whole-brain TRK/TCK
Atlas
GT SC weight
GT SC length (가능하면)
```

### 4.1 Whole-brain TRK → ROI-pair bundle

```text
Whole-brain GT TRK
↓
Endpoint ROI assignment
↓
ROI_i ↔ ROI_j bundle
```

SC matrix의 edge 하나를 하나의 ROI-pair streamline bundle로 본다.

```text
SC(i,j) ≈ ROI_i ↔ ROI_j streamline bundle
```

### 4.2 128-point resampling

각 streamline의 XYZ 좌표를 읽고:

```text
variable points
↓
equidistant resampling
↓
[128,3]
```

으로 변환한다.

---

## 5. 최종 모델 구조

```text
                         T1
                          ↓
                Trainable T1 Encoder
                          ↓
             Subject Anatomy Feature
                          │
             ┌────────────┴─────────────┐
             │                          │
      ROI-pair Embedding           Latent z
             │                          │
             └────────────┬─────────────┘
                          ↓
               ATM Streamline Decoder
                          ↓
                Streamline [128,3]
                          │
          ┌───────────────┼────────────────┐
          ↓               ↓                ↓
      Geometry       Endpoint ROI      Weight Head
          │               │                │
          └───────────────┼────────────────┘
                          ↓
                 Differentiable SC
                          ↓
                SC Weight / Length
```

---

## 6. 최종 Loss

```text
L_total =
L_recon
+ λ_KL       * L_KL
+ λ_geom     * L_geometry
+ λ_endpoint * L_endpoint
+ λ_edge     * L_edge
+ λ_corr     * L_SC_corr
+ λ_mag      * L_SC_mag
+ λ_len      * L_length
```

### 역할

| Loss | 목적 |
|---|---|
| L_recon | GT streamline 경로/좌표 재현 |
| L_KL | latent regularization |
| L_geometry | smoothness / adjacent-point consistency |
| L_endpoint | 올바른 ROI_i ↔ ROI_j 연결 |
| L_edge | edge 존재 여부 |
| L_SC_corr | whole-brain SC pattern |
| L_SC_mag | 실제 SC weight scale |
| L_length | tract-length matrix |

---

## 7. 최종 학습 순서

### Phase 0 — ATM pretrained reproduction
- 원본 inference 확인
- decoder 정상 여부
- runtime / GPU baseline

### Phase 1 — ROI-pair dataset 구축
```text
GT TRK
→ ROI-pair assignment
→ ROI-pair bundles
→ 128-point resampling
```

### Phase 2 — Streamline generator adaptation
먼저 VAE encoder / decoder를 현재 GT streamline distribution에 맞춘다.

```text
GT streamline
→ VAE Encoder
→ z
→ Decoder
→ Pred streamline
```

Loss:
```text
L_recon + L_KL + L_geometry
```

### Phase 3 — T1 Encoder 재학습
T1 encoder를 unfreeze한다.

```text
T1
→ Trainable Encoder
→ subject feature
+ ROI pair
+ latent
→ Decoder
→ streamline
```

이 단계부터 current MRI domain과 subject-specific tractography에 맞는 representation을 학습시킨다.

### Phase 4 — Endpoint supervision
```text
+ L_endpoint
```

### Phase 5 — Edge existence
```text
+ L_edge
```

### Phase 6 — SC pattern
```text
+ L_SC_corr
```

SC gradient가 decoder뿐 아니라 T1 encoder까지 흐르게 한다.

### Phase 7 — SC magnitude
```text
+ L_SC_mag
```

Streamline Weight Head도 함께 학습한다.

### Phase 8 — Tract length
```text
+ L_length
```

### Phase 9 — Final joint fine-tuning
최종적으로 다음을 모두 trainable:

```text
T1 Encoder
VAE Encoder
Streamline Decoder
ROI-pair Embedding
Edge Head
Weight Head
```

작은 LR에서 전체 objective로 joint fine-tuning한다.

---

## 8. Learning Rate 전략

Pretrained 부분과 신규 부분은 LR을 다르게 시작한다.

초기 예시:

```text
Pretrained T1 Encoder      : 1e-5
Pretrained VAE Encoder     : 1e-5 ~ 5e-5
Pretrained Decoder         : 1e-5 ~ 5e-5

ROI-pair Embedding         : 1e-4
Edge Head                  : 1e-4
Weight Head                : 1e-4
```

이는 최종값이 아니라 validation 기반 tuning 시작점이다.

---

## 9. T1 Encoder 최종 전략

반드시 다음을 ablation한다.

### A. ATM pretrained encoder frozen
기준선.

### B. ATM pretrained encoder fine-tuned
가장 중요한 후보.

### C. New/trainable T1 encoder
필요 시 3D CNN / 3D ResNet 등으로 교체.

최종 선택은 validation의:

- TRK quality
- SC corr / CCC
- subject generalization

으로 결정한다.

---

## 10. MRI Domain Shift 검증

원 ATM과 현재 데이터의 차이가 pretrained encoder 문제의 원인인지 확인한다.

검토:
- scanner
- manufacturer
- protocol
- voxel size
- intensity range
- template
- registration
- preprocessing

필요 시:
- intensity normalization
- scanner-balanced split
- augmentation
- harmonization 전략

을 검토한다.

단, 영상 자체 harmonization은 anatomical signal을 훼손하지 않는지 반드시 검증한다.

---

## 11. T1 Feature QC

Fine-tuning 전/후 각 subject의 feature:

```text
a_s ∈ R^D
```

를 추출하고 다음을 측정한다.

- feature norm
- pairwise cosine similarity
- pairwise Pearson correlation
- Euclidean distance
- dimension-wise variance
- PCA explained variance

핵심은 absolute norm보다 **subject 간 variance가 존재하는지**다.

나쁜 경우:

```text
a1 ≈ a2 ≈ a3 ≈ ...
```

좋은 방향:

```text
subject별 feature 차이가 존재
+
동일 subject에서는 안정적
```

또한 feature가 scanner/site만 구분하는지 확인한다.

---

## 12. Data Split

반드시 subject-level split.

금지:

```text
같은 subject의 streamline/ROI pair를
train과 test에 동시에 포함
```

Train / Validation / Test를 subject 단위로 나눈다.

scanner/site가 여러 개인 경우 split imbalance를 확인한다.

---

## 13. GPU 전략

ATM의 장점인 complete-streamline batch generation을 유지한다.

금지:

```text
p1 → p2 → ... → p500
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
Decoder
↓
[N,128,3]
```

우선순위:

1. T1 encoder subject당 1회
2. streamline batch 증가
3. mixed precision은 수치 검증 후
4. subject batch 증가
5. multi-GPU는 subject-level DDP

현재 실측에서 BF16/FP16 좌표 오차가 컸으므로 FP32를 reference로 유지하고 mixed precision은 별도 검증 후 사용한다.

---

## 14. 최종 Inference

학습 완료 후 입력은 T1만 필요하다.

```text
New T1
↓
Fine-tuned T1 Encoder
↓
Subject Feature
↓
ROI-pair candidates
↓
Edge Head
↓
Positive ROI pairs
↓
Latent batch
↓
Fine-tuned Streamline Decoder
↓
ROI-pair streamlines
↓
Whole-brain TRK/TCK
↓
SC Weight
SC Length
```

GT TRK / GT SC는 inference에 사용하지 않는다.

---

## 15. 필수 Evaluation

### TRK
- geometry
- endpoint distance
- valid streamline ratio
- length distribution
- ROI-pair/bundle overlap

### ROI pair
- start/end ROI accuracy
- unordered pair accuracy
- edge F1

### SC Weight
- Pearson
- Spearman
- CCC
- MAE
- RMSE

### SC Length
- Pearson
- CCC
- MAE
- RMSE

### Generalization
- unseen subject
- scanner/site robustness
- final test set 1회 평가

---

## 16. 필수 Ablation

최소:

```text
A. Frozen pretrained ATM encoder
B. Fine-tuned ATM encoder
C. New/trainable encoder
D. + Endpoint
E. + Edge
F. + SC corr
G. + SC magnitude
H. + Tract length
```

가장 중요한 비교:

```text
Frozen Encoder
vs
Fine-tuned Encoder
vs
New Encoder
```

이 비교가 MRI domain shift / pretrained encoder mismatch 가설을 직접 검증한다.

---

## 17. 이것이 “최종 전략”이라는 의미

**아키텍처와 학습 단계는 이 버전으로 고정하는 것을 권장한다.**

앞으로 남는 것은 구조를 계속 추가하는 것이 아니라:

- learning rate
- loss weight λ
- batch size
- unfreeze timing
- latent sampling 수
- edge threshold
- regularization

같은 **validation 기반 hyperparameter tuning**이다.

즉 새로운 핵심 module은 현재 구조가 명확한 실패 원인을 보일 때만 추가한다.

---

## 18. 최종 원칙

```text
ATM pretrained weights
        ↓
warm-start
        ↓
현재 ROI-pair tractography 데이터로 geometry adaptation
        ↓
T1 Encoder unfreeze / 재학습
        ↓
Endpoint
        ↓
Edge
        ↓
SC Corr
        ↓
SC Magnitude
        ↓
Tract Length
        ↓
전체 network Joint Fine-tuning
```

> **ATM의 pretrained 가중치는 시작점으로 사용한다.  
> 최종 모델의 학습 가능한 구성요소는 모두 현재 T1 + GT tractography + GT SC 데이터에 맞게 재학습한다.  
> 특히 T1 encoder는 MRI acquisition/domain 차이와 subject-specific SC 목적을 고려해 반드시 fine-tuning 또는 재학습한다.**

---

## 19. 현재 프로젝트에서 바로 다음 작업

1. frozen encoder baseline 보존
2. T1 encoder unfreeze 옵션 구현
3. parameter group별 learning rate 구현
4. pretrained vs fine-tuned feature QC
5. Phase 2 geometry adaptation
6. Endpoint
7. Edge
8. SC corr
9. SC magnitude
10. Length
11. Final joint fine-tuning
12. Frozen vs Fine-tuned vs New Encoder ablation
13. CoRNN 대비 speed / quality benchmark

---

## 20. 최종 연구 가설

> ATM의 complete-streamline generation 구조와 pretrained streamline decoder를 warm-start로 활용하되, 원 ATM과 현재 데이터셋 간 MRI acquisition/domain 및 학습 목적 차이를 고려하여 T1 encoder를 포함한 모든 학습 가능한 module을 현재 T1 + GT tractography + GT SC 데이터에 맞게 재학습하면, 빠른 T1-only tractography generation을 유지하면서 subject-specific TRK와 SC 재현도를 향상시킬 수 있다.
