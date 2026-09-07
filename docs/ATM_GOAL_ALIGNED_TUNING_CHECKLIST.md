# ATM Goal-Aligned Tuning Checklist
## T1-only → Whole-brain TRK → SC Weight / Tract-Length

> 목적: 원 ATM의 각 stage를 현재 목표인 **subject-specific T1 → ROI-pair streamlines → whole-brain TRK → SC weight/length**에 맞게 조정하고, 실제 수정 코드와 한 항목씩 대조하기 위한 체크리스트.
>
> 현재 상태 평가는 업로드된 `CODE_CHANGES_FROM_UPSTREAM.md`와 `ATM_ROI_PAIR_SMOKE_TEST_REPORT.md` 기준이며, 실제 repository에서는 다시 직접 검증해야 한다.

---

## 0. 최종 목표

```text
New subject T1
    ↓
Trainable T1 Encoder
    ↓
Subject-specific anatomy feature
    ↓
ROI-pair conditioning + latent z
    ↓
Trainable ATM streamline generator
    ↓
ROI-pair complete streamlines
    ↓
Whole-brain TRK/TCK
    ↓
SC Weight
SC Tract-Length
```

### 핵심 원칙

- ATM pretrained weights는 **warm-start**로만 사용한다.
- 최종 출력에 영향을 주는 trainable neural-network module은 현재 데이터셋으로 fine-tuning한다.
- deterministic preprocessing, atlas assignment, SC 계산식은 학습 대상이 아니다.
- 원 ATM의 30 anatomical bundle을 최종 bundle 정의로 사용하지 않는다.
- 최종 SC loss의 정의는 실제 GT SC 생성 방식과 동일해야 한다.
- 아래 core checklist가 통과하기 전 새로운 핵심 module을 추가하지 않는다.

### Status

```text
✅ DONE
⚠️ PARTIAL
❌ TODO
🧪 VERIFY
```

---

# Stage 1. MRI / Coordinate-space 정합

## 원 ATM
- T1 → MNI152 registration
- intensity normalization
- streamline도 MNI space

## 현재 목표
GT tractogram, atlas, SC가 정의된 공간을 기준으로 T1을 맞춘다.

현재 프로젝트 보고서 기준:
```text
GT tractogram = QSDR/NLin6 계열 template space
```

따라서:
```text
T1
→ SyN registration
→ NLin6/GT tractogram space
→ ATM working grid
```

## Checklist
- [ ] T1, GT tractogram, atlas가 동일한 world space
- [ ] 좌우 반전 없음
- [ ] 5명 이상 tract overlay 육안 QC
- [ ] affine/orientation/voxel size 기록
- [ ] registration transform 저장
- [ ] train/val/test 동일 preprocessing

## 코드 확인
```text
src/atm_sc/data/prepare_t1.py
src/atm_sc/spaces.py
scripts/01_qc_coordinate_space.py
```

## 현재 상태
- [x] SyN → NLin6
- [x] GT tractogram→atlas→SC 재현 r≈0.9986
- [x] 좌표 QC 수행
- [ ] 여러 subject 반복 QC

## PASS
```text
reference SC reproduction Pearson > 0.99
edge F1 > 0.95
```

---

# Stage 2. MRI Intensity / Scanner Domain Adaptation

## 원 ATM
TractoInferno의 MRI domain에 맞춰 T1 normalization이 학습되었다.

## 현재 문제
현재 데이터는 원 ATM과 다음이 다를 수 있다.
- scanner/vendor
- acquisition protocol
- voxel size
- intensity distribution
- preprocessing
- registration/template
- population

이 차이는 pretrained T1 encoder feature 저하 원인이 될 수 있다.

## Core tuning
- pretrained ATM normalization은 baseline으로 보존
- 현재 training data의 intensity distribution 분석
- training subjects만 이용해 robust normalization 후보 비교
- validation/test statistics를 normalization에 사용하지 않음

예:
```text
brain mask
→ robust percentile clipping
→ subject-wise scaling 또는 z-normalization
```

## Checklist
- [ ] 원 ATM normalization baseline
- [ ] current dataset intensity histogram
- [ ] scanner/vendor별 histogram
- [ ] normalization 전/후 비교
- [ ] train-only statistics
- [ ] T1=0 vs real-T1 feature 비교
- [ ] scanner/site feature confound 확인

## 현재 상태
- [x] upstream BundleNorm 사용
- [ ] current-dataset normalization 비교
- [ ] scanner-domain QC
- [ ] train-only normalization statistics

### 중요
현재 `init_bundle=AF_L`의 bundle-specific normalization constant가 whole-brain ROI-pair model에 최적인지 검증 필요.

---

# Stage 3. Whole-brain TRK → ROI-pair Bundle

## 원 ATM
```text
30 anatomical bundles
AF_L, CST_L, ILF_R, ...
```

## 현재 목표
```text
SC edge ≈ ROI_i ↔ ROI_j streamline bundle
```

각 streamline:
```text
start endpoint → ROI_i
end endpoint   → ROI_j
```

Undirected:
```text
(i,j) == (j,i)
```

## Checklist
- [ ] endpoint ROI assignment
- [ ] background endpoint 처리
- [ ] same-ROI streamline 처리
- [ ] canonical pair `(min,max)`
- [ ] positive pair 저장
- [ ] negative pair 생성
- [ ] GT streamline↔pair mapping 보존

## 코드
```text
src/atm_sc/data/trk_to_roi_pairs.py
scripts/02_assign_roi_pairs.py
scripts/03_build_roi_pair_bundles.py
```

## 현재 상태
- [x] endpoint assignment
- [x] canonical ROI pair
- [x] bundles.npz
- [x] endpoint→ROI 일치 100%

---

# Stage 4. Streamline 128-point Representation

## 원 ATM
각 streamline을 128 equidistant points로 resampling.

## 우리 목표
그대로 유지.

이유:
- ATM pretrained VAE와 호환
- complete-streamline generation 유지
- recurrent tracking 불필요

## Checklist
- [ ] arc-length 기반 128-point resampling
- [ ] endpoint 보존
- [ ] length error 최소
- [ ] NaN/Inf 없음
- [ ] [N,128,3] shape

## 현재 상태
- [x] vectorized 128-point resampling
- [x] endpoint error 0
- [x] length error p99≈0.28%

## PASS
```text
endpoint error = 0
length error p99 < 1%
```

---

# Stage 5. T1 Encoder EA — 가장 중요한 수정

## 원 ATM
```text
T1 → EA → anatomy feature a
```

EA는 원래 bundle segmentation decoder와 함께 특정 30 bundle의 occupancy를 설명하도록 학습됨.

## 우리 목표
```text
T1 → EA → subject-specific whole-brain connectivity feature
```

즉 pretrained encoder를 최종적으로 frozen 상태로 사용하면 안 된다.

## Core tuning
1. pretrained weight는 initialization으로 사용 가능
2. EA unfreeze
3. downstream losses의 gradient가 EA까지 도달
4. final joint fine-tuning에서 EA trainable

## Checklist — 코드
- [ ] `requires_grad=True` option
- [ ] optimizer param group에 EA 포함
- [ ] encoder LR 별도 설정
- [ ] frozen baseline option 유지
- [ ] full fine-tuning option

## Checklist — gradient
- [ ] `L_recon → EA`
- [ ] `L_endpoint → EA`
- [ ] `L_edge → EA`
- [ ] `L_SC_corr → EA`
- [ ] `L_SC_mag → EA`
- [ ] `L_length → EA`
- [ ] optimizer step 후 EA parameter 변경

## Checklist — feature QC
- [ ] 5~10 subjects feature 추출
- [ ] norm
- [ ] cosine similarity
- [ ] Pearson correlation
- [ ] Euclidean distance
- [ ] dimension-wise variance
- [ ] PCA
- [ ] scanner/site association

## 현재 상태
- [ ] T1 encoder trainable
- [ ] SC loss→EA gradient
- [ ] EA optimizer update
- [x] anatomy_gain 추가
- [x] pretrained feature가 매우 작다는 문제 확인

### 최우선 수정
```text
현재: EA ❄ frozen
최종: EA 🔥 trainable
```

---

# Stage 6. Segmentation Decoder DA

## 원 ATM
```text
EA
→ DA
→ bundle binary occupancy map
→ Dice loss L_SEG
```

이 branch는 EA를 tract-aware하게 학습시키는 auxiliary task.

## 우리 목표에서의 tuning
원 30-bundle segmentation objective는 ROI-pair whole-brain 목적과 직접 맞지 않는다.

### Core final baseline
```text
원 30-bundle DA objective 제거
```

EA는 downstream:
- reconstruction
- endpoint
- edge
- SC
- length

loss로 직접 학습.

### Optional — encoder collapse가 지속될 때만
- WM segmentation
- atlas-region segmentation
- ROI-pair occupancy auxiliary head

를 고려.

## Checklist
- [ ] 원 30-bundle segmentation loss가 final objective에 없음
- [ ] auxiliary head 기본 OFF
- [ ] collapse 발생 시에만 추가

## 현재 상태
- [x] 원 segmentation decoder branch 사용 안 함

---

# Stage 7. Streamline Encoder ES / VAE

## 원 ATM
```text
GT streamline + anatomy
→ ES
→ μ, σ
→ z
```

## 우리 목표
pretrained ES를 warm-start로 사용하되 current ROI-pair streamline distribution에 맞게 fine-tune.

## Checklist
- [ ] ES pretrained load
- [ ] ES trainable
- [ ] ROI-pair/anatomy condition 적용
- [ ] reparameterization 정상
- [ ] KL finite
- [ ] μ/σ distribution logging
- [ ] posterior collapse monitor

## 현재 상태
- [x] ConvVAE encoder 사용
- [x] trainable
- [x] reconstruction path 존재

---

# Stage 8. Latent Sampling / KDE

## 원 ATM
training streamline latent로 bundle-specific KDE를 만들고 inference에서 KDE sampling.

## 우리 목표의 문제
ROI-pair가 수백~수천 개라 pair별 KDE가 안정적이지 않을 수 있다.

## Core baseline
```text
z ~ N(0,I)
```

그리고 KL로 latent를 regularize.

## Checklist
- [ ] training posterior reparameterization
- [ ] inference `N(0,I)`
- [ ] KL target `N(0,I)`
- [ ] seed 고정
- [ ] latent diversity metric

## 현재 상태
- [x] ROI-pair model = N(0,I)
- [x] original KDE path는 reproduction용 보존

### Optional only if failure
- conditional prior
- learned pair-specific prior

---

# Stage 9. ROI-pair Conditioning / FiLM

## 원 ATM
anatomy feature가 FiLM을 통해 streamline encoder/decoder를 조절.

## 우리 모델
```text
cond =
gain * anatomy_feature
+
Proj(Emb(ROI_a)+Emb(ROI_b))
```

## Checklist
- [ ] pair order invariant
- [ ] ROI embedding trainable
- [ ] projection trainable
- [ ] anatomy branch trainable
- [ ] tensor shape 일치
- [ ] `||anatomy_condition||` logging
- [ ] `||roi_pair_condition||` logging
- [ ] 한 branch가 압도하지 않는지 확인

## 현재 상태
- [x] order-invariant embedding
- [x] projection 0-init
- [x] pretrained start 보존
- [ ] fine-tuned EA feature와 재검증

---

# Stage 10. Streamline Decoder DS

## 원 ATM
```text
z + anatomy
→ DS
→ complete streamline [128,3]
```

## 우리 목표
ATM decoder의 complete-streamline generation 능력은 유지하되 current ROI-pair GT tract geometry에 맞게 fine-tune.

## Checklist
- [ ] pretrained decoder load
- [ ] decoder trainable
- [ ] output `[N,128,3]`
- [ ] training graph torch-only
- [ ] whole-brain coordinate normalization 일관
- [ ] recon loss 감소

## 현재 상태
- [x] decoder trainable
- [x] torch-only inverse normalization
- [x] batch complete-streamline generation
- [ ] whole-brain bbox adaptation 충분히 수렴했는지 확인

### 중요
bundle-specific bbox → whole-brain bbox 변경으로 초기 geometry error가 큼.
따라서 SC loss 전에 geometry adaptation이 먼저 필요.

---

# Stage 11. Geometry Adaptation Loss

## 원 ATM
```text
L_VAE = MSE reconstruction + β KL
L_ADJ = adjacent-point regularization
```

## 우리 모델
```text
L_recon + λ_KL L_KL + λ_geom L_geometry
```

## 현재 문제
보고서상:
```text
L_recon gradient ≈ 10^4
다른 loss gradient ≈ 10^0~10^1
```

따라서 원 ATM처럼 모든 weight=1을 그대로 사용할 수 없다.

## Checklist
- [ ] raw loss logging
- [ ] weighted loss logging
- [ ] loss별 grad norm
- [ ] total grad norm
- [ ] gradient clipping
- [ ] NaN/Inf detector
- [ ] λ config화

## 현재 상태
- [x] clipping 사용
- [x] gradient imbalance 확인
- [ ] 최종 λ tuning

---

# Stage 12. Endpoint Supervision

## 원 ATM
explicit ROI-pair endpoint loss 없음.

## 우리 모델
생성 streamline의 양 끝이 conditioning된 ROI pair를 연결하게 한다.

## Checklist
- [ ] differentiable soft endpoint assigner
- [ ] direct/reverse orientation invariant
- [ ] background class
- [ ] probability sum=1
- [ ] endpoint accuracy
- [ ] unordered pair accuracy

## 현재 상태
- [x] 구현
- [x] synthetic PASS
- [x] real smoke PASS

---

# Stage 13. Edge-existence Head

## 목적
모든 ROI pair를 생성하는 over-connectivity를 방지.

```text
T1 anatomy + ROI pair
→ Edge Head
→ P(edge exists)
```

## 중요한 정의
Generation candidate는 **endpoint-defined pair existence**를 target으로 사용.

Final SC target이 pass-based라면 별도로 유지.

## Checklist
- [ ] positive endpoint pairs
- [ ] negative pair sampling
- [ ] class imbalance
- [ ] BCE/pos_weight
- [ ] ROC-AUC
- [ ] PR-AUC
- [ ] edge F1
- [ ] threshold는 validation에서 결정

## 현재 상태
- [x] EdgeHead
- [x] BCE
- [x] negative pairs

---

# Stage 14. Streamline Weight Head

## 목적
고정 streamline 수로도 subject-specific edge strength를 표현.

```text
streamline k → w_k >= 0
```

## Checklist
- [ ] softplus/non-negative
- [ ] init≈1
- [ ] gradient finite
- [ ] SC builder에 반영
- [ ] weight distribution logging
- [ ] exploding weight monitor

## 현재 상태
- [x] 구현
- [x] softplus
- [x] init≈1
- [x] SC builder 연동

---

# Stage 15. SC Definition 일치 — 두 번째 최우선

## 현재 프로젝트 핵심 문제
보고서:
```text
GT .mat SC = pass definition
sc_pass vs GT ≈ r 0.9986
sc_end  vs GT ≈ r 0.673
```

따라서 final SC loss는 반드시 GT와 동일한 `pass` 정의를 사용해야 한다.

## Final rule

### ROI-pair bundle generation
```text
endpoint pair
```

### Edge Head target
```text
endpoint existence
```

### Final SC loss/evaluation
```text
PASS-SC
```

## Checklist
- [ ] `sc_end` / `sc_pass` 별도 저장
- [ ] dataset mode 구분
- [ ] final SC loss default=`pass`
- [ ] final hard evaluation도 pass
- [ ] endpoint pair target과 pass-SC target 혼동 금지

## 현재 상태
- [x] sc_end/sc_pass 분리
- [x] pass mode 지원
- [ ] final training config 기본값이 pass인지 확인
- [ ] full training command가 pass인지 확인

## PASS
```text
GT tractogram → soft pass-SC
vs reference GT SC
Pearson > 0.99
```

---

# Stage 16. Differentiable SC Builder

## 원 ATM
SC는 사후 evaluation.

## 우리 모델
SC를 training supervision으로 사용.

## Checklist
- [ ] fully differentiable
- [ ] endpoint/pass mode
- [ ] subject-level aggregation
- [ ] chunk별 SC loss 금지
- [ ] partial SC 합산 후 loss 1회
- [ ] symmetric SC
- [ ] diagonal policy
- [ ] FP32 SC math

## 현재 상태
- [x] soft SC
- [x] ChunkedSC
- [x] BundleAccumulator
- [x] 2-pass exact gradient
- [x] single graph와 gradient 일치

---

# Stage 17. SC Correlation Loss

```text
L_SC_corr = 1 - Pearson(SC_pred, SC_gt)
```

## Checklist
- [ ] upper triangle
- [ ] diagonal 제외
- [ ] zero-variance guard
- [ ] FP32
- [ ] pass-SC target
- [ ] train/val 동일 vectorization

## 현재 상태
- [x] 구현
- [ ] final pass target 확인

---

# Stage 18. SC Magnitude Loss

Correlation만으로 실제 edge strength는 맞지 않는다.

```text
L_SC_mag =
MAE(log(SC_pred+eps), log(SC_gt+eps))
```

## Checklist
- [ ] eps
- [ ] FP32
- [ ] positive/all-edge policy 명확
- [ ] GT SC scale 정의 고정
- [ ] GT `SC_weight` 의미 문서화

## 현재 상태
- [x] 구현
- [ ] GT SC weight가 count/normalized/기타인지 최종 문서화

---

# Stage 19. Tract-Length Loss

```text
length_k = Σ ||p(t+1)-p(t)||
```

## Checklist
- [ ] unit=mm
- [ ] GT SC_length 정의 일치
- [ ] empty edge 처리
- [ ] weighted aggregation
- [ ] pass mode와 일관
- [ ] log/linear loss 기록

## 현재 상태
- [x] 구현
- [ ] final pass aggregation 검증

---

# Stage 20. Final Loss

```text
L_total =
L_recon
+ λ_KL       L_KL
+ λ_geom     L_geometry
+ λ_endpoint L_endpoint
+ λ_edge     L_edge
+ λ_corr     L_SC_corr
+ λ_mag      L_SC_mag
+ λ_len      L_length
```

## Checklist
- [ ] 모든 λ config화
- [ ] phase별 active loss
- [ ] raw/weighted loss logging
- [ ] loss별 grad norm
- [ ] validation 기반 λ tuning
- [ ] final joint fine-tuning에서 작은 LR

---

# Stage 21. Training Phase

## Phase 0 — ATM reproduction
- [x] pretrained load
- [x] original inference
- [x] baseline runtime

## Phase 1 — Data preprocessing
- [ ] 206 subjects 전체 preprocessing
- [ ] subject-level split 먼저 확정
- [ ] split 후 train-only normalization stats

## Phase 2 — Geometry adaptation
Train:
```text
ES + DS + ROI condition
```
Loss:
```text
L_recon + L_KL + L_geom
```

PASS:
- [ ] recon 감소
- [ ] geometry 개선
- [ ] NaN 없음

## Phase 3 — T1 Encoder training
```text
EA unfreeze
```
- [ ] EA optimizer 포함
- [ ] EA gradient
- [ ] EA update
- [ ] feature QC

**현재 가장 중요한 TODO**

## Phase 4 — Endpoint
```text
+ L_endpoint
```
- [ ] pair accuracy 상승
- [ ] endpoint distance 감소

## Phase 5 — Edge
```text
+ L_edge
```
- [ ] PR-AUC/F1 개선
- [ ] over-connectivity 감소

## Phase 6 — SC Corr
```text
+ L_SC_corr
```
- [ ] `sc_mode=pass`
- [ ] val SC Pearson 상승

## Phase 7 — SC Magnitude
```text
+ L_SC_mag
```
- [ ] CCC 상승
- [ ] MAE/RMSE 감소

## Phase 8 — Length
```text
+ L_length
```
- [ ] SC_length corr/CCC 상승

## Phase 9 — Joint Fine-tuning
Trainable:
```text
EA
ES
DS
ROI embedding
Edge head
Weight head
conditioning layers
```
- [ ] small LR
- [ ] val TRK 유지/개선
- [ ] val SC 개선
- [ ] overfit 없음

---

# Stage 22. Subject-level Split / Leakage

## Checklist
- [ ] train subjects
- [ ] validation subjects
- [ ] test subjects
- [ ] 동일 subject가 두 split에 없음
- [ ] scanner/site stratification
- [ ] normalization stats train-only
- [ ] test set으로 hyperparameter tuning 금지

---

# Stage 23. T1 Feature QC — 필수

pretrained EA 문제가 있었으므로 optional이 아님.

## 비교
```text
a_pretrained
vs
a_finetuned
```

## Checklist
- [ ] norm
- [ ] cosine
- [ ] pairwise corr
- [ ] Euclidean distance
- [ ] per-dim variance
- [ ] PCA
- [ ] scanner/site association

## 성공 방향
```text
subject-specific variation 존재
동일 subject에서는 안정적
scanner만 표현하지 않음
```

---

# Stage 24. Inference

## 최종
```text
T1
→ Fine-tuned EA
→ anatomy
→ Edge Head
→ candidate endpoint pairs
→ z ~ N(0,I)
→ DS
→ ROI-pair streamlines
→ merge
→ whole-brain TRK/TCK
→ hard reference SC
→ SC weight/length
```

## Checklist
- [ ] inference input = T1 only
- [ ] GT TRK 사용 안 함
- [ ] GT SC 사용 안 함
- [ ] edge candidate selection
- [ ] batched decode
- [ ] CPU export
- [ ] hard SC calculation
- [ ] soft vs hard SC consistency

## 현재 상태
- [ ] full inference pipeline 최종 검증

---

# Stage 25. Filtering / Trimming

## 원 ATM
1. brain 밖 streamline 제거
2. <20 mm 제거
3. WM surface trimming

## 우리 전략

### Training
```text
hard filter OFF
```

### Final inference/evaluation
optional post-processing으로:
```text
raw output
vs
filtered output
```
둘 다 평가.

## Checklist
- [ ] raw TRK metric
- [ ] optional filtered metric
- [ ] filtering 전/후 SC 비교

---

# Stage 26. GPU / Performance

## 유지해야 할 ATM 장점

금지:
```text
p1→p2→...→p500
```

유지:
```text
z batch → DS → [N,128,3]
```

## Checklist
- [ ] T1 Encoder subject당 1회
- [ ] anatomy feature reuse
- [ ] large streamline batch
- [ ] recurrent tracking 없음
- [ ] SC chunk aggregation
- [ ] final file write CPU
- [ ] GPU util
- [ ] peak VRAM
- [ ] streamlines/sec

## 현재 상태
- [x] recurrent tracking 없음
- [x] decode throughput 매우 높음
- [x] T1 encoder 1회/cache

---

# Stage 27. Precision

현재 report에서 mixed precision 좌표 오차가 커서:

```text
FP32 = reference
```

## Checklist
- [ ] FP32 baseline
- [ ] FP16/BF16 separate ablation
- [ ] coordinate error
- [ ] SC corr
- [ ] SC magnitude
- [ ] length error

---

# Stage 28. Smoke Test Final Gate

## Synthetic
- [ ] forward
- [ ] finite loss
- [ ] backward
- [ ] SC symmetry
- [ ] endpoint prob sum=1
- [ ] pair order invariance

## Real subject
- [ ] T1 encoder forward
- [ ] ES gradient
- [ ] DS gradient
- [ ] EA gradient
- [ ] ROI embedding gradient
- [ ] EdgeHead gradient
- [ ] WeightHead gradient
- [ ] SC loss→EA gradient
- [ ] optimizer step 후 EA 변경

## 현재 상태
- [x] 기존 synthetic/real smoke PASS
- [ ] **EA-unfreeze smoke를 새로 PASS해야 함**

---

# Stage 29. A10 Sanity Training Gate

Smoke PASS 후 바로 full training 금지.

먼저:
```text
1~5 subjects
50~200 steps
```

## Checklist
- [ ] L_recon 감소
- [ ] total finite
- [ ] EA update
- [ ] DS update
- [ ] feature variance
- [ ] loss별 grad norm
- [ ] VRAM
- [ ] step time
- [ ] NaN/OOM 없음

PASS 후 full GPU job.

---

# Stage 30. Final Evaluation

## TRK
- [ ] geometry
- [ ] endpoint distance
- [ ] valid streamline ratio
- [ ] length distribution
- [ ] coverage
- [ ] overreach

## ROI pair
- [ ] pair accuracy
- [ ] edge F1
- [ ] PR-AUC

## SC Weight
- [ ] Pearson
- [ ] Spearman
- [ ] CCC
- [ ] MAE
- [ ] RMSE
- [ ] density
- [ ] degree correlation

## SC Length
- [ ] Pearson
- [ ] CCC
- [ ] MAE
- [ ] RMSE

## Generalization
- [ ] unseen subjects
- [ ] scanner/site robustness
- [ ] final test set는 마지막에 1회

---

# Stage 31. 필수 Ablation

## Encoder
- [ ] Frozen pretrained EA
- [ ] Fine-tuned pretrained EA
- [ ] New encoder — fine-tuned EA가 실패할 때만

## Loss
- [ ] Geometry only
- [ ] + Endpoint
- [ ] + Edge
- [ ] + SC corr
- [ ] + SC mag
- [ ] + Length

## SC definition
- [ ] endpoint
- [ ] pass

최종 main result는 GT 정의에 맞는 pass.

---

# 현재 코드에서 가장 먼저 확인할 5개

## 1. T1 Encoder
```text
❌ frozen → ✅ trainable
```
- [ ] optimizer 포함
- [ ] gradient
- [ ] parameter update
- [ ] feature QC

## 2. Final SC definition
```text
final sc_mode = pass
```
- [ ] config
- [ ] training command
- [ ] validation
- [ ] hard final evaluation

## 3. Loss scale
- [ ] recon vs endpoint/SC gradient balance
- [ ] λ config/logging

## 4. Decoder geometry adaptation
- [ ] whole-brain bbox 변경 후 geometry 먼저 수렴

## 5. MRI domain
- [ ] pretrained normalization vs current-data normalization 비교
- [ ] scanner/intensity confound 확인

---

# Code Review Master Checklist

## Model
- [ ] T1 Encoder trainable option
- [ ] encoder param group
- [ ] VAE Encoder trainable
- [ ] Decoder trainable
- [ ] ROI-pair embedding
- [ ] Edge Head
- [ ] Weight Head
- [ ] pair-order invariant
- [ ] recurrent tracking 없음

## Data
- [ ] T1/tract/atlas same space
- [ ] ROI-pair assignments
- [ ] 128 points
- [ ] `sc_end`
- [ ] `sc_pass`
- [ ] GT SC_weight
- [ ] GT SC_length

## Loss
- [ ] recon
- [ ] KL
- [ ] geometry
- [ ] endpoint
- [ ] edge
- [ ] SC corr
- [ ] SC mag
- [ ] length
- [ ] 모든 λ config
- [ ] gradient norm logging

## Training
- [ ] geometry phase
- [ ] encoder phase
- [ ] endpoint phase
- [ ] edge phase
- [ ] SC corr phase
- [ ] SC mag phase
- [ ] length phase
- [ ] joint phase
- [ ] subject-level split
- [ ] best validation checkpoint

## SC
- [ ] final training mode=`pass`
- [ ] subject-level aggregation
- [ ] chunk별 SC loss 금지
- [ ] hard final SC
- [ ] soft-hard consistency

## Smoke
- [ ] pytest
- [ ] synthetic
- [ ] real
- [ ] EA gradient
- [ ] EA parameter update
- [ ] SC→EA gradient

## GPU
- [ ] A10 sanity run
- [ ] 50~200 stable steps
- [ ] VRAM
- [ ] step time
- [ ] GPU utilization
- [ ] sanity PASS 후 full job

---

# Architecture를 더 바꾸지 않아도 되는 조건

아래가 모두 충족되면 architecture 변경을 멈추고 hyperparameter tuning으로 이동한다.

```text
[ ] T1 encoder current dataset에서 trainable
[ ] T1 feature subject-specific variance 확인
[ ] ROI-pair streamline generation 정상
[ ] decoder geometry adaptation 수렴
[ ] endpoint 연결 정상
[ ] edge existence 정상
[ ] final SC target = pass
[ ] soft pass-SC reproduces GT SC >0.99
[ ] SC loss gradient가 EA/DS까지 전달
[ ] SC weight와 length 모두 학습 가능
[ ] synthetic + real smoke PASS
[ ] A10 sanity training stable
[ ] subject-level validation 성능 개선
```

이후 남는 것은:
- learning rate
- loss weights λ
- batch size
- unfreeze timing
- edge threshold
- latent sample 수
- regularization

등의 validation 기반 tuning이다.

---

# Original ATM vs Final Project

| Stage | Original ATM | Final Project |
|---|---|---|
| MRI | TractoInferno T1 | current T1 domain |
| Space | MNI152 | GT/atlas-aligned NLin6 |
| Bundle | 30 anatomical bundles | ROI-pair bundles |
| T1 Encoder | bundle-aware | subject-specific whole-brain |
| Segmentation DA | bundle occupancy | core에서 제거 |
| Streamline Encoder | bundle VAE | ROI-pair VAE |
| Latent | bundle KDE | N(0,I) baseline |
| Conditioning | anatomy FiLM | anatomy + ROI pair |
| Decoder | bundle coordinates | whole-brain ROI-pair coordinates |
| Geometry | VAE + Adj | recon + KL + geometry |
| Endpoint | 없음 | 추가 |
| Edge existence | 없음 | 추가 |
| SC | evaluation only | training target |
| SC definition | connectome evaluation | GT-compatible pass-SC |
| SC magnitude | 없음 | 추가 |
| Tract length | geometry evaluation | training target |
| Encoder | pretrained inference | current-data fine-tuning |
| Output | 30 bundles | whole-brain TRK + SC |

---

# 근거

## ATM 논문에서 직접 가져온 요소
- T1 Encoder `EA`
- Segmentation Decoder `DA`
- Streamline VAE Encoder `ES`
- Streamline Decoder `DS`
- FiLM anatomy conditioning
- 64-D latent
- 128-point streamline representation
- KDE inference
- `L_SEG + L_VAE + L_ADJ`
- filtering/trimming
- connectome은 원 논문에서 evaluation으로 사용

## 현재 프로젝트에서 추가/변경한 요소
다음은 원 논문이 아니라 현재 연구 목적에 맞춘 project-specific tuning이다.
- 30 bundle → ROI-pair bundle
- endpoint loss
- edge-existence head
- streamline weight head
- differentiable SC training loss
- SC correlation/magnitude loss
- tract-length loss
- pass-SC target
- current-data T1 encoder fine-tuning

## 현재 보고서상 핵심 미해결/재검증 항목
1. pretrained T1 anatomy feature가 매우 작음
2. T1 encoder가 frozen
3. pass GT SC와 endpoint SC mismatch
4. reconstruction gradient scale imbalance
5. whole-brain bbox 변경에 따른 initial geometry distortion
