# ROI-pair ATM의 Pass-Edge 문제 해결 + Route Loss + TractoLearn/GESTA 증강 전략

## 최종 목표

```text
T1-only
→ subject-specific whole-brain TRK/TCK
→ PASS-based SC weight
→ SC tract-length
```

현재 핵심 문제는 두 종류다.

1. **Pass-edge 문제**
   - 현재 explicit anatomical supervision은 endpoint 중심
   - 특히 SUB-SUB 연결의 약 63%가 endpoint bundle이 아니라 다른 streamline이 중간에 통과하며 만들어지는 pass-edge
   - 따라서 endpoint만 맞춰서는 SUB-SUB connectivity를 제대로 보장하기 어려움

2. **Bundle-size imbalance 문제**
   - streamline이 많은 endpoint-defined ROI-pair bundle은 geometry 학습이 풍부
   - streamline이 적은 bundle은 반복 sampling만 하면 diversity 부족
   - TractoLearn/GESTA latent-space augmentation으로 under-represented bundle을 보완

```text
Pass/transit 문제
→ Route/Visit supervision + block-wise PASS-SC

Endpoint bundle imbalance
→ GESTA-based synthetic augmentation
```

---

# 1. 현재 loss가 실제로 무엇을 보고 있는가?

현재 구조에서 `L_recon`, `L_geometry`는 streamline 전체 좌표를 보기 때문에 중간 경로를 완전히 무시하는 것은 아니다.

하지만 해부학적 connectivity supervision은 주로 `L_endpoint`를 통해:

```text
streamline 시작 ROI
+
streamline 끝 ROI
```

를 직접 맞춘다.

즉 explicit connectivity question은:

> 어디서 출발해서 어디로 도착했는가?

에 가깝다.

---

# 2. Endpoint loss의 구조적 한계

GT:

```text
CTX A
 ↓
SUB 1
 ↓
SUB 2
 ↓
CTX B
```

Prediction:

```text
CTX A
 ↓
SUB 5
 ↓
SUB 9
 ↓
CTX B
```

둘 다 endpoint는:

```text
CTX A ↔ CTX B
```

이므로 endpoint loss는 상당 부분 맞다고 볼 수 있다.

하지만 중간 route는 완전히 다르다.

---

# 3. 왜 SUB-SUB에서 특히 문제가 되는가?

현재 데이터 분석에서:

> **SUB-SUB 연결의 약 63%는 endpoint bundle이 아니라 다른 streamline이 중간에 통과하며 만들어지는 pass-edge**

이다.

즉 GT SC에서:

```text
SUB1 ↔ SUB2 = positive
```

라고 해도 반드시:

```text
SUB1에서 시작해서 SUB2에서 끝나는 streamline
```

이 존재하는 것이 아니다.

오히려:

```text
CTX3 → SUB1 → SUB2 → CTX8
CTX4 → SUB1 → SUB2 → CTX9
CTX5 → SUB1 → SUB2 → CTX7
```

같은 streamline들이 SUB1과 SUB2를 지나가면서 PASS-SC에 기여할 수 있다.

---

# 4. 이게 왜 중요한가?

우리 preprocessing의 ROI-pair bundle은 endpoint ROI pair 기준이다.

예:

```text
CTX3 ↔ CTX8 bundle
```

안에:

```text
CTX3 → SUB1 → SUB2 → CTX8
```

streamline이 들어있다.

따라서 `SUB1 ↔ SUB2` SC edge를 잘 만들기 위해 반드시 `SUB1 ↔ SUB2 endpoint bundle`을 생성할 필요는 없다.

오히려 다른 endpoint bundle들이 올바르게 SUB1과 SUB2를 통과해야 한다.

---

# 5. GESTA만으로 pass-edge 문제를 해결할 수 없는 이유

GESTA는 endpoint-defined bundle의 geometry diversity 보강에 적합하다.

하지만 `SUB1 ↔ SUB2`가 endpoint bundle 자체로 존재하지 않는 pass-edge라면, GESTA로 `SUB1↔SUB2 bundle`을 새로 만들어 증강하는 것은 GT 정의와 맞지 않는다.

```text
GESTA
=
endpoint bundle imbalance 해결

Route loss
=
intermediate passage 해결
```

---

# 6. 문제를 역할별로 분리

```text
Endpoint-defined bundle imbalance
→ GESTA augmentation

Intermediate route / pass-edge mismatch
→ Route / ROI-visitation loss

Subject-level SC topology imbalance
→ Overall + CC + CS + SS PASS-SC loss

SC strength
→ Weight Head + log magnitude loss
```

---

# 7. PASS-SC와 Endpoint SC의 차이

## Endpoint SC

streamline의 first ROI와 last ROI만 보고 edge를 만든다.

## PASS-SC

streamline이 실제로 지나간 ROI pair를 기반으로 SC contribution을 만든다.

예:

```text
CTX A
→ SUB1
→ SUB2
→ CTX B
```

이면:

```text
CTX A ↔ SUB1
SUB1 ↔ SUB2
SUB2 ↔ CTX B
```

같은 pass 관계에 기여할 수 있다.

현재 GT `.mat` SC가 pass-based라면 최종 SC supervision도 pass-SC 정의에 맞춰야 한다.

---

# 8. 왜 block-wise PASS-SC loss가 필요한가?

전체 PASS-SC correlation `r_all` 하나만 쓰면 edge 수가 많은 block이 전체 correlation을 지배할 수 있다.

예:

```text
CTX-CTX = 많음
CTX-SUB = 중간
SUB-SUB = 적음
```

이면 CTX-CTX가 잘 맞는 것만으로 overall r이 높아질 수 있고, SUB-SUB 오차는 묻힐 수 있다.

---

# 9. Detailed PASS-SC correlation

다음 4개를 계산한다.

```text
r_all
r_CC
r_CS
r_SS
```

```text
L_type =
(
    (1-r_CC)
  + (1-r_CS)
  + (1-r_SS)
) / 3
```

```text
L_SC_corr =
λ_global * (1-r_all)
+
λ_type * L_type
```

필요하면 `λ_CC`, `λ_CS`, `λ_SS`를 분리할 수 있다.

---

# 10. block-wise SC loss만으로 geometry를 보장할 수 없는 이유

SC correlation은 subject-level 결과를 본다.

모델은 실제 route를 고치기보다 Weight Head를 조절해 SC loss를 줄일 수도 있다.

즉:

```text
SC_pred ≈ SC_GT
```

가 되었다고 해서:

```text
streamline route ≈ GT route
```

가 보장되지는 않는다.

---

# 11. Weight Head shortcut 문제

현재 SC strength는 Weight Head가 담당한다.

그러면 모델 입장에서는:

```text
geometry 수정
```

대신:

```text
weight 수정
```

으로 SC loss를 줄이는 shortcut이 생길 수 있다.

그래서 route-level supervision이 필요하다.

---

# 12. Route Loss의 목적

Endpoint loss:

> 출발지와 도착지가 맞는가?

Route loss:

> 중간에 지나가야 할 ROI도 맞게 지나가는가?

---

# 13. GT ROI visitation signature

GT streamline:

```text
CTX A
→ SUB1
→ SUB2
→ CTX B
```

이면:

```text
CTX A = 1
SUB1  = 1
SUB2  = 1
CTX B = 1
others = 0
```

즉:

```text
v_GT ∈ {0,1}^N_ROI
```

를 만든다.

---

# 14. Predicted streamline의 soft ROI visitation

Predicted streamline의 128개 point에 atlas ROI distance 또는 soft assignment를 적용해:

```text
P(ROI_i visited | streamline)
```

를 계산한다.

결과:

```text
v_pred ∈ [0,1]^N_ROI
```

---

# 15. Route loss 후보

### BCE

```text
L_route =
BCE(v_pred, v_GT)
```

### Dice

```text
L_route_dice =
1 -
2 * sum(v_pred * v_GT)
/
(sum(v_pred) + sum(v_GT) + eps)
```

### Focal BCE

rare ROI visit가 매우 불균형하면 후보.

---

# 16. ROI visitation과 pass-edge signature

## ROI visitation

```text
어떤 ROI를 지나갔는가?
```

장점:
- 단순
- 안정적
- gradient 설계 쉬움

## Pass-edge signature

```text
어떤 ROI pair transition/pass를 만들었는가?
```

예:

```text
CTX A-SUB1 = 1
SUB1-SUB2 = 1
SUB2-CTX B = 1
```

PASS-SC와 더 직접적이지만 구현이 복잡하다.

### 권장

1차 baseline은 ROI visitation loss.
2차 ablation으로 pass-edge/transition loss 추가.

---

# 17. 왜 Route Loss가 SUB-SUB에 특히 유리한가?

SUB-SUB pass-edge는 endpoint bundle 존재 여부가 아니라, 다른 streamline이 두 SUB ROI를 실제로 지나야 하는 문제다.

따라서 route loss가 intermediate passage에 직접 supervision을 준다.

---

# 18. 도로 비유

GT 도로:

```text
서울 → 대전 → 대구 → 부산
```

Endpoint loss는:

```text
서울 ↔ 부산
```

만 본다.

Prediction:

```text
서울 → 광주 → 순천 → 부산
```

이어도 endpoint는 맞다.

하지만 `대전 ↔ 대구`라는 pass-edge는 사라진다.

Route loss는:

```text
서울 → 대전 → 대구 → 부산
```

이라는 중간 경로까지 맞추라고 한다.

---

# 19. SUB-SUB 63% pass-edge의 의미

> SUB-SUB 연결의 대부분이 `SUB_i에서 시작해서 SUB_j에서 끝나는 streamline`이 아니라 다른 endpoint를 가진 streamline이 `SUB_i와 SUB_j를 통과`해서 만들어지는 연결이라면, endpoint 중심 supervision은 SUB-SUB connectivity에 직접적인 학습 신호를 충분히 제공하지 못한다.

---

# 20. 추가 옵션: SUB-SUB pass presence loss

GT:

```text
A_GT(i,j) = 1 if SC_GT(i,j) > 0 else 0
```

Pred soft pass probability:

```text
P_pred(i,j)
```

Loss:

```text
L_presence_SS =
BCE(
    P_pred[SS],
    A_GT[SS]
)
```

즉:

```text
Presence
→ 실제 pass-edge가 존재하는가?

Magnitude
→ 그 edge strength가 얼마인가?
```

로 분리.

---

# 21. SUB-SUB 최종 loss 예

```text
L_SS =
λ_corr_SS     * (1-r_SS)
+
λ_mag_SS      * L_logmag_SS
+
λ_presence_SS * L_presence_SS
```

그리고 streamline level에서:

```text
+ λ_route * L_route
```

---

# 22. SC magnitude loss

```text
L_SC_mag =
MAE(
    log(SC_pred + eps),
    log(SC_GT + eps)
)
```

Correlation은 pattern을 보고, log magnitude는 scale을 본다.

---

# 23. 권장 전체 loss

```text
L_total =
L_recon
+ λ_KL       * L_KL
+ λ_geom     * L_geometry
+ λ_endpoint * L_endpoint
+ λ_route    * L_route
+ λ_edge     * L_edge
+ λ_global   * L_SC_global
+ λ_type     * L_SC_type
+ λ_mag      * L_SC_mag
+ λ_len      * L_length
```

선택적:

```text
+ λ_presence_SS * L_presence_SS
```

---

# 24. Weight Head는 너무 일찍 강하게 학습하지 않기

권장 순서:

```text
geometry
→ endpoint
→ route
→ global PASS-SC
→ block-wise PASS-SC
→ Weight Head + magnitude
→ length
→ joint fine-tune
```

이유:

```text
geometry correction보다 weight shortcut이 먼저 생기는 것 방지
```

---

# 25. 권장 training phase

## Phase 1
`L_recon + L_KL + L_geometry`

## Phase 2
`+ L_endpoint`

## Phase 3
`+ L_route`

## Phase 4
`+ Global PASS-SC`

## Phase 5
`+ CC / CS / SS block-wise PASS-SC`

## Phase 6
`+ Weight Head + log magnitude`

## Phase 7
`+ tract-length`

## Phase 8
joint fine-tuning

---

# 26. 여기서 GESTA가 해결하는 문제

GESTA는:

```text
pass-edge 문제
```

가 아니라:

```text
endpoint-defined bundle imbalance
```

를 해결하는 데 사용한다.

예:

```text
ROI A↔B = 5000 streamlines
ROI C↔D = 500
ROI E↔F = 20
```

이면 A-B는 geometry diversity가 충분하지만 E-F는 매우 부족하다.

---

# 27. 단순 oversampling의 한계

E-F 20개를 반복해서 쓰면:

```text
20 real
→ repeat
→ repeat
```

학습 노출량은 늘어나지만 geometry diversity는 늘어나지 않는다.

GESTA는 이 부분을 latent-space synthetic generation으로 보완한다.

---

# 28. GESTA synthetic generation — 전체 흐름

```text
Seed bundle
↓
Autoencoder Encoder
↓
Latent seed cloud
↓
KDE / Parzen density estimation
↓
Proposal distribution
↓
Rejection sampling
↓
Accepted new latent samples
↓
Decoder
↓
Synthetic complete streamlines
↓
Geometric / anatomical filtering
↓
Accepted synthetic bundle
```

---

# 29. TractoLearn 구현 위치

Repository:

```text
https://github.com/scil-vital/tractolearn
```

주요 파일:

```text
tractolearn/generative/generate_points.py
scripts/ae_generate_streamlines.py
tractolearn/filtering/streamline_space_filtering.py
```

---

# 30. TractoLearn `RejectionSampler`

실제 구현에는 다음이 포함된다.

```text
KernelDensity
multivariate_normal
GaussianMixture
GridSearchCV
Silverman bandwidth
rejection sampling
batch parallel sampling
```

---

# 31. GESTA Step 1 — Seed bundle

원 GESTA/TractoLearn:
- subject-specific bundle
- optional atlas bundle

우리 training adaptation:
- TRAIN GT endpoint-defined ROI-pair bundle

예:

```text
ROI E↔F
20 real streamlines
```

---

# 32. Step 2 — Streamline → latent

원 TractoLearn은 자체 AE encoder 사용.

우리 pipeline에서는:

```text
ATM Streamline Encoder ES
```

를 사용.

즉:

```text
20 real streamlines
↓
ES
↓
z1 ... z20
```

---

# 33. Step 3 — Latent seed cloud

예:

```text
Z_seed ∈ R^(20×64)
```

이 seed cloud가 해당 bundle의 streamline shape distribution을 나타낸다.

---

# 34. Step 4 — KDE / Parzen density estimation

```text
p(z | bundle)
≈
KDE(Z_seed)
```

즉:

> 이 bundle과 비슷한 streamline shape가 latent space에서 어느 영역에 존재할 가능성이 높은가?

를 추정한다.

---

# 35. KDE bandwidth

TractoLearn은:
- manual bandwidth
- Cross-Validation
- Silverman1986
- `kde_bw_factor`

를 지원한다.

`kde_bw_factor`로 synthetic generation을 더 strict/permissive하게 조절 가능.

---

# 36. Step 5 — Proposal distribution

지원:

```text
multivariate_normal
GMM
```

### Gaussian

```text
mean = mean(Z_seed)
cov  = cov(Z_seed)
z_candidate ~ N(mean, cov)
```

### GMM

latent distribution이 multi-modal일 때 사용 가능.

TractoLearn에는 silhouette/BIC 기반 component selection logic도 있다.

---

# 37. Step 6 — Rejection sampling

목표 분포:

```text
p(z) = KDE(seed latent)
```

proposal:

```text
q(z)
```

candidate:

```text
z ~ q(z)
```

그리고 KDE density와 proposal density 비율을 기반으로 accept/reject.

직관:

```text
proposal candidate
↓
seed bundle latent distribution과 충분히 비슷한가?
↓
YES → accept
NO  → reject
```

---

# 38. Step 7 — Accepted latent

예:

```text
20 seed latent
↓
RejectionSampler
↓
80 new accepted latent
```

결과:

```text
Z_new
```

---

# 39. Step 8 — Decoder

우리 adaptation에서는:

```text
ATM Decoder DS
```

사용.

그리고 단순히 `z_new`만 넣지 않고:

```text
z_new
+
subject T1 anatomy
+
ROI-pair condition
```

을 함께 넣는다.

즉:

```text
DS(z_new, anatomy_subject, ROI_i, ROI_j)
```

---

# 40. 우리 adaptation의 장점

원 GESTA generative concept:

```text
z
→ streamline
```

우리:

```text
z
+
subject anatomy
+
ROI pair
→ streamline
```

따라서 synthetic streamline을 해당 subject와 해당 connection에 더 강하게 condition 가능.

---

# 41. Step 9 — Raw synthetic streamline

accepted latent를 decode해:

```text
[128,3]
```

complete streamline을 생성.

예:

```text
z21' → s21'
z22' → s22'
...
```

---

# 42. Raw synthetic은 자동으로 realistic하지 않음

가능한 실패:

```text
endpoint 오류
길이 이상
curvature 이상
brain 밖 이탈
WM 이탈
duplicate
```

따라서:

```text
Generated
≠
Guaranteed valid
```

이다.

---

# 43. TractoLearn/GESTA filtering 철학

원 pipeline에는:
- length filtering
- WM filtering
- ROI filtering
- local orientation / peaks filtering

등이 있다.

원 dMRI 기반 pipeline에서는 FA/FODF peaks도 사용한다.

---

# 44. 우리 T1-only에서 사용할 수 있는 filter

최종 inference 입력이 T1 only이므로:

```text
Endpoint ROI consistency
Brain mask
WM mask
Length
Curvature
Self-intersection / abnormal geometry
Duplicate control
```

등 T1/atlas/geometry 기반 filter 사용.

---

# 45. T1-only에서 필수로 쓰면 안 되는 filter

```text
DWI
FA
FODF peaks
```

가 필요한 filter는 최종 inference 필수 조건으로 쓰면 안 된다.

Training-only QC로 참고할 수는 있지만 inference dependency와 분리해야 한다.

---

# 46. GESTA synthetic은 얼마나 realistic한가?

정확한 표현은:

```text
biological ground truth
```

가 아니라:

```text
tractography-plausible synthetic streamline
```

이다.

Realism의 근거:
1. real seed streamline latent distribution에서 sampling
2. KDE density 기반
3. rejection sampling
4. learned decoder manifold
5. post-generation plausibility filtering

---

# 47. Synthetic을 GT와 동일 신뢰도로 쓰지 않기

권장:

```text
Real GT streamline
→ primary supervision

GESTA synthetic
→ augmentation
```

필요하면:

```text
L_recon =
L_real
+
λ_syn * L_synthetic
```

에서 `λ_syn < 1`로 시작.

예:

```text
real = 1.0
synthetic = 0.25~0.5
```

수치는 validation에서 결정.

---

# 48. Bundle-size balancing에 GESTA 적용

예:

```text
Bundle A = 5000 real
Bundle B = 500 real
Bundle C = 20 real
```

학습에서 pair당 16개를 쓴다고 하자.

### Large
```text
A → random downsample → 16
```

### Medium
```text
B → random sample → 16
```

### Small
```text
C → real + GESTA synthetic pool → 16
```

목표:

```text
pair당 training exposure는 비슷하게
geometry diversity는 GESTA로 보완
```

---

# 49. Synthetic count ≠ SC strength

예:

```text
weak edge
GT SC = 10
```

geometry augmentation 때문에 200 synthetic candidate를 만들어도:

```text
SC target = 10
```

을 유지해야 한다.

augmentation은 geometry training exposure만 조절한다.

---

# 50. Weight Head 역할

각 generated streamline:

```text
w_k >= 0
```

예측.

PASS-SC contribution:

```text
SC_pred(i,j)
=
Σ_k w_k * C_k(i,j)
```

개념.

즉:

```text
candidate 수
→ geometry / coverage

weight
→ actual SC strength
```

로 분리.

---

# 51. GESTA와 Route Loss는 대체 관계가 아님

GESTA:

```text
endpoint-defined bundle under-representation
```

해결.

Route loss:

```text
intermediate ROI passage
```

해결.

둘은 서로 다른 문제를 담당한다.

---

# 52. 전체 최종 구조

```text
TRAIN T1 + GT TRK + Atlas + GT PASS-SC
                    │
                    ↓
         Endpoint ROI-pair bundles
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
 High-count bundle        Low-count bundle
        │                       │
   downsample                    ↓
        │                     ATM ES
        │                       ↓
        │                  latent seeds
        │                       ↓
        │             TractoLearn GESTA
        │         KDE + proposal + rejection
        │                       ↓
        │                 synthetic z
        │                       ↓
        │                     ATM DS
        │                       ↓
        │             synthetic streamline
        │                       ↓
        │               T1/atlas filtering
        │                       │
        └────────────┬──────────┘
                     ↓
           Balanced geometry training
                     ↓
       L_recon + L_geom + L_endpoint
                     ↓
                 L_route
                     ↓
             Whole-brain TRK
                     ↓
           Differentiable PASS-SC
                     ↓
    ┌───────────┬───────────┬───────────┐
    ↓           ↓           ↓           ↓
 Overall      CTX-CTX     CTX-SUB     SUB-SUB
  Corr         Corr        Corr         Corr
    └───────────┴───────────┴───────────┘
                     ↓
          Log magnitude + Length
                     ↓
                Weight Head
```

---

# 53. Training vs Inference의 GESTA seed

### Training
```text
TRAIN GT endpoint bundle
→ latent seed
```

### Inference
```text
New T1
↓
ATM initial predicted bundle
↓
ES
↓
GESTA sampler
↓
DS
↓
completion
```

Validation/Test/Inference에서 GT streamline을 seed로 쓰면 leakage.

---

# 54. Route loss 구현 체크리스트

- [ ] GT streamline ROI visitation extraction
- [ ] Pred streamline soft ROI assignment
- [ ] differentiable visitation probability
- [ ] BCE / Dice 구현
- [ ] route loss gradient가 DS까지 도달
- [ ] route loss gradient가 EA까지 도달 가능한지 확인
- [ ] GT route vs Pred route visualization
- [ ] endpoint는 맞지만 route가 틀린 synthetic test 작성

---

# 55. PASS-SC 체크리스트

- [ ] GT `.mat` pass-SC definition 확인
- [ ] predicted pass-SC builder 동일 정의
- [ ] Overall correlation
- [ ] CTX-CTX correlation
- [ ] CTX-SUB correlation
- [ ] SUB-SUB correlation
- [ ] log magnitude
- [ ] diagonal 제외
- [ ] upper triangle
- [ ] zero edge 처리 정책
- [ ] FP32

---

# 56. SUB-SUB 진단 체크리스트

- [ ] endpoint-defined SUB-SUB 비율
- [ ] pass-only SUB-SUB 비율
- [ ] 현재 분석의 약 63% 재현 확인
- [ ] endpoint-supported SUB-SUB performance
- [ ] pass-only SUB-SUB performance
- [ ] route loss 전/후 비교
- [ ] block-wise SS corr 전/후
- [ ] SS magnitude error
- [ ] SS edge recall

---

# 57. GESTA augmentation 체크리스트

- [ ] TractoLearn RejectionSampler adapter
- [ ] ATM ES를 encoder로 사용
- [ ] ATM DS를 decoder로 사용
- [ ] seed latent extraction
- [ ] KDE bandwidth mode
- [ ] Gaussian proposal baseline
- [ ] GMM ablation
- [ ] rejection acceptance rate logging
- [ ] max sampling budget
- [ ] min seed count
- [ ] max synthetic ratio
- [ ] synthetic filtering
- [ ] duplicate suppression
- [ ] GT SC target invariance

---

# 58. Synthetic realism 체크리스트

- [ ] endpoint accuracy
- [ ] length distribution
- [ ] WM occupancy
- [ ] brain occupancy
- [ ] curvature distribution
- [ ] bundle voxel coverage
- [ ] overreach
- [ ] duplicate ratio
- [ ] latent distance to real seed cloud

---

# 59. Validation

## SC
```text
Overall
CTX-CTX
CTX-SUB
SUB-SUB
```

각각:
```text
Pearson
Spearman
CCC
MAE
RMSE
```

## SUB-SUB
```text
Endpoint-supported SS
vs
Pass-only SS
```

를 별도로 본다.

이게 route loss 효과 검증에 가장 중요하다.

## Bundle size
```text
low-count
mid-count
high-count
```

로 나눠 GESTA 효과 확인.

---

# 60. 필수 Ablation — Route

```text
A. No route loss
B. + ROI visitation loss
C. + pass-edge presence loss
D. + both
```

핵심 metric:
```text
pass-only SUB-SUB corr / recall
```

---

# 61. 필수 Ablation — SC loss

```text
A. Overall only
B. Overall + CC/CS/SS
C. B + log magnitude
D. C + SS presence
```

---

# 62. 필수 Ablation — GESTA

```text
A. No balancing
B. Pair-balanced only
C. Pair-balanced + duplicate oversampling
D. Pair-balanced + GESTA Gaussian
E. Pair-balanced + GESTA rejection sampling
F. E + filtering
```

---

# 63. 실패로 보는 경우

### Route
```text
SS corr ↑
BUT
route geometry 악화
```

### GESTA
```text
coverage ↑
BUT
overreach ↑↑
```

### Weight Head
```text
SC corr ↑
BUT
route loss / endpoint accuracy 악화
```

이면 Weight shortcut 의심.

---

# 64. PASS 기준

- [ ] endpoint accuracy 유지/개선
- [ ] route visitation accuracy 개선
- [ ] pass-only SUB-SUB corr 개선
- [ ] pass-only SUB-SUB recall 개선
- [ ] SUB-SUB magnitude error 개선/유지
- [ ] low-count bundle geometry 개선
- [ ] GESTA synthetic overreach 제한
- [ ] Overall SC 유지/개선
- [ ] CTX-CTX 유지
- [ ] CTX-SUB 유지
- [ ] tract-length 유지/개선
- [ ] Weight Head shortcut 없음
- [ ] inference input은 T1 only

---

# 65. 최종 핵심 정리

```text
1. Endpoint supervision의 한계
→ 중간 route를 직접 보지 않음

2. SUB-SUB 특성
→ 약 63%가 endpoint bundle이 아니라 pass-edge

3. 해결
→ Route / ROI visitation loss

4. SC block imbalance
→ Overall + CC + CS + SS PASS-SC loss

5. Strength
→ Weight Head + log magnitude

6. Endpoint bundle imbalance
→ TractoLearn/GESTA latent augmentation
```

---

# 66. 한 문장 결론

> **SUB-SUB 연결의 다수가 다른 endpoint bundle의 중간 통과로 만들어지는 pass-edge이므로, endpoint loss와 GESTA 증강만으로는 충분하지 않다. Endpoint-defined bundle의 데이터 부족은 GESTA로 보완하되, pass-only SUB-SUB 연결은 streamline-level Route/ROI-visitation loss와 block-wise PASS-SC supervision을 통해 직접 학습시키는 것이 핵심이다.**

---

# 67. TractoLearn/GESTA source note

본 문서에서 TractoLearn/GESTA 구현에 기반한 부분은 다음 공개 코드 구조를 바탕으로 한다.

```text
Repository:
scil-vital/tractolearn

Generative sampling:
tractolearn/generative/generate_points.py

Generation script:
scripts/ae_generate_streamlines.py

Filtering:
tractolearn/filtering/streamline_space_filtering.py
```

확인된 주요 구현 요소:

```text
RejectionSampler
KernelDensity
Gaussian proposal
GMM proposal
Cross-Validation bandwidth
Silverman bandwidth
batch sampling
per-bundle generation target
max total sampling
length / WM / orientation filtering components
```

본 프로젝트에서 새로 추가하는 부분:

```text
ATM ES/DS와 TractoLearn sampler 통합
T1 anatomy + ROI-pair-conditioned decoding
ROI-pair balanced training
Route / ROI-visitation loss
pass-only SUB-SUB validation
Overall + CC + CS + SS PASS-SC loss
Weight Head와 synthetic count 분리
T1-only filtering subset
```
