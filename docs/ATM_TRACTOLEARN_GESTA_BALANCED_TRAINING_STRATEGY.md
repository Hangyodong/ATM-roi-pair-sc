# ATM × TractoLearn(GESTA) 기반 ROI-pair Bundle 균형 학습 전략
## Under-represented bundle augmentation + detailed SC supervision + validation checklist

> **목적**
>
> 현재 ROI-pair ATM이 streamline 수가 많거나 SC가 큰 연결에 편향되지 않도록,
> `scil-vital/tractolearn`의 GESTA generative sampling 구현을 활용하여
> **streamline이 적은 ROI-pair bundle의 학습 노출량과 geometry diversity를 보완**한다.
>
> 동시에 최종 SC는 단순 전체 correlation 하나가 아니라
> `Overall + CTX-CTX + CTX-SUB + SUB-SUB`으로 세분화하여 supervision한다.
>
> 최종 inference 목표는 그대로 유지한다.
>
> ```text
> New T1
> → subject-specific whole-brain TRK/TCK
> → SC weight
> → SC tract-length
> ```
>
> **중요:** TractoLearn의 GESTA 코드는 그 자체로 “imbalance-aware training sampler”는 아니다.
> GESTA는 seed bundle의 latent distribution에서 새로운 streamline을 생성하는 generative module이다.
> 따라서 본 전략에서는 **GESTA를 under-represented ROI-pair bundle augmentation 엔진으로 사용하고,
> ROI-pair balanced sampler와 결합**한다.

---

# 1. 사용 근거: TractoLearn에서 실제로 제공하는 GESTA 관련 코드

Repository:

```text
https://github.com/scil-vital/tractolearn
```

주요 코드:

```text
tractolearn/generative/generate_points.py
scripts/ae_generate_streamlines.py
tractolearn/filtering/streamline_space_filtering.py
```

README는 generative tractography 목적으로 `tractolearn`을 사용할 경우
GESTA와 FIESTA를 인용하도록 명시한다.

---

# 2. TractoLearn에서 가져올 핵심 모듈

## 2.1 `RejectionSampler`

경로:

```text
tractolearn/generative/generate_points.py
```

TractoLearn 구현은 seed latent data에 대해:

```text
KernelDensity
+
proposal distribution
+
rejection sampling
```

을 수행한다.

지원되는 핵심 기능:

```text
KDE / Parzen density estimation

KDE bandwidth:
- 직접 지정
- Cross-Validation
- Silverman1986

Proposal:
- multivariate_normal
- GaussianMixture(GMM)

Scaling:
- max
- 3rd_quartile

Sampling:
- batch-wise
- reproducible SeedSequence
- acceptance-rate logging
```

---

# 3. `generate_points()`에서 참고할 기능

경로:

```text
tractolearn/generative/generate_points.py
```

입력 개념:

```text
bundle
atlas_bundle
max_seeds
composition
bandwidth
use_rs
gmm_n_component
num_generate_points
```

원 TractoLearn에서는:

```text
subject bundle
+
optional atlas bundle
↓
AE encoding
↓
latent seeds
↓
sampling
↓
AE decoding
↓
new streamlines
```

구조다.

---

# 4. `ae_generate_streamlines.py`에서 참고할 기능

경로:

```text
scripts/ae_generate_streamlines.py
```

특히 다음 아이디어를 가져온다.

```text
per-bundle desired number of generated streamlines
per-bundle maximum total sampling
maximum number of seed streamlines
Parzen bandwidth
rejection sampling on/off
sampling batch size
minimum / maximum streamline length
bundle-wise generation loop
```

즉 **각 bundle마다 생성 목표량과 최대 sampling budget을 따로 설정**할 수 있다.

이 특성은 우리 ROI-pair imbalance 문제에 매우 유용하다.

---

# 5. TractoLearn filtering에서 참고할 부분

경로:

```text
tractolearn/filtering/streamline_space_filtering.py
```

확인되는 기능 예:

```text
filter_grid_roi()
cut_streamlines_outside_mask()
streamline local orientation analysis
WM occupancy 관련 feature
length / region / curvature 관련 tractography feature
```

원 pipeline에는 FA와 FODF peaks를 활용하는 filtering도 존재한다.

하지만 우리의 최종 inference는:

```text
T1 only
```

이므로,

```text
FA
FODF peaks
dMRI
```

가 필요한 filter를 최종 필수 조건으로 사용하면 안 된다.

---

# 6. 가장 중요한 설계 원칙

이번 전략의 핵심은:

```text
GESTA augmentation만 적용
```

이 아니다.

최종 구조는:

```text
ROI-pair balanced sampling
+
GESTA latent augmentation
+
fixed training exposure per pair
+
detailed SC supervision
```

이다.

---

# 7. 문제 정의

예를 들어 한 subject에서:

```text
ROI A ↔ ROI B : 5000 streamlines
ROI C ↔ ROI D :  800 streamlines
ROI E ↔ ROI F :   20 streamlines
```

전체 streamline pool에서 random sampling하면:

```text
A-B
```

가 training에 훨씬 많이 등장한다.

결과:

```text
large bundle geometry → 잘 학습
small bundle geometry → 학습 부족
```

가 발생할 수 있다.

---

# 8. 단순 oversampling의 문제

small bundle의 20개 streamline을 단순히 반복하면:

```text
20개
→ 20개
→ 20개
→ 20개
```

같은 streamline을 계속 보게 된다.

이는:

```text
training exposure는 증가
BUT
geometry diversity는 증가하지 않음
```

이라는 문제가 있다.

---

# 9. GESTA를 사용하는 이유

GESTA-style augmentation은:

```text
small bundle의 실제 seed streamline
↓
latent representation
↓
seed distribution 주변에서 새로운 latent 생성
↓
decode
↓
new complete streamlines
```

을 수행한다.

따라서:

```text
단순 duplicate oversampling
```

보다:

```text
geometry-aware synthetic augmentation
```

을 할 수 있다.

---

# 10. 우리 모델에는 별도 TractoLearn AE를 그대로 넣지 않는다

현재 ATM에는 이미:

```text
Streamline Encoder ES
Streamline Decoder DS
```

가 있다.

따라서 권장 구조는:

```text
TractoLearn:
RejectionSampler / KDE / GMM / sampling logic

ATM:
ES / DS / T1 anatomy conditioning / ROI-pair conditioning
```

의 조합이다.

즉 TractoLearn의 autoencoder를 통째로 추가해
두 개의 streamline autoencoder를 운영하지 않는다.

---

# 11. 최종 역할 분담

## ATM

```text
EA
→ subject T1 anatomy encoding

ES
→ streamline → latent

DS
→ latent + anatomy + ROI-pair → complete streamline
```

## TractoLearn에서 port

```text
RejectionSampler
KDE / Parzen bandwidth logic
Gaussian/GMM proposal
batch sampling
max sampling budget
seed limit logic
```

---

# 12. Training 전 ROI-pair bundle index

각 subject GT tractogram에서:

```text
streamline endpoint
↓
ROI_i, ROI_j
↓
canonical pair:
(min(i,j), max(i,j))
```

로 그룹화.

결과:

```text
Bundle(i,j)
=
{s1, s2, ..., sN}
```

각 ROI-pair bundle마다 저장:

```text
subject_id
roi_i
roi_j
N_streamlines
SC_GT
tract_length_GT
anatomy_type
```

---

# 13. Anatomy type metadata

각 pair를:

```text
CTX-CTX
CTX-SUB
SUB-SUB
```

로 tagging.

이 값은:

```text
training batch를 9개 case로 강제 분할
```

하기 위한 것이 아니라,

```text
SC detailed loss
+
validation
```

에 사용한다.

---

# 14. Bundle imbalance statistics를 먼저 계산

training subjects에서:

```text
N_ij = number of streamlines in bundle(i,j)
```

distribution을 계산.

반드시 저장:

```text
min
Q1
median
Q3
95th percentile
max
histogram
```

---

# 15. 목표는 모든 bundle을 최대 bundle 크기로 만드는 것이 아니다

예:

```text
largest bundle = 50,000
```

이라고 해서 모든 bundle을:

```text
50,000
```

까지 생성하면 안 된다.

그것은:

```text
synthetic domination
compute 폭증
tiny/noisy bundle 과대증폭
```

을 일으킬 수 있다.

---

# 16. Bundle training exposure target

추천 개념:

```text
B_target
```

를 둔다.

예:

```text
B_target =
training positive bundle count distribution의
median 또는 Q3 기반
```

정확한 값은 validation으로 선정.

핵심:

```text
N_ij >> B_target
→ downsample

N_ij ≈ B_target
→ 그대로

N_ij < B_target
→ GESTA augmentation
```

---

# 17. 권장 adaptive target

완전 동일 target보다 다음이 더 안전하다.

```text
B_ij =
clip(
    B_base,
    lower = B_min,
    upper = B_max
)
```

또는 bundle size를 완전히 무시하지 않고:

```text
B_ij ∝ N_ij^α
```

with:

```text
0 < α < 1
```

를 사용할 수 있다.

시작 hyperparameter 후보:

```text
α = 0.3 ~ 0.5
```

이는 본 프로젝트의 tuning proposal이며
TractoLearn 원 코드의 고정 규칙은 아니다.

---

# 18. Tiny/noisy bundle gate

매우 작은 bundle에 KDE를 바로 fitting하면 위험하다.

예:

```text
N = 1
N = 2
N = 5
```

latent density 자체가 신뢰하기 어렵다.

따라서:

```text
if N_ij < min_seed_count:
    KDE augmentation 금지 또는 fallback
```

를 둔다.

---

# 19. Tiny bundle fallback

선택지:

```text
A. 실제 streamline replacement sampling
B. ATM pair-conditioned N(0,I) prior 사용
C. training subjects의 동일 ROI-pair pooled latent prior
D. anatomy-type pooled prior
```

우선순위 추천:

```text
same ROI-pair across training subjects
→ pair-conditioned ATM prior
→ anatomy-type prior
```

---

# 20. Dataset leakage 방지

pooled latent를 만들 때:

```text
validation subject
test subject
```

의 GT streamline을 절대 사용하지 않는다.

즉:

```text
TRAIN GT only
↓
latent augmentation prior
```

이어야 한다.

---

# 21. Training augmentation — 핵심 과정

under-represented bundle:

```text
GT ROI-pair bundle
↓
128-point resampling
↓
ATM ES
↓
latent seeds Z_ij
↓
TractoLearn RejectionSampler
↓
Z_synthetic
↓
ATM DS
+
subject T1 anatomy
+
ROI-pair condition
↓
synthetic complete streamlines
↓
T1/atlas/geometry filtering
↓
accepted augmented bundle
```

---

# 22. TractoLearn RejectionSampler 적용

권장 adapter:

```python
sampler = RejectionSampler(
    data=seed_z,
    kde_bw=...,
    kde_bw_factor=...,
    proposal_distribution_name="multivariate_normal",
    scaling_mode="max",
)
```

그리고:

```python
z_new, n_trials, elapsed = sampler.sample(
    nb_samples=n_needed,
    batch_size=...
)
```

개념을 사용.

---

# 23. Gaussian proposal부터 시작

TractoLearn은:

```text
multivariate_normal
GMM
```

proposal을 지원한다.

첫 baseline:

```text
multivariate_normal
```

추천.

이유:

```text
구현 단순
seed가 적은 bundle에 상대적으로 안정
parameter 수 적음
```

---

# 24. GMM은 ablation으로

seed가 충분한 bundle에서는:

```text
GMM
```

이 multi-modal latent structure를 더 잘 표현할 가능성이 있다.

하지만 작은 bundle에서:

```text
GMM component 과다
```

는 불안정하다.

따라서:

```text
Gaussian proposal
vs
GMM proposal
```

은 ablation으로 비교.

---

# 25. KDE bandwidth

TractoLearn 코드는:

```text
Cross-Validation
Silverman1986
manual bandwidth
```

를 지원한다.

권장:

```text
baseline: Silverman
final: CV vs Silverman 비교
```

이유:

```text
수천 ROI-pair마다 full CV
→ 너무 비쌀 수 있음
```

---

# 26. Rejection acceptance rate를 반드시 기록

TractoLearn의 sampler는:

```text
accepted / total trials
```

개념을 기록한다.

우리도 bundle별:

```text
acceptance_rate
n_trials
sampling_time
```

을 저장.

---

# 27. Acceptance rate가 매우 낮은 bundle

예:

```text
acceptance < threshold
```

라면:

```text
latent density fitting 불안정
proposal 부적절
seed insufficient
```

가능성이 있다.

이 경우:

```text
fallback sampler
```

로 전환.

---

# 28. Synthetic streamline filter

T1-only 조건에서 반드시 확인:

```text
1. target ROI_i ↔ ROI_j endpoint 일치
2. brain mask 밖으로 크게 이탈하지 않음
3. WM occupancy
4. minimum length
5. maximum length
6. extreme curvature 제외
7. NaN / Inf 없음
8. duplicate 또는 near-duplicate 과다 없음
```

---

# 29. TractoLearn filter 중 그대로 사용 가능한 것

개념적으로 재사용 가능:

```text
filter_grid_roi()
cut_streamlines_outside_mask()
length filtering
WM occupancy 관련 검사
```

---

# 30. 그대로 사용하면 안 되는 filter

최종 inference에서 dMRI가 없으므로:

```text
FA-dependent mandatory filter
FODF peak orientation filter
```

를 핵심 생성 조건으로 두면 안 된다.

연구용 training QC로 참고할 수는 있지만
T1-only inference requirement와 분리해야 한다.

---

# 31. 가장 중요한 원칙 — augmentation count ≠ SC strength

예:

```text
GT SC weak edge = 10
```

그 bundle의 geometry를 학습하기 위해:

```text
synthetic 200 streamline
```

을 만들었다고 해서:

```text
SC target = 200
```

으로 바꾸면 안 된다.

---

# 32. GT SC는 절대 augmentation으로 변경하지 않는다

항상:

```text
GT SC matrix
=
원본 subject GT
```

유지.

Synthetic streamline은:

```text
geometry 학습 노출량
```

을 조절하는 데이터일 뿐이다.

---

# 33. Weight Head 역할

최종 생성 candidate 수와
SC strength를 분리하기 위해:

```text
w_k >= 0
```

를 예측.

예:

```text
SC_pred(i,j)
=
Σ_k w_k C_k(i,j)
```

---

# 34. 따라서 두 레벨을 분리

```text
Bundle balancing
→ streamline geometry representation

SC supervision
→ connection strength representation
```

이 두 개를 섞지 않는다.

---

# 35. ROI-pair balanced batch sampler

GESTA augmentation만으로는 충분하지 않다.

augmentation 후에도 전체 synthetic pool에서
random streamline sampling하면:

```text
큰 bundle 우세
```

가 다시 생길 수 있다.

따라서 batch의 sampling unit은:

```text
streamline
```

이 아니라:

```text
ROI pair
```

이어야 한다.

---

# 36. 권장 batch 생성

```text
Step 1
subject 선택

Step 2
positive ROI pairs 선택

Step 3
각 pair에서 fixed N_train streamline 선택

Step 4
부족하면 real + synthetic pool 사용
```

예:

```text
24 ROI pairs / batch
16 streamlines / pair
```

이면:

```text
384 streamlines
```

이다.

수치는 hyperparameter.

---

# 37. Real / Synthetic mixing

small bundle을 synthetic만으로 채우지 않는다.

예:

```text
real fraction >= 0.5
```

같은 minimum real ratio를 둘 수 있다.

예:

```text
16 / pair

real 8
synthetic 8
```

단 실제 seed가 적으면 유연하게 조정.

---

# 38. Synthetic domination 방지

각 bundle마다:

```text
max_synthetic_ratio
```

를 둔다.

예:

```text
synthetic : real <= 4 : 1
```

같은 cap을 validation starting point로 둘 수 있다.

정확한 값은 실험으로 결정.

---

# 39. Epoch 간 다양성

가능하면 모든 synthetic streamline을
한 번 offline 생성해 고정하기보다:

```text
latent samples를 epoch/refresh 주기마다 일부 재생성
```

하면 geometry diversity가 증가할 수 있다.

단 compute가 크면:

```text
pre-generate synthetic cache
+
periodic refresh
```

전략 사용.

---

# 40. 권장 2-stage implementation

## Stage A — offline augmentation cache

먼저 구현:

```text
TRAIN GT bundles
↓
ES
↓
RejectionSampler
↓
DS
↓
filter
↓
synthetic bundle cache
```

장점:

```text
debug 쉬움
재현성 높음
training 속도 안정
```

---

## Stage B — optional online refresh

baseline 검증 후:

```text
epoch N마다
under-represented bundle synthetic pool refresh
```

추가 가능.

---

# 41. SC detailed supervision

augmentation과 별도로
SC는 다음 4개 correlation을 계산한다.

```text
r_all
r_CC
r_CS
r_SS
```

---

# 42. Global SC correlation

```text
r_all =
corr(
    vector(SC_pred upper triangle),
    vector(SC_GT upper triangle)
)
```

Loss:

```text
L_global = 1 - r_all
```

---

# 43. CTX-CTX

```text
r_CC =
corr(
    SC_pred[CTX-CTX],
    SC_GT[CTX-CTX]
)
```

---

# 44. CTX-SUB

```text
r_CS =
corr(
    SC_pred[CTX-SUB],
    SC_GT[CTX-SUB]
)
```

---

# 45. SUB-SUB

```text
r_SS =
corr(
    SC_pred[SUB-SUB],
    SC_GT[SUB-SUB]
)
```

---

# 46. Type-aware correlation loss

추천:

```text
L_type =
(
  (1-r_CC)
+ (1-r_CS)
+ (1-r_SS)
) / 3
```

최종:

```text
L_SC_corr =
λ_global * (1-r_all)
+
λ_type * L_type
```

---

# 47. 왜 9개 case correlation을 사용하지 않는가?

다음까지 학습 loss로 만들면:

```text
CTX-CTX Small
CTX-CTX Medium
CTX-CTX Large
...
```

총 9개.

문제:

```text
일부 cell 표본 부족
correlation 불안정
loss 복잡도 증가
overfitting 위험
```

따라서 training loss는:

```text
Overall
CC
CS
SS
```

까지만 사용.

---

# 48. SC magnitude loss

큰 edge가 MAE를 독점하지 않도록:

```text
L_SC_mag =
MAE(
    log(SC_pred + eps),
    log(SC_GT + eps)
)
```

사용.

---

# 49. tract-length loss

```text
L_length
```

도 유지.

최종 목표:

```text
SC weight
+
SC tract-length
```

모두 subject-specific 재현.

---

# 50. 최종 SC loss

```text
L_SC =
λ_global * L_global
+
λ_type   * L_type
+
λ_mag    * L_SC_mag
+
λ_len    * L_length
```

---

# 51. 전체 모델 loss

```text
L_total =
L_recon
+ λ_KL       L_KL
+ λ_geom     L_geometry
+ λ_endpoint L_endpoint
+ λ_edge     L_edge
+ L_SC
```

---

# 52. 권장 training sequence

## Phase 0

```text
기존 ATM reproduction
```

---

## Phase 1

```text
ROI-pair bundle preprocessing
+
bundle size distribution 분석
```

---

## Phase 2

```text
ATM ES / DS geometry domain adaptation
```

---

## Phase 3

```text
TractoLearn RejectionSampler adapter 구현
```

---

## Phase 4

```text
under-represented bundle augmentation cache 생성
```

---

## Phase 5

```text
ROI-pair balanced geometry training
real + synthetic
```

---

## Phase 6

```text
T1 EA fine-tuning
+
endpoint
+
edge head
```

---

## Phase 7

```text
Overall + CC + CS + SS SC correlation
```

---

## Phase 8

```text
log magnitude
+
Weight Head
```

---

## Phase 9

```text
tract-length
+
joint fine-tuning
```

---

# 53. Validation에서는 GT 기반 GESTA augmentation 금지

validation/test subject에서:

```text
GT streamline
→ GESTA seed
```

를 사용하면 inference leakage.

따라서 validation은 두 종류를 구분.

---

# 54. Validation A — model accuracy

입력:

```text
T1 only
```

결과:

```text
TRK
SC
length
```

평가.

GT는 오직 metric 계산에만 사용.

---

# 55. Validation B — augmentation quality

TRAIN split 내부에서만:

```text
real bundle
vs
GESTA-augmented bundle
```

geometry QC를 별도 수행 가능.

---

# 56. 최종 inference에서 GESTA-style completion을 사용할 경우

GT seed가 없으므로:

```text
T1
↓
ATM initial ROI-pair streamlines
↓
ES
↓
TractoLearn RejectionSampler
↓
DS
↓
additional candidates
```

로 해야 한다.

즉:

```text
training augmentation seed = TRAIN GT bundle

inference completion seed = ATM predicted initial bundle
```

로 엄격히 분리.

---

# 57. Validation 기본 metrics

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

---

# 58. SC strength diagnostic

training loss case로 만들지 않고
validation에서:

```text
Small
Medium
Large
```

확인.

목적:

```text
weak edge가 사라지는가?
strong edge만 잘 맞는가?
```

---

# 59. Bundle-size diagnostic

GT bundle size 기준으로:

```text
low-count
mid-count
high-count
```

를 나눠 validation.

여기서는 SC strength와 별개로:

```text
원래 streamline이 적은 bundle이
실제로 개선됐는가?
```

를 확인.

---

# 60. Bundle geometry metrics

가능한 평가:

```text
endpoint pair accuracy
valid streamline ratio
length distribution error
bundle voxel coverage
overreach
coverage-to-overreach ratio
duplicate ratio
```

---

# 61. 가장 중요한 비교 실험

## Sampling / augmentation ablation

```text
A. Original unbalanced
B. ROI-pair balanced only
C. B + duplicate oversampling
D. B + TractoLearn Gaussian/KDE augmentation
E. B + TractoLearn RejectionSampler
F. E + filtering
```

핵심 비교:

```text
C vs D/E
```

단순 duplicate보다
latent generative augmentation이 실제로 도움이 되는지 검증.

---

# 62. SC supervision ablation

```text
A. Overall corr
B. Overall + CC/CS/SS
C. B + log magnitude
D. C + length
```

---

# 63. TractoLearn sampler ablation

```text
Gaussian direct sampling
vs
KDE direct sampling
vs
KDE + rejection sampling
vs
KDE + GMM proposal + rejection sampling
```

---

# 64. 반드시 기록할 training 로그

bundle마다:

```text
N_real
N_synthetic
N_used
augmentation_ratio
KDE bandwidth
proposal type
acceptance rate
sampling trials
filter pass rate
mean length
endpoint pass rate
```

---

# 65. Batch imbalance 로그

매 epoch:

```text
ROI-pair별 sampling count
bundle-size bin별 exposure
SC-strength bin별 exposure
CC/CS/SS exposure
```

를 기록.

---

# 66. 중요한 판단

목표는:

```text
CC = CS = SS
Small = Medium = Large
```

를 정확히 1:1:1로 만드는 것이 아니다.

목표는:

```text
특정 large bundle이 training gradient를 독점하지 못하게 함
+
under-represented bundle에도 충분한 학습 기회 제공
```

이다.

---

# 67. Too-small bundle 보호

GESTA augmentation이 오히려 noise를 증폭할 수 있다.

따라서:

```text
min_seed_count
min_reproducibility
max_synthetic_ratio
```

를 둔다.

---

# 68. Subject reproducibility 옵션

특정 ROI pair가:

```text
training subjects 대부분에서 존재
```

하지만 한 subject에서 count가 작다면:

```text
small but likely real
```

일 가능성이 있다.

반대로:

```text
한 subject에서 1~2 streamline만 존재
```

하면 false-positive 가능성이 있다.

추후:

```text
group prevalence
```

를 augmentation priority에 포함 가능.

---

# 69. 권장 augmentation priority score

선택적 확장:

```text
Priority_ij =
underrepresentation
×
group reproducibility
×
edge confidence
```

즉:

```text
적지만 반복적으로 존재하는 connection
```

을 먼저 보호.

이는 본 프로젝트 제안이며
TractoLearn 원 코드 자체의 기능은 아니다.

---

# 70. 코드 구조 권장

예:

```text
src/atm_sc/
├── data/
│   ├── roi_pair_dataset.py
│   ├── bundle_statistics.py
│   └── balanced_pair_sampler.py
│
├── generative/
│   ├── tractolearn_adapter.py
│   ├── latent_sampler.py
│   └── bundle_augmenter.py
│
├── filtering/
│   └── t1_streamline_filter.py
│
├── models/
│   ├── anatomy_encoder.py
│   ├── streamline_vae.py
│   ├── edge_head.py
│   └── weight_head.py
│
└── training/
    ├── losses.py
    ├── trainer.py
    └── validation.py
```

---

# 71. `tractolearn_adapter.py`

역할:

```text
TractoLearn RejectionSampler 호출 interface
```

ATM 코드 전체에 TractoLearn dependency를 퍼뜨리지 않는다.

예:

```python
sample_latents(
    seed_z,
    n_samples,
    proposal="multivariate_normal",
    use_rejection=True,
    bandwidth_mode="silverman",
)
```

---

# 72. `bundle_augmenter.py`

역할:

```text
GT bundle
↓
ES encode
↓
latent sampling
↓
DS decode
↓
filter
↓
synthetic cache
```

---

# 73. `balanced_pair_sampler.py`

역할:

```text
ROI pair first
↓
fixed number of streamlines per pair
↓
real/synthetic mixing
```

GESTA와 별개의 필수 모듈.

---

# 74. 권장 config 예

```yaml
bundle_balance:
  enabled: true
  sampling_unit: roi_pair
  streamlines_per_pair: 16

  target_mode: quantile
  target_quantile: 0.50

  min_seed_count: 20
  max_synthetic_ratio: 4.0

  real_fraction_min: 0.50

tractolearn_gesta:
  enabled: true
  proposal: multivariate_normal
  use_rejection_sampling: true

  bandwidth_mode: silverman
  kde_bw_factor: 1.0

  scaling_mode: max
  allow_singular: false

  max_seeds: 256
  sampling_batch_size: 1024

  gmm:
    enabled: false
    n_components: 4

filter:
  endpoint: true
  brain_mask: true
  wm_occupancy: true
  min_length_mm: 20
  max_length_mm: 220

sc_loss:
  global_corr: true
  type_corr: true
  log_magnitude: true
  tract_length: true
```

숫자는 starting proposal.
최종 값은 validation으로 결정.

---

# 75. Smoke Test 1 — RejectionSampler adapter

Synthetic latent:

```text
N = 100
D = 64
```

입력.

확인:

- [ ] required sample count 반환
- [ ] NaN 없음
- [ ] shape 정확
- [ ] reproducible seed
- [ ] acceptance rate > 0
- [ ] runtime 로그

---

# 76. Smoke Test 2 — Small bundle augmentation

예:

```text
real = 20
target = 100
```

확인:

- [ ] synthetic 80 생성 가능
- [ ] decoded shape `[80,128,3]`
- [ ] endpoint pass
- [ ] length pass
- [ ] finite coordinates

---

# 77. Smoke Test 3 — Large bundle downsampling

예:

```text
real = 5000
training target = 100
```

확인:

- [ ] 5000개 모두 loss에 들어가지 않음
- [ ] pair당 training exposure 고정
- [ ] epoch 간 random subset 변화

---

# 78. Smoke Test 4 — Batch balance

가상 데이터:

```text
Bundle A = 5000
Bundle B = 500
Bundle C = 20
```

1000 training steps 후:

```text
pair exposure A/B/C
```

가 raw count 250:25:1 비율이 되지 않는지 확인.

---

# 79. Smoke Test 5 — SC target invariance

augmentation 전/후:

```text
GT SC matrix
```

가 byte-wise 또는 numeric-wise 동일해야 한다.

- [ ] augmentation이 SC_GT를 수정하지 않음
- [ ] synthetic count가 GT weight로 사용되지 않음

---

# 80. Smoke Test 6 — SC detailed correlation

synthetic matrix에서:

```text
overall
CC
CS
SS
```

correlation이 각각 정확히 계산되는지 unit test.

---

# 81. Training Checklist

## Data

- [ ] Subject-level split 완료
- [ ] ROI-pair bundle index는 TRAIN에서만 augmentation용 생성
- [ ] Validation/Test GT는 augmentation seed로 사용하지 않음
- [ ] GT SC는 augmentation과 독립

## Bundle statistics

- [ ] N_streamline histogram
- [ ] quantiles
- [ ] pair prevalence
- [ ] CC/CS/SS count
- [ ] SC strength distribution

## TractoLearn

- [ ] RejectionSampler adapter
- [ ] KDE bandwidth
- [ ] Gaussian proposal
- [ ] GMM optional
- [ ] acceptance rate logging
- [ ] max sampling trials
- [ ] reproducible RNG

## Augmentation

- [ ] min seed gate
- [ ] max synthetic ratio
- [ ] real/synthetic mixture
- [ ] decoded geometry QC
- [ ] endpoint filter
- [ ] length filter
- [ ] WM/brain filter
- [ ] duplicate control

## Batch sampler

- [ ] sampling unit = ROI pair
- [ ] fixed streamline count / pair
- [ ] large bundle downsampling
- [ ] small bundle augmentation
- [ ] no raw-count-proportional exposure

## SC loss

- [ ] Overall corr
- [ ] CTX-CTX corr
- [ ] CTX-SUB corr
- [ ] SUB-SUB corr
- [ ] log magnitude
- [ ] tract length
- [ ] pass-SC definition

---

# 82. Validation Checklist

## Overall

- [ ] SC Pearson
- [ ] SC Spearman
- [ ] SC CCC
- [ ] MAE
- [ ] RMSE

## Anatomy type

- [ ] CTX-CTX
- [ ] CTX-SUB
- [ ] SUB-SUB

## Strength diagnostic

- [ ] Small
- [ ] Medium
- [ ] Large
- [ ] weak-edge recall

## Bundle-size diagnostic

- [ ] low-count bundle
- [ ] mid-count bundle
- [ ] high-count bundle

## Geometry

- [ ] endpoint accuracy
- [ ] valid ratio
- [ ] length distribution
- [ ] coverage
- [ ] overreach
- [ ] duplicate ratio

---

# 83. Bias Prevention PASS Criteria

다음이 만족되어야
“training bundle-size bias가 통제됐다”고 판단.

- [ ] large bundle이 raw streamline count 비율대로 더 자주 학습되지 않음
- [ ] low-count bundle training exposure가 baseline보다 증가
- [ ] 단순 duplicate oversampling보다 GESTA augmentation이 geometry diversity 개선
- [ ] low-count bundle endpoint accuracy가 개선 또는 유지
- [ ] low-count bundle validation error 감소
- [ ] high-count bundle 성능 심각한 저하 없음
- [ ] Overall SC correlation 유지 또는 개선
- [ ] CTX-CTX 성능 유지 또는 개선
- [ ] CTX-SUB 성능 유지 또는 개선
- [ ] SUB-SUB 성능 유지 또는 개선
- [ ] weak-edge recall 악화 없음
- [ ] SC magnitude error 악화 없음
- [ ] tract-length error 악화 없음
- [ ] validation/test inference는 T1 only

---

# 84. 반드시 실패로 보는 경우

다음 중 하나면 GESTA balancing을 그대로 채택하지 않는다.

```text
bundle coverage ↑
BUT
SC error ↑↑
```

또는:

```text
small bundle performance ↑
BUT
large/overall 성능 크게 ↓
```

또는:

```text
synthetic ratio 과다
→ training은 좋음
→ validation 악화
```

또는:

```text
tiny/noisy bundle이 과대 증폭
```

---

# 85. 가장 중요한 Ablation Table

| Model | Pair-balanced | GESTA Aug | Rejection | Type SC Corr | Log Mag |
|---|---:|---:|---:|---:|---:|
| A Baseline | No | No | No | No | No |
| B | Yes | No | No | No | No |
| C | Yes | duplicate only | No | No | No |
| D | Yes | Yes | No/Gaussian | No | No |
| E | Yes | Yes | Yes | No | No |
| F | Yes | Yes | Yes | Yes | No |
| G Final | Yes | Yes | Yes | Yes | Yes |

핵심 질문:

```text
B vs E
→ GESTA sampling 자체의 효과

C vs E
→ 단순 oversampling 대비 generative augmentation의 효과

E vs F
→ detailed SC corr 효과

F vs G
→ log magnitude 효과
```

---

# 86. Source-derived vs Project-specific 구분

## TractoLearn에서 직접 확인되는 기능

```text
RejectionSampler
KDE / KernelDensity
Cross-Validation bandwidth
Silverman bandwidth
multivariate normal proposal
GMM proposal
max / 3rd_quartile scaling
batch rejection sampling
per-bundle desired generation count
per-bundle max sampling
max seeds
Parzen bandwidth
length constraints
WM / FA / peaks filtering pipeline
```

## 본 프로젝트에서 추가하는 전략

```text
ROI-pair balanced sampler
ATM ES/DS를 TractoLearn sampler와 연결
GT bundle 기반 TRAIN augmentation
ATM prediction 기반 inference completion
fixed streamline exposure per ROI pair
GT SC target invariance
Overall + CC + CS + SS SC loss
Weight Head로 candidate count와 SC strength 분리
T1-only filtering subset
```

둘을 논문 Methods에서 명확하게 구분해야 한다.

---

# 87. License Note

`tractolearn` repository는 permissive MIT/BSD license가 아니라
SCIL의 별도 End User License Agreement를 사용한다.

확인되는 조건에는:

```text
academic / research use 허용
commercial use 제한
```

이 포함된다.

따라서:

```text
직접 코드 port / 재사용 시 LICENSE 확인
TractoLearn + GESTA + FIESTA citation
원 코드가 포함되면 license notice 유지
```

가 필요하다.

---

# 88. 논문 Methods에서 추천 표현

예:

> To mitigate bundle-size imbalance, we adopted the latent-space generative
> sampling strategy implemented in TractoLearn/gesta. Under-represented
> ROI-pair bundles were encoded into the ATM streamline latent space, and
> additional latent samples were generated using KDE-based rejection sampling.
> The generated latent samples were decoded by the anatomy- and ROI-pair-
> conditioned ATM decoder. Importantly, synthetic streamline count was used
> only to balance geometric training exposure and did not alter the empirical
> SC target.

그리고 SC:

> Whole-brain SC similarity was supervised globally and separately for
> cortico-cortical, cortico-subcortical, and subcortico-subcortical edges to
> prevent the numerically dominant edge class from masking poorer performance
> in less represented connection types.

---

# 89. 최종 파이프라인

```text
                         TRAINING

TRAIN T1 + GT TRK + Atlas + GT SC
                │
                ↓
       ROI-pair bundle index
                │
        ┌───────┴────────┐
        ↓                ↓
  High-count         Low-count
   bundle              bundle
        │                │
  downsample             ↓
        │             ATM ES
        │                ↓
        │          latent seeds
        │                ↓
        │     TractoLearn RejectionSampler
        │       KDE + Gaussian/GMM
        │                ↓
        │          synthetic z
        │                ↓
        │             ATM DS
        │                ↓
        │       synthetic streamlines
        │                ↓
        │        T1/atlas filtering
        │                │
        └────────┬───────┘
                 ↓
       Balanced ROI-pair pool
                 ↓
     Fixed N streamline / pair
                 ↓
          Geometry training
                 ↓
      T1-conditioned ATM model
                 ↓
        Whole-brain prediction
                 ↓
          Weight Head
                 ↓
     Differentiable PASS-SC
                 ↓
   ┌─────────────┼────────────────┐
   ↓             ↓                ↓
Overall Corr   CC/CS/SS       Log Magnitude
                                  +
                               Length
```

---

# 90. 최종 결론

이 전략의 메인 아이디어는:

```text
“작은 bundle을 별도 모델로 만든다”
```

가 아니다.

또한:

```text
“Small / Medium / Large를 무조건 1:1:1 case로 나눈다”
```

도 아니다.

핵심은:

```text
1. ROI pair를 training sampling unit으로 사용
2. 큰 bundle은 downsample하여 gradient 독점 방지
3. 작은 bundle은 TractoLearn GESTA latent sampling으로
   단순 duplicate가 아닌 synthetic geometry를 보완
4. pair당 training exposure를 제한/균형화
5. augmentation 개수와 실제 SC strength는 분리
6. SC는 Overall + CC + CS + SS로 세밀하게 supervision
7. validation에서 low/mid/high bundle과 weak/medium/strong SC를 별도 평가
```

이다.

따라서 최종 모델은:

```text
large bundle만 잘 만드는 ATM
```

이 아니라:

```text
under-represented ROI-pair bundle까지 충분히 학습하면서
whole-brain SC topology와 strength를 동시에 재현하는
balanced T1-only tractogram generator
```

를 목표로 한다.
