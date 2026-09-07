# ATM + GESTA Synthetic Streamline Generation & QC Pipeline
## 현재 T1-only → Whole-brain TRK → SC Weight / Length 파이프라인용

## 1. 목적

GESTA synthetic streamline/segment를 사용하는 목적은 **SC 값을 인위적으로 늘리는 것**이 아니라,

```text
가닥 수가 적은 bundle / SC edge
→ 학습 geometry 다양성 부족
→ large bundle 중심 학습 편향
```

을 줄이기 위한 **training augmentation**이다.

따라서 반드시 다음 세 가지를 분리한다.

```text
1. 언제 synthetic을 생성할 것인가?
2. 몇 개를 생성할 것인가?
3. 생성된 synthetic이 realistic한가?
```

중요:

```text
Synthetic sample 수 ≠ GT SC edge 값
```

예:

```text
GT SC[C,D] = 30
GESTA synthetic C-D = 100개
```

를 training에 추가해도:

```text
GT SC[C,D] = 30
```

은 그대로 유지한다.

---

# 2. 현재 프로젝트에서 Synthetic의 단위

현재 GT SC는 하나의 streamline이 **지나간 모든 ROI pair**를 count하는 PASS-SC 방식이다.

예:

```text
A → C → D → B
```

이면 해당 streamline은 다음 SC edge에 기여할 수 있다.

```text
A-C
A-D
A-B
C-D
C-B
D-B
```

따라서 synthetic augmentation 대상은 두 종류로 생각한다.

## A. Full streamline

```text
A → C → D → B
```

목적:
- 전체 tract geometry
- endpoint
- route
- whole-brain TRK realism

## B. SC edge-aligned segment

예:

```text
C-D sub-path
A-D sub-path
```

목적:
- PASS-SC edge 직접 supervision
- 작은 edge의 geometry 부족 보완
- SUB-SUB pass-only edge 보완

---

# 3. Synthetic 생성 전체 흐름

```text
Real streamline / segment
        ↓
ATM Streamline Encoder (ES)
        ↓
Real latent seeds
        ↓
KDE / latent bank
        ↓
GESTA-style latent sampling
        ↓
Rejection Sampling
        ↓
Accepted latent z_syn
        ↓
ATM Streamline Decoder (DS)
        ↓
Synthetic streamline / segment
        ↓
Synthetic Realism QC
        ↓
QC PASS only
        ↓
Balanced training pool
```

---

# 4. 언제 Synthetic을 생성할 것인가?

단순히:

```text
if count < 100:
    generate synthetic
```

처럼 하나의 arbitrary threshold만 사용하는 것은 권장하지 않는다.

Synthetic generation 여부는 **TRAIN set의 실제 bundle/edge 분포**를 기준으로 결정한다.

---

# 5. 추천 Generation Eligibility

각 edge/bundle의 real training sample 수를:

```text
R_e
```

라고 한다.

먼저 TRAIN positive edge들의 sample-count distribution을 구한다.

예:

```text
Q25 = TRAIN bundle size distribution의 25 percentile
```

그 다음:

```text
R_e < Q25
```

인 edge/bundle을 synthetic augmentation 후보로 둔다.

즉:

```text
Large bundle
→ synthetic 불필요

Medium bundle
→ 필요 시 소량

Small / under-represented bundle
→ GESTA augmentation
```

---

# 6. 현재 프로젝트에 맞는 추천 기준

현재 파이프라인은 이미:

```text
bundle exposure ∝ sqrt(bundle size)
```

를 사용하고 있으므로 GESTA도 이 balancing 전략과 연결한다.

권장:

```text
Synthetic eligibility:
TRAIN sample count가 하위 quantile에 위치

Synthetic target:
현재 sqrt exposure target을 만족할 정도까지만 증가
```

즉 모든 작은 bundle을 가장 큰 bundle 크기까지 채우지 않는다.

---

# 7. 몇 개를 생성할 것인가?

각 edge에 대해 원하는 training diversity/exposure target을:

```text
T_e
```

라 두면:

```text
N_syn(e) = max(0, T_e - R_e)
```

로 생각할 수 있다.

단 반드시 cap을 둔다.

예:

```text
N_syn(e) <= 4 × R_e
```

현재 프로젝트의 안전장치:

```text
Real sample contribution ≥ 50 %
Synthetic ≤ 4 × real
Synthetic training weight = 0.5
```

는 유지 가능한 초기 기준이다.

---

# 8. 너무 작은 Bundle은 어떻게 처리할 것인가?

Real seed가 너무 적으면 edge-specific KDE 자체가 불안정할 수 있다.

예:

```text
C-D real streamline = 3개
```

만 가지고 KDE를 만들면 실제 C-D distribution을 잘 대표한다고 보기 어렵다.

따라서 최소 seed 수:

```text
N_min_seed
```

를 둔다.

## Seed 충분

```text
R_e >= N_min_seed
→ edge-specific KDE
```

## Seed 부족

```text
R_e < N_min_seed
→ same-edge TRAIN subjects pooled latent
→ 또는 CC / CS / SS pooled prior
→ 또는 pair-conditional prior
```

로 fallback한다.

`N_min_seed`는 고정된 과학적 상수가 아니다.

추천:

```text
10 / 20 / 30 / 50
```

등을 validation에서 비교하여:
- KDE 안정성
- synthetic acceptance rate
- coverage
- overreach
- downstream SC/TRK 성능

을 보고 결정한다.

---

# 9. GESTA의 Rejection Sampling

GESTA는 단순히 Gaussian noise를 streamline 좌표에 더하는 방식이 아니다.

개념:

```text
Real streamline
↓
Encoder
↓
latent seed z_real
↓
KDE로 p(z) 추정
↓
Gaussian/GMM proposal q(z)에서 candidate 생성
↓
Rejection Sampling
↓
p(z) 기준으로 plausible latent 선택
↓
Decoder
↓
Synthetic streamline
```

중요:

```text
GESTA rejection sampling
≠
Synthetic anatomical QC
```

latent가 KDE 분포에서 accepted되었다고 해서
decoded streamline이 자동으로 anatomically realistic한 것은 아니다.

따라서 decoder 이후 별도의 QC가 필요하다.

---

# 10. Synthetic Realism QC 기본 원칙

원 GESTA의 평가 철학을 기반으로 한다.

GESTA의 핵심 개념:

```text
A = Anatomy
D = Direction
G = Geometry
C = Connectivity
```

하지만 현재 프로젝트는 최종 inference가 T1-only이므로
dMRI/fODF 기반 Direction(D)은 inference 기준으로 사용할 수 없다.

따라서 현재 프로젝트에서는:

```text
A + G + C
+
ROI / SC-edge consistency
+
Coverage / Overreach
```

를 핵심으로 사용한다.

---

# 11. Level 1 — 개별 Synthetic Streamline Hard QC

Synthetic candidate가 training pool에 들어가기 전에 반드시 통과해야 하는 단계.

## 11.1 ROI / SC-edge Consistency

### Full streamline

조건된 endpoint ROI pair를 제대로 연결하는지 확인.

예:

```text
condition = A-B
synthetic endpoint = A-B
→ PASS
```

### Edge segment

조건된 SC edge를 실제로 포함/연결하는지 확인.

예:

```text
target edge = C-D
synthetic segment가 C-D를 연결
→ PASS
```

이 항목은 **Hard reject criterion**으로 둔다.

---

## 11.2 Brain / WM Occupancy

streamline point 중 brain/WM mask 내부 비율:

```text
occupancy =
N_inside / N_total
```

을 계산한다.

임의의 고정 threshold보다 TRAIN real streamline distribution을 기준으로 설정한다.

예:

```text
τ_WM = TRAIN real occupancy의 Q1~Q5
```

통과:

```text
occupancy_syn >= τ_WM
```

---

## 11.3 Length

streamline length:

```text
L = Σ ||p[t+1] - p[t]||
```

Synthetic length가 real bundle/edge 분포와 유사한지 확인한다.

권장:

```text
Q1(real) <= L_syn <= Q99(real)
```

또는 더 보수적으로:

```text
Q5(real) <= L_syn <= Q95(real)
```

threshold는 validation에서 결정.

sample이 부족하면:

```text
edge-specific
→ CC/CS/SS pooled
→ global
```

순으로 fallback한다.

---

## 11.4 Curvature / Turning Angle

각 point의 local turning angle을 계산한다.

```text
v1 = p[t] - p[t-1]
v2 = p[t+1] - p[t]

θ_t = angle(v1, v2)
```

Synthetic이 비정상적으로 꺾이지 않는지 확인한다.

권장:

```text
max_angle_syn <= Q99(max_angle_real)
```

또는:

```text
P95(angle_syn) <= Q99(P95(angle_real))
```

현재 synthetic에서 turning-angle failure가 많으므로
**기존 임의 threshold가 너무 강한지 먼저 real TRAIN distribution으로 검증해야 한다.**

---

## 11.5 Winding / Loop

과도한 loop 또는 비정상적인 winding을 검사한다.

권장:

```text
winding_syn <= TRAIN real Q99
```

필요하면:
- self-intersection
- near-loop
- 반복 좌표

등도 추가한다.

---

## 11.6 Coordinate Sanity

필수:

```text
NaN 없음
Inf 없음
zero-length streamline 없음
working-space bbox 밖으로 심하게 이탈하지 않음
```

---

# 12. Level 2 — Bundle / Edge Distribution QC

개별 streamline이 QC를 통과했다고 해서 synthetic bundle 전체가 realistic한 것은 아니다.

따라서 real bundle과 synthetic bundle의 **분포 자체**를 비교한다.

---

## 12.1 Length Distribution

비교:

```text
Real length distribution
vs
Synthetic length distribution
```

추천:
- Wasserstein distance
- KS statistic
- median difference
- IQR ratio

---

## 12.2 Curvature Distribution

추천:
- Wasserstein distance
- KS statistic
- P95 curvature difference

---

## 12.3 Spatial Coverage

Real bundle voxel set:

```text
V_real
```

Synthetic bundle voxel set:

```text
V_syn
```

Coverage:

```text
Coverage =
|V_real ∩ V_syn| / |V_real|
```

높을수록 real bundle 공간을 잘 재현한다.

---

## 12.4 Overreach

Coverage만 높으면 synthetic을 너무 넓게 생성하는 모델도 좋아 보일 수 있다.

따라서:

```text
Overreach =
|V_syn - V_real| / |V_syn|
```

도 반드시 같이 본다.

핵심:

```text
High Coverage
+
Low Overreach
```

---

## 12.5 Dice / Jaccard

```text
Dice =
2|V_real ∩ V_syn| /
(|V_real| + |V_syn|)
```

```text
Jaccard =
|V_real ∩ V_syn| /
|V_real ∪ V_syn|
```

---

## 12.6 Duplicate Ratio

Synthetic가 real streamline을 거의 그대로 복사하는지 확인한다.

각 synthetic streamline과 가장 가까운 real streamline의 distance를 계산한다.

후보:
- MDF
- point-wise RMS
- Hausdorff distance

너무 가까운 synthetic 비율을:

```text
duplicate ratio
```

로 기록한다.

목표:

```text
Real manifold과 유사
BUT
단순 복사는 아님
```

---

## 12.7 Latent-space Plausibility

```text
z_real
vs
z_syn
```

비교:
- MMD
- Energy distance
- nearest-neighbor distance

해석:

```text
너무 멀다
→ out-of-distribution

너무 가깝다
→ memorization / duplicate
```

---

# 13. Level 3 — Downstream Validation

가장 중요한 최종 검증.

다음 조건을 비교한다.

```text
A0. ATM only
A1. ATM + balanced sampling
A2. ATM + GESTA raw
A3. ATM + GESTA + Hard QC
A4. ATM + GESTA + Hard QC + Distribution QC
```

평가는 synthetic가 아니라 **real validation/test subjects**에서 수행한다.

---

# 14. TRK 평가

- endpoint accuracy
- ROI pair accuracy
- route F1
- valid streamline ratio
- length distribution
- coverage
- overreach

---

# 15. SC 평가

- Pearson
- Spearman
- CCC
- MAE
- RMSE

Block:

```text
Overall
CTX-CTX
CTX-SUB
SUB-SUB
```

Strength:

```text
Small
Medium
Large
```

특히:

```text
SUB-SUB endpoint-supported
vs
SUB-SUB pass-only
```

를 분리한다.

---

# 16. Threshold를 어떻게 설정할 것인가?

핵심 원칙:

```text
논문 지표는 가져오되,
threshold는 현재 TRAIN real data에서 결정
```

추천 예:

```text
WM occupancy
→ TRAIN real Q1~Q5 이상

Length
→ TRAIN real Q1~Q99

Curvature / turning angle
→ TRAIN real Q99 이하

Winding
→ TRAIN real Q99 이하

Latent distance
→ TRAIN real-real distribution 범위
```

---

# 17. 단일 Quality Score만 사용하는 것은 권장하지 않음

예:

```text
Q =
w1*WM
+w2*length
+w3*curvature
+w4*ROI
+w5*latent
```

로 하나의 score를 만들 수는 있지만,
이 score 하나로 모든 QC를 대체하는 것은 위험하다.

예:

```text
WM occupancy는 매우 높지만
ROI edge가 완전히 틀림
```

인데 평균 score가 높아서 통과할 수 있기 때문이다.

따라서 추천:

```text
Hard anatomical constraints
+
Soft quality ranking
```

이다.

---

# 18. 추천 Hard QC

Training에 들어가려면 최소한 다음은 모두 만족:

```text
[ ] target ROI / SC edge consistency
[ ] brain/WM occupancy
[ ] plausible length
[ ] plausible curvature
[ ] plausible winding
[ ] finite coordinates
```

---

# 19. 추천 Bundle-level QC

```text
[ ] Coverage 충분
[ ] Overreach 낮음
[ ] Length distribution 유사
[ ] Curvature distribution 유사
[ ] Duplicate ratio 낮음
[ ] Latent distribution plausible
```

---

# 20. 현재 파이프라인에서 반드시 구분해야 할 Count

현재 edge segment는 저장량 때문에 capped될 수 있다.

따라서:

```text
N_GT_edge
=
원본 전체 tractogram에서 계산한 PASS-SC edge count
```

와:

```text
N_cached_segment
=
학습용으로 저장된 segment 수
```

를 절대 혼동하면 안 된다.

## Edge Count Head / SC target

```text
N_GT_edge
```

사용.

## GESTA geometry augmentation eligibility

```text
실제 학습에 노출되는 unique real geometry 수
```

를 기준으로 사용.

---

# 21. Synthetic Generation Logging

각 edge마다 최소 다음을 저장한다.

```text
edge_id
ROI_i
ROI_j
source = self_KDE / latent_bank / pooled
n_real_seed
n_requested
n_candidate
n_latent_accepted
n_decoded
n_roi_pass
n_wm_pass
n_length_pass
n_angle_pass
n_winding_pass
n_final_accept
acceptance_rate
```

---

# 22. 낮은 Synthetic Pass Rate 해석

낮은 acceptance rate는 다음 중 하나일 수 있다.

```text
1. KDE / proposal mismatch
2. latent가 real manifold에서 멂
3. Decoder가 sampled latent에 약함
4. ROI/T1 conditioning이 약함
5. QC threshold가 너무 강함
6. 특정 edge가 실패율을 지배
```

따라서 global pass rate뿐 아니라:

```text
per-edge
per-source
per-failure-reason
```

를 반드시 기록한다.

---

# 23. 최종 권장 Pipeline

```text
TRAIN real edge/bundle distribution 계산
        ↓
Under-represented edge 선택
        ↓
충분한 seed?
   ├─ YES → edge-specific KDE
   └─ NO  → pooled TRAIN latent prior
        ↓
필요한 exposure만큼 synthetic 요청
        ↓
GESTA latent rejection sampling
        ↓
ATM decoder
        ↓
Synthetic streamline / segment
        ↓
Hard Realism QC
        ↓
Bundle Distribution QC
        ↓
QC-passed synthetic only
        ↓
Real + Synthetic balanced training
        ↓
Real validation subjects에서 평가
        ↓
No GESTA / Raw GESTA / QC-GESTA 비교
```

---

# 24. 최종 판단 기준

GESTA 성공 기준은:

```text
Synthetic 때문에 training loss가 내려갔다
```

가 아니다.

최종 기준은:

```text
Synthetic가 real streamline distribution을 보존하면서
unseen real subject에서
TRK geometry + SC reconstruction을 개선했는가?
```

이다.

---

# 25. 핵심 요약

```text
언제 생성?
→ TRAIN에서 under-represented한 bundle/SC edge

몇 개 생성?
→ sqrt exposure target을 채우는 정도
→ synthetic ≤ 4×real 등 cap 유지

어떻게 생성?
→ ES → KDE/GMM → rejection sampling → DS

어떤 synthetic을 사용할까?
→ GESTA latent acceptance 후
   Anatomy + Geometry + Connectivity/ROI QC 통과한 것만

Threshold?
→ 임의 상수보다 TRAIN real distribution percentile 기반

Realistic한지 최종 확인?
→ Coverage↑ + Overreach↓
→ Length/curvature 분포 유사
→ duplicate 낮음
→ real validation subject에서 TRK와 SC 동시 개선
```
