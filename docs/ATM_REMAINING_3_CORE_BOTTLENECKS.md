# ATM 현재 남은 핵심 병목 3개
## T1-only Whole-brain Tractography / SC Reconstruction

현재 파이프라인은 전처리, PASS-SC 정의, route/segment supervision, balanced sampling, GESTA augmentation까지 대부분 구현되어 있다.

현재 성능을 제한하는 핵심 병목은 아래 3개로 정리한다.

---

# 1. T1 Anatomy Feature가 Subject-specific 정보를 거의 사용하지 못함

## 현재 문제

현재 모델은:

```text
T1
↓
UNet Encoder
↓
anatomy feature a [512]
↓
ROI-pair embedding과 결합
↓
streamline / SC 생성
```

구조이지만, 실제 측정에서는 T1을 바꾸어도 예측이 거의 변하지 않는다.

```text
자기 T1   → SC r ≈ 0.8108
남의 T1   → SC r ≈ 0.8101
T1 = 0    → SC r ≈ 0.8067
```

즉 현재 count/edge prediction은 사실상:

```text
ROI pair ID
→ TRAIN population 평균 연결 패턴
```

을 외우는 쪽에 가깝다.

## 원인

### 1) Feature scale 불균형

```text
anatomy feature L2 ≈ 0.069
pair embedding L2 ≈ 1.521
```

ROI-pair feature가 anatomy보다 약 22배 크다.

따라서 optimizer가 T1보다 ROI-pair embedding을 사용하는 것이 훨씬 쉽다.

### 2) Global anatomy vector 하나만 사용

현재 subject당 anatomy vector가 하나이고,
이 동일한 512차원 벡터가 3,321개 ROI pair 모두에 들어간다.

```text
Subject anatomy a
   ├─ A-B
   ├─ A-C
   ├─ A-D
   └─ ...
```

따라서 특정 ROI pair 주변의 국소 해부학적 차이를 직접 표현하기 어렵다.

## 결과

현재 생성 SC의 subject 간 상관:

```text
GT subject끼리       ≈ 0.897
생성 subject끼리     ≈ 0.9995
```

즉 모델 출력의 개인차가 거의 없다.

## 우선 개선안

### 1차
- anatomy feature LayerNorm
- anatomy/pair feature scale balancing
- T1 encoder learning rate 재조정
- T1 shuffle / zero ablation을 매 validation마다 수행

### 2차
- ROI-pair-specific local anatomy feature 추가
- ROI_i / ROI_j 주변 feature pooling
- 해당 두 ROI 사이 WM corridor feature 사용

### 성공 기준

```text
자기 T1 prediction
>
남의 T1 prediction
>
T1=0 prediction
```

차이가 명확해야 한다.

또한 subject-specific residual correlation이 현재 약 0.026 수준에서 의미 있게 상승해야 한다.

---

# 2. Streamline Decoder의 Geometry Reconstruction 정확도가 부족함

## 현재 문제

latent sampling만의 문제가 아니다.

실측:

```text
Prior latent       → pair Dice ≈ 0.053
Train bank latent  → pair Dice ≈ 0.137
Oracle posterior   → pair Dice ≈ 0.167
Real vs Real ceiling → ≈ 0.597
```

정답에 가까운 oracle latent를 넣어도 실제 bundle spatial distribution의 약 28% 수준만 재현한다.

즉 현재 병목은:

```text
latent prior
```

보다

```text
Decoder 자체의 geometry reconstruction
```

에 더 가깝다.

## 현재 reconstruction 오차

```text
streamline reconstruction RMSE ≈ 3.55 mm
atlas resolution = 2 mm
```

약 1.8 voxel의 오차가 발생한다.

ROI boundary 근처에서는 이 정도 오차만으로도:

```text
올바른 ROI 방문
→ 다른 ROI 방문
```

으로 바뀔 수 있다.

## 결과

전체 brain-level Dice는 비교적 높아 보이지만:

```text
Whole-brain Dice ≈ 0.565
Pair-specific Dice ≈ 0.059
```

이다.

즉 전체적으로는 tractogram처럼 보이지만,
개별 ROI pair의 trajectory는 정확하지 않다.

## 우선 개선안

### Decoder reconstruction 자체 먼저 개선
- reconstruction loss 중심 pre-finetuning
- endpoint / route / SC loss를 동시에 강하게 넣기 전에 geometry 안정화
- decoder LR 재탐색
- pretrained decoder freeze vs partial/full fine-tuning 비교
- 128-point coordinate reconstruction error를 ROI boundary 기준으로 분석

### 추가 평가
- MDF
- endpoint distance
- pair-wise Dice
- coverage
- overreach
- length distribution
- ROI visitation accuracy

## 성공 기준

최소 목표:

```text
reconstruction RMSE: 3.55 mm → 2 mm 근처 이하
pair-wise Dice: 0.059 → 명확한 상승
```

SC correlation만 상승하고 pair-wise geometry가 개선되지 않으면
tractography reconstruction 성공으로 판단하지 않는다.

---

# 3. Synthetic Streamline의 Realism이 부족함

## 현재 문제

GESTA-inspired augmentation에서:

```text
Real streamline QC pass ≈ 94.7%
Synthetic QC pass       ≈ 12%
```

로 synthetic candidate 대부분이 탈락한다.

특히 주요 병목은 curvature / turning angle이다.

현재 QC threshold는 TRAIN real streamline 분포에서 계산했으며,
synthetic은 기준을 느슨하게 해도 많이 탈락하였다.

따라서 단순히 QC threshold가 너무 엄격한 문제라기보다:

```text
latent sampling
+
decoder
```

가 실제 streamline distribution 밖의 geometry를 자주 생성하는 문제로 해석한다.

## 현재 적용된 Hard QC

- length
- maximum turning angle
- winding
- endpoint-distance / length ratio
- brain occupancy
- duplicate removal

이 부분은 적절하게 구현되어 있다.

## 아직 필요한 부분

현재는 개별 streamline QC 중심이다.

추가로 **bundle-level distribution QC**가 필요하다.

### Real vs Synthetic bundle 비교

```text
Coverage ↑
Overreach ↓
Dice ↑
Length distribution 유사
Curvature distribution 유사
Duplicate ratio ↓
```

를 확인해야 한다.

## 중요

GESTA synthetic은:

```text
training loss를 낮추기 위한 데이터
```

가 아니라:

```text
under-represented bundle의 real geometry distribution을 보완하는 데이터
```

여야 한다.

따라서 synthetic이 많아져 SC 성능만 올라가도,
geometry가 real distribution과 다르면 성공으로 보지 않는다.

## 권장 Ablation

```text
A0. ATM only
A1. + Balanced sampling
A2. + Raw synthetic
A3. + Synthetic Hard QC
A4. + Hard QC + Bundle-level QC
```

각 조건에서:

- pair-wise Dice
- coverage / overreach
- SC Pearson / CCC
- small-edge performance
- SUB-SUB pass-only performance

를 비교한다.

## 성공 기준

```text
Synthetic QC pass rate 상승
+
Real/Synthetic bundle distribution 차이 감소
+
Pair-wise geometry 개선
+
Real validation subject의 SC/TRK 성능 개선
```

이 동시에 나타나야 한다.

---

# 4. 세 병목의 관계

세 문제는 서로 독립적이지 않다.

```text
T1 subject-specific 정보 부족
        ↓
개인 anatomy conditioning 약함
        ↓
Decoder가 population-average trajectory 생성
        ↓
Pair-specific geometry 부정확
        ↓
Synthetic도 decoder를 거치며 geometry 오류 발생
        ↓
QC pass rate 저하
```

따라서 우선순위는 다음과 같이 잡는 것이 좋다.

```text
1. Decoder geometry 정확도 개선
        ↓
2. T1 anatomy conditioning 강화
        ↓
3. Synthetic bundle-level QC 및 augmentation 재검증
```

단, 1과 2는 병렬로 작은 실험을 해도 된다.

---

# 5. 지금 당장 하지 않아도 되는 것

현재는 새로운 head/loss를 계속 추가하기보다
기존 핵심 경로를 먼저 안정화하는 것이 중요하다.

우선 보류:
- 새로운 복잡한 GESTA sampling 방식
- 추가적인 SC loss 다수
- synthetic 비율 대폭 증가
- 새로운 output head 추가

현재 핵심은:

```text
T1 정보가 실제로 사용되는가?
Decoder가 정확한 trajectory를 재현하는가?
Synthetic이 real geometry를 보존하는가?
```

세 질문에 답하는 것이다.

---

# 6. 최종 요약

| 병목 | 현재 증거 | 핵심 해결 방향 |
|---|---|---|
| **T1 anatomy 무시** | 자기/남의/zero T1 결과 거의 동일 | anatomy conditioning 및 local ROI feature 강화 |
| **Decoder geometry 부족** | recon 3.55 mm, pair Dice 0.059 | decoder reconstruction accuracy 우선 개선 |
| **Synthetic realism 부족** | Real QC 94.7% vs Synthetic 12% | hard QC + bundle-level distribution QC |

최종 목표는 단순히 SC correlation을 높이는 것이 아니라:

```text
T1-specific anatomy
→ realistic streamline geometry
→ realistic subject-specific SC
```

가 실제로 성립하도록 만드는 것이다.
