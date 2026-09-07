# T1 → TRK → SC 재현 평가 지표 정리

## 0. 평가 목적

본 연구의 최종 목표는 **T1 MRI 한 장만으로 subject-specific whole-brain tractogram(TRK/TCK)을 생성하고, 그로부터 SC matrix와 tract-length matrix를 재현하는 것**이다.

따라서 평가는 하나의 지표만 보는 것이 아니라 아래 4개 축으로 나누는 것이 가장 적절하다.

1. **Tractography geometry 재현**
2. **Structural connectivity(SC) 재현**
3. **Subject-specificity 재현**
4. **Tract-length 재현**

---

# 1. Tractography Geometry 재현 지표

## 1-1. Pair-wise Dice

각 ROI pair의 GT bundle과 생성 bundle을 voxelize한 뒤 공간적 겹침 정도를 계산한다.

\[
Dice = \frac{2|V_{GT} \cap V_{Pred}|}{|V_{GT}| + |V_{Pred}|}
\]

### 의미
- 특정 ROI pair의 streamline이 **실제와 비슷한 공간을 지나가는지** 평가
- 1에 가까울수록 좋음

### 방향

```text
높을수록 좋음 ↑
```

### 중요도

**매우 높음**

Whole-brain Dice보다 **pair-wise Dice가 더 중요**하다.  
전체 tractogram이 대충 같은 WM 영역을 지나가더라도 개별 연결이 틀릴 수 있기 때문이다.

---

## 1-2. Whole-brain Dice

전체 GT tractogram과 전체 생성 tractogram을 voxelize하여 공간 겹침을 계산한다.

### 의미
- 전체 tractogram의 전반적인 공간 범위가 비슷한지 확인

### 방향

```text
높을수록 좋음 ↑
```

### 주의

Whole-brain Dice가 높더라도 개별 ROI-pair trajectory가 틀릴 수 있으므로 반드시 pair-wise Dice와 같이 본다.

---

## 1-3. Coverage / Overlap

GT bundle 영역 중 생성 bundle이 얼마나 재현했는지 본다.

\[
Coverage = \frac{|V_{GT} \cap V_{Pred}|}{|V_{GT}|}
\]

### 의미

```text
GT가 차지하는 실제 공간을
생성 streamline이 얼마나 덮었는가?
```

### 방향

```text
높을수록 좋음 ↑
```

---

## 1-4. Overreach

생성 bundle이 GT 밖으로 얼마나 퍼졌는지 평가한다.

예시 정의:

\[
Overreach =
\frac{|V_{Pred} \setminus V_{GT}|}{|V_{Pred}|}
\]

### 의미

```text
생성 streamline이
실제 tract 영역 밖으로 얼마나 벗어났는가?
```

### 방향

```text
낮을수록 좋음 ↓
```

### 해석

가장 이상적인 경우:

```text
Coverage ↑
Overreach ↓
```

---

## 1-5. MDF (Minimum Direct-Flip Distance)

GT streamline과 prediction streamline의 대응 point 간 평균 거리를 계산하되,
streamline 방향이 반대일 수 있으므로 direct / flipped 두 방향 중 작은 값을 선택한다.

\[
MDF = \min(d_{direct}, d_{flip})
\]

### 의미

개별 streamline의 **전체 shape이 얼마나 가까운가**를 평가한다.

### 방향

```text
낮을수록 좋음 ↓
```

### 장점

128-point로 resampling된 streamline을 사용하는 현재 구조에서 적용하기 좋다.

---

## 1-6. Endpoint Distance

GT와 prediction의 양 끝점 위치 차이를 측정한다.

예:

\[
d_{end}
=
\frac{
\|p_1-\hat{p}_1\|
+
\|p_N-\hat{p}_N\|
}{2}
\]

### 의미

- 올바른 ROI 근처에서 시작/종료했는지
- endpoint assignment가 정확한지

### 방향

```text
낮을수록 좋음 ↓
```

---

## 1-7. Hausdorff Distance

GT streamline과 prediction streamline 사이의 최악의 공간 차이를 측정한다.

### 의미

평균적으로는 비슷하지만 특정 부분이 크게 벗어난 streamline을 탐지할 수 있다.

### 방향

```text
낮을수록 좋음 ↓
```

---

## 1-8. Valid Connection Rate / Invalid Connection Rate

생성 streamline이 해부학적으로 올바른 ROI pair를 연결했는지 평가한다.

### 예시

```text
Valid Connection Rate
= 올바른 ROI pair 연결 수 / 전체 생성 streamline 수
```

### 방향

```text
Valid Connection ↑
Invalid Connection ↓
```

---

# 2. Structural Connectivity(SC) Matrix 재현 지표

현재 연구에서는 streamline이 통과한 **모든 ROI pair**에 contribution하는 PASS-SC를 사용한다.

---

## 2-1. Pearson Correlation

\[
r = corr(SC_{Pred}, SC_{GT})
\]

### 의미

SC edge들의 **강약 패턴이 얼마나 비슷한가**를 평가한다.

예:

```text
GT에서 강한 edge
→ Prediction에서도 강한가?

GT에서 약한 edge
→ Prediction에서도 약한가?
```

### 방향

```text
1에 가까울수록 좋음 ↑
```

### 주의

Pearson r이 높아도 절대 scale은 틀릴 수 있다.

따라서 CCC와 같이 보는 것이 중요하다.

---

## 2-2. Log-Pearson Correlation

\[
r_{log}
=
corr(\log(1+SC_{Pred}), \log(1+SC_{GT}))
\]

### 의미

강한 edge가 결과를 지배하는 것을 줄이고,
약한 연결까지 비교한다.

### 방향

```text
높을수록 좋음 ↑
```

---

## 2-3. Spearman Correlation

SC edge 순위가 얼마나 비슷한지 평가한다.

### 의미

정확한 절대값보다:

```text
어떤 edge가 더 강하고
어떤 edge가 더 약한가
```

의 순서를 본다.

### 방향

```text
높을수록 좋음 ↑
```

---

## 2-4. Concordance Correlation Coefficient (CCC)

Pearson correlation이 패턴만 보는 단점을 보완한다.

\[
CCC =
\frac{2\,cov(P,G)}
{var(P)+var(G)+(\mu_P-\mu_G)^2}
\]

### 의미

동시에 평가:

- 패턴 유사성
- 평균값 차이
- 분산 차이
- 절대 scale

### 방향

```text
1에 가까울수록 좋음 ↑
```

### 중요도

**SC 평가에서 Pearson과 함께 핵심 지표**

---

## 2-5. RMSE

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_i
(P_i-G_i)^2
}
\]

### 의미

SC edge 값의 절대 오차를 평가한다.

### 방향

```text
낮을수록 좋음 ↓
```

---

## 2-6. MAE

\[
MAE =
\frac{1}{N}
\sum_i |P_i-G_i|
\]

### 의미

RMSE보다 extreme edge에 덜 민감한 절대 오차 지표.

### 방향

```text
낮을수록 좋음 ↓
```

---

## 2-7. Log-MAE

\[
LogMAE =
mean(
|\log(1+P)-\log(1+G)|
)
\]

### 의미

SC가 heavy-tail distribution을 가지므로
약한 edge와 강한 edge를 상대적 scale에서 비교하기 좋다.

### 방향

```text
낮을수록 좋음 ↓
```

---

## 2-8. Edge F1

GT와 prediction의 연결 존재 여부를 이진화하여 비교한다.

### 계산

- Precision
- Recall
- F1 score

### 의미

```text
연결이 존재하는 edge를
제대로 찾아냈는가?
```

### 방향

```text
높을수록 좋음 ↑
```

---

# 3. Subject-specificity 재현 지표

이 부분은 현재 연구에서 매우 중요하다.

단순 SC correlation이 높더라도 모든 subject에게 거의 같은 SC를 출력하면
**subject-specific model이라고 보기 어렵다.**

---

## 3-1. Residual Correlation

Population-average component를 제거한 뒤
개인의 residual pattern을 GT와 비교한다.

권장 방식은 **leave-one-out centering**이다.

\[
P'_s = P_s - \frac{\sum_{k\neq s} P_k}{N-1}
\]

\[
G'_s = G_s - \frac{\sum_{k\neq s} G_k}{N-1}
\]

그 후:

\[
r_{residual}
=
corr(P'_s,G'_s)
\]

### 의미

```text
그룹 평균이 아니라
이 subject만의 SC 차이를 맞췄는가?
```

### 방향

```text
높을수록 좋음 ↑
```

---

## 3-2. Subject-to-Subject Prediction Similarity

생성된 여러 subject의 SC끼리 correlation을 계산한다.

### 비교

```text
GT subject 간 평균 correlation
vs
Predicted subject 간 평균 correlation
```

### 의미

prediction끼리 너무 비슷하면 population template collapse를 의미한다.

### 이상적인 방향

```text
Predicted subject 간 variability
≈
GT subject 간 variability
```

---

## 3-3. Self-T1 vs Shuffled-T1 Test

동일한 downstream 조건에서 T1만 바꾸어 비교한다.

```text
본인 T1
남의 T1
Zero T1
```

### 성공 조건

```text
Self-T1 performance
>
Shuffled-T1 performance
>
Zero-T1 performance
```

또는 최소한 본인 T1이 명확하게 가장 좋아야 한다.

### 의미

T1 encoder가 실제 subject-specific anatomy 정보를 사용하고 있는지 직접 검증한다.

---

## 3-4. Population Template Baseline

TRAIN subject들의 평균 SC를 모든 TEST subject에게 동일하게 제공한다.

\[
SC_{template}
=
\frac{1}{N_{train}}
\sum_s SC_s
\]

### 의미

T1을 전혀 사용하지 않는 baseline.

### 중요

모델이 population template보다 못하면:

```text
T1에서 개인 정보를 충분히 활용하지 못했다
```

고 해석해야 한다.

---

# 4. Tract-Length Matrix 재현 지표

Tract length는 TRK/SC보다 secondary endpoint로 둘 수 있다.

하지만 TVB와 같은 whole-brain model에 사용할 경우 중요성이 커진다.

---

## 4-1. Tract-Length Pearson Correlation

\[
r_L
=
corr(L_{Pred}, L_{GT})
\]

### 의미

긴 연결과 짧은 연결의 패턴을 재현했는지 평가한다.

### 방향

```text
높을수록 좋음 ↑
```

---

## 4-2. Tract-Length CCC

길이의 패턴뿐 아니라 절대 길이까지 맞는지 평가한다.

### 방향

```text
1에 가까울수록 좋음 ↑
```

---

## 4-3. Tract-Length RMSE

\[
RMSE_L
=
\sqrt{
\frac{1}{N}
\sum_i
(L_{Pred,i}-L_{GT,i})^2
}
\]

단위:

```text
mm
```

### 방향

```text
낮을수록 좋음 ↓
```

---

## 4-4. Length Distribution Error

전체 tractogram의 streamline length distribution을 비교한다.

추천:

- mean / median difference
- Wasserstein distance
- Kolmogorov-Smirnov statistic

### 의미

```text
생성된 tractogram의 길이 분포가
실제 tractogram과 비슷한가?
```

---

# 5. 논문에서 권장하는 핵심 평가 세트

모든 지표를 Main Table에 넣을 필요는 없다.

## Primary metrics

### TRK

```text
Pair-wise Dice
Coverage
Overreach
MDF 또는 Endpoint Distance
```

### SC

```text
Pearson r
CCC
RMSE
Edge F1
```

### Subject-specificity

```text
Residual r
Self-T1 vs Shuffled-T1
Population-template baseline
```

## Secondary metrics

```text
Whole-brain Dice
Spearman
Log-Pearson
Log-MAE
Hausdorff
Length r
Length CCC
Length RMSE
KS / Wasserstein
```

---

# 6. 논문 결과 해석 기준

좋은 결과의 예시는 다음과 같은 형태이다.

```text
Tractography
-------------
Pair-wise Dice      ↑
Coverage            ↑
Overreach           ↓
MDF                 ↓

SC
-------------
Pearson r           > 0.9
CCC                 높음
RMSE                낮음
Edge F1             높음

Subject-specificity
-------------
Residual r          > 0
Self-T1             > Shuffled-T1
Pred subject 다양성 ≈ GT subject 다양성

Tract length
-------------
Length r            양호
Length RMSE         허용 범위
```

단, **SC Pearson r > 0.9만으로 성공을 주장해서는 안 된다.**

Population-average SC 자체가 높은 correlation을 만들 수 있으므로:

```text
TRK geometry
+
SC reconstruction
+
Subject-specificity
```

세 축이 같이 좋아야 한다.

---

# 7. 최종 권장 평가 프레임

```text
                    T1
                     ↓
                Generated TRK
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   Geometry       SC Matrix   Tract Length
        │            │            │
 Pair Dice       Pearson r     Length r
 Coverage        CCC           RMSE
 Overreach       RMSE          CCC
 MDF             Edge F1
        │
        └────────────┬────────────┘
                     ↓
             Subject-specificity
                     │
              Residual r
              Self vs Shuffle T1
              Template baseline
```

---

# 8. 가장 중요한 결론

현재 연구에서 최우선 평가는 다음 3개다.

1. **생성 TRK의 geometry가 실제 tractogram과 비슷한가**
2. **그 TRK에서 추출한 SC matrix가 GT와 비슷한가**
3. **그 결과가 단순한 population 평균이 아니라 subject-specific한가**

Tract length는 주요 응용이 TVB라면 함께 평가하되,
논문의 primary endpoint는 **TRK geometry + SC reconstruction + subject-specificity**로 두는 것이 가장 적절하다.
