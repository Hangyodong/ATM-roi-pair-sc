# T1 기반 subject specific SC 및 TRK collapse 문제 해결 전략

## 문서 목적

이 문서는 T1 MRI를 입력으로 생성한 structural connectivity matrix와 tractogram이 피험자별 차이를 반영하지 못하고 population mean에 수렴하는 문제를 해결하기 위한 실행 계획이다. 핵심은 문제를 다음 두 축으로 분리하는 것이다.

1. **Subject specificity 축**: T1에서 개인별 SC 편차를 복원한다.
2. **Tract geometry 축**: 생성 prior와 해부학적 제약을 고쳐 streamline의 위치와 형태를 개선한다.

두 축은 평가 지표와 실패 원인이 다르므로 처음부터 하나의 joint loss로 동시에 해결하지 않는다. 먼저 독립 실험으로 병목을 확인한 뒤, 각 축에서 유효성이 검증된 구성만 최종 모델에 결합한다.

## 현재 문제에 대한 판단

현재 기록에서 관찰된 핵심 현상은 population mean collapse와 prior mismatch가 동시에 존재한다는 것이다.

- 생성 SC의 피험자 간 상관이 약 `0.99914`로, 피험자별 출력이 거의 동일하다.
- 전체 SC 상관은 약 `0.788`이지만 train population template을 그대로 사용한 baseline은 약 `0.945`이다.
- population template을 제거한 residual correlation은 약 `0.023`이다.
- 학습 단계가 진행되면서 residual correlation이 `0.124 -> 0.009 -> -0.031`로 감소했다. 절대 크기와 RMSE 중심 loss가 개인 신호를 억제했을 가능성이 크다.
- spatial feature map에서는 피험자 간 차이가 남아 있지만 global average pooling 뒤의 anatomy vector는 거의 동일하다.
- BN 재보정 후 oracle posterior를 사용하면 whole brain Dice 약 `0.680`, pair Dice 약 `0.524`까지 도달하지만, 생성 prior에서는 pair Dice가 약 `0.068`이다.
- posterior의 조건부 표준편차는 약 `0.18`인데 현재 prior 표준편차는 약 `1.0`으로 기록되어 있다. 생성 시 latent를 지나치게 넓게 샘플링할 가능성이 크다.
- 현재 prior NLL은 약 `71.62`, arch additive 후보는 약 `15.60`으로 기록되어 있다.

따라서 현재 모델의 높은 전체 SC correlation은 개인별 T1 정보를 잘 사용했다는 증거가 아니다. 공통 connectome 구조를 재생한 결과일 수 있으므로, 향후 실험의 중심 지표는 전체 SC correlation보다 **residual 예측력, self T1 우위, 피험자 식별력, 생성 다양성**이어야 한다.

## 1 Template plus residual SC 학습

### 1.1 목표

모델이 SC 전체를 처음부터 예측하게 두면, edge별 population mean을 외우는 것만으로도 loss를 크게 줄일 수 있다. 이를 막기 위해 공통 구조는 고정 template으로 제공하고, 신경망은 피험자별 편차만 예측하게 한다.

### 1.2 Target 정의

SC count의 긴 꼬리 분포를 안정화하기 위해 먼저 log transform을 적용한다.

$$
y_{s,ij}=\log(1+SC_{s,ij})
$$

train set만 사용하여 edge별 population template을 계산한다.

$$
\mu_{ij}=\frac{1}{N_{train}}\sum_{s\in train}y_{s,ij}
$$

피험자별 residual target은 다음과 같다.

$$
\Delta_{s,ij}=y_{s,ij}-\mu_{ij}
$$

모델은 전체 SC가 아니라 residual만 예측한다.

$$
\widehat{\Delta}_{s,ij}=f_\theta(T1_s,i,j)
$$

최종 SC는 template과 예측 residual을 합쳐 복원한다.

$$
\widehat{SC}_{s,ij}=\exp(\mu_{ij}+\widehat{\Delta}_{s,ij})-1
$$

비음수 정수 count가 필요하면 마지막 단계에서만 `clamp(min=0)`와 반올림 또는 stochastic rounding을 적용한다. 학습 중에는 연속값을 유지한다.

### 1.3 Data leakage 방지

- `mu_ij`는 반드시 train split에서만 계산한다.
- validation과 test 피험자는 template 계산에 포함하지 않는다.
- cross validation을 사용하면 fold마다 template을 다시 계산한다.
- 희소 edge를 제거하거나 thresholding할 때도 threshold 통계는 train split에서만 결정한다.
- template, residual normalization mean과 std, edge mask를 checkpoint와 함께 저장한다.

### 1.4 Residual 표준화

edge별 residual scale 차이가 큰 경우 train set의 표준편차로 정규화한다.

$$
\widetilde{\Delta}_{s,ij}=\frac{\Delta_{s,ij}}{\sigma_{ij}+\epsilon}
$$

단, 분산이 거의 없는 edge는 큰 정규화 값과 불안정한 gradient를 만들 수 있다. 다음 중 하나를 사용한다.

- `sigma_ij`에 하한값을 둔다.
- train prevalence와 variance 기준으로 edge mask를 만든다.
- variance가 낮은 edge는 residual loss의 weight를 낮춘다.

### 1.5 최소 구현 형태

```python
# train split only
log_sc_train = torch.log1p(sc_train)
template = log_sc_train.mean(dim=0)
residual_std = log_sc_train.std(dim=0).clamp_min(std_floor)

# training target
target_residual = (torch.log1p(sc_subject) - template) / residual_std
pred_residual = model(t1_subject, edge_index)

# reconstruction for evaluation
pred_log_sc = template + pred_residual * residual_std
pred_sc = torch.expm1(pred_log_sc).clamp_min(0)
```

### 1.6 반드시 비교할 baseline

1. 모든 피험자에게 train template만 출력
2. 기존 absolute SC prediction 모델
3. template plus residual 모델
4. template plus residual 모델에 shuffled T1 입력
5. template plus residual 모델에 zero 또는 mean T1 입력

전체 SC correlation만 보면 template baseline이 강하므로, 비교의 핵심은 residual과 개인 식별 지표다.

## 2 Decoder와 prior 축 분리

### 2.1 분리 원칙

현재 oracle posterior 성능과 generated prior 성능의 차이가 크므로, decoder capacity와 prior quality를 같은 문제로 취급하면 안 된다.

```text
Axis A  Subject specificity
T1 -> residual SC 또는 subject conditioned count

Axis B  Tract geometry
latent prior -> decoder -> streamline geometry
```

### 2.2 Axis A 실험 설정

첫 residual 실험에서는 다음 구성만 학습한다.

- T1 encoder
- residual count head
- 필요하면 local anatomy projection layer

다음 구성은 고정한다.

- streamline decoder
- latent prior
- 기존 tract generation 경로

이 실험의 질문은 하나다.

> T1에 GT SC residual을 설명할 재현 가능한 신호가 있으며, end to end residual supervision으로 그 신호를 읽을 수 있는가?

### 2.3 Axis B 실험 설정

geometry 실험에서는 decoder checkpoint와 count 설정을 고정하고 prior sampling만 변경한다. 그 다음 필터링을 별도로 평가한다.

1. oracle posterior 조건의 상한 성능 확인
2. prior mean만 사용하는 deterministic decoding
3. prior temperature sweep
4. prior architecture 교체
5. WM GM filtering과 length filtering

### 2.4 Joint training 재개 조건

다음 조건을 모두 만족한 뒤에만 joint fine tuning을 고려한다.

- residual SC가 template baseline보다 피험자별 지표에서 우수하다.
- self T1이 shuffled T1보다 일관되게 우수하다.
- prior temperature 또는 새 prior가 geometry를 개선한다.
- 각 단독 실험의 개선이 서로의 지표를 훼손하지 않는다.

joint fine tuning을 수행할 때는 두 축의 validation metric을 따로 기록하고 Pareto trade off를 확인한다. SC 개선과 geometry 악화가 동시에 일어나면 하나의 합산 점수로 숨기지 않는다.

## 3 Global pooling 문제와 local anatomy conditioning

### 3.1 현재 구조의 한계

현재와 같이 T1 전체를 하나의 global anatomy vector로 압축하고 모든 ROI pair에 같은 vector를 제공하면, 각 연결이 봐야 할 국소 해부학 정보가 희석된다.

```text
T1 -> 3D encoder -> global average pooling -> anatomy vector
                                             |-> edge A B
                                             |-> edge A C
                                             `-> 모든 edge
```

spatial feature map에 남아 있던 피험자 차이가 global average pooling 뒤에 거의 사라진다는 관찰은 이 구조가 collapse 지점일 수 있음을 시사한다.

### 3.2 권장 구조

3D encoder의 spatial feature map `F_s(x,y,z)`를 유지하고, edge `(i,j)`별 anatomy representation을 만든다.

$$
h_{s,ij}=[f_{s,i},f_{s,j},f_{s,ij}^{path},f_s^{global},e_{ij}]
$$

- `f_s,i`: ROI i 내부 또는 경계에서 pooled feature
- `f_s,j`: ROI j 내부 또는 경계에서 pooled feature
- `f_s,ij_path`: 두 ROI 사이의 예상 WM corridor에서 pooled feature
- `f_s,global`: global context
- `e_ij`: ROI pair identity 또는 anatomical pair embedding

예측은 다음과 같이 수행한다.

$$
\widehat{\Delta}_{s,ij}=g_\phi(h_{s,ij})
$$

### 3.3 Local feature 추출 후보

낮은 구현 비용부터 순서대로 시험한다.

1. **Endpoint ROI pooling**: atlas mask를 downsample하여 feature map에서 masked mean과 max pooling을 수행한다.
2. **ROI boundary pooling**: GM WM 경계의 얇은 band에서 feature를 추출한다.
3. **Straight corridor pooling**: ROI centroid를 잇는 선 주변의 tube mask를 사용한다.
4. **Atlas pathway pooling**: train set의 population tract density를 사용해 edge별 soft mask를 만든다.
5. **Deformable or attention sampling**: pair query가 공간 위치를 선택하도록 학습한다.

처음에는 endpoint ROI pooling과 global context만 사용한다. pathway mask는 성능 필요성이 확인된 뒤 추가한다.

### 3.4 Pair embedding 우세 방지

강한 pair embedding은 모델이 T1을 무시하고 edge lookup table처럼 작동하게 만들 수 있다. 다음을 점검한다.

- anatomy feature와 pair embedding의 norm과 분산을 로깅한다.
- 각각 LayerNorm 후 동일한 projection dimension으로 맞춘다.
- pair embedding dropout 또는 stochastic masking을 적용한다.
- 일부 step에서는 pair embedding 없이 residual을 예측하게 한다.
- anatomy branch를 제거했을 때와 T1을 shuffle했을 때 성능이 실제로 하락하는지 본다.

### 3.5 Local architecture로 넘어가는 기준

현재 encoder를 end to end로 학습한 template plus residual 실험에서 residual correlation이 의미 있게 상승하면 local conditioning으로 확장한다. 반대로 end to end residual supervision에서도 residual correlation이 반복 실험에서 0 근처라면, 대수술 전에 T1과 GT SC 사이의 재현 가능한 개인 신호 및 정합 품질을 먼저 조사한다.

## 4 Residual loss와 difference loss

### 4.1 기본 residual loss

절대 SC magnitude보다 residual target에 직접 loss를 건다.

$$
L_{res}=\operatorname{SmoothL1}(\widehat{\Delta},\Delta)
$$

희소 edge와 outlier에 덜 민감하므로 초기 실험에는 MSE보다 Smooth L1을 권장한다.

### 4.2 Residual correlation loss

피험자 내부에서 edge별 증감 패턴을 보존하기 위해 correlation loss를 추가한다.

$$
L_{corr}=1-\operatorname{corr}(\widehat{\Delta}_s,\Delta_s)
$$

분산이 거의 없는 batch 또는 edge subset에서는 correlation이 불안정할 수 있으므로 epsilon을 사용하고 유효 edge 수를 확인한다.

### 4.3 Subject difference loss

두 피험자의 차이를 직접 맞추면 모든 피험자에게 동일 residual을 출력하는 shortcut을 억제할 수 있다.

$$
L_{diff}=\left\| (\widehat{\Delta}_s-\widehat{\Delta}_t)-(\Delta_s-\Delta_t) \right\|_1
$$

동일 출력을 내면 예측 차이가 0이 되므로 실제 피험자 차이가 존재하는 한 loss를 피할 수 없다.

### 4.4 Variance preservation loss 선택안

필요할 경우 batch 내 예측 residual variance가 GT variance보다 지나치게 작아지는 것을 막는다.

$$
L_{var}=\left|\operatorname{Std}_s(\widehat{\Delta}_{s,ij})-\operatorname{Std}_s(\Delta_{s,ij})\right|
$$

이 loss는 batch에 여러 피험자가 있어야 안정적이며, 과도한 weight는 노이즈를 개인차로 증폭할 수 있으므로 후순위 ablation으로 둔다.

### 4.5 초기 권장 조합

$$
L_{personal}=L_{res}+0.2L_{corr}+0.2L_{diff}
$$

초기 실험에서는 absolute SC RMSE, 전체 scale loss, whole SC magnitude loss를 끈다. residual이 살아난 뒤 최종 calibration이 필요할 때만 작은 weight로 다시 도입한다.

권장 ablation은 다음과 같다.

| 실험 | Loss 구성 | 목적 |
|---|---|---|
| A | `L_res` | residual target 자체의 학습 가능성 확인 |
| B | `L_res + L_corr` | edge별 증감 패턴 보존 효과 확인 |
| C | `L_res + L_diff` | 동일 출력 collapse 억제 효과 확인 |
| D | `L_res + L_corr + L_diff` | 권장 조합 검증 |
| E | D plus `L_var` | 분산 부족이 남을 때만 평가 |

## 5 Subject batch 전략

### 5.1 최소 요구사항

한 optimization step의 loss에는 최소 두 명의 서로 다른 피험자가 참여해야 한다. 그래야 subject difference loss와 batch level diversity 지표를 계산할 수 있다.

### 5.2 메모리가 충분한 경우

- subject batch size를 2 이상으로 설정한다.
- 각 피험자에서 같은 edge subset을 샘플링하면 차이 계산이 단순해진다.
- 강한 edge와 약한 edge를 함께 포함하는 stratified edge sampling을 사용한다.

### 5.3 메모리가 부족한 경우

두 피험자를 동시에 GPU에 올리지 않고 sequential forward를 사용한다.

```text
Subject A forward -> residual A 보관
Subject B forward -> residual B 보관
pairwise difference loss 계산
gradient accumulation 또는 activation checkpointing
backward
```

두 번째 forward까지 계산 graph를 유지하기 어렵다면 다음 대안을 사용한다.

- encoder feature를 낮은 해상도 또는 mixed precision으로 저장
- edge head 단계에서만 두 피험자를 함께 처리
- microbatch 두 개의 prediction을 모아 loss를 계산
- activation checkpointing 적용

### 5.4 Pair 구성

- 무작위 subject pair만 사용하면 쉬운 쌍과 어려운 쌍이 섞인다.
- 초기에는 무작위 pairing으로 시작한다.
- 이후 GT residual distance가 너무 작거나 큰 쌍에 치우치지 않도록 거리 구간별 sampling을 고려한다.
- 같은 피험자의 augmentation 두 개는 invariance 학습에는 유용하지만 subject difference pair를 대체하지 않는다.

### 5.5 Batch에서 로깅할 값

- GT subject pair residual distance
- predicted subject pair residual distance
- 두 거리의 ratio
- batch 내 predicted residual standard deviation
- batch 내 GT residual standard deviation
- anatomy feature의 subject variance
- pair embedding norm 대비 anatomy feature norm

## 6 Prior temperature sweep와 arch additive prior

### 6.1 Temperature sweep

현재 prior 표준편차가 posterior보다 넓다면 재학습 전에 sampling temperature부터 조정한다.

$$
z=\mu+s\sigma\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

권장 sweep은 다음과 같다.

```text
s = 0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0
```

`s=0.0`은 prior mean deterministic baseline이다. 각 temperature에서 같은 subject, 같은 pair, 같은 streamline count와 고정된 seed set을 사용하여 공정하게 비교한다. 단일 seed 결과가 아니라 여러 seed의 평균과 표준편차를 보고한다.

### 6.2 Temperature 선택 기준

다음 지표를 함께 본다.

- pair wise Dice 증가
- MDF와 endpoint error 감소
- whole brain Dice 유지 또는 증가
- overreach 감소
- streamline diversity가 지나치게 붕괴하지 않음
- invalid streamline 비율 감소

temperature가 낮아져 Dice만 오르고 생성 다양성이 0에 가까워지면 posterior mode를 복사하는 deterministic generator가 될 수 있다. geometry 정확도와 다양성의 Pareto curve를 확인한다.

### 6.3 Arch additive prior

temperature 조절은 잘못된 분산 scale을 보정할 뿐, prior의 구조적 mismatch를 해결하지는 못한다. 이후에는 ROI별 성분과 pair interaction을 분해한 additive prior를 시험한다.

예시 parameterization은 다음과 같다.

$$
\mu_{s,ij}=\mu_i+\mu_j+\mu_{ij}^{interaction}+A_{s,ij}^{\mu}
$$

$$
\log\sigma_{s,ij}=b_i+b_j+b_{ij}^{interaction}+A_{s,ij}^{\sigma}
$$

- `mu_i`, `mu_j`: ROI별 공통 latent 성분
- `mu_ij_interaction`: 해당 ROI pair의 추가 성분
- `A_s,ij`: 선택적인 subject anatomy conditioning
- `log sigma`: 양수 분산을 안정적으로 보장

초기 arch additive 실험은 subject conditioning 없이 pair prior만 개선하고, 이후 anatomy conditioned residual prior를 추가한다. 이렇게 하면 geometry 개선이 prior 구조 때문인지 T1 conditioning 때문인지 분리할 수 있다.

### 6.4 Prior 평가

- held out posterior latent에 대한 NLL
- prior와 posterior의 차원별 mean 및 std 비교
- calibration plot
- latent interpolation 시 streamline 변화의 연속성
- oracle posterior, prior mean, sampled prior의 geometry gap

낮은 NLL이 반드시 좋은 tract geometry를 보장하지 않으므로, NLL과 geometry를 모두 통과해야 한다.

## 7 WM GM filtering과 overreach 개선

### 7.1 목적

prior를 개선해도 생성 streamline이 백질 밖으로 벗어나거나 endpoint가 목표 GM에 도달하지 않으면 overreach가 남는다. filtering은 prior와 decoder의 구조적 오류를 숨기기 위한 수단이 아니라, 해부학적으로 명백히 잘못된 streamline을 제거하는 독립 후처리 단계로 평가한다.

### 7.2 권장 filtering 순서

1. 좌표계와 affine 일치 확인
2. brain mask 밖 point 제거
3. streamline 내부 point의 WM 통과 비율 확인
4. 양 endpoint의 GM 또는 ROI 도달 여부 확인
5. 목표 ROI pair와 endpoint label 일치 확인
6. minimum 및 maximum length 적용
7. 곡률 또는 급격한 방향 전환 기준 적용
8. 중복 또는 거의 동일한 streamline 정리

### 7.3 Soft score 우선

초기에는 hard threshold 하나만 적용하지 말고 각 streamline에 품질 score를 계산한다.

```text
quality =
    w_wm       * WM occupancy
  + w_endpoint * endpoint validity
  + w_pair     * target pair consistency
  - w_outside  * outside brain fraction
  - w_curve    * curvature penalty
```

score threshold를 sweep하여 coverage와 overreach의 trade off를 본 뒤 operating point를 고른다.

### 7.4 WM 기준

- streamline point 중 WM 또는 허용한 GM WM transition band에 속한 비율을 계산한다.
- endpoint 주변의 짧은 GM 구간은 허용하고 중간 경로의 GM 통과는 엄격히 제한한다.
- voxel nearest neighbor만 쓰면 경계에서 불안정할 수 있으므로 trilinear sampling 또는 distance transform 기반 soft mask를 고려한다.

### 7.5 GM endpoint와 pair consistency

- 시작점과 끝점이 목표 ROI의 GM mask 또는 허용 거리 안에 있는지 확인한다.
- 방향성이 없는 streamline은 `(i,j)`와 `(j,i)`를 동일하게 처리한다.
- endpoint가 인접 ROI에 걸칠 때 label ambiguity를 기록한다.
- endpoint snap을 사용한다면 이동 거리 분포를 보고하고, 큰 이동으로 잘못된 streamline을 정상처럼 보이게 만들지 않는다.

### 7.6 Length와 curvature

minimum length는 짧은 spurious connection을 줄이지만 실제 short range fiber도 제거할 수 있다. 따라서 전체에 하나의 threshold를 고정하기보다 다음을 비교한다.

- global minimum length
- ROI pair별 train distribution 기반 lower percentile
- tract class별 threshold

curvature filtering 역시 threshold별 coverage와 overreach를 함께 보고한다.

### 7.7 Filtering 전후 모두 보고

후처리 효과를 명확히 하기 위해 다음 네 조건을 분리한다.

1. raw generation
2. WM filter only
3. endpoint and pair filter only
4. full filter

각 조건에서 retained streamline rate, pair Dice, coverage, overreach, MDF, endpoint error를 함께 기록한다. retained rate가 지나치게 낮은데 지표만 좋아지는 경우는 실질적 개선으로 간주하지 않는다.

## 8 평가 지표와 성공 기준

### 8.1 Subject specificity 핵심 지표

| 지표 | 계산 | 해석 및 성공 기준 |
|---|---|---|
| Residual correlation | GT와 예측에서 train template 제거 후 correlation | 현재 약 `0.023`에서 반복 실험 평균 `0.10` 이상이면 다음 구조 실험으로 진행할 근거. 논문 주장을 위해서는 더 높은 값과 신뢰구간 필요 |
| Self versus shuffled T1 | 같은 head와 pair에서 입력 T1만 교환 | self T1이 shuffled 및 zero T1보다 paired test에서 유의하게 우수해야 함 |
| Subject identification | 각 예측 SC가 어느 GT subject와 가장 유사한지 검색 | chance보다 명확히 높고 permutation test에서 유의해야 함 |
| Inter subject correlation | 예측 SC끼리의 평균 correlation | 현재 약 `0.99914`보다 낮아져야 하며 GT의 피험자 간 분포에 가까워야 함 |
| Variance ratio | predicted residual variance divided by GT residual variance | 0에 수렴하면 collapse. 1에 가까운 것이 이상적이나 noise amplification 여부도 확인 |
| Difference correlation | 모든 subject pair의 GT difference와 predicted difference 비교 | 동일 출력 shortcut 여부를 직접 평가 |

`Residual r >= 0.10`은 조기 개발 단계의 go or no go 기준이지 최종 논문 성공선이 아니다. 최종 판단은 confidence interval, test retest reliability, shuffled control, template baseline 대비 개선을 함께 본다.

### 8.2 SC 재현 지표

- Pearson correlation
- Spearman correlation
- Concordance correlation coefficient
- MAE와 RMSE
- edge prevalence 또는 binary topology 지표
- node strength와 network level graph metric

전체 SC Pearson `r > 0.90`은 유용하지만 충분조건이 아니다. 현재 template baseline이 약 `0.945`이므로 전체 correlation이 0.9를 넘더라도 residual과 subject specificity가 실패하면 개인 T1 기반 복원 주장은 성립하지 않는다.

### 8.3 TRK geometry 지표

- pair wise Dice
- whole brain Dice
- coverage
- overreach
- MDF 또는 bundle distance
- endpoint distance와 endpoint hit rate
- invalid streamline rate
- retained streamline rate after filtering

pair Dice는 현재 generated 값 약 `0.068`과 oracle posterior 값 약 `0.524` 사이의 gap을 얼마나 줄였는지가 중요하다. 절대 threshold 하나보다 동일 split과 동일 평가 코드에서 baseline 대비 개선량을 우선한다.

### 8.4 성공 판정의 최소 논리

다음 세 문장이 모두 데이터로 성립해야 한다.

1. **T1을 올바른 피험자에게 넣었을 때만 SC residual이 더 잘 복원된다.**
2. **prior 개선으로 oracle posterior와 generated tractogram의 geometry gap이 줄어든다.**
3. **해부학적 filtering 후 overreach가 줄면서 coverage가 실용적인 수준으로 유지된다.**

## 9 실행 우선순위

### Phase 0 재학습 없는 진단

| 우선순위 | 실험 | 산출물 | Go 기준 |
|---:|---|---|---|
| 0.1 | 31개 SC의 bitwise equality 및 pairwise correlation 확인 | collapse 유형 판정 | 완전 동일이면 cache seed reuse bug 우선 조사 |
| 0.2 | T1 파일 hash와 intensity 통계 확인 | 입력 중복 여부 | 피험자별 입력이 실제로 다름 |
| 0.3 | anatomy feature의 subject variance 확인 | collapse layer 위치 | spatial map과 pooled vector를 모두 비교 |
| 0.4 | same pair same latent에서 T1만 교환 | decoder의 anatomy sensitivity | streamline과 count가 측정 가능한 수준으로 변함 |
| 0.5 | random seed cache latent bank 재사용 점검 | 구현 오류 배제 | subject loop 안 seed reset 또는 output reuse 없음 |

완전 동일 출력이면 구조 변경 전에 inference bug를 먼저 해결한다. 거의 동일하지만 bitwise identical하지 않다면 population mean collapse 실험으로 진행한다.

### Phase 1 재학습 없는 geometry 실험

1. prior mean deterministic baseline `s=0.0`
2. temperature `0.2, 0.3, 0.4, 0.5, 0.7, 1.0` sweep
3. 기존 count 총량 calibration은 별도 ablation으로 평가
4. WM GM endpoint length filter threshold sweep

이 단계는 빠르게 실행해 geometry 개선 가능성과 최적 operating range를 찾는다.

### Phase 2 Template plus residual 최소 실험

1. train only template과 residual normalization 생성
2. T1 encoder와 residual count head만 학습
3. decoder와 prior는 freeze
4. `L_res`부터 시작
5. `L_corr`와 `L_diff` ablation
6. self, shuffled, zero T1 평가
7. 3개 이상 seed로 평균과 confidence interval 기록

**분기 기준**

- residual correlation이 반복적으로 `0.10` 이상이고 self T1 우위가 확인되면 Phase 3으로 간다.
- residual correlation이 0 근처이고 self T1 우위가 없으면 T1과 GT 정합, GT reliability, preprocessing, population registration이 개인 신호를 지웠는지 조사한다.

### Phase 3 Local anatomy conditioning

1. endpoint ROI pooling 추가
2. global plus local feature 결합
3. pair embedding norm 조정과 dropout
4. pathway corridor feature ablation
5. local feature를 shuffle하는 negative control

local 구조의 성공은 전체 SC correlation이 아니라 residual과 self shuffle gap의 추가 개선으로 판단한다.

### Phase 4 Prior 재구조화

1. arch additive pair prior 구현
2. held out posterior NLL 평가
3. temperature 재조정
4. geometry 평가
5. 선택적으로 subject anatomy conditioned prior residual 추가

### Phase 5 제한적 결합

각 축에서 가장 좋은 checkpoint를 결합하고 짧은 low learning rate fine tuning만 수행한다. 다음을 동시에 감시한다.

- residual correlation
- self shuffled gap
- SC CCC와 scale error
- pair Dice
- coverage와 overreach
- latent NLL

한 지표를 개선하면서 다른 축이 악화되면 joint loss weight를 계속 추가하기보다 두 단계 inference 또는 frozen module 결합을 유지한다.

## 10 실험 관리 표준

### 10.1 각 run에 저장할 설정

- data split과 subject ID 목록
- template 파일 hash
- edge mask와 normalization 통계
- model commit hash
- checkpoint와 optimizer state
- random seed
- loss weight
- subject batch와 edge sampling 방식
- prior temperature
- filtering threshold
- 평가 코드 버전

### 10.2 필수 결과 파일

```text
run_dir/
  config.yaml
  split.json
  template_stats.pt
  metrics_subject.csv
  metrics_group.json
  self_shuffle_zero.csv
  subject_similarity_matrix.npy
  residual_predictions.npz
  geometry_metrics.csv
  filtering_curve.csv
  checkpoint.pt
```

### 10.3 권장 Figure

1. GT와 예측 SC의 subject similarity matrix
2. template 제거 전후 correlation 비교
3. self, shuffled, zero T1 성능 분포
4. GT와 예측 residual scatter 또는 edge profile
5. temperature별 Dice, overreach, diversity curve
6. filtering threshold별 coverage overreach Pareto curve
7. oracle posterior와 prior generation의 대표 bundle 비교

## 11 Claude Code 구현 체크리스트

### 데이터 계층

- [ ] train split 전용 `SC_template_log1p` 생성
- [ ] residual std와 edge mask 저장
- [ ] validation과 test leakage unit test 작성
- [ ] subject pair sampler 작성
- [ ] self, shuffled, zero T1 evaluation loader 작성

### 모델 계층

- [ ] count head의 target을 absolute SC에서 normalized residual로 변경
- [ ] reconstruction 함수 `template + residual -> SC` 분리
- [ ] decoder와 prior freeze option 추가
- [ ] spatial feature map 반환 option 추가
- [ ] ROI mask pooling module 추가
- [ ] anatomy와 pair embedding norm logging 추가

### Loss 계층

- [ ] Smooth L1 residual loss
- [ ] numerically stable residual correlation loss
- [ ] subject difference loss
- [ ] 선택적 variance preservation loss
- [ ] absolute magnitude loss를 독립 flag로 제어

### Inference 계층

- [ ] subject loop 밖에서 RNG 초기화하거나 subject별 seed 명시
- [ ] cache key에 subject ID와 checkpoint hash 포함
- [ ] prior temperature argument 추가
- [ ] deterministic prior mean mode 추가
- [ ] raw와 filtered tractogram을 모두 저장

### 평가 계층

- [ ] 전체 SC와 residual SC 지표 분리
- [ ] template baseline 자동 계산
- [ ] self shuffled zero T1 paired comparison
- [ ] subject identification과 permutation test
- [ ] predicted inter subject similarity 분포
- [ ] pair Dice, MDF, endpoint, coverage, overreach 통합
- [ ] filtering retained rate 기록

## 12 최종 의사결정 규칙

### Case A Residual 학습 성공

조건:

- residual correlation이 안정적으로 상승한다.
- self T1이 shuffled 및 zero T1보다 우수하다.
- predicted subject variance가 GT 방향으로 회복된다.

결론:

- 기존 실패는 T1에 개인 정보가 없어서가 아니라 학습 target과 architecture가 population template shortcut을 허용했기 때문일 가능성이 높다.
- local anatomy conditioning과 subject conditioned count 또는 prior로 확장한다.

### Case B Residual 학습 실패

조건:

- end to end residual supervision에서도 residual correlation이 0 근처다.
- self T1과 shuffled T1 차이가 없다.

다음 조사:

- T1과 diffusion derived GT의 registration 정확도
- preprocessing에서의 과도한 spatial normalization
- GT SC의 test retest reliability
- tractography noise가 subject effect보다 큰지 여부
- sample size와 confound
- residual target의 신호 대 잡음비

이 경우 모델을 더 크게 만드는 것보다 먼저 데이터가 예측 가능한 개인 신호를 포함하는지 검증한다.

### Case C Temperature만으로 geometry 개선

조건:

- 낮은 temperature에서 pair Dice와 endpoint 지표가 개선되고 diversity가 유지된다.

결론:

- 단기 inference는 최적 temperature를 사용한다.
- 장기적으로 posterior scale을 더 잘 맞추는 prior calibration 또는 arch additive prior를 학습한다.

### Case D Filtering만으로 지표 개선

조건:

- overreach는 감소하지만 retained rate 또는 coverage가 급락한다.

결론:

- filtering은 증상을 줄였을 뿐 generator를 해결하지 못했다.
- raw generation 성능 개선을 계속하고, filter 결과만으로 모델 성능을 주장하지 않는다.

## 결론

가장 먼저 수행할 학습 실험은 **train template plus T1 predicted residual**이다. 이때 decoder와 prior를 고정하고 absolute SC loss를 끈 상태에서 residual supervision이 개인차를 살릴 수 있는지 확인한다. 동시에 재학습이 필요 없는 prior temperature sweep과 WM GM filtering을 독립적으로 수행해 geometry 개선 가능성을 측정한다.

실험의 핵심 성공 기준은 전체 SC correlation 하나가 아니다. 올바른 T1이 shuffled T1보다 우수하고, template을 제거한 residual을 예측하며, 생성 피험자 간 다양성이 GT 방향으로 회복되어야 한다. geometry에서는 oracle posterior와 generated prior 사이의 gap을 줄이고, filtering 뒤에도 충분한 coverage를 유지하면서 overreach를 낮춰야 한다.

이 순서를 지키면 현재 collapse가 학습 target, global pooling, prior mismatch, 해부학적 제약, 또는 데이터 자체의 낮은 개인 신호 중 어디에서 발생하는지 단계별로 판정할 수 있다.
