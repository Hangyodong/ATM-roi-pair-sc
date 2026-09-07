# ROI-pair Bundle을 SC Edge 단위에 맞추는 수정 전략
## Full Streamline + Edge-aligned Segment Dual Representation

> **핵심 목표**
>
> 기존 endpoint-defined ROI-pair bundle과 PASS-SC edge 정의가 어긋나는 문제를 줄이고,
> **SC matrix의 edge 값이 실제 streamline/segment 수와 더 직접적으로 대응**하도록
> bundle representation을 SC edge 단위에 맞게 수정한다.
>
> 최종 inference 목표:
>
> ```text
> T1-only
> → whole-brain TRK/TCK
> → SC weight
> → SC tract-length
> ```

---

# 1. 기존 Endpoint Bundle의 문제

현재 bundle 정의:

```text
Bundle(i,j)
=
endpoint가 ROI_i와 ROI_j인 streamline들의 집합
```

예를 들어:

```text
GT streamline
A → C → D → B
```

는 현재:

```text
A-B bundle
```

에만 포함된다.

하지만 PASS-SC에서는 중간에 통과하는 ROI 때문에:

```text
A-C
C-D
D-B
```

같은 edge에도 contribution이 생길 수 있다.

따라서:

```text
Endpoint Bundle
≠
PASS-SC Edge
```

가 된다.

---

# 2. 왜 이 불일치가 문제인가?

현재 구조에서는 A-B bundle을 잘 생성해도 streamline이 실제로:

```text
A → C → D → B
```

가 아니라:

```text
A → E → F → B
```

를 지나면 endpoint는 맞지만 PASS-SC는 달라진다.

즉:

```text
Bundle geometry learning unit
```

과:

```text
SC matrix edge unit
```

이 서로 다르다.

---

# 3. 특히 SUB-SUB에서 문제가 커지는 이유

현재 분석에서 SUB-SUB edge의 약 63%가 endpoint-defined SUB-SUB bundle이 아니라,
다른 endpoint streamline이 중간에 두 SUB ROI를 통과하면서 만들어지는 pass-edge다.

예:

```text
CTX1 → SUB3 → SUB7 → CTX8
```

현재 endpoint bundle:

```text
CTX1-CTX8
```

하지만 SC에서는:

```text
SUB3-SUB7
```

도 중요한 edge가 된다.

따라서 기존 endpoint bundle만으로는 SUB-SUB edge를 직접 학습하기 어렵다.

---

# 4. 수정 아이디어

원 streamline:

```text
A → C → D → B
```

를 SC edge 관점에서:

```text
A-C
C-D
D-B
```

로 분해한다.

각 인접 ROI transition을:

```text
Edge-aligned Segment Bundle
```

로 정의한다.

---

# 5. 새로운 Bundle 정의

## 기존

```text
Bundle = Endpoint ROI Pair
```

예:

```text
A-B Bundle
```

## 수정

```text
Bundle = SC Edge-aligned ROI Pair Segment
```

예:

```text
A-C Bundle
C-D Bundle
D-B Bundle
```

각 bundle은 해당 ROI pair 사이를 실제로 통과하는 streamline segment들의 집합으로 정의한다.

---

# 6. 가장 중요한 장점

이상적으로:

```text
Bundle(i,j)의 segment 수
≈
SC(i,j)의 edge 값
```

이 된다.

즉:

```text
SC edge
↔
Edge-aligned Segment Bundle
```

관계가 직접적이 된다.

---

# 7. 예시

GT whole-brain에서 C-D transition을 통과하는 segment가 327개라면:

```text
C-D Edge Bundle
=
327 segments
```

GT SC 정의가 동일한 transition-count 기반이면:

```text
SC[C,D] = 327
```

이 된다.

즉:

```text
Generated segment count
→ SC edge value
```

로 직접 연결할 수 있다.

---

# 8. 왜 SC realism에 유리한가?

기존 방식:

```text
Endpoint bundle 생성
↓
중간 route가 맞아야 함
↓
PASS-SC 계산
```

수정 방식:

```text
SC edge에 해당하는 segment를 직접 학습
↓
edge count를 직접 재현
```

즉 SC가 더 이상 2차적인 결과가 아니라 학습 단위 자체와 정렬된다.

---

# 9. 도로 비유

실제 도로:

```text
서울 → 대전 → 대구 → 부산
```

기존 endpoint 방식:

```text
서울-부산 Bundle = 1
```

하지만 교통 matrix가 구간별 연결량을 본다면:

```text
서울-대전
대전-대구
대구-부산
```

를 직접 세는 것이 더 자연스럽다.

SC도 같은 논리다.

---

# 10. SUB-SUB 문제 해결

예:

```text
CTX1 → SUB3 → SUB7 → CTX8
```

기존에는:

```text
CTX1-CTX8 Bundle
```

만 존재한다.

수정 후:

```text
CTX1-SUB3 Bundle
SUB3-SUB7 Bundle
SUB7-CTX8 Bundle
```

이 된다.

따라서 SUB3-SUB7이 더 이상 다른 bundle이 지나가서 간접적으로 생기는 edge가 아니라
직접 학습되는 edge-bundle이 된다.

---

# 11. Route Loss의 역할 변화

기존 구조:

```text
Endpoint Bundle
+
Route Loss
```

수정 구조:

```text
A-C
C-D
D-B
```

를 직접 edge-bundle로 학습하므로
Route Loss의 중요성은 크게 줄어든다.

즉 edge-aligned segment bundle 자체가 route supervision을 구조적으로 포함한다.

---

# 12. Segment-only 구조의 문제

원 streamline:

```text
A → C → D → B
```

는 하나의 연속된 white-matter trajectory다.

이를:

```text
A-C
C-D
D-B
```

만으로 학습하면 최종적으로 full streamline continuity를 복원하는 문제가 남는다.

즉:

```text
SC realism ↑
BUT
Full tractography continuity ↓
```

위험이 있다.

---

# 13. 추천: Dual Representation

원 GT streamline을 버리지 않는다.

하나의 GT streamline에서:

```text
Full Streamline
+
Edge-aligned Segments
```

를 동시에 만든다.

## Branch 1 — Full Streamline

```text
A → C → D → B
```

용도:
- complete streamline geometry
- tract continuity
- whole-brain TRK
- tract length

## Branch 2 — Edge Segments

```text
A-C
C-D
D-B
```

용도:
- SC edge supervision
- edge count
- bundle balancing
- GESTA augmentation
- SUB-SUB direct learning

---

# 14. 최종 구조

```text
                    GT Full Streamline
                    A → C → D → B
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     Full Streamline Branch      Edge Segment Branch
              │                         │
        A → C → D → B             A-C / C-D / D-B
              │                         │
      Geometry / Length            Edge Count / SC
              │                         │
              │                  GESTA Balancing
              │                         │
              └────────────┬────────────┘
                           ▼
                    Whole-brain Output
                           ↓
                      SC Matrix
```

---

# 15. GESTA 적용 위치도 변경

기존 GESTA balancing은 endpoint-defined bundle size를 기준으로 했다.

수정 후에는:

```text
SC edge-aligned segment bundle size
```

를 기준으로 한다.

예:

```text
A-C : 5000 segments
C-D :   40 segments
D-B :  700 segments
```

이면:

```text
A-C → downsample
C-D → GESTA synthetic augmentation
D-B → normal sampling
```

---

# 16. 왜 이 GESTA 적용이 더 직접적인가?

기존에는 small endpoint bundle을 보강했지만,
그 bundle이 반드시 small SC edge와 일치하지 않았다.

수정 후에는:

```text
small edge-bundle
```

을 직접 보강한다.

즉:

```text
GESTA augmentation
→ small SC edge geometry representation 보완
```

이 된다.

---

# 17. GESTA synthetic generation

각 low-count edge bundle:

```text
C-D real segments
↓
ATM ES
↓
Latent Seeds
↓
KDE / Parzen
↓
Gaussian / GMM Proposal
↓
Rejection Sampling
↓
Synthetic Latent
↓
ATM DS
↓
Synthetic C-D segments
↓
Filtering
```

---

# 18. Synthetic Count와 GT SC는 분리

예:

```text
GT C-D SC = 40
```

인데 training balance를 위해:

```text
200 synthetic C-D segments
```

를 만들었다고 해도:

```text
GT SC = 40
```

은 유지해야 한다.

즉:

```text
Training Pool Size
≠
SC Target
```

이다.

---

# 19. Weight Head를 다시 생각해야 하는 이유

기존 구조에서는 candidate count와 SC strength가 달라 Weight Head가 필요했다.

하지만 새 구조에서 GT SC가 실제로 edge를 구성하는 streamline/segment 수라면:

```text
SC_ij
=
Number of generated valid segments
```

가 더 자연스럽다.

따라서 Weight Head 대신:

```text
Edge Count Head
```

를 우선 고려할 수 있다.

---

# 20. Edge Count Head

입력:

```text
Subject anatomy
+
ROI_i
+
ROI_j
```

출력:

```text
N_hat(i,j)
=
해당 SC edge에서 생성해야 할 segment 수
```

---

# 21. Inference 예

```text
T1
↓
Anatomy Encoder
↓
Edge Count Head
↓
C-D = 42 segments
↓
C-D Segment Generator
↓
42 valid segments 생성
↓
SC[C,D] = 42
```

즉:

```text
Predicted edge count
=
Generated segment count
=
SC edge value
```

가 된다.

---

# 22. Edge Count Loss

GT:

```text
N_GT(i,j)
```

Pred:

```text
N_pred(i,j)
```

Loss 후보:

```text
L_count =
MAE(
    log(N_pred + eps),
    log(N_GT + eps)
)
```

또는 count distribution에 맞춰:
- Poisson loss
- Negative Binomial loss

를 비교할 수 있다.

---

# 23. SC correlation은 유지

count를 직접 맞춰도 whole-brain topology 평가를 위해:

```text
Overall
CTX-CTX
CTX-SUB
SUB-SUB
```

SC correlation은 유지한다.

예:

```text
L_SC_corr =
λ_global * (1-r_all)
+
λ_type * [
    (1-r_CC)
  + (1-r_CS)
  + (1-r_SS)
] / 3
```

---

# 24. 권장 최종 Loss

## Full Streamline Branch

```text
L_full =
L_recon
+
λ_geom * L_geometry
+
λ_endpoint * L_endpoint
+
λ_len * L_full_length
```

## Edge Segment Branch

```text
L_edge =
λ_count * L_edge_count
+
λ_seggeom * L_segment_geometry
+
λ_sc * L_SC_corr
+
λ_mag * L_SC_logmag
```

## Total

```text
L_total =
L_full
+
λ_edgebranch * L_edge
```

---

# 25. Route Loss는 선택적

새 edge-aligned segment branch가:

```text
A-C
C-D
D-B
```

를 직접 학습하므로 기존 ROI-visitation route loss는 필수성이 줄어든다.

추천:

```text
Baseline:
Edge Segment Branch

Ablation:
+ Route Loss
```

---

# 26. 가장 중요한 전제: GT SC 정의 확인

이 전략의 핵심은:

> **현재 GT SC matrix의 edge 값이 어떤 streamline/pass 정의로 계산되는가?**

이다.

edge-bundle definition을 GT SC 생성 정의와 정확히 맞춰야 한다.

---

# 27. Case A — 인접 ROI transition count

만약:

```text
A → C → D → B
```

streamline에서 GT SC가:

```text
A-C
C-D
D-B
```

만 count한다면 현재 제안이 정확히 맞는다.

---

# 28. Case B — Same-streamline all-pairs count

만약 같은 streamline에서 통과한 모든 ROI pair를 연결로 센다면:

```text
A-C
A-D
A-B
C-D
C-B
D-B
```

까지 contribution될 수 있다.

이 경우 단순히:

```text
A-C
C-D
D-B
```

만 segment로 정의하면 GT SC를 완전히 재현할 수 없다.

따라서 현재 `.mat` 생성 정의 확인이 필수다.

---

# 29. 권장 Preprocessing 검증

streamline 하나:

```text
A → C → D → B
```

를 선택하고 GT SC 생성 코드에서 어떤 edge들이 +1 되는지 직접 추적한다.

확인:

```text
[ ] A-C
[ ] C-D
[ ] D-B
[ ] A-D
[ ] A-B
[ ] C-B
```

---

# 30. GT 정의에 따른 Edge Bundle

## Adjacent-transition 방식이면

```text
Edge Bundle(i,j)
=
ROI_i에서 ROI_j로 직접 이어지는 segment
```

## All-pass-pair 방식이면

```text
Edge Bundle(i,j)
=
같은 streamline에서 ROI_i와 ROI_j의 연결에 기여하는
sub-path / contribution
```

로 정의해야 한다.

---

# 31. Edge-aligned Preprocessing 예

원 streamline:

```text
p1 ... p128
```

ROI visitation sequence:

```text
A A A C C C D D D B B
```

연속 중복 제거:

```text
A → C → D → B
```

그 후:

```text
A-C segment
C-D segment
D-B segment
```

추출.

---

# 32. Segment coordinate 추출

각 transition마다 실제 streamline의 해당 ROI 경계 사이 좌표를 잘라낸다.

구현 시 경계 정의를 모든 subject에서 동일하게 해야 한다.

---

# 33. Segment Resampling

각 segment를 fixed N points로 resampling할 수 있다.

예:

```text
32 / 64 / 128 points
```

full streamline과 동일하게 128을 쓸지,
segment에는 더 적은 point 수를 쓸지는 validation으로 결정한다.

---

# 34. 매우 짧은 Segment 처리

인접 ROI가 붙어 있으면 segment가 매우 짧을 수 있다.

따라서:

```text
minimum segment length
```

QC가 필요하다.

너무 짧으면:
- interpolation artifact
- atlas boundary jitter
- geometry learning 의미 저하

가능성이 있다.

---

# 35. Atlas Boundary Jitter

ROI 경계에서:

```text
A ↔ C ↔ A ↔ C
```

처럼 label oscillation이 생길 수 있다.

따라서 visitation sequence preprocessing에:
- minimum dwell length
- label smoothing
- short transition removal

등이 필요할 수 있다.

---

# 36. 최종 추천 Architecture

```text
                       T1
                        ↓
                 Anatomy Encoder
                        ↓
               Subject Anatomy Feature
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
 Full Streamline Branch        Edge Segment Branch
          │                           │
 Endpoint pair condition        ROI_i ↔ ROI_j edge
          │                           │
 Complete streamline            Edge Count Head
          │                           │
 Geometry / Length             Segment generation
          │                           │
          │                      GESTA balancing
          │                           │
          └─────────────┬─────────────┘
                        ↓
                 Whole-brain TRK
                        ↓
                    SC Matrix
```

---

# 37. 핵심 개념

```text
Full Streamline
=
Tractography realism

Edge Segment Bundle
=
SC realism
```

둘을 동시에 유지한다.

---

# 38. 기존 구조와 수정 구조 비교

| 항목 | 기존 | 수정 |
|---|---|---|
| Bundle 기준 | Endpoint ROI pair | **SC edge-aligned segment** |
| 예 | A-B | **A-C / C-D / D-B** |
| 중간 ROI | bundle 내부에 숨음 | **직접 edge로 분리** |
| SUB-SUB pass-edge | 간접 생성 | **직접 학습 가능** |
| SC edge 대응 | 불완전 | **직접 대응** |
| GESTA 적용 | endpoint bundle | **small edge-bundle** |
| SC strength | Weight Head | **Edge Count Head 우선 고려** |
| Full TRK continuity | 유지 | **Dual branch로 유지** |
| Route loss | 중요 | **선택적/보조적** |

---

# 39. 필수 Ablation

## A. 기존
```text
Endpoint bundle
+ Weight Head
+ PASS-SC
```

## B. Edge Segment
```text
Edge-aligned segment bundle
+ Edge Count Head
```

## C. Dual Representation
```text
Full Streamline
+
Edge Segment
```

## D. Final
```text
Dual Representation
+
GESTA edge balancing
+
Block-wise SC loss
```

---

# 40. Validation

## SC
- Overall Pearson
- CTX-CTX
- CTX-SUB
- SUB-SUB
- MAE
- RMSE
- CCC

## Edge Count
- `corr(N_pred, N_GT)`
- MAE(log count)
- edge recall
- zero-edge specificity

## Segment
- segment endpoint accuracy
- segment length
- segment geometry
- WM occupancy

## Full Tractography
- complete streamline geometry
- tract length
- whole-brain coverage
- continuity

---

# 41. GESTA 효과 평가

특히 low-count edge bundle에서:

```text
No augmentation
vs
Duplicate oversampling
vs
GESTA augmentation
```

비교.

평가:
- geometry diversity
- edge count prediction
- SC correlation
- SUB-SUB performance

---

# 42. 구현 전 가장 먼저 확인할 것

> **현재 GT SC matrix의 edge 값이 정확히 어떤 streamline/pass 정의로 계산되는가?**

이 정의가 최종 edge-bundle definition의 ground truth가 된다.

---

# 43. 최종 결론

기존 endpoint bundle:

```text
A-B bundle
```

은 SC 관점에서:

```text
A-C
C-D
D-B
```

같은 중간 pass-edge를 직접 표현하지 못한다.

따라서 SC realism을 높이려면:

```text
Bundle definition
→ SC edge-aligned segment
```

로 맞추는 것이 더 직접적이다.

다만 complete streamline continuity를 잃지 않도록:

```text
Full Streamline Branch
+
Edge Segment Branch
```

의 dual representation을 권장한다.

최종 역할:

```text
Full Streamline
→ TRK realism

Edge-aligned Bundle
→ SC realism

GESTA
→ low-count edge-bundle imbalance 완화

Block-wise SC loss
→ CC / CS / SS 편향 완화

Edge Count Head
→ SC edge magnitude 직접 예측
```
