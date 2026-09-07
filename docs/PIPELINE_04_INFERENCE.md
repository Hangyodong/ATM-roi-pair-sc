# 추론 — T1 만으로 TRK + SC 만들기

`src/atm_sc/inference/generate_sc.py`, `src/atm_sc/inference/latent_bank.py`

GT 는 지표 계산에만 쓰고 생성에는 일절 쓰지 않는다.

---

## 1. 전체 흐름

```
T1 [1,1,193,229,193]
  ↓ encode_anatomy
a [1,512]
  ↓ ① select_pairs      (edge head, thr=0.5)      → 생성할 ROI 쌍 K개
  ↓ ② template_counts   (train 평균 sc_end 비율)   → 쌍마다 몇 가닥
  ↓ ③ LatentBank.sample (train streamline KDE)    → z [N,64]
  ↓ ④ decode(z, cond)                             → streamline [N,128,3] mm
  ↓ ⑤ hard_sc(..., mode="pass")                   → SC [82,82] + 평균 길이
```

## 2. ① 생성할 ROI 쌍 고르기

```python
pairs, prob = select_pairs(model, anatomy, n_roi=82, thr=0.5)
```

edge head 확률 > 0.5 인 쌍. test 평균 **1,812 쌍** 선택.

GT 와 무관한 대안: **템플릿에서 `sc_end ≥ 1` 인 쌍** (2,849개). 실측상 두 방식의 SC 상관이
0.880 vs 0.882 로 사실상 같다 — edge head 가 GT pair 목록에 기대고 있지 않다는 확인이다.

## 3. ② 배분 — 쌍마다 몇 가닥을 만들 것인가

**가장 큰 영향 요인이다.** 같은 pair 목록·같은 총 가닥 수로 고정한 실측 (test 4명):

| 기하 \ 배분 | 균등 | **GT 배분(오라클)** | `count_head_end` 예측 |
|---|---|---|---|
| GT 기하 (실제 streamline) | 0.824 | **0.969** | 0.597 |
| 생성 기하 | 0.678 | 0.863 | 0.567 |

- 배분 효과 (균등 → GT): **+0.185**
- 기하 효과 (GT 기하 → 생성 기하): **−0.146**
- **학습된 `count_head_end` 는 균등보다도 나쁘다** (−0.111)

`count_head_end` 가 망가진 이유: **로그 공간에서 1.7~1.9배 과분산**이다.

| subject | std(log GT) | std(log 예측) | 배율 |
|---|---|---|---|
| sub-000004 | 2.241 | 4.136 | 1.85× |
| sub-000005 | 2.234 | 3.932 | 1.76× |

`exp()` 후 극단적 편중이 된다:

| 배분 | 상위 10 % 쌍에 몰린 비율 | 지니 | GT 배분과 상관 |
|---|---|---|---|
| GT | 0.72 | 0.83 | 1.000 |
| **예측** | **0.94** | 0.94 | **0.09** |
| 균등 | 0.10 | 0.00 | 0.00 |

전체 가닥의 94 % 를 상위 10 % 쌍에 쏟아붓는데 그 쌍 선택이 거의 무작위다.
원인은 `edge_count_loss` 가 log1p MAE 라 **분산 보정 압력이 없다**는 것.
`sc_end` 는 `sc_pass` 보다 16배 희소하고 꼬리가 길어 그대로 터졌다.

### 현재 채택: 그룹 템플릿 배분

```python
n(i,j) = round( 템플릿_sc_end(i,j) / Σ템플릿 * total )
```

train 144명의 `sc_end` 평균 비율을 그대로 쓴다. 학습이 전혀 필요 없다.

| 배분 | 배분 자체 정확도 | SC 상관 |
|---|---|---|
| 균등 | — | 0.679 |
| **템플릿** | **0.836** | **0.853** (오라클의 **99 %**) |
| 템플릿 × pass헤드 변조 | 0.714 | 0.851 |
| GT (오라클) | 1.000 | 0.863 |

**pass head 로 변조하면 오히려 나빠진다** (배분 정확도 0.836 → 0.714). count head 가
개인 정보를 담고 있지 않으므로 곱해봐야 잡음만 더한다. 순수 템플릿을 쓴다.

### 총 가닥 수는 상관에 영향이 없다

| 총 가닥 | SC 상관 |
|---|---|
| 14,497 (pair 당 8개) | 0.680 |
| 100,000 (균등 51개) | 0.678 |

**6.5배 늘려도 변화가 없다.** 4명 전원 일관. 다만 CCC(절대 크기)는 총합에 직접 좌우되므로
GT 규모(**460,000**, train 평균 459,716)에 맞춰야 한다.

## 4. ③ latent — 사전분포 대신 bank

`src/atm_sc/inference/latent_bank.py`

`z ~ N(mu_pair, I)` 로 뽑으면 디코더가 실제 다발이 없는 자리를 그린다.
train subject 의 실제 streamline 을 인코더로 통과시켜 pair 별로 모아두고
그 주변에서 KDE 샘플링한다.

```python
bank = build_bank(model, train[:12], bundle_path, n_per_pair=24)
z, hit = bank.sample(pairs, rng)          # 전부 벡터화 (46만 가닥도 한 번에)
# bank 에 없는 pair 만 N(mu_pair, I) 로 채운다
```

**대역폭**: Silverman, 차원마다 `std * n^(-1/(D+4))`.

실측 (test 3명, 균등 배분, pair 당 16개):

| latent 출처 | SC 상관 | pair 별 복셀 dice |
|---|---|---|
| `prior` = N(mu_pair, I) | 0.669 | 0.053 |
| **`bank`** = train KDE | **0.726** | 0.090 |
| `post` = 실제 streamline latent (오라클) | 0.702 | 0.110 |
| 실제 16개 vs 다른 실제 16개 (**천장**) | — | **0.597** |

두 가지가 드러난다.

1. **KDE bank 가 오라클 posterior 보다 좋다** — 오라클은 특정 가닥 하나에 대응해 다양성이
   좁은 반면 KDE 는 주변을 퍼뜨려 번들 전체를 더 넓게 덮는다. SC 는 개별 가닥의 정확도보다
   번들의 공간적 분포에 좌우된다.
2. **오라클 latent 를 줘도 dice 가 천장의 28 % 다** — 사전분포가 아니라 **디코더 자체가
   병목**이다. 복원 오차 3.55 mm 인데 아틀라스 복셀이 2 mm 라 1.8칸이 어긋난다.

bank 는 **모든 subject 에게 동일**하다 (train 으로만 만든다). 따라서 개인차를 담지 못한다.

## 5. ④⑤ 생성과 SC 추출

```python
sc, n_total, S = generate_by_count(model, anatomy, pairs, counts, atlas, affine,
                                   n_roi=82, bank=bank, keep=True)
```

메모리 절약을 위해 20,000 가닥씩 나눠 생성하고 SC 를 누적한다 (46만 가닥 × 128 × 3 은 700 MB).

SC 추출은 학습의 soft 규칙이 아니라 **hard 규칙**을 쓴다:

| mode | 정의 |
|---|---|
| `pass` | 통과한 ROI 들의 **모든 쌍** (= GT `.mat` 정의) |
| `end` | 양 끝점만 (= `sc_end` 정의) |

## 6. 추론 사전정보 만들기

```bash
python scripts/34_build_inference_prior.py --n-bank-subj 12
```

train 으로만 만들고 test 는 일절 보지 않는다. 산출물:

| 파일 | 내용 |
|---|---|
| `outputs/inference/template.npz` | `sc_end`, `sc_pass` train 평균 |
| `outputs/inference/latent_bank.npz` | 3,087 pair / 348,456 latent / D=64 |

안전장치: bank 가 생성 대상 pair 의 90 % 미만을 덮으면 assert 로 중단한다.
덮이지 않은 pair 는 조용히 사전분포로 떨어져 성능이 새는데 눈에 띄지 않기 때문이다.

## 7. 실행

```bash
python scripts/29_final_evaluation.py \
  --ckpt outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt \
  --by-count --use-bank --total-streamlines 460000 \
  --template-subjects outputs/splits/train.txt --trk-eval
```

검증 스크립트: `scripts/32_verify_bank_inference.py` (31명, edge head / 템플릿 pair 두 방식 비교).

## 8. 개선 요약

| 구성 | SC r | CCC | 길이 RMSE | 길이 r |
|---|---|---|---|---|
| 학습된 그대로 (prior + 균등) | 0.711 | 0.682 | 55.0 mm | 0.566 |
| **bank + 템플릿 배분** | **0.880** | 0.737 | **41.3 mm** | **0.735** |

재학습 없이 추론 코드만 바꿔 **+0.17**. `count_head_end` 와 `allocate_counts` 는 지우지 않고
남겨뒀다 — 과분산을 고치면 템플릿을 넘을 여지가 있고, 지금 지우면 비교 근거가 사라진다.
