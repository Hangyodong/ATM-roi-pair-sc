# 학습

`src/atm_sc/training/trainer.py`, `src/atm_sc/training/run.py`

---

## 1. 한 step 의 구조

anatomy 는 subject 당 **1회**만 계산하고 leaf 로 분리한다. 여러 손실이 각자 backward 하되
UNet backward 는 마지막에 **1회**만 한다 — stage1 이 193³ 에서 64채널이라 여러 번 흘리면
메모리가 터진다.

```
a_full = encode_anatomy_grad(T1)          # 1회
a_leaf = a_full.detach().requires_grad_()  # 여기서 분리, dL/da 를 누적

[G] 생성 pass   generate(z~prior, cond) → endpoint / route_gen / gen_length / SC / count 가중
[R] 복원 pass   encode(GT) → decode → recon / kl / geom / route
[S] segment     SC edge segment 에 대해 [R] 과 같은 것 (mode=1)
[C] count       count_head, count_head_end
[E] edge        edge_head

a_full.backward(dLda)                      # UNet backward 1회
```

각 pass 뒤 `dL/da` 를 누적하고(`take_dLda`), gradient norm 을 기록해 어느 항이 지배하는지
로그로 볼 수 있게 한다.

## 2. 손실 함수 전체

`LossWeights` 기본값. 실제 값은 phase 별 config 가 덮어쓴다.

### 기하 (streamline 모양)

| 이름 | 가중치 | 정의 |
|---|---|---|
| `recon` | 1.0 | mm 단위 RMSE, **방향 대칭** (뒤집힌 순서와 비교해 작은 쪽) |
| `kl` | 0.1 | 조건부 사전분포 `N(mu_pair, I)` 로의 KL |
| `geom` | 1.0 | 인접 점 간격 균일성 (`adjacency_loss`) |
| `seg_recon/kl/geom` | 동일 | SC edge segment 분기 (mode=1) |

### 경로 / 끝점

| 이름 | 가중치 | 정의 |
|---|---|---|
| `endpoint` | 1.0 | 양 끝점이 목표 ROI 에 오도록. log-softmax, τ=5 |
| `seg_endpoint` | 0.5 | segment 용 |
| `route` | 1.0 | **복원** streamline 이 지나는 ROI 집합 BCE (per-streamline GT) |
| `route_gen` | 0.5 | **생성** streamline 의 통과 ROI (pair 별 GT 통과 비율이 target) |
| `gen_length` | 1.0 | `|log1p(L_pred) − log1p(L_gt_pair)|` |

`route` 는 log 확률 공간에서 계산한다 (`losses/route.py`). `visit_log_probs(mm, tau)` 는
`point_log_probs(...).amax(dim=1)` — streamline 의 어느 점이든 그 ROI 에 가까우면 방문으로 본다.

**`gen_length` 가 없으면 안 된다.** route loss 는 "많은 ROI 를 지나라"고 요구하는데,
길이 제약이 없으면 모델이 **길게 헤매는 쪽으로 도망간다**. 실측: 생성 길이가
111 mm → **323 mm** 로 폭주했고 (GT 103 mm), `route_pos_weight` 를 5 → 2 로 낮추고
`gen_length` 를 넣어 잡았다.

### SC 값

| 이름 | 가중치 | 정의 |
|---|---|---|
| `corr` | 0.5 | **4-term**: ctx-ctx / ctx-sub / sub-sub 블록별 로그 Pearson 평균 + λ·전체 |
| `mag` | 0.5 | log1p L1. `normalize='none'` (s3 부터) 이면 절대 스케일까지 본다 |
| `scale` | 1.0 | `|log Σpred − log Σgt|` — 전역 배율. 실측: 배율 하나로 CCC 0.024 → 0.817 |
| `rmse` | 0.2 | GT 표준편차로 무차원화한 SC RMSE, 블록 균형 |
| `count` | 1.0 | log1p MAE, 블록 균형. `count_head` 와 `count_head_end` 둘 다 |
| `presence` | 0.5 | `1 − exp(−SC/scale)` 에 대한 BCE (기본 sub-sub) |
| `edge` | 0.5 | edge 존재 BCE |
| `length` | 0.2 | tract 길이 행렬 |

`sc_global_weight=0.5`, `sc_groups='block'`.

**절대 스케일 안전장치**:

```python
absolute = (cfg.sc_mag_normalize == "none" and "mag" in cfg.active) or bool(cfg.active & {"scale","rmse"})
assert not (absolute and cfg.max_pairs_per_step is not None), \
    "절대 스케일 손실은 pair 부분집합에서 계산할 수 없다"
```

## 3. 5단계 phase

`configs/pipeline_route.yaml`. baseline 의 sc_corr 체크포인트에서 이어받는다.

| # | config | phase | steps | 새로 켜는 것 |
|---|---|---|---|---|
| s1 | `route/s1_route.yaml` | `seg_route` | 3000 | route + 균형 노출 + GESTA + segment 분기 + count head |
| s2 | `route/s2_presence.yaml` | `seg_route_presence` | — | presence (sub-sub 존재 여부) |
| s3 | `route/s3_mag.yaml` | `seg_count_mag` | — | magnitude 절대 스케일 (`normalize='none'`), scale, rmse |
| s4 | `route/s4_length.yaml` | `seg_full` | — | tract length |
| s5 | `route/s5_joint.yaml` | `seg_full` | 4000 | 전체 joint, LR 을 1/3 로 낮춤 |

**학습률** (파라미터 그룹별):

| 그룹 | s1 | s5 |
|---|---|---|
| `lr_t1` (UNet) | 1e-5 | 5e-6 |
| `lr_vae_enc` | 3e-5 | 1e-5 |
| `lr_dec` | 3e-5 | 1e-5 |
| `lr_heads` | 1e-4 | 3e-5 |

`grad_clip=50`, `seed=0`, `gt_pairs_per_step=128`, `n_gen_per_pair=8`, `chunk=2048`.

**드라이버 주의**: phase 경계에서 코드를 고쳤으면 **드라이버를 반드시 재시작**해야 한다.
실행 중인 드라이버는 옛 코드를 들고 있어 새 config 필드에서
`TypeError: TrainConfig.__init__() got an unexpected keyword argument` 로 죽는다.
체크포인트는 정확히 이어지므로 재시작 비용은 없다 (`scripts/30_restart_driver.sh`).

## 4. 균형 샘플링 — 큰 번들 편향 방지

`src/atm_sc/data/balanced_pair_sampler.py`

SC edge 값은 1 ~ 수만으로 250:1 까지 차이난다. 그대로 샘플링하면 큰 번들만 학습된다.

```python
B_k = clip(100 * (N_k/100)**0.5, 16, 256)     # alpha=0.5
```

노출 비율 **250:1 → 5.7:1**. segment 쪽도 같은 방식(`segment_sampler.py`, 실측 5.40).

강도 구간 (`roi_groups.tier_of_strength`, `TIER_EDGES=(100, 1000)`):
약함 < 100 ≤ 중간 < 1000 ≤ 강함.

`BalanceConfig`: `alpha=0.5, b_base=100, b_min=16, b_max=256, real_fraction_min=0.5,
max_synthetic_ratio=4.0, min_seed_count=20, lambda_syn=0.5`.

## 5. GESTA synthetic 증강

`src/atm_sc/generative/` — 가닥이 적은 번들을 합성으로 채운다.

```
ATM 인코더 ─▶ latent seed ─▶ KDE 샘플링 ─▶ 디코더 ─▶ QC 필터 ─▶ synthetic streamline
```

- **거부 샘플링은 못 쓴다**: 64차원에서 수락률이 ~1e-11. `latent_sampler.py` 가
  수락률 < 1e-3 이면 assert 로 중단한다.
- 가닥이 적은 번들은 **cross-subject 통합 latent bank** 를 쓴다.
- `min_seed_count=20` 미만이면 증강하지 않는다.

**QC 임계값은 실제 TRAIN 분포에서 뽑았다** (60,000 가닥, `scripts/27_qc_thresholds.py`):

| 기준 | 값 |
|---|---|
| 길이 | 19.3 ~ 241.5 mm |
| 최대 굴절각 | ≤ 44.8° |
| 총 감김 | ≤ 1265° |
| 끝점 간 거리 / 길이 | ≥ 0.15 |
| 뇌 안 비율 | ≥ 0.95 |
| 중복 제거 허용오차 | 0.1 mm |

실제 streamline 의 통과율 0.947. **synthetic 통과율은 0.12 로 낮고 병목은 곡률(0.26)**
이다 — 실제 p99 최대 굴절이 44.8° 인데 우리 임계값이 60° 였으므로, 임계값이 실제보다
느슨한데도 떨어진다는 것은 생성 가닥이 진짜로 실제 분포 밖이라는 뜻이다.

## 6. 중요도 가중 (`weight_mode='count'`)

pair 당 8개만 생성하지만, 학습 시점의 SC 가 100만 가닥 전체의 불편 추정이 되도록
각 가닥에 `N̂_end(pair)/n_gen` 을 곱한다.

```python
if cfg.weight_mode != "head":
    nh = m.edge_log_counts_end(a_leaf, pc).exp() / max(cfg.n_gen_per_pair, 1)
    w = nh if cfg.weight_mode == "count" else w * nh
```

**부작용**: 이렇게 하면 자유 `weight_head` 가 gradient 를 받지 못해 초기값 ~1.0 에 머문다.
그런데 검증/추론의 `tractogram_sc` 는 그 head 를 쓴다 → val 총합비 0.002, CCC 0.002.
학습과 평가가 서로 다른 것을 재는 상태다 (`PIPELINE_06_FINDINGS.md` §3).

## 7. 선점 (A10 ↔ H100 인계)

`preempt_file` 이 생기면 `Preempted` 예외를 던지고 체크포인트를 저장한 뒤 대기한다.
10 step 마다 확인한다. 큐가 열리면 H100 으로 옮겨 이어서 학습한다.

## 8. 모니터링

```bash
python scripts/20_monitor.py
```

phase 진행, config, 현재 step 상세, val 결과, 블록/구간별 지표를 표로 보여준다.
선점 상태도 표시한다.

## 9. 실행

```bash
python scripts/19_train_pipeline.py --config configs/pipeline_route.yaml
```

마지막 phase 가 끝나면 `final_eval: true` 로 test 31명 T1-only 평가를 1회 자동 실행한다
(OOM 이 나면 표본을 줄여 수동 재실행).
