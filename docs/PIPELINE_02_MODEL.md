# 모델 구조

`src/atm_sc/models/roi_atm.py` — 클래스 `ROIPairATM`

---

## 1. 전체 구조

```
[T1 | WM] [1,2,193,229,193]      ch0 rigid T1 [0,1] · ch1 WM 확률
   │
   └─ rigid_UNet 인코더 ────────────────────▶ anatomy  a [1,512]
                                                 │
(ROI_i, ROI_j) ─ Emb ─▶ pair_vec [N,64] ─ Proj ─┤
                                                 ▼
                cond [N,512] = anatomy_gain·LayerNorm(a) + Proj(pair_vec) + mode_emb[mode]
                                                 │
   GT streamline [N,128,3] ─ ConvVAE 인코더(cond) ─▶ (mu, logvar) ─▶ z [N,64]   (학습 전용)
                              또는  z ~ N(mu_pair, I) / latent bank              (추론)
                                                 │
                            ConvVAE 디코더(z, cond) ─▶ [N,3,128] tanh
                                                 │ 역정규화 (affine)
                                                 ▼
                                   streamline [N,128,3] mm
```

상수: `ANATOMICAL_DIM=512`, `LATENT_DIM=64`, `N_POINTS=128`, `emb_dim=64`, `n_roi=82`.

ATM 상류(30개 번들별 모델) 중 하나(`init_bundle="AF_L"`)의 ConvVAE 가중치로 초기화한다.
30개를 평균하는 것은 서로 독립 학습된 모델이라 의미가 없다.

## 2. anatomy feature 추출 — T1 → [1, 512]

`src/atm_sc/models/atm_adapter.py`

```
[T1 | WM] [1,2,193,229,193]
 stage1  conv1_1 → conv1_2 → conv1_3, 잔차합       64ch  @ 193³
 stage2  conv2_1 → conv2_2 → conv2_3, 잔차합      128ch  @  97³
 stage3  conv3_1 → conv3_2 → conv3_3, 잔차합      256ch  @  49³
 stage4  conv4_x → 전역 pooling → fc
 → a [1,512]
```

**인코더 가지만 실행한다.** 상류 `rigid_UNet` 은 segmentation 디코더 가지도 갖지만
그 가지는 anatomical condition 에 영향을 주지 않고(상류 `infer.py` 도 결과를 버린다),
전체 forward 는 193³ 에서 128채널 concat(4.4 GB)까지 만들어 A10 23GB 에서 OOM 이다.
수치 동일성은 `scripts/00_check_env.py` 가 CPU 전체 forward 와 대조해 확인한다.

**중요한 구조적 사실**: subject 당 벡터 **하나**다. 이 512차원이 3,321개 ROI 쌍 전부에
**같은 값**으로 들어간다. 따라서 anatomy 는 전체 레벨만 움직일 수 있고 쌍별 패턴을 바꿀 수 없다.

**학습 경로** (`encode_anatomy_grad`): `unet_level` 로 어디부터 gradient 를 흘릴지 정한다.

| `unet_level` | 학습되는 stage |
|---|---|
| `none` | 없음 (동결, 디스크 캐시 사용) |
| `stage4` | conv4_x + fc |
| `stage3` / `stage2` | 그 이상 |
| `full` | stage1 부터 전부 (현재 설정) |

메모리 때문에 학습 stage 는 `torch.utils.checkpoint` 로 감싼다. stage1 은 193³ 에서
64채널 텐서가 개당 2.2 GB 라 checkpoint 없이는 23 GB 에서 backward 가 불가능하다.
**subject 당 step 마다 1회만 호출**하고, trainer 가 leaf 로 분리해 backward 도 1회만 한다.

`unet_level='stage4'` 일 때는 stage3 출력 [1,256,49,58,49] 을 fp16 으로 디스크 캐시(72 MB)해
매 step 앞단을 다시 돌리지 않는다 (`cache_stage3`).

## 3. streamline feature 추출 — TRK → [N, 64]

```python
def encode_streamline(mm, a):          # mm [N,128,3]
    s = (mm - coord_min) / coord_scale - 1.0    # → [-1,1]
    return ae.encode(s.permute(0,2,1), a)       # → (mu [N,64], logvar [N,64])

z = mu + randn_like(mu) * exp(0.5*logvar)       # reparameterize
```

**학습 때만 존재한다.** 복원 손실을 계산하려면 GT streamline 을 latent 로 보내야 한다.
추론에는 그 subject 의 streamline 이 없으므로 사전분포나 latent bank 에서 뽑는다
(`PIPELINE_04_INFERENCE.md`).

좌표 정규화 상수 `coord_min/coord_scale` 은 **MNI152NLin6 brain mask 의 bounding box**(+5mm)
에서 계산한다. 상류의 번들별 상수는 whole-brain 을 덮지 못한다 (AF_L 은 x ≤ −3.7).
따라서 **학습 시작 시점의 출력 geometry 는 pretrained AF_L 을 whole-brain 박스로 늘린 것**이고
복원 손실이 이를 다시 맞춘다. 이 사실은 Methods 에 적어야 한다.

## 4. 조건화 (두 feature 가 만나는 곳)

`src/atm_sc/models/roi_pair_embedding.py`

```python
pair_vec = Emb(i) + Emb(j)                       # 덧셈 → (i,j)와 (j,i)가 같다 (순서 불변)
cond     = anatomy_gain * LayerNorm(a) + Proj(pair_vec) + mode_emb[mode]
mu_pair  = prior_mu(pair_vec) + mode_prior[mode]  # 조건부 사전분포 평균
```

`LayerNorm` 은 재학습에서 추가했다 (아래 크기 불균형 때문). `eps=1e-8` 로 낮춘 이유:
anatomy 의 원소 분산이 ~1e-5 라 기본 `eps=1e-5` 면 정규화가 절반만 먹힌다 (실측 L2 0.92 → 0.9999).
스케일을 **새 파라미터에만** 넣어 구 checkpoint 가 missing-key 경로로 안전하게 들어온다.

`mode`: 0 = 전체 streamline, 1 = SC edge 정렬 segment. 같은 디코더를 쓰고 조건만 다르다.
`mode_emb` / `mode_prior` 는 0 으로 초기화해 기존 동작을 보존한다.

**실측된 크기 불균형** (test 31명):

| 항목 | L2 크기 | 1층 기여 std | subject/pair 간 변동 |
|---|---|---|---|
| anatomy a | **0.069** | 0.0228 | 0.0069 |
| pair_vec | **1.521** (22배) | 0.1741 (8배) | 0.1443 (**21배**) |

가중치 자체는 죽지 않았다 (anatomy 쪽 rms 0.065 vs pair 0.066). **입력이 22배 작다.**

**LayerNorm 적용 후 (실측):** anatomy L2 **0.168 → 1.000**, pair_vec 평균 L2 1.14
→ 크기 비율 **6.8× → 1.14×**. 다만 사전학습 디코더가 보던 조건 분포가 그만큼 커지므로
P0 예열 단계가 필수다 (`PIPELINE_08_RETRAIN_DESIGN.md`).

## 5. Head 들

| head | 입력 | 출력 | 용도 |
|---|---|---|---|
| `weight_head` | cond, z | w [N] | streamline 별 가중치 |
| `edge_head` | a, pair_vec, pairs | logit [K] | 그 쌍에 연결이 있는가 (템플릿 위) |
| `count_head` | a, pair_vec, pairs | log count [K] | **pass** SC 값 (템플릿 위) |
| `count_head_end` | a, pair_vec, pairs | log count [K] | **끝점** 기준 가닥 수 (템플릿 위, 배분용) |

`count_head` 가 필요한 이유: `sc_magnitude_loss` 는 총합을 정규화한 뒤 비교하므로 절대
스케일 정보가 손실에서 사라진다 (실측: 패턴 r=0.81 인데 CCC=0.02). 개수를 직접 맞히는
head 가 있어야 절대값이 맞는다.

구조 (`edge_count_head.py`): `[a(512) | pair_vec(64)] → Linear(576→256) → GELU →
Linear(256→256) → GELU → Linear(256→1)`. 마지막 층 0-초기화 + bias = `init_log_count`
→ 시작 시 모든 edge 가 상수를 예측하고 "평균은 이미 맞고 편차만 학습" 하는 지점에서 출발한다.

출력이 log count 인 이유: count 분포가 heavy tail (1 ~ 수만) 이라 선형 회귀는 큰 edge 에만
끌려간다. log 면 상대오차가 균등해지고 `exp() ≥ 0` 이 구조적으로 보장된다.

**템플릿 인수분해** (재학습에서 추가): head 가 그룹 평균을 외우는 데 용량을 쓰지 않도록
train 평균을 buffer 로 주고 그 위의 **개인차만** 학습한다.

```python
log SC(i,j) = log1p(template(i,j)) + f(a, pair_vec)     # count_head=sc_pass, count_head_end=sc_end
logit(i,j)  = logit(template_prob(i,j)) + g(a, pair_vec) # edge_head=edge_prob
```

마지막 층이 0-초기화라 **시작 시점에 정확히 템플릿을 재현한다** (실측 r = 1.000000).
템플릿은 `register_buffer` 로 고정하고 학습하지 않는다. `outputs/inference/template.npz`
(`scripts/34_build_inference_prior.py`) 에서 온다.

## 6. 체크포인트 호환

```python
def load_checkpoint(self, sd, strict=False):
    missing, unexpected = self.load_state_dict(sd, strict=False)
    assert not unexpected, f"checkpoint 에만 있는 키 (구조 불일치): {unexpected[:8]}"
    if missing:
        print(f"checkpoint 에 없는 새 파라미터 {len(missing)}개는 초기값 사용")
```

나중에 추가한 head 는 초기값을 유지하고 나머지는 그대로 싣는다. checkpoint 에만 있는
키가 있으면 구조가 바뀐 것이므로 **중단**한다.

## 7. 생성 API

```python
model.generate(anatomy, pairs, n_per_pair, chunk=8192, generator=None, z=None)
# -> (mm [K*n,128,3], w [K*n], pairs_rep [K*n,2])
```

`z` 를 주면 사전분포 대신 그것을 쓴다 (latent bank 용). 주지 않으면 `sample_z` 가
`N(mu_pair, I)` 에서 뽑는다.

## 8. WM 채널 (적용됨)

입력은 `[1, 2, 193, 229, 193]` 이다: ch0 = rigid T1 [0,1] 정규화, ch1 = WM 확률 0~1.
UNet 첫 conv 를 `Conv3d(1,64,3) → Conv3d(2,64,3)` 으로 확장하고 ch0 은 사전학습 가중치를
복사, **ch1 은 0 으로 초기화**한다. 실측: WM 채널을 0 으로 채우면 1채널 결과와
**차이 0.0** (bit 단위 동일) — 사전학습 동작이 정확히 보존된다.

`in_channels` / `template` / `t1_source` 는 checkpoint 에 기록되고
`roi_atm.from_checkpoint()` 가 그것으로 모델을 만든다. 직접 `ROIPairATM(...)` 을 만들면
프로토콜이 어긋나도 **오류 없이 결과만 나빠진다** (실측: own r 0.81 → 0.72).

주의: WM 맵은 T1 에서 결정론적으로 계산된 것이라 **정보가 새로 생기지는 않는다.**
학습을 쉽게 만드는 유도 편향으로서의 가치이고, §4 의 크기·구조 불균형을 같이 고치지
않으면 anatomy 경로가 여전히 무시될 수 있다.
