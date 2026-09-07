# ROI-pair ATM — 구현 및 Smoke Test 보고서

날짜: 2026-09-02 · 기준: `docs/ATM_ROI_PAIR_SC_FINETUNING_PIPELINE.md`
결과: **synthetic PASS · real-data PASS · pytest 35 passed** · phase 스크립트 07/10 동작 확인

이전 보고서(`ATM_SC_SMOKE_TEST_REPORT.md`, 30-bundle 버전)의 upstream 분석 §1 은 그대로 유효하다.

---

## 1. 무엇이 바뀌었나 (30-bundle → ROI-pair)

| | 이전 (v2 프레임워크) | 이번 (ROI-pair 파이프라인) |
|---|---|---|
| bundle 단위 | ATM 원본 30 해부학 bundle, 모델 30개 | SC edge = ROI pair, **decoder 1개 공유** |
| 조건 | 어느 .pth 를 쓰느냐 | `cond = gain·a + Proj(Emb(ROI_a)+Emb(ROI_b))` |
| latent prior | bundle 별 KDE | N(0,I) (KL 로 맞춤) |
| T1 encoder 호출 | subject × 30 bundle | **subject 당 1회** |
| SC 기여도 | count | `w_k = softplus(MLP(cond, z))` (§16) |
| edge 존재 | — | `EdgeHead(a, pair_vec)` + BCE (§10) |
| GT 분해 | — | whole-brain `.tt.gz` → endpoint ROI → canonical pair → 128점 (§5-8) |

upstream 은 여전히 무수정. ROI-pair 조건은 ConvVAE 의 FiLM 이 받는 `anatomical_info` 자리에
합쳐 넣는다 (`model.py:296-298, 316-317`). Proj 마지막 층 0-init, gain=1 → **시작 시점 decoder 는
pretrained 와 동일한 함수**(단, 좌표 박스는 whole-brain 으로 바뀜 — §5 참조).

## 2. 파일

```
src/atm_sc/
  models/atm_adapter.py            (기존) ATMVAE 래퍼, D2~D5 우회, 인코더 전용 UNet
  models/roi_pair_embedding.py     Emb(a)+Emb(b) → Proj → +gain·anatomy. 순서 불변, 0-init
  models/streamline_weight_head.py softplus MLP, init w=1
  models/edge_head.py              P(edge), init 0.5
  models/roi_atm.py                ROIPairATM: 위를 조립. brain_box_mm() 좌표 정규화
  models/endpoint_assigner.py      (기존) endpoint_probs / visit_probs, 배경 클래스
  models/sc_builder.py             endpoint_sc / pass_sc 에 weights 추가, ChunkedSC, BundleAccumulator
  losses/{endpoint,edge,sc_corr,sc_magnitude,tract_length,geometry,metrics}.py
  data/trk_to_roi_pairs.py         endpoint 할당 → canonical pair, end/pass SC + length
  data/resample_streamlines.py     벡터화 등간격 128점 (끝점 정확)
  data/build_distance_maps.py
  data/dataset.py                  ROIPairSubject (npz 메모리 상주, get_pair, sample_pairs, negative_pairs, gt_for)
  data/tt_io.py, paths.py, prepare_t1.py (기존)
  training/trainer.py              2-pass 생성 + 재구성 + edge, subject-level SC
  training/run.py                  phase 실행기 (roi_atm/endpoint/edge/sc/full)
  training/synthetic.py            §27 합성 subject
scripts/
  00_reproduce_atm.py  01_qc_coordinate_space.py  02_assign_roi_pairs.py  03_build_roi_pair_bundles.py
  04_resample_to_128.py  05_build_distance_maps.py  06_smoke_test.py
  07_train_roi_atm.py  08_train_endpoint.py  09_train_sc.py  10_train_full.py  11_benchmark.py
  legacy/  (30-bundle 버전 스크립트)
tests/  test_endpoint, test_sc_builder, test_losses, test_roi_pair, test_smoke  (35 passed)
docs/ROI_PAIR_DATA_FORMAT.md       npz 계약
```

## 3. Tensor flow (학습 1 step, subject 1명)

```
T1_W [1,1,193,229,193] ──UNet(동결, 인코더만)──▶ a [1,512]     subject 당 1회, 디스크 캐시
                                                    ×gain
pairs [K,2] (양성 전부 또는 subset) ─repeat n_gen─▶ [N,2] ──Emb+Proj──▶ + a = cond [N,512]
z ~ N(0,I) [N,64]
[G] pass 1 (no_grad, chunk): decode → S [c,128,3] → w [c] → q_s,q_e [c,R] → 부분 SC/Num 합산
      SC_total [R,R] ──L_corr + L_mag + L_length (fp32)──▶ dL/dSC = G, dL/dNum
    pass 2 (grad, chunk): 같은 z 로 재생성 → (SC_c·G).sum() + λ_end·L_endpoint_c → backward
[R] GT bundle 샘플 [n,128,3] ──encode(cond)──▶ μ,logσ² → z → decode → L_recon + β·KL + L_adj → backward
[E] 양성/음성 pair ──EdgeHead──▶ BCE → backward
clip → AdamW step.   trainable 2.10 M (ConvVAE 1.515 M + heads), UNet 49.86 M 동결
```

SC 는 chunk 마다 loss 를 걸지 않고 합친 뒤 한 번 계산한다 (§24). 2-pass gradient 는 단일 그래프와
**오차 0** (tests/test_sc_builder.py). recurrent tracking 없음.

## 4. 결과

### 4.1 데이터 전처리 (sub-000001, 1,000,000 streamline)

| 항목 | 값 |
|---|---|
| endpoint 할당 | 433,278 (43.3 %). 배경 start/end 30.5 %/23.6 %, 양끝 같은 ROI 12.6 % |
| 양성 pair K | **1,759** / 3,321 (density 0.530, endpoint rule) |
| `sc_pass` vs `.mat` GT | r = **0.9986** (재확인) · `sc_end` vs `.mat` r = 0.673 |
| bundles.npz (cap 256) | 150,721 streamline, 100 MB, fp16 오차 0.031 mm, 끝점→ROI 일치 **100 %** |
| 128점 재샘플 | 길이 오차 mean 0.09 % / p99 0.28 %, 끝점 오차 0, 1.8 s / 1e5 |
| 좌표 QC | streamline 99.96 % 가 warp 된 T1 뇌 안, 좌우 반전 없음 (`outputs/qc/*.png` 육안 확인) |

### 4.2 `scripts/06_smoke_test.py` (A10, endpoint 모드)

```
[A] synthetic  ROI=6, 양성 pair 3, 8/pair, P=128
  ROI-pair embedding [N,512] / pair order invariance / streamline [24,128,3] /
  w>=0 & init 1 / endpoint prob [N,6] sum==1 / SC [6,6] symmetry 0.0 / finite loss (8항) /
  backward / streamline gradient (max 5.4e-1) / model gradient (pair_emb, weight_head,
  edge_head, convvae 전부) / optimizer step 53/66 tensors 변경 / UNet 동결       → 전부 PASS
[B] real  sub-000001, 8 pairs × 16 = 128 streamlines
  T1 encoder 0.38 s 1회 → [1,512] · K=1759 · L_recon 680 / L_kl 94.5 / L_endpoint 35.6 /
  L_edge 0.693 / L_corr 1.006 / L_mag 1.94 / L_length 3.35 · grad 유한 · 2 step 연속 OK  → PASS
peak VRAM 0.58 GB · 21.4 s
FINAL: PASS
```

### 4.3 phase 스크립트

`07_train_roi_atm.py --max-steps 3 --max-pairs 32`: L_recon 879 → 646 → 591 (3 step, 하강 확인).
`10_train_full.py --max-steps 1 --n-gen 4`: 양성 1,759 pair × 4 = 7,036 streamline 전체 SC 로 1 step 동작.
step 시간 (npz 메모리 상주 수정 후): 07 = 1.3 s (1 step 째, warm-up 포함) → <0.1 s, 10(full, 7,036 생성) = 2.1 s. 수정 전 40 s.

### 4.4 참고 수치 (`00_reproduce_atm.py`, `11_benchmark.py`)

- 공식 예제 재현 (AF_L, sub-1135): 3000 streamline, GT 와 3 mm 이내 점 비율 0.942 / 0.998
- 생성 throughput: decode-only **~260k streamlines/s** (fp32, batch ≥ 2048 에서 포화), T1 encoder 0.53 s
- `.trk` 쓰기 32,768 streamline 1.0 s

## 5. 발견한 문제 (중요도 순)

1. **pretrained ATM 의 anatomy feature 가 거의 0 이다.** PPMI sub-000001: ‖a‖ = 0.086, T1=0 을 넣어도
   0.053 (bias). 공식 예제 sub-1135 는 T1 강도 0–223 인데 정규화 상수가 8330 이라 입력이 [0, 0.026] —
   그런데도 AF_L 을 잘 재현한다. 즉 **pretrained 모델에서 T1 조건화는 사실상 비활성이고 bundle
   geometry 는 KDE latent 은행이 실어 나른다.** subject-specific 생성이라는 전제에 직접 관련된 발견이다.
   대응: `anatomy_gain`(학습 가능) 추가, feature ≈ 0 이면 assert. **다음 단계에서 subject 간 feature
   분산을 반드시 측정**해야 한다 — 분산이 없으면 T1 → SC 학습은 성립하지 않는다.
2. **GT SC 는 pass, 파이프라인은 endpoint.** endpoint 모드 학습 target 은 같은 tractogram 의
   `sc_end`(K=1759, density 0.53) 이고 `.mat` 의 pass-SC(density 0.84) 는 최종 평가 참조다. 섞으면
   상한 r=0.67. `--sc-mode pass` 도 동작한다 (visit_probs, d_bg=2.0).
3. **시작 시점 geometry**: 좌표 박스를 whole-brain 으로 바꾸므로 pretrained AF_L 출력이 박스에 맞춰
   늘어난 상태에서 시작한다 (L_recon ≈ 680 mm² → RMSE 26 mm). L_ATM 이 다시 맞춰야 한다.
4. **gradient 크기 불균형**: L_recon 이 mm² 스케일이라 grad norm 이 10⁴ 이고 나머지 항은 10⁰~10¹.
   clip 5.0 으로 막았지만 λ 재조정이 필요하다 (§17 지시대로 각 항 grad norm 을 로그에 기록함).
5. `ROIPairSubject` 가 `NpzFile` 을 들고 있어 `get_pair` 마다 100 MB 를 재압축해제 → step 40 s.
   메모리 상주로 수정.
6. 공간: GT 가 QSDR 템플릿 공간이라 T1 을 SyN 으로 그쪽에 맞춘다. subject 별 "형상" 은 GT 에 없다.

## 5b. 우선순위 1·2 검증 결과 (2026-09-02 후반)

### 우선순위 1 — T1 anatomy feature 가 subject 를 구별하는가

**먼저 발견한 함정: T1 강도 스케일.** PPMI native T1 max 가 878 ~ 203,163 (230배). upstream 고정
상수(max 8330)로 정규화하면 두 subject 는 voxel 39 % 가 >1 이 되어 feature 가 40배 커진다
(‖a‖ 3.6 vs 0.08). 즉 그대로 쓰면 feature 는 해부가 아니라 **스캐너 강도 스케일**을 인코딩한다.
→ `BundleNorm.normalize_t1(robust=True)`: subject 별 뇌 안 p99.5 를 0.6 으로 맞춘 뒤 upstream 식 적용
(`tests/test_roi_pair.py::test_robust_t1_normalization_is_scale_invariant`).

**robust 정규화 후** (해부 성분 a−a₀, a₀ = T1=0 일 때의 bias):

| 정합 | subject 수 | ‖a−a₀‖ | 쌍별 cosine mean / min | 쌍별 L2 / ‖a−a₀‖ |
|---|---|---|---|---|
| rigid | 8 | 0.090 | **0.92** / 0.74 | 0.41 |
| SyN | 5 | 0.097 | **0.97** / 0.94 | 0.23 |

동결된 pretrained UNet 의 feature 는 subject 간에 거의 같다 (특히 GT 공간인 SyN 에서 0.97).
bundle 을 구분하도록 학습된 인코더지 subject 를 구분하는 인코더가 아니다.
**decoder 만 fine-tuning 해서는 T1 조건화가 subject-specific SC 를 만들 수 없다.**

대응: `trainable="vae+unet4"` — conv4_x + fc (17.96 M) 만 다시 연다. conv1~3 출력을 subject 당
1회 캐시(fp16 72 MB)하므로 메모리 peak 0.7 GB, stage4 forward 는 동결 경로와 bit-exact.
5 subject 로 Phase 2(L_ATM 만) 600 step 을 돌려도 cosine 0.973 → 0.967 로 **판별력은 생기지 않았다**
— 재구성 손실은 z 가 다 설명하므로 T1 을 쓸 이유가 없다. 판별력은 subject 별 target 을 가진
loss(endpoint/SC)에서만 나올 수 있고, 이는 배치 전처리 완료 후 held-out subject 의 SC corr 로 판정한다.

### 우선순위 2 — 학습 SC 정의 (근거)

| | endpoint rule | pass rule |
|---|---|---|
| GT `.mat` SC 와의 일치 (같은 tractogram, hard) | r = 0.673 | **r = 0.9986** |
| density (sub-000001) | 0.530 (K=1759) | 0.839 |
| soft(미분가능) 버전 vs GT | — | r = 0.9939, ccc = 0.985 |
| 파이프라인 문서의 bundle 정의 | **endpoint** (§5) | — |

권장: **bundle 분해·조건화·L_endpoint 는 endpoint (문서 그대로), SC-level loss/평가는 pass**
(TVB 에 실제로 쓰는 `.mat` 정의). endpoint 로 조건화된 streamline 도 중간 ROI 를 지나므로 생성
tractogram 의 pass-SC 는 잘 정의되고 `.mat` 과 직접 비교된다. 둘을 섞어 endpoint soft-SC 를 `.mat` 에
맞추면 상한이 0.67 이다. `--sc-mode` 한 플래그로 전환된다.

### Phase 2 결과 (sub-000001, 500 step, β 스윕)

| β | L_recon | RMSE | KL | prior z 생성 pair acc | posterior-bank z pair acc |
|---|---|---|---|---|---|
| 0.01 | 13.6 | 3.7 mm | 161 | 0.000 | **0.663** (jitter 0) / 0.19 (0.5) |
| 0.1 | 15.1 | 3.9 mm | 94 | 0.000 | 0.636 |
| 1.0 | 28.6 | 5.3 mm | 42 | 0.000 | 0.653 |

재구성은 수렴하지만 **prior z~N(0,I) 에서 생성하면 pair 정확도 0, 길이 3배** — decoder 가 pair 조건
대신 z 에 실린 정보만 쓴다 (posterior 은행에서 z 를 뽑으면 0.66). upstream ATM 이 KDE 은행을
쓰는 이유가 바로 이것이다. 조건이 작동하게 만드는 것은 Phase 3 (endpoint loss) 의 역할.

Phase 3 첫 시도에서 L_endpoint 가 34 에서 전혀 안 움직였다: 끝점이 목표에서 멀면
softmax(−d/0.5) 의 목표 확률이 1e-8 아래로 떨어져 `clamp` 에 걸리고 **gradient 가 0** 이 되는
버그 (34 ≈ 2·log 1e-8). `endpoint_log_probs` + `endpoint_loss(log_input=True)` 로 log-softmax 경로를
만들어 고쳤고 (`test_far_endpoint_still_has_gradient`), endpoint 전용 τ=5 mm 를 분리했다.

## 6. GPU / runtime

A10 23 GB · smoke peak VRAM 0.58 GB · T1 encoder 0.4 s/subject (peak 9 GB, 캐시됨) ·
ConvVAE decode ~260k streamlines/s · SyN 정합 ~6 min/subject (1회) · `.tt.gz` 전처리 ~100 s/subject.
AMP: fp16 max 6 mm / bf16 max 57 mm 좌표 오차 → 기본 off, bf16 사용 금지.

## 7. 실행

```bash
python scripts/01_qc_coordinate_space.py --sub sub-000001      # 정합 + PNG
python scripts/02_assign_roi_pairs.py --sub sub-000001         # ~90 s
python scripts/03_build_roi_pair_bundles.py --sub sub-000001   # ~20 s
python scripts/05_build_distance_maps.py
python -m pytest tests -q
python scripts/06_smoke_test.py                                 # synthetic + real
python scripts/07_train_roi_atm.py --max-steps 5 --max-pairs 64
python scripts/10_train_full.py  --max-steps 5 --n-gen 4
```

206 명 전체 전처리: `02 --all` 은 약 5 h → background 로.

## 8. 다음 단계 (파이프라인 §33)

1. **subject 간 anatomy feature 분산 측정** (5-10 명 SyN 정합 후) — 프로젝트 성립 여부를 가르는 검사
2. 206 명 전처리 배치 (01 → 02 → 03), split 은 group × batch2 stratify
3. Phase 2 ROI-pair baseline (L_ATM) 수백 step, L_recon 수렴과 λ 재조정
4. Phase 3 endpoint → Phase 4 edge → Phase 6-7 SC corr/mag (+ weight head) → Phase 8 length
5. inference 스크립트 (§22): edge head 로 pair 선택 → 생성 → concat → `.trk` → hard SC → `.mat` GT 와 평가
6. ablation (§31), CoRNN 대비 benchmark (§30)
