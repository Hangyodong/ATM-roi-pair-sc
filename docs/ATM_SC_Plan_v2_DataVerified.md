# ATM SC-aware Fine-tuning — v2 프레임워크 + 실데이터 검증 분석

작성일: 2026-09-02
대상: `ATM_SC_Aware_Finetuning_Framework_v2.md` (최종 파이프라인) + `PPMI_QC263_tracto.zip` + `stable.zip`
상태: **`ATM_Upstream_Analysis_and_SC_Finetuning_Plan.md`(v1 분석)를 대체한다.**
      upstream 코드 분석 결과는 v1 문서가 여전히 유효하므로 여기서는 중복하지 않고 참조만 한다.

---

## 0. 핵심 요약

v2 문서를 읽고, **주장을 검증하지 않고 받아들이는 대신 실제 데이터로 전부 확인했다.**
그 결과 v2 설계와 실제 데이터가 **세 군데에서 충돌**한다. 특히 첫 두 개는 구현 전에 반드시 해결해야 한다.

| # | 충돌 | 근거 | 영향 |
|---|---|---|---|
| **C1** | v2의 SC/endpoint 설계는 **endpoint 기반**인데, GT SC는 **pass(통과) 기반** | endpoint SC vs GT: **r=0.673** / pass SC vs GT: **r=0.9986** (실측) | Differentiable SC Builder(§8)와 `L_endpoint`(§7) 재설계 필요 |
| **C2** | GT tractogram은 **QSDR 템플릿 공간**(비선형 정규화), ATM은 **rigid-MNI**(subject 형상 보존) | 5개 subject의 `trans_to_mni`·`dimension`이 완전히 동일. `.mat` info에 "QSDR 1e6 tracts" 명시 | ATM 출력 공간 ≠ GT 공간. 프로젝트 최대 리스크 |
| **C3** | v2 §28의 `05_build_gt_sc.py`가 불필요 | `FC_DKPD25_82_ppmi_all_nomed_qc.mat`에 **SC_weight / SC_length가 이미 238명분 존재** | 스크립트 1개 제거, 대신 `.tt.gz` 디코더 필요 |

추가로 **환경에 외부 도구가 하나도 없다** (MRtrix / ANTs / FreeSurfer / MATLAB / DSI Studio / dipy / scilpy 전부 없음).
ATM 공식 `infer.py`는 이들 없이는 절반만 실행된다.

---

## 1. v2 문서 이해 — v1 대비 무엇이 바뀌었나

v2는 v1에 대해 다음을 추가/변경했다.

1. **`L_endpoint` 신설** (§7). SC_corr는 subject-level global objective라서 개별 streamline이
   올바른 ROI pair를 잇는지 감독하지 않는다 → streamline-level supervision을 추가.
2. **Loss를 local / global 2계층으로 명시적 구분** (§15-16).
   local = `L_ATM`, `L_endpoint` / global = `L_SC_corr`, `L_SC_mag`, `L_length`.
3. **Bundle coverage test를 SC loss보다 먼저 배치** (§5.1, Stage 2).
   낮은 SC 성능이 loss 문제인지 30-bundle 표현력 문제인지 먼저 분리하겠다는 것 — 타당하다.
4. **"SC_corr를 넣으면 반드시 좋아진다"고 주장하지 않음** (§9, §22 Case A~D).
   ablation으로 검증하고 실패 양상까지 미리 분류해 둔 점이 v1보다 낫다.
5. **λ 축소** (§17): `λ_corr` 0.1→0.05, `λ_mag` 0.1→0.05, `λ_len` 0.05→0.02, `λ_endpoint`=0.1 신설.
6. **DWI 불필요 선언** (§3): GT tractography와 GT SC를 이미 보유하므로 맞다.
7. **Stage 9단계로 세분화**, `endpoint_assigner.py` / `endpoint.py` / `endpoint_labels.py` /
   `evaluate_endpoint.py` 파일 추가 (§28).

설계 방향 자체는 동의한다. 문제는 아래 실측 결과와 맞물리는 부분이다.

---

## 2. 실측한 데이터 사실

### 2.1 `PPMI_QC263_tracto.zip` (27.28 GB)

```
PPMI_QC263_tracto/
├── manifest.csv                     263 rows
└── sub-XXXXXX/
    ├── sub-XXXXXX_T1w.json
    ├── sub-XXXXXX_T1w.nii.gz
    └── sub-XXXXXX_tract.tt.gz       ← .trk 가 아니라 DSI Studio TinyTrack
```

manifest 집계: `group` PD 197 / HC 66 · `proto` tr25 223 / ep2d 40 ·
`dwi_qc` pass 238 / FAIL 25 · `how` 전부 keep · `btable_refit` 1인 subject 4명.
`fib` 컬럼은 `/mnt/d/PPMI_CTR_PD_NIFTI/derivatives/DSI_SC_len/.../dwi.fib.gz`를 가리키는데
**`/mnt/c`, `/mnt/d`는 이 머신에 없다** → `.fib.gz` 접근 불가 (C2에서 중요).

**T1w**: subject **native** 공간. 1mm iso, RAS. shape은 프로토콜에 따라 `(192,256,256)`(tr25) 또는
`(176,240,256)`(ep2d), origin은 subject마다 전부 다름. GE SIGNA Architect, SAG 3D T1 FSPGR.

**`.tt.gz`**: gzip + MATLAB v4. `scipy.io.loadmat`으로 읽힌다. 변수:

| 변수 | 값 |
|---|---|
| `dimension` | `[80, 100, 80]` |
| `voxel_size` | `[2, 2, 2]` mm |
| `trans_to_mni` | 대각 `diag(-2,-2,+2)`, offset `(79.5, 81.5, -72)` |
| `report` | DSI Studio deterministic (Yeh 2013 + Yeh 2020), aniso th 0.5–0.7 otsu, angle 60°, **step 1.00 mm**, length 20–250 mm |
| `parameter_id` | `c9A99193Fba3Fb803FcbA041b7A4340420Fcba803Fdc` |
| `track`, `track1` | uint8 buffer — 1/32-voxel 정밀도 int8 delta 인코딩 |

디코더를 작성해 검증했다 (버퍼를 **정확히 소진**, `consumed_exactly=True`):

```
sub-100001   track  635,859 + track1  364,141 = 1,000,000 streamlines
             41,983,963 + 24,057,010 = 66,040,973 points
             평균 66.0 point/streamline  (step 1mm → 평균 약 65 mm)
```

포맷: `[uint32 size=3*npts][int32 x,y,z ×32][int8 dx,dy,dz] × (npts-1)`

**결정적 확인 — 이건 native 공간이 아니다.** 5개 subject(tr25 3, ep2d 2, T1 shape 상이)에서
`dimension`, `voxel_size`, `trans_to_mni`가 **완전히 동일**했고 off-diagonal이 정확히 0이었다.
`.mat`의 info 문자열도 "**QSDR 1e6 tracts**"라고 명시한다.
→ tractogram은 **DSI Studio QSDR로 비선형 정규화된 템플릿 공간**에 있다.

### 2.2 Atlas

`DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz`
`(91,109,91)` @2mm, LAS, affine = FSL MNI152 표준. 라벨 **82개 전부 존재**(소실 없음).
최소 ROI 15 voxel (label 71, 72) — 2mm에서 매우 작으므로 soft assignment 시 주의.

### 2.3 GT SC는 이미 존재한다

`FC_DKPD25_82_ppmi_all_nomed_qc.mat` → `data` struct array **238 rows** (dwi_qc==pass만).

```
atlas : "DesikanCortexPD25_2mm (82: 1-66 DK cortex, 67-82 PD25 subcortex)"
info  : "... SC_weight=streamline count(pass), SC_length=mean length mm (QSDR 1e6 tracts) ..."
```

필드: `subject, session, group, batch, proto, batch2, age, sex, TR, nvol,`
`FC_raw(82,82), FC_raw_z(82,82), SC_weight(82,82), SC_length(82,82), dwi_qc,`
`updrs3, updrs3_gap_months, updrs3_tremor, updrs3_rigidity, updrs3_brady`

sub-100001 기준: SC upper-tri 3321개 중 **nonzero 2785 (density 83.9 %)**,
총합 6.90e6, 상위 10 % edge가 전체 weight의 **68.9 %**, GT edge length 평균 152.4 mm.

### 2.4 GT SC 재현 검증 — 좌표 체인 전체를 확인했다

```
tt voxel/32 ──trans_to_mni──▶ MNI mm ──inv(atlas.affine)──▶ atlas voxel ──▶ ROI label
```

sub-100001의 1,000,000 streamline 전부를 디코딩해 두 가지 정의로 SC를 만들고 GT와 비교했다.

| 정의 | sum (GT 6.90e6) | nnz (GT 2785) | Pearson r | r(log1p) | edge F1 |
|---|---|---|---|---|---|
| **`end`** (양 끝점만) | 4.33e5 | 1759 | **0.6728** | 0.7530 | 0.7738 |
| **`pass`** (통과 ROI 집합의 모든 쌍) | 6.78e6 | 2774 | **0.9986** | 0.9912 | 0.9815 |

| tract length | Pearson r | MAE |
|---|---|---|
| `end` 기반 | 0.7675 | 40.64 mm |
| **`pass` 기반** | **0.9744** | **4.91 mm** |

per-edge 비율(mine/GT) median 0.980 (p25 0.842, p75 1.050). 남은 ~2 % 차이는
DSI Studio의 subvoxel/보간 방식과 내 nearest-neighbor 반올림 차이로 설명된다.

**→ 좌표 체인 전체가 검증되었다. 그리고 GT SC의 정의가 `pass`임이 확정되었다.**

### 2.5 실행 환경

```
GPU        NVIDIA A10  23,028 MiB   (1장)
torch 2.6.0+cu124 · numpy · scipy 1.17.1 · nibabel 5.4.2 · scikit-learn 1.7.2 · joblib 1.5.2
없음: dipy, scilpy, nilearn, ants
없음(바이너리): dsi_studio, MRtrix(tckedit/tck2connectome), ANTs, FreeSurfer, MATLAB, singularity/apptainer
컨테이너 이미지는 있으나 실행기가 없음: tractseg_eval_master.sif, slant_deep_brain_seg_v1_1_0.sif
```

ATM KDE 로드 테스트 (`kde_models/AF_L/kde_model.joblib`, 277 MB):
로드 33.6 s, sklearn 1.6.1 pickle → 1.7.2 경고만 발생하고 **정상 동작**.
`kernel='tophat'`, `bandwidth=1`, **447,000 × 64** 학습 샘플.
→ ATM의 latent prior는 사실상 **학습 latent 은행에 반경 1의 균일 jitter를 더한 것**이다.
   `N(0,I)` prior가 아니므로 z를 임의로 샘플링하면 안 된다.

---

## 3. 충돌 C1 — GT SC는 endpoint가 아니라 pass 기반이다

### 문제

v2 §7.4는 `L_endpoint = CE(q_start, i) + CE(q_end, j)`,
v2 §8.1은 `W_k(i,j) = q_start(k,i) · q_end(k,j)` 로 정의한다. 둘 다 **양 끝점만** 본다.

그런데 실측상 endpoint 기반 SC는 GT와 **r = 0.673**밖에 되지 않는다.
v2 Stage 5의 통과 기준("soft SC가 reference SC와 충분히 유사")을 endpoint 정의로는 **구조적으로 만족할 수 없다.**
loss를 아무리 잘 만들어도 목표 자체가 다른 양을 가리킨다.

### 수정 — Differentiable **pass** SC Builder

streamline `k`의 128개 점 각각에 대해 soft ROI 확률 `q_t ∈ R^82`를 구한 뒤,
**끝점이 아니라 "통과 여부"를 noisy-OR로 집계**한다.

```
q_t(i)   = softmax_i( -d_i(p_t) / τ )          # ROI distance map + grid_sample (v2 §7.3 그대로)

u_k(i)   = 1 - Π_t ( 1 - q_t(i) )              # streamline k가 ROI i를 통과할 확률 (noisy-OR)

SC_k(i,j)= u_k(i) · u_k(j)      (i ≠ j)
SC_pred  = Σ_k SC_k
```

- noisy-OR는 hard `max`의 매끄러운 대체이고 미분 가능하다. 수치 안정성을 위해
  `log(1-q)` 합으로 계산한다: `u = 1 - exp(Σ_t log(1 - q_t + eps))`.
- tract length도 같은 `u`로:
  `Length(i,j) = Σ_k u_k(i)u_k(j)·L_k / (Σ_k u_k(i)u_k(j) + eps)`

### 수정 — `L_endpoint`의 역할 재정의

`L_endpoint`를 버리지는 않는다. 다만 **GT SC를 재현하는 supervision이 아니라는 점**을 분명히 한다.

- **주 supervision (신설) `L_roi_visit`**: GT streamline이 통과한 ROI 집합을 multi-hot
  `y_k ∈ {0,1}^82`로 만들고, `BCE(u_k, y_k)`. GT SC 정의와 정확히 같은 양을 감독한다.
- **보조 `L_endpoint`**: 기존 CE 유지. 끝점이 GM에 안착하도록 해서 해부학적 타당성을 지킨다.
  (`pass` 정의에서는 SC에 직접 기여하지 않지만 streamline이 백질 밖으로 새는 것을 막는다.)

### 검증 기준 (Stage 5 대체)

Soft `pass` SC의 목표는 MRtrix가 아니라 **내가 이미 검증한 hard `pass` SC**다
(그것이 GT와 r=0.9986). 즉:

```
GT .tt.gz ──hard pass──▶ SC_hard   (GT와 r=0.9986, 이미 확인)
GT .tt.gz ──soft pass──▶ SC_soft   ← 이것이 SC_hard와 얼마나 같은가를 τ 튜닝으로 맞춘다
```

MRtrix가 없으므로 이 방식이 오히려 더 직접적이고, 외부 도구 의존도 사라진다.

---

## 4. 충돌 C2 — 공간 불일치 (최대 리스크)

### 문제

```
GT tractogram : QSDR 템플릿 공간. 전 subject 동일 격자 (80,100,80)@2mm.
                → 비선형 정규화로 subject 뇌 형상이 이미 템플릿에 맞춰짐

ATM           : subject T1 ──antsRegistrationSyN.sh -t r (rigid)──▶ MNI 1mm (193,229,193)
                → subject 뇌 형상이 그대로 보존됨. 출력 streamline도 이 공간

제공된 T1     : subject native (192,256,256) 등, origin 전부 다름
```

**ATM 출력 공간과 GT streamline 공간이 비선형 warp만큼 다르다.**
이 상태로 `L_ATM`이나 SC loss를 걸면 모델은 자기가 볼 수 없는 warp를 학습하라는 요구를 받는다.

QSDR 역변환은 `.fib.gz` 안에 있는데, 그 경로(`/mnt/d/...`)가 이 머신에 없다.

### 선택지

| | 방법 | 필요한 것 | 평가 |
|---|---|---|---|
| **A** | ATM 입력 T1을 **GT와 같은 템플릿 공간으로 비선형 정규화**해서 넣는다 | T1→MNI 비선형 정합 도구 (ANTs 등, 현재 없음) | **권장.** 입력 해부와 출력 streamline 공간이 일치. 다만 ATM 인코더는 rigid T1로 학습되었으므로 domain shift 존재 |
| **B** | GT streamline을 native로 되돌린 뒤 rigid-MNI로 보낸다 | 263개 `.fib.gz` (QSDR 역변환 포함) | 파일 접근 불가. 사용자가 복사해 주면 가능 |
| **C** | native 공간에서 tractography 재수행 | 원본 DWI | zip에 DWI 없음. 사실상 불가 |

### A를 택할 때의 과학적 함의 — 반드시 인지해야 함

GT가 **이미 템플릿 공간으로 정규화된** tractogram이므로,
**"subject-specific streamline geometry"의 상한이 데이터에 의해 제한된다.**
subject 간 차이는 뇌 형상이 아니라 **streamline 밀도/경로 분포**에만 남아 있다.

이것이 프로젝트를 무의미하게 만들지는 않는다 — SC weight/length는 subject마다 실제로 다르고,
그것이 최종 목표(TVB 입력)이기 때문이다. 다만 논문 Methods에서
"ATM 원본은 rigid 공간에서 subject-specific geometry를 학습, 본 연구는 템플릿 공간에서
subject-specific connectivity를 학습"으로 **정확히 구분해 기술해야 한다.**

### A를 택할 때 필요한 부수 작업

- ATM의 bundle별 정규화 상수 `supp/{bundle}_{min,max}_coords_rigid_transformed_streamlines.npy`는
  rigid-2009c-1mm 기준이다. 템플릿 공간이 바뀌면 **우리 데이터로 새 상수를 계산**해야 한다.
  (fine-tuning을 하는 이상 어차피 필요한 작업이다.)
- ATM 격자 (193,229,193)@1mm vs GT 격자 (80,100,80)@2mm — SC builder는 mm 좌표로 통일해 처리한다.
  격자를 맞출 필요는 없고 affine만 정확하면 된다.

---

## 5. 충돌 C3 — GT SC 생성 스크립트는 불필요, 대신 `.tt.gz` 디코더가 필요

v2 §28의 `scripts/05_build_gt_sc.py`는 삭제한다 (SC_weight/SC_length가 이미 있음).

대신 다음이 필요하고, **이미 작성·검증했다** (`scratchpad/verify_sc.py`):

- `.tt.gz` → streamline 배열 디코더 (버퍼 정확 소진 검증 완료, 1e6 tracks ≈ 66 s/subject)
- hard `pass` SC / `end` SC / edge length 계산기 (GT와 r=0.9986 확인)

이 디코더는 세 곳에서 재사용된다: bundle coverage test(Stage 2), GT `L_roi_visit` 라벨 생성,
soft SC builder 검증(Stage 5).

---

## 6. 그 외 실무 이슈

| 이슈 | 내용 | 대응 |
|---|---|---|
| **streamline 수 스케일** | GT 1,000,000 vs ATM 3000×30 bundle = 90,000 (11.1배) | `L_SC_mag`에서 양쪽을 총합 정규화하거나 스케일 계수를 명시. 그냥 log-MAE를 쓰면 상수 offset이 그대로 손실이 된다 |
| **streamline 길이 표현** | GT 가변 길이 (평균 66점, 1mm step, 20–250 mm) vs ATM 고정 128점 등간격 | GT를 128점 등간격으로 resample. **원래 길이 `L_k`는 resample 전에 계산해 따로 저장** (등간격 128점은 길이 정보를 보존하지만 굴곡 손실 가능) |
| **`pass` 정의의 특성** | 긴 streamline 하나가 수십 개 edge를 만든다. density 84 %, 상위 10 % edge가 weight의 69 % | 기존 TVB 파이프라인 정의와 일치시키는 것이 목적이므로 **그대로 유지**. 다만 평가 시 density/degree를 반드시 함께 보고 |
| **작은 ROI** | label 71, 72가 2mm에서 15 voxel | soft assignment의 τ가 크면 이 ROI들이 뭉개진다. τ 스윕 시 per-ROI recall 확인 |
| **ATM 30-bundle coverage** | v2 §5.1 / Stage 2. TractSeg가 필요한데 실행기(singularity) 없음 | 대안: ATM 공식 `sub-1135`의 30개 GT bundle `.trk`(zip 내 존재)를 템플릿 공간에 올려 **어떤 ROI pair를 덮는지** 근사 측정. 정밀 측정은 TractSeg 확보 후 |
| **`infer.py` 실행 가능 범위** | 외부 도구 부재 | **생성 단계는 오늘 실행 가능**(torch+numpy+joblib+nibabel+sklearn만 필요). `affine_registration`/`filtering`/`trimming`/`to_native`는 전부 불가 |
| **upstream 결함** | v1 문서 §5의 D1–D9 전부 유효 | 특히 **D2(`.eval()` 미호출 → Dropout3d 활성 → inference 비결정적)**, D3(3000 하드코딩), D4(`supp/` vs `data/` 경로) |

---

## 7. 수정된 Loss 설계

```
L_total =
    λ_geo    * L_geo          # ATM 기하 (아래 참조)
  + λ_visit  * L_roi_visit    # ★신설: 통과 ROI 집합 multi-label BCE (GT SC 정의와 일치)
  + λ_end    * L_endpoint     # v2 §7 유지, 역할은 보조로 강등
  + λ_corr   * L_SC_corr      # 1 - pearson(upper_tri)
  + λ_mag    * L_SC_mag       # log-MAE, 총합 정규화 후
  + λ_len    * L_length       # GT edge mask 안에서 log-MAE
```

`L_geo`는 upstream에 loss 코드가 없으므로(D1) 우리가 정의한다. 둘 중 선택:

- **G-1** GT bundle 라벨이 확보되면: `MSE(S_pred, S_gt_resampled) + β·KL` (원 VAE objective 근사)
- **G-2 (기본값)**: **anchor loss** — 동결한 원본 decoder의 출력을 기준으로
  `L_anchor = mean ||S_pred(z,a) - S_orig(z,a)||²`. GT bundle 라벨이 필요 없고
  v2 §14("SC loss만 강하게 주면 경로가 비현실적으로 변한다")를 직접 방어한다.

여기에 등간격 정규화 `L_adj`를 더한다 (128점 등간격 전제):
`d_t = ||p_{t+1}-p_t||`, `L_adj = mean((d_t - mean(d_t))²)`

초기 가중치는 v2 §17을 따르되 `λ_visit`을 새로 잡아야 한다.
**첫 실험 전에 각 항의 gradient norm을 반드시 기록**하고 그 값으로 재조정한다 (v2 §17 지시).

---

## 8. 수정된 파일 계획

`external/atm_upstream/`은 **한 줄도 수정하지 않는다.** D2/D3/D4/D5는 adapter에서 우회한다.

### v2 §28 대비 변경

| v2 원안 | 변경 |
|---|---|
| `models/endpoint_assigner.py` | 유지하되 **`roi_assigner.py`**로 역할 확대 (per-point soft ROI + noisy-OR `u_k`) |
| `models/sc_builder.py` | **`pass` 정의로 재작성** (endpoint outer-product 아님) |
| `losses/endpoint.py` | **`losses/local.py`** — `L_roi_visit` + `L_endpoint` + `L_adj` |
| `losses/sc_corr.py` / `sc_magnitude.py` / `tract_length.py` | **`losses/sc.py` 하나로 통합** (3파일로 나눌 만큼 크지 않음) |
| `data/endpoint_labels.py` | **`data/tt_io.py`** — `.tt.gz` 디코더 + hard SC (이미 검증) + `L_roi_visit` 라벨 생성 |
| `scripts/05_build_gt_sc.py` | **삭제** (GT SC 이미 존재) |
| — | **신설 `scripts/02b_verify_gt_sc.py`** — `.tt.gz`에서 SC 재현해 `.mat`과 대조 (회귀 테스트) |

### 최종 신규 파일 (9개)

```
src/atm_sc/
├── models/atm_adapter.py     ATMVAE 로드·UNet freeze·anatomy feature 캐시·미분가능 decode+역정규화
├── models/roi_assigner.py    ROI distance map + grid_sample → q_t, noisy-OR → u_k
├── models/sc_builder.py      pass 기반 SC_weight / (Num,Den) length. bundle 가산 누적 지원
├── losses/local.py           L_roi_visit, L_endpoint, L_adj, L_anchor
├── losses/sc.py              L_SC_corr, L_SC_mag, L_length
├── data/tt_io.py             .tt.gz 디코더, hard pass/end SC, 128점 resample, GT 라벨
├── data/dataset.py           subject split, T1 로드, GT SC/length, anatomy cache 조회
├── training/trainer.py       2-pass bundle gradient 누적 (v1 문서 §6(b))
└── inference/generate.py     30 bundle 생성 → 합성 → trk export
```

`trainer.py`의 2-pass 누적은 v1 문서 §6(b)에 유도해 둔 그대로 유효하다.
`SC_total = Σ_b SC_b` 가 bundle에 대해 가산적이므로 `∂L/∂SC_b = ∂L/∂SC_total`.
length는 `(Num, Den)` 쌍으로 누적해 같은 트릭을 적용한다.
UNet(49.86 M, 97 %)을 동결하면 학습 대상은 **decoder 627 K params**뿐이고,
A10 23 GB에서 3D UNet이 학습 그래프에서 빠지므로 여유가 생긴다.

---

## 9. 수정된 Stage 순서

| Stage | 내용 | 상태 |
|---|---|---|
| **0** | `stable.zip` 나머지 압축 해제 → `sub-1135` 공식 inference 재현 (**생성 단계까지만**; filtering/trimming/to_native는 도구 부재로 불가). D2 `.eval()` on/off 양쪽 기록 | 즉시 가능 |
| **1** | **좌표계 결정 (C2).** A/B/C 중 선택. A라면 T1→템플릿 비선형 정합 도구 확보가 선행 | **차단 중 — 사용자 결정 필요** |
| **2** | `.tt.gz` 디코더 회귀 테스트를 여러 subject로 확대 + GT SC 재현 검증 | 코드 검증 완료, 확대만 남음 |
| **3** | ATM 30-bundle → GT SC edge coverage 측정 (v2 §5.1) | TractSeg 필요 / 근사 대안 있음 |
| **4** | Soft `pass` SC builder 구현 + τ 튜닝 → hard `pass` SC 재현 | Stage 1 이후 |
| **5** | `L_geo(anchor) + L_adj` 만으로 decoder baseline fine-tune | |
| **6** | `+ λ_visit·L_roi_visit` (+보조 `L_endpoint`) | |
| **7** | `+ λ_corr·L_SC_corr` → `+ λ_mag·L_SC_mag` → `+ λ_len·L_length` | |
| **8** | ablation (v2 §21) + CoRNN 대비 benchmark (동일 streamline 수) | |

subject split: 238명(dwi_qc pass)을 **subject 단위**로 분할하고,
`group`(PD/HC)과 `batch2`(scanner_proto)로 **stratify**한다.
FC 데이터에 batch/proto 필드가 이미 있으므로 배치 효과가 있는 코호트다 —
train/test에 스캐너가 몰리지 않도록 해야 한다.
(v2 §27 준수: test SC는 λ 튜닝에 절대 사용하지 않는다.)

---

## 10. 지금 결정이 필요한 것

1. **[차단] C2 공간 문제** — A(ATM 입력을 템플릿 공간으로) / B(`.fib.gz` 확보해 GT를 native로) 중 어느 쪽인가?
   B를 원하면 `/mnt/d/.../derivatives/DSI_SC_len/*/dwi.fib.gz`를 이 서버로 복사할 수 있는지.
2. **외부 도구** — ANTs(또는 최소한 비선형 정합 수단), dipy, singularity/apptainer를 설치할 수 있는가?
   dipy는 `infer.py`의 `.trk` export에 필요하고, ANTs는 Stage 1 A안에 필요하다.
   (nibabel만으로 `.trk`를 쓰는 것도 가능하므로 dipy는 우회 가능하다.)
3. **`L_geo` 선택** — 30-bundle segmentation을 만들 수 있으면 G-1, 아니면 G-2(anchor). 기본값은 G-2.
4. **`L_roi_visit` 도입 동의 여부** — v2 §7의 `L_endpoint`를 보조로 내리고
   통과-집합 감독을 주 supervision으로 올리는 변경.

---

## 부록 A. 검증에 사용한 코드

`scratchpad/verify_sc.py` — `.tt.gz` 디코딩 → MNI mm → atlas ROI → `end`/`pass` SC 및 length,
`.mat`의 GT와 대조. sub-100001 기준 약 70 초. `src/atm_sc/data/tt_io.py`의 기반이 된다.

## 부록 B. 이 문서가 근거로 삼은 실측 목록

- `stable.zip` 코드 전량 (`infer.py`, `model/model.py`, `.pyc` 3개, `matlab_post/*.m`, `supp/*.npy` 120개)
- `atmvae_AF_L.pth` `strict=True` 로드 · 파라미터 51.38 M / UNet 49.86 M / decoder 627 K
- `ConvVAE.decode` 실제 forward: `[N,64] → [N,3,128] → [N,128,3]`
- `kde_models/AF_L/kde_model.joblib` 로드 · tophat/bw=1/447,000×64
- `PPMI_QC263_tracto.zip` manifest 263행 집계 + 5 subject 실측 (`sub-100001, 100005, 100268, 3385, 4011`)
- `sub-100001` 1,000,000 streamline 전량 디코딩 및 GT SC 재현
- `DesikanCortexPD25_*.nii.gz` 82 라벨 확인
- `FC_DKPD25_82_ppmi_all_nomed_qc.mat` 238행 스키마
- 실행 환경 (GPU/패키지/바이너리) 조사
