# 이 학습에 맞는 tractogram 추출 방법

작성 2026-09-09. 근거는 전부 이 저장소에서 실측한 값이다 (§1 표의 출처는 §7).

지금 GT(`PPMI_QC263_tracto/*/sub-*_tract.tt.gz`, DSI Studio QSDR)로 학습한 결과가 어디서
막혔는지에서 거꾸로 나온 요구사항 문서다. 새 데이터를 만들 때 이대로 하면 지금의 실패
지점들이 사라진다.

---

## 1. 지금 GT 의 무엇이 문제인가

| 측정 | 값 | 뜻 |
|---|---|---|
| latent 조건평균의 subject 성분 | **4.4 %** | 연결의 **모양**에 개인차가 거의 없다 |
| GT SC 카운트의 subject 성분 | 18.5 % | 개인차는 **세기**에만 남아 있다 |
| T1 → SC 잔차 선형 상한 (val 31명) | **0.153** | 세기의 개인차조차 T1 으로 잘 안 보인다 |
| 생성 SC 의 "맞는 분산" 몫 | 0.25 ~ 0.74 % | subject 구별이 사실상 안 된다 |

**원인은 QSDR 이다.** DSI Studio 가 tractogram 을 QSDR 로 **비선형** 정규화해 템플릿 공간에
올려놓았다 (전 subject 동일 공간). 개인 뇌 형태가 그 워프에 흡수돼 사라졌다. 그래서 디코더가
아무리 잘 배워도 가닥 **모양**에서 개인차를 만들 수 없다.

되돌리려면 그 subject 의 QSDR 워프 필드가 필요하고, 그건 `.fib.gz` 에 있다. 지금 없다
(subject 폴더에 `T1w.nii.gz` 와 `tract.tt.gz` 뿐).

부수 문제 하나 더: GT 는 **가닥 가중치가 없다**. SC 가 순수 가닥 수라 tractography 의 알려진
편향(길이·곡률 의존)이 그대로 SC 에 들어간다. SIFT2 가 이걸 고친다.

---

## 2. 요구사항 (실패에서 거꾸로 도출)

| # | 요구 | 왜 |
|---|---|---|
| R1 | **native space 로 뽑고 rigid 로만 공통 공간에 맞춘다** | QSDR 이 지운 개인 형태를 보존한다. T1 전처리는 이미 rigid 프로토콜이다 |
| R2 | **ACT (해부학적 제약)** | 지금 생성 가닥의 백질 점유가 0.393 (GT 0.825) 이다. GT 쪽에서 미리 백질을 지키게 한다 |
| R3 | **SIFT2 가중치** | 가닥 수 편향 제거. SC 목표값 자체의 질이 올라간다 |
| R4 | **100 만 가닥, 길이 20~250 mm** | 지금 GT 와 같은 규모라야 비교가 성립한다 |
| R5 | **SC 는 pass 모드 Case B** | 이미 검증됨: 100 만 가닥 재현 시 `.mat` 과 r **0.9987** |
| R6 | **워프/변환 파일을 전부 보관** | 나중에 공간을 바꿀 수 있어야 한다. 지금 이게 없어서 막혔다 |

---

## 3. 필요한 입력

subject 당 이것들이 있어야 한다. **현재 하나도 없다** (T1 과 이미 만들어진 tractogram 뿐).

```
sub-XXXXX_dwi.nii.gz      확산 강조 영상
sub-XXXXX_dwi.bval        b-value
sub-XXXXX_dwi.bvec        확산 방향
sub-XXXXX_T1w.nii.gz      T1 (이미 있음)
(선택) 역위상 b0          topup 용. 없으면 왜곡 보정 품질이 떨어진다
```

PPMI 확산 데이터는 LONI 에서 받는다. 이 코호트(`PPMI_QC263`)의 T1 을 이미 갖고 있으므로
같은 subject 의 DWI 를 받으면 된다.

---

## 4. 도구 현황

| 단계 | 명령 | 상태 |
|---|---|---|
| 잡음 제거 | `dwidenoise` | 있음 (MRtrix 3.0.8) |
| Gibbs 링 | `mrdegibbs` | 있음 |
| **와전류·움직임 보정** | `eddy` (FSL) | **없음 — 유일한 차단점** |
| 편향 보정 | `dwibiascorrect ants` | 있음 (ANTs 0.6.3) |
| 응답함수 | `dwi2response dhollander` | 있음 |
| FOD | `dwi2fod msmt_csd` | 있음 |
| 5TT | `5ttgen` (FSL/FreeSurfer 필요) | **불가 → §5.4 로 자체 조립** |
| Tractography | `tckgen -act -backtrack` | 있음 |
| 가닥 가중치 | `tcksift2` | 있음 |
| 커넥톰 | `tck2connectome` | 있음 (우리는 자체 `hard_sc` 를 쓴다) |

**FSL 을 설치하거나 컨테이너로 `eddy` 를 확보해야 한다.** `/scratch/home/wog3597/containers/`
가 있으니 FSL singularity 이미지가 현실적인 경로다. `dwifslpreproc` 가 내부에서 FSL 을
호출하므로 FSL 만 있으면 MRtrix 쪽은 그대로 쓴다.

---

## 5. 추출 절차

### 5.1 DWI 전처리

```bash
mrconvert sub_dwi.nii.gz dwi.mif -fslgrad sub_dwi.bvec sub_dwi.bval
dwidenoise dwi.mif dwi_dn.mif -noise noise.mif
mrdegibbs dwi_dn.mif dwi_dg.mif
dwifslpreproc dwi_dg.mif dwi_pp.mif -rpe_none -pe_dir AP    # eddy. 역위상 b0 있으면 -rpe_pair
dwibiascorrect ants dwi_pp.mif dwi_bc.mif -bias bias.mif
dwi2mask dwi_bc.mif mask.mif
```

QC (조용히 실패하는 지점이다):
- `noise.mif` 의 뇌 안 평균이 0 이 아닐 것 (0 이면 denoise 가 아무 일도 안 했다)
- `mask.mif` 복셀 수가 코호트 중앙값의 ±30 % 안일 것
- eddy 의 움직임 파라미터 최댓값이 3 mm 를 넘는 subject 는 표시해 둘 것

### 5.2 응답함수와 FOD

```bash
dwi2response dhollander dwi_bc.mif wm.txt gm.txt csf.txt -voxels rf_voxels.mif
dwi2fod msmt_csd dwi_bc.mif wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif -mask mask.mif
mtnormalise wmfod.mif wmfod_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask mask.mif
```

PPMI 확산이 단일 shell(b=1000) 이면 `msmt_csd` 대신 다음을 쓴다.

```bash
dwi2response tournier dwi_bc.mif wm.txt
dwi2fod csd dwi_bc.mif wm.txt wmfod.mif -mask mask.mif
```

QC: `wmfod.mif` 의 l=0 성분이 백질에서 회백질보다 클 것. 아니면 gradient 방향(bvec) 부호가
틀린 것이다 — **이 프로젝트에서 가장 조용히 틀리는 종류다**.

### 5.3 T1 정합 (native 유지가 핵심)

R1 때문에 **DWI 를 T1 에 맞추지 말고 T1 을 DWI 에 맞춘다** (혹은 강체 변환만 쓴다).
비선형 정합을 쓰면 여기서 개인 형태가 다시 지워진다.

```bash
dwiextract dwi_bc.mif - -bzero | mrmath - mean b0.mif -axis 3
mrconvert sub_T1w.nii.gz T1.mif
# 강체(6 자유도)만. affine/nonlinear 금지
flirt -in T1.nii.gz -ref b0.nii.gz -dof 6 -omat t1_to_dwi.mat   # FSL 확보 시
transformconvert t1_to_dwi.mat T1.nii.gz b0.nii.gz flirt_import t1_to_dwi.txt
mrtransform T1.mif -linear t1_to_dwi.txt T1_indwi.mif
```

FSL 없이 하려면 ANTs 강체로 대체한다 (`antsRegistration -t Rigid[0.1]`, 파이썬 `ants` 모듈로
호출 가능). **`-t SyN` 을 쓰면 안 된다.**

### 5.4 5TT 조립 (5ttgen 대체)

`5ttgen` 이 FSL/FreeSurfer 를 요구하므로 우리 Atropos 3-class 산출물로 직접 만든다.
`src/atm_sc/data/wm_segment.py` 의 `wm_probability(..., return_all=True)` 가 CSF/GM/WM 확률과
subject 뇌 마스크를 준다.

5TT 는 4차원 [X,Y,Z,5] 이고 채널 순서가 정해져 있다.

```
0 cortical GM      Atropos GM 확률 x (피질 아틀라스 라벨 마스크)
1 subcortical GM   Atropos GM 확률 x (피질하 아틀라스 라벨 마스크)
2 WM               Atropos WM 확률
3 CSF              Atropos CSF 확률
4 pathological     0 (해당 없음)
```

각 복셀에서 합이 1 이 되도록 정규화하고 `mrconvert` 로 `.mif` 저장. `5ttcheck` 로 검증한다.

주의: 이 5TT 는 T1 기반이라 **DWI 공간으로 옮겨야** 한다 (§5.3 의 변환 사용).

### 5.5 Tractography

```bash
tckgen wmfod_norm.mif tracks_2M.tck \
  -act 5tt_indwi.mif -backtrack -crop_at_gmwmi \
  -seed_dynamic wmfod_norm.mif \
  -select 2000000 -maxlength 250 -minlength 20 \
  -cutoff 0.06 -nthreads 8
```

- `-act -backtrack -crop_at_gmwmi` 가 R2 다. 백질 안에서 전파하고 GM/WM 경계에서 끝난다.
- `-seed_dynamic` 이 SIFT 편향을 미리 줄인다.
- 200 만을 뽑고 SIFT2 를 거친 뒤 100 만으로 맞춘다 (R4).

### 5.6 SIFT2

```bash
tcksift2 tracks_2M.tck wmfod_norm.mif weights.txt \
  -act 5tt_indwi.mif -out_mu mu.txt -nthreads 8
```

`weights.txt` 는 가닥당 실수 가중치다. **SC 는 가닥 수가 아니라 이 가중치의 합**이 된다 (R3).
`mu.txt` 는 전역 배율이라 subject 간 비교에 필요하다 — 반드시 보관한다.

### 5.7 공통 공간으로 (R1)

학습은 공통 공간에서 한다. **강체만** 쓴다.

```bash
# DWI(native) -> MNI152NLin6, 강체 6 자유도
mrtransform tracks_2M.tck ... # tck 는 tcktransform 이 아니라 warp 로 옮긴다
tcktransform tracks_2M.tck rigid_warp.mif tracks_mni.tck
```

강체 변환은 `transformcalc`/`warpinit` 로 워프 필드를 만들어 적용한다. 결과는 개인 뇌 형태를
그대로 유지한 채 위치·방향만 정렬된 tractogram 이다.

**보관할 것 (R6)**: `t1_to_dwi.txt`, native→MNI 강체 행렬, `mu.txt`, `weights.txt`, 5TT.
지금 데이터가 막힌 이유가 이것들이 없어서다.

---

## 6. 이 저장소에서 바뀌는 것

새 데이터가 오면 코드가 이만큼 바뀐다. 미리 알고 짜야 한다.

| 대상 | 변경 |
|---|---|
| `data/tt_io.py` | `.tt.gz` 대신 `.tck` 읽기 추가. `hard_sc` 에 **가닥 가중치 인자** 추가 (SIFT2) |
| `data/trk_to_roi_pairs.py` | 가중치를 함께 집계 |
| SC 정의 | 가닥 수 → 가중 합. `sc_end`/`sc_pass` 둘 다 |
| 템플릿·통계 | `sc_template_stats.npz`, `anat_tier1_pair_stats.npz`, 능선 가중치 전부 재생성 |
| 좌표 공간 | native+rigid. `spaces.py` 의 W 격자 재확인 |
| 학습 | 인코더부터 전면 재학습 |

`hard_sc` 에 가중치를 넣는 것과 좌표 공간을 인자로 빼는 것은 **지금 미리 해둘 수 있다**.

---

## 7. 비용

| 단계 | subject 당 | 206 명 (8 코어) |
|---|---|---|
| 전처리 (denoise~bias) | 15 ~ 30 분 | 1 ~ 2 일 |
| FOD | 5 ~ 10 분 | 4 시간 |
| tckgen 200 만 (ACT) | 30 ~ 60 분 | 1 ~ 2 일 |
| SIFT2 | 15 ~ 30 분 | 1 일 |
| **합계** | **1 ~ 2 시간** | **4 ~ 6 일** |

`base_8` 큐가 48 시간 walltime 이므로 재제출 안전하게(산출물 있으면 건너뛰기) 나눠 돌린다.
`base_32` 는 `acl_groups = base_32_group` 접근 제어가 걸려 있어 그룹 추가 요청이 필요하다.

디스크: 200 만 가닥 `.tck` 가 subject 당 약 2 GB, 100 만으로 줄이면 1 GB. 206 명이면 200 GB.
`/scratch` 에 448 TB 여유가 있다.

---

## 8. 단계별 검증 게이트

각 단계에서 이걸 통과 못 하면 다음으로 넘어가지 않는다. 신경영상 파이프라인은 조용히
실패하고 숫자는 그럴듯하게 나온다.

| 단계 | 게이트 |
|---|---|
| 전처리 | 마스크 복셀 수가 코호트 중앙값 ±30 %, 움직임 최댓값 < 3 mm |
| FOD | 백질 l=0 성분 > 회백질 (bvec 부호 검증) |
| 정합 | T1 뇌 안에 b0 뇌가 90 % 이상 포함 |
| 5TT | `5ttcheck` 통과, 복셀별 합 = 1 |
| tckgen | 가닥 수가 목표의 95 % 이상, 평균 길이 60~120 mm |
| SIFT2 | 가중치 분포가 한 점에 몰리지 않을 것 (변동계수 > 0.3) |
| SC | 비영 edge 2500 ~ 3000, 0 비율 0.15 ~ 0.30, 가닥당 pair 기여 5 ~ 10 |

마지막 줄이 지금 GT 의 실측값이다 (비영 2558, 0 비율 0.230, 가닥당 pair 7.50).

---

## 9. 요약

- **차단점은 두 개다**: 원본 DWI 가 없다, FSL `eddy` 가 없다. 나머지 도구는 다 있다.
- **가장 중요한 설계 결정은 R1** (native 로 뽑고 강체로만 정렬) 이다. 지금 데이터가 실패한
  근본 원인이 여기다.
- SIFT2 가중치와 ACT 는 SC 목표값의 질을 올린다. 지금 우리가 추론에서 하는 역산이
  SIFT2 와 같은 발상인데, 원본에서 제대로 하면 훨씬 낫다.
- 데이터를 준비하는 동안 §6 의 코드 작업과 `docs/PIPELINE_13` §10 의 P3·P4(z 용량과 prior
  정렬) 를 진행할 수 있다. 그 둘은 새 데이터가 와도 그대로 필요하다.
