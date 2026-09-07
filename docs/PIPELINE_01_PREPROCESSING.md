# 전처리

---

## 1. 좌표 공간

세 공간이 mm 단위로 서로 맞아야 한다. 어긋나면 ROI 할당과 SC 가 통째로 틀리는데
**코드는 정상 종료하고 숫자도 그럴듯하게 나온다.** `scripts/01_qc_coordinate_space.py` 가
overlay 그림까지 만들어 눈으로 확인하게 한다.

| 공간 | 정의 | 크기 |
|---|---|---|
| **W 격자** (모델 입력) | ATM 상류가 쓰는 격자 | 193×229×193 @ 1mm |
| 아틀라스 | `DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz` | 91×109×91 @ 2mm |
| 템플릿 | `tpl-MNI152NLin6Asym_res-01_T1w.nii.gz` | 182×218×182 @ 1mm |

`W_AFFINE` (`src/atm_sc/spaces.py`): 회전 없음, 원점 (−96, −132, −78), 1mm 등방.

```
[[  1  0  0  -96]
 [  0  1  0 -132]
 [  0  0  1  -78]
 [  0  0  0    1]]
```

## 2. T1 전처리

`src/atm_sc/data/prepare_t1.py`

```
native T1 (PPMI, subject 마다 크기/강도 제각각)
  ↓ ants.registration(fixed=TEMPLATE, moving=T1, type_of_transform="Rigid" | "SyN")
템플릿 공간 영상
  ↓ resample_to_W(order=1)   -- mm 로만 연결, 격자 가정 없음
W 격자 float32 [193,229,193]  → outputs/cache/{sub}_T1w_{mode}_W.npy
```

**정합 방식**

| | Rigid | SyN (기존) |
|---|---|---|
| 변환 | 회전 + 평행이동 (6 DOF) | 비선형 워프 |
| 개인 뇌 형태 | **보존** | 템플릿 모양으로 구부러짐 |
| 복셀 대응 | 근사 | 정확 |
| subject 간 영상 상관 (8명 실측) | 0.731 | 0.900 |
| 소요 시간 | ~155초/명 | ~6분/명 |

SyN 은 개인 형태 정보를 **워프 필드로 옮기는데 그 필드를 저장하지 않았다.**
현재 프로토콜은 **Rigid** 로 확정 (`scripts/37_build_rigid_t1.py`).

**강도 정규화** (`BundleNorm.normalize_t1`, `src/atm_sc/models/atm_adapter.py`)

PPMI T1 은 native max 가 878 ~ 203,163 으로 subject 마다 230배 차이난다 (스캐너/프로토콜).
ATM 상류의 고정 상수(8330)만 쓰면 정규화 후 >1 인 voxel 이 39 % 인 subject 가 생기고
anatomy feature 가 40배 커진다.

```python
# unit=True (현재 프로토콜) — [0,1] 보장
scale = percentile(vol[vol > 0], 99.5)
out   = clip(vol / scale, 0, 1)
```

99.5 백분위를 1.0 에 맞추고 위를 자른다. 밝은 이상치(지방·혈관)가 스케일을 지배하지 않으면서
범위는 확실히 [0,1] 이 된다.

| subject | 기존 방식 최대 | `unit=True` 최대 |
|---|---|---|
| sub-000004 | 0.890 | 1.000 |
| sub-000001 | **1.208** | 1.000 |
| sub-4030 | **1.237** | 1.000 |

## 3. Streamline (GT tractogram)

`src/atm_sc/data/tt_io.py`

```
.tt.gz (DSI Studio, QSDR 템플릿 공간)
  ↓ mat 헤더의 trans_to_mni [4,4]:  tt voxel -> MNI mm
streamline [N, ?, 3] mm  (점 개수 가변)
  ↓ 호 길이 등간격 재샘플
[N, 128, 3] mm
```

- subject 당 정확히 **1,000,000** streamline
- 정합 검증(`check_alignment`): warp 된 T1 의 뇌 안에 그 subject 의 streamline 이
  **90 % 이상** 들어가야 한다. 이 검사가 이 파이프라인에서 가장 조용히 실패하기 쉬운 지점이다.

## 4. SC 정의 — 측정으로 확정

DSI Studio `.mat` 의 `SC_weight` 가 무엇을 세는지 두 가설을 실측 비교했다.

| 가설 | 정의 | GT 와의 상관 | 재현된 edge |
|---|---|---|---|
| Case A (인접 전이) | 연속으로 지나는 ROI 쌍만 | 0.781 | 700 / 2,785 |
| **Case B (모든 방문 쌍)** | 지나간 ROI 들의 **모든 쌍** | **0.9986** | 전부 |

**Case B 가 정답이다.** A→B→C→D 를 지나는 streamline 은 (A,B) (A,C) (A,D) (B,C) (B,D) (C,D)
여섯 쌍 모두에 +1 한다. GT edge 의 77 % 는 인접 전이만으로는 만들 수 없다.

`hard_sc(mm, npts, atlas, affine, n_roi, mode)` 가 두 정의를 모두 제공한다.

| 배열 | 정의 | 206명 평균 | 표준편차 | 비영 edge |
|---|---|---|---|---|
| `sc_end` | 양 **끝점**이 서로 다른 ROI | **459,716** | 24,273 | 1,868.6 |
| `sc_pass` | **통과**한 모든 ROI 쌍 (= GT `.mat`) | **7,408,486** | 995,099 | 2,823.0 |
| `len_pass` | 그 edge 를 만든 streamline 들의 평균 길이 | — | — | — |

- 100만 가닥 중 **46 %** 만 양 끝점이 ROI 안에 있다 (범위 413,647 ~ 513,294)
- streamline 하나가 평균 **16.1 개** ROI 쌍을 방문한다 (7.41M / 459.7k)

## 5. ROI 쌍 bundle

`scripts/02_assign_roi_pairs.py` → `assignments.npz`, `scripts/03_build_roi_pair_bundles.py` → `bundles.npz`

```
assignments.npz   start_roi[1M] end_roi[1M] pair[1M,2] length_mm[1M]
                  sc_end[82,82] sc_pass[82,82] len_end len_pass n_total n_assigned
bundles.npz       streamlines[N,128,3] float16   pair_ids[K,2]  pair_offsets[K+1]
                  pair_count_full[K]  cap=256  (pair 당 최대 256개만 보관)
```

cap 256 때문에 bundle 에 남는 것은 subject 당 평균 **153,940** 가닥 (전체의 15 %).
학습용 표본이지 SC 계산용이 아니다 — SC 는 항상 100만 가닥 전체로 계산된 `sc_pass` 를 쓴다.

## 6. SC edge 정렬 segment

`src/atm_sc/data/edge_segments.py`, `scripts/26_build_edge_segments.py`

번들 정의를 SC edge 와 1:1 로 맞추기 위해, 하나의 streamline 을 **방문한 모든 ROI 쌍**에 대해
부분 경로로 쪼갠다 (Case B 정의와 일관).

```python
dwell_intervals(labels, min_dwell=2)   # 아틀라스 지터 제거
streamline_segments(...)               # 쌍마다 가장 긴 부분 경로 1개
SEG_POINTS = 32,  MIN_LEN_MM = 4.0
```

205명 실측:

| 항목 | 평균 |
|---|---|
| streamline 당 segment | 11.64 |
| 원 segment 수 | 1,798,416 |
| 보관된 segment 수 | 203,846 |
| segment SC 와 GT SC 의 상관 | **0.937** (로그 0.954) |
| GT edge 재현율 | 0.891 |

## 7. White matter 확률 맵

`src/atm_sc/data/wm_segment.py`, `scripts/36_build_wm_maps.py`

FreeSurfer `recon-all` 은 subject 당 6~10시간(206명이면 10~14일)이고 이 머신에 설치되어
있지 않다. 인코더가 3D UNet 이라 표면 메시를 받아도 볼륨으로 되돌려야 하므로,
**ANTs Atropos 3-class 분할의 WM 확률 볼륨**으로 대체한다 (subject 당 **~35초**).

```python
mask = brain_mask_W(erode_mm=4)        # 템플릿 뇌 마스크
seg  = ants.atropos(a=T1, x=mask, i="kmeans[3]", m="[0.2,1x1x1]", c="[5,0]")
wm   = 확률맵 중 가중평균 강도가 가장 높은 클래스   # T1 에서 WM > GM > CSF
```

클래스 번호를 고정으로 가정하지 않고 강도 순위로 고른다.

**뇌 마스크 처리 — 여기서 결함을 하나 잡았다.**

템플릿 뇌 마스크는 뇌실을 구멍으로 남긴다(10,714 복셀). 그대로 침식하면 구멍이
**66,329 복셀**로 부풀어 뇌실 주변 백질이 통째로 잘려나가고, 정육면체 구조요소는
상자 모양 인공물을 만든다. 수정:

```python
m = binary_fill_holes(m)          # 먼저 구멍을 메운다 (뇌실은 Atropos 가 CSF 로 분류)
m = binary_erosion(m, ball(4))    # 구형 구조요소로 침식
assert not (binary_fill_holes(m) & ~m).any()
```

**품질 관문** (조용한 실패를 막는다):

```python
assert 0.20 < WM비율 < 0.70        # 뇌 안에서
assert isfinite(wm).all() and wm.max() > 0.9
```

실측: 마스크 1,528,955 복셀, WM 42.1 % — 정상 범위(40~45 %).

## 8. 전처리 산출물

`outputs/cache/` 아래:

| 파일 | 내용 |
|---|---|
| `{sub}_T1w_rigid_W.npy` | rigid 정합 T1, float32 [193,229,193], **정규화 전** |
| `{sub}_T1w_syn_W.npy` | (구) SyN 정합 T1 |
| `{sub}_WM_W.npy` | WM 확률, float32 [193,229,193], 0~1 |
| `{sub}_anat_{bundle}.npy` | anatomy feature 캐시 (동결 인코더용) |

`outputs/roi_pairs/{sub}/` 아래: `assignments.npz` `bundles.npz` `visit.npz` `edge_segments.npz`

## 9. 실행 순서

```bash
python scripts/37_build_rigid_t1.py --workers 6        # T1 -> rigid MNI152
python scripts/36_build_wm_maps.py  --workers 6        # WM 확률 (rigid T1 기준)
python scripts/12_preprocess_batch.py                  # 02 ROI 할당 -> 03 bundle
python scripts/24_build_visitation.py                  # route loss 용 통과 ROI
python scripts/26_build_edge_segments.py               # SC edge 정렬 segment
python scripts/27_qc_thresholds.py                     # 실제 분포 기반 QC 임계값
python scripts/22_build_synthetic_cache.py             # GESTA synthetic
```
