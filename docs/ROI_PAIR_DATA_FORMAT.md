# ROI-pair 데이터 포맷 (구현 계약)

모든 좌표는 **MNI mm (MNI152NLin6Asym), float32**. `.tt.gz` → `tt_io.load_streamlines()` 가 이미 mm 로 준다.
ROI 인덱스는 **0-based** (atlas label 1..82 → 0..81). `R = 82`.
canonical pair: `a = min(i,j), b = max(i,j)`, `pair_index = a*R + b`.

## `outputs/roi_pairs/{sub}/assignments.npz`  (scripts/02_assign_roi_pairs.py)

| key | shape / dtype | 의미 |
|---|---|---|
| `start_roi` | [N] int16 | 첫 점의 ROI (nearest voxel). 배경이면 -1 |
| `end_roi` | [N] int16 | 마지막 점의 ROI |
| `pair` | [N,2] int16 | canonical (a,b). start/end 중 하나라도 -1 이거나 a==b 면 (-1,-1) |
| `length_mm` | [N] float32 | 재샘플 **전** 원래 길이 |
| `sc_end` | [R,R] int64 | endpoint rule SC (대칭, 대각 0) — **endpoint 모드 학습 target** |
| `len_end` | [R,R] float32 | endpoint rule 평균 길이 (mm), edge 없으면 0 |
| `sc_pass` | [R,R] int64 | pass rule SC (`.mat` GT 와 같은 정의, 평가 참조) |
| `len_pass` | [R,R] float32 | |
| `n_total`, `n_assigned` | int | |

N 은 streamline 순서 그대로 (`track` 다음 `track1`). 순서를 바꾸지 않는다.

## `outputs/roi_pairs/{sub}/bundles.npz`  (scripts/03_build_roi_pair_bundles.py)

pair 별 최대 `cap` 개(기본 256)를 **결정적으로**(seed 고정) 골라 128점으로 재샘플해 pair_index 오름차순으로 저장.

| key | shape / dtype | 의미 |
|---|---|---|
| `streamlines` | [M,128,3] float16 | 등간격 128점, mm |
| `lengths` | [M] float32 | 원래 길이 (재샘플 전) |
| `pair` | [M,2] int16 | canonical |
| `pair_index` | [M] int32 | a*R+b, **오름차순 정렬** |
| `src_index` | [M] int64 | assignments 의 원래 streamline 인덱스 |
| `pair_ids` | [K,2] int16 | 양성 pair 목록 (정렬) |
| `pair_offsets` | [K+1] int64 | pair k 의 행 = `offsets[k]:offsets[k+1]` |
| `pair_count_full` | [K] int64 | cap 이전 GT 개수 (= sc_end[a,b]) |
| `cap`, `seed`, `n_roi` | int | |

## `outputs/cache/dist_maps.npy`  (scripts/05_build_distance_maps.py)
[R, 91, 109, 91] float32 mm. `models/endpoint_assigner.build_distance_maps` 로 생성. atlas affine 은 atlas nii 에서.

## `outputs/cache/{sub}_T1w_syn_W.npy`  (scripts/01 / data/prepare_t1.py)
[193,229,193] float32, 정규화 전. W 격자 (`spaces.W_AFFINE`).

## 학습 target 규칙
endpoint 모드 → `sc_end`/`len_end`. pass 모드 → `.mat` 의 `SC_weight`/`SC_length` (= `sc_pass`).
둘을 섞으면 상관 상한이 0.67 이다 (실측). 절대 섞지 않는다.
