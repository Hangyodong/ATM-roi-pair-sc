"""UNet 중간 feature map 의 ROI 별 국소 풀링과 아틀라스 정렬.

전뇌 global average pooling 이 개인차를 지운다는 것은 측정으로 확정됐다
(stage3 공간맵 subject 간 상관 0.575 -> pooling 직후 0.9996, FINDINGS §B-6).
전뇌 평균 대신 82 ROI 별로 풀링해 ``f[82, C]`` 를 얻고, pair (i,j) 조건화에
``f[i], f[j]`` 를 쓴다.

격자 대응은 추측이 아니라 conv 산술이다
(``stable/stable/model/model.py`` 148-166 의 ``rigid_UNet`` 인코더):

  stage1  conv1_1/1_2/1_3  k=3 s=1 p=1  -> 크기 불변 (193,229,193)
  stage2  conv2_1          k=3 **s=2** p=1  -> (97,115,97), 나머지는 s=1
  stage3  conv3_1          k=3 **s=2** p=1  -> (49,58,49),  나머지는 s=1

  k=3, s=2, p=1 은 출력 j 가 입력 [2j-1, 2j, 2j+1] 을 보므로 receptive field 중심이
  **정확히 2j** 다. 두 번 거치면 stage3 출력 index k <-> UNet 입력 index 4k.
  k = 0..48 -> 입력 0, 4, ..., 192 로 193 축을 정확히 덮는다. 193/4 = 48.25 라서
  '4배가 아니다' 로 보이지만 실제 대응은 (193-1)/4 + 1 = 49 로 딱 맞는다.

  => feature 격자 affine = 입력 affine 의 3x3 에 4 를 곱한 것. **원점(translation)은 그대로**다
     (feature index 0 이 입력 index 0 에 대응하므로).

아틀라스(2mm)·UNet 입력(1mm 193^3)·feature(49x58x49) 세 격자가 전부 다르므로
재표본화는 mm 를 경유해서만 한다. 라벨 데이터라 **nearest neighbor 만** 쓴다.
W 격자의 x 축은 +1mm/voxel 인데 아틀라스는 -2mm/voxel 이라 voxel index 를 직접
비교하면 좌우가 조용히 뒤집힌다 -- 그래서 ``check_lateralization`` 이 필수다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.paths import ATLAS, SC_MAT
from ..spaces import W_AFFINE, W_SHAPE, apply_affine

N_ROI = 82
STAGE3_STRIDE = 4          # conv2_1(s=2) x conv3_1(s=2)
STAGE3_SHAPE = (49, 58, 49)


# --- 격자 -------------------------------------------------------------------
def feature_grid(in_shape=W_SHAPE, in_affine=W_AFFINE, stride: int = STAGE3_STRIDE):
    """다운샘플된 feature 격자의 (shape, affine).

    stride-2, k=3, p=1 conv 는 출력 j <-> 입력 2j 이므로 원점은 이동하지 않는다.
    """
    in_affine = np.asarray(in_affine, np.float64)
    assert in_affine.shape == (4, 4), in_affine.shape
    shape = tuple((int(s) - 1) // stride + 1 for s in in_shape)
    aff = in_affine.copy()
    aff[:3, :3] = in_affine[:3, :3] * float(stride)
    return shape, aff


def _labels_from_atlas(atlas_path, shape, affine, n_roi: int) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(atlas_path))
    lab = np.asanyarray(img.dataobj)
    assert lab.size > 0, f"아틀라스가 비었다: {atlas_path}"
    lab = np.rint(lab).astype(np.int16)
    present = np.unique(lab)
    assert present.min() >= 0 and present.max() == n_roi, (
        f"아틀라스 라벨 범위 {present.min()}..{present.max()} != 0..{n_roi}")
    assert len(present) == n_roi + 1, (
        f"아틀라스에 라벨이 {len(present) - 1}개뿐 (기대 {n_roi}). 아틀라스 파일 확인")

    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vox = np.stack([ii, jj, kk], -1).reshape(-1, 3).astype(np.float64)
    mm = apply_affine(affine, vox)                                   # feature -> mm
    av = np.rint(apply_affine(np.linalg.inv(img.affine), mm)).astype(np.int64)  # mm -> atlas voxel (nearest)
    ok = np.all((av >= 0) & (av < np.asarray(lab.shape)), axis=1)
    out = np.zeros(len(av), np.int16)
    out[ok] = lab[av[ok, 0], av[ok, 1], av[ok, 2]]
    return out.reshape(shape)


def atlas_on_feature_grid(atlas_path=ATLAS, feat_shape=STAGE3_SHAPE, in_shape=W_SHAPE,
                          in_affine=W_AFFINE, n_roi: int = N_ROI,
                          return_affine: bool = False) -> np.ndarray:
    """82-ROI 아틀라스를 feature 격자로 재표본화 (nearest). 라벨 소실을 assert 로 잡는다.

    feat_shape 는 in_shape 에서 유도한 격자와 일치해야 한다 (stride 를 역산해 확인).
    """
    feat_shape = tuple(int(s) for s in feat_shape)
    stride = round((int(in_shape[0]) - 1) / (feat_shape[0] - 1))
    shape, aff = feature_grid(in_shape, in_affine, stride)
    assert shape == feat_shape, (
        f"feature 격자 {feat_shape} 가 입력 {tuple(in_shape)} / stride {stride} 에서 유도한 {shape} 와 다르다")

    labels = _labels_from_atlas(atlas_path, shape, aff, n_roi)
    cnt = roi_voxel_counts(labels, n_roi)
    missing = np.flatnonzero(cnt == 0) + 1
    assert missing.size == 0, (
        f"재표본화에서 ROI {missing.tolist()} 가 소실됐다 (voxel 0개). "
        f"feature 격자 {shape} 가 너무 성기다 -- 상위 stage 나 soft 할당을 써야 한다")
    assert labels.dtype == np.int16 and labels.shape == shape
    return (labels, aff) if return_affine else labels


def roi_voxel_counts(labels: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """[n_roi] 배경(0) 제외 ROI 별 voxel 개수. index i 는 라벨 i+1."""
    return np.bincount(np.asarray(labels).ravel(), minlength=n_roi + 1)[1:n_roi + 1]


# --- 좌우 검사 ---------------------------------------------------------------
def roi_names(mat_path=SC_MAT) -> list[str]:
    """.mat 의 region_names (82개, 라벨 1..82 순서). 이름 앞의 L_/R_ 이 반구다."""
    import scipy.io as sio
    m = sio.loadmat(str(mat_path), variable_names=["region_names"], squeeze_me=True)
    names = [str(x).strip() for x in np.asarray(m["region_names"]).ravel()]
    assert len(names) == N_ROI, len(names)
    return names


def roi_centroids_mm(labels: np.ndarray, affine: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """[n_roi, 3] ROI 무게중심 (mm). voxel 이 없는 ROI 는 assert 로 걸린다."""
    cnt = roi_voxel_counts(labels, n_roi)
    assert (cnt > 0).all(), f"voxel 0 인 ROI: {(np.flatnonzero(cnt == 0) + 1).tolist()}"
    idx = np.argwhere(np.asarray(labels) > 0).astype(np.float64)
    lab = np.asarray(labels)[np.asarray(labels) > 0].astype(np.int64) - 1
    mm = apply_affine(affine, idx)
    s = np.zeros((n_roi, 3))
    np.add.at(s, lab, mm)
    return s / cnt[:, None]


def check_lateralization(labels: np.ndarray, affine: np.ndarray, names=None,
                         n_roi: int = N_ROI) -> dict:
    """L_*/R_* ROI 무게중심의 x 부호가 affine 기준으로 맞는지 (RAS: +x = 오른쪽).

    아틀라스나 격자를 바꿀 때마다 돌린다. 1mm/2mm 사이에서 좌우가 뒤집힌 적이 있다.
    """
    names = names or roi_names()
    c = roi_centroids_mm(labels, affine, n_roi)
    side = np.array([1 if n.startswith("R_") else (-1 if n.startswith("L_") else 0) for n in names])
    assert (side != 0).all(), "L_/R_ 접두어가 없는 ROI 이름이 있다"
    bad = [{"roi": int(i + 1), "name": names[i], "x_mm": round(float(c[i, 0]), 2)}
           for i in np.flatnonzero(np.sign(c[:, 0]) != side)]
    out = {"n_bad": len(bad), "bad": bad,
           "mean_x_left": round(float(c[side < 0, 0].mean()), 3),
           "mean_x_right": round(float(c[side > 0, 0].mean()), 3),
           "min_abs_x": round(float(np.abs(c[:, 0]).min()), 3)}
    assert not bad, f"좌우 반전 의심: {bad[:5]} (L 평균 x {out['mean_x_left']}, R {out['mean_x_right']})"
    assert out["mean_x_left"] < 0 < out["mean_x_right"], out
    return out


# --- 풀링 -------------------------------------------------------------------
def roi_pool(feat: torch.Tensor, labels, n_roi: int = N_ROI) -> torch.Tensor:
    """feat [1,C,D,H,W] · labels [D,H,W] -> [n_roi, C] ROI 내 평균.

    라벨 0(배경)은 버린다. voxel 이 없는 ROI 는 0 을 돌려주지 않고 assert 로 실패시킨다.
    """
    assert feat.ndim == 5 and feat.shape[0] == 1, feat.shape
    lab = torch.as_tensor(np.ascontiguousarray(labels), dtype=torch.long, device=feat.device)
    assert tuple(lab.shape) == tuple(feat.shape[2:]), (
        f"labels {tuple(lab.shape)} != feature 격자 {tuple(feat.shape[2:])}")
    assert torch.isfinite(feat).all(), "feature 에 NaN/Inf"
    assert float(feat.abs().max()) > 0, "feature 가 전부 0 -- T1 정규화/입력 확인"

    c = feat.shape[1]
    flat = feat.reshape(c, -1).t()                                   # [V, C]
    idx = lab.reshape(-1)
    cnt = torch.bincount(idx, minlength=n_roi + 1).to(flat.dtype)
    assert (cnt[1:] > 0).all(), (
        f"voxel 0 인 ROI: {(torch.nonzero(cnt[1:] == 0).flatten() + 1).tolist()}")
    s = torch.zeros(n_roi + 1, c, dtype=flat.dtype, device=flat.device)
    s.index_add_(0, idx, flat)
    out = s[1:] / cnt[1:, None]
    assert out.shape == (n_roi, c) and torch.isfinite(out).all()
    return out


def global_pool(feat: torch.Tensor) -> torch.Tensor:
    """대조군: 전뇌 global average pooling [C] (현재 encode_anatomy 가 하는 일)."""
    assert feat.ndim == 5 and feat.shape[0] == 1, feat.shape
    return feat.reshape(feat.shape[1], -1).mean(1)
