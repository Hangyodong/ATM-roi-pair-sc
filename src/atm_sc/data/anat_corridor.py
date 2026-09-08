"""pair corridor / WM 표면 feature — 새 데이터 추출 없이 기존 WM 확률맵에서 나온다.

왜: 지금 tier1 pair feature 9개의 최강 항은 centroid **직선거리**다. 두 ROI 사이에 실제로
백질이 있는지는 안 본다. corridor(두 centroid 를 잇는 선 위의 WM 확률)는 SC 의 강한 예측자이고
`{sub}_WM_W.npy` 만으로 계산된다 (subject 당 밀리초). WM 표면적은 같은 맵의 0.5 등고면에
marching cubes 를 걸어 ROI 별로 모은다 (0.6초).

캐시: `outputs/cache/anat_corridor/{sub}_{source}.npz`
  corridor [P, CORRIDOR_DIM] float32   P = R(R-1)/2, upper-triangle 순서
  wm_area  [R] float32                 ROI 별 WM 경계 면적 (mm^2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import CACHE

OUT_DIR = CACHE / "anat_corridor"
CORRIDOR_DIM = 7
N_SAMPLE = 64


def path(sub: str, source: str = "rigid") -> Path:
    return OUT_DIR / f"{sub}_{source}.npz"


def _sample_line(vol: np.ndarray, a_mm: np.ndarray, b_mm: np.ndarray,
                 inv: np.ndarray, n: int) -> np.ndarray:
    """[K,3] -> [K,3] 두 점 사이를 n 등분해 최근접 voxel 값 [K, n]. 밖은 0."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :, None]
    pts = a_mm[:, None, :] * (1 - t) + b_mm[:, None, :] * t                  # [K,n,3] mm
    ijk = np.einsum("ij,knj->kni", inv[:3, :3], pts) + inv[:3, 3]
    ijk = np.round(ijk).astype(np.int32)
    ok = np.ones(ijk.shape[:2], bool)
    for d in range(3):
        ok &= (ijk[..., d] >= 0) & (ijk[..., d] < vol.shape[d])
    out = np.zeros(ijk.shape[:2], np.float32)
    idx = np.where(ok)
    out[idx] = vol[ijk[..., 0][idx], ijk[..., 1][idx], ijk[..., 2][idx]]
    return out


def corridor_features(wm: np.ndarray, t1n: np.ndarray, centroid: np.ndarray,
                      pairs: np.ndarray, affine: np.ndarray, n: int = N_SAMPLE) -> np.ndarray:
    """[P, 7]. centroid 는 mm (WM 가중), affine 은 W 격자의 voxel->mm.

      0 wm_mean     선 위 평균 WM 확률 -- "두 ROI 사이에 백질이 있나"
      1 wm_mid      가운데 절반의 평균 (끝점 GM 을 뺀 값)
      2 wm_min      최솟값 -- 경로가 끊기는 지점
      3 wm_p10      10 백분위 (min 은 voxel 하나에 흔들린다)
      4 wm_frac     WM > 0.5 인 표본 비율
      5 gap_frac    WM < 0.3 이 이어지는 최장 구간의 길이 비율 -- 관통 불가 구간
      6 t1_mean     선 위 평균 T1 (정규화). 조직 대비의 대리
    """
    assert centroid.ndim == 2 and centroid.shape[1] == 3, centroid.shape
    inv = np.linalg.inv(affine)
    i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    w = _sample_line(wm, centroid[i], centroid[j], inv, n)
    t = _sample_line(t1n, centroid[i], centroid[j], inv, n)
    lo, hi = n // 4, n - n // 4
    gap = w < 0.3
    # 최장 연속 gap 길이: 누적합 트릭 (K x n 이라 파이썬 루프 없이)
    run = np.zeros_like(gap, np.int32)
    run[:, 0] = gap[:, 0]
    for k in range(1, n):                       # n=64 라 벡터화된 64 스텝, pair 3321 개는 한 번에
        run[:, k] = np.where(gap[:, k], run[:, k - 1] + 1, 0)
    X = np.stack([w.mean(1), w[:, lo:hi].mean(1), w.min(1), np.percentile(w, 10, axis=1),
                  (w > 0.5).mean(1), run.max(1) / float(n), t.mean(1)], 1).astype(np.float32)
    assert X.shape[1] == CORRIDOR_DIM and np.isfinite(X).all(), X.shape
    return X


def wm_surface_area(wm: np.ndarray, labels: np.ndarray, n_roi: int,
                    level: float = 0.5, step: int = 1) -> np.ndarray:
    """[R] ROI 별 WM 경계 면적 (mm^2). marching cubes 삼각형을 면 중심의 아틀라스 라벨로 모은다."""
    from skimage import measure
    v, f, _, _ = measure.marching_cubes(wm, level=level, step_size=step)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    c = np.round(tri.mean(1)).astype(np.int32)
    for d in range(3):
        np.clip(c[:, d], 0, labels.shape[d] - 1, out=c[:, d])
    lab = labels[c[:, 0], c[:, 1], c[:, 2]]
    out = np.zeros(n_roi, np.float64)
    m = lab > 0
    np.add.at(out, lab[m] - 1, area[m])
    assert np.isfinite(out).all() and out.sum() > 0, "WM 표면적이 전부 0"
    return out.astype(np.float32)


def load(sub: str, source: str = "rigid") -> dict:
    p = path(sub, source)
    assert p.exists(), f"corridor feature 가 없다: {p} (scripts/70_corridor_feats.py 로 생성)"
    z = np.load(p)
    assert z["corridor"].shape[1] == CORRIDOR_DIM, z["corridor"].shape
    assert np.isfinite(z["corridor"]).all(), f"{sub}: corridor 에 NaN/Inf"
    return {k: z[k] for k in z.files}
