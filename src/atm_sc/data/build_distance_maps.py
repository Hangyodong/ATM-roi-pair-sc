"""ROI 거리맵 [R, X, Y, Z] 생성/캐시. models.endpoint_assigner.build_distance_maps 의 얇은 래퍼."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

from ..models.endpoint_assigner import build_distance_maps
from .paths import ATLAS, CACHE

N_ROI = 82


def load_atlas(path: Path = ATLAS):
    img = nib.load(path)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    labels = np.unique(atlas)
    assert atlas.max() == N_ROI, f"ROI 최대 라벨 {atlas.max()} != {N_ROI}"
    assert len(labels) - 1 == N_ROI, f"라벨 {len(labels)-1} 개 — ROI 소실"
    return img, atlas


def get_distance_maps(force: bool = False, out: Path = CACHE / "dist_maps.npy") -> np.ndarray:
    img, atlas = load_atlas()
    if out.exists() and not force:
        dm = np.load(out)
        assert dm.shape == (N_ROI,) + atlas.shape, f"캐시 shape {dm.shape} 이 atlas 와 다름"
        assert np.isfinite(dm).all() and dm.max() > 0
        return dm
    dm = build_distance_maps(atlas, N_ROI, img.header.get_zooms()[:3])
    assert dm.shape == (N_ROI,) + atlas.shape
    for r in range(N_ROI):                        # ROI 안은 0, 밖은 양수
        assert dm[r][atlas == r + 1].max() == 0.0
        assert dm[r][atlas != r + 1].min() > 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, dm)
    return dm
