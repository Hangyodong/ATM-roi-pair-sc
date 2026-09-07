import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

R_SMALL, P = 6, 128          # ROI 6개, ATM 실제 point 수


@pytest.fixture(scope="session")
def synth():
    """작은 합성 atlas + 거리맵. ROI 를 x 축을 따라 6개 블록으로 배치한다."""
    from atm_sc.models.endpoint_assigner import build_distance_maps
    shape = (24, 20, 18)
    atlas = np.zeros(shape, np.int16)
    for r in range(R_SMALL):
        atlas[2 + r * 3: 4 + r * 3, 4:16, 4:14] = r + 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-24.0, -20.0, -18.0]
    dm = build_distance_maps(atlas, R_SMALL, (2.0, 2.0, 2.0))
    return {"atlas": atlas, "affine": affine, "dist": dm, "shape": shape, "n_roi": R_SMALL}


@pytest.fixture(scope="session")
def streamlines(synth):
    """N=12 개의 합성 streamline. ROI a 중심 -> ROI b 중심을 잇는 직선 + 잡음."""
    import torch
    from atm_sc.spaces import voxel_to_mm
    rng = np.random.default_rng(0)
    pairs = [(0, 5), (1, 4), (2, 3), (0, 3), (1, 5), (2, 4)] * 2
    out = []
    for a, b in pairs:
        pa = voxel_to_mm(np.array([3.0 + a * 3, 10.0, 9.0]), synth["affine"])
        pb = voxel_to_mm(np.array([3.0 + b * 3, 10.0, 9.0]), synth["affine"])
        t = np.linspace(0, 1, P)[:, None]
        out.append(pa * (1 - t) + pb * t + rng.normal(0, 0.3, (P, 3)))
    return torch.tensor(np.stack(out), dtype=torch.float32), pairs
