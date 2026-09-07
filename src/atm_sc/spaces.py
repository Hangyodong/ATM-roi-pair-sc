"""좌표계 정의와 변환.

이 프로젝트에는 네 개의 공간이 있다. 전부 mm(MNI RAS) 를 경유해서만 변환한다.

  1. TT     : DSI Studio QSDR tractogram 격자. (80,100,80) @2mm, 1/32 voxel 정밀도 정수 저장.
              ``trans_to_mni`` 로 mm 로 간다. 전 subject 동일 (실측 확인).
  2. ATLAS  : DesikanCortexPD25 MNI152NLin6 2mm. (91,109,91). ROI 82개.
  3. W      : ATM 작업 격자. (193,229,193) @1mm, origin (-96,-132,-78).
              rigid_UNet 의 Upsample size 가 이 값들로 하드코딩되어 있어 변경 불가
              (external/atm_upstream/stable/model/model.py:170,175,180).
  4. NATIVE : subject T1 원본 공간. 여기서는 W 로 보내는 출발점으로만 쓴다.

TT 와 ATLAS 가 같은 MNI 를 쓴다는 것은 추측이 아니라 실측이다: TT 를 ATLAS 로
투영해 만든 pass-SC 가 GT SC 와 r=0.9986 (edge F1 0.982) 로 일치한다.
scripts/01_verify_gt_sc.py 가 이 검증을 재실행한다.
"""
from __future__ import annotations

import numpy as np

# ATM 작업 격자 W --------------------------------------------------------------
W_SHAPE = (193, 229, 193)
W_AFFINE = np.array([[1.0, 0.0, 0.0, -96.0],
                     [0.0, 1.0, 0.0, -132.0],
                     [0.0, 0.0, 1.0, -78.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# 해부학 정합 대상. GT tractogram(QSDR) 과 atlas 가 사는 공간.
TEMPLATE_KEY = "MNI152NLin6Asym"


def apply_affine(aff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts [..., 3] 에 4x4 affine 적용."""
    assert aff.shape == (4, 4), aff.shape
    assert pts.shape[-1] == 3, pts.shape
    return pts @ aff[:3, :3].T + aff[:3, 3]


def mm_to_voxel(mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """mm -> 연속 voxel index (반올림하지 않음)."""
    return apply_affine(np.linalg.inv(affine), mm)


def voxel_to_mm(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return apply_affine(affine, vox)


def mm_to_grid(mm, affine: np.ndarray, shape) -> np.ndarray:
    """mm -> torch.nn.functional.grid_sample(align_corners=True) 용 정규화 좌표.

    grid_sample 은 마지막 축부터 (x, y, z) 순으로 읽는다. 즉 입력 볼륨이
    [N, C, D0, D1, D2] 일 때 grid[..., 0] 은 D2, [..., 1] 은 D1, [..., 2] 는 D0 이다.
    voxel index (i, j, k) -> ( 2k/(K-1)-1, 2j/(J-1)-1, 2i/(I-1)-1 ) 로 뒤집어야 한다.
    이 축 순서를 틀리면 좌우/전후가 조용히 바뀐다.

    numpy ndarray 또는 torch.Tensor 를 그대로 받는다.
    """
    size = np.asarray(shape, dtype=np.float64)
    assert size.shape == (3,), shape
    inv = np.linalg.inv(affine)
    if hasattr(mm, "detach"):                      # torch (numpy 2.x 도 .device 를 갖는다)
        import torch
        A = torch.as_tensor(inv[:3, :3].T, dtype=mm.dtype, device=mm.device)
        b = torch.as_tensor(inv[:3, 3], dtype=mm.dtype, device=mm.device)
        s = torch.as_tensor(size, dtype=mm.dtype, device=mm.device)
        vox = mm @ A + b
        g = 2.0 * vox / (s - 1.0) - 1.0
        return torch.flip(g, dims=[-1])
    vox = apply_affine(inv, np.asarray(mm, dtype=np.float64))
    return (2.0 * vox / (size - 1.0) - 1.0)[..., ::-1]


def _selftest() -> None:
    """축 순서와 왕복 변환을 확인한다. import 시 자동 실행."""
    shape = (5, 7, 9)
    aff = np.diag([2.0, 3.0, 4.0, 1.0]); aff[:3, 3] = [-10.0, 5.0, -2.0]
    vox = np.array([[0., 0., 0.], [4., 6., 8.], [2., 3., 4.]])
    mm = voxel_to_mm(vox, aff)
    assert np.allclose(mm_to_voxel(mm, aff), vox), "mm<->voxel 왕복 실패"
    g = mm_to_grid(mm, aff, shape)
    assert np.allclose(g[0], [-1, -1, -1]), g[0]          # voxel (0,0,0) -> 전부 -1
    assert np.allclose(g[1], [+1, +1, +1]), g[1]          # 반대 코너 -> 전부 +1
    # 축이 뒤집혔는지: i 만 움직였을 때 마지막 성분만 변해야 한다
    g2 = mm_to_grid(voxel_to_mm(np.array([[1., 0., 0.]]), aff), aff, shape)
    assert g2[0, 2] > -1 + 1e-9 and abs(g2[0, 0] + 1) < 1e-9, g2
    assert W_SHAPE == (193, 229, 193)


_selftest()
