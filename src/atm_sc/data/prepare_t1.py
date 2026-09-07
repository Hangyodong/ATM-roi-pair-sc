"""subject native T1 -> ATM 작업 격자 W.

ATM 원본은 T1 을 MNI(ICBM152 2009c) 로 **rigid** 정합한다 (infer.py:45, `-t r`;
docstring 은 affine 이라고 하지만 실제 플래그는 rigid 다). 그런데 우리 GT tractogram
은 DSI Studio QSDR 로 **비선형 정규화된** 템플릿 공간에 있다. 따라서 rigid 로만 맞추면
입력 해부와 출력 streamline 공간이 서로 어긋난다.

QSDR 의 역변환은 .fib.gz 안에 있는데 이 서버에서 접근할 수 없으므로, T1 을 GT 쪽
공간(MNI152NLin6Asym)으로 가져오는 방향을 기본값으로 한다.

  mode='syn'   (기본) 비선형. 입력 해부와 GT streamline 공간이 일치한다.
  mode='rigid'        ATM 원본과 같은 정합. 비교/ablation 용.

정합 대상이 NLin6 인 근거는 추측이 아니다: NLin6 좌표의 DK-PD25 atlas 로 GT
tractogram 을 투영해 만든 pass-SC 가 .mat 의 GT SC 와 r=0.9986 으로 일치한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

from ..spaces import W_AFFINE, W_SHAPE, apply_affine

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "templates" / "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz"
TEMPLATE_MASK = ROOT / "templates" / "tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"


def _nib_to_ants(img: nib.Nifti1Image):
    """nibabel -> ANTsImage. LPS/RAS 규약 차이를 direction 행렬로 넘긴다."""
    import ants
    a = np.asanyarray(img.dataobj).astype(np.float32)
    aff = img.affine
    spacing = tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))
    direction = np.array([aff[:3, i] / spacing[i] for i in range(3)]).T
    lps = np.diag([-1.0, -1.0, 1.0])                 # ANTs 는 LPS, nibabel 은 RAS
    return ants.from_numpy(a, origin=tuple(lps @ aff[:3, 3]),
                           spacing=spacing, direction=lps @ direction)


def resample_to_W(img: nib.Nifti1Image, order: int = 1) -> np.ndarray:
    """임의 MNI 격자의 볼륨을 W (193,229,193 @1mm) 로 재샘플. mm 로만 연결한다."""
    from scipy.ndimage import map_coordinates
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in W_SHAPE], indexing="ij")
    mm = apply_affine(W_AFFINE, np.stack([ii, jj, kk], -1).reshape(-1, 3).astype(np.float64))
    vox = apply_affine(np.linalg.inv(img.affine), mm)
    out = map_coordinates(np.asanyarray(img.dataobj).astype(np.float32),
                          vox.T, order=order, mode="constant", cval=0.0)
    return out.reshape(W_SHAPE).astype(np.float32)


def register_to_template(t1_path, mode: str = "syn", out_dir=None, cache: bool = True):
    """native T1 -> 템플릿 공간 nibabel 이미지. 변환은 out_dir 에 캐시한다."""
    import ants
    assert mode in ("syn", "rigid"), mode
    t1_path = Path(t1_path)
    out_dir = Path(out_dir) if out_dir else t1_path.parent
    warped_path = out_dir / f"{t1_path.name.split('.')[0]}__to{mode}.nii.gz"
    if cache and warped_path.exists():
        return nib.load(warped_path)

    fixed = ants.image_read(str(TEMPLATE))
    moving = ants.image_read(str(t1_path))
    tx = "SyN" if mode == "syn" else "Rigid"
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=tx)
    out_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(reg["warpedmovout"], str(warped_path))
    return nib.load(warped_path)


def check_alignment(t1_W: np.ndarray, tract_mm: np.ndarray, thresh_frac: float = 0.90):
    """warp 된 T1 의 뇌 안에 그 subject 의 GT streamline 이 실제로 들어가는지 확인.

    이 검사가 이 파이프라인에서 가장 조용히 실패하기 쉬운 지점이다. 정합이 통째로
    실패해도 뒤 단계는 그럴듯한 숫자를 계속 만들어 낸다.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes
    b = t1_W > (0.10 * float(t1_W.max()))
    b = binary_fill_holes(binary_closing(b, np.ones((5, 5, 5))))
    inv = np.linalg.inv(W_AFFINE)
    ijk = np.rint(apply_affine(inv, tract_mm.astype(np.float64))).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(W_SHAPE)), axis=1)
    inside = np.zeros(len(ijk), bool)
    inside[ok] = b[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    frac = float(inside.mean())
    assert frac >= thresh_frac, (
        f"streamline 의 {frac:.3f} 만 warp 된 T1 뇌 안에 있음 (< {thresh_frac}). "
        "정합 실패 또는 좌표계 오류.")
    return frac


def prepare_subject(t1_path, mode: str = "syn", out_dir=None) -> np.ndarray:
    """native T1 -> W 격자 float32 [193,229,193]. 정규화는 하지 않는다
    (bundle 마다 상수가 다르므로 BundleNorm.normalize_t1 에서 처리)."""
    img = register_to_template(t1_path, mode=mode, out_dir=out_dir)
    vol = resample_to_W(img)
    assert vol.shape == W_SHAPE, vol.shape
    assert np.isfinite(vol).all(), "재샘플 결과에 NaN/Inf"
    assert vol.max() > 0, "재샘플 결과가 전부 0 — 정합 실패"
    return vol
