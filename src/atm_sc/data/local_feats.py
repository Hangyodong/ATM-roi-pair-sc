"""ROI 국소 anatomy feature — global average pooling 이 지운 개인차를 되살리는 입력.

실측(175명, `outputs/cache/s1b_feats/`):

| feature | subject 간 코사인 | subject 성분 비중 |
|---|---|---|
| `a512` global avg pool (지금까지 head 가 받던 것) | 0.9994 | **2.1%** |
| `g3` global stage3 | 0.9945 | 3.3% |
| `f_roi` ROI 국소 pooling | 0.8917 | **13.4%** |

GT SC 카운트의 subject 성분이 18.5% 이므로 국소 feature 는 같은 자릿수이고 전역 벡터는
한 자릿수 부족하다. 전략 문서 §3.2-3.3 의 endpoint ROI pooling 이 이것이다.

캐시는 `scripts/42_space_probe.py` (S1-b) 가 만들어 둔 것을 그대로 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FEAT_DIR = ROOT / "outputs" / "cache" / "s1b_feats"


def roi_feature_path(sub: str, source: str = "rigid") -> Path:
    return FEAT_DIR / f"{sub}_{source}.npz"


def load_roi_feats(sub: str, source: str = "rigid", device="cuda",
                   n_roi: int | None = None) -> torch.Tensor:
    """[R, D] ROI 별 국소 anatomy. 없으면 조용히 넘어가지 않고 죽는다."""
    p = roi_feature_path(sub, source)
    assert p.exists(), f"ROI 국소 feature 가 없다: {p} (scripts/42_space_probe.py 로 생성)"
    z = np.load(p)
    assert "f_roi" in z.files, f"{p} 에 f_roi 가 없다: {z.files}"
    f = np.asarray(z["f_roi"], np.float32)
    assert f.ndim == 2, (sub, f.shape)
    if n_roi is not None:
        assert f.shape[0] == n_roi, (sub, f.shape, n_roi)
    assert np.isfinite(f).all(), f"{sub}: f_roi 에 NaN/Inf"
    assert float(np.abs(f).max()) > 0, f"{sub}: f_roi 가 전부 0"
    return torch.from_numpy(f).to(device)


def pair_local(f_roi: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """[R,D] + [K,2] -> [K, 2D]. pair 는 canonical (i<j) 이라 concat 순서가 일관된다."""
    assert f_roi.ndim == 2 and pairs.ndim == 2 and pairs.shape[1] == 2, (f_roi.shape, pairs.shape)
    i, j = pairs[:, 0].long(), pairs[:, 1].long()
    assert int(i.min()) >= 0 and int(torch.maximum(i, j).max()) < f_roi.shape[0], "ROI 인덱스 범위 밖"
    return torch.cat([f_roi[i], f_roi[j]], dim=-1)


def local_dim(source: str = "rigid", n_roi: int | None = None) -> int:
    """캐시 하나를 열어 2*D 를 알아낸다 (config 에 차원을 손으로 적지 않게)."""
    ps = sorted(FEAT_DIR.glob(f"*_{source}.npz"))
    assert ps, f"{FEAT_DIR} 에 {source} 캐시가 없다"
    d = int(np.load(ps[0])["f_roi"].shape[1])
    return 2 * d
