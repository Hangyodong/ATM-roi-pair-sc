"""pair 별 선형 anatomy -> SC 잔차 head (train 전용 능선회귀).

왜 있나: `count_head` 는 pair 별 tier1 가중치를 갖고도 val resid_r 0.086 인데, 같은 정보를 쓴
단순 능선회귀가 0.147 이다 (`outputs/eval/ridge_ceiling.json`). 신경망이 절대 SC 적합에
끌려가 잔차를 못 배운 것이다. 잔차 목표를 만들 때는 이쪽이 낫다.

입력은 subject T1 에서만 나온다 (tier1 자로 잰 값 + ROI 국소 anatomy PCA). 적합에 쓰는
템플릿/PCA/평균/능선 가중치는 **train split 144명**으로만 만든다 -> 추론에 그룹 정보 없음.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "outputs" / "cache" / "pair_ridge_{source}.npz"


def _pair_list(n_roi: int) -> np.ndarray:
    return np.stack(np.triu_indices(n_roi, 1), 1)


def raw_blocks(sub: str, n_roi: int, source: str = "rigid") -> dict:
    """표준화 전 feature 블록. fit(train 통계 계산)과 추론이 같은 함수를 쓰게 한다.

      tier1     자로 잰 ROI feature 에서 만든 pair feature 9개
      corridor  두 centroid 를 잇는 선 위의 WM 확률 7개 (`data/anat_corridor.py`)
      surf      ROI 별 WM 경계 면적 3개 (marching cubes)
      tissue    GM 부피/GM 비중/CSF 비중/GM-WM 비, 양 ROI -> 8개 (Atropos 3-class)
    """
    from ..data import anat_corridor as AC
    from ..data.anat_tier1 import load as load_tier1, pair_features
    from ..data.paths import CACHE
    from ..data.wm_segment import tissue_path
    pairs = _pair_list(n_roi)
    out = {"tier1": pair_features(load_tier1(sub, source), pairs).astype(np.float64)}
    c = AC.load(sub, source)
    out["corridor"] = np.asarray(c["corridor"], np.float64)
    la = np.log1p(np.asarray(c["wm_area"], np.float64))
    out["surf"] = np.stack([la[pairs[:, 0]], la[pairs[:, 1]],
                            0.5 * (la[pairs[:, 0]] + la[pairs[:, 1]])], -1)
    tp = tissue_path(CACHE, sub, source)
    if tp.exists():
        from ..models.roi_pool import atlas_on_feature_grid
        from ..spaces import W_AFFINE, W_SHAPE
        global _LAB, _NVOX
        if "_LAB" not in globals() or _LAB is None:
            lab, _ = atlas_on_feature_grid(feat_shape=W_SHAPE, in_shape=W_SHAPE,
                                           in_affine=W_AFFINE, return_affine=True)
            _LAB = np.asarray(lab, np.int64).ravel()
            _NVOX = np.maximum(np.bincount(_LAB, minlength=n_roi + 1)[1:], 1.0)
        z = np.load(tp)
        g, cs, wv = (z[k].astype(np.float32).ravel() / 255.0 for k in ("gm", "csf", "wm"))
        agg = lambda v: np.bincount(_LAB, weights=v, minlength=n_roi + 1)[1:]
        gs, css, ws = agg(g), agg(cs), agg(wv)
        tr = np.stack([np.log1p(gs), gs / _NVOX, css / _NVOX,
                       np.log((gs + 1.0) / (ws + 1.0))], 1)
        out["tissue"] = np.concatenate([tr[pairs[:, 0]], tr[pairs[:, 1]]], -1)
    return out


_LAB = None
_NVOX = None


def features(sub: str, w: dict, source: str = "rigid") -> np.ndarray:
    """[P, F] pair feature (절편 포함). w 는 fit 이 만든 train 전용 통계."""
    from ..data.local_feats import roi_feature_path
    n_roi = int(w["n_roi"]); pairs = _pair_list(n_roi)
    blocks = raw_blocks(sub, n_roi, source)
    names = [b for b in w["blocks"].tolist()]
    parts = []
    for b in names:
        assert b in blocks, f"{sub}: 적합에 쓴 블록 '{b}' 이 없다"
        parts.append((blocks[b] - w[f"mu_{b}"]) / w[f"sd_{b}"])
    f = np.asarray(np.load(roi_feature_path(sub, source))["f_roi"], np.float64)
    assert f.shape[0] == n_roi, (sub, f.shape)
    z = np.einsum("rd,rdq->rq", f - w["roi_mean"], w["comp"])
    X = np.concatenate(parts + [z[pairs[:, 0]], z[pairs[:, 1]], np.ones((len(pairs), 1))], -1)
    assert np.isfinite(X).all(), f"{sub}: pair ridge feature 에 NaN/Inf"
    assert X.shape[1] == w["W"].shape[1], (X.shape, w["W"].shape)
    # train 분포 밖으로 크게 벗어나면 예측이 폭발한다 (표준화 하한이 없던 시절 val 최대 36789).
    mx = float(np.abs(X).max())
    assert mx < 100.0, f"{sub}: 표준화 feature 가 train 분포 밖이다 (최대 {mx:.1f})"
    return X


def load(source: str = "rigid") -> dict:
    p = Path(str(WEIGHTS).format(source=source))
    assert p.exists(), f"pair ridge 가중치가 없다: {p} (scripts/67_fit_pair_ridge.py 로 생성)"
    z = np.load(p, allow_pickle=False)
    w = {k: z[k] for k in z.files}
    assert int(w["n_train"]) == 144, f"train 144명으로 적합되지 않았다 ({int(w['n_train'])})"
    return w


def resid_log(sub: str, w: dict, source: str = "rigid") -> np.ndarray:
    """[P] log1p 공간의 subject 잔차 예측 (템플릿 대비)."""
    r = np.einsum("pf,pf->p", features(sub, w, source), w["W"])
    assert np.isfinite(r).all(), f"{sub}: ridge 잔차에 NaN/Inf"
    return r


def predict_sc(sub: str, w: dict, source: str = "rigid", alpha: float = 1.0) -> np.ndarray:
    """[P] 선형 SC 예측. alpha 는 잔차 진폭 배율.

    증폭은 **선형 공간**에서 한다:  pred = 템플릿 + alpha * (expm1(템플릿+r) - 템플릿).
    로그 공간에서 alpha*r 을 넣고 expm1 하면 지수가 큰 edge 를 폭발시켜 정확도가 무너진다
    (실측 val: 능선 alpha 8 에서 resid_r 0.147 -> 0.018). 선형 증폭은 LOO 중심화 후 잔차의
    **배율**일 뿐이라 resid_r 이 (거의) 불변이고 subject 간 상관만 내려간다:
        alpha  1 -> resid_r 0.147 inter 0.999
        alpha 12 -> resid_r 0.145 inter 0.875   (GT inter 0.904)
    즉 "얼마나 정확한가" 와 "얼마나 서로 다른가" 가 분리된 손잡이가 된다.
    """
    base = np.expm1(w["template"])
    d = np.expm1(w["template"] + resid_log(sub, w, source)) - base
    return np.maximum(base + float(alpha) * d, 0.0)
