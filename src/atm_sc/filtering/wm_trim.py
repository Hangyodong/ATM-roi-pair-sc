"""백질 경계 기준 trimming + subject 뇌 마스크 filtering (상류 ATM 의 Filtered/Trimmed bundle).

왜: 생성 가닥이 ROI 를 평균 6.4개 지나는데 GT 는 4.4개다 (`outputs/eval/alloc_to_sc_val.json`).
가닥 하나가 pair 20개에 count 를 더하는 바람에 배분에 실린 개인차가 pass SC 로 가면서 전달률
0.27 로 떨어진다. 백질 밖으로 새어 나간 구간을 잘라내면 그 희석이 줄어든다.

상류 ATM 은 템플릿 GM/WM 마스크와 FreeSurfer white surface 를 쓴다. 여기서는 subject 자신의
Atropos 3-class 산출물을 쓴다 (`{sub}_tissue_W.npz`: wm/gm/csf 확률 + brain 마스크).
따라서 후처리 자체가 개인화 통로가 된다.

  filtering  뇌 마스크 밖 비율이 크거나 길이가 비정상인 가닥을 버린다
  trimming   백질 확률 > thr 인 **최장 연속 구간**으로 자르고, 끝점이 피질에 닿도록 margin 만큼 편다
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrimConfig:
    wm_thr: float = 0.3          # 백질로 볼 확률 하한. GT 가닥의 wm_occupancy 5 백분위가 0.031 이라
                                 # 0.5 는 GT 를 41 % 지운다 -- 보정값은 scripts/73 이 정한다
    gm_margin_pts: int = 6       # 백질 구간 양끝에서 피질 쪽으로 더 펴는 점 수 (끝점을 ROI 에 닿게)
    min_pts: int = 16            # 자른 뒤 이보다 짧으면 버린다
    min_length_mm: float = 20.0  # GT 통과율 0.988
    max_outside_brain: float = 0.05   # 뇌 마스크 밖 점 비율 상한 (GT 0.02 기준 통과율 0.878)


def sample_at(S: np.ndarray, vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """[N,T,3] mm -> [N,T] 최근접 voxel 값. 격자 밖은 0."""
    inv = np.linalg.inv(affine)
    ijk = np.round(np.einsum("ij,ntj->nti", inv[:3, :3], S) + inv[:3, 3]).astype(np.int32)
    ok = np.ones(ijk.shape[:2], bool)
    for d in range(3):
        ok &= (ijk[..., d] >= 0) & (ijk[..., d] < vol.shape[d])
    out = np.zeros(ijk.shape[:2], np.float32)
    idx = np.where(ok)
    out[idx] = np.asarray(vol, np.float32)[ijk[..., 0][idx], ijk[..., 1][idx], ijk[..., 2][idx]]
    return out


def longest_run(mask: np.ndarray):
    """[N,T] bool -> (start [N], end [N] 포함, length [N]). True 가 없으면 length 0."""
    N, T = mask.shape
    run = np.zeros((N, T), np.int32)
    run[:, 0] = mask[:, 0]
    for t in range(1, T):
        run[:, t] = np.where(mask[:, t], run[:, t - 1] + 1, 0)
    end = run.argmax(1)
    length = run[np.arange(N), end]
    start = end - length + 1
    return start, end, length


def trim_and_filter(S: np.ndarray, wm: np.ndarray, brain: np.ndarray, affine: np.ndarray,
                    cfg: TrimConfig | None = None):
    """[N,T,3] mm -> (points [M,3] 평탄화, npts [K], keep [N] bool).

    hard_sc(points, npts, ...) 에 그대로 넣을 수 있는 형태로 돌려준다.
    """
    cfg = cfg or TrimConfig()
    S = np.asarray(S, np.float64)
    assert S.ndim == 3 and S.shape[2] == 3, S.shape
    N, T = S.shape[:2]
    w = sample_at(S, wm, affine)
    b = sample_at(S, brain.astype(np.float32), affine) > 0.5
    outside = 1.0 - b.mean(1)
    inside = (w > cfg.wm_thr) & b
    st, en, ln = longest_run(inside)
    st = np.maximum(st - cfg.gm_margin_pts, 0)
    en = np.minimum(en + cfg.gm_margin_pts, T - 1)
    npts_all = en - st + 1
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)              # [N, T-1]
    idx = np.arange(T - 1)[None]
    in_seg = (idx >= st[:, None]) & (idx < en[:, None])
    length = (seg * in_seg).sum(1)
    keep = (ln > 0) & (npts_all >= cfg.min_pts) & (length >= cfg.min_length_mm) \
        & (outside <= cfg.max_outside_brain)
    assert keep.shape == (N,)
    if not keep.any():
        return np.zeros((0, 3)), np.zeros(0, np.int64), keep
    ks, ke = st[keep], en[keep]
    cols = np.arange(T)[None]
    sel = (cols >= ks[:, None]) & (cols <= ke[:, None])
    pts = S[keep][sel]
    npts = (ke - ks + 1).astype(np.int64)
    assert len(pts) == int(npts.sum()), (len(pts), int(npts.sum()))
    return pts, npts, keep
