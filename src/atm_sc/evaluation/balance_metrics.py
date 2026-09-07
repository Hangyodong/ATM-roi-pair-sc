"""균형 학습 검증 지표 (GESTA 전략 §57–60, §65, §78, §82–83). gradient 없음, 부작용 없음.

- SC: losses.metrics.sc_metrics 에 Spearman · weak-edge recall · strong-edge precision 을 더한다.
  raw count 는 heavy tail 이라 Pearson 은 큰 edge 몇 개가 결정한다 — GT <= 100 인 edge (~79 %) 를 전부 0 으로
  만들어도 r 은 0.001 정도밖에 안 떨어진다 (tests/test_balance_metrics.py). 약한 edge 소실은 recall 로만 보인다 (§58).
- bundle geometry: Tractometer 식 voxel coverage / overreach / dice, 길이 분포, valid·duplicate ratio (§60).
- exposure: 학습 노출이 raw count 비례가 아님을 size-bin / tier / block 별로 보인다 (§65, §78, §83).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import ks_2samp

from ..data.bundle_statistics import SIZE_EDGES, size_bin_of
from ..data.roi_groups import BLOCKS, TIER_EDGES, TIERS
from ..losses.metrics import sc_metrics
from ..losses.sc_corr import upper

__all__ = ["SIZE_EDGES", "SIZE_BINS", "size_bin_of", "spearman", "sc_metrics_extended",
           "bundle_geometry_metrics", "exposure_report"]

SIZE_BINS = ("low", "mid", "high")
STRONG_PCT = 5            # 상위 5 % edge 가 SC 질량의 절반 (losses/sc_corr.py)


# ----------------------------------------------------------------------------- SC
def _avg_rank(x: torch.Tensor) -> torch.Tensor:
    """동률은 평균 순위 (scipy.stats.rankdata(method='average') 와 동일)."""
    x = x.double().flatten()
    order = torch.argsort(x, stable=True)
    s = x[order]
    new = torch.ones_like(s, dtype=torch.bool); new[1:] = s[1:] != s[:-1]
    gid = torch.cumsum(new, 0) - 1                                   # 동률 그룹 id (정렬 순서)
    pos = torch.arange(1, s.numel() + 1, dtype=torch.float64, device=x.device)
    avg = torch.bincount(gid, weights=pos) / torch.bincount(gid)
    ranks = torch.empty_like(s); ranks[order] = avg[gid]
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.ndim == 1 and a.shape == b.shape and a.numel() > 1, (a.shape, b.shape)
    with torch.no_grad():
        return float(torch.corrcoef(torch.stack([_avg_rank(a), _avg_rank(b)]))[0, 1])


def _upper_masked(sc_pred, sc_gt, mask):
    p, g = upper(sc_pred).double(), upper(sc_gt).double()
    if mask is not None:
        mm = upper(torch.as_tensor(mask, device=sc_pred.device)).bool()
        p, g = p[mm], g[mm]
    return p, g


def _top_set(x: torch.Tensor, k: int) -> torch.Tensor:
    """값이 큰 k 개 edge 의 index. 0 은 edge 가 아니므로 뺀다 (0 끼리의 topk 순서는 임의라 precision 이 우연이 된다)."""
    idx = torch.topk(x, k).indices
    return idx[x[idx] > 0]


def sc_metrics_extended(sc_pred: torch.Tensor, sc_gt: torch.Tensor, mask: torch.Tensor | None = None,
                        weak_edge_max: float = TIER_EDGES[0]) -> dict:
    """sc_metrics + spearman
    + weak_edge_recall: GT 0 < g <= weak_edge_max 인 edge (weak_edge_n 개) 중 pred > 0 인 비율
    + strong_edge_precision: pred 상위 5 % (strong_edge_n = ceil(0.05·n_edges) 개, 0 제외) 중 GT 상위 5 % 에도 드는 비율."""
    out = sc_metrics(sc_pred, sc_gt, mask)
    with torch.no_grad():
        p, g = _upper_masked(sc_pred, sc_gt, mask)
        weak = (g > 0) & (g <= weak_edge_max)
        k = max(1, math.ceil(g.numel() * STRONG_PCT / 100))
        tp, tg = _top_set(p, k), _top_set(g, k)
        out.update(spearman=spearman(p, g),
                   weak_edge_recall=float((p[weak] > 0).double().mean()) if weak.any() else float("nan"),
                   weak_edge_n=int(weak.sum()),
                   strong_edge_precision=float(torch.isin(tp, tg).double().mean()) if tp.numel() else float("nan"),
                   strong_edge_n=k)
    return out


# ----------------------------------------------------------------------------- bundle geometry
def _lengths(S: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)


def _duplicate_mask(G: np.ndarray, tol_mm: float = 1.0, n_key: int = 16, key_mm: float = 2.0) -> np.ndarray:
    """앞선 streamline 과 평균 점별 거리 < tol_mm 이면 duplicate. 후보는 16 점 resample 을 key_mm 로 반올림한
    hash 가 같은 것만 비교해 O(n²) 을 피한다 — hash 경계를 넘는 sub-mm jitter 는 놓치므로 하한이고,
    복제 oversampling (§19-A, §83 'duplicate oversampling') 의 정확한 사본은 전부 잡는다."""
    n, P = G.shape[:2]
    dup = np.zeros(n, bool)
    if n < 2:
        return dup
    key = np.round(G[:, np.linspace(0, P - 1, n_key).round().astype(int)] / key_mm).astype(np.int64).reshape(n, -1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")                             # bucket 안에서는 원래 순서 유지
    for members in np.split(order, np.flatnonzero(np.diff(inv[order])) + 1):
        for j in range(1, len(members)):
            d = np.linalg.norm(G[members[:j]] - G[members[j]], axis=-1).mean(1)
            dup[members[j]] = bool((d < tol_mm).any())
    return dup


def bundle_geometry_metrics(S_gen: np.ndarray, S_gt: np.ndarray, voxel_mm: float = 2.0,
                            length_range=(20.0, 220.0)) -> dict:
    """S_gen [n,P,3], S_gt [m,P,3] mm (§60).
    coverage = |vox(gen) ∩ vox(gt)| / |vox(gt)|, overreach = |vox(gen) \\ vox(gt)| / |vox(gt)| (Tractometer OR),
    dice = 2|∩| / (|vox(gen)| + |vox(gt)|); voxel 은 두 bundle 공통 원점(최소 좌표) 기준 floor(mm / voxel_mm).
    length_err_mm = |mean L_gen − mean L_gt|, length_ks = KS 통계량. valid = 유한하고 길이가 length_range 안.
    voxel·길이 지표는 유한한 gen streamline 전부 (길이 필터 없이), duplicate_ratio 분모는 n."""
    S_gen, S_gt = np.asarray(S_gen, np.float64), np.asarray(S_gt, np.float64)
    assert S_gen.ndim == 3 and S_gt.ndim == 3 and S_gen.shape[2] == S_gt.shape[2] == 3, (S_gen.shape, S_gt.shape)
    assert len(S_gen) > 0 and len(S_gt) > 0 and np.isfinite(S_gt).all(), "빈 bundle 또는 비유한 GT"
    fin = np.isfinite(S_gen).all(axis=(1, 2))
    G = S_gen[fin]
    L_gen = np.full(len(S_gen), np.nan); L_gen[fin] = _lengths(G)
    L_gt = _lengths(S_gt)
    valid = fin & (L_gen >= length_range[0]) & (L_gen <= length_range[1])

    pg, pt = G.reshape(-1, 3), S_gt.reshape(-1, 3)
    origin = np.concatenate([pg, pt]).min(0)
    vg, vt = (np.floor((p - origin) / voxel_mm).astype(np.int64) for p in (pg, pt))
    dims = np.concatenate([vg, vt]).max(0) + 1
    lin = lambda v: np.unique((v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2])
    sg, st = lin(vg), lin(vt)
    inter = np.intersect1d(sg, st, assume_unique=True).size
    return {"coverage": inter / st.size, "overreach": (sg.size - inter) / st.size,
            "dice": 2 * inter / (sg.size + st.size),
            "length_err_mm": float(abs(L_gen[fin].mean() - L_gt.mean())) if fin.any() else float("nan"),
            "length_ks": float(ks_2samp(L_gen[fin], L_gt).statistic) if fin.any() else float("nan"),
            "valid_ratio": float(valid.mean()),
            "duplicate_ratio": float(_duplicate_mask(G).sum() / len(S_gen)),
            "n_gen": int(len(S_gen)), "n_gt": int(len(S_gt))}


# ----------------------------------------------------------------------------- exposure
def _group_shares(exposure: np.ndarray, counts: np.ndarray, labels: np.ndarray, names) -> dict:
    return {nm: {"n_pairs": int((labels == i).sum()),
                 "exposure_share": float(exposure[labels == i].sum() / exposure.sum()),
                 "count_share": float(counts[labels == i].sum() / counts.sum())} for i, nm in enumerate(names)}


def exposure_report(pair_counts: np.ndarray, exposure: np.ndarray, pair_block: np.ndarray, pair_tier: np.ndarray) -> dict:
    """pair 별 N_ij · 학습 노출 · block(0/1/2) · tier(0/1/2) -> size-bin / tier / block 별 share (§65 로그).
    노출이 raw count 비례 (§78: 250:25:1) 가 아님을 보이는 요약 (§83):
      max_pair_share        = 노출이 가장 큰 pair 하나의 share
      ratio_high_to_low     = high size-bin 의 pair 당 평균 노출 / low size-bin 의 것 (한쪽이 비면 nan)
      ratio_max_to_min_pair = count 최대 pair 의 노출 / count 최소 pair 의 노출 (§78 의 A/C; 3 pair 가상 데이터에서는
                              C = 20 이 mid bin 이라 ratio_high_to_low 가 정의되지 않으므로 따로 둔다)."""
    c, e = np.asarray(pair_counts, np.float64), np.asarray(exposure, np.float64)
    bl, tr = np.asarray(pair_block), np.asarray(pair_tier)
    assert c.ndim == 1 and c.shape == e.shape == bl.shape == tr.shape and len(c) > 0, (c.shape, e.shape, bl.shape, tr.shape)
    assert (c >= 1).all() and (e >= 0).all() and e.sum() > 0, "count < 1 이거나 노출이 전혀 없음"
    sb = size_bin_of(c)
    per_pair = lambda m: float(e[m].mean()) if m.any() else float("nan")
    return {"size": _group_shares(e, c, sb, SIZE_BINS), "tier": _group_shares(e, c, tr, TIERS),
            "block": _group_shares(e, c, bl, BLOCKS),
            "max_pair_share": float(e.max() / e.sum()),
            "ratio_high_to_low": per_pair(sb == 2) / per_pair(sb == 0),
            "ratio_max_to_min_pair": float(e[c.argmax()] / e[c.argmin()])}
