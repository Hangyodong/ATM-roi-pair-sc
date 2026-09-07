"""ROI 그룹(피질/피질하)과 SC edge 강도 구간(소/중/대).

DesikanCortexPD25 82 ROI: label 1~66 피질(Desikan), 67~82 피질하(PD25) — .mat `region_names` 순서:
red nucleus, substantia nigra, subthalamic nucleus, caudate, putamen, GPe, GPi, thalamus (L/R 쌍).
block 별 pair 수: ctx-ctx 2145 · ctx-sub 1056 · sub-sub 120 (합 3321).
"""
from __future__ import annotations

import numpy as np

N_CTX = 66
BLOCKS = ("ctx-ctx", "ctx-sub", "sub-sub")
TIERS = ("small", "mid", "large")
# GT pass-SC(.mat, 206명) nonzero edge 의 33/67 백분위 = 92 / 1282 streamline 을 반올림한 고정 경계.
# subject 별 3분위가 아니라 고정값이어야 '균형' 이 의미가 있다 (3분위면 균등 샘플링과 같다).
TIER_EDGES = (100.0, 1000.0)


def ctx_mask(n_roi: int, n_ctx: int = N_CTX) -> np.ndarray:
    assert 0 < n_ctx < n_roi, (n_ctx, n_roi)
    return np.arange(n_roi) < n_ctx


def block_masks(n_roi: int, n_ctx: int = N_CTX) -> dict[str, np.ndarray]:
    """대칭 [R,R] bool, 대각 False. 세 block 이 upper triangle 을 정확히 분할한다."""
    c = ctx_mask(n_roi, n_ctx)
    cc, ss = np.outer(c, c), np.outer(~c, ~c)
    off = ~np.eye(n_roi, dtype=bool)
    m = {"ctx-ctx": cc & off, "ctx-sub": ~cc & ~ss & off, "sub-sub": ss & off}
    iu = np.triu_indices(n_roi, 1)
    assert sum(int(v[iu].sum()) for v in m.values()) == len(iu[0])
    return m


def block_of_pairs(pairs: np.ndarray, n_ctx: int = N_CTX) -> np.ndarray:
    """[K,2] -> 0 ctx-ctx, 1 ctx-sub, 2 sub-sub."""
    p = np.asarray(pairs)
    return (p[:, 0] >= n_ctx).astype(np.int64) + (p[:, 1] >= n_ctx).astype(np.int64)


def tier_of_strength(strength: np.ndarray, edges=TIER_EDGES) -> np.ndarray:
    """edge 강도(streamline 수) -> 0 small (<= edges[0]), 1 mid, 2 large (> edges[1])."""
    s = np.asarray(strength, np.float64)
    return (s > edges[0]).astype(np.int64) + (s > edges[1]).astype(np.int64)


def tier_masks(sc_gt: np.ndarray, edges=TIER_EDGES) -> dict[str, np.ndarray]:
    """GT 값 기준 [R,R] bool. 0 인 edge(음성) 는 어느 구간에도 없다."""
    g = np.asarray(sc_gt, np.float64)
    off = ~np.eye(g.shape[0], dtype=bool)
    t = tier_of_strength(g, edges)
    return {name: (t == i) & (g > 0) & off for i, name in enumerate(TIERS)}


def balanced_choice(groups: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """존재하는 그룹마다 n 을 균등 분배해 인덱스를 뽑는다 (그룹 안은 균등, 부족하면 복원 추출)."""
    groups = np.asarray(groups)
    present = np.unique(groups)
    quota = np.full(len(present), n // len(present))
    quota[: n % len(present)] += 1
    out = []
    for gi, q in zip(present, quota):
        idx = np.flatnonzero(groups == gi)
        out.append(rng.choice(idx, size=int(q), replace=int(q) > len(idx)))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out
