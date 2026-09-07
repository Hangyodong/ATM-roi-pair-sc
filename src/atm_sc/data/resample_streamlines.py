"""가변 길이 streamline -> 등간격 n_out 점 (ATM 입력 형식, 기본 128).

`tt_io.resample_128` 은 streamline 마다 Python 루프를 돈다. 여기서는 ragged 배열 전체를
한 번에 처리하는 벡터화 구현을 두고, 참조 구현(tt_io.resample_128)과 앞 몇 백 개를
대조해 같은 결과인지 확인한다.

보장:
  * 양 끝점은 원래 끝점과 **정확히** 같다 ((1-w)*p0 + w*p1 형태라 w=0,1 에서 비트 단위 동일).
  * 재샘플 점은 전부 원래 polyline 위에 놓인다 (선형 보간). 따라서 재샘플 길이 <= 원래 길이.
"""
from __future__ import annotations

import numpy as np

from .tt_io import resample_128, segment_lengths


def _starts(npts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(npts)[:-1]]).astype(np.int64)


def resample_equidistant(mm: np.ndarray, npts: np.ndarray, n_out: int = 128,
                         verify: bool = True) -> np.ndarray:
    """mm [P,3] (concatenated), npts [n] -> [n, n_out, 3] float32."""
    npts = np.asarray(npts, np.int64)
    assert npts.ndim == 1 and (npts >= 1).all(), "npts 는 1 이상"
    P, n = int(npts.sum()), len(npts)
    assert mm.shape == (P, 3), (mm.shape, P)
    mm64 = np.asarray(mm, np.float64)
    starts = _starts(npts)
    sid = np.repeat(np.arange(n), npts)

    # streamline 내부 누적 호장 (다른 streamline 으로 넘어가는 구간은 0)
    seg = np.linalg.norm(np.diff(mm64, axis=0), axis=1)
    seg = np.where(sid[1:] == sid[:-1], seg, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cum = cum - cum[starts][sid]                        # streamline 시작에서 0
    total = cum[starts + npts - 1]                      # streamline 별 총 길이
    ok = (total > 0) & (npts >= 2)

    # u_local in [0,1] 에 streamline id 를 더해 전역 단조 좌표를 만든다
    u_local = np.where(ok[sid], cum / np.where(total[sid] > 0, total[sid], 1.0), 0.0)
    u_local[starts + npts - 1] = np.where(ok, 1.0, 0.0)  # 마지막 점은 정확히 1
    u_glob = u_local + sid

    t = np.linspace(0.0, 1.0, n_out)
    tg = (np.arange(n)[:, None] + t[None, :]).ravel()   # [n*n_out]
    idx = np.searchsorted(u_glob, tg, side="right") - 1
    lo = np.repeat(starts, n_out)
    hi = np.repeat(starts + np.maximum(npts - 2, 0), n_out)  # idx+1 이 같은 streamline 안에 있도록
    idx = np.clip(idx, lo, hi)
    idx1 = np.minimum(idx + 1, np.repeat(starts + npts - 1, n_out))
    du = u_glob[idx1] - u_glob[idx]
    w = np.where(du > 0, (tg - u_glob[idx]) / np.where(du > 0, du, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    out = (1.0 - w)[:, None] * mm64[idx] + w[:, None] * mm64[idx1]
    out = out.reshape(n, n_out, 3)

    # 퇴화 (점 1개 또는 길이 0): 첫 점으로 채운다 (참조 구현과 동일)
    if (~ok).any():
        out[~ok] = mm64[starts[~ok]][:, None, :]

    out = out.astype(np.float32)
    assert np.isfinite(out).all(), "재샘플 결과에 NaN/Inf"
    # 끝점은 원래 끝점과 정확히 같아야 한다
    assert np.array_equal(out[:, 0], mm[starts].astype(np.float32)), "시작점 불일치"
    assert np.array_equal(out[ok, -1], mm[(starts + npts - 1)[ok]].astype(np.float32)), "끝점 불일치"

    if verify and n > 0:
        m = min(300, n)
        ref = resample_128(mm[: int(npts[:m].sum())], npts[:m], n_out)
        assert np.allclose(out[:m], ref, atol=1e-3), \
            f"벡터화 재샘플이 참조 구현과 불일치 (max {np.abs(out[:m]-ref).max():.4f} mm)"
    return out


def polyline_lengths(s: np.ndarray) -> np.ndarray:
    """[n, T, 3] -> [n] 재샘플 polyline 길이."""
    return np.linalg.norm(np.diff(s.astype(np.float64), axis=1), axis=2).sum(1)


def check_fidelity(mm: np.ndarray, npts: np.ndarray, out: np.ndarray,
                   straight_ratio: float = 1.2, tol: float = 0.02) -> dict:
    """길이 보존 통계. 직선에 가까운(원래 길이/현 길이 < straight_ratio) streamline 은
    재샘플 길이 오차가 tol 이내여야 한다 (assert)."""
    starts = _starts(npts)
    L0 = segment_lengths(mm, npts)
    L1 = polyline_lengths(out)
    chord = np.linalg.norm(mm[starts + npts - 1] - mm[starts], axis=1)
    valid = L0 > 0
    rel = np.zeros(len(npts)); rel[valid] = (L0[valid] - L1[valid]) / L0[valid]
    assert (rel[valid] >= -1e-6).all(), "재샘플 길이가 원래보다 길다 — 보간 오류"
    straight = valid & (chord > 0) & (L0 / np.maximum(chord, 1e-9) < straight_ratio)
    if straight.any():
        worst = float(rel[straight].max())
        assert worst < tol, f"직선형 streamline 길이 오차 {worst:.4f} > {tol}"
    end_err = np.abs(out[:, -1] - mm[starts + npts - 1]).max() if len(npts) else 0.0
    return {"n": int(len(npts)), "rel_mean": float(rel[valid].mean()),
            "rel_p99": float(np.percentile(rel[valid], 99)), "rel_max": float(rel[valid].max()),
            "n_straight": int(straight.sum()),
            "straight_max": float(rel[straight].max()) if straight.any() else 0.0,
            "endpoint_max_err": float(end_err)}
