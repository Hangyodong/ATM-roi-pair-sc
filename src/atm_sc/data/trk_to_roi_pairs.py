"""Whole-brain tractogram -> endpoint ROI 할당 -> canonical ROI pair (pipeline §5).

GT SC 를 만들 때 쓴 것과 같은 atlas / nearest-voxel 규칙을 쓴다 (tt_io.point_labels).
endpoint 규칙과 pass 규칙을 둘 다 계산해 저장한다:
  sc_end  : 학습 target (endpoint 모드). 이 파일의 pair 와 정확히 같은 규칙.
  sc_pass : .mat 의 GT SC 와 같은 정의 (r≈0.9986). 평가 참조.
둘을 섞으면 안 된다 (docs/ROI_PAIR_DATA_FORMAT.md).
"""
from __future__ import annotations

import numpy as np

from .tt_io import hard_sc, load_streamlines, point_labels, segment_lengths

N_ROI = 82


def canonical_pairs(start_roi: np.ndarray, end_roi: np.ndarray) -> np.ndarray:
    """0-based ROI (-1=배경) -> [N,2] int16 canonical (a<b). 무효는 (-1,-1)."""
    a = np.minimum(start_roi, end_roi).astype(np.int16)
    b = np.maximum(start_roi, end_roi).astype(np.int16)
    bad = (start_roi < 0) | (end_roi < 0) | (start_roi == end_roi)
    pair = np.stack([a, b], 1)
    pair[bad] = -1
    return pair


def pair_to_index(pair: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """[N,2] canonical -> [N] a*R+b (무효는 -1)."""
    idx = pair[:, 0].astype(np.int64) * n_roi + pair[:, 1].astype(np.int64)
    idx[(pair < 0).any(1)] = -1
    return idx


def assign_roi_pairs(tt_path, atlas: np.ndarray, affine: np.ndarray,
                     n_roi: int = N_ROI, chunk: int = 200_000, verbose: bool = True) -> dict:
    """chunk 단위로 .tt.gz 를 읽어 streamline 별 endpoint 라벨과 두 종류의 SC 를 만든다."""
    hdr, gen = load_streamlines(str(tt_path), chunk=chunk)
    start_roi, end_roi, length = [], [], []
    sc_end = np.zeros((n_roi, n_roi), np.int64)
    ls_end = np.zeros((n_roi, n_roi), np.float64)
    sc_pass = np.zeros((n_roi, n_roi), np.int64)
    ls_pass = np.zeros((n_roi, n_roi), np.float64)
    n = 0
    for mm, npts in gen():
        starts = np.concatenate([[0], np.cumsum(npts)[:-1]])
        ends = starts + npts - 1
        lab = point_labels(mm[np.concatenate([starts, ends])], atlas, affine).astype(np.int16)
        s, e = lab[: len(npts)] - 1, lab[len(npts):] - 1          # 0-based, 배경 -1
        L = segment_lengths(mm, npts).astype(np.float32)
        pair = canonical_pairs(s, e)
        ok = pair[:, 0] >= 0
        np.add.at(sc_end, (pair[ok, 0], pair[ok, 1]), 1)
        np.add.at(ls_end, (pair[ok, 0], pair[ok, 1]), L[ok])
        w, sl = hard_sc(mm, npts, atlas, affine, n_roi, "pass")
        sc_pass += w; ls_pass += sl
        start_roi.append(s); end_roi.append(e); length.append(L)
        n += len(npts)
        if verbose:
            print(f"  {n:>9,} streamlines", flush=True)
    assert n > 0, "streamline 0개"
    sc_end = sc_end + sc_end.T                     # canonical 은 상삼각에만 쌓였다
    ls_end = ls_end + ls_end.T
    len_end = np.divide(ls_end, sc_end, out=np.zeros_like(ls_end), where=sc_end > 0)
    len_pass = np.divide(ls_pass, sc_pass, out=np.zeros_like(ls_pass), where=sc_pass > 0)
    start_roi = np.concatenate(start_roi); end_roi = np.concatenate(end_roi)
    length = np.concatenate(length)
    pair = canonical_pairs(start_roi, end_roi)
    n_assigned = int((pair[:, 0] >= 0).sum())

    assert len(start_roi) == n and np.isfinite(length).all() and (length > 0).all()
    assert sc_end.sum() > 0, "endpoint SC 가 전부 0 — 좌표계/atlas 확인"
    assert np.array_equal(sc_end, sc_end.T) and sc_end.diagonal().sum() == 0
    assert np.array_equal(sc_pass, sc_pass.T) and sc_pass.diagonal().sum() == 0
    assert sc_end.sum() // 2 == n_assigned, (sc_end.sum(), n_assigned)
    return dict(start_roi=start_roi, end_roi=end_roi, pair=pair, length_mm=length,
                sc_end=sc_end, len_end=len_end.astype(np.float32),
                sc_pass=sc_pass, len_pass=len_pass.astype(np.float32),
                n_total=n, n_assigned=n_assigned, n_roi=n_roi)


def select_capped(pair_index: np.ndarray, cap: int, seed: int):
    """pair 별 최대 cap 개를 seed 로 결정적으로 고른다. (선택된 src_index 오름차순) 을 돌려준다."""
    valid = np.where(pair_index >= 0)[0]
    key = np.random.default_rng(seed).random(len(pair_index))
    order = np.lexsort((key[valid], pair_index[valid]))     # pair, 그 안에서 랜덤 key
    v = valid[order]
    pi = pair_index[v]
    first = np.concatenate([[0], np.where(np.diff(pi) != 0)[0] + 1])
    rank = np.arange(len(v)) - np.repeat(first, np.diff(np.concatenate([first, [len(v)]])))
    return np.sort(v[rank < cap])
