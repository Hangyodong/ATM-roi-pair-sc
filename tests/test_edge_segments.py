"""SC edge-aligned segment 분해 (docs/ATM_SC_EDGE_ALIGNED_BUNDLE_DUAL_REPRESENTATION_STRATEGY.md §4-7, §30-35)."""
import numpy as np

from atm_sc.data.edge_segments import (decompose_bundle, dwell_intervals, segment_sc, streamline_segments)


def _line(pa, pb, n=128):
    t = np.linspace(0, 1, n)[:, None]
    return pa * (1 - t) + pb * t


def test_dwell_intervals_ignores_boundary_jitter():
    lab = np.array([0, 0, 1, 1, 1, 2, 1, 3, 3, 3, 0])     # 중간의 단독 1 은 경계 흔들림
    iv = dwell_intervals(lab, min_dwell=2)
    assert [(r, i, j) for r, i, j in iv] == [(0, 2, 4), (2, 7, 9)]
    assert len(dwell_intervals(lab, min_dwell=1)) == 4     # min_dwell=1 이면 스친 것도 방문


def test_all_visited_pairs_not_only_adjacent():
    """GT 가 Case B(모든 ROI 쌍) 이므로 A→C→D→B 는 6개 pair 를 만든다 (인접 3개가 아니라)."""
    lab = np.repeat([1, 3, 4, 2], 32)                      # A(0) C(2) D(3) B(1)
    mm = _line(np.zeros(3), np.array([100.0, 0, 0]))
    segs = streamline_segments(mm, lab, n_points=16, min_dwell=2, min_length_mm=0.0)
    pairs = sorted(p for p, _, _ in segs)
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for _, s, L in segs:
        assert s.shape == (16, 3) and np.isfinite(s).all() and L > 0


def test_segment_spans_between_the_two_rois():
    lab = np.concatenate([np.full(40, 1), np.zeros(48, int), np.full(40, 2)])
    mm = _line(np.zeros(3), np.array([128.0, 0, 0]))
    segs = {p: (s, L) for p, s, L in streamline_segments(mm, lab, 16, 2, 0.0)}
    s, L = segs[(0, 1)]
    assert np.isclose(s[0, 0], 0.0, atol=1.5) and np.isclose(s[-1, 0], 127.0, atol=1.5)
    assert 120 < L < 130                                   # 전체 길이에 가깝다


def test_short_segments_dropped():
    lab = np.concatenate([np.full(64, 1), np.full(64, 2)])
    mm = _line(np.zeros(3), np.array([2.0, 0, 0]))         # 총 2 mm
    assert streamline_segments(mm, lab, 16, 2, min_length_mm=4.0) == []
    assert len(streamline_segments(mm, lab, 16, 2, min_length_mm=0.0)) == 1


def test_decompose_bundle_and_sc():
    atlas = np.zeros((20, 10, 10), np.int16)
    atlas[1:4] = 1; atlas[8:11] = 2; atlas[16:19] = 3
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-20.0, -10.0, -10.0]
    S = np.stack([_line(np.array([-16.0, 0, 0]), np.array([14.0, 0, 0])) for _ in range(5)]).astype(np.float32)
    key, seg, L = decompose_bundle(S, atlas, affine, n_roi=4, n_points=16, min_dwell=2, min_length_mm=1.0)
    assert seg.shape == (len(key), 16, 3) and len(key) == 15          # 5 streamline x 3 pair
    M = segment_sc(key, 4)
    assert M[0, 1] == 5 and M[0, 2] == 5 and M[1, 2] == 5 and (M == M.T).all() and M[3].sum() == 0
