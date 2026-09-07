"""T1-only streamline filter: 합성 atlas/mask 위에서 기준을 하나씩 떨어뜨리는 케이스."""
import numpy as np
import pytest

from atm_sc.filtering.t1_streamline_filter import (FilterConfig, _dedup, filter_streamlines,
                                                   max_turn_angles_deg, streamline_lengths_mm)

P = 128
A, B = np.array([-28.0, 0.0, 0.0]), np.array([28.0, 0.0, 0.0])   # ROI 1 / ROI 2 안의 기준점 (mm)


def _line(p0, p1, n=P):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p0 * (1 - t) + p1 * t


@pytest.fixture(scope="module")
def grid():
    """atlas 40^3 @2mm (mm [-40,38]) + brain mask 80^3 @1mm.

    ROI 1/2 는 x=0 에서 맞닿은 두 블록 (x mm [-32,0) / [0,32), y,z mm [-16,16)) — 그래야 20 mm
    미만 streamline 도 endpoint 는 통과해 length 만 떨어진다. mask 는 y 방향만 좁다 (mm [-16,16))
    — (f) 의 반원이 y 로 벗어나면서 endpoint/length/curvature 는 유지한다.
    """
    atlas = np.zeros((40, 40, 40), np.int16)
    atlas[4:20, 12:28, 12:28] = 1
    atlas[20:36, 12:28, 12:28] = 2
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = -40.0
    mask = np.zeros((80, 80, 80), bool)
    mask[4:76, 24:56, 4:76] = True
    maff = np.eye(4); maff[:3, 3] = -40.0
    return atlas, affine, mask, maff


def _cases():
    """(a)..(h) 순서. 각 케이스는 의도한 기준 하나만 떨어지도록 만든다 ((h) 는 전부)."""
    a = _line(A, B)                                                     # 56 mm 직선
    b = _line(A, np.array([-28.0, 0.0, 30.0]))                          # 끝점 z idx 35: 배경
    c = _line(np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))    # 4 mm
    t = np.linspace(0.0, 1.0, P)                                        # 반지름 10 나선 4회전 ≈ 256 mm, 꺾임 ≈ 11°
    d = np.stack([-28 + 56 * t, 10 * np.sin(8 * np.pi * t), 10 - 10 * np.cos(8 * np.pi * t)], 1)
    apex = np.array([0.0, 0.0, 10 * np.sqrt(3.0)])                      # 두 다리 ±60° → 꼭짓점(index 64)에서 120°
    e = np.concatenate([_line(np.array([-10.0, 0.0, 0.0]), apex, 65),
                        _line(apex, np.array([10.0, 0.0, 0.0]), 64)[1:]])
    th = np.linspace(0.0, np.pi, P)
    f = np.stack([-28 * np.cos(th), 28 * np.sin(th), np.zeros(P)], 1)   # y 로 부푼 반원: 61% 가 mask 밖
    g = a + np.array([0.0, 0.2, 0.0])
    h = a.copy(); h[50, 1] = np.nan
    S = np.stack([a, b, c, d, e, f, g, h]).astype(np.float32)
    return S, np.tile(np.array([0, 1], np.int64), (len(S), 1))


def test_helpers():
    S, _ = _cases()
    L = streamline_lengths_mm(S)
    assert L[0] == pytest.approx(56.0) and L[2] == pytest.approx(4.0) and L[3] > 220 and np.isnan(L[7])
    ang = max_turn_angles_deg(S)
    assert ang[0] < 1e-3 and ang[4] == pytest.approx(120.0, abs=1e-3) and ang[3] < 60 and np.isnan(ang[7])


def test_each_criterion(grid):
    atlas, affine, mask, maff = grid
    S, pairs = _cases()
    cfg = FilterConfig(dedup_tol_mm=1.0)      # 케이스 (g) 는 0.2 mm 복사본이라 기본값(0.1)보다 큰 tol 로 검사
    keep, st = filter_streamlines(S, pairs, atlas, affine, mask, maff, cfg)
    assert keep.tolist() == [True] + [False] * 7
    fails = {"finite": {7}, "endpoint": {1, 7}, "length": {2, 3, 7},
             "curvature": {4, 7}, "brain": {5, 7}, "dedup": {6}}
    for k, bad in fails.items():
        assert st[k] == pytest.approx(1 - len(bad) / len(S)), (k, st[k])
    assert st["n_in"] == 8 and st["n_keep"] == 1 and st["pass_rate"] == pytest.approx(1 / 8)


def test_endpoint_order_and_disable(grid):
    atlas, affine, mask, maff = grid
    S, pairs = _cases()
    keep, st = filter_streamlines(S[:1, ::-1], pairs[:1], atlas, affine, mask, maff)   # b -> a 순서
    assert keep.all() and st["endpoint"] == 1.0
    keep, st = filter_streamlines(S, pairs, atlas, affine, mask, maff,
                                  FilterConfig(endpoint=False, dedup_tol_mm=1.0))
    assert keep[1] and st["endpoint"] == 1.0 and st["n_keep"] == 2


def test_no_mask_and_reversed_duplicate(grid):
    atlas, affine, _, _ = grid
    S, pairs = _cases()
    keep, st = filter_streamlines(S[[0, 5]], pairs[:2], atlas, affine)      # mask 없으면 (f) 도 통과
    assert keep.tolist() == [True, True] and st["brain"] == 1.0
    keep, st = filter_streamlines(np.stack([S[0], S[0, ::-1]]), pairs[:2], atlas, affine)
    assert keep.tolist() == [True, False] and st["dedup"] == 0.5             # 뒤집힌 복사본도 중복


def _ref_dedup(S, pairs, cand, tol):
    """O(N^2) 참조 구현."""
    keep = np.ones(len(S), bool)
    for i in np.flatnonzero(cand):
        for j in np.flatnonzero(cand[:i] & keep[:i]):
            if (pairs[j] == pairs[i]).all() and min(np.linalg.norm(S[j] - S[i], axis=-1).mean(),
                                                    np.linalg.norm(S[j] - S[i, ::-1], axis=-1).mean()) < tol:
                keep[i] = False
                break
    return keep


def test_dedup_matches_bruteforce():
    """격자 hash 가 후보를 놓치지 않는지: 무작위 이동(0~2 mm, tol 1 mm 양쪽)·뒤집기·pair 섞기."""
    rng = np.random.default_rng(0)
    S, pairs = [], []
    for _ in range(30):
        base = _line(*rng.uniform(-50, 50, (2, 3)))
        sh = rng.normal(size=(8, 3)); sh *= (rng.uniform(0, 2, 8) / np.linalg.norm(sh, axis=1))[:, None]
        for s in base[None] + sh[:, None]:
            S.append(s[::-1] if rng.random() < 0.5 else s)
            pairs.append([[0, 1], [0, 2], [1, 2]][rng.integers(3)])
    perm = rng.permutation(len(S))
    S, pairs = np.stack(S)[perm], np.asarray(pairs, np.int64)[perm]
    cand = rng.random(len(S)) > 0.2
    k = _dedup(S, pairs, cand, 1.0)
    assert np.array_equal(k, _ref_dedup(S, pairs, cand, 1.0))
    assert 0 < (~k).sum() < len(S) and k[~cand].all()
