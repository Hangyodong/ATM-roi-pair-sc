"""TRAIN 분포 기반 QC 임계값 (GESTA QC 문서 §11, §16)."""
import numpy as np
import pytest

from atm_sc.filtering.qc_thresholds import QCThresholds, end_to_end_ratio, measure, winding_deg
from atm_sc.filtering.t1_streamline_filter import FilterConfig, filter_streamlines


def _straight(n=200, L=100.0, P=128, seed=0, noise=0.01):
    """128점으로 재샘플된 GT 처럼 매끄러운 직선. 잡음이 크면 winding 이 잡음으로 채워진다
    (실측: 실제 GT 의 총 회전량 중앙값 605도 — 매끄러운 곡률에서 나온다)."""
    t = np.linspace(0, 1, P)[None, :, None]
    d = np.random.default_rng(seed).normal(0, noise, (n, P, 3))
    return (np.array([0., 0, 0]) * (1 - t) + np.array([L, 0, 0]) * t + d).astype(np.float64)


def test_winding_and_end_ratio():
    s = _straight(10, seed=1)
    assert winding_deg(s).max() < 400                       # 매끄러운 직선
    assert (end_to_end_ratio(s) > 0.99).all()
    th = np.linspace(0, 2 * np.pi, 128)                     # 원 한 바퀴
    loop = np.stack([np.cos(th), np.sin(th), np.zeros(128)], 1)[None] * 20
    assert 300 < winding_deg(loop)[0] < 400                 # 약 360도
    assert end_to_end_ratio(loop)[0] < 0.05                 # 제자리로 돌아옴 -> 직선비가 잡아낸다


def test_measure_makes_real_pass():
    s = _straight(500, seed=2)
    th = measure(s, q=(1.0, 99.0))
    L = np.linalg.norm(np.diff(s, axis=1), axis=-1).sum(1)
    keep = (L >= th.min_length_mm) & (L <= th.max_length_mm)
    assert keep.mean() >= 0.97                              # 자기 분포는 통과해야 한다
    assert th.n_real == 500 and th.max_turn_deg > 0


def test_save_load_roundtrip(tmp_path):
    th = measure(_straight(300, seed=3))
    p = th.save(tmp_path / "qc.json")
    th2 = QCThresholds.load(p)
    assert th2.max_turn_deg == th.max_turn_deg and th2.n_real == th.n_real
    cfg = th2.to_filter_config()
    assert isinstance(cfg, FilterConfig) and cfg.max_turn_deg == th.max_turn_deg


def test_winding_criterion_rejects_loops():
    atlas = np.zeros((30, 30, 30), np.int16); atlas[1:4] = 1; atlas[26:29] = 2
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-30.0, -30.0, -30.0]
    good = _straight(4, L=52.0, seed=4) + np.array([-26.0, 0, 0])
    th = np.linspace(0, 6 * np.pi, 128)                     # 세 바퀴 -> 제자리 복귀
    loop = np.stack([np.cos(th) * 26, np.sin(th) * 3, np.zeros(128)], 1)[None].repeat(2, 0)
    S = np.concatenate([good, loop])
    P = np.zeros((len(S), 2), np.int64); P[:, 1] = 1
    cfg = FilterConfig(endpoint=False, dedup_tol_mm=0.0, min_end_ratio=0.15)   # 실측 real p1
    keep, st = filter_streamlines(S, P, atlas, affine, cfg=cfg)
    assert keep[:4].all() and not keep[4:].any() and st["winding"] == pytest.approx(4 / 6)
