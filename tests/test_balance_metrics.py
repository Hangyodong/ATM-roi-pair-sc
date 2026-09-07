"""균형 학습 검증 지표 (GESTA 전략 §57–60, §78, §82). CPU 만."""
import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from atm_sc.data.roi_groups import TIER_EDGES, block_masks, block_of_pairs, tier_of_strength
from atm_sc.evaluation.balance_metrics import (SIZE_EDGES, bundle_geometry_metrics, exposure_report,
                                                sc_metrics_extended, size_bin_of, spearman)
from atm_sc.losses.sc_corr import upper

R, P = 82, 128


def _gt(seed=0):
    rng = np.random.default_rng(seed)
    g = np.triu(rng.lognormal(3, 2, (R, R)), 1)
    return torch.as_tensor(g + g.T, dtype=torch.float32)


def _bundle(n=40, seed=0):
    """x 축을 따라 ~90 mm 진행하며 z 로 휘는 합성 bundle [n,P,3] mm. 끝점 산포 ±5 mm (서로 1 mm 안에 없음)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, P)
    start = rng.uniform(-5, 5, (n, 1, 3)); end = start + np.array([90.0, 0, 0]) + rng.uniform(-5, 5, (n, 1, 3))
    S = start + (end - start) * t[None, :, None]
    S[:, :, 2] += 8 * np.sin(np.pi * t)
    return S


def test_spearman():
    x = torch.linspace(-3, 3, 200)
    assert spearman(x, torch.exp(x) + x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x ** 3) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    b[:100] = a[:100].round(0); a[200:250] = 1.5                       # 동률 (평균 순위)
    assert spearman(torch.as_tensor(a), torch.as_tensor(b)) == pytest.approx(spearmanr(a, b).statistic, abs=1e-6)


def test_sc_metrics_extended_weak_and_strong_edges():
    g = _gt()
    m = sc_metrics_extended(g, g)
    assert m["weak_edge_recall"] == 1.0 and m["strong_edge_precision"] == 1.0
    assert m["spearman"] == pytest.approx(1.0) and m["r"] == pytest.approx(1.0)
    assert m["n_edges"] == 3321 and m["strong_edge_n"] == 167
    assert m["weak_edge_n"] == int((upper(g) <= TIER_EDGES[0]).sum()) and 0.7 < m["weak_edge_n"] / 3321 < 0.9
    # GT <= 100 인 edge (~79 %) 를 전부 0 으로: Pearson 은 거의 안 움직이고 strong precision 도 1 인데 recall 만 0 —
    # 약한 edge 소실은 이 지표 없이는 보이지 않는다 (§58).
    p = g * (g > TIER_EDGES[0])
    w = sc_metrics_extended(p, g)
    assert w["weak_edge_recall"] == 0.0 and w["weak_edge_n"] == m["weak_edge_n"]
    assert w["strong_edge_precision"] == 1.0
    assert m["r"] - w["r"] < 0.02 and w["spearman"] < 0.9
    # 순서를 뒤집으면 강한 edge 예측은 전부 틀린다
    inv = torch.where(g > 0, 1.0 / g.clamp(min=1e-6), torch.zeros(()))
    i = sc_metrics_extended(inv, g)
    assert i["spearman"] == pytest.approx(-1.0) and i["strong_edge_precision"] == 0.0 and i["weak_edge_recall"] == 1.0
    # mask: sub-sub block (120 edge) 만
    mask = block_masks(R)["sub-sub"]
    gs = upper(g)[upper(torch.as_tensor(mask))]
    s = sc_metrics_extended(p, g, mask=mask)
    assert s["n_edges"] == 120 and s["strong_edge_n"] == 6 and s["weak_edge_n"] == int((gs <= TIER_EDGES[0]).sum())
    assert s["weak_edge_recall"] == 0.0 and s["strong_edge_precision"] == 1.0
    full = sc_metrics_extended(g, g, mask=mask)
    assert full["weak_edge_recall"] == 1.0 and full["spearman"] == pytest.approx(1.0)


def test_size_bin_boundaries():
    assert SIZE_EDGES == (20, 256)
    assert size_bin_of(np.array([19, 20, 255, 256])).tolist() == [0, 1, 1, 2]


def test_bundle_geometry_metrics():
    S = _bundle()
    same = bundle_geometry_metrics(S, S)
    assert same["coverage"] == 1.0 and same["overreach"] == 0.0 and same["dice"] == 1.0
    assert same["duplicate_ratio"] == 0.0 and same["valid_ratio"] == 1.0
    assert same["length_err_mm"] == 0.0 and same["length_ks"] == 0.0 and same["n_gen"] == same["n_gt"] == 40
    # y 로 30 mm 평행이동 (= 15 voxel): 겹치는 voxel 없음, voxel 수는 같음
    shifted = bundle_geometry_metrics(S + np.array([0, 30.0, 0]), S)
    assert shifted["coverage"] == 0.0 and shifted["overreach"] == pytest.approx(1.0, abs=0.01) and shifted["dice"] == 0.0
    assert shifted["length_err_mm"] == pytest.approx(0.0, abs=1e-9)
    # 전부 한 번씩 복제 -> 절반이 duplicate
    dup = bundle_geometry_metrics(np.concatenate([S, S]), S)
    assert dup["duplicate_ratio"] == 0.5 and dup["coverage"] == 1.0 and dup["overreach"] == 0.0
    # 5 mm 짜리 조각과 NaN streamline 은 invalid (voxel 지표는 유한한 것만)
    short = np.zeros((1, P, 3)); short[0, :, 0] = np.linspace(0, 5, P)
    bad = np.full((1, P, 3), np.nan)
    inv = bundle_geometry_metrics(np.concatenate([S, short, bad]), S)
    assert inv["valid_ratio"] == pytest.approx(40 / 42) and inv["coverage"] == 1.0 and inv["n_gen"] == 42
    assert 0 < inv["overreach"] < 0.05 and inv["length_ks"] > 0
    with pytest.raises(AssertionError):
        bundle_geometry_metrics(S[:0], S)


def test_exposure_report():
    # §78: A = 5000 / B = 500 / C = 20 (ctx-ctx / ctx-sub / sub-sub), 노출 ∝ count 면 250:25:1
    counts = np.array([5000, 500, 20]); pairs = np.array([[0, 1], [0, 70], [70, 71]])
    bl, tr = block_of_pairs(pairs), tier_of_strength(counts)
    rep = exposure_report(counts, counts * 3, bl, tr)
    assert rep["ratio_max_to_min_pair"] == pytest.approx(250.0)
    assert rep["max_pair_share"] == pytest.approx(0.906, abs=1e-3)
    assert np.isnan(rep["ratio_high_to_low"])                          # 20 은 mid bin -> low bin 이 비어 있다
    assert rep["size"]["high"] == pytest.approx({"n_pairs": 2, "exposure_share": 5500 / 5520, "count_share": 5500 / 5520})
    assert rep["size"]["low"]["n_pairs"] == 0 and rep["size"]["low"]["exposure_share"] == 0.0
    assert [rep["block"][b]["n_pairs"] for b in ("ctx-ctx", "ctx-sub", "sub-sub")] == [1, 1, 1]
    assert [rep["tier"][t]["n_pairs"] for t in ("small", "mid", "large")] == [1, 1, 1]
    assert rep["tier"]["small"]["exposure_share"] == pytest.approx(20 / 5520)
    uni = exposure_report(counts, np.full(3, 7), bl, tr)
    assert uni["ratio_max_to_min_pair"] == 1.0 and uni["max_pair_share"] == pytest.approx(1 / 3)
    assert uni["size"]["high"] == pytest.approx({"n_pairs": 2, "exposure_share": 2 / 3, "count_share": 5500 / 5520})
    # low bin 이 있으면 bin 평균 비율이 정의된다
    c4 = np.array([5000, 500, 20, 10]); bl4, tr4 = np.array([0, 1, 2, 2]), tier_of_strength(c4)
    assert exposure_report(c4, c4 * 2, bl4, tr4)["ratio_high_to_low"] == pytest.approx(2750 / 10)
    assert exposure_report(c4, np.ones(4), bl4, tr4)["ratio_high_to_low"] == 1.0
    with pytest.raises(AssertionError):
        exposure_report(counts, np.zeros(3), bl, tr)
