"""edge-aligned segment bundle sampler smoke test (전략 문서 §13-16, §48).

실제 데이터 파일에 의존하지 않도록 FakeSubject 로 ROIPairSubject 의 segment API 5 개만 흉내낸다.
"""
import numpy as np
import pytest
import torch

from atm_sc.data.roi_groups import N_CTX, block_of_pairs
from atm_sc.data.segment_sampler import (BalancedSegmentSampler, SegmentBalanceConfig, upsample_segments)


def _bundle(m: int, g: torch.Generator) -> torch.Tensor:
    """[m, 32, 3] 길이가 0 이 아닌 가짜 segment (누적합이라 되돌아가도 호길이 > 0)."""
    step = torch.randn(m, 31, 3, generator=g) * 0.5 + torch.tensor([1.0, 0.2, 0.1])
    return torch.cat([torch.zeros(m, 1, 3), step.cumsum(1)], 1).float()


class FakeSubject:
    """ROIPairSubject 의 edge segment 인터페이스만 노출한다."""

    def __init__(self, counts=(5000, 500, 20), pairs=((0, 1), (2, 3), (4, 5)), stored=None, seed=0):
        self.sub = "fake"
        self.has_edge_segments = True
        self.edge_pair_ids = np.asarray(pairs, np.int16)
        self.edge_count_full = np.asarray(counts, np.int32)
        g = torch.Generator().manual_seed(seed)
        stored = [min(int(c), 64) for c in counts] if stored is None else list(stored)
        self._segs = [_bundle(int(m), g) for m in stored]
        self._lens = [torch.rand(int(m), generator=g) * 40.0 + 5.0 for m in stored]

    def edge_segments(self, e):
        return self._segs[e]

    def edge_segment_lengths(self, e):
        return self._lens[e]


def _block_subject(n_cc=60, n_cs=8, n_ss=3, seed=1):
    """ctx-ctx 가 압도적으로 많은 subject (실제 82 ROI 비율: 2145 / 1056 / 120)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_cc):
        a, b = sorted(rng.choice(N_CTX, 2, replace=False))
        pairs.append((a, b))
    for _ in range(n_cs):
        pairs.append((int(rng.integers(0, N_CTX)), int(rng.integers(N_CTX, 82))))
    for _ in range(n_ss):
        a, b = sorted(rng.choice(np.arange(N_CTX, 82), 2, replace=False))
        pairs.append((a, b))
    pairs = np.asarray(pairs, np.int64)
    counts = rng.integers(10, 5000, len(pairs))
    return FakeSubject(counts=tuple(counts), pairs=tuple(map(tuple, pairs)), seed=seed)


# ---------------------------------------------------------------- 1. upsample

def test_upsample_preserves_endpoints_and_arclength():
    t = torch.linspace(0.0, 1.0, 32)
    line = torch.stack([t * 10.0, t * 3.0, -t * 2.0], 1)[None]              # [1,32,3] 등간격 직선
    out = upsample_segments(line, 128)
    assert out.shape == (1, 128, 3) and out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 0], line[0, 0]) and torch.equal(out[0, -1], line[0, -1])
    d = torch.linalg.norm(out[0, 1:] - out[0, :-1], dim=1)
    assert (d > 0).all()                                                     # 호길이 단조 증가
    assert float(d.max() - d.min()) < 1e-4                                   # 직선이면 등간격이어야 한다
    assert float(d.sum()) == pytest.approx(float(np.sqrt(100 + 9 + 4)), rel=1e-5)


def test_upsample_identity_and_curved_endpoints():
    S = _bundle(16, torch.Generator().manual_seed(3))
    assert upsample_segments(S, 32) is S                                     # p == n_points 면 그대로
    out = upsample_segments(S, 128)
    assert out.shape == (16, 128, 3) and torch.isfinite(out).all()
    assert torch.equal(out[:, 0], S[:, 0]) and torch.equal(out[:, -1], S[:, -1])
    # 굽은 곡선은 원 vertex 가 새 sample 사이에 끼면 모서리가 잘려 아주 조금 짧아진다 (5 % 이내)
    L_in = torch.linalg.norm(S[:, 1:] - S[:, :-1], dim=2).sum(1)
    L_out = torch.linalg.norm(out[:, 1:] - out[:, :-1], dim=2).sum(1)
    assert (L_out <= L_in + 1e-4).all() and (L_out > 0.95 * L_in).all()
    d = torch.linalg.norm(out[:, 1:] - out[:, :-1], dim=2)
    assert (d > 0).all()                                                     # 호길이 단조 증가 (되돌아가지 않는다)


# ---------------------------------------------------------------- 2. 노출 압축

def test_exposure_is_not_proportional_to_counts():
    sub = FakeSubject(counts=(5000, 500, 20))                                # raw count 비 = 250
    s = BalancedSegmentSampler(sub, SegmentBalanceConfig(enabled=True))
    rng = np.random.default_rng(0)
    for _ in range(3000):
        s.sample_batch(rng, n_edges=1, n_per_edge=4)
    assert s.exposure.sum() == 3000 * 4
    ratio = s.exposure[0] / s.exposure[2]
    assert 1.0 < ratio < 15.0, ratio                                         # 250 이 아니라 target 비 256/44.7 ≈ 5.7
    assert ratio == pytest.approx(s.targets[0] / s.targets[2], rel=0.2)


# ---------------------------------------------------------------- 3. batch 형식

def test_sample_batch_shapes_and_info():
    sub = _block_subject()
    cfg = SegmentBalanceConfig(enabled=True, n_points=128)
    s = BalancedSegmentSampler(sub, cfg)
    S, P, L, info = s.sample_batch(np.random.default_rng(7), n_edges=12, n_per_edge=5)
    n = 12 * 5
    assert S.shape == (n, 128, 3) and S.dtype == torch.float32 and torch.isfinite(S).all()
    assert P.shape == (n, 2) and P.dtype == torch.int64
    assert (P[:, 0] < P[:, 1]).all() and int(P.min()) >= 0                    # canonical · 0-based
    assert L.shape == (n,) and L.dtype == torch.float32 and (L > 0).all()

    pid = torch.as_tensor(np.asarray(sub.edge_pair_ids, np.int64))
    grp = P.view(12, 5, 2)
    assert (grp == grp[:, :1]).all()                                         # edge 당 5 개가 같은 pair
    assert all((pid == g).all(1).any() for g in grp[:, 0])                   # 실제 edge 에서 나왔다

    for k in ("seg_real_frac", "seg_mean_length_mm", "seg_edges"):
        assert k in info
    assert info["seg_real_frac"] == 1.0
    assert 5.0 < info["seg_mean_length_mm"] < 45.0
    assert 1 <= info["seg_edges"] <= 12
    shares = [info[f"seg_block_{b}"] for b in ("ctx-ctx", "ctx-sub", "sub-sub")]
    assert sum(shares) == pytest.approx(1.0)

    s2 = BalancedSegmentSampler(sub, cfg)                                    # 같은 rng seed -> 같은 batch
    S2, P2, L2, _ = s2.sample_batch(np.random.default_rng(7), n_edges=12, n_per_edge=5)
    assert torch.equal(S, S2) and torch.equal(P, P2) and torch.equal(L, L2)


# ---------------------------------------------------------------- 4. 빈약한 edge 제외

def test_thin_edges_are_never_sampled():
    sub = FakeSubject(counts=(5000, 500, 20), stored=(64, 1, 20))            # edge 1 은 segment 1 개
    s = BalancedSegmentSampler(sub, SegmentBalanceConfig(enabled=True, min_segments=2))
    assert s.valid.tolist() == [0, 2] and s.p[1] == 0.0
    rng = np.random.default_rng(1)
    es = np.concatenate([s.sample_edges(16, rng) for _ in range(200)])
    assert not (es == 1).any()
    _, P, _, _ = s.sample_batch(rng, n_edges=32, n_per_edge=3)
    assert not ((P[:, 0] == 2) & (P[:, 1] == 3)).any()                       # edge 1 의 pair
    assert s.exposure[1] == 0


def test_all_thin_edges_raises():
    sub = FakeSubject(counts=(5, 5), pairs=((0, 1), (2, 3)), stored=(1, 1))
    with pytest.raises(AssertionError):
        BalancedSegmentSampler(sub, SegmentBalanceConfig(enabled=True, min_segments=2))


# ---------------------------------------------------------------- 5. block 균형

def test_block_sampling_equalizes_blocks():
    sub = _block_subject()                                                   # 60 / 8 / 3 edge
    s = BalancedSegmentSampler(sub, SegmentBalanceConfig(enabled=True, edge_sampling="block"))
    es = s.sample_edges(300, np.random.default_rng(0))
    blk = block_of_pairs(np.asarray(sub.edge_pair_ids, np.int64))[es]
    for i in range(3):
        share = float((blk == i).mean())
        assert 0.25 <= share <= 0.45, (i, share)

    s2 = BalancedSegmentSampler(sub, SegmentBalanceConfig(enabled=True, edge_sampling="target"))
    blk2 = block_of_pairs(np.asarray(sub.edge_pair_ids, np.int64))[s2.sample_edges(300, np.random.default_rng(0))]
    assert float((blk2 == 0).mean()) > 0.6                                   # target 모드는 edge 수를 그대로 따라간다
