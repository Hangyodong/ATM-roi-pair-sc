"""SC edge-aligned segment bundle 균형 sampler (전략 문서 §13-16, §48).

endpoint bundle(A-B) 대신 **edge-aligned segment bundle** 을 sampling unit 으로 쓴다.
balanced_pair_sampler 와 같은 GESTA 노출 규칙을 그대로 쓴다:
    B_e = clip(B_base · (N_e / B_base)^α, B_min, B_max)      (α < 1 이라 큰 edge 가 count 비례로 뽑히지 않는다)
edge e 선택 확률 p_e ∝ B_e, 선택된 edge 마다 고정 n 개 segment 를 채운다.
segment 는 32 point 로 저장되어 있고 ATM decoder 는 128 point 를 쓰므로 호길이 보존 보간으로 올린다.
GT SC 는 건드리지 않는다: 여기서 만든 batch 는 segment branch 의 recon 에만 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .balanced_pair_sampler import exposure_targets
from .roi_groups import BLOCKS, balanced_choice, block_of_pairs

EDGE_SAMPLING = ("target", "uniform", "block")


@dataclass
class SegmentBalanceConfig:
    enabled: bool = False
    alpha: float = 0.5
    b_base: float = 100.0
    b_min: float = 16.0
    b_max: float = 256.0
    n_points: int = 128                 # ATM decoder 입력 길이. 저장된 segment 는 32 point (edge_segments.SEG_POINTS)
    min_segments: int = 2               # 저장 segment 가 이보다 적은 edge 는 뽑지 않는다 (geometry 다양성이 없다)
    edge_sampling: str = "target"       # target: p ∝ 노출 목표 · uniform: edge 균등 · block: ctx/sub block 균등


def upsample_segments(S: torch.Tensor, n_points: int) -> torch.Tensor:
    """[m, p, 3] -> [m, n_points, 3] 호길이 등간격 선형 보간 (edge_segments._resample 의 batch 판).

    p == n_points 면 그대로 돌려준다. 양 끝점은 정확히 보존된다 (t=0, t=총길이 에서 가중치가 0/1 이 된다).
    """
    assert S.ndim == 3 and S.shape[2] == 3, S.shape
    assert n_points >= 2, n_points
    p = S.shape[1]
    if p == n_points:
        return S
    assert p >= 2, S.shape
    d = torch.linalg.norm(S[:, 1:] - S[:, :-1], dim=2)                       # [m, p-1] 점간 거리
    s = torch.cat([torch.zeros_like(d[:, :1]), torch.cumsum(d, 1)], 1)       # [m, p] 누적 호길이
    total = s[:, -1:].clamp_min(1e-8)                                        # 길이 0 (한 점에 뭉친 segment) 방어
    t = torch.linspace(0.0, 1.0, n_points, dtype=S.dtype, device=S.device)[None] * total
    hi = torch.searchsorted(s.contiguous(), t.contiguous(), right=True).clamp_(1, p - 1)
    lo = hi - 1
    s0, s1 = s.gather(1, lo), s.gather(1, hi)
    w = ((t - s0) / (s1 - s0).clamp_min(1e-12)).unsqueeze(2)                 # [m, n, 1]
    i3 = lo.unsqueeze(2).expand(-1, -1, 3)
    j3 = hi.unsqueeze(2).expand(-1, -1, 3)
    out = S.gather(1, i3) * (1.0 - w) + S.gather(1, j3) * w
    assert out.shape == (S.shape[0], n_points, 3), out.shape
    assert torch.isfinite(out).all()
    return out


def _stored_counts(subject, E: int) -> np.ndarray:
    """edge 별 저장 segment 수. offsets 이 있으면 그걸 쓴다 (edge 마다 slice 를 복사하면 npz 전체를 한 번 복사하게 된다)."""
    z = getattr(subject, "_seg", None)
    if isinstance(z, dict) and "offsets" in z:
        return np.diff(np.asarray(z["offsets"], np.int64))
    return np.array([int(subject.edge_segments(e).shape[0]) for e in range(E)], np.int64)


class BalancedSegmentSampler:
    def __init__(self, subject, cfg: SegmentBalanceConfig):
        assert cfg.edge_sampling in EDGE_SAMPLING, cfg.edge_sampling
        assert getattr(subject, "has_edge_segments", False), "edge_segments.npz 없음 (scripts/26_build_edge_segments.py)"
        self.subject, self.cfg = subject, cfg
        self.pair_ids = np.asarray(subject.edge_pair_ids, np.int64)
        assert self.pair_ids.ndim == 2 and self.pair_ids.shape[1] == 2, self.pair_ids.shape
        assert self.pair_ids.min(initial=0) >= 0, "ROI id 는 0-based 여야 한다"
        assert (self.pair_ids[:, 0] < self.pair_ids[:, 1]).all(), "edge_pair_ids 가 canonical(a<b) 이 아니다"
        E = len(self.pair_ids)
        counts = np.asarray(subject.edge_count_full, np.int64)
        assert counts.shape == (E,) and (counts >= 1).all(), (counts.shape, E)

        self.targets = exposure_targets(counts, cfg)                          # [E] float, 무효 edge 포함
        self.blocks = block_of_pairs(self.pair_ids)                           # [E] 0 ctx-ctx / 1 ctx-sub / 2 sub-sub
        self.n_stored = _stored_counts(subject, E)
        assert self.n_stored.shape == (E,), (self.n_stored.shape, E)
        self.valid = np.flatnonzero(self.n_stored >= cfg.min_segments)
        assert len(self.valid) > 0, f"min_segments={cfg.min_segments} 를 넘는 edge 가 하나도 없다"

        w = np.zeros(E, np.float64)
        w[self.valid] = self.targets[self.valid]
        self.p = w / w.sum()                                                  # 무효 edge 는 확률 0
        self.exposure = np.zeros(E, np.int64)                                 # edge 별 누적 노출 (segment 수)

    def sample_edges(self, n_edges: int, rng: np.random.Generator) -> np.ndarray:
        """[n_edges] edge index (subject.edge_pair_ids 행 번호)."""
        assert n_edges >= 1, n_edges
        K = len(self.valid)
        if self.cfg.edge_sampling == "block":
            return self.valid[balanced_choice(self.blocks[self.valid], n_edges, rng)]
        if self.cfg.edge_sampling == "uniform":
            return rng.choice(self.valid, size=n_edges, replace=n_edges > K)
        return rng.choice(len(self.p), size=n_edges, replace=n_edges > K, p=self.p)

    def _fill(self, e: int, n: int, rng: np.random.Generator):
        """edge e 에서 n 개 segment: 가능하면 비복원, 저장 수가 모자라면 복원 추출 (duplicate oversampling)."""
        segs = self.subject.edge_segments(e)                                  # [m, p, 3]
        m = int(segs.shape[0])
        assert m >= self.cfg.min_segments, (e, m)
        i = rng.choice(m, size=n, replace=n > m)
        ti = torch.as_tensor(i, dtype=torch.long)
        S = upsample_segments(segs[ti].float(), self.cfg.n_points)
        L = self.subject.edge_segment_lengths(e)[ti].float()
        return S, L

    def sample_batch(self, rng: np.random.Generator, n_edges: int, n_per_edge: int):
        """-> (S [n_edges*n_per_edge, n_points, 3] float32, P [.,2] int64, lengths [.] float32, info)."""
        assert n_per_edge >= 1, n_per_edge
        es = self.sample_edges(n_edges, rng)
        S, P, L = [], [], []
        for e in es:
            e = int(e)
            s, l = self._fill(e, n_per_edge, rng)
            S.append(s)
            L.append(l)
            P.append(torch.as_tensor(self.pair_ids[e]).repeat(s.shape[0], 1))
            self.exposure[e] += s.shape[0]
        S = torch.cat(S).float()
        P = torch.cat(P).long()
        L = torch.cat(L).float()
        n = n_edges * n_per_edge
        assert S.shape == (n, self.cfg.n_points, 3), S.shape
        assert P.shape == (n, 2) and (P[:, 0] < P[:, 1]).all() and int(P.min()) >= 0
        assert L.shape == (n,) and torch.isfinite(L).all()
        blk = self.blocks[es]
        info = {"seg_real_frac": 1.0,                                          # synthetic segment 는 아직 없다 (§48)
                "seg_mean_length_mm": float(L.mean()),
                "seg_edges": int(len(np.unique(es)))}
        info.update({f"seg_block_{b}": float((blk == i).mean()) for i, b in enumerate(BLOCKS)})
        return S, P, L, info
