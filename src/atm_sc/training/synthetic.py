"""합성 subject (pipeline §27 smoke test 용). dataset.ROIPairSubject 와 같은 인터페이스."""
from __future__ import annotations

import numpy as np
import torch

from ..data.roi_groups import balanced_choice, block_of_pairs, tier_of_strength
from ..models.endpoint_assigner import build_distance_maps
from ..spaces import voxel_to_mm


class InMemorySubject:
    """pair_ids [K,2], per-pair streamlines, GT 행렬을 메모리에 들고 있는 subject."""

    def __init__(self, n_roi, pair_ids, bundles, lengths, sc_end, len_end, sub="synthetic"):
        self.n_roi = n_roi
        self.sub = sub                       # Trainer 가 subject 별 sampler 를 캐시할 때 키로 쓴다
        self.pair_ids = np.asarray(pair_ids, np.int64)
        self._bundles = [np.asarray(b, np.float32) for b in bundles]
        self._lengths = [np.asarray(l, np.float32) for l in lengths]
        self.pair_count_full = np.array([len(b) for b in self._bundles], np.int64)
        self.sc_end, self.len_end = sc_end, len_end
        self.sc_pass, self.len_pass = sc_end, len_end      # 합성에서는 구분하지 않는다
        self.sc_mat, self.len_mat = sc_end, len_end

    def get_pair(self, k):
        return torch.from_numpy(self._bundles[k]), torch.from_numpy(self._lengths[k])

    @property
    def pair_strength(self):
        p = self.pair_ids
        s = np.asarray(self.sc_mat, np.float64)[p[:, 0], p[:, 1]].copy()
        z = s <= 0
        s[z] = self.pair_count_full[z]
        return s

    @property
    def pair_tier(self):
        return tier_of_strength(self.pair_strength)

    @property
    def pair_block(self):
        return block_of_pairs(self.pair_ids)

    def synthetic_pair(self, k):
        return torch.zeros(0, 128, 3)

    def set_visitation(self, atlas, affine):
        """route loss 용 GT 통과 ROI (scripts/24 와 같은 hard 규칙)."""
        from ..data.tt_io import point_labels
        self._visit = []
        for b in self._bundles:
            lab = point_labels(b.reshape(-1, 3).astype(np.float64), atlas, affine).astype(np.int64).reshape(len(b), -1)
            v = np.zeros((len(b), self.n_roi), np.float32)
            rows = np.repeat(np.arange(len(b)), lab.shape[1])
            keep = lab.ravel() > 0
            v[rows[keep], lab.ravel()[keep] - 1] = 1.0
            self._visit.append(v)
        self.pair_marginal = np.stack([v.mean(0) for v in self._visit]).astype(np.float32)
        self.has_visitation = True

    has_visitation = False

    def visitation(self, k):
        return torch.from_numpy(self._visit[k])

    def sample_pairs(self, n, rng, weighted=True, mode=None):
        mode = mode or ("log" if weighted else "uniform")
        if mode == "tier":
            return balanced_choice(self.pair_tier, min(n, len(self.pair_ids)), rng)
        p = np.log1p(self.pair_count_full) if mode == "log" else np.ones(len(self.pair_ids))
        return rng.choice(len(self.pair_ids), size=min(n, len(self.pair_ids)),
                          replace=False, p=p / p.sum())

    def negative_pairs(self, n, rng):
        iu = np.stack(np.triu_indices(self.n_roi, 1), 1)
        neg = iu[self.sc_end[iu[:, 0], iu[:, 1]] == 0]
        return neg[rng.choice(len(neg), size=min(n, len(neg)), replace=False)]

    @property
    def positive_ratio(self):
        return len(self.pair_ids) / (self.n_roi * (self.n_roi - 1) / 2)


def make_synthetic(n_roi: int = 6, n_pos_pairs: int = 3, n_per_pair: int = 8,
                   n_points: int = 128, seed: int = 0):
    """작은 atlas + 직선 streamline. (subject, atlas, affine, dist_maps, coord_min, coord_max)"""
    rng = np.random.default_rng(seed)
    shape = (24, 20, 18)
    atlas = np.zeros(shape, np.int16)
    for r in range(n_roi):
        atlas[2 + r * 3: 4 + r * 3, 4:16, 4:14] = r + 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-24.0, -20.0, -18.0]
    dist = build_distance_maps(atlas, n_roi, (2.0, 2.0, 2.0))

    all_pairs = [(a, b) for a in range(n_roi) for b in range(a + 1, n_roi)]
    pos = [all_pairs[i] for i in rng.choice(len(all_pairs), n_pos_pairs, replace=False)]
    bundles, lengths = [], []
    sc = np.zeros((n_roi, n_roi), np.float64); ln = np.zeros_like(sc)
    t = np.linspace(0, 1, n_points)[:, None]
    for a, b in pos:
        pa = voxel_to_mm(np.array([3.0 + a * 3, 10.0, 9.0]), affine)
        pb = voxel_to_mm(np.array([3.0 + b * 3, 10.0, 9.0]), affine)
        S = np.stack([pa * (1 - t) + pb * t + rng.normal(0, 0.3, (n_points, 3))
                      for _ in range(n_per_pair)]).astype(np.float32)
        L = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)
        bundles.append(S); lengths.append(L)
        sc[a, b] = sc[b, a] = n_per_pair
        ln[a, b] = ln[b, a] = L.mean()
    lo = voxel_to_mm(np.zeros(3), affine) - 2; hi = voxel_to_mm(np.array(shape) - 1.0, affine) + 2
    subj = InMemorySubject(n_roi, np.array(sorted(pos)), bundles, lengths, sc, ln)
    # 정렬 순서에 맞춰 bundles 재정렬
    order = np.argsort([a * n_roi + b for a, b in pos])
    subj._bundles = [bundles[i] for i in order]; subj._lengths = [lengths[i] for i in order]
    subj.pair_count_full = np.array([len(b) for b in subj._bundles])
    return subj, atlas, affine, dist, lo.astype(np.float32), hi.astype(np.float32)
