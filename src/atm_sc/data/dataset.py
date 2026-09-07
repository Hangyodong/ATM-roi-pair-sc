"""subject 하나의 ROI-pair 학습 데이터 (pipeline §3, §6).

    s = ROIPairSubject("sub-100001")
    s.pair_ids            [K,2] 양성 canonical pair
    s.get_pair(k)         -> (streamlines [n,128,3] float32 tensor, lengths [n] tensor)
    s.sample_pairs(32, rng)
    s.negative_pairs(32, rng)

학습 target 은 모드별로 다르다 (docs/ROI_PAIR_DATA_FORMAT.md):
    endpoint 모드 -> sc_end / len_end      pass 모드 -> sc_pass (= .mat 의 sc_mat) / len_pass
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from .paths import CACHE, ROOT, SC_MAT
from .roi_groups import balanced_choice, block_of_pairs, tier_of_strength

ROI_PAIRS_DIR = ROOT / "outputs" / "roi_pairs"


def load_mat_gt(sub: str):
    """FC_DKPD25_*.mat 의 (SC_weight, SC_length). dwi_qc==pass 인 238명만 있다."""
    import scipy.io as sio
    m = sio.loadmat(SC_MAT)
    rows = [r for r in m["data"][0] if str(r["subject"][0]) == sub]
    assert rows, f"{sub} 가 {SC_MAT.name} 에 없음 (dwi_qc==pass 만 수록)"
    w = np.asarray(rows[0]["SC_weight"], np.float64)
    l = np.asarray(rows[0]["SC_length"], np.float64)
    assert w.shape == l.shape and w.shape[0] == w.shape[1] and w.sum() > 0
    return w, l


class ROIPairSubject:
    def __init__(self, sub: str, roi_pairs_dir: Path = ROI_PAIRS_DIR, cache: Path = CACHE):
        self.sub = sub
        self.dir = Path(roi_pairs_dir) / sub
        self.cache = Path(cache)
        self.synthetic_dir = ROOT / "outputs" / "synthetic"
        assert (self.dir / "assignments.npz").exists(), f"{self.dir}/assignments.npz 없음 (scripts/02)"
        assert (self.dir / "bundles.npz").exists(), f"{self.dir}/bundles.npz 없음 (scripts/03)"

    # --- lazy 로드 -----------------------------------------------------------
    @cached_property
    def _asg(self):
        z = dict(np.load(self.dir / "assignments.npz"))      # NpzFile 은 키 접근마다 재해제 -> 한 번에 메모리로
        assert int(z["n_total"]) > 0 and z["sc_end"].sum() > 0
        return z

    @cached_property
    def _bnd(self):
        # NpzFile 은 z["streamlines"] 를 부를 때마다 100MB 전체를 다시 압축 해제한다.
        # pair 마다 get_pair 를 부르면 step 당 수십 초가 여기서 샌다. 한 번에 메모리로 올린다.
        z = dict(np.load(self.dir / "bundles.npz"))
        s, off = z["streamlines"], z["pair_offsets"]
        assert s.ndim == 3 and s.shape[1:] == (128, 3), s.shape
        assert off[0] == 0 and off[-1] == len(s) and (np.diff(off) > 0).all()
        assert (np.diff(z["pair_index"]) >= 0).all(), "pair_index 가 정렬되어 있지 않음"
        assert np.isfinite(s.astype(np.float32)).all()
        return z

    @cached_property
    def _visit(self):
        """scripts/24 가 만든 GT ROI-visitation (없으면 None). route loss 용."""
        p = self.dir / "visit.npz"
        if not p.exists():
            return None
        z = dict(np.load(p))
        assert int(z["n_streamlines"]) == len(self._bnd["streamlines"]), f"{p}: bundles.npz 와 개수 불일치"
        return z

    def visitation(self, k: int) -> torch.Tensor:
        """pair k 의 streamline 별 통과 ROI multi-hot [n, R] float32."""
        z = self._visit
        assert z is not None, f"{self.sub}: visit.npz 없음 (scripts/24_build_visitation.py)"
        a, b = int(self._bnd["pair_offsets"][k]), int(self._bnd["pair_offsets"][k + 1])
        v = np.unpackbits(z["visit_packed"][a:b], axis=1, count=int(z["n_roi"]))
        return torch.from_numpy(v.astype(np.float32))

    @property
    def pair_marginal(self) -> np.ndarray:
        """pair 별 통과 ROI 비율 [K, R] (생성 pass 의 route target)."""
        z = self._visit
        assert z is not None, f"{self.sub}: visit.npz 없음 (scripts/24_build_visitation.py)"
        return z["pair_marginal"]

    @property
    def has_visitation(self) -> bool:
        return (self.dir / "visit.npz").exists()

    @cached_property
    def _seg(self):
        """scripts/26 이 만든 SC edge-aligned segment bundle (없으면 None)."""
        p = self.dir / "edge_segments.npz"
        if not p.exists():
            return None
        z = dict(np.load(p))
        assert z["segments"].ndim == 3 and z["offsets"][-1] == len(z["segments"]), p
        return z

    @property
    def has_edge_segments(self) -> bool:
        return (self.dir / "edge_segments.npz").exists()

    @property
    def edge_pair_ids(self) -> np.ndarray:
        return self._seg["pair_ids"]

    @property
    def edge_count_full(self) -> np.ndarray:
        """edge 별 분해 segment 수 (cap 전). 학습 노출 균형용이며 GT SC 값이 아니다 (§18)."""
        return self._seg["count_full"]

    def edge_segments(self, e: int) -> torch.Tensor:
        """edge e 의 segment [m, n_points, 3] float32."""
        z = self._seg
        assert z is not None, f"{self.sub}: edge_segments.npz 없음 (scripts/26_build_edge_segments.py)"
        a, b = int(z["offsets"][e]), int(z["offsets"][e + 1])
        return torch.from_numpy(z["segments"][a:b].astype(np.float32))

    def edge_segment_lengths(self, e: int) -> torch.Tensor:
        z = self._seg
        a, b = int(z["offsets"][e]), int(z["offsets"][e + 1])
        return torch.from_numpy(z["lengths"][a:b].astype(np.float32))

    @cached_property
    def _syn(self):
        """scripts/22 가 만든 synthetic.npz (없으면 None). pair 별 offsets 로 잘라 쓴다."""
        p = self.synthetic_dir / self.sub / "synthetic.npz"
        if not p.exists():
            return None
        z = dict(np.load(p))
        assert z["streamlines"].shape[1:] == (128, 3) and len(z["pair_offsets"]) == self.n_pairs + 1, p
        return z

    def synthetic_pair(self, k: int) -> torch.Tensor:
        """pair k 의 synthetic streamlines [m,128,3] float32 (없으면 [0,128,3])."""
        z = self._syn
        if z is None:
            return torch.zeros(0, 128, 3)
        a, b = int(z["pair_offsets"][k]), int(z["pair_offsets"][k + 1])
        return torch.from_numpy(z["streamlines"][a:b].astype(np.float32))

    @property
    def n_roi(self) -> int:
        return int(self._asg["n_roi"])

    @property
    def sc_end(self) -> np.ndarray: return self._asg["sc_end"]
    @property
    def len_end(self) -> np.ndarray: return self._asg["len_end"]
    @property
    def sc_pass(self) -> np.ndarray: return self._asg["sc_pass"]
    @property
    def len_pass(self) -> np.ndarray: return self._asg["len_pass"]

    @cached_property
    def _mat(self):
        return load_mat_gt(self.sub)

    @property
    def sc_mat(self) -> np.ndarray: return self._mat[0]
    @property
    def len_mat(self) -> np.ndarray: return self._mat[1]

    @property
    def pair_ids(self) -> np.ndarray: return self._bnd["pair_ids"]
    @property
    def pair_count_full(self) -> np.ndarray: return self._bnd["pair_count_full"]
    @property
    def n_pairs(self) -> int: return len(self.pair_ids)

    @cached_property
    def pair_strength(self) -> np.ndarray:
        """bundle pair 의 GT edge 강도 = .mat pass-SC 값 (0 이면 endpoint count 로 대체)."""
        p = np.asarray(self.pair_ids, np.int64)
        s = self.sc_mat[p[:, 0], p[:, 1]].astype(np.float64)
        z = s <= 0
        s[z] = self.pair_count_full[z]
        return s

    @cached_property
    def pair_tier(self) -> np.ndarray:
        """0 small / 1 mid / 2 large (roi_groups.TIER_EDGES)."""
        return tier_of_strength(self.pair_strength)

    @cached_property
    def pair_block(self) -> np.ndarray:
        """0 ctx-ctx / 1 ctx-sub / 2 sub-sub."""
        return block_of_pairs(self.pair_ids)

    @property
    def positive_ratio(self) -> float:
        R = self.n_roi
        return self.n_pairs / (R * (R - 1) / 2)

    @cached_property
    def t1_w(self):
        """W 격자 T1 (정규화 전). 없으면 None (scripts/01 또는 data.prepare_t1)."""
        p = self.cache / f"{self.sub}_T1w_syn_W.npy"
        if not p.exists():
            return None
        v = np.load(p)
        assert v.shape == (193, 229, 193) and np.isfinite(v).all() and v.max() > 0
        return v

    # --- 접근 ----------------------------------------------------------------
    def get_pair(self, k: int):
        """pair k 의 (streamlines [n,128,3] float32, lengths [n] float32) torch tensor."""
        z = self._bnd
        a, b = int(z["pair_offsets"][k]), int(z["pair_offsets"][k + 1])
        assert b > a, f"pair {k} 가 비어 있음"
        s = torch.from_numpy(z["streamlines"][a:b].astype(np.float32))
        L = torch.from_numpy(z["lengths"][a:b].astype(np.float32))
        return s, L

    def gt_for(self, mode: str):
        """모드에 맞는 (SC, length) target. 섞어 쓰지 않도록 한 곳에서만 고른다."""
        if mode == "endpoint":
            return self.sc_end.astype(np.float32), self.len_end.astype(np.float32)
        if mode == "pass":
            return self.sc_mat.astype(np.float32), self.len_mat.astype(np.float32)
        raise ValueError(mode)

    def sample_pairs(self, n_pairs: int, rng: np.random.Generator, weighted: bool = True,
                     mode: str | None = None):
        """양성 pair 인덱스 [n_pairs].
        mode 'log'     : log1p(count) 비례 (강한 edge 를 더 자주)      — weighted=True 와 같음
             'uniform' : pair 균등                                     — weighted=False 와 같음
             'tier'    : 소/중/대 강도 구간에 같은 개수씩 (구간 안은 균등). 큰 bundle 편향 방지."""
        K = self.n_pairs
        mode = mode or ("log" if weighted else "uniform")
        if mode == "log":
            p = np.log1p(self.pair_count_full.astype(np.float64)); p /= p.sum()
            return rng.choice(K, size=n_pairs, replace=n_pairs > K, p=p)
        if mode == "uniform":
            return rng.choice(K, size=n_pairs, replace=n_pairs > K)
        if mode == "tier":
            return balanced_choice(self.pair_tier, n_pairs, rng)
        raise ValueError(mode)

    def negative_pairs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """sc_end == 0 인 canonical (a,b) [n,2]. 양성/음성 불균형 점검용 (pipeline §10, §34)."""
        R = self.n_roi
        iu = np.triu_indices(R, 1)
        neg = np.stack(iu, 1)[self.sc_end[iu] == 0]
        assert len(neg) > 0, "음성 pair 가 없음 (SC 가 완전 연결)"
        return neg[rng.choice(len(neg), size=n, replace=n > len(neg))].astype(np.int16)
