"""ROI-pair 별 latent bank (추론용 사전분포 대체).

왜 필요한가: 생성 시 z ~ N(mu_pair, I) 에서 뽑으면 디코더가 실제 다발이 없는 자리를 그린다.
측정(test 3명, 균등 배분): prior 0.669 / bank 0.726, pair 별 복셀 dice 0.053 -> 0.137
(실제 streamline 두 표본끼리의 천장은 0.597). train subject 의 실제 streamline 을 encoder 로
통과시켜 pair 별로 모아 두고 그 주변에서 KDE 샘플링하면 다발이 놓인 자리에 머문다.

test subject 의 streamline 은 쓰지 않는다 (T1 만 입력). bank 는 train 전용이라 모든 subject 에
같은 것이 쓰이며, 따라서 개인차를 담지는 못한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class LatentBank:
    """pair -> latent 무더기. 전부 벡터화되어 있어 46만 가닥도 한 번에 뽑는다."""

    def __init__(self, Z, offsets, bw, slot, n_roi):
        self.Z, self.offsets, self.bw, self.slot, self.n_roi = Z, offsets, bw, slot, int(n_roi)
        assert Z.ndim == 2 and offsets.ndim == 1 and bw.shape == (len(offsets) - 1, Z.shape[1])
        assert offsets[-1] == len(Z), (offsets[-1], len(Z))
        self.dim = Z.shape[1]

    @property
    def n_pairs(self) -> int:
        return len(self.offsets) - 1

    def sample(self, pairs: np.ndarray, rng: np.random.Generator):
        """pairs [K,2] -> (z [K,D] float32, hit [K] bool). hit=False 면 호출자가 prior 로 채운다."""
        p = np.asarray(pairs, np.int64)
        i, j = np.minimum(p[:, 0], p[:, 1]), np.maximum(p[:, 0], p[:, 1])
        bi = self.slot[i * self.n_roi + j]
        hit = bi >= 0
        z = np.zeros((len(p), self.dim), np.float32)
        if not hit.any():
            return z, hit
        b = bi[hit]
        cnt = (self.offsets[b + 1] - self.offsets[b]).astype(np.int64)
        idx = self.offsets[b] + (rng.random(len(b)) * cnt).astype(np.int64)
        z[hit] = self.Z[idx] + self.bw[b] * rng.standard_normal((len(b), self.dim)).astype(np.float32)
        return z, hit

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), Z=self.Z, offsets=self.offsets,
                            bw=self.bw, slot=self.slot, n_roi=self.n_roi)
        return path.with_suffix(".npz")

    @staticmethod
    def load(path) -> "LatentBank":
        d = np.load(path)
        b = LatentBank(d["Z"], d["offsets"], d["bw"], d["slot"], int(d["n_roi"]))
        assert b.n_pairs > 0 and np.isfinite(b.Z).all(), "latent bank 가 비었거나 NaN"
        return b


@torch.no_grad()
def build_bank(model, subjects, bundle_path, n_per_pair: int = 24, batch: int = 20000,
               seed: int = 0, verbose: bool = True) -> LatentBank:
    """train subject 의 실제 streamline 을 encode 해 pair 별 latent 무더기를 만든다.

    bundle_path(sub) -> bundles.npz 경로, anatomy 는 각 subject 의 것을 쓴다
    (z 는 그 subject 의 condition 기준이지만, 디코딩은 대상 subject 의 condition 으로 한다).
    """
    from ..training.run import t1_input
    acc: dict[tuple[int, int], list] = {}
    rng = np.random.default_rng(seed)
    for s in subjects:
        b = np.load(bundle_path(s))
        pairs, off, SS = b["pair_ids"].astype(np.int64), b["pair_offsets"], b["streamlines"]
        feat = model.atm.encode_anatomy(t1_input(model, s))
        idx, pr = [], []
        for k in range(len(pairs)):
            avail = int(off[k + 1] - off[k])
            n = min(n_per_pair, avail)
            if n <= 0:
                continue
            idx.append(off[k] + rng.choice(avail, n, replace=False))
            pr.append(np.repeat(pairs[k][None], n, axis=0))
        assert idx, f"{s}: bundles.npz 에 streamline 이 없음"
        idx, pr = np.concatenate(idx), np.concatenate(pr)
        for i in range(0, len(idx), batch):
            S = torch.as_tensor(SS[idx[i:i + batch]].astype(np.float32), device=model.device)
            P = torch.as_tensor(pr[i:i + batch], device=model.device)
            mu, _ = model.encode_streamlines(S, model.condition(feat, P))
            mu = mu.cpu().numpy().astype(np.float32)
            for t, (a_, b_) in enumerate(pr[i:i + batch]):
                acc.setdefault((int(min(a_, b_)), int(max(a_, b_))), []).append(mu[t])
        if verbose:
            print(f"  bank += {s}  ({len(acc)} pair)", flush=True)

    n_roi = int(model.n_roi)
    keys = sorted(acc)
    Zs, offs, bws = [], [0], []
    for k in keys:
        z = np.stack(acc[k])
        Zs.append(z); offs.append(offs[-1] + len(z))
        # Silverman: 차원마다 std * n^(-1/(D+4))
        bws.append(z.std(0) * len(z) ** (-1.0 / (z.shape[1] + 4)))
    Z = np.concatenate(Zs).astype(np.float32)
    slot = np.full(n_roi * n_roi, -1, np.int32)
    for b_, (i, j) in enumerate(keys):
        slot[i * n_roi + j] = b_
    bank = LatentBank(Z, np.array(offs, np.int64), np.stack(bws).astype(np.float32), slot, n_roi)
    assert np.isfinite(Z).all(), "latent 에 NaN/Inf"
    if verbose:
        print(f"latent bank: {bank.n_pairs} pair, {len(Z):,} latent, D={bank.dim}", flush=True)
    return bank
