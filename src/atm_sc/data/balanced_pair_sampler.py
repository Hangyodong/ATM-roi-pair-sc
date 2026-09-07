"""ROI-pair 균형 batch sampler (GESTA 전략 §16–17, §35–38, §77–78).

sampling unit 은 streamline 이 아니라 ROI pair 다. pair k 의 학습 노출 목표
    B_k = clip(B_base · (N_k / B_base)^α, B_min, B_max)          (α < 1: 큰 bundle 을 눌러 count 비례 노출을 막는다)
로 pair 선택 확률 p_k ∝ B_k 를 정하고, 선택된 pair 마다 고정 n 개를 real + synthetic 풀에서 채운다
(real ≥ real_fraction_min, synthetic ≤ max_synthetic_ratio × real, 모자라면 real 복원 추출).
GT SC 는 건드리지 않는다 (§31–32): 여기서 만든 batch 는 recon(ES/DS geometry) 에만 쓰인다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .roi_groups import TIERS


@dataclass
class BalanceConfig:
    enabled: bool = False
    alpha: float = 0.5
    b_base: float = 100.0
    b_min: float = 16.0
    b_max: float = 256.0
    real_fraction_min: float = 0.5
    max_synthetic_ratio: float = 4.0
    lambda_syn: float = 0.5             # synthetic streamline 의 recon 가중치 (§47: GT 와 같은 신뢰도로 쓰지 않는다)
    min_seed_count: int = 20            # 이 미만 pair 는 per-subject KDE 대신 pooled bank/duplicate (augmenter 가 결정)
    synthetic_dir: str = "outputs/synthetic"


def exposure_targets(counts: np.ndarray, cfg: BalanceConfig) -> np.ndarray:
    c = np.asarray(counts, np.float64)
    assert (c >= 1).all()
    return np.clip(cfg.b_base * (c / cfg.b_base) ** cfg.alpha, cfg.b_min, cfg.b_max)


class BalancedPairSampler:
    def __init__(self, subject, cfg: BalanceConfig):
        self.subject, self.cfg = subject, cfg
        self.targets = exposure_targets(subject.pair_count_full, cfg)
        self.p = self.targets / self.targets.sum()
        self.exposure = np.zeros(len(self.targets), np.int64)      # pair 별 누적 노출 (streamline 수)
        self.n_real_used = self.n_synth_used = 0

    def sample_pairs(self, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
        K = len(self.p)
        return rng.choice(K, size=n_pairs, replace=n_pairs > K, p=self.p)

    def _fill(self, k: int, n: int, rng: np.random.Generator, want_visit: bool = False):
        """pair k 에서 n 개: real 우선, synthetic 으로 보충, 그래도 모자라면 real 복원 추출.
        -> (S [n,128,3], V [n,R] | None (real 만; synthetic 은 0 이고 route mask 로 제외), is_syn [n] bool)"""
        real, _ = self.subject.get_pair(k)
        synth = self.subject.synthetic_pair(k)
        n_real_avail, n_syn_avail = real.shape[0], synth.shape[0]
        n_real = min(n_real_avail, max(int(np.ceil(n * self.cfg.real_fraction_min)), n - n_syn_avail))
        n_syn = min(n - n_real, n_syn_avail, int(self.cfg.max_synthetic_ratio * n_real))
        i_real = rng.choice(n_real_avail, n_real, replace=False)
        parts, idx_real, is_syn = [real[torch.as_tensor(i_real)]], [i_real], [np.zeros(n_real, bool)]
        if n_syn > 0:
            parts.append(synth[torch.as_tensor(rng.choice(n_syn_avail, n_syn, replace=False))])
            is_syn.append(np.ones(n_syn, bool))
        rest = n - n_real - n_syn
        if rest > 0:                                                   # duplicate oversampling fallback (§19-A)
            i_rest = rng.choice(n_real_avail, rest, replace=True)
            parts.append(real[torch.as_tensor(i_rest)]); idx_real.append(i_rest); is_syn.append(np.zeros(rest, bool))
        self.n_real_used += n_real + rest; self.n_synth_used += n_syn
        S = torch.cat(parts); syn = np.concatenate(is_syn)
        V = None
        if want_visit:
            vk = self.subject.visitation(k)                            # [n_real_avail, R]
            V = torch.zeros(S.shape[0], vk.shape[1])
            keep = ~syn
            V[torch.as_tensor(keep)] = vk[torch.as_tensor(np.concatenate(idx_real))]
        return S, V, syn, n_real + rest, n_syn

    def sample_batch(self, rng: np.random.Generator, n_pairs: int, n_per_pair: int, want_visit: bool = False):
        """-> (S [n_pairs*n_per_pair,128,3] float32, P [.,2] int64, V [.,R] | None, syn [.] bool, info)"""
        ks = self.sample_pairs(n_pairs, rng)
        S, P, V, SY, n_real, n_syn = [], [], [], [], 0, 0
        pid = np.asarray(self.subject.pair_ids, np.int64)
        for k in ks:
            s, v, syn, r, y = self._fill(int(k), n_per_pair, rng, want_visit)
            S.append(s); P.append(torch.as_tensor(pid[k]).repeat(s.shape[0], 1)); SY.append(syn)
            if v is not None:
                V.append(v)
            n_real += r; n_syn += y
            self.exposure[k] += s.shape[0]
        tiers = np.asarray(self.subject.pair_tier)[ks]
        info = {"recon_real_frac": n_real / (n_real + n_syn), "recon_synth_frac": n_syn / (n_real + n_syn)}
        info.update({f"recon_tier_{t}": float((tiers == i).mean()) for i, t in enumerate(TIERS)})
        return (torch.cat(S).float(), torch.cat(P), (torch.cat(V) if V else None),
                np.concatenate(SY), info)
