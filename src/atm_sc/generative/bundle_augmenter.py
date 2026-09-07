"""GESTA 식 offline augmentation (전략 §21–27, §40 Stage A, §72).

    TRAIN GT bundle → ATM ES(mu) → latent seeds → latent_sampler(KDE/rejection) → ATM DS(anatomy + pair 조건)
    → T1-only filter → outputs/synthetic/<sub>/synthetic.npz

원칙: GT SC 는 절대 건드리지 않는다 (§31–32). synthetic 은 recon(geometry) 노출량만 바꾼다.
seed 가 min_seed_count 미만인 pair 는 per-subject KDE 대신 TRAIN subject 들의 같은 pair pooled latent bank
(§19-C, §20: train 만) 를 seed 로 쓰고, 그것도 없으면 pair prior N(mu_pair, I) (§19-B) 로 만든 뒤 같은 filter 를 통과시킨다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.balanced_pair_sampler import BalanceConfig, exposure_targets
from ..data.bundle_statistics import size_bin_of
from ..data.roi_groups import BLOCKS, block_of_pairs
from ..filtering.qc_thresholds import DEFAULT_PATH as QC_PATH, QCThresholds
from ..filtering.t1_streamline_filter import FilterConfig, filter_streamlines
from .latent_sampler import sample_latents

ROOT = Path(__file__).resolve().parents[3]
SOURCE = {"kde": 0, "bank": 1, "mixed": 2, "prior": 3}
SIZE_BINS = ("low", "mid", "high")


@dataclass
class AugmentConfig:
    balance: BalanceConfig = field(default_factory=lambda: BalanceConfig(enabled=True))
    min_seed_count: int = 20             # §18: 이 미만이면 per-subject KDE 불안정 -> bank/prior
    max_synthetic_ratio: float = 4.0     # §38, §67: pair 당 synthetic <= ratio x N_real
    over_generate: float = 3.0           # filter 탈락 대비 여유 생성 (실측 통과율: GT 0.97, 생성물은 endpoint 기준으로 갈림)
    method: str = "kde"                  # 'kde' | 'gaussian' | 'rejection'
    bandwidth: float | str = "silverman"
    bw_factor: float = 1.0
    bank_cap_per_pair: int = 512         # pair 당 bank 최대 latent 수 (subject 간 pooled)
    bank_per_subject: int = 32           # subject 당 pair 마다 인코딩할 real streamline 수
    chunk: int = 4096
    seed: int = 0
    # GESTA QC 문서 §5-6: 임계값은 임의 상수가 아니라 TRAIN real 분포 분위수에서 (scripts/27)
    qc_thresholds: str | None = str(QC_PATH)
    eligibility_quantile: float = 25.0   # TRAIN edge 크기 분포의 하위 이 분위수 미만만 증강 대상 (§5)
    filter: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self):
        if self.qc_thresholds and Path(self.qc_thresholds).exists():
            th = QCThresholds.load(Path(self.qc_thresholds))
            self.filter = th.to_filter_config()
            self.filter.max_winding_deg = th.max_winding_deg
            self.filter.min_end_ratio = th.min_end_ratio


# --------------------------------------------------------------------------- ES / DS
@torch.no_grad()
def encode_mu(model, anatomy: torch.Tensor, S: torch.Tensor, pair: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """real streamlines [n,128,3] mm -> posterior mean [n,64] (numpy). 조건 = (subject anatomy, pair)."""
    out = []
    P = torch.as_tensor(np.asarray(pair, np.int64), device=model.device)
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk].to(model.device).float()
        mu, _ = model.encode_streamlines(s, model.condition(anatomy, P.expand(s.shape[0], 2)))
        out.append(mu.float().cpu().numpy())
    z = np.concatenate(out) if out else np.zeros((0, 64), np.float32)
    assert np.isfinite(z).all(), "ES 출력 NaN"
    return z


@torch.no_grad()
def decode_z(model, anatomy: torch.Tensor, z: np.ndarray, pair: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """latent [n,64] -> streamlines mm [n,128,3] float32."""
    out = []
    P = torch.as_tensor(np.asarray(pair, np.int64), device=model.device)
    zt = torch.as_tensor(np.asarray(z, np.float32), device=model.device)
    for i in range(0, zt.shape[0], chunk):
        zc = zt[i:i + chunk]
        out.append(model.decode(zc, model.condition(anatomy, P.expand(zc.shape[0], 2))).float().cpu().numpy())
    S = np.concatenate(out)
    assert S.shape[1:] == (128, 3) and np.isfinite(S).all(), "DS 출력 NaN"
    return S


# --------------------------------------------------------------------------- pooled latent bank (§19-C, §20)
class LatentBank:
    """TRAIN subject 들의 같은 ROI-pair real latent 모음. pair 당 cap 개까지."""

    def __init__(self, pair_ids, offsets, z, subject_idx, subjects):
        self.pair_ids, self.offsets, self.z, self.subject_idx, self.subjects = pair_ids, offsets, z, subject_idx, list(subjects)
        self.index = {tuple(p): k for k, p in enumerate(np.asarray(pair_ids).tolist())}
        assert offsets[0] == 0 and offsets[-1] == len(z)

    def get(self, pair, exclude_subject: str | None = None) -> np.ndarray:
        k = self.index.get(tuple(int(x) for x in pair))
        if k is None:
            return np.zeros((0, self.z.shape[1]), np.float32)
        a, b = int(self.offsets[k]), int(self.offsets[k + 1])
        z, s = self.z[a:b].astype(np.float32), self.subject_idx[a:b]
        if exclude_subject is not None and exclude_subject in self.subjects:
            z = z[s != self.subjects.index(exclude_subject)]
        return z

    @classmethod
    def load(cls, path: Path):
        d = np.load(path, allow_pickle=False)
        return cls(d["pair_ids"], d["offsets"], d["z"], d["subject_idx"], json.loads(str(d["subjects"])))


@torch.no_grad()
def encode_mu_batch(model, anatomy: torch.Tensor, S: torch.Tensor, pairs: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """streamline 마다 다른 pair 조건으로 한 번에 인코딩. pair 별 호출(subject 당 1900회) 대비 ~50배 빠르다."""
    P = torch.as_tensor(np.asarray(pairs, np.int64), device=model.device)
    out = []
    for i in range(0, S.shape[0], chunk):
        s = S[i:i + chunk].to(model.device).float()
        mu, _ = model.encode_streamlines(s, model.condition(anatomy, P[i:i + chunk]))
        out.append(mu.float().cpu().numpy().astype(np.float16))
    z = np.concatenate(out) if out else np.zeros((0, 64), np.float16)
    assert np.isfinite(z.astype(np.float32)).all(), "ES 출력 NaN"
    return z


@torch.no_grad()
def build_latent_bank(model, subjects, anatomy_of, out_path: Path, cfg: AugmentConfig, log=print) -> LatentBank:
    """TRAIN subject 만 (§20). subject 당 pair 마다 최대 per_subject 개를 뽑아 한 번에 인코딩한 뒤
    pair 별로 cap 까지 모은다 (여러 subject 가 골고루 들어가도록)."""
    from ..data.dataset import ROIPairSubject
    rng = np.random.default_rng(cfg.seed)
    pool: dict[tuple, list] = {}
    t0 = time.time()
    for si, sub in enumerate(subjects):
        subj = ROIPairSubject(sub); a = anatomy_of(sub)
        pid = np.asarray(subj.pair_ids, np.int64)
        off = np.asarray(subj._bnd["pair_offsets"], np.int64)
        idx = np.concatenate([rng.choice(np.arange(off[k], off[k + 1]),
                                         size=min(off[k + 1] - off[k], cfg.bank_per_subject), replace=False)
                              for k in range(len(pid))])
        idx.sort()
        S = torch.from_numpy(subj._bnd["streamlines"][idx].astype(np.float32))
        kk = np.searchsorted(off, idx, side="right") - 1                 # streamline -> pair index
        mu = encode_mu_batch(model, a, S, pid[kk], cfg.chunk)
        for k in np.unique(kk):
            pool.setdefault(tuple(pid[k].tolist()), []).append((si, mu[kk == k]))
        del subj
        if (si + 1) % 10 == 0 or si == len(subjects) - 1:
            log(f"[bank] {si + 1}/{len(subjects)} subjects, {len(pool)} pairs, {time.time() - t0:.0f}s")
    pairs = sorted(pool)
    zs, sidx, offsets = [], [], [0]
    for p in pairs:
        z = np.concatenate([m for _, m in pool[p]])
        sj = np.concatenate([np.full(len(m), si, np.int16) for si, m in pool[p]])
        if len(z) > cfg.bank_cap_per_pair:
            sel = np.sort(rng.choice(len(z), cfg.bank_cap_per_pair, replace=False)); z, sj = z[sel], sj[sel]
        zs.append(z); sidx.append(sj); offsets.append(offsets[-1] + len(z))
    pair_ids = np.array(pairs, np.int64); z = np.concatenate(zs); sidx = np.concatenate(sidx)
    offsets = np.array(offsets, np.int64)
    assert len(z) > 0 and np.isfinite(z.astype(np.float32)).all()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pair_ids=pair_ids, offsets=offsets, z=z, subject_idx=sidx, subjects=json.dumps(list(subjects)))
    log(f"[bank] {out_path.name}: {len(pair_ids)} pairs, {len(z):,} latents, {out_path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s")
    return LatentBank(pair_ids, offsets, z, sidx, subjects)


# --------------------------------------------------------------------------- subject 증강
@torch.no_grad()
def augment_subject(model, subject, anatomy, bank, cfg: AugmentConfig, atlas, affine,
                    brain_mask, mask_affine, out_dir: Path, log=print) -> dict:
    pid = np.asarray(subject.pair_ids, np.int64); counts = np.asarray(subject.pair_count_full, np.int64)
    targets = exposure_targets(counts, cfg.balance)
    # §5: 큰 bundle 은 증강 대상이 아니다. TRAIN 분포 하위 분위수 미만만 후보.
    elig_cut = float(np.percentile(counts, cfg.eligibility_quantile)) if cfg.eligibility_quantile else np.inf
    blocks, sbins = block_of_pairs(pid), size_bin_of(counts)
    S_all, k_all, src_all, rows = [], [], [], []
    t0 = time.time()
    for k in range(subject.n_pairs):
        N = int(counts[k])
        need = min(int(np.ceil(targets[k])) - N, int(cfg.max_synthetic_ratio * N))
        row = {"k": k, "pair": pid[k].tolist(), "n_real": N, "target": float(targets[k]), "need": max(need, 0),
               "block": BLOCKS[blocks[k]], "size_bin": SIZE_BINS[sbins[k]]}
        if need <= 0 or N > elig_cut:
            row.update(source="none", n_gen=0, n_accept=0,
                       skip=("target 충족" if need <= 0 else f"크기 상위 (N>{elig_cut:.0f})"))
            rows.append(row); continue
        own = encode_mu(model, anatomy, subject.get_pair(k)[0], pid[k], cfg.chunk) if N >= 2 else np.zeros((0, 64), np.float32)
        pooled = bank.get(pid[k], exclude_subject=subject.sub) if bank is not None else np.zeros((0, 64), np.float32)
        if N >= cfg.min_seed_count:
            seeds, source = own, "kde"
        elif len(pooled) >= cfg.min_seed_count:
            seeds, source = (np.concatenate([own, pooled]) if len(own) else pooled), "bank"
        elif len(own) + len(pooled) >= 2:
            seeds, source = np.concatenate([own, pooled]), "mixed"
        else:
            seeds, source = None, "prior"
        n_gen = int(np.ceil(need * cfg.over_generate))
        if seeds is not None:
            z_new, info = sample_latents(seeds, n_gen, method=cfg.method, bandwidth=cfg.bandwidth,
                                         bw_factor=cfg.bw_factor, seed=cfg.seed * 100003 + k)
        else:                                             # §19-B pair prior
            P = torch.as_tensor(pid[k][None], device=model.device)
            g = torch.Generator(device=model.device); g.manual_seed(cfg.seed * 100003 + k)
            z_new = (model.prior_mean(P) + torch.randn(n_gen, 64, device=model.device, generator=g)).cpu().numpy()
            info = {"acceptance_rate": 1.0, "n_trials": n_gen, "bandwidth": None, "elapsed_sec": 0.0}
        S_new = decode_z(model, anatomy, z_new, pid[k], cfg.chunk)
        keep, fst = filter_streamlines(S_new, np.repeat(pid[k][None], len(S_new), 0), atlas, affine,
                                       brain_mask, mask_affine, cfg.filter)
        acc = S_new[keep][:need]
        row.update(source=source, n_seeds=int(len(seeds)) if seeds is not None else 0, n_gen=n_gen,
                   n_accept=int(len(acc)), sampler_acceptance=float(info["acceptance_rate"]),
                   n_trials=int(info["n_trials"]),
                   bandwidth=(None if info.get("bandwidth") is None else float(info["bandwidth"])),
                   filter_pass=float(fst["pass_rate"]),
                   # §21: 실패 사유별 통과 수를 남긴다 (낮은 통과율의 원인을 나중에 추적하려면 필수)
                   filter={c: float(fst[c]) for c in ("finite", "endpoint", "length", "curvature",
                                                      "winding", "brain", "dedup") if c in fst},
                   n_pass={c: int(round(fst[c] * n_gen)) for c in ("endpoint", "length", "curvature",
                                                                   "winding", "brain") if c in fst})
        rows.append(row)
        if len(acc):
            S_all.append(acc.astype(np.float16)); k_all.append(np.full(len(acc), k, np.int32))
            src_all.append(np.full(len(acc), SOURCE[source], np.int8))
    if S_all:
        S = np.concatenate(S_all); kk = np.concatenate(k_all); src = np.concatenate(src_all)
        order = np.argsort(kk, kind="stable"); S, kk, src = S[order], kk[order], src[order]
    else:
        S = np.zeros((0, 128, 3), np.float16); kk = np.zeros(0, np.int32); src = np.zeros(0, np.int8)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(kk, minlength=subject.n_pairs))]).astype(np.int64)
    assert offsets[-1] == len(S) and len(offsets) == subject.n_pairs + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "synthetic.tmp.npz"          # 학습이 읽는 중에 재생성해도 깨지지 않게 원자적 교체
                                                 # (np.savez 는 .npz 로 끝나지 않으면 확장자를 붙인다)
    np.savez_compressed(tmp, streamlines=S, pair_index=kk, pair_offsets=offsets,
                        source=src, pair_ids=pid, cfg=json.dumps(asdict(cfg), default=str))
    tmp.replace(out_dir / "synthetic.npz")
    gen = [r for r in rows if r["n_gen"] > 0]
    need_total = int(sum(r["need"] for r in rows))
    summary = {"subject": subject.sub, "n_pairs": int(subject.n_pairs), "n_real": int(counts.sum()),
               "n_synthetic": int(len(S)), "n_pairs_augmented": int(sum(r["n_accept"] > 0 for r in rows)),
               "need_total": need_total, "fill_rate": float(len(S) / max(need_total, 1)),
               "by_source": {s: int(sum(r["n_accept"] for r in rows if r.get("source") == s)) for s in SOURCE},
               "by_size_bin": {b: int(sum(r["n_accept"] for r in rows if r["size_bin"] == b)) for b in SIZE_BINS},
               "by_block": {b: int(sum(r["n_accept"] for r in rows if r["block"] == b)) for b in BLOCKS},
               "mean_sampler_acceptance": float(np.mean([r["sampler_acceptance"] for r in gen])) if gen else None,
               "mean_filter_pass": float(np.mean([r["filter_pass"] for r in gen])) if gen else None,
               "eligibility_cut": float(elig_cut),
               "n_pairs_eligible": int(sum(1 for r in rows if r.get("n_gen", 0) > 0)),
               "filter_pass_by_criterion": ({c: float(np.mean([r["filter"][c] for r in gen if c in r["filter"]]))
                                             for c in ("endpoint", "length", "curvature", "winding", "brain", "dedup")}
                                            if gen else None),
               "qc_thresholds": (asdict(QCThresholds.load(Path(cfg.qc_thresholds)))
                                 if cfg.qc_thresholds and Path(cfg.qc_thresholds).exists() else None),
               "elapsed_sec": time.time() - t0, "pairs": rows}
    (out_dir / "augment_stats.json").write_text(json.dumps(summary, ensure_ascii=False))
    log(f"[augment] {subject.sub}: real {summary['n_real']:,} + synth {len(S):,} "
        f"({summary['n_pairs_augmented']} pairs, fill {summary['fill_rate']:.2f}, "
        f"acc {summary['mean_sampler_acceptance']}, filter {summary['mean_filter_pass']}, {summary['elapsed_sec']:.0f}s)")
    return summary
