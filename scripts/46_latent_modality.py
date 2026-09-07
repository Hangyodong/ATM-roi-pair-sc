#!/usr/bin/env python
"""W1-e: 조건부 prior p(z|pair) 의 GT 잠재분포가 단봉인가 다봉인가.

  # 인코딩(GPU) + 분석 한 번에
  python scripts/46_latent_modality.py
  # 캐시가 있으면 GPU 없이 재분석
  python scripts/46_latent_modality.py --stage analyze

배경
----
현재 prior 는 `z ~ N(mu_pair, I)` 다 (`models/roi_pair_embedding.py:prior_mean`,
`models/roi_atm.py:sample_z`). 단봉·등방·분산 고정이고 anatomy 를 받지도 않는다.
이 prior 로 생성하면 pair dice 0.053, 오라클 posterior latent 를 주면 0.167 (천장 0.704).

가설: 한 ROI 쌍의 GT 다발이 여러 갈래(다봉)라서, 단봉 가우시안의 평균 근처에서 뽑으면
어느 갈래도 아닌 가운데가 나온다. 이 스크립트는 그 가설을 **검정**한다.

무엇을 재는가
-------------
(subject, pair) 마다 GT 다발을 인코더에 통과시켜 posterior 평균 mu [n, 64] 를 얻고
(= 오라클 posterior, `38_decoder_capacity.py` 의 `encode_streamlines` 경로와 동일),
그 집합이 단봉인지 다봉인지를 **세 가지 독립적인 검정**으로 판정한다.

  1. GMM BIC/AIC   PCA 상위 d 차원에서 K=1..5 full-cov 혼합을 적합해 BIC 최소 K.
  2. Hartigan dip  PC1 투영 위의 dip test (귀무분포=균등, 단봉성의 최악 경우).
  3. silhouette    K=2..5 kmeans 실루엣을, **같은 n·같은 공분산의 가우시안**에서 나온
                   실루엣 분포와 비교하는 모수 부트스트랩. 방향 탐색까지 귀무에 포함되므로
                   "데이터에서 고른 축" 편향이 없다.

판정: 3개 중 2개 이상이 다봉이라고 하면 그 pair 를 다봉으로 센다.

대조군 (없으면 결과를 믿을 수 없다)
-----------------------------------
실제 pair 하나하나에 대해 **같은 n, 같은 공분산**의 합성 데이터를 만들어 같은 검정을 건다.

  uni_gauss    진짜 단봉 (가우시안)              -> 다봉 판정률이 낮아야 한다 (거짓 양성률)
  uni_t3       단봉이지만 두꺼운 꼬리 (t, df=3)   -> 꼬리를 모드로 오인하지 않는지
  bi_pc1_4sd   명백한 2봉 (PC1 방향 4 sigma 분리) -> 다봉 판정률이 높아야 한다 (검출력)
  bi_pc1_2sd   약한 2봉 (2 sigma)                -> 검출 한계 눈금
  bi_rand_4sd  무작위 64차원 방향 4 sigma 분리    -> **사각지대 기록용** (검출력 0, 아래 참고)

실측 latent 는 유효차원이 ~3 (PC1 이 분산의 54 %, 상위 4개가 89 %) 이다. 그래서 무작위
방향으로 벌린 2봉은 지배적 축들에 묻혀 어떤 검정으로도 보이지 않는다. 실제 다발이 두
갈래라면 갈래 간 분산이 그 방향을 상위 주성분으로 밀어올리므로 `bi_pc1_*` 이 옳은 대조군이고,
`bi_rand_4sd` 는 "이 검정이 못 보는 것" 을 명시적으로 남기기 위한 것이다.

부수 산출: `evaluation/prior_metrics.py` 로 현재 prior N(mu_pair, I) 의 잠재지표 기준값.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc.data.roi_groups import BLOCKS, TIERS                       # noqa: E402
from atm_sc.evaluation import prior_metrics as pm                      # noqa: E402

CACHE_NPZ = ROOT / "outputs/eval/w1e_latent_cache.npz"
TESTS_NPZ = ROOT / "outputs/eval/w1e_modality_tests.json"
OUT_JSON = ROOT / "outputs/eval/w1e_latent_modality.json"
LOCK = ROOT / "outputs/gpu.lock"


# --------------------------------------------------------------------------- GPU lock
def acquire_lock(lock: Path, stale_sec: int = 900, wait_sec: int = 0) -> bool:
    """`scripts/19_train_pipeline.py` 와 같은 규약. 살아 있는 보유자가 있으면 False."""
    me = f"{socket.gethostname()}:{os.getpid()}"
    t0 = time.time()
    while True:
        if lock.exists():
            try:
                host, pid = lock.read_text().strip().split(":")
            except ValueError:
                host, pid = "?", "0"
            alive = False
            if host == socket.gethostname():
                try:
                    os.kill(int(pid), 0); alive = True
                except (OSError, ValueError):
                    alive = False
            fresh = time.time() - lock.stat().st_mtime < stale_sec
            if alive or (host != socket.gethostname() and fresh):
                if time.time() - t0 < wait_sec:
                    time.sleep(10); continue
                print(f"lock 보유 중: {host}:{pid} -> 종료", flush=True)
                return False
            print(f"stale lock 인수: {host}:{pid}", flush=True)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(me)
        return True


def release_lock(lock: Path):
    if lock.exists() and lock.read_text().strip() == f"{socket.gethostname()}:{os.getpid()}":
        lock.unlink()


# --------------------------------------------------------------------------- 인코딩
def select_pairs(subj, n_pairs: int, min_n: int, rng) -> tuple[np.ndarray, dict]:
    """tier x block 9칸을 라운드로빈으로 채워 pair index 를 고른다. 부족한 칸은 건너뛴다."""
    off = subj._bnd["pair_offsets"]
    cnt = np.diff(off)
    ok = cnt >= min_n
    tier, block = subj.pair_tier, subj.pair_block
    cells = {}
    for t in range(3):
        for b in range(3):
            idx = np.flatnonzero(ok & (tier == t) & (block == b))
            if len(idx):
                cells[(t, b)] = list(rng.permutation(idx))
    picked = []
    order = sorted(cells, key=lambda c: len(cells[c]))       # 희소한 칸부터 -> 반드시 포함
    while len(picked) < n_pairs and any(cells.values()):
        for c in order:
            if not cells[c]:
                continue
            picked.append(cells[c].pop())
            if len(picked) >= n_pairs:
                break
    stat = {"n_pairs_total": int(len(cnt)), "n_pairs_ge_min_n": int(ok.sum()),
            "n_excluded_too_few": int((~ok).sum()),
            "cells_available": {f"{TIERS[t]}|{BLOCKS[b]}": len(v) for (t, b), v in cells.items()}}
    return np.asarray(sorted(picked), np.int64), stat


def encode(a) -> dict:
    import torch
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.training.run import CACHE, T1_SOURCES, t1_input

    assert acquire_lock(LOCK, wait_sec=a.lock_wait), "GPU lock 획득 실패"
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
        t1_src = sd.get("t1_source", "syn")
        suffix = T1_SOURCES[t1_src][0]
        subs = [s.strip() for s in open(ROOT / a.split) if s.strip()]
        subs = [s for s in subs if (CACHE / f"{s}{suffix}").exists()][:a.n_subj]
        assert len(subs) >= 3, f"T1 캐시가 있는 subject 3명 이상 필요 (now {len(subs)})"
        print(f"ckpt={a.ckpt} t1_source={t1_src} dev={dev} subjects={len(subs)}", flush=True)

        rng = np.random.default_rng(a.seed)
        Z, LV, MU, offs, meta, stats = [], [], [], [0], [], []
        for si, sub in enumerate(subs):
            subj = ROIPairSubject(sub)
            ks, st = select_pairs(subj, a.pairs_per_subject, a.min_n, rng)
            st["subject"] = sub
            stats.append(st)
            with torch.no_grad():
                feat = m.atm.encode_anatomy(t1_input(m, sub, t1_src))
                assert feat.shape == (1, 512) and torch.isfinite(feat).all(), feat.shape
                for k in ks:
                    S, _ = subj.get_pair(int(k))
                    S = S.to(dev)
                    pid = np.asarray(subj.pair_ids[k], np.int64)[None]
                    pt = torch.as_tensor(pid, device=dev).repeat(S.shape[0], 1)
                    cond = m.condition(feat, pt)
                    mu, logvar = m.encode_streamlines(S, cond)
                    mu_p = m.prior_mean(torch.as_tensor(pid, device=dev))[0]
                    mu, logvar = mu.float().cpu().numpy(), logvar.float().cpu().numpy()
                    # 필수 assert: shape / NaN / 전부 0
                    assert mu.shape == (S.shape[0], 64), mu.shape
                    assert np.isfinite(mu).all() and np.isfinite(logvar).all(), f"{sub} pair {k}: NaN/Inf"
                    assert np.abs(mu).max() > 0, f"{sub} pair {k}: latent 이 전부 0"
                    Z.append(mu); LV.append(logvar); MU.append(mu_p.float().cpu().numpy())
                    offs.append(offs[-1] + mu.shape[0])
                    meta.append((si, int(subj.pair_ids[k][0]), int(subj.pair_ids[k][1]),
                                 int(subj.pair_tier[k]), int(subj.pair_block[k]),
                                 int(subj.pair_count_full[k])))
            print(f"  {sub}: pair {len(ks)}개 / 가닥 {offs[-1]:,} 누적", flush=True)
    finally:
        release_lock(LOCK)

    meta = np.asarray(meta, np.int64)
    out = {"z": np.concatenate(Z).astype(np.float32), "logvar": np.concatenate(LV).astype(np.float32),
           "prior_mu": np.stack(MU).astype(np.float32), "offsets": np.asarray(offs, np.int64),
           "subject_idx": meta[:, 0], "pair_ids": meta[:, 1:3].astype(np.int16),
           "tier": meta[:, 3].astype(np.int8), "block": meta[:, 4].astype(np.int8),
           "pair_count_full": meta[:, 5], "subjects": np.asarray(subs),
           "ckpt": np.asarray(a.ckpt), "t1_source": np.asarray(t1_src),
           "encode_args": np.asarray(json.dumps(
               {k: getattr(a, k) for k in ("ckpt", "split", "n_subj", "pairs_per_subject",
                                           "min_n", "seed")}, ensure_ascii=False)),
           "select_stats": np.asarray(json.dumps(stats, ensure_ascii=False))}
    assert out["z"].shape == (out["offsets"][-1], 64), out["z"].shape
    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_NPZ, **out)
    print(f"캐시 저장: {CACHE_NPZ}  z{out['z'].shape}  pair {len(out['prior_mu'])}개", flush=True)
    return out


# --------------------------------------------------------------------------- 다봉성 검정
def _kmeans(X, k, rng, iters=30):
    """작은 배열용 Lloyd + kmeans++ (sklearn 호출 오버헤드를 피한다. 부트스트랩이 수만 번 돈다)."""
    n = X.shape[0]
    c = X[rng.integers(n)][None]
    for _ in range(k - 1):
        d = ((X[:, None] - c[None]) ** 2).sum(-1).min(1)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        c = np.vstack([c, X[rng.choice(n, p=p)]])
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        d = ((X[:, None] - c[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = X[lab == j].mean(0)
    return lab


def _silhouette(D, lab, k):
    """사전계산된 거리행렬 D 로 평균 실루엣. 빈 군집이 생기면 -1 (무효)."""
    n = len(lab)
    sizes = np.bincount(lab, minlength=k)
    if (sizes < 2).any():
        return -1.0
    M = np.zeros((n, k))
    for j in range(k):
        M[:, j] = D[:, lab == j].sum(1)
    own = M[np.arange(n), lab] / (sizes[lab] - 1)
    other = M / np.maximum(sizes[None], 1)
    other[np.arange(n), lab] = np.inf
    b = other.min(1)
    return float(np.mean((b - own) / np.maximum(own, b)))


def _best_silhouette(X, rng, max_k=5):
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    best, bk = -1.0, 1
    for k in range(2, max_k + 1):
        s = _silhouette(D, _kmeans(X, k, rng), k)
        if s > best:
            best, bk = s, k
    return best, bk


def _pca(z, d):
    zc = z - z.mean(0)
    U, S, _ = np.linalg.svd(zc, full_matrices=False)
    return (U * S)[:, :d], zc


def modality_tests(z, seed=0, boot=100, max_k=5, sil_max_k=5) -> dict:
    """z [n,64] 가 단봉인지 다봉인지. 세 검정 + 판정."""
    import diptest
    z = np.asarray(z, np.float64)
    n, D = z.shape
    assert n >= 4 and D == 64, (n, D)
    rng = np.random.default_rng(seed)
    d = int(min(4, max(2, n // 25), min(n - 1, D)))
    X, zc = _pca(z, d)
    ev = np.linalg.svd(zc, compute_uv=False) ** 2
    var_captured = float(ev[:d].sum() / max(ev.sum(), 1e-30))
    eff_dim = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))
    X = X / (X.std() + 1e-12)                     # 전체 스칼라 한 개로만 정규화 (성분 간 비율 보존)

    # (1) GMM BIC/AIC
    from sklearn.mixture import GaussianMixture
    bic, aic = [], []
    for k in range(1, max_k + 1):
        if n <= k * (d + d * (d + 1) / 2) + 2:            # 파라미터가 표본보다 많으면 무의미
            bic.append(np.inf); aic.append(np.inf); continue
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=5,
                            random_state=seed).fit(X)
        bic.append(float(g.bic(X))); aic.append(float(g.aic(X)))
    k_bic, k_aic = int(np.argmin(bic)) + 1, int(np.argmin(aic)) + 1
    # argmin 은 상한(max_k)에 붙기 쉽다. "BIC 개선의 90 % 를 달성하는 최소 K" 가 더 안정적인
    # '성분 몇 개짜리 구조인가' 의 요약이다.
    drop = bic[0] - min(bic)
    k_bic90 = next((i + 1 for i, v in enumerate(bic) if drop <= 0 or (bic[0] - v) >= 0.9 * drop), 1)

    # (2) Hartigan dip (PC1)
    dip, p_dip = diptest.diptest(np.ascontiguousarray(X[:, 0]))

    # (3) 실루엣 모수 부트스트랩 (귀무 = 같은 n / 같은 공분산의 가우시안)
    sil, k_sil = _best_silhouette(X, rng, sil_max_k)
    null = np.empty(boot)
    for b in range(boot):
        # G @ zc / sqrt(n-1) 은 정확히 경험 공분산을 갖는 가우시안이다 (64차원 그대로).
        zn = rng.standard_normal((n, n)) @ zc / np.sqrt(n - 1)
        Xn, _ = _pca(zn, d)
        Xn = Xn / (Xn.std() + 1e-12)
        null[b] = _best_silhouette(Xn, rng, sil_max_k)[0]
    p_sil = float((null >= sil).sum() + 1) / (boot + 1)

    votes = {"gmm": k_bic >= 2, "dip": p_dip < 0.05, "sil": p_sil < 0.05}
    return {"n": n, "pca_dim": d, "var_captured": var_captured, "eff_dim": eff_dim,
            "k_bic": k_bic, "k_aic": k_aic, "k_bic90": k_bic90,
            "k_bic_at_ceiling": bool(k_bic == max_k),
            "bic_drop_1_to_2": float(bic[0] - bic[1]) if len(bic) > 1 else 0.0,
            "bic": bic, "delta_bic_1_minus_best": float(bic[0] - min(bic)),
            "dip": float(dip), "p_dip": float(p_dip),
            "silhouette": float(sil), "k_sil": int(k_sil), "p_sil": p_sil,
            "sil_null_q95": float(np.quantile(null, 0.95)),
            "votes": votes, "n_votes": int(sum(votes.values())),
            "multimodal": bool(sum(votes.values()) >= 2)}


# --------------------------------------------------------------------------- 합성 대조군
def synth_like(z, kind, rng) -> np.ndarray:
    """실제 pair 와 **같은 n·같은 공분산**의 합성 데이터. 검정력을 실제와 맞춘다.

    2봉 대조군의 분리 방향이 중요하다. 실측 latent 는 유효차원 ~3 (PC1 이 분산의 54 %,
    상위 4개가 89 %) 이라 **무작위 64차원 방향**으로 4 sigma 를 벌리면 그 방향의 분산 자체가
    지배적 축들보다 훨씬 작아 어떤 검정으로도 안 보인다 (실측 검출력 0.00). 실제로 다발이
    두 갈래면 갈래 간 분산이 그 방향의 분산을 지배하므로 그 방향이 상위 주성분이 된다.
    -> 주 대조군은 **PC1 방향 분리**, 무작위 방향은 사각지대를 기록하는 용도로만 남긴다.
    """
    z = np.asarray(z, np.float64)
    n, D = z.shape
    zc = z - z.mean(0)
    base = rng.standard_normal((n, n)) @ zc / np.sqrt(n - 1)      # 경험 공분산 가우시안
    if kind == "uni_gauss":
        return base
    if kind == "uni_t3":                                          # 두꺼운 꼬리 (df=3), 단봉
        s = np.sqrt(3.0 / rng.chisquare(3, size=(n, 1)))
        return base * s
    if kind.startswith("bi_"):
        _, axis, sd = kind.split("_")                             # "bi_pc1_4sd"
        sd = float(sd[:-2])
        if axis == "rand":
            u = rng.standard_normal(D)
        else:
            u = np.linalg.svd(zc, full_matrices=False)[2][int(axis[2:]) - 1]
        u = u / np.linalg.norm(u)
        sigma_u = float(np.std(base @ u))
        sign = np.where(rng.random(n) < 0.5, -0.5, 0.5)
        return base + (sd * sigma_u) * sign[:, None] * u[None]
    raise ValueError(kind)


# --------------------------------------------------------------------------- 요약
def _summarize(rows, key="multimodal") -> dict:
    if not rows:
        return {"n": 0}
    ks = np.asarray([r["k_bic"] for r in rows])
    return {"n": len(rows), "multimodal_frac": float(np.mean([r[key] for r in rows])),
            "vote_frac": {v: float(np.mean([r["votes"][v] for r in rows])) for v in ("gmm", "dip", "sil")},
            "k_bic_hist": {str(k): int((ks == k).sum()) for k in range(1, int(ks.max()) + 1)},
            "k_bic_median": float(np.median(ks)),
            "k_bic_at_ceiling_frac": float(np.mean([r["k_bic_at_ceiling"] for r in rows])),
            "k_bic90_median": float(np.median([r["k_bic90"] for r in rows])),
            "k_bic90_hist": {str(k): int(sum(r["k_bic90"] == k for r in rows))
                             for k in sorted({r["k_bic90"] for r in rows})},
            "median_var_captured": float(np.median([r["var_captured"] for r in rows])),
            "median_eff_dim": float(np.median([r["eff_dim"] for r in rows])),
            "median_p_dip": float(np.median([r["p_dip"] for r in rows])),
            "median_p_sil": float(np.median([r["p_sil"] for r in rows])),
            "median_silhouette": float(np.median([r["silhouette"] for r in rows])),
            "median_n": float(np.median([r["n"] for r in rows]))}


def _nbin(r) -> str:
    return "n<100" if r["n"] < 100 else ("100<=n<200" if r["n"] < 200 else "n>=200")


def _group(rows, keyfn) -> dict:
    g = {}
    for r in rows:
        g.setdefault(keyfn(r), []).append(r)
    return {k: _summarize(v) for k, v in sorted(g.items())}


# --------------------------------------------------------------------------- 분석
def analyze(a, cache) -> dict:
    import torch
    z_all, off = cache["z"], cache["offsets"]
    tier, block, mu_p = cache["tier"], cache["block"], cache["prior_mu"]
    pids = cache["pair_ids"]
    K = len(off) - 1
    assert z_all.shape[1] == 64 and np.isfinite(z_all).all() and np.abs(z_all).max() > 0
    assert K >= 50, f"pair 가 부족하다 ({K} < 50) -- --pairs-per-subject / --n-subj 를 올려라"
    print(f"pair {K}개, latent {z_all.shape}", flush=True)

    sets = ["real"] + list(a.controls.split(","))
    sig = {"n_pairs": K, "boot": a.boot, "max_k": a.max_k, "sil_max_k": a.sil_max_k,
           "controls": a.controls, "z_crc": int(zlib.crc32(z_all.tobytes()))}
    if TESTS_NPZ.exists():
        old = json.load(open(TESTS_NPZ))
        if old.get("sig") == sig:
            print(f"검정 결과 캐시 재사용: {TESTS_NPZ}", flush=True)
            return _analyze_from(a, cache, old["res"])
    res = {s: [] for s in sets}
    t0 = time.time()
    for k in range(K):
        z = z_all[off[k]:off[k + 1]].astype(np.float64)
        for s in sets:
            # (pair, 종류) 마다 결정론적 독립 seed -> --controls 목록을 바꿔도 각 결과가 그대로
            # 재현된다. str.__hash__ 는 프로세스마다 달라지므로 쓰지 않는다.
            zz = z if s == "real" else synth_like(
                z, s, np.random.default_rng([k, zlib.crc32(s.encode())]))
            r = modality_tests(zz, seed=k, boot=a.boot, max_k=a.max_k, sil_max_k=a.sil_max_k)
            r.update(tier=int(tier[k]), block=int(block[k]), pair=k)
            res[s].append(r)
        if (k + 1) % 20 == 0:
            print(f"  검정 {k+1}/{K}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"sig": sig, "res": res}, open(TESTS_NPZ, "w"))
    return _analyze_from(a, cache, res)


def _analyze_from(a, cache, res) -> dict:
    """검정 결과(res)로부터 요약 + prior 기준값. 검정을 다시 돌리지 않는다."""
    import torch
    z_all, off = cache["z"], cache["offsets"]
    tier, block, mu_p = cache["tier"], cache["block"], cache["prior_mu"]
    pids = cache["pair_ids"]
    K = len(off) - 1
    sets = list(res)
    real = res["real"]
    modality = {"overall": _summarize(real),
                "by_tier": _group(real, lambda r: TIERS[r["tier"]]),
                "by_block": _group(real, lambda r: BLOCKS[r["block"]]),
                "by_n": _group(real, _nbin)}
    controls = {s: _summarize(res[s]) | {"by_n": _group(res[s], _nbin)} for s in sets if s != "real"}

    # 검정력은 n 에 의존한다. 판정은 **대조군으로 검정력을 확인한 부분집합**(n >= power_n)에서
    # 내리고, 대조군 assert 도 같은 부분집합에서 건다. by_n 표에 구간별 검정력이 그대로 남는다.
    powered = [r for r in real if r["n"] >= a.power_n]
    use_powered = len(powered) >= 20
    modality["powered_subset"] = _summarize(powered) | {"power_n": a.power_n, "used_for_verdict": use_powered}
    def _sub(name):
        rows = [r for r in res[name] if r["n"] >= a.power_n]
        return float(np.mean([r["multimodal"] for r in rows])) if rows else None
    fp = _sub("uni_gauss") if "uni_gauss" in controls else None
    tp = _sub(a.power_control) if a.power_control in controls else None

    # ---- 현재 prior 의 잠재지표 기준값 -------------------------------------
    # 오라클(그 pair 의 GT 로 적합한 prior)은 **절반으로 적합하고 나머지 절반으로 채점**한다.
    # 같은 데이터로 적합하고 채점하면 성분이 많을수록 무조건 좋아 보여(퇴화) 상한이 무의미해진다.
    # 현재 prior 와 N(0,I) 는 데이터를 쓰지 않으므로 같은 평가 절반에서 그대로 비교된다.
    from collections import defaultdict
    base = defaultdict(list)
    geo = []
    for k in range(K):
        zf = z_all[off[k]:off[k + 1]].astype(np.float32)
        n_k = len(zf)
        perm = np.random.default_rng(2000 + k).permutation(n_k)
        z_fit = torch.as_tensor(zf[perm[: n_k // 2]])
        z_ev = torch.as_tensor(zf[perm[n_k // 2:]])
        z = torch.as_tensor(zf)
        mu = torch.as_tensor(mu_p[k])
        kb = int(np.clip(real[k]["k_bic"], 1, max(2, len(z_fit) // 30)))   # 표본 대비 성분 수 제한
        reg = 1e-3 * float(z_fit.var(0).mean())      # GT sd ~ 0.18 -> 절대값 1e-6 은 너무 작다
        priors = {"current_N(mu_pair,I)": pm.current_prior(mu),
                  "N(0,I)": pm.GaussianPrior(torch.zeros(64), 0.0),
                  "oracle_gauss_fit": pm.fit_gaussian(z_fit),
                  "oracle_mixture_k2": pm.fit_mixture(z_fit, 2, seed=0, reg=reg),
                  "oracle_mixture_kbic": (pm.fit_mixture(z_fit, kb, seed=0, reg=reg) if kb >= 2
                                          else pm.fit_gaussian(z_fit))}
        # 성분을 늘리면 held-out 에서도 계속 좋아지는가 (= 구조가 2봉보다 풍부한가).
        # 표본이 넉넉한 pair 에서만, 같은 적합/평가 분할로 K 사다리를 만든다 (성분당 >=20 표본).
        if n_k >= a.power_n:
            priors["ladder_current"] = pm.current_prior(mu)
            priors["ladder_gauss"] = pm.fit_gaussian(z_fit)
            for kk in [int(x) for x in a.k_ladder.split(",") if x]:
                if len(z_fit) >= 20 * kk:
                    priors[f"ladder_mix{kk}"] = pm.fit_mixture(z_fit, kk, seed=0, reg=reg)
        for name, p in priors.items():
            base[name].append(pm.evaluate_prior(p, z_ev, n_sample=a.n_prior_sample, seed=k,
                                                n_perm=a.n_perm) | {"k": kb if "kbic" in name else None})

        sd = z.std(0)
        geo.append({"gt_sd_mean": float(sd.mean()), "gt_sd_max": float(sd.max()),
                    "posterior_sd_mean": float(torch.exp(0.5 * torch.as_tensor(
                        cache["logvar"][off[k]:off[k + 1]])).mean()),
                    "mean_offset_from_mu_pair": float((z.mean(0) - mu).norm()),
                    "gt_norm_mean": float(z.norm(dim=1).mean()),
                    "eff_dim_diag": float(sd.pow(2).sum() ** 2 / sd.pow(4).sum())})

    def agg(rows, keys):
        return {k: float(np.median([r[k] for r in rows])) for k in keys}

    mkeys = ("nll", "nll_per_dim", "mmd_mmd2", "mmd_p", "mmd_mmd2_over_null_q95",
             "precision_ratio", "recall_ratio", "d_gt_self")
    prior_baseline = {n: agg(rows, mkeys) | {
        "frac_mmd_significant": float(np.mean([r["mmd_p"] <= 0.05 for r in rows])),
        "median_k": (float(np.median([r["k"] for r in rows])) if rows[0]["k"] else None),
        "n_pairs": len(rows),
        "n_eval_median": float(np.median([r["n_gt"] for r in rows]))} for n, rows in base.items()}
    k_ladder = {n[7:]: v for n, v in prior_baseline.items() if n.startswith("ladder_")}
    prior_baseline = {n: v for n, v in prior_baseline.items() if not n.startswith("ladder_")}
    latent_geometry = agg(geo, tuple(geo[0]))

    # ---- 같은 pair 가 subject 마다 다른가 (anatomy 조건화의 근거) -----------
    bykey = {}
    for k in range(K):
        bykey.setdefault((int(pids[k][0]), int(pids[k][1])), []).append(k)
    shared = {p: v for p, v in bykey.items() if len(v) >= 3}
    inter = None
    if shared:
        b_between, b_within = [], []
        for p, ks in shared.items():
            means = np.stack([z_all[off[k]:off[k + 1]].mean(0) for k in ks])
            withins = np.stack([z_all[off[k]:off[k + 1]].var(0) for k in ks]).mean(0)
            b_between.append(means.var(0).sum()); b_within.append(withins.sum())
        b_between, b_within = np.asarray(b_between), np.asarray(b_within)
        inter = {"n_pairs_shared_by_ge3_subjects": len(shared),
                 "between_subject_var": float(np.median(b_between)),
                 "within_bundle_var": float(np.median(b_within)),
                 "icc_between_over_total": float(np.median(b_between / (b_between + b_within)))}

    frac = (modality["powered_subset"] if use_powered else modality["overall"])["multimodal_frac"]
    fpr = fp if fp is not None else 0.0
    # 거짓 양성률을 빼고도 남는가. 단봉 대조군보다 높지 않으면 단봉이다.
    verdict = ("multimodal" if frac >= max(0.5, 3 * fpr) else
               "unimodal" if frac <= max(0.25, 1.5 * fpr) else "mixed")
    return {"modality": modality, "controls": controls, "k_ladder": k_ladder,
            "control_check": {"uni_gauss_fpr": fp, "bi_power": tp, "power_control": a.power_control,
                              "power_n": a.power_n,
                              "verdict_frac": frac, "verdict_basis": "powered_subset" if use_powered else "overall"},
            "prior_baseline": prior_baseline, "latent_geometry": latent_geometry,
            "inter_subject": inter, "verdict": verdict,
            "per_pair": [{k: r[k] for k in ("pair", "n", "tier", "block", "k_bic", "p_dip",
                                            "p_sil", "silhouette", "k_bic90", "var_captured",
                                            "multimodal")} for r in real]}


def main(a):
    cache = None
    if a.stage in ("encode", "both"):
        cache = encode(a)
    if a.stage == "encode":
        return
    if cache is None:
        assert CACHE_NPZ.exists(), f"캐시 없음 ({CACHE_NPZ}) -> --stage encode 먼저"
        cache = dict(np.load(CACHE_NPZ, allow_pickle=False))
    print("지표 자기검증 (합성 데이터)...", flush=True)
    val = pm.validate_metrics(n_perm=a.n_perm)
    print(f"  {len(val['checks'])}개 항목 전부 통과", flush=True)

    out = analyze(a, cache)
    out["metric_validation"] = {"checks": val["checks"], "note": val["note"],
                                "cases": {k: {m: round(v[m], 4) for m in
                                              ("nll", "mmd_mmd2", "mmd_p", "precision_ratio", "recall_ratio")}
                                          for k, v in list(val["cases"].items()) + list(val["lowdim_d4"].items())}}
    out["args"] = vars(a)
    e = json.loads(str(cache["encode_args"])) if "encode_args" in cache else vars(a)
    out["encode_args"] = e
    out["cmd"] = (f"python scripts/46_latent_modality.py --stage encode --ckpt {e['ckpt']} "
                  f"--split {e['split']} --n-subj {e['n_subj']} "
                  f"--pairs-per-subject {e['pairs_per_subject']} --min-n {e['min_n']} "
                  f"--seed {e['seed']}"
                  f"  &&  python scripts/46_latent_modality.py --stage analyze "
                  f"--boot {a.boot} --max-k {a.max_k} --sil-max-k {a.sil_max_k} "
                  f"--controls {a.controls} --power-n {a.power_n}")
    out["cache"] = str(CACHE_NPZ.relative_to(ROOT))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)

    # 대조군 검증은 **저장 뒤**에 건다: 검정이 못 미더워도 계산 결과는 남긴다.
    cc = out["control_check"]
    if cc["uni_gauss_fpr"] is not None and cc["bi_power"] is not None and cc["verdict_basis"] == "powered_subset":
        assert cc["uni_gauss_fpr"] < 0.25, \
            f"단봉 대조군 거짓 양성률이 너무 높다 ({cc['uni_gauss_fpr']:.2f}) -- 검정을 믿을 수 없다"
        assert cc["bi_power"] > 0.70, \
            f"2봉 대조군({a.power_control}) 검출력이 너무 낮다 ({cc['bi_power']:.2f}) -- 검정을 믿을 수 없다"

    m = out["modality"]["overall"]
    ps = out["modality"]["powered_subset"]
    print(f"\n=== 다봉성 ({m['n']} pair) ===")
    print(f"검정력 확인 구간 n>={a.power_n}: {ps['n']} pair, 다봉률 {ps['multimodal_frac']:.3f} "
          f"(판정 근거: {out['control_check']['verdict_basis']})")
    print(f"다봉 판정 비율 {m['multimodal_frac']:.3f}   K_bic 중앙값 {m['k_bic_median']:.0f}  분포 {m['k_bic_hist']}")
    print(f"검정별 양성률 {m['vote_frac']}")
    print(f"{'대조군':12s} {'다봉률':>7s} {'K_bic 중앙':>10s}  검정별")
    for k, v in out["controls"].items():
        print(f"{k:12s} {v['multimodal_frac']:7.3f} {v['k_bic_median']:10.0f}  {v['vote_frac']}")
    print(f"\ntier {{k: v['multimodal_frac'] for ...}} = "
          f"{{{', '.join(f'{k}: {v['multimodal_frac']:.2f}' for k, v in out['modality']['by_tier'].items())}}}")
    print(f"block = {{{', '.join(f'{k}: {v['multimodal_frac']:.2f}' for k, v in out['modality']['by_block'].items())}}}")
    print(f"\n=== 현재 prior 기준값 (중앙값) ===")
    print(f"{'prior':24s} {'NLL':>9s} {'MMD^2':>9s} {'p':>6s} {'prec':>6s} {'rec':>6s}")
    for k, v in out["prior_baseline"].items():
        print(f"{k:24s} {v['nll']:9.2f} {v['mmd_mmd2']:9.4f} {v['mmd_p']:6.3f} "
              f"{v['precision_ratio']:6.2f} {v['recall_ratio']:6.2f}")
    print(f"\n=== 오라클 K 사다리 (n>={a.power_n}, 같은 held-out 절반) ===")
    print(f"{'prior':10s} {'NLL':>9s} {'MMD^2':>9s} {'p<=.05':>7s} {'prec':>6s} {'rec':>6s} {'pair':>5s}")
    for k, v in out["k_ladder"].items():
        print(f"{k:10s} {v['nll']:9.2f} {v['mmd_mmd2']:9.4f} {v['frac_mmd_significant']:7.2f} "
              f"{v['precision_ratio']:6.2f} {v['recall_ratio']:6.2f} {v['n_pairs']:5d}")
    print(f"\nlatent 기하 {out['latent_geometry']}")
    print(f"subject 간 {out['inter_subject']}")
    print(f"\n판정: {out['verdict']}")
    print(f"저장: {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["encode", "analyze", "both"])
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt")
    ap.add_argument("--split", default="outputs/splits/val.txt")
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--pairs-per-subject", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=40, help="이보다 가닥이 적은 pair 는 검정에서 제외")
    ap.add_argument("--boot", type=int, default=100, help="실루엣 모수 부트스트랩 반복")
    ap.add_argument("--max-k", type=int, default=8, help="GMM 성분 상한")
    ap.add_argument("--sil-max-k", type=int, default=5, help="실루엣/부트스트랩 군집 상한 (비용 지배)")
    ap.add_argument("--n-prior-sample", type=int, default=256)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--controls",
                    default="uni_gauss,uni_t3,bi_pc1_4sd,bi_pc1_2sd,bi_rand_4sd")
    ap.add_argument("--power-control", default="bi_pc1_4sd",
                    help="검출력 assert 에 쓰는 대조군 (bi_rand_* 는 사각지대라 제외)")
    ap.add_argument("--k-ladder", default="2,3,5,8",
                    help="held-out 오라클 혼합 성분 사다리 (n >= power_n 인 pair 에서만)")
    ap.add_argument("--power-n", type=int, default=200,
                    help="이 이상의 n 을 가진 pair 에서만 최종 판정 (64차원 검정력 한계, 대조군으로 확인)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lock-wait", type=int, default=600, help="GPU lock 대기 초")
    main(ap.parse_args())
