#!/usr/bin/env python
"""W2-b: 조건부 prior p(z|pair) 사다리 오르기 — 잠재공간에서만 채점 (GPU 불필요).

  python scripts/48_prior_ladder.py

무엇을 하나
-----------
W1-e 가 오라클 사다리를 이미 재 두었다 (`outputs/eval/w1e_latent_modality.json:k_ladder`,
precision_ratio 낮을수록 좋음):

  N(mu_pair, I) 36.6  ->  pair별 대각 가우시안 7.32  ->  mix2 5.08  ->  mix3 4.27  ->  mix5 3.36

그 사다리는 sklearn 으로 적합한 **오라클**이다. 이 스크립트는 같은 held-out 분할·같은 지표로
`models/latent_prior.py` 의 수학(= 실제 모델이 쓸 수학)이 그 값에 도달하는지 확인하고,
"학습으로 그 값을 실현할 수 있는가" 를 갈라 본다.

후보 (전부 같은 절반으로 적합, 나머지 절반으로 채점 — 46번 스크립트와 동일한 rng seed)
-------------------------------------------------------------------------------------
  current          N(mu_pair, I)                       현행. 36.6 을 재현해야 프로토콜이 맞다.
  kl_fit_sigma     N(mu_pair, s_kl^2 I)                **KL 손실이 실제로 몰고 갈 분산.**
                   s_kl^2 = E[var_q] + E[(z-mu_pair)^2] (KL(q||p) 의 p 에 대한 모멘트 정합).
                   posterior 가 부분 붕괴해 있으면 이 값이 1 근처라 아무것도 안 변한다.
  s1_var_fixed_mu  N(mu_pair, diag)                    평균은 그대로, 분산만 학습했을 때의 상한.
  s1_full          N(mu_fit, diag)                     1단계 목표 (오라클 gauss 7.32).
  s2_mixK          K-혼합 (kmeans++ 초기화 + Adam)      2단계 목표 (mix2 5.08 / mix5 3.36).
  arch_additive_*  pair 임베딩이 표현 가능한 함수족만으로 **처음 보는 pair** 에 일반화
                   (prior_mu(Emb(a)+Emb(b)) 는 곧 mu = c_a + c_b + const 인 가법 모형이다).

부수 측정
---------
  posterior_resolution  가닥별 posterior 반경(잡음) 대 mu 퍼짐/모드 간격.
                        decoder 가 학습 중 본 z 는 mu + 0.81*eps 다. 모드 간격이 그 잡음보다
                        작으면 잠재지표가 좋아져도 생성이 따라오지 않는다 — 이득의 상한.
  aggregate_posterior   z_gt 를 posterior 평균 대신 **샘플**로 바꿔 같은 사다리를 다시 잰다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc.evaluation import prior_metrics as pm                      # noqa: E402
from atm_sc.models import latent_prior as lp                           # noqa: E402
from atm_sc.models.roi_pair_embedding import ROIPairEmbedding, canonical_pairs  # noqa: E402

CACHE_NPZ = ROOT / "outputs/eval/w1e_latent_cache.npz"
W1E_JSON = ROOT / "outputs/eval/w1e_latent_modality.json"
OUT_JSON = ROOT / "outputs/eval/w2b_prior_ladder.json"
MKEYS = ("nll", "nll_per_dim", "mmd_mmd2", "mmd_p", "precision_ratio", "recall_ratio", "d_gt_self")


# ------------------------------------------------------------------ 필수 자기검증
def selfcheck(latent_dim: int = 64, n_roi: int = 82) -> dict:
    """log_sigma 0-init 이 기존 샘플링과 bit-exact 같은지. 실패하면 여기서 멈춘다."""
    torch.manual_seed(0)
    e = ROIPairEmbedding(n_roi, 64, 512, latent_dim=latent_dim)
    with torch.no_grad():                       # 0-init 인 prior_mu 를 흔들어 mu != 0 으로
        e.prior_mu.weight.normal_(std=0.2); e.prior_mu.bias.normal_(std=0.2)
    p = canonical_pairs(torch.randint(0, n_roi, (64, 2)))
    mu, ls = e.prior_params(p)
    assert ls.shape == mu.shape == (64, latent_dim), (mu.shape, ls.shape)
    assert torch.equal(ls, torch.zeros_like(ls)), "log_sigma 가 0-init 이 아니다"
    assert torch.equal(mu, e.prior_mean(p)), "prior_params 의 mu 가 기존 prior_mean 과 다르다"
    ga = torch.Generator().manual_seed(7); gb = torch.Generator().manual_seed(7)
    old = e.prior_mean(p) + torch.randn(64, latent_dim, generator=ga)      # roi_atm.sample_z
    new = e.sample_prior(p, generator=gb)
    assert torch.equal(old, new), "0-init 인데 샘플이 bit-exact 하지 않다"
    assert torch.isfinite(new).all()
    # 혼합 헤드 K=1 은 1단계와 같은 분포여야 한다.
    m1 = lp.PairMixturePrior(64, latent_dim, k=1)
    v, z = e.pair_vec(p), mu + 0.3
    d = float((m1.log_prob(z, v, mu, ls) - lp.diag_log_prob(z, mu, ls)).abs().max())
    assert d < 1e-4, f"K=1 혼합이 대각 가우시안과 다르다 ({d})"
    # KL 등가성: prior 분산이 I 면 새 KL 식이 기존 식과 같아야 한다 (아래 patch 제안의 근거).
    mu_q, lv_q = torch.randn(32, latent_dim), torch.randn(32, latent_dim) * 0.5
    mu_p = torch.randn(32, latent_dim)
    old_kl = (-0.5 * (1 + lv_q - (mu_q - mu_p).pow(2) - lv_q.exp()).sum(1)).mean()
    lv_p = torch.zeros_like(lv_q)
    new_kl = (0.5 * (lv_p - lv_q + (lv_q - lv_p).exp()
                     + (mu_q - mu_p).pow(2) * (-lv_p).exp() - 1.0).sum(1)).mean()
    assert float((old_kl - new_kl).abs()) < 1e-3, (float(old_kl), float(new_kl))
    return {"bit_exact_zero_init": True, "prior_mean_unchanged": True,
            "mix_k1_equals_diag": True, "kl_patch_matches_at_unit_var": True,
            "kl_old": float(old_kl), "kl_new_logvar_prior0": float(new_kl)}


# ------------------------------------------------------------------------ 데이터
def conditions(cache) -> list[dict]:
    z, lv, off = cache["z"], cache["logvar"], cache["offsets"]
    mu_p, pid = cache["prior_mu"], cache["pair_ids"]
    assert z.shape[1] == 64 and np.isfinite(z).all(), (z.shape, "NaN/Inf")
    out = []
    for k in range(len(off) - 1):
        s, e = int(off[k]), int(off[k + 1])
        zk = torch.as_tensor(z[s:e].astype(np.float32))
        n = zk.shape[0]
        # ** 46번과 같은 seed/분할 ** -> 채점 절반이 글자 그대로 같아 값이 직접 비교된다.
        perm = np.random.default_rng(2000 + k).permutation(n)
        out.append({"k": k, "n": n, "z": zk,
                    "fit": torch.as_tensor(perm[: n // 2].copy()),
                    "ev": torch.as_tensor(perm[n // 2:].copy()),
                    "mu": torch.as_tensor(mu_p[k].astype(np.float32)),
                    "var_q": torch.as_tensor(np.exp(lv[s:e].astype(np.float32))),
                    "roi": (int(pid[k, 0]), int(pid[k, 1]))})
    return out


def agg(rows: list[dict]) -> dict:
    return {k: float(np.median([r[k] for r in rows])) for k in MKEYS} | {
        "frac_mmd_significant": float(np.mean([r["mmd_p"] <= 0.05 for r in rows])),
        "n_pairs": len(rows), "n_eval_median": float(np.median([r["n_gt"] for r in rows]))}


# ---------------------------------------------------------------- 가법 ROI 모형 (구조 한계)
def additive_design(rois, n_roi: int) -> np.ndarray:
    """prior_mu(Emb(a)+Emb(b)) = (W E)_a + (W E)_b + b  <- 정확히 이 가법 모형이다."""
    X = np.zeros((len(rois), n_roi + 1), np.float64)
    for i, (a, b) in enumerate(rois):
        X[i, a] += 1.0; X[i, b] += 1.0; X[i, -1] = 1.0
    return X


def ridge_fit(X, Y, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[-1, -1] -= lam                                    # intercept 는 벌하지 않는다
    return np.linalg.solve(A, X.T @ Y)


def arch_additive(conds, n_roi: int, n_fold: int, lams, mix_k=(), mix_steps: int = 400,
                  mix_lr: float = 0.05, seed: int = 0) -> dict:
    """가법 ROI 모형으로 **처음 보는 pair** 의 mu / log_sigma 를 예측 (조건 단위 교차검증)."""
    rois = [c["roi"] for c in conds]
    X = additive_design(rois, n_roi)
    Y = np.stack([c["z"][c["fit"]].mean(0).numpy() for c in conds]).astype(np.float64)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(conds)) % n_fold
    best, cv = None, {}
    for lam in lams:
        err = 0.0
        for f in range(n_fold):
            tr, te = fold != f, fold == f
            err += float(((X[te] @ ridge_fit(X[tr], Y[tr], lam) - Y[te]) ** 2).sum())
        cv[str(lam)] = err / Y.size
        if best is None or cv[str(lam)] < cv[str(best)]:
            best = lam
    base = sum(float(((Y[fold == f] - Y[fold != f].mean(0)) ** 2).sum()) for f in range(n_fold))
    pred = np.zeros_like(Y)
    ls_glob = np.zeros((len(conds), 64), np.float64)
    book: dict[int, list] = {}
    for f in range(n_fold):
        tr, te = fold != f, fold == f
        pred[te] = X[te] @ ridge_fit(X[tr], Y[tr], best)
        # 분산·다봉 구조도 pair 를 넘어 일반화해야 한다. 조건 안 잔차를 모아 **전 pair 공통**
        # 대각분산(1단계)과 K-혼합 codebook(2단계)을 만든다 -- 둘 다 학습으로 실현 가능한 형태.
        r = np.concatenate([conds[i]["z"][conds[i]["fit"]].numpy() - Y[i] for i in np.where(tr)[0]])
        ls_glob[te] = 0.5 * np.log(np.maximum((r ** 2).mean(0), 1e-12))
        for kk in mix_k:
            rr = torch.as_tensor(r[np.random.default_rng(3000 + f).choice(
                len(r), min(len(r), 4000), replace=False)].astype(np.float32))
            book.setdefault(kk, [None] * len(conds))
            mm, ll, lg = lp.fit_mixture(rr, kk, steps=mix_steps, lr=mix_lr, seed=0)
            for i in np.where(te)[0]:
                book[kk][i] = (mm, ll, lg)
    return {"pred_mu": torch.as_tensor(pred.astype(np.float32)),
            "pred_log_sigma": torch.as_tensor(ls_glob.astype(np.float32)),
            "codebook": book,
            "lam": float(best), "cv_mse": cv, "n_fold": n_fold,
            "cv_r2": 1.0 - float(((pred - Y) ** 2).sum()) / base,
            "mu_pair_r2": None,
            "mean_offset": float(np.linalg.norm(pred - Y, axis=1).mean())}


# --------------------------------------------------------------------------- 후보
def build(c, a, extra) -> dict:
    """조건 하나에 대한 후보 prior 들. 전부 fit 절반만 본다."""
    zf, mu = c["z"][c["fit"]], c["mu"]
    out = {"current": pm.current_prior(mu)}

    # KL(q||p) 를 p 에 대해 최소화하면 p 는 aggregate posterior 의 모멘트로 간다.
    # 평균이 구조상 mu_pair 에 묶여 있으므로 분산은 E[var_q] + E[(z-mu_pair)^2] 로 수렴한다.
    s2 = float(c["var_q"][c["fit"]].mean() + (zf - mu).pow(2).mean())
    out["kl_fit_sigma"] = pm.GaussianPrior(mu, float(np.log(s2)))

    m, ls = lp.fit_diag(zf, mu=mu)
    out["s1_var_fixed_mu"] = lp.as_metric_prior(m, ls)
    out["s1_mean_only"] = pm.GaussianPrior(zf.mean(0), 0.0)     # 평균만 맞추고 분산은 I
    m, ls = lp.fit_diag(zf)
    out["s1_full"] = lp.as_metric_prior(m, ls)

    for kk in a.k_ladder:
        if zf.shape[0] >= 20 * kk:                       # 46번과 같은 성분당 표본 하한
            mm, ll, lg = lp.fit_mixture(zf, kk, steps=a.mix_steps, lr=a.mix_lr, seed=0)
            out[f"s2_mix{kk}"] = lp.as_metric_prior(mm, ll, lg)

    i = c["k"]
    out["arch_additive_mean"] = pm.GaussianPrior(extra["pred_mu"][i], 0.0)
    out["arch_additive"] = lp.as_metric_prior(extra["pred_mu"][i], extra["pred_log_sigma"][i])
    for kk, bk in extra["codebook"].items():
        mm, ll, lg = bk[i]
        out[f"arch_additive_mix{kk}"] = lp.as_metric_prior(extra["pred_mu"][i] + mm, ll, lg)
    return out


def score(conds, a, extra, target="mu") -> dict:
    rows: dict[str, list] = {}
    for c in conds:
        zev = c["z"][c["ev"]]
        if target == "sample":       # decoder 가 학습 중 실제로 본 z (aggregate posterior)
            g = torch.Generator().manual_seed(9000 + c["k"])
            zev = zev + c["var_q"][c["ev"]].sqrt() * torch.randn(zev.shape, generator=g)
        assert zev.ndim == 2 and zev.shape[1] == 64 and torch.isfinite(zev).all()
        for name, p in build(c, a, extra).items():
            r = pm.evaluate_prior(p, zev, n_sample=a.n_prior_sample, seed=c["k"],
                                  n_perm=a.n_perm)
            rows.setdefault(name, []).append(r | {"k": c["k"], "n": c["n"]})
    return rows


def subject_ceiling(conds) -> dict:
    """pair 만 보는 prior 의 **원리적** 상한. 조건 평균을 pair 로 묶어 일원분산분석.

    같은 pair 라도 subject 마다 조건 평균이 다르면 그 몫은 pair 만 보는 어떤 함수로도
    설명할 수 없다 -> 데이터를 아무리 늘려도 남는 평균 오차의 하한이다.
    """
    from collections import defaultdict
    M = np.stack([c["z"].mean(0).numpy() for c in conds]).astype(np.float64)
    g = defaultdict(list)
    for i, c in enumerate(conds):
        g[tuple(sorted(c["roi"]))].append(i)
    rep = {k: v for k, v in g.items() if len(v) >= 2}
    within = sum(float(((M[v] - M[v].mean(0)) ** 2).sum()) for v in rep.values())
    dof = sum(len(v) - 1 for v in rep.values())
    tot = float(((M - M.mean(0)) ** 2).sum()) / len(M)
    mse = within / dof if dof else float("nan")            # subject 성분 (차원 합)
    return {"n_pairs_repeated": len(rep), "n_conditions_repeated": sum(len(v) for v in rep.values()),
            "within_pair_across_subject_var": mse, "total_condition_mean_var": tot,
            "subject_share": mse / tot,
            "irreducible_mean_offset": float(np.sqrt(mse)),
            "note": ("pair 만 보는 prior 는 subject 성분을 원리적으로 못 맞춘다. "
                     "이 오프셋을 pair 안 GT 구름 반경과 비교할 것.")}


# ------------------------------------------------------- posterior 잡음 대 모드 간격
def resolution(conds, a) -> dict:
    out = []
    for c in conds:
        z, zf = c["z"], c["z"][c["fit"]]
        rq = float(c["var_q"].sum(1).sqrt().mean())                  # 가닥별 posterior 반경
        rmu = float(z.var(0, unbiased=True).sum().sqrt())            # pair 안 mu 퍼짐 반경
        d = torch.cdist(z, z).fill_diagonal_(float("inf"))
        nn = float(d.min(1).values.median())
        sep = float("nan")
        if zf.shape[0] >= 20 * a.sep_k:
            mm, _, lg = lp.fit_mixture(zf, a.sep_k, steps=a.mix_steps, lr=a.mix_lr, seed=0)
            keep = mm[torch.softmax(lg, 0) > 0.05]
            if keep.shape[0] >= 2:
                dd = torch.cdist(keep, keep).fill_diagonal_(float("inf"))
                sep = float(dd.min(1).values.median())
        out.append({"posterior_radius": rq, "mu_spread_radius": rmu, "mu_nn_dist": nn,
                    "mode_sep": sep, "sep_over_noise": sep / rq, "n": c["n"]})
    fin = [r for r in out if np.isfinite(r["mode_sep"])]
    return {k: float(np.median([r[k] for r in out])) for k in
            ("posterior_radius", "mu_spread_radius", "mu_nn_dist")} | {
        "mode_sep": float(np.median([r["mode_sep"] for r in fin])) if fin else None,
        "sep_over_noise": float(np.median([r["sep_over_noise"] for r in fin])) if fin else None,
        "n_pairs_with_modes": len(fin),
        "note": ("decoder 는 학습 중 z = mu + sigma_q*eps 를 봤다. 모드 간격이 posterior 반경보다 "
                 "작으면 잠재지표 개선이 생성으로 옮겨간다는 보장이 없다.")}


# ---------------------------------------------------------------------------- main
def main(a):
    assert CACHE_NPZ.exists(), f"latent 캐시 없음: {CACHE_NPZ} (scripts/46 --stage encode)"
    chk = selfcheck()
    print(f"자기검증 통과: {[k for k, v in chk.items() if v is True]}", flush=True)

    cache = dict(np.load(CACHE_NPZ, allow_pickle=False))
    conds = conditions(cache)
    n_roi = int(cache["pair_ids"].max()) + 1
    print(f"조건 {len(conds)}개 (n>={a.power_n} 인 것 {sum(c['n'] >= a.power_n for c in conds)}개), "
          f"n_roi={n_roi}", flush=True)

    extra = arch_additive(conds, n_roi, a.n_fold, [float(x) for x in a.ridge_lam.split(",")],
                          mix_k=a.k_ladder, mix_steps=a.mix_steps, mix_lr=a.mix_lr)
    M = np.stack([c["z"].mean(0).numpy() for c in conds]).astype(np.float64)
    mu_p = np.stack([c["mu"].numpy() for c in conds]).astype(np.float64)
    extra["mu_pair_r2"] = 1.0 - float(((M - mu_p) ** 2).sum() / ((M - M.mean(0)) ** 2).sum())
    extra["mu_pair_offset"] = float(np.linalg.norm(M - mu_p, axis=1).mean())
    extra["gt_cloud_radius"] = float(np.median(
        [float(c["z"].var(0, unbiased=True).sum().sqrt()) for c in conds]))
    print(f"가법 ROI 모형 (조건 단위 {a.n_fold}-fold CV): lambda={extra['lam']:g}, "
          f"held-out R^2={extra['cv_r2']:.3f}, ||mu_hat - mean_gt||={extra['mean_offset']:.3f}\n"
          f"  현재 prior_mu: R^2={extra['mu_pair_r2']:.3f}, "
          f"||mu_pair - mean_gt||={extra['mu_pair_offset']:.3f}   "
          f"(pair 안 GT 구름 반경 {extra['gt_cloud_radius']:.3f})", flush=True)

    rows = score(conds, a, extra, target="mu")
    ladder = {"all": {n: agg(r) for n, r in rows.items()},
              "powered": {n: agg([x for x in r if x["n"] >= a.power_n])
                          for n, r in rows.items()
                          if sum(x["n"] >= a.power_n for x in r) >= 20}}

    sub = [c for c in conds if c["n"] >= a.power_n]
    sens = {n: agg(r) for n, r in score(sub, a, extra, target="sample").items()}

    w1e = json.loads(W1E_JSON.read_text())
    oracle = w1e["k_ladder"]
    got = ladder["powered"]
    # 프로토콜이 46번과 같은지: 데이터를 안 쓰는 current 는 값이 재현되어야 한다.
    d_cur = abs(got["current"]["precision_ratio"] - oracle["current"]["precision_ratio"])
    assert d_cur < 0.05, (f"current 재현 실패 {got['current']['precision_ratio']:.3f} vs "
                          f"{oracle['current']['precision_ratio']:.3f} -- 분할/지표가 46번과 다르다")

    out = {"selfcheck": chk,
           "ladder": ladder,
           "oracle_k_ladder": oracle,
           "vs_oracle": {m: {"ours": got[o]["precision_ratio"],
                             "oracle": oracle[m]["precision_ratio"],
                             "ratio": got[o]["precision_ratio"] / oracle[m]["precision_ratio"]}
                         for m, o in [("current", "current"), ("gauss", "s1_full"),
                                      ("mix2", "s2_mix2"), ("mix3", "s2_mix3"),
                                      ("mix5", "s2_mix5")] if o in got},
           "arch_additive_fit": {k: extra[k] for k in
                                 ("lam", "cv_mse", "n_fold", "mean_offset", "cv_r2",
                                  "mu_pair_r2", "mu_pair_offset", "gt_cloud_radius")},
           "subject_ceiling": subject_ceiling(conds),
           "posterior_resolution": resolution(conds, a),
           "aggregate_posterior_sensitivity": sens,
           "current_reproduced_delta": d_cur,
           "latent_geometry_w1e": w1e["latent_geometry"],
           "args": vars(a),
           "cache": str(CACHE_NPZ.relative_to(ROOT)),
           "cmd": (f"python scripts/48_prior_ladder.py --k-ladder {a.k_ladder_raw} "
                   f"--mix-steps {a.mix_steps} --mix-lr {a.mix_lr} --n-fold {a.n_fold} "
                   f"--ridge-lam {a.ridge_lam} --n-prior-sample {a.n_prior_sample} "
                   f"--n-perm {a.n_perm} --power-n {a.power_n} --seed {a.seed}")}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)

    def table(d, title):
        print(f"\n=== {title} ===")
        print(f"{'prior':20s} {'NLL':>9s} {'MMD^2':>9s} {'p<=.05':>7s} {'prec':>7s} {'rec':>7s} {'pair':>5s}")
        for k, v in d.items():
            print(f"{k:20s} {v['nll']:9.2f} {v['mmd_mmd2']:9.4f} {v['frac_mmd_significant']:7.2f} "
                  f"{v['precision_ratio']:7.2f} {v['recall_ratio']:7.2f} {v['n_pairs']:5d}")
    table(ladder["powered"], f"사다리 (n>={a.power_n}, 46번과 같은 held-out 절반)")
    table(ladder["all"], "사다리 (전체 280 조건)")
    print("\n=== 오라클 대비 (precision_ratio) ===")
    for m, v in out["vs_oracle"].items():
        print(f"  {m:8s} ours {v['ours']:6.2f}  oracle {v['oracle']:6.2f}  ratio {v['ratio']:.3f}")
    table(sens, "민감도: z_gt 를 posterior 샘플로 (decoder 가 실제로 본 z)")
    print(f"\npair 만 보는 prior 의 상한 {out['subject_ceiling']}")
    print(f"posterior 해상도 {out['posterior_resolution']}")
    print(f"저장: {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-ladder", default="2,3,5")
    ap.add_argument("--mix-steps", type=int, default=400)
    ap.add_argument("--mix-lr", type=float, default=0.05)
    ap.add_argument("--sep-k", type=int, default=5, help="모드 간격 측정에 쓸 성분 수")
    ap.add_argument("--n-fold", type=int, default=5, help="가법 ROI 모형 조건 단위 CV")
    ap.add_argument("--ridge-lam", default="0.1,1,10,100")
    ap.add_argument("--n-prior-sample", type=int, default=256)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--power-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.k_ladder_raw = args.k_ladder
    args.k_ladder = [int(x) for x in args.k_ladder.split(",") if x]
    main(args)
