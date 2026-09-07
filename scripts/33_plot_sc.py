#!/usr/bin/env python
"""bank + 템플릿 배분으로 생성한 SC 를 GT 와 비교해 그림 4장으로 저장한다.

  python scripts/33_plot_sc.py --subjects sub-101070 sub-101124 sub-101476
결과: outputs/figures/sc_{1_overview,2_template,3_blocks,4_tiers}.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import ATLAS                                # noqa: E402
from atm_sc.data.roi_groups import block_masks, tier_masks         # noqa: E402
from atm_sc.data.tt_io import hard_sc                              # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                       # noqa: E402
from atm_sc.training.run import t1_input                           # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
IU = np.triu_indices(82, 1)
FIG = ROOT / "outputs/figures"


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else np.nan


def ccc(p, g):
    mp, mg = p.mean(), g.mean()
    return float(2 * ((p - mp) * (g - mg)).mean() / (p.var() + g.var() + (mp - mg) ** 2 + 1e-8))


def build_template(subs, key):
    acc = np.zeros((82, 82))
    for s in subs:
        acc += np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")[key]
    return acc / len(subs)


def build_bank(m, subs, n_per_pair, dev):
    bank, rng = {}, np.random.default_rng(0)
    for s in subs:
        b = np.load(ROOT / f"outputs/roi_pairs/{s}/bundles.npz")
        pairs, off, SS = b["pair_ids"].astype(np.int64), b["pair_offsets"], b["streamlines"]
        with torch.no_grad():
            feat = m.atm.encode_anatomy(t1_input(m, s))
        idx, pr = [], []
        for k in range(len(pairs)):
            avail = off[k + 1] - off[k]
            n = min(n_per_pair, avail)
            if n <= 0:
                continue
            idx.append(off[k] + rng.choice(avail, n, replace=False))
            pr.append(np.repeat(pairs[k][None], n, axis=0))
        idx, pr = np.concatenate(idx), np.concatenate(pr)
        for i in range(0, len(idx), 20000):
            S = torch.as_tensor(SS[idx[i:i + 20000]].astype(np.float32), device=dev)
            P = torch.as_tensor(pr[i:i + 20000], device=dev)
            with torch.no_grad():
                mu, _ = m.encode_streamlines(S, m.condition(feat, P))
            mu = mu.cpu().numpy().astype(np.float32)
            for t, (a_, b_) in enumerate(pr[i:i + 20000]):
                bank.setdefault((int(a_), int(b_)), []).append(mu[t])
    return {k: (np.stack(v), np.stack(v).std(0) * len(v) ** (-1.0 / (np.stack(v).shape[1] + 4)))
            for k, v in bank.items()}


@torch.no_grad()
def generate(m, feat, pairs, counts, bank, atlas, affine, dev, D, batch=20000, seed=0):
    rep = np.repeat(np.asarray(pairs, np.int64), np.asarray(counts, np.int64), axis=0)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    rs = np.random.default_rng(seed)
    W = np.zeros((82, 82), np.float64)
    for i in range(0, len(rep), batch):
        blk = rep[i:i + batch]
        P = torch.as_tensor(blk, device=dev)
        z = m.prior_mean(P) + torch.randn(len(blk), D, device=dev, generator=g)
        zs = np.empty((len(blk), D), np.float32); hit = np.zeros(len(blk), bool)
        for t in range(len(blk)):
            e = bank.get((int(blk[t, 0]), int(blk[t, 1])))
            if e is None:
                continue
            v, h = e
            zs[t] = v[rs.integers(0, len(v))] + h * rs.standard_normal(D).astype(np.float32)
            hit[t] = True
        if hit.any():
            z = torch.where(torch.as_tensor(hit, device=dev)[:, None], torch.as_tensor(zs, device=dev), z)
        S = m.decode(z, m.condition(feat, P)).cpu().numpy().astype(np.float32)
        npts = np.full(len(S), S.shape[1], np.int64)
        W += hard_sc(S.reshape(-1, 3), npts, atlas, affine, 82, "pass")[0]
    assert W.sum() > 0, "생성 SC 가 비었음"
    return W


def mat(ax, M, title, vmax):
    im = ax.imshow(np.maximum(M, 0.5), norm=LogNorm(vmin=1, vmax=vmax), cmap="magma", interpolation="nearest")
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(65.5, color="cyan", lw=0.6); ax.axvline(65.5, color="cyan", lw=0.6)
    return im


def scat(ax, p, g, title, note=""):
    ax.loglog(np.maximum(g, 0.5), np.maximum(p, 0.5), ".", ms=1.5, alpha=0.25, color="#1f77b4")
    lo, hi = 1, max(g.max(), p.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("원본 GT SC (가닥 수)"); ax.set_ylabel("생성 SC (가닥 수)")
    ax.set_title(f"{title}\nr = {corr(p, g):.3f}" + (f"   {note}" if note else ""), fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def main(a):
    cache = FIG / "sc_cache.npz"
    if a.replot:                       # 이미 생성한 결과로 그림만 다시 그린다 (GPU 불필요)
        assert cache.exists(), f"{cache} 없음 -- --replot 없이 먼저 실행"
        z = np.load(cache, allow_pickle=False)
        G, P, t_pass = list(z["G"]), list(z["P"]), z["t_pass"]
        a.subjects = [str(x) for x in z["subjects"]]
        print(f"캐시 사용: {a.subjects}", flush=True)
        return draw(a, G, P, t_pass)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    lv = sd.get("unet_level", "full")
    m = ROIPairATM(n_roi=82, trainable="full" if lv == "full" else "vae", unet_level=lv, device=dev)
    m.load_checkpoint(sd["model"]); m.eval()
    img = nib.load(ATLAS); atlas = np.asarray(img.dataobj).astype(np.int16); affine = img.affine
    train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()]

    t_end, t_pass = build_template(train, "sc_end"), build_template(train, "sc_pass")
    bank = build_bank(m, train[:a.n_bank_subj], 24, dev)
    D = next(iter(bank.values()))[0].shape[1]
    keep = t_end[IU] >= 1.0
    pairs = np.stack([IU[0][keep], IU[1][keep]], 1)
    cnt = t_end[pairs[:, 0], pairs[:, 1]]
    cnt = np.maximum(np.round(cnt * a.total / cnt.sum()), 1).astype(np.int64)
    print(f"bank {len(bank)} pair / 생성 pair {len(pairs)} / 총 {cnt.sum():,} 가닥", flush=True)

    G, P = [], []
    for s in a.subjects:
        z = np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")
        with torch.no_grad():
            feat = m.atm.encode_anatomy(t1_input(m, s))
        W = generate(m, feat, pairs, cnt, bank, atlas, affine, dev, D)
        G.append(z["sc_pass"].astype(np.float64)); P.append(W)
        print(f"  {s}: r={corr(W[IU], G[-1][IU]):.3f}", flush=True)
    FIG.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, G=np.stack(G), P=np.stack(P),
                        t_pass=t_pass, subjects=np.array(a.subjects))
    return draw(a, G, P, t_pass)


def draw(a, G, P, t_pass):
    FIG.mkdir(parents=True, exist_ok=True)
    s0, g0, p0 = a.subjects[0], G[0], P[0]
    vmax = max(g0.max(), p0.max())
    gu, pu, tu = g0[IU], p0[IU], t_pass[IU]

    # ---- 1. 개요 --------------------------------------------------------------
    f, ax = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    mat(ax[0], g0, f"원본 GT SC  ({s0})", vmax)
    im = mat(ax[1], p0, "생성 SC  (bank + 템플릿 배분)", vmax)
    f.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02, label="가닥 수 (로그)")
    scat(ax[2], pu, gu, "edge 별 비교", f"CCC = {ccc(pu, gu):.3f}")
    f.suptitle("T1 만 입력해 만든 tractogram 에서 뽑은 SC vs 원본 SC\n"
               f"r = {corr(pu, gu):.3f} : 3,321개 연결의 강약 패턴이 얼마나 같은 방향으로 움직이는가 "
               "(1 이면 완전 일치, 0 이면 무관)", fontsize=12)
    f.savefig(FIG / "sc_1_overview.png", dpi=130, bbox_inches="tight"); plt.close(f)

    # ---- 2. 그룹 템플릿과의 대조 -----------------------------------------------
    f, ax = plt.subplots(1, 4, figsize=(21, 6.0), constrained_layout=True)
    mat(ax[0], g0, f"원본 GT SC ({s0})", vmax)
    mat(ax[1], p0, f"생성 SC   r = {corr(pu, gu):.3f}", vmax)
    im = mat(ax[2], t_pass, f"그룹 평균 템플릿   r = {corr(tu, gu):.3f}", vmax)
    f.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02, label="가닥 수 (로그)")
    rr_p, rr_t = corr(pu - tu, gu - tu), np.nan
    ax[3].plot(gu - tu, pu - tu, ".", ms=1.5, alpha=0.25, color="#d62728")
    ax[3].axhline(0, color="k", lw=0.6); ax[3].axvline(0, color="k", lw=0.6)
    ax[3].set_xlabel("원본 − 템플릿  (이 사람의 개인차)"); ax[3].set_ylabel("생성 − 템플릿")
    ax[3].set_title(f"개인차만 남기면\n잔차 r = {rr_p:.3f}  (0 이면 개인차를 전혀 못 잡음)", fontsize=10)
    f.suptitle("주의: 학습셋 144명의 평균 행렬(T1 을 아예 안 봄)이 생성 SC 보다 원본과 더 닮았다.\n"
               "따라서 이 결과는 '개인 예측' 이 아니라 '평균적 연결 패턴의 재현' 이다.", fontsize=12)
    f.savefig(FIG / "sc_2_template.png", dpi=130, bbox_inches="tight"); plt.close(f)

    # ---- 3. 블록별 -------------------------------------------------------------
    bm = block_masks(82)
    names = {"ctx-ctx": "피질 ↔ 피질", "ctx-sub": "피질 ↔ 피질하", "sub-sub": "피질하 ↔ 피질하"}
    f, ax = plt.subplots(1, 4, figsize=(20, 5.8), constrained_layout=True)
    im = ax[0].imshow(np.select([bm["ctx-ctx"], bm["ctx-sub"], bm["sub-sub"]], [1, 2, 3], 0),
                      cmap="Set2", interpolation="nearest")
    ax[0].set_title("블록 구분\n1 피질↔피질 · 2 피질↔피질하 · 3 피질하↔피질하", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    for k, (key, lab) in enumerate(names.items(), 1):
        mm = bm[key][IU]
        scat(ax[k], pu[mm], gu[mm], f"{lab}  ({int(mm.sum()):,}개 연결)")
    f.suptitle("연결 종류별 재현도 — 피질하 구조가 포함된 연결이 상대적으로 어렵다", fontsize=12)
    f.savefig(FIG / "sc_3_blocks.png", dpi=130, bbox_inches="tight"); plt.close(f)

    # ---- 4. 강도 구간별 + 오차 --------------------------------------------------
    tm = tier_masks(g0)
    labs = {"small": "약한 연결 (<100 가닥)", "mid": "중간 (100–1,000)", "large": "강한 연결 (>1,000)"}
    f, ax = plt.subplots(1, 4, figsize=(20, 5.8), constrained_layout=True)
    for k, (key, lab) in enumerate(labs.items()):
        mm = tm[key][IU]
        scat(ax[k], pu[mm], gu[mm], f"{lab}  ({int(mm.sum()):,}개)")
    d = np.zeros((82, 82)); d[IU] = np.log1p(pu) - np.log1p(gu); d += d.T
    v = np.abs(d).max()
    im = ax[3].imshow(d, cmap="coolwarm", vmin=-v, vmax=v, interpolation="nearest")
    ax[3].axhline(65.5, color="k", lw=0.6); ax[3].axvline(65.5, color="k", lw=0.6)
    ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].set_title("오차 지도  log(생성) − log(원본)\n빨강 = 과다 생성, 파랑 = 과소 생성", fontsize=10)
    f.colorbar(im, ax=ax[3], fraction=0.046)
    f.suptitle("연결 강도 구간별 재현도 — 전체 r 은 강한 연결이 끌어올리고, 약한 연결은 재현이 어렵다",
               fontsize=12)
    f.savefig(FIG / "sc_4_tiers.png", dpi=130, bbox_inches="tight"); plt.close(f)

    print("\n저장:")
    for p in sorted(FIG.glob("sc_*.png")):
        print(f"  {p}")
    print(f"\n{len(a.subjects)}명 평균 r = "
          f"{np.mean([corr(P[i][IU], G[i][IU]) for i in range(len(G))]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt")
    ap.add_argument("--subjects", nargs="+", default=["sub-101070", "sub-101124", "sub-101476"])
    ap.add_argument("--total", type=int, default=460_000)
    ap.add_argument("--n-bank-subj", type=int, default=12)
    ap.add_argument("--replot", action="store_true", help="저장된 결과로 그림만 다시 그린다")
    main(ap.parse_args())
