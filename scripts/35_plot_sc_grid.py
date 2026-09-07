#!/usr/bin/env python
"""test subject 별 SC 행렬을 격자로 비교 (그림 1장에 8명).

각 그림: 세로 3줄 x 가로 8명
  1줄 원본 GT SC
  2줄 현재 방식      (prior + 균등 배분)
  3줄 bank + 템플릿 배분

  python scripts/35_plot_sc_grid.py --total 100000
결과: outputs/figures/sc_grid_1.png ... sc_grid_4.png
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
from atm_sc.data.paths import ATLAS                                 # noqa: E402
from atm_sc.data.tt_io import hard_sc                               # noqa: E402
from atm_sc.inference.generate_sc import select_pairs, template_counts   # noqa: E402
from atm_sc.inference.latent_bank import LatentBank, build_bank     # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                        # noqa: E402
from atm_sc.training.run import t1_input                            # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
IU = np.triu_indices(82, 1)
FIG = ROOT / "outputs/figures"


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else np.nan


def sc_of(S, atlas, affine):
    npts = np.full(len(S), S.shape[1], np.int64)
    return hard_sc(S.reshape(-1, 3).astype(np.float32), npts, atlas, affine, 82, "pass")[0].astype(np.float64)


@torch.no_grad()
def gen(m, feat, pairs, counts, bank, atlas, affine, dev, seed=0, batch=20000):
    rep = np.repeat(pairs.astype(np.int64), counts.astype(np.int64), axis=0)
    g = torch.Generator(device=dev); g.manual_seed(seed); rs = np.random.default_rng(seed)
    W = np.zeros((82, 82), np.float64)
    for i in range(0, len(rep), batch):
        blk = rep[i:i + batch]; P = torch.as_tensor(blk, device=dev)
        z = m.sample_z(P, g)
        if bank is not None:
            zb, hit = bank.sample(blk, rs)
            if hit.any():
                z = torch.where(torch.as_tensor(hit, device=dev)[:, None],
                                torch.as_tensor(zb, device=dev), z)
        W += sc_of(m.decode(z, m.condition(feat, P)).cpu().numpy().astype(np.float32), atlas, affine)
    return W


def main(a):
    cache = FIG / "sc_grid_cache.npz"
    if a.replot and cache.exists():
        d = np.load(cache)
        G, C, B, subs = d["G"], d["C"], d["B"], [str(x) for x in d["subs"]]
        print(f"캐시 사용 ({len(subs)}명)", flush=True)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sd = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
        lv = sd.get("unet_level", "full")
        m = ROIPairATM(n_roi=82, trainable="full" if lv == "full" else "vae", unet_level=lv, device=dev)
        m.load_checkpoint(sd["model"]); m.eval()
        img = nib.load(ATLAS); atlas = np.asarray(img.dataobj).astype(np.int16); affine = img.affine
        train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()]
        subs = [l.strip() for l in open(ROOT / "outputs/splits/test.txt") if l.strip()]

        te = None
        for s in train:
            z = np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")
            te = z["sc_end"].astype(np.float64) if te is None else te + z["sc_end"]
        te /= len(train)
        bp = ROOT / "outputs/inference/latent_bank.npz"
        bank = (LatentBank.load(bp) if bp.exists() else
                build_bank(m, train[:a.n_bank_subj],
                           lambda s: ROOT / f"outputs/roi_pairs/{s}/bundles.npz", 24))
        G, C, B = [], [], []
        for n, s in enumerate(subs, 1):
            z = np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")
            with torch.no_grad():
                feat = m.atm.encode_anatomy(t1_input(m, s))
                pe, _ = select_pairs(m, feat, 82, thr=0.5)
            G.append(z["sc_pass"].astype(np.float64))
            C.append(gen(m, feat, pe, np.full(len(pe), max(a.total // len(pe), 1)),
                         None, atlas, affine, dev))
            B.append(gen(m, feat, pe, template_counts(te, pe, a.total), bank, atlas, affine, dev))
            print(f"[{n}/{len(subs)}] {s}  현재 r={corr(C[-1][IU], G[-1][IU]):.3f}  "
                  f"bank r={corr(B[-1][IU], G[-1][IU]):.3f}", flush=True)
        G, C, B = np.stack(G), np.stack(C), np.stack(B)
        FIG.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, G=G, C=C, B=B, subs=np.array(subs))

    FIG.mkdir(parents=True, exist_ok=True)
    if a.maxnorm:
        # 행렬마다 자기 최댓값으로 나눈다. GT 와 생성의 절대 스케일이 달라도 패턴을 직접 비교할 수 있다.
        G, C, B = [X / X.reshape(len(X), -1).max(1)[:, None, None] for X in (G, C, B)]
        vmin, vmax = a.vmin, 1.0
    else:
        vmin, vmax = 1.0, max(G.max(), B.max())
    rows = [("원본 GT SC", G, None),
            ("현재 방식 (prior + 균등 배분)", C, True),
            ("bank + 템플릿 배분", B, True)]
    per = 8
    for f_i in range(0, len(G), per):
        sl = slice(f_i, min(f_i + per, len(G)))
        k = sl.stop - sl.start
        fig, ax = plt.subplots(3, k, figsize=(2.35 * k + 1.6, 8.2), constrained_layout=True,
                               squeeze=False)
        for ri, (lab, M, show_r) in enumerate(rows):
            for ci in range(k):
                A = ax[ri][ci]
                im = A.imshow(np.maximum(M[sl][ci], vmin * 0.5), norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap="magma", interpolation="nearest")
                A.set_xticks([]); A.set_yticks([])
                A.axhline(65.5, color="cyan", lw=0.5); A.axvline(65.5, color="cyan", lw=0.5)
                if ri == 0:
                    A.set_title(subs[sl][ci].replace("sub-", ""), fontsize=10)
                if show_r:
                    A.set_xlabel(f"r = {corr(M[sl][ci][IU], G[sl][ci][IU]):.3f}", fontsize=9)
                if ci == 0:
                    A.set_ylabel(lab, fontsize=10)
        fig.colorbar(im, ax=ax[:, -1], fraction=0.035, pad=0.02,
                     label="자기 최댓값 대비 (로그)" if a.maxnorm else "가닥 수 (로그)")
        rc = np.nanmean([corr(C[sl][i][IU], G[sl][i][IU]) for i in range(k)])
        rb = np.nanmean([corr(B[sl][i][IU], G[sl][i][IU]) for i in range(k)])
        fig.suptitle(f"test subject SC 행렬 비교{' — 행렬마다 최댓값으로 정규화' if a.maxnorm else ''} "
                     f"({f_i + 1}–{sl.stop}번째, {k}명)   "
                     f"평균 r: 현재 {rc:.3f}  →  bank+템플릿 {rb:.3f}", fontsize=13)
        p = FIG / f"sc_grid_{'maxnorm_' if a.maxnorm else ''}{f_i // per + 1}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  저장 {p}")

    print(f"\n전체 {len(G)}명 평균 r:  현재 "
          f"{np.nanmean([corr(C[i][IU], G[i][IU]) for i in range(len(G))]):.4f}  "
          f"bank+템플릿 {np.nanmean([corr(B[i][IU], G[i][IU]) for i in range(len(G))]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt")
    ap.add_argument("--total", type=int, default=100_000)
    ap.add_argument("--n-bank-subj", type=int, default=12)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--maxnorm", action="store_true", help="행렬마다 자기 최댓값으로 나눠서 그린다")
    ap.add_argument("--vmin", type=float, default=1e-4, help="maxnorm 일 때 로그 하한")
    main(ap.parse_args())
