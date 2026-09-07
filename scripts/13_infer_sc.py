#!/usr/bin/env python
"""checkpoint -> subject T1 -> whole-brain tractogram -> SC, GT(.mat pass / sc_end) 와 비교 (pipeline §22, §29)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                       # noqa: E402
from atm_sc.data.paths import ATLAS                                   # noqa: E402
from atm_sc.inference.generate_sc import generate_tractogram, select_pairs, tractogram_sc, save_trk  # noqa: E402
from atm_sc.losses import sc_metrics                                  # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                          # noqa: E402
from atm_sc.training.run import anatomy_feature, stage3_cache         # noqa: E402


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subj = ROIPairSubject(a.sub)
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    sd = torch.load(a.checkpoint, map_location=dev, weights_only=True)
    trainable = sd.get("trainable", "vae")
    m = ROIPairATM(n_roi=subj.n_roi, device=dev, trainable=trainable)
    m.load_state_dict(sd["model"]); m.eval()
    if trainable == "vae+unet4":
        with torch.no_grad():
            feat = m.anatomy_from_stage3(stage3_cache(m, a.sub, "AF_L"))
    else:
        feat = anatomy_feature(m, a.sub, "AF_L")

    if a.pairs == "gt":
        pairs = np.asarray(subj.pair_ids); src = f"GT 양성 pair {len(pairs)} (평가용)"
    else:
        pairs, prob = select_pairs(m, feat, subj.n_roi, thr=a.edge_thr); src = f"edge head > {a.edge_thr}: {len(pairs)} pairs"
    assert len(pairs) > 0, "선택된 pair 없음"
    t0 = time.time()
    S, w, pr = generate_tractogram(m, feat, pairs, a.n_per_pair, seed=0)
    t_gen = time.time() - t0
    L = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)
    print(f"[{a.sub}] {src} x {a.n_per_pair} -> {len(S):,} streamlines in {t_gen:.1f}s "
          f"({len(S)/t_gen:,.0f}/s) | 길이 mean {L.mean():.1f} p5 {np.percentile(L,5):.1f} p95 {np.percentile(L,95):.1f} mm | w mean {w.mean():.3f}")
    t0 = time.time(); sc = tractogram_sc(S, w, atlas, img.affine, subj.n_roi); print(f"hard SC 계산 {time.time()-t0:.1f}s")

    gt = {"pass": (np.asarray(subj.sc_mat, np.float64), np.asarray(subj.len_mat, np.float64)),
          "end": (np.asarray(subj.sc_end, np.float64), np.asarray(subj.len_end, np.float64))}
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    iu = np.triu_indices(subj.n_roi, 1)
    print(f"{'rule':5s} {'variant':9s} {'r':>7s} {'r_log':>7s} {'ccc':>7s} {'F1':>6s} {'density':>8s} {'sum':>10s} {'GT sum':>10s} | {'len r':>6s} {'len MAE':>8s}")
    for rule in ("pass", "end"):
        gw, gl = gt[rule]
        for variant in ("sc", "sc_w"):
            P = sc[rule][variant]
            mm_ = sc_metrics(T(P), T(gw))
            lp = sc[rule]["len"]; msk = (gw[iu] > 0) & (lp[iu] > 0)
            lr = np.corrcoef(lp[iu][msk], gl[iu][msk])[0, 1] if msk.sum() > 2 else float("nan")
            lmae = np.abs(lp[iu][msk] - gl[iu][msk]).mean() if msk.sum() else float("nan")
            print(f"{rule:5s} {variant:9s} {mm_['r']:7.4f} {mm_['r_log']:7.4f} {mm_['ccc']:7.4f} {mm_['edge_f1']:6.3f} "
                  f"{mm_['density']:8.3f} {P[iu].sum():10.0f} {gw[iu].sum():10.0f} | {lr:6.3f} {lmae:8.2f}")
    if a.save_trk:
        out = ROOT / "outputs" / "tractograms" / f"{a.sub}_{Path(a.checkpoint).stem}.trk"
        out.parent.mkdir(parents=True, exist_ok=True); save_trk(S, out); print("trk ->", out)
    if a.save_npz:
        out = ROOT / "outputs" / "sc" / f"{a.sub}_{Path(a.checkpoint).stem}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{f"{r}_{k}": v for r in sc for k, v in sc[r].items()}, lengths=L, weights=w, pairs=pr)
        print("sc  ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-000001")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", default="gt", choices=["gt", "edge"])
    ap.add_argument("--edge-thr", type=float, default=0.5)
    ap.add_argument("--n-per-pair", type=int, default=32)
    ap.add_argument("--save-trk", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    main(ap.parse_args())
