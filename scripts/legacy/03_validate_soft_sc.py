#!/usr/bin/env python
"""Stage: differentiable soft SC builder 단독 검증 (v2 Stage 5 의 수정판).

v2 는 soft SC 를 MRtrix SC 와 비교하라고 하지만 이 서버에 MRtrix 가 없고, 더 중요하게는
GT SC 가 이미 .mat 에 있고 그것을 hard pass 규칙으로 r=0.9986 까지 재현했다.
따라서 비교 대상은 MRtrix 가 아니라 **GT SC 자체**로 잡는다.

    GT .tt.gz --128점 재샘플--> soft pass SC(tau)  vs  .mat 의 SC_weight

tau 를 스윕해 최적값을 고른다. 128점 재샘플은 ATM 출력 형식과 같으므로, 이 검증은
"ATM 이 GT streamline 을 완벽히 재현했을 때 도달 가능한 SC 상한"도 함께 알려준다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atm_sc.data import tt_io                                      # noqa: E402
from atm_sc.models.endpoint_assigner import RoiAssigner, build_distance_maps   # noqa: E402
from atm_sc.models.sc_builder import pass_sc, streamline_lengths   # noqa: E402
from atm_sc.losses import sc_metrics                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
from atm_sc.data.paths import ATLAS, SC_MAT as MAT, tt_path
N_ROI = 82


def gt_sc(sub):
    m = sio.loadmat(MAT)
    r = [x for x in m["data"][0] if str(x["subject"][0]) == sub]
    assert r, sub
    return (np.asarray(r[0]["SC_weight"], np.float64), np.asarray(r[0]["SC_length"], np.float64))


def main(sub, taus, aggs, dbgs, max_tracks, device):
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    cache = ROOT / "outputs" / "cache" / "dist_maps.npy"
    if cache.exists():
        dm = np.load(cache)
    else:
        print("ROI 거리맵 생성 중 (82 x 91x109x91) ...", flush=True)
        dm = build_distance_maps(atlas, N_ROI, img.header.get_zooms()[:3])
        cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, dm)
    print(f"거리맵 {dm.shape} {dm.dtype} max={dm.max():.1f}mm")

    tt = tt_path(sub)
    _, gen = tt_io.load_streamlines(str(tt))
    mms, Ls, n = [], [], 0
    for mm, npts in gen():
        Ls.append(tt_io.segment_lengths(mm, npts))
        mms.append(tt_io.resample_128(mm, npts))
        n += len(npts)
        if n >= max_tracks:
            break
    S = torch.from_numpy(np.concatenate(mms)).to(device)
    L = torch.from_numpy(np.concatenate(Ls).astype(np.float32)).to(device)
    print(f"[{sub}] {len(S):,} streamlines 를 128점으로 재샘플, 길이 평균 {L.mean():.1f}mm")

    Gw, Gl = gt_sc(sub)
    scale = len(S) / 1_000_000                       # GT 는 1e6 streamline 기준
    Gw_s = torch.from_numpy(Gw * scale).float().to(device)

    # hard 기준선 (같은 128점 재샘플 streamline 으로)
    hw, hs = tt_io.hard_sc(np.concatenate(mms).reshape(-1, 3),
                           np.full(len(S), 128, np.int64), atlas, img.affine, N_ROI, "pass")
    hard = torch.from_numpy(hw.astype(np.float64)).float().to(device)
    m = sc_metrics(hard, Gw_s)
    print(f"  hard(128점 재샘플)  r={m['r']:.4f} r_log={m['r_log']:.4f} "
          f"ccc={m['ccc']:.4f} F1={m['edge_f1']:.4f} density={m['density']:.3f}")

    best = None
    for tau in taus:
      for agg in aggs:
       for dbg in dbgs:
        ra = RoiAssigner(dm, img.affine, tau=tau, device=device, aggregate=agg,
                         d_bg=(None if dbg < 0 else dbg))
        us = []
        for i in range(0, len(S), 20000):
            with torch.no_grad():
                us.append(ra.visit_probs(S[i:i + 20000]))
        u = torch.cat(us)
        sc, num = pass_sc(u, L)
        mm_ = sc_metrics(sc, Gw_s)
        lp = (num / (sc + 1e-8)).cpu().numpy()
        msk = (Gl > 0) & (lp > 0)
        rl = float(np.corrcoef(lp[msk], Gl[msk])[0, 1])
        print(f"  tau={tau:4.2f} {agg:4s} bg={dbg:>4.1f} r={mm_['r']:.4f} r_log={mm_['r_log']:.4f} "
              f"ccc={mm_['ccc']:.4f} F1={mm_['edge_f1']:.4f} density={mm_['density']:.3f} "
              f"| length r={rl:.4f} MAE={np.abs(lp[msk]-Gl[msk]).mean():.1f}mm")
        if best is None or mm_["r"] > best[1]:
            best = ((tau, agg, dbg), mm_["r"])
        del ra, u, sc, num
        torch.cuda.empty_cache() if device == "cuda" else None
    print(f"\nbest tau = {best[0]} (r = {best[1]:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--aggs", nargs="+", default=["max", "lse"])
    ap.add_argument("--dbgs", type=float, nargs="+", default=[-1, 1.0, 2.0, 4.0])
    ap.add_argument("--max-tracks", type=int, default=200000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    main(a.sub, a.taus, a.aggs, a.dbgs, a.max_tracks, a.device)
