#!/usr/bin/env python
"""GT tractogram -> endpoint ROI 할당 -> canonical ROI pair + end/pass SC (pipeline §5).

출력: outputs/roi_pairs/{sub}/assignments.npz  (docs/ROI_PAIR_DATA_FORMAT.md)
검증: sc_pass vs .mat GT r >= 0.99 (assert), 자체 end-SC == tt_io.hard_sc('end') (assert).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data import tt_io                                              # noqa: E402
from atm_sc.data.build_distance_maps import load_atlas                     # noqa: E402
from atm_sc.data.dataset import ROI_PAIRS_DIR, load_mat_gt                 # noqa: E402
from atm_sc.data.paths import subjects, tt_path                            # noqa: E402
from atm_sc.data.trk_to_roi_pairs import assign_roi_pairs                  # noqa: E402

N_ROI = 82
MIN_R_PASS = 0.99


def _corr(a, b):
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.corrcoef(a[iu].astype(float), b[iu].astype(float))[0, 1])


def run(sub, img, atlas, cross_check=True):
    t0 = time.time()
    tt = tt_path(sub)
    assert tt.stat().st_size > 0, f"{tt} 가 0 바이트"
    print(f"[{sub}] {tt.name}")
    r = assign_roi_pairs(tt, atlas, img.affine, N_ROI)
    n, na = r["n_total"], r["n_assigned"]
    R = N_ROI
    iu = np.triu_indices(R, 1)
    K = int((r["sc_end"][iu] > 0).sum())
    print(f"  n_total={n:,}  n_assigned={na:,} ({100*na/n:.1f}%)  "
          f"positive pairs K={K}  density={K/len(iu[0]):.3f}")
    top = np.argsort(r["sc_end"][iu])[::-1][:5]
    print("  top-5 pairs (a,b,count):",
          [(int(iu[0][t]), int(iu[1][t]), int(r["sc_end"][iu][t])) for t in top])
    print(f"  배경 endpoint: start {100*(r['start_roi']<0).mean():.1f}%  "
          f"end {100*(r['end_roi']<0).mean():.1f}%  같은 ROI 양끝 "
          f"{100*((r['start_roi']==r['end_roi'])&(r['start_roi']>=0)).mean():.1f}%")

    # --- 검증 ----------------------------------------------------------------
    gw, gl = load_mat_gt(sub)
    r_pass, r_end = _corr(r["sc_pass"], gw), _corr(r["sc_end"], gw)
    m = (gl[iu] > 0) & (r["len_pass"][iu] > 0)
    r_len = float(np.corrcoef(r["len_pass"][iu][m], gl[iu][m])[0, 1])
    print(f"  sc_pass vs .mat GT: r={r_pass:.4f}  (len r={r_len:.4f})   sc_end vs .mat: r={r_end:.4f}")
    assert r_pass >= MIN_R_PASS, f"pass SC 상관 {r_pass:.4f} < {MIN_R_PASS} — 좌표계/atlas 오류"

    if cross_check:
        W = np.zeros((R, R), np.int64); S = np.zeros((R, R), np.float64)
        _, gen = tt_io.load_streamlines(str(tt))
        for mm, npts in gen():
            w, s = tt_io.hard_sc(mm, npts, atlas, img.affine, R, "end")
            W += w; S += s
        assert np.array_equal(W, r["sc_end"]), "자체 end-SC 가 tt_io.hard_sc('end') 와 다름"
        Lref = np.divide(S, W, out=np.zeros_like(S), where=W > 0)
        assert np.allclose(Lref, r["len_end"], atol=1e-3), "end 평균 길이 불일치"
        print("  cross-check: end-SC == tt_io.hard_sc('end')  OK")

    out = ROI_PAIRS_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "assignments.npz", **r)
    print(f"  -> {out/'assignments.npz'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-cross-check", action="store_true")
    a = ap.parse_args()
    img, atlas = load_atlas()
    for s in (subjects() if a.all else [a.sub]):
        run(s, img, atlas, cross_check=not a.no_cross_check)
