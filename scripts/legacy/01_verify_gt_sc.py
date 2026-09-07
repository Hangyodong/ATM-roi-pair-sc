#!/usr/bin/env python
"""GT 검증: .tt.gz 에서 SC 를 재계산해 FC_DKPD25_*.mat 의 SC_weight/SC_length 와 대조.

이 스크립트가 통과하지 못하면 좌표계 가정이 틀린 것이므로 그 뒤 단계는 전부 무의미하다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atm_sc.data import tt_io                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
from atm_sc.data.paths import ATLAS, SC_MAT as MAT, tt_path
N_ROI = 82
# 통과 기준. 실측 baseline: pass r=0.9986 / end r=0.673
MIN_R, MIN_F1, MIN_LEN_R = 0.99, 0.95, 0.95


def load_gt(sub):
    m = sio.loadmat(MAT)
    rows = [r for r in m["data"][0] if str(r["subject"][0]) == sub]
    assert rows, f"{sub} 가 {MAT.name} 에 없음 (dwi_qc==pass 만 수록됨)"
    return (np.asarray(rows[0]["SC_weight"], np.float64),
            np.asarray(rows[0]["SC_length"], np.float64))


def main(sub, tt_path):
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    assert int(atlas.max()) == N_ROI, f"ROI 수 불일치: {atlas.max()}"
    assert len(np.unique(atlas)) - 1 == N_ROI, "atlas 라벨 소실"

    hdr, gen = tt_io.load_streamlines(tt_path)
    print(f"[{sub}] tt dim={hdr.dimension} vox={hdr.voxel_size} "
          f"diag={np.diag(hdr.trans_to_mni)[:3]}")

    W = np.zeros((N_ROI, N_ROI), np.int64)
    S = np.zeros((N_ROI, N_ROI), np.float64)
    We = np.zeros_like(W); Se = np.zeros_like(S)
    n = 0
    for mm, npts in gen():
        w, s = tt_io.hard_sc(mm, npts, atlas, img.affine, N_ROI, "pass")
        W += w; S += s
        w, s = tt_io.hard_sc(mm, npts, atlas, img.affine, N_ROI, "end")
        We += w; Se += s
        n += len(npts)
        print(f"  {n:>9,} streamlines", flush=True)

    assert n > 0 and np.isfinite(W).all()
    assert W.sum() > 0, "SC 전부 0 — endpoint 가 atlas 밖. 좌표계 확인 필요"
    assert np.array_equal(W, W.T), "SC 비대칭"

    Gw, Gl = load_gt(sub)
    iu = np.triu_indices(N_ROI, 1)
    Lp = np.divide(S, W, out=np.zeros_like(S), where=W > 0)
    Le = np.divide(Se, We, out=np.zeros_like(Se), where=We > 0)

    def report(name, X, G):
        x, g = X[iu].astype(float), G[iu]
        tp = int(((x > 0) & (g > 0)).sum()); fp = int(((x > 0) & (g == 0)).sum())
        fn = int(((x == 0) & (g > 0)).sum())
        r = float(np.corrcoef(x, g)[0, 1]); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"  {name:9s} sum={x.sum():>12,.0f} (GT {g.sum():>12,.0f})  "
              f"nnz={int((x>0).sum()):>5} (GT {int((g>0).sum()):>5})  r={r:.4f}  "
              f"r_log={np.corrcoef(np.log1p(x), np.log1p(g))[0,1]:.4f}  F1={f1:.4f}")
        return r, f1

    def report_len(name, X, G):
        m = (G[iu] > 0) & (X[iu] > 0)
        r = float(np.corrcoef(X[iu][m], G[iu][m])[0, 1])
        print(f"  {name:9s} r={r:.4f}  MAE={np.abs(X[iu][m]-G[iu][m]).mean():.2f} mm  n={int(m.sum())}")
        return r

    print(f"\n[{sub}] {n:,} streamlines")
    report("SC_end", We, Gw)
    r_p, f1_p = report("SC_pass", W, Gw)
    report_len("LEN_end", Le, Gl)
    r_lp = report_len("LEN_pass", Lp, Gl)

    assert r_p >= MIN_R, f"pass SC 상관 {r_p:.4f} < {MIN_R} — 좌표계/atlas 확인"
    assert f1_p >= MIN_F1, f"pass SC edge F1 {f1_p:.4f} < {MIN_F1}"
    assert r_lp >= MIN_LEN_R, f"pass length 상관 {r_lp:.4f} < {MIN_LEN_R}"
    out = ROOT / "outputs" / "cache" / f"{sub}_hardsc.npz"
    np.savez_compressed(out, SC_pass=W, LEN_pass=Lp, SC_end=We, LEN_end=Le, GT_W=Gw, GT_L=Gl)
    print(f"\nPASS  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--tt", default=None, help="기본: ppmi_probe/PPMI_QC263_tracto/<sub>/<sub>_tract.tt.gz")
    a = ap.parse_args()
    main(a.sub, a.tt or str(tt_path(a.sub)))
