"""corridor pair feature + ROI 별 WM 표면적 추출 (기존 WM 맵만 사용)."""
import argparse, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data import anat_corridor as AC
from atm_sc.data import anat_tier1 as T1F
from atm_sc.data.paths import CACHE
from atm_sc.models.roi_pool import atlas_on_feature_grid
from atm_sc.spaces import W_AFFINE, W_SHAPE


def main(a):
    labels, aff = atlas_on_feature_grid(feat_shape=W_SHAPE, in_shape=W_SHAPE,
                                        in_affine=W_AFFINE, return_affine=True)
    n_roi = int(labels.max()); assert n_roi == 82, n_roi
    pairs = np.stack(np.triu_indices(n_roi, 1), 1)
    subs = sorted({p.name.split("_T1w_")[0] for p in CACHE.glob(f"*_T1w_{a.source}_W.npy")})
    subs = subs[a.start::a.stride]
    AC.OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = 0
    for i, s in enumerate(subs):
        p = AC.path(s, a.source)
        if p.exists() and not a.force:
            n_skip += 1; continue
        t0 = time.time()
        try:
            wm = np.load(CACHE / f"{s}_WM_W.npy").astype(np.float32)
            t1 = np.load(CACHE / f"{s}_T1w_{a.source}_W.npy").astype(np.float32)
            t1n = t1 / max(float(np.percentile(t1[wm > 0.5], 99.5)), 1e-6)
            cent = np.asarray(T1F.load(s, a.source)["centroid"], np.float32)
            assert cent.shape == (n_roi, 3), cent.shape
            cor = AC.corridor_features(wm, t1n, cent, pairs, aff)
            ar = AC.wm_surface_area(wm, labels, n_roi)
            np.savez_compressed(p, corridor=cor, wm_area=ar)
            assert p.stat().st_size > 0
            n_ok += 1
            print(f"[{i+1}/{len(subs)}] {s} wm_mean {cor[:,0].mean():.3f} "
                  f"gap {cor[:,5].mean():.3f} area {ar.sum():.0f}mm2 {time.time()-t0:.1f}s", flush=True)
        except Exception:
            n_err += 1
            print(f"[{i+1}/{len(subs)}] {s} FAIL\n{traceback.format_exc()}", flush=True)
    print(f"CORRIDOR DONE ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    assert n_err == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rigid"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    main(ap.parse_args())
