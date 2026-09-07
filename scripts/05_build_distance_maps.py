#!/usr/bin/env python
"""ROI 거리맵 [82, 91, 109, 91] 생성/캐시 (pipeline §13)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data.build_distance_maps import get_distance_maps, load_atlas   # noqa: E402
from atm_sc.data.paths import CACHE                                         # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t = time.time()
    dm = get_distance_maps(force=a.force)
    img, atlas = load_atlas()
    print(f"dist_maps {dm.shape} {dm.dtype}  max={dm.max():.1f} mm  "
          f"{time.time()-t:.1f}s  -> {CACHE/'dist_maps.npy'}")
    cnt = np.bincount(atlas.ravel())[1:]
    print(f"atlas voxel/ROI: min={cnt.min()} (label {cnt.argmin()+1}) max={cnt.max()} "
          f"zooms={img.header.get_zooms()[:3]}")
