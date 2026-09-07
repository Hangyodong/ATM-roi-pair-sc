#!/usr/bin/env python
"""native T1 -> ATM 작업 격자 W. 그 subject 의 GT streamline 으로 정합을 검증한다."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atm_sc.data import tt_io                                     # noqa: E402
from atm_sc.data.prepare_t1 import prepare_subject, check_alignment   # noqa: E402
from atm_sc.data.paths import t1_path, tt_path                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main(sub, mode, probe_tracks):
    t1, tt = t1_path(sub), tt_path(sub)
    assert t1.exists() and tt.exists(), (t1, tt)

    cache = ROOT / "outputs" / "cache"; cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vol = prepare_subject(t1, mode=mode, out_dir=cache)
    print(f"[{sub}] {mode} 정합 + W 재샘플: {time.time()-t0:.0f}s  "
          f"shape={vol.shape} max={vol.max():.1f}")

    _, gen = tt_io.load_streamlines(str(tt), chunk=probe_tracks)
    mm, npts = next(gen())
    frac = check_alignment(vol, mm)
    print(f"[{sub}] GT streamline 의 {frac:.4f} 가 warp 된 T1 뇌 안에 있음 "
          f"({len(npts):,} streamlines 로 검사)")

    out = cache / f"{sub}_T1w_{mode}_W.npy"
    np.save(out, vol)
    print(f"PASS -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--mode", default="syn", choices=["syn", "rigid"])
    ap.add_argument("--probe-tracks", type=int, default=20000)
    a = ap.parse_args()
    main(a.sub, a.mode, a.probe_tracks)
