#!/usr/bin/env python
"""128점 등간격 재샘플 fidelity 점검 (pipeline §8)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data import tt_io                                              # noqa: E402
from atm_sc.data.paths import tt_path                                      # noqa: E402
from atm_sc.data.resample_streamlines import check_fidelity, resample_equidistant   # noqa: E402


def main(sub, n):
    _, gen = tt_io.load_streamlines(str(tt_path(sub)), chunk=n)
    mm, npts = next(gen())
    t = time.time()
    out = resample_equidistant(mm, npts, 128)
    dt = time.time() - t
    assert out.shape == (len(npts), 128, 3)
    st = check_fidelity(mm, npts, out)
    print(f"[{sub}] {st['n']:,} streamlines  원래 점수 평균 {npts.mean():.1f}")
    print(f"  길이 오차(원래 대비 감소율): mean {100*st['rel_mean']:.3f}%  "
          f"p99 {100*st['rel_p99']:.3f}%  max {100*st['rel_max']:.3f}%")
    print(f"  직선형({st['n_straight']:,}개) 최대 오차 {100*st['straight_max']:.3f}%  (< 2% 확인)")
    print(f"  끝점 오차 max {st['endpoint_max_err']:.2e} mm  (0 이어야 함)")
    assert st["endpoint_max_err"] == 0.0
    print(f"  시간: {dt:.2f}s / {len(npts):,} -> {dt/len(npts)*1e5:.1f}s per 1e5 streamlines")
    print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    main(a.sub, a.n)
