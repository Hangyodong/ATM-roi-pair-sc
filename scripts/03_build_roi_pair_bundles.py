#!/usr/bin/env python
"""assignments.npz -> pair 별 최대 cap 개를 골라 128점으로 재샘플 -> bundles.npz (pipeline §6, §8)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data import tt_io                                              # noqa: E402
from atm_sc.data.build_distance_maps import load_atlas                     # noqa: E402
from atm_sc.data.dataset import ROI_PAIRS_DIR                              # noqa: E402
from atm_sc.data.paths import subjects, tt_path                            # noqa: E402
from atm_sc.data.resample_streamlines import resample_equidistant          # noqa: E402
from atm_sc.data.trk_to_roi_pairs import pair_to_index, select_capped      # noqa: E402

N_ROI = 82


def run(sub, img, atlas, cap, seed):
    t0 = time.time()
    d = ROI_PAIRS_DIR / sub
    asg = np.load(d / "assignments.npz")
    pair, length, sc_end = asg["pair"], asg["length_mm"], asg["sc_end"]
    N = len(pair)
    pidx = pair_to_index(pair, N_ROI)
    sel = select_capped(pidx, cap, seed)                 # src_index 오름차순
    assert len(sel) > 0
    print(f"[{sub}] {N:,} streamlines, 할당 {int((pidx>=0).sum()):,}, 선택 {len(sel):,} (cap={cap})")

    # 선택된 것만 디코딩하며 재샘플 (chunk 순서 = src 순서)
    keep = np.zeros(N, bool); keep[sel] = True
    S, cur = [], 0
    _, gen = tt_io.load_streamlines(str(tt_path(sub)))
    for mm, npts in gen():
        k = keep[cur:cur + len(npts)]
        if k.any():
            starts = np.concatenate([[0], np.cumsum(npts)[:-1]])
            idx = np.where(k)[0]
            pts = np.concatenate([mm[starts[i]:starts[i] + npts[i]] for i in idx])
            S.append(resample_equidistant(pts, npts[idx], 128))
        cur += len(npts)
    assert cur == N
    S = np.concatenate(S)
    assert S.shape == (len(sel), 128, 3)

    # pair_index 오름차순 (같은 pair 안에서는 src_index 순) 으로 정렬
    order = np.lexsort((sel, pidx[sel]))
    sel, S = sel[order], S[order]
    pi = pidx[sel]
    assert (pi >= 0).all() and (np.diff(pi) >= 0).all()
    uniq, first = np.unique(pi, return_index=True)
    offsets = np.concatenate([first, [len(sel)]]).astype(np.int64)
    pair_ids = np.stack([uniq // N_ROI, uniq % N_ROI], 1).astype(np.int16)
    count_full = sc_end[pair_ids[:, 0], pair_ids[:, 1]].astype(np.int64)

    # --- 검증 ----------------------------------------------------------------
    iu = np.triu_indices(N_ROI, 1)
    K_pos = int((sc_end[iu] > 0).sum())
    assert len(pair_ids) == K_pos, f"양성 pair {K_pos} 중 {len(pair_ids)} 만 저장됨"
    assert (np.diff(offsets) == np.minimum(count_full, cap)).all(), "pair 별 개수 != min(count, cap)"
    assert (count_full > 0).all()
    S16 = S.astype(np.float16)
    qerr = float(np.abs(S16.astype(np.float32) - S).max())
    assert qerr < 0.1, f"fp16 양자화 오차 {qerr:.3f} mm"
    # 저장된(fp16) 끝점이 할당된 ROI 에 실제로 떨어지는가
    ends = S16.astype(np.float32)[:, [0, -1]].reshape(-1, 3)
    lab = tt_io.point_labels(ends, atlas, img.affine).reshape(-1, 2).astype(np.int64) - 1
    got = np.sort(lab, 1)
    frac = float((got == pair[sel]).all(1).mean())
    print(f"  pairs K={len(pair_ids)}  저장 {len(sel):,}  fp16 max err {qerr:.4f} mm  "
          f"끝점∈할당ROI {100*frac:.2f}%")
    assert frac >= 0.95, f"끝점이 할당 ROI 에 있는 비율 {frac:.3f} < 0.95"
    lens = length[sel].astype(np.float32)
    assert (lens > 0).all() and np.isfinite(S16.astype(np.float32)).all()

    np.savez_compressed(d / "bundles.npz", streamlines=S16, lengths=lens,
                        pair=pair[sel].astype(np.int16), pair_index=pi.astype(np.int32),
                        src_index=sel.astype(np.int64), pair_ids=pair_ids, pair_offsets=offsets,
                        pair_count_full=count_full, cap=cap, seed=seed, n_roi=N_ROI)
    sz = (d / "bundles.npz").stat().st_size / 1e6
    print(f"  -> {d/'bundles.npz'}  {sz:.0f} MB  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    img, atlas = load_atlas()
    for s in (subjects() if a.all else [a.sub]):
        run(s, img, atlas, a.cap, a.seed)
