#!/usr/bin/env python
"""Route loss 용 GT ROI-visitation 전처리 (ROUTE 전략 §13, §54).

bundles.npz 의 128점 streamline 마다 "지나간 ROI" 를 hard atlas 규칙(GT pass-SC 와 동일)으로 구해
  outputs/roi_pairs/<sub>/visit.npz
    visit_packed  [n_streamlines, ceil(R/8)] uint8   (np.packbits, streamline 순서 = bundles.npz 순서)
    pair_marginal [n_pairs, R] float16               (pair 안에서 그 ROI 를 지나는 streamline 비율)
    n_roi, n_visits_mean
를 만든다. 읽기 전용(bundles.npz 는 건드리지 않음)이라 학습 중에 돌려도 안전하다.

  python scripts/24_build_visitation.py --workers 4
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import ATLAS                      # noqa: E402
from atm_sc.data.tt_io import point_labels               # noqa: E402

CHUNK = 20000                                            # streamline 단위 (20k x 128 점 = 2.6M lookup)


def visitation(S: np.ndarray, atlas: np.ndarray, affine: np.ndarray, n_roi: int) -> np.ndarray:
    """[n,128,3] mm -> [n,R] bool. 점 라벨 lookup 후 streamline 별 unique."""
    n, t = S.shape[0], S.shape[1]
    out = np.zeros((n, n_roi), bool)
    for i in range(0, n, CHUNK):
        s = S[i:i + CHUNK].astype(np.float64)
        lab = point_labels(s.reshape(-1, 3), atlas, affine).astype(np.int64).reshape(-1, t)
        rows = np.repeat(np.arange(lab.shape[0]), t)
        keep = lab.ravel() > 0
        out[i + rows[keep], lab.ravel()[keep] - 1] = True
    return out


def run(sub: str, atlas: np.ndarray, affine: np.ndarray, overwrite: bool) -> dict:
    d = ROOT / "outputs" / "roi_pairs" / sub
    out = d / "visit.npz"
    if out.exists() and not overwrite:
        return {"subject": sub, "skipped": True}
    t0 = time.time()
    z = np.load(d / "bundles.npz")
    S, off = z["streamlines"], z["pair_offsets"]
    n_roi = int(z["n_roi"])
    v = visitation(S, atlas, affine, n_roi)
    assert v.shape == (len(S), n_roi) and v.any(1).all(), f"{sub}: 통과 ROI 가 없는 streamline 존재"
    marg = np.stack([v[a:b].mean(0) for a, b in zip(off[:-1], off[1:])]).astype(np.float16)
    assert marg.shape == (len(off) - 1, n_roi) and np.isfinite(marg.astype(np.float32)).all()
    np.savez_compressed(out, visit_packed=np.packbits(v, axis=1), pair_marginal=marg,
                        n_roi=n_roi, n_streamlines=len(S))
    r = {"subject": sub, "n_streamlines": int(len(S)), "n_visits_mean": float(v.sum(1).mean()),
         "sec": time.time() - t0, "mb": out.stat().st_size / 1e6}
    print(f"[{sub}] {r['n_streamlines']:,} streamlines, 평균 통과 ROI {r['n_visits_mean']:.1f}, "
          f"{r['sec']:.0f}s, {r['mb']:.1f} MB", flush=True)
    return r


def main(a):
    import nibabel as nib
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (ROOT / "outputs" / "roi_pairs" / s / "bundles.npz").exists()]
    if a.limit:
        subs = subs[: a.limit]
    print(f"{len(subs)} subjects, workers={a.workers}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda s: run(s, atlas, img.affine, a.overwrite), subs))
    done = [r for r in res if not r.get("skipped")]
    if done:
        print(f"완료 {len(done)}/{len(subs)}, 평균 통과 ROI {np.mean([r['n_visits_mean'] for r in done]):.2f}, "
              f"평균 {np.mean([r['sec'] for r in done]):.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="outputs/subjects_train_eval_206.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    sys.exit(main(ap.parse_args()))
