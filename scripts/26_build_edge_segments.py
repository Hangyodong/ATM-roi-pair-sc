#!/usr/bin/env python
"""SC edge-aligned segment bundle 전처리 (EDGE_ALIGNED 전략 §4-7, §30-35).

bundles.npz 의 full streamline 을 "같은 streamline 안에서 ROI_i 와 ROI_j 를 잇는 부분경로" 로 분해해
  outputs/roi_pairs/<sub>/edge_segments.npz
    pair_ids   [E,2] int16      분해로 나온 edge (a<b)
    offsets    [E+1] int64      edge 별 segment 구간
    segments   [M,n_points,3] float16
    lengths    [M] float16
    count_full [E] int32        cap 전 segment 수 (학습 노출 균형용; GT SC 값이 아니다)
    n_roi, n_points, cap, min_dwell, min_length_mm
를 만든다. 읽기 전용이라 학습 중에도 안전하다.

GT 정의 실측(sub-100001, 1M streamline): Case B(모든 ROI 쌍) r=0.9986 vs Case A(인접) r=0.781
-> 인접 transition 이 아니라 모든 방문 ROI 쌍의 부분경로로 분해한다 (§30).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.edge_segments import MIN_DWELL, MIN_LEN_MM, SEG_POINTS, decompose_bundle, segment_sc  # noqa: E402
from atm_sc.data.paths import ATLAS                                                                    # noqa: E402


def run(sub, atlas, affine, a) -> dict:
    d = ROOT / "outputs" / "roi_pairs" / sub
    out = d / "edge_segments.npz"
    if out.exists() and not a.overwrite:
        return {"subject": sub, "skipped": True}
    t0 = time.time()
    S = np.load(d / "bundles.npz")["streamlines"]
    if a.max_streamlines:
        S = S[: a.max_streamlines]
    key, seg, L = decompose_bundle(S, atlas, affine, 82, a.n_points, a.min_dwell, a.min_length)
    assert len(seg) > 0, f"{sub}: segment 0개"
    order = np.argsort(key, kind="stable")
    key, seg, L = key[order], seg[order], L[order]
    uniq, first, cnt = np.unique(key, return_index=True, return_counts=True)
    rng = np.random.default_rng(a.seed)
    take = [first[i] + (np.sort(rng.choice(cnt[i], a.cap, replace=False)) if cnt[i] > a.cap else np.arange(cnt[i]))
            for i in range(len(uniq))]
    sel = np.concatenate(take)
    kept = np.minimum(cnt, a.cap)
    offsets = np.concatenate([[0], np.cumsum(kept)]).astype(np.int64)
    pair_ids = np.stack([uniq // 82, uniq % 82], 1).astype(np.int16)
    assert (pair_ids[:, 0] < pair_ids[:, 1]).all() and offsets[-1] == len(sel)
    np.savez_compressed(out, pair_ids=pair_ids, offsets=offsets, segments=seg[sel].astype(np.float16),
                        lengths=L[sel].astype(np.float16), count_full=cnt.astype(np.int32),
                        n_roi=82, n_points=a.n_points, cap=a.cap, min_dwell=a.min_dwell,
                        min_length_mm=a.min_length, n_streamlines=len(S))
    # QC: 분해 SC 가 GT .mat 와 얼마나 맞는가 (bundles 가 pair 당 cap 256 이라 절대값은 다르다)
    from atm_sc.data.dataset import load_mat_gt
    M = segment_sc(key, 82); W, _ = load_mat_gt(sub)
    iu = np.triu_indices(82, 1)
    r = float(np.corrcoef(M[iu], W[iu])[0, 1]); rl = float(np.corrcoef(np.log1p(M[iu]), np.log1p(W[iu]))[0, 1])
    rec = float(((M[iu] > 0) & (W[iu] > 0)).sum() / max((W[iu] > 0).sum(), 1))
    res = {"subject": sub, "n_streamlines": int(len(S)), "n_segments_full": int(len(key)),
           "n_segments_kept": int(len(sel)), "n_edges": int(len(uniq)), "seg_per_streamline": len(key) / len(S),
           "sc_r": r, "sc_r_log": rl, "gt_edge_recall": rec, "sec": time.time() - t0,
           "mb": out.stat().st_size / 1e6}
    print(f"[{sub}] {len(S):,} streamline -> segment {len(key):,} (edge {len(uniq)}, 저장 {len(sel):,}) "
          f"SC r={r:.3f} log r={rl:.3f} GT edge 재현 {rec:.3f} · {res['sec']:.0f}s {res['mb']:.0f}MB", flush=True)
    return res


def main(a):
    import nibabel as nib
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (ROOT / "outputs" / "roi_pairs" / s / "bundles.npz").exists()]
    if a.reverse:
        subs = subs[::-1]
    if a.limit:
        subs = subs[: a.limit]
    print(f"{len(subs)} subjects, workers={a.workers}, n_points={a.n_points}, cap={a.cap}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(lambda s: run(s, atlas, img.affine, a), subs) if not r.get("skipped")]
    if res:
        agg = {k: float(np.mean([r[k] for r in res])) for k in ("seg_per_streamline", "sc_r", "sc_r_log",
                                                                "gt_edge_recall", "sec", "mb")}
        agg["n_subjects"] = len(res)
        (ROOT / "outputs" / "stats").mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs" / "stats" / "edge_segments_summary.json").write_text(json.dumps({"mean": agg, "per_subject": res}, indent=1))
        print("평균:", json.dumps({k: round(v, 3) for k, v in agg.items()}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="outputs/subjects_train_eval_206.txt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-points", type=int, default=SEG_POINTS)
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--min-dwell", type=int, default=MIN_DWELL)
    ap.add_argument("--min-length", type=float, default=MIN_LEN_MM)
    ap.add_argument("--max-streamlines", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
