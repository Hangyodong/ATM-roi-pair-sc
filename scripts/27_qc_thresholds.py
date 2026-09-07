#!/usr/bin/env python
"""TRAIN real streamline 분포에서 synthetic QC 임계값을 만든다 (GESTA QC 문서 §11, §16).

  python scripts/27_qc_thresholds.py --subjects outputs/splits/train.txt --n-per-subject 3000

임의 상수 대신 "real 의 몇 %가 통과하는가"로 정의한다. 결과: outputs/stats/qc_thresholds.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.tt_io import point_labels                                  # noqa: E402
from atm_sc.filtering.qc_thresholds import (DEFAULT_PATH, end_to_end_ratio,  # noqa: E402
                                            measure, winding_deg)
from atm_sc.filtering.t1_streamline_filter import max_turn_angles_deg, streamline_lengths_mm  # noqa: E402
from atm_sc.models.roi_atm import TEMPLATE_MASK                             # noqa: E402


def main(a):
    import nibabel as nib
    rng = np.random.default_rng(a.seed)
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (ROOT / "outputs/roi_pairs" / s / "bundles.npz").exists()][: a.limit or None]
    mimg = nib.load(TEMPLATE_MASK); mask = np.asanyarray(mimg.dataobj) > 0
    S_all, ins_all = [], []
    for s in subs:
        z = np.load(ROOT / "outputs/roi_pairs" / s / "bundles.npz")
        St = z["streamlines"]
        idx = rng.choice(len(St), min(a.n_per_subject, len(St)), replace=False)
        S = St[np.sort(idx)].astype(np.float64)
        S_all.append(S)
        ins = point_labels(S.reshape(-1, 3), mask, mimg.affine) > 0
        ins_all.append(ins.reshape(len(S), -1).mean(1))
    S = np.concatenate(S_all); ins = np.concatenate(ins_all)
    th = measure(S, ins, q=(a.q_low, a.q_high))
    p = th.save(ROOT / a.out)
    L, A, W, R = streamline_lengths_mm(S), max_turn_angles_deg(S), winding_deg(S), end_to_end_ratio(S)
    print(f"TRAIN real {len(S):,} 가닥 ({len(subs)}명) 분포:")
    for name, v, unit in (("길이", L, "mm"), ("최대 꺾임각", A, "도"), ("총 회전량", W, "도"),
                          ("직선비", R, ""), ("뇌 안 비율", ins, "")):
        q = np.percentile(v, [1, 5, 50, 95, 99])
        print(f"  {name:10s} p1/p5/p50/p95/p99 = " + " / ".join(f"{x:.2f}{unit}" for x in q))
    print(f"\n임계값 (q={a.q_low}~{a.q_high}) -> {p}")
    for k, v in th.__dict__.items():
        print(f"  {k}: {v}")
    keep = ((L >= th.min_length_mm) & (L <= th.max_length_mm) & (A <= th.max_turn_deg)
            & (W <= th.max_winding_deg) & (R >= th.min_end_ratio) & (ins >= th.brain_inside_min))
    print(f"\n이 임계값으로 real 자신의 통과율: {keep.mean():.3f} (설계상 약 {(1 - 5 * (a.q_low / 100)):.2f} 이상이어야 함)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="outputs/splits/train.txt")
    ap.add_argument("--n-per-subject", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--q-low", type=float, default=1.0)
    ap.add_argument("--q-high", type=float, default=99.0)
    ap.add_argument("--out", default="outputs/stats/qc_thresholds.json")
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
