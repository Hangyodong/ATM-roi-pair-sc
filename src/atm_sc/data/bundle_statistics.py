"""ROI-pair bundle 크기 불균형 통계 (GESTA 전략 §14, §65). 학습 split 만 사용한다 (§20 leakage 방지)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .roi_groups import BLOCKS, TIERS, block_of_pairs, tier_of_strength

ROOT = Path(__file__).resolve().parents[3]
SIZE_EDGES = (20, 256)          # N_ij: low < 20 (KDE 금지, fallback) / mid / high ≥ 256 (cap)


def size_bin_of(counts: np.ndarray, edges=SIZE_EDGES) -> np.ndarray:
    c = np.asarray(counts)
    return (c >= edges[0]).astype(np.int64) + (c >= edges[1]).astype(np.int64)


def bundle_statistics(subjects: list[str], roi_pairs_dir: Path = ROOT / "outputs" / "roi_pairs") -> dict:
    """subject 별 bundles.npz 의 pair_count_full 과 .mat pass-SC 값으로 분포·prevalence·block/size/tier 수를 만든다."""
    from .dataset import load_mat_gt
    counts, strengths, blocks, prev = [], [], [], {}
    per_subject = []
    for s in subjects:
        z = np.load(roi_pairs_dir / s / "bundles.npz")
        c = z["pair_count_full"].astype(np.int64); pid = np.asarray(z["pair_ids"], np.int64)
        assert len(c) == len(pid) and (c >= 1).all(), s
        w, _ = load_mat_gt(s)
        st = w[pid[:, 0], pid[:, 1]]
        counts.append(c); strengths.append(st); blocks.append(block_of_pairs(pid))
        for p in map(tuple, pid.tolist()):
            prev[p] = prev.get(p, 0) + 1
        sb = size_bin_of(c)
        per_subject.append({"subject": s, "n_pairs": int(len(c)), "n_streamlines": int(c.sum()),
                            "size_bins": {b: int((sb == i).sum()) for i, b in enumerate(("low", "mid", "high"))}})
    c = np.concatenate(counts); st = np.concatenate(strengths); bl = np.concatenate(blocks)
    sb, tr = size_bin_of(c), tier_of_strength(st)
    q = [0, 5, 25, 50, 75, 95, 100]
    hist_edges = [1, 2, 5, 10, 20, 50, 100, 256, 1000, 5000, 100000]
    pv = np.array(list(prev.values()))
    out = {"n_subjects": len(subjects), "n_bundle_pairs": int(len(c)),
           "count_quantiles": {f"p{k}": float(v) for k, v in zip(q, np.percentile(c, q))},
           "count_histogram": {f"[{a},{b})": int(((c >= a) & (c < b)).sum()) for a, b in zip(hist_edges[:-1], hist_edges[1:])},
           "size_bins": {b: {"n_pairs": int((sb == i).sum()), "streamline_share": float(c[sb == i].sum() / c.sum())}
                         for i, b in enumerate(("low", "mid", "high"))},
           "size_edges": list(SIZE_EDGES),
           "blocks": {b: {"n_pairs": int((bl == i).sum()), "streamline_share": float(c[bl == i].sum() / c.sum()),
                          "median_count": float(np.median(c[bl == i])) if (bl == i).any() else None} for i, b in enumerate(BLOCKS)},
           "strength_tiers": {t: {"n_pairs": int((tr == i).sum()), "median_count": float(np.median(c[tr == i])) if (tr == i).any() else None}
                              for i, t in enumerate(TIERS)},
           "pair_prevalence": {"n_unique_pairs": int(len(prev)),
                               "ge_50pct_subjects": int((pv >= len(subjects) / 2).sum()),
                               "ge_10pct_subjects": int((pv >= len(subjects) / 10).sum()),
                               "single_subject": int((pv == 1).sum())},
           "per_subject": per_subject}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default="outputs/splits/train.txt")
    ap.add_argument("--out", default="outputs/stats/bundle_statistics.json")
    a = ap.parse_args()
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    st = bundle_statistics(subs)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in st.items() if k != "per_subject"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
