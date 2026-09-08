#!/usr/bin/env python
"""test 31명의 GT SC vs generated SC 를 fig 당 8명, 총 4장으로 그린다.

입력: scripts/49_eval_frozen.py 가 남긴 `outputs/eval/final_<ckpt>_vectors.npz`
      (gt / pred_generated / pred_best: [N, 3321] upper-triangle 벡터, subjects: [N]).
`pred_best` 는 --by-count 일 때 alloc 경로(잔차 변조 배분) = 실제 제안 추론 경로다.
색은 log1p(SC) 이고 **subject 마다 GT 와 generated 가 같은 색 범위**를 쓴다 (안 그러면 크기 차이가 안 보인다).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def to_mat(v, n_roi):
    m = np.zeros((n_roi, n_roi), np.float64)
    iu = np.triu_indices(n_roi, 1)
    m[iu] = v; m.T[iu] = v
    return m


def main(a):
    z = np.load(a.vectors, allow_pickle=False)
    gk = a.gt_key if a.gt_key in z.files else ("gt" if "gt" in z.files else "gt_sc")
    assert a.pred_key in z.files, f"{a.pred_key} 없음. 있는 키: {sorted(z.files)}"
    subs, G = z["subjects"], z[gk]
    P = z[a.pred_key]
    assert G.shape == P.shape and G.ndim == 2, (G.shape, P.shape)
    n_roi = int(round((1 + np.sqrt(1 + 8 * G.shape[1])) / 2))
    assert n_roi * (n_roi - 1) // 2 == G.shape[1], (n_roi, G.shape)
    assert np.isfinite(G).all() and np.isfinite(P).all(), "SC 벡터에 NaN/Inf"
    iu = np.triu_indices(n_roi, 1)
    out = ROOT / "outputs/figs"; out.mkdir(parents=True, exist_ok=True)
    per = a.per_fig
    n_fig = int(np.ceil(len(subs) / per))
    for f in range(n_fig):
        idx = list(range(f * per, min((f + 1) * per, len(subs))))
        fig, axes = plt.subplots(4, 4, figsize=(16, 15))
        axes = axes.ravel()
        for k, i in enumerate(idx):
            g, p = to_mat(G[i], n_roi), to_mat(P[i], n_roi)
            lg, lp = np.log1p(g), np.log1p(p)
            vmax = max(lg.max(), lp.max())
            r = float(np.corrcoef(G[i], P[i])[0, 1])
            for j, (m, ttl) in enumerate(((lg, "GT"), (lp, a.pred_label))):
                ax = axes[2 * k + j]
                im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_title(f"{subs[i]}  {ttl}" + (f"  r={r:.3f}" if j else ""), fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=axes[2 * k + 1], fraction=0.046, pad=0.02)
        for k in range(2 * len(idx), 16):
            axes[k].axis("off")
        fig.suptitle(f"test SC  log1p  —  GT vs {a.pred_label}   (fig {f + 1}/{n_fig}, subjects {idx[0] + 1}–{idx[-1] + 1})",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fp = out / f"test_sc_gt_vs_{a.pred_key}_{f + 1}of{n_fig}.png"
        fig.savefig(fp, dpi=110); plt.close(fig)
        assert fp.stat().st_size > 10_000, f"{fp} 가 비어 있다"
        print(f"저장: {fp.relative_to(ROOT)}  ({len(idx)}명)", flush=True)
    # 부수: 31명 전체 상관 요약 (그림에 적은 r 과 같은 정의)
    rs = [float(np.corrcoef(G[i], P[i])[0, 1]) for i in range(len(subs))]
    print(f"subject 별 r: 평균 {np.mean(rs):.4f}  최소 {np.min(rs):.4f}  최대 {np.max(rs):.4f}  (n={len(rs)})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="outputs/eval/final_<ckpt>_vectors.npz")
    ap.add_argument("--pred-key", default="pred_best")
    ap.add_argument("--gt-key", default="gt")
    ap.add_argument("--pred-label", default="generated (alloc)")
    ap.add_argument("--per-fig", type=int, default=8)
    sys.exit(main(ap.parse_args()))
