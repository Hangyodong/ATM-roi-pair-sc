"""SC 재현 프로파일 — 상관 하나로 판정하지 않는다.

왜: 절대 r 0.810 / edge F1 0.93 / subject 간 상관 0.886 이 전부 좋아 보였는데, 실제로는
0 인 edge 를 0.7 % 만 맞추고, pair 의 3분의 2 에서 상관이 0.06~0.16 이고, 생성 subject 차이의
99.3 % 가 틀린 차이였다 (resid_r 0.081 -> 잔차 분산의 0.66 %). 크기 가중 상관이 큰 연결만
보여준 것이다.

GT 기준선 (100만 가닥 원본을 우리 hard_sc 로 재현, r=0.9987):
  가닥당 pair 기여 7.50 · 0 비율 0.230 · 비영 edge 2558
"""
from __future__ import annotations

import numpy as np

from .reproduction_metrics import _corr, loo_center


def tier_masks_by_mean(G: np.ndarray, n_tier: int = 3) -> dict:
    """GT 평균 크기로 나눈 구간. 크기 가중 상관이 가리는 작은 연결을 분리해서 본다."""
    gm = G.mean(0)
    q = np.quantile(gm, np.linspace(0, 1, n_tier + 1)[1:-1])
    names = ["small", "mid", "large"][:n_tier] if n_tier == 3 else [f"t{i}" for i in range(n_tier)]
    edges = np.concatenate([[-np.inf], q, [np.inf]])
    return {names[i]: (gm > edges[i]) & (gm <= edges[i + 1]) for i in range(n_tier)}


def profile(P: np.ndarray, G: np.ndarray, n_streamlines: np.ndarray | None = None) -> dict:
    """P, G: [n_subj, n_edge] 선형 SC. 판정에 필요한 것을 전부 낸다."""
    assert P.shape == G.shape and P.ndim == 2, (P.shape, G.shape)
    n = len(P)
    gz = G == 0
    out = {
        "n_subjects": n,
        # --- 희소 구조 (여기가 가장 크게 틀려 있었다) ---
        "zero_frac_gt": float(gz.mean()),
        "zero_frac_pred": float((P == 0).mean()),
        "zero_specificity": float(((P == 0) & gz).sum() / max(gz.sum(), 1)),
        "nonzero_edges_gt": float((G > 0).sum(1).mean()),
        "nonzero_edges_pred": float((P > 0).sum(1).mean()),
        # --- 분포 ---
        "sum_ratio": float((P.sum(1) / np.maximum(G.sum(1), 1)).mean()),
        "nonzero_median_gt": float(np.median(G[G > 0])),
        "nonzero_median_pred": float(np.median(P[P > 0])) if (P > 0).any() else 0.0,
        "top1pct_mass_gt": float(np.sort(G, 1)[:, -max(G.shape[1] // 100, 1):].sum(1).mean() / G.sum(1).mean()),
        "top1pct_mass_pred": float(np.sort(P, 1)[:, -max(P.shape[1] // 100, 1):].sum(1).mean() / max(P.sum(1).mean(), 1)),
        # --- 상관: 선형은 큰 값이 만든다. log/순위를 같이 본다 ---
        "r_linear": float(np.mean([_corr(P[i], G[i]) for i in range(n)])),
        "r_log": float(np.mean([_corr(np.log1p(P[i]), np.log1p(G[i])) for i in range(n)])),
        "r_spearman": float(np.mean([_corr(np.argsort(np.argsort(P[i])), np.argsort(np.argsort(G[i])))
                                     for i in range(n)])),
    }
    for k, m in tier_masks_by_mean(G).items():
        out[f"r_{k}"] = float(np.mean([_corr(P[i][m], G[i][m]) for i in range(n)]))
        out[f"zero_spec_{k}"] = float(((P[:, m] == 0) & gz[:, m]).sum() / max(gz[:, m].sum(), 1))
    # --- 개인차: resid_r 만 보면 안 된다. 분산비와 함께 봐야 "맞는 다양성" 인지 알 수 있다 ---
    Pc, Gc = loo_center(P), loo_center(G)
    rr = np.array([_corr(Pc[i], Gc[i]) for i in range(n)])
    r = float(np.nanmean(rr))
    vr = float(np.mean(Pc.std(1) / np.maximum(Gc.std(1), 1e-9)))
    iu = np.triu_indices(n, 1)
    out.update({
        "resid_r": r,
        "resid_var_ratio": vr,
        "correct_var_frac": r * r,          # 생성 subject 차이 중 실제로 맞는 몫
        "inter_subj_r_pred": float(np.corrcoef(P)[iu].mean()),
        "inter_subj_r_gt": float(np.corrcoef(G)[iu].mean()),
    })
    if n_streamlines is not None:
        out["pairs_per_streamline"] = float((P.sum(1) / np.maximum(np.asarray(n_streamlines, float), 1)).mean())
    return out


GT_REF = {"pairs_per_streamline": 7.50, "zero_frac": 0.230}


def verdict(p: dict) -> dict:
    """게이트. 하나라도 어기면 "SC 를 재현했다" 고 말하지 않는다."""
    g = {
        "희소성": (p["zero_specificity"] >= 0.5, f"0 특이도 {p['zero_specificity']:.3f} (>=0.5)"),
        "밀도": (abs(p["zero_frac_pred"] - p["zero_frac_gt"]) <= 0.05,
               f"0 비율 {p['zero_frac_pred']:.3f} vs GT {p['zero_frac_gt']:.3f}"),
        "총합": (0.9 <= p["sum_ratio"] <= 1.1, f"합비 {p['sum_ratio']:.3f}"),
        "작은연결": (p["r_small"] >= 0.3, f"small r {p['r_small']:.3f} (>=0.3)"),
        "중간연결": (p["r_mid"] >= 0.3, f"mid r {p['r_mid']:.3f} (>=0.3)"),
        "순위": (p["r_spearman"] >= 0.8, f"스피어만 {p['r_spearman']:.3f} (>=0.8)"),
        "개인차": (p["correct_var_frac"] >= 0.10,
                f"맞는 분산 {100*p['correct_var_frac']:.2f}% (>=10%)"),
    }
    if "pairs_per_streamline" in p:
        g["가닥당pair"] = (abs(p["pairs_per_streamline"] - GT_REF["pairs_per_streamline"]) <= 2.5,
                        f"{p['pairs_per_streamline']:.1f} vs GT {GT_REF['pairs_per_streamline']}")
    return {"gates": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in g.items()},
            "n_pass": int(sum(v[0] for v in g.values())), "n_total": len(g)}
