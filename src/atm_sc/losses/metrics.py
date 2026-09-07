"""평가 지표 (Framework v2 §26). gradient 없음."""
from __future__ import annotations

import torch

from .sc_corr import upper

EPS = 1e-8


def sc_metrics(sc_pred: torch.Tensor, sc_gt: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    """mask ([R,R] bool) 를 주면 그 edge 들만 (block/tier 별 지표)."""
    with torch.no_grad():
        p, g = upper(sc_pred).double(), upper(sc_gt).double()
        if mask is not None:
            mm = upper(torch.as_tensor(mask, device=sc_pred.device)).bool()
            p, g = p[mm], g[mm]
        r = torch.corrcoef(torch.stack([p, g]))[0, 1]
        rl = torch.corrcoef(torch.stack([torch.log1p(p.clamp(min=0)), torch.log1p(g)]))[0, 1]
        # 공분산은 편향(1/n) 추정이므로 분산도 같은 1/n 으로 맞춰야 한다.
        # torch.var 의 기본값(unbiased, 1/(n-1))을 쓰면 동일 입력에서도 CCC 가
        # (n-1)/n 로 나온다.
        mp, mg = p.mean(), g.mean()
        vp, vg = p.var(correction=0), g.var(correction=0)
        ccc = 2 * ((p - mp) * (g - mg)).mean() / (vp + vg + (mp - mg) ** 2 + EPS)
        tp = ((p > 0) & (g > 0)).sum(); fp = ((p > 0) & (g == 0)).sum(); fn = ((p == 0) & (g > 0)).sum()
        return {"r": float(r), "r_log": float(rl), "ccc": float(ccc),
                "mae": float((p - g).abs().mean()), "rmse": float(((p - g) ** 2).mean().sqrt()),
                "edge_f1": float(2 * tp / (2 * tp + fp + fn + EPS)),
                "log_mae": float((torch.log1p(p.clamp(min=0)) - torch.log1p(g)).abs().mean()),
                "density": float((p > 0).float().mean()), "n_edges": int(p.numel())}


def sc_group_metrics(sc_pred: torch.Tensor, sc_gt: torch.Tensor, groups: dict) -> dict:
    """{group: sc_metrics(...)}  groups = {name: [R,R] bool} (data.roi_groups.block_masks / tier_masks)."""
    return {name: sc_metrics(sc_pred, sc_gt, mask=m) for name, m in groups.items()}
