"""L_length (Framework v2 §13).

Length_pred(i,j) = sum_k w_k(i,j) L_k / (sum_k w_k(i,j) + eps)
  = Num / SC     (sc_builder 가 두 개를 같이 돌려준다)

GT edge 가 존재하는 곳에서만 계산한다. GT 가 0 인 edge 의 length 는 정의되지 않는다.
"""
from __future__ import annotations

import torch

from .sc_corr import upper

EPS = 1e-8


def predicted_length(num_pred: torch.Tensor, sc_pred: torch.Tensor) -> torch.Tensor:
    return num_pred / (sc_pred + EPS)


def tract_length_loss(num_pred, sc_pred, len_gt, sc_gt,
                      min_gt_weight: float = 0.0, masks: dict | None = None) -> torch.Tensor:
    """masks 를 주면 group(ctx/sub block) 별 masked 평균의 평균."""
    mask = (sc_gt > min_gt_weight) & (len_gt > 0)
    m = upper(mask.float())
    assert float(m.sum()) > 0, "GT edge mask 가 비어 있음"
    lp = predicted_length(num_pred, sc_pred)
    d = (torch.log1p(upper(lp).clamp(min=0)) - torch.log1p(upper(len_gt))).abs()
    if masks is None:
        return (d * m).sum() / m.sum()
    vals = []
    for gm in masks.values():
        mm = m * upper(gm.float())
        if float(mm.sum()) > 0:
            vals.append((d * mm).sum() / mm.sum())
    assert vals, "빈 mask"
    return torch.stack(vals).mean()
