"""Streamline 단위 supervision (Framework v2 §7).

L_endpoint : predicted streamline 의 양 끝점이 GT 와 같은 ROI pair 를 잇도록.
             tractography 는 방향이 없으므로 (i,j) 와 (j,i) 를 같게 취급한다.
L_roi_visit: 이 프로젝트의 GT SC 는 endpoint 가 아니라 pass 정의라서 (실측 r 0.673 vs
             0.9986) endpoint 만 맞춰서는 GT SC 를 재현할 수 없다. 통과 ROI 집합을
             직접 감독하는 항을 함께 제공한다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def endpoint_loss(q_start, q_end, i_gt, j_gt, symmetric: bool = True, log_input: bool = False):
    """q_* [N,R] 확률(또는 log_input=True 면 log-확률), i_gt/j_gt [N] 0-based ROI 인덱스.

    학습에서는 반드시 EndpointAssigner.endpoint_log_probs + log_input=True 를 쓴다.
    확률을 clamp 해서 log 를 취하면 끝점이 목표에서 멀 때 gradient 가 0 이 된다.
    """
    assert q_start.shape == q_end.shape, (q_start.shape, q_end.shape)
    assert i_gt.shape == j_gt.shape == (q_start.shape[0],), (i_gt.shape, j_gt.shape)
    if log_input:
        ls, le = q_start, q_end
    else:
        ls, le = torch.log(q_start.clamp_min(EPS)), torch.log(q_end.clamp_min(EPS))
    fwd = F.nll_loss(ls, i_gt, reduction="none") + F.nll_loss(le, j_gt, reduction="none")
    if not symmetric:
        return fwd.mean()
    rev = F.nll_loss(ls, j_gt, reduction="none") + F.nll_loss(le, i_gt, reduction="none")
    return torch.minimum(fwd, rev).mean()


def endpoint_accuracy(q_start, q_end, i_gt, j_gt) -> dict:
    """보고용. ROI 정확도와 pair 정확도 (순서 무시)."""
    with torch.no_grad():
        a, b = q_start.argmax(1), q_end.argmax(1)
        roi = 0.5 * ((a == i_gt).float().mean() + (b == j_gt).float().mean())
        pair = (((a == i_gt) & (b == j_gt)) | ((a == j_gt) & (b == i_gt))).float().mean()
        return {"endpoint_roi_acc": float(roi), "endpoint_pair_acc": float(pair)}


def roi_visit_loss(u: torch.Tensor, y: torch.Tensor, pos_weight: float | None = None):
    """u [N,R] 통과 확률, y [N,R] multi-hot GT -> BCE. GT SC(pass) 와 같은 양을 감독."""
    assert u.shape == y.shape, (u.shape, y.shape)
    u = u.clamp(EPS, 1 - EPS)
    if pos_weight is None:
        return F.binary_cross_entropy(u, y)
    w = torch.where(y > 0, torch.full_like(y, pos_weight), torch.ones_like(y))
    return (F.binary_cross_entropy(u, y, reduction="none") * w).mean()
