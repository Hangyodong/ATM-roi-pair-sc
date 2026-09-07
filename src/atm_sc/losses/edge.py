"""L_edge = BCE(P(edge), GT_edge)  (pipeline §10).

positive/negative 불균형을 반드시 확인한다 (§34). sub-000001 endpoint rule 기준 양성
pair 는 3321 개 중 약 절반이라 심하지 않지만, subject 와 rule 에 따라 달라지므로
pos_weight 를 밖에서 넘길 수 있게 둔다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_loss(logits: torch.Tensor, target: torch.Tensor,
              pos_weight: float | None = None) -> torch.Tensor:
    assert logits.shape == target.shape, (logits.shape, target.shape)
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pw)


def edge_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> dict:
    with torch.no_grad():
        p = (torch.sigmoid(logits) > thr).float(); t = target.float()
        tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
        return {"edge_acc": float((p == t).float().mean()),
                "edge_f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
                "pos_ratio": float(t.mean())}
