"""Route / ROI-visitation loss (ATM_ROUTE_LOSS_PASS_SC_GESTA_DETAILED_STRATEGY.md §12–21, §54).

endpoint loss 는 "어디서 출발해 어디로 도착했나" 만 본다. GT SUB-SUB 연결의 63 % 는 그 두 ROI 를
끝점으로 갖는 streamline 이 아니라 **다른 bundle 이 지나가며** 만들어지는 pass-edge 이므로,
중간 통과 ROI 를 직접 감독하지 않으면 학습 신호가 없다 (§3, §19).

    v_GT   ∈ {0,1}^R    GT streamline 이 지나간 ROI (hard atlas, GT pass-SC 와 같은 규칙)
    v_pred ∈ [0,1]^R    생성/복원 streamline 의 soft 통과 확률 (EndpointAssigner.visit_probs)

확률이 아니라 **log 확률**을 입력받는다: 목표 ROI 가 수십 mm 떨어지면 exp(-d/tau) 가 float32 아래로
내려가 clamp 에 걸리고 gradient 가 0 이 된다 (endpoint loss 에서 실측된 버그와 같은 원인).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sc_corr import upper

EPS = 1e-8


def _log1m_exp(log_u: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(log_u)), log_u <= 0. u -> 1 일 때만 위험하므로 그쪽만 clamp."""
    return torch.log1p(-torch.exp(log_u).clamp(max=1.0 - 1e-6))


def route_bce(log_u: torch.Tensor, y: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """log_u [N,R] 통과 log-확률, y [N,R] multi-hot(또는 [0,1] soft target) -> BCE (§15)."""
    assert log_u.shape == y.shape, (log_u.shape, y.shape)
    assert float(log_u.max()) <= 1e-4, "log_u 는 log-확률이어야 한다 (<=0)"
    pos, neg = -log_u, -_log1m_exp(log_u)
    w = 1.0 if pos_weight is None else float(pos_weight)
    return (w * y * pos + (1 - y) * neg).mean()


def route_dice(log_u: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - 2<u,y>/(sum u + sum y). streamline 별로 계산 후 평균 (§15).

    eps 는 0 나눗셈 방지용만 (통과 ROI 는 항상 >=1 개). eps=1 같은 smoothing 을 쓰면 통과 ROI 가
    ~5개뿐이라 완벽한 예측에서도 손실이 0.14 로 남는다.
    """
    assert log_u.shape == y.shape, (log_u.shape, y.shape)
    u = torch.exp(log_u)
    num = 2 * (u * y).sum(-1)
    den = u.sum(-1) + y.sum(-1) + eps
    return (1 - num / den).mean()


def route_loss(log_u: torch.Tensor, y: torch.Tensor, mode: str = "bce",
               pos_weight: float | None = None) -> torch.Tensor:
    if mode == "bce":
        return route_bce(log_u, y, pos_weight)
    if mode == "dice":
        return route_dice(log_u, y)
    if mode == "both":
        return route_bce(log_u, y, pos_weight) + route_dice(log_u, y)
    raise ValueError(mode)


@torch.no_grad()
def route_metrics(log_u: torch.Tensor, y: torch.Tensor, thr: float = 0.3) -> dict:
    """gradient 없는 진단: 통과 ROI 의 precision / recall / F1 / Jaccard (§54, §64)."""
    p = (torch.exp(log_u) >= thr).float()
    t = (y >= 0.5).float()
    tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
    return {"route_precision": float(tp / (tp + fp + EPS)), "route_recall": float(tp / (tp + fn + EPS)),
            "route_f1": float(2 * tp / (2 * tp + fp + fn + EPS)),
            "route_jaccard": float(tp / (tp + fp + fn + EPS)),
            "route_pred_visits": float(p.sum(-1).mean()), "route_gt_visits": float(t.sum(-1).mean())}


def pass_presence_loss(sc_pred: torch.Tensor, sc_gt: torch.Tensor, mask: torch.Tensor | None = None,
                       scale: float = 1.0) -> torch.Tensor:
    """"이 pass-edge 가 존재하는가" 를 strength 와 분리해 감독한다 (§20–21).

    P(edge) = 1 - exp(-SC_pred/scale)  (Poisson 발생 확률; SC_pred=0 -> 0, 커지면 1 로 포화)
    target  = 1[SC_GT > 0]
    SUB-SUB 처럼 값이 작은 block 에서는 magnitude 보다 존재 여부가 먼저 맞아야 한다.
    """
    p, g = upper(sc_pred), upper(sc_gt)
    if mask is not None:
        m = upper(mask).bool()
        assert m.any(), "presence mask 가 비어 있음"
        p, g = p[m], g[m]
    lam = p.clamp(min=0) / float(scale)
    log_p = torch.log(-torch.expm1(-lam.clamp(min=1e-12)) + EPS)      # log(1 - exp(-lam))
    log_1mp = -lam                                                     # log(exp(-lam))
    y = (g > 0).float()
    return -(y * log_p + (1 - y) * log_1mp).mean()


@torch.no_grad()
def presence_metrics(sc_pred: torch.Tensor, sc_gt: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    p, g = upper(sc_pred), upper(sc_gt)
    if mask is not None:
        m = upper(mask).bool(); p, g = p[m], g[m]
    pred, t = (p > 0).float(), (g > 0).float()
    tp = (pred * t).sum(); fp = (pred * (1 - t)).sum(); fn = ((1 - pred) * t).sum()
    return {"presence_recall": float(tp / (tp + fn + EPS)), "presence_precision": float(tp / (tp + fp + EPS)),
            "presence_f1": float(2 * tp / (2 * tp + fp + fn + EPS))}


def visitation_from_labels(labels: torch.Tensor, n_roi: int) -> torch.Tensor:
    """[N, T] 0=배경, 1..R 라벨 -> [N, R] multi-hot. GT 전처리(script 24) 와 같은 규칙."""
    assert labels.ndim == 2, labels.shape
    y = torch.zeros(labels.shape[0], n_roi, device=labels.device)
    idx = labels.long() - 1
    valid = idx >= 0
    rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(idx)
    y[rows[valid], idx[valid]] = 1.0
    return y


_ = F
