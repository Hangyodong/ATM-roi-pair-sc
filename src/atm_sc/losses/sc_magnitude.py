"""L_SC_mag (Framework v2 §12).

correlation 만 최적화하면 스케일이 틀려도 r=1 이 가능하다 (v2 §11).
게다가 이 프로젝트에서는 GT 가 1,000,000 streamline 기준인데 ATM 은 bundle 당 3000 개
(30 bundle 이면 90,000) 를 만들어 총량이 10배 이상 다르다. 따라서 normalize='sum' 으로
총합을 맞춘 뒤 비교하는 것이 기본값이다. normalize='none' 이면 그 상수 offset 이
그대로 loss 에 들어간다.
"""
from __future__ import annotations

import torch

from .sc_corr import upper

EPS = 1e-8


def rescale(pred: torch.Tensor, gt: torch.Tensor, mode: str = "sum") -> torch.Tensor:
    if mode == "none":
        return pred
    if mode == "sum":
        return pred * (gt.sum() / (pred.sum() + EPS))
    raise ValueError(mode)


def sc_magnitude_loss(sc_pred, sc_gt, normalize: str = "sum", masks: dict | None = None) -> torch.Tensor:
    """masks 를 주면 group 별 평균의 평균: sub-sub(120 edge) 오차가 3321 edge 평균에 묻히지 않는다."""
    p = rescale(sc_pred, sc_gt, normalize)
    d = (torch.log1p(upper(p).clamp(min=0)) - torch.log1p(upper(sc_gt))).abs()
    if masks is None:
        return d.mean()
    vals = [d[upper(m).bool()].mean() for m in masks.values() if bool(upper(m).any())]
    assert vals, "빈 mask"
    return torch.stack(vals).mean()


def sc_scale_loss(sc_pred, sc_gt, masks: dict | None = None) -> torch.Tensor:
    """총합의 로그 차이 |log(sum pred) - log(sum gt)|. 스칼라 하나로 전역 배율을 직접 학습한다.

    실측(test sub-101070, phase6 checkpoint): 예측 합 161,056 vs GT 7,673,899 (48배).
    배율 하나만 곱하면 CCC 가 0.024 -> 0.817 로 뛴다. 즉 절대 스케일 문제의 대부분이 이 상수다.
    총합 정규화된 magnitude loss 는 이 상수를 볼 수 없으므로 별도 항이 필요하다.
    """
    def one(p, g):
        return (torch.log(p.clamp(min=0).sum() + EPS) - torch.log(g.sum() + EPS)).abs()
    if masks is None:
        return one(upper(sc_pred), upper(sc_gt))
    vals = [one(upper(sc_pred)[upper(m).bool()], upper(sc_gt)[upper(m).bool()])
            for m in masks.values() if bool(upper(m).any())]
    assert vals, "빈 mask"
    return torch.stack(vals).mean()


def sc_rmse_loss(sc_pred, sc_gt, masks: dict | None = None, normalize: str = "gt_std") -> torch.Tensor:
    """SC 행렬의 RMSE. 절대 오차를 직접 본다.

    raw RMSE 는 heavy tail 때문에 상위 1 % edge 가 오차의 32 % 를 차지한다(실측). 그래서 기본값은
    GT 표준편차로 나눠 무차원화하고(다른 항과 자릿수를 맞춤), group 별 평균의 평균을 쓴다.
    normalize: 'gt_std' | 'none' | 'log'(log1p 공간 RMSE, tail 에 가장 둔감)
    """
    def one(p, g):
        if normalize == "log":
            d = torch.log1p(p.clamp(min=0)) - torch.log1p(g)
            return (d ** 2).mean().sqrt()
        d = (p - g) ** 2
        r = d.mean().sqrt()
        if normalize == "gt_std":
            return r / (g.std(correction=0) + EPS)
        assert normalize == "none", normalize
        return r
    if masks is None:
        return one(upper(sc_pred), upper(sc_gt))
    vals = [one(upper(sc_pred)[upper(m).bool()], upper(sc_gt)[upper(m).bool()])
            for m in masks.values() if bool(upper(m).any())]
    assert vals, "빈 mask"
    return torch.stack(vals).mean()
