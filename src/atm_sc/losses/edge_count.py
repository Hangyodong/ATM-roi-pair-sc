"""L_count = MAE(log1p(N_pred), log1p(N_GT))  (strategy §22).

sc_magnitude_loss 는 normalize='sum' 으로 총합을 맞추고 비교하므로 절대 스케일이 loss 에서
빠진다. 여기서는 rescale 없이 count 를 그대로 비교하므로 "10 배 크게 예측" 이 실제로 벌을
받는다 (그게 CCC 를 올리는 항이다).

log1p 도메인인 이유: count 는 heavy tail 이고 GT 0 도 흔하다. log1p 는 0 에서 정의되므로
eps 를 끼워 넣지 않아도 되고, 상위 몇 개 edge 가 loss 를 독점하지 않는다.

log1p(exp(x)) == softplus(x) 라 예측 쪽은 softplus 로 계산한다 (x 가 크면 exp 가 overflow).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .sc_corr import upper

EPS = 1e-8


def _targets(gt_count: torch.Tensor, zero_target: float | None,
             dtype: torch.dtype) -> torch.Tensor:
    t = torch.log1p(gt_count.to(dtype))
    if zero_target is None:
        return t
    # GT 0 edge 의 목표를 정확히 0 으로 두면 log_pred 를 -inf 로 미는 항이 되어 head 가
    # 발산한다. zero_target(=count 단위 바닥값) 을 주면 "1 개 미만" 정도까지만 누르고 멈춘다.
    assert zero_target >= 0, zero_target
    return torch.where(gt_count > 0, t, torch.full_like(t, math.log1p(zero_target)))


def _grouped(d: torch.Tensor, masks: dict | None) -> torch.Tensor:
    """masks 가 있으면 group 별 평균의 평균. sub-sub(120 edge) 가 ctx-ctx(2145 edge) 에 묻히지 않는다."""
    if masks is None:
        return d.mean()
    vals = []
    for name, m in masks.items():
        mm = torch.as_tensor(m, device=d.device).bool().reshape(-1)
        assert mm.shape == d.shape, (name, mm.shape, d.shape)
        assert bool(mm.any()), f"빈 mask: {name}"
        vals.append(d[mm].mean())
    assert vals, "mask dict 가 비어 있음"
    return torch.stack(vals).mean()


def edge_count_loss(log_pred: torch.Tensor, gt_count: torch.Tensor, masks: dict | None = None,
                    zero_target: float | None = None) -> torch.Tensor:
    """log_pred [K] (자연로그 예측), gt_count [K] (>= 0, 0 가능). masks = {name: bool [K]}."""
    assert log_pred.ndim == 1 and gt_count.ndim == 1, (log_pred.shape, gt_count.shape)
    assert log_pred.shape == gt_count.shape, (log_pred.shape, gt_count.shape)
    assert torch.isfinite(log_pred).all(), "log_pred 에 NaN/Inf"
    assert torch.isfinite(gt_count).all(), "gt_count 에 NaN/Inf"
    assert bool((gt_count >= 0).all()), "count 가 음수"
    # dtype 을 float32 로 못 박지 않는다: float64 로 부르면 "완전 일치 -> loss 0" 을
    # 반올림 오차 없이 확인할 수 있어야 한다.
    assert log_pred.is_floating_point(), log_pred.dtype
    d = (F.softplus(log_pred) - _targets(gt_count, zero_target, log_pred.dtype)).abs()
    return _grouped(d, masks)


def edge_count_matrix_loss(log_pred_mat: torch.Tensor, gt_mat: torch.Tensor,
                           masks: dict | None = None) -> torch.Tensor:
    """[R,R] 판. upper triangle 만 쓴다. masks 는 [R,R] bool (data.roi_groups.block_masks)."""
    assert log_pred_mat.shape == gt_mat.shape, (log_pred_mat.shape, gt_mat.shape)
    p, g = upper(log_pred_mat), upper(gt_mat)
    um = None if masks is None else {k: upper(torch.as_tensor(m, device=g.device).bool())
                                     for k, m in masks.items()}
    return edge_count_loss(p, g, masks=um)


def _flat(x: torch.Tensor) -> torch.Tensor:
    """[K] 또는 [R,R] -> [K]. 행렬이면 upper triangle."""
    x = torch.as_tensor(x).double()
    return upper(x) if (x.ndim == 2 and x.shape[0] == x.shape[1]) else x.reshape(-1)


@torch.no_grad()
def edge_count_metrics(pred_count: torch.Tensor, gt_count: torch.Tensor) -> dict:
    """count 단위(로그 아님) 예측/GT -> 지표. pred/gt 는 [K] 또는 [R,R].

    sc_metrics 는 [R,R] 만 받는데(내부에서 upper) count head 는 임의 길이 pair 목록을 채점하므로
    공식만 동일하게(CCC 는 correction=0) 맞춰 여기서 계산한다.
    """
    p, g = _flat(pred_count), _flat(gt_count)
    assert p.shape == g.shape, (p.shape, g.shape)
    assert torch.isfinite(p).all() and torch.isfinite(g).all(), "NaN/Inf"
    assert bool((g >= 0).all()), "GT count 가 음수"
    lp, lg = torch.log1p(p.clamp(min=0)), torch.log1p(g)
    r = torch.corrcoef(torch.stack([p, g]))[0, 1]
    rl = torch.corrcoef(torch.stack([lp, lg]))[0, 1]
    mp, mg = p.mean(), g.mean()
    vp, vg = p.var(correction=0), g.var(correction=0)
    ccc = 2 * ((p - mp) * (g - mg)).mean() / (vp + vg + (mp - mg) ** 2 + EPS)
    # 1 이 임계값인 이유: count 가 1 미만이면 생성할 segment 가 0 개다 (§21 에서 count 가 곧 개수).
    zpos, znull = g > 0, g == 0
    nan = float("nan")
    return {"r": float(r), "r_log": float(rl), "ccc": float(ccc),
            "mae": float((p - g).abs().mean()), "log_mae": float((lp - lg).abs().mean()),
            "zero_specificity": float((p[znull] < 1).double().mean()) if bool(znull.any()) else nan,
            "recall": float((p[zpos] >= 1).double().mean()) if bool(zpos.any()) else nan,
            "n_edges": int(p.numel())}
