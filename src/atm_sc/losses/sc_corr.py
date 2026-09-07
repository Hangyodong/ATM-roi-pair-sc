"""L_SC_corr = 1 - Pearson(SC_pred, SC_gt)  (Framework v2 §10).

upper triangle 만 쓴다. 분산이 0 이면 상관이 정의되지 않으므로 그 경우 gradient 가
NaN 이 되지 않도록 분모에 eps 를 두고, 완전히 상수인 입력은 assert 로 막는다.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def upper(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[0] == x.shape[1], x.shape
    i, j = torch.triu_indices(x.shape[0], x.shape[1], offset=1, device=x.device)
    return x[i, j]


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a * b).sum() / (a.norm() * b.norm() + EPS)


def sc_corr_loss(sc_pred: torch.Tensor, sc_gt: torch.Tensor,
                 mask: torch.Tensor | None = None, log: bool = False) -> torch.Tensor:
    """log=True 면 log1p 도메인 Pearson. raw count 는 heavy tail (상위 5 % edge 가 질량 50 %) 이라
    큰 edge 몇 개가 r 을 결정하므로, 작은 edge 도 패턴 loss 에 들어가게 하려면 log 를 쓴다."""
    p, g = upper(sc_pred), upper(sc_gt)
    if mask is not None:
        m = upper(mask).bool()
        assert m.any(), "valid edge mask 가 비어 있음"
        p, g = p[m], g[m]
    if log:
        p, g = torch.log1p(p.clamp(min=0)), torch.log1p(g)
    assert float(g.var()) > 0, "SC_gt 의 분산이 0 — 상관을 정의할 수 없음"
    return 1.0 - pearson(p, g)


def sc_corr_group_loss(sc_pred: torch.Tensor, sc_gt: torch.Tensor, masks: dict,
                       weights: dict | None = None, log: bool = True):
    """group(ctx-ctx / ctx-sub / sub-sub 등) 별 (1 - r) 의 가중 평균 -> (loss, {group: r}).
    whole-brain 하나로 보면 sub-sub(질량 0.9 %) 가 ctx-ctx(87 %) 에 묻히므로 group 마다 따로 상관을 잰다.
    GT 분산이 0 인 group 은 건너뛴다 (r = nan)."""
    tot, wsum, rs = 0.0, 0.0, {}
    for name, m in masks.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        g = upper(sc_gt)[upper(m).bool()]
        if w <= 0 or g.numel() < 3 or float(g.var()) == 0:
            rs[name] = float("nan")
            continue
        l = sc_corr_loss(sc_pred, sc_gt, mask=m, log=log)
        rs[name] = 1.0 - float(l)
        tot = tot + w * l
        wsum += w
    assert wsum > 0, "유효한 group 이 없음"
    return tot / wsum, rs


class ResidualCorr:
    """subject 평균을 뺀 **개인차**에 대한 Pearson 손실.

    왜 필요한가 (실측): P2 에서 count 손실만으로 잔차 상관이 0.026 -> 0.124 로 올랐는데,
    P3 에서 corr/mag/scale 을 켜자 0.009 로 무너졌다. SC 행렬 분산의 90 %는 **쌍 간 변동**
    (어느 연결이 굵은가)이고 그건 그룹 템플릿이 이미 맞히는 부분이라, corr 손실의 기울기가
    대부분 "템플릿 패턴을 더 잘 맞춰라" 로 간다. 개인차(전체 분산의 10.6 %)는 묻힌다.

    한 step 에 subject 하나만 처리하므로 subject 간 평균을 **이동평균(EMA)** 으로 유지하고
    그것으로 중심화한다. EMA 는 train 템플릿에서 시작한다 -- 모델의 초기 출력이 정확히
    템플릿이므로(템플릿 인수분해) 시작 시점에 잔차가 0 이고 편향이 없다.

    EMA 는 통계량이지 파라미터가 아니므로 gradient 를 흘리지 않는다 (detach).
    """

    def __init__(self, template_pred: torch.Tensor, template_gt: torch.Tensor | None = None,
                 momentum: float = 0.02, log: bool = True, warmup: int = 50):
        t = template_pred.detach().float()
        assert t.ndim == 1 and t.numel() > 2, f"upper-triangle 벡터를 기대한다: {tuple(t.shape)}"
        self.ema_p = t.clone()
        self.ema_g = (template_gt.detach().float().clone() if template_gt is not None else t.clone())
        assert self.ema_g.shape == self.ema_p.shape, (self.ema_g.shape, self.ema_p.shape)
        self.m, self.log, self.warmup, self.n = float(momentum), bool(log), int(warmup), 0

    @torch.no_grad()
    def _update(self, p: torch.Tensor, g: torch.Tensor):
        self.ema_p.mul_(1 - self.m).add_(p.detach().float(), alpha=self.m)
        self.ema_g.mul_(1 - self.m).add_(g.detach().float(), alpha=self.m)
        self.n += 1

    def __call__(self, sc_pred: torch.Tensor, sc_gt: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        p, g = upper(sc_pred), upper(sc_gt)
        if self.log:
            p, g = torch.log1p(p.clamp(min=0)), torch.log1p(g)
        return self.on_upper(p, g, None if mask is None else upper(mask).bool())

    def on_upper(self, p: torch.Tensor, g: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """이미 upper-triangle 로 펴고 도메인 변환까지 끝낸 벡터용 진입점.

        count head 는 [R,R] 를 만들지 않고 softplus(log_count) = log1p 예측을 바로 주므로
        행렬을 되조립하지 않고 여기로 들어온다. `__call__` 이 이 함수로 환원되므로 기존 경로의
        수치는 바뀌지 않는다.
        """
        assert p.ndim == 1 and p.shape == g.shape, (p.shape, g.shape)
        assert p.shape == self.ema_p.shape, (p.shape, self.ema_p.shape)
        ep, eg = self.ema_p, self.ema_g
        self._update(p, g)
        if self.n <= self.warmup:                 # EMA 가 자리잡기 전에는 기울기를 흘리지 않는다
            return p.sum() * 0.0
        rp, rg = p - ep, g - eg
        if mask is not None:
            assert mask.any(), "valid edge mask 가 비어 있음"
            rp, rg = rp[mask], rg[mask]
        if float(rg.var()) < 1e-12:               # 이 subject 가 평균과 같으면 배울 개인차가 없다
            return p.sum() * 0.0
        return 1.0 - pearson(rp, rg)


# ─────────────────────────── 전략 문서 §4: 잔차/차분 손실 ───────────────────────────
# 여기부터는 **고정된 train 템플릿**으로 정규화한 잔차를 직접 다룬다. `ResidualCorr` 의 EMA 와
# 달리 warmup 도 초기값 편향도 없다 (템플릿이 train split 통계라 누수도 없다).


def residual_smooth_l1(d_pred: torch.Tensor, d_gt: torch.Tensor,
                       mask: torch.Tensor | None = None, beta: float = 1.0) -> torch.Tensor:
    """L_res = SmoothL1(pred_delta, gt_delta). 희소 edge/이상치에 MSE 보다 둔감하다 (§4.1)."""
    assert d_pred.shape == d_gt.shape, (d_pred.shape, d_gt.shape)
    if mask is not None:
        assert mask.any(), "잔차 edge mask 가 비어 있음"
        d_pred, d_gt = d_pred[mask], d_gt[mask]
    return torch.nn.functional.smooth_l1_loss(d_pred, d_gt.detach(), beta=beta)


def residual_corr_loss(d_pred: torch.Tensor, d_gt: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
    """L_corr = 1 - corr(pred_delta, gt_delta) (§4.2). GT 잔차 분산이 0 이면 기울기를 끊는다."""
    assert d_pred.shape == d_gt.shape, (d_pred.shape, d_gt.shape)
    if mask is not None:
        assert mask.any(), "잔차 edge mask 가 비어 있음"
        d_pred, d_gt = d_pred[mask], d_gt[mask]
    if float(d_gt.var()) < 1e-12:
        return d_pred.sum() * 0.0
    return 1.0 - pearson(d_pred, d_gt.detach())


def subject_diff_loss(dp_a: torch.Tensor, dp_b: torch.Tensor,
                      dg_a: torch.Tensor, dg_b: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    """L_diff = |(pred_a - pred_b) - (gt_a - gt_b)|_1 (§4.3).

    모든 subject 에 같은 값을 내는 shortcut 을 직접 막는다: 출력이 같으면 예측 차이가 0 이라
    실제 subject 차이가 있는 한 손실을 피할 수 없다.
    """
    dp, dg = dp_a - dp_b, (dg_a - dg_b).detach()
    assert dp.shape == dg.shape, (dp.shape, dg.shape)
    if mask is not None:
        assert mask.any(), "잔차 edge mask 가 비어 있음"
        dp, dg = dp[mask], dg[mask]
    return (dp - dg).abs().mean()


def subject_var_loss(dp_a: torch.Tensor, dp_b: torch.Tensor,
                     dg_a: torch.Tensor, dg_b: torch.Tensor,
                     mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """L_var — 예측 잔차의 **진폭**을 GT 에 맞춘다 (전략 문서 §4.4).

    왜 필요한가 (실측): SmoothL1 은 불확실할 때 평균으로 수축하는 게 최적이라(regression to
    the mean) 예측 개인차가 눌린다. `a2_residual_D_local` 학습 로그에서 diff_ratio 가 0.055 --
    예측한 두 subject 차이가 GT 의 5.5% 뿐이고, 그래서 inter_subj_r 이 0.9997 에서 안 내려간다.
    정보(train resid_r 0.10)는 있는데 출력으로 안 나오는 상태다.

    두 subject 로 분산을 재는 근거: 독립 표본이면 E‖Δa − Δb‖² = 2·Var(Δ) 이므로 쌍차 노름을
    맞추는 것이 표준편차를 맞추는 것과 같다 (상수배). 한 step 에 2명뿐이라 std 를 직접 재는
    것보다 안정적이다.

    로그 비를 쓰는 이유: 과소·과대를 대칭으로 벌하고, 비가 0.05 처럼 작을 때도 기울기가 산다.

    **주의**: 이 손실만 키우면 잡음을 개인차로 증폭한다. self vs shuffled 격차와 식별 정확도를
    반드시 함께 감시해야 한다 (문서 §4.4 의 경고).
    """
    dp, dg = dp_a - dp_b, (dg_a - dg_b).detach()
    if mask is not None:
        assert mask.any(), "잔차 edge mask 가 비어 있음"
        dp, dg = dp[mask], dg[mask]
    np_, ng = dp.norm(), dg.norm()
    assert float(ng) > 0, "GT 두 subject 의 잔차 차이가 0 -- 같은 subject 인가"
    return torch.log((np_ + eps) / (ng + eps)).abs()
