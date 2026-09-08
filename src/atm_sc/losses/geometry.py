"""L_ATM: streamline geometry / reconstruction (Framework v2 §14).

**upstream 배포본에는 학습 코드도 loss 함수도 없다** (.py 는 infer.py 와 model/model.py
둘뿐이고, model/__pycache__ 의 .pyc 세 개에도 loss 이름이 없다). 따라서 원 논문의
reconstruction/adjacent-point loss 를 그대로 가져올 수 없고 아래를 우리가 정의한다.

  G-1 stream_recon_loss + kl_loss : GT bundle 라벨이 있을 때 (원 VAE objective 근사)
  G-2 anchor_loss                 : 동결한 원본 decoder 출력에서 멀어지지 않게. GT bundle
                                    라벨이 필요 없어 기본값으로 쓴다.
  adjacency_loss                  : ATM 이 128점 등간격 재샘플로 학습되었다는 전제를
                                    제약으로 되돌린 것.
"""
from __future__ import annotations

import torch


def adjacency_loss(mm: torch.Tensor) -> torch.Tensor:
    d = torch.linalg.norm(mm[:, 1:] - mm[:, :-1], dim=-1)
    return ((d - d.mean(dim=1, keepdim=True)) ** 2).mean()


def anchor_loss(mm_pred: torch.Tensor, mm_ref: torch.Tensor) -> torch.Tensor:
    assert mm_pred.shape == mm_ref.shape, (mm_pred.shape, mm_ref.shape)
    return ((mm_pred - mm_ref) ** 2).sum(-1).mean()


def stream_recon_loss(mm_pred: torch.Tensor, mm_gt: torch.Tensor, squared: bool = False,
                      weights: torch.Tensor | None = None) -> torch.Tensor:
    """방향 모호성을 고려해 정방향/역방향 중 작은 쪽. 기본은 streamline 별 RMSE (mm).

    mm^2 그대로 두면 gradient norm 이 ~10^3 으로 다른 항(~10^0)을 압도하고 global grad clip 이
    다른 항의 실효 학습률을 1/200 로 만든다 (실측). mm 단위로 두면 항끼리 비교 가능하다.
    """
    a = ((mm_pred - mm_gt) ** 2).sum(-1).mean(-1)
    b = ((mm_pred - mm_gt.flip(1)) ** 2).sum(-1).mean(-1)
    m = torch.minimum(a, b)
    v = m if squared else (m + 1e-6).sqrt()
    if weights is None:
        return v.mean()
    # synthetic streamline 은 GT 가 아니다 -> 낮은 가중치 (ROUTE 전략 §47)
    assert weights.shape == v.shape, (weights.shape, v.shape)
    return (v * weights).sum() / weights.sum().clamp(min=1e-8)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, mu_prior: torch.Tensor | None = None,
            logvar_prior: torch.Tensor | None = None) -> torch.Tensor:
    """KL( N(mu, diag exp(logvar)) || N(mu_prior, diag exp(logvar_prior)) ).

    mu_prior=None 이면 평균 0, logvar_prior=None 이면 단위분산 -> 기존 식과 **bit-exact** 동일한
    분기를 탄다 (W2-b selfcheck: kl_old == kl_new_logvar_prior0).

    prior 파라미터를 여기에 학습시키면 안 된다 (C11 실측): KL(q||p) 의 p 최적해는 aggregate
    posterior 의 모멘트라, 평균이 mu_pair 에 묶인 지금 구조에서 sigma* = 1.074 -> 지금의 1.0 이
    이미 KL 최적점이고 그 sigma 로 채점하면 precision 36.61 -> 38.74 로 나빠진다.
    호출부는 mu_prior/logvar_prior 를 **detach 해서** 넘긴다. prior 는 별도 적합항으로 학습한다.
    """
    d = mu if mu_prior is None else mu - mu_prior
    if logvar_prior is None:
        return (-0.5 * (1 + logvar - d.pow(2) - logvar.exp()).sum(1)).mean()
    return (0.5 * (logvar_prior - logvar + (logvar - logvar_prior).exp()
                   + d.pow(2) * (-logvar_prior).exp() - 1.0).sum(1)).mean()


def wm_occupancy_loss(mm: torch.Tensor, wm: torch.Tensor, inv_affine: torch.Tensor,
                      interior: float = 0.125, target: float = 0.9) -> torch.Tensor:
    """생성/복원 streamline 이 백질 안을 지나게 하는 손실 (미분 가능).

    측정(val 3명, SyN 공간): GT 가닥은 중간구간 WM 점유 평균 0.825 / 중앙 0.988 이고 86 %가
    0.5 를 넘는데, 생성 가닥은 평균 0.393 / 중앙 0.384 로 36 %만 넘는다. 디코더가 백질 밖을
    헤매고, 그래서 가닥 하나가 ROI 를 6.4개(GT 4.4) 지나 pair 20개에 count 를 더한다. 그 희석이
    배분에 실린 개인차를 pass SC 로 전달하는 비율을 0.27 까지 떨어뜨린다.

    양끝 `interior` 비율은 뺀다 -- 끝점은 피질(GM)에 닿아야 하므로 WM 을 요구하면 안 된다.
    hinge 라 target 을 넘긴 점은 더 밀지 않는다 (백질 중심으로 뭉쳐 피질에 못 닿는 것을 막는다).

    mm [N,T,3] (W 격자 mm), wm [X,Y,Z] (0~1, streamline 과 **같은 공간**이어야 한다 -- GT 는
    QSDR/NLin6 이므로 rigid 조직맵을 쓰면 조직 경계가 어긋난다), inv_affine [4,4] mm->voxel.
    """
    import torch.nn.functional as F
    assert mm.ndim == 3 and mm.shape[2] == 3, mm.shape
    assert wm.ndim == 3, wm.shape
    N, T = mm.shape[:2]
    lo = int(T * interior); hi = T - lo
    assert hi > lo, (T, interior)
    x = mm[:, lo:hi]
    ijk = torch.einsum("ij,ntj->nti", inv_affine[:3, :3].to(x.dtype), x) + inv_affine[:3, 3].to(x.dtype)
    size = torch.tensor(wm.shape, device=x.device, dtype=x.dtype)
    g = (2.0 * ijk / (size - 1.0) - 1.0).flip(-1)          # grid_sample 은 (x,y,z) = (k,j,i) 순서
    occ = F.grid_sample(wm[None, None].to(x.dtype), g.reshape(1, 1, 1, -1, 3),
                        align_corners=True, padding_mode="zeros").reshape(N, hi - lo)
    return torch.relu(target - occ).mean()
