"""Edge-existence head (pipeline §10).

T1 feature + ROI pair -> P(edge exists). positive 만 학습하면 over-connectivity 가
생기므로 negative pair 를 함께 넣어 BCE 로 학습한다. inference 에서는 이 확률로
생성할 pair 를 고른다 (§22).

마지막 층 0 초기화 -> 시작 시 p = 0.5.

템플릿 인수분해 (재학습 설계 §2 ③): `template_prob` (train 에서 그 pair 에 edge 가 있던 비율)
을 주면

    logit(i,j) = logit(template_prob(i,j)) + g(anatomy, pair_vec)

가 되고 g 의 마지막 층이 0-init 이라 **시작 시점 확률이 정확히 그룹 빈도**다. head 가
그룹 평균을 외우는 데 용량을 쓰지 않고 개인차만 학습한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .edge_count_head import as_template


class EdgeHead(nn.Module):
    def __init__(self, cond_dim: int = 512, emb_dim: int = 64, hidden: int = 256,
                 template_prob=None, prob_eps: float = 1e-3):
        """template_prob: [R,R] edge 존재 확률 (0~1). 주면 인수분해 모드 (없으면 기존 동작).
        prob_eps: logit 이 무한대가 되지 않게 [eps, 1-eps] 로 자른다."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if template_prob is None:
            self.register_buffer("template_logit", None)
        else:
            pr = as_template(template_prob, "template_prob")
            assert float(pr.max()) <= 1.0, f"확률이 1 을 넘는다 ({float(pr.max())})"
            assert 0 < prob_eps < 0.5, prob_eps
            pr = pr.clamp(prob_eps, 1.0 - prob_eps)
            # 고정 buffer -- 학습하지 않는다 (train 평균이고 개인차만 g 가 맞춘다).
            self.register_buffer("template_logit", torch.log(pr) - torch.log1p(-pr))

    def forward(self, anatomy: torch.Tensor, pair_vec: torch.Tensor,
                pairs: torch.Tensor | None = None) -> torch.Tensor:
        """anatomy [1,C] 또는 [N,C], pair_vec [N,E] -> logits [N].
        pairs [N,2] 는 템플릿 인수분해 모드에서만 필요하다."""
        n = pair_vec.shape[0]
        if anatomy.shape[0] == 1:
            anatomy = anatomy.expand(n, -1)
        out = self.net(torch.cat([anatomy, pair_vec], dim=-1)).squeeze(-1)
        if self.template_logit is not None:
            assert pairs is not None, "템플릿 인수분해 head 는 pair 인덱스가 필요하다 (forward(..., pairs=...))"
            assert pairs.ndim == 2 and pairs.shape == (n, 2), (pairs.shape, n)
            i, j = pairs[:, 0].long(), pairs[:, 1].long()
            r = self.template_logit.shape[0]
            assert int(i.min()) >= 0 and int(torch.maximum(i, j).max()) < r, (
                f"ROI 인덱스가 템플릿({r}x{r}) 범위 밖")
            out = out + self.template_logit.to(out.device)[i, j]
        return out
