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


from .pair_anchor import upper_index


class EdgeHead(nn.Module):
    def __init__(self, cond_dim: int = 512, emb_dim: int = 64, hidden: int = 256,
                 template_prob=None, prob_eps: float = 1e-3,
                 local_dim: int = 0, tier1_dim: int = 0, tier1_n_roi: int = 0):
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
        # 국소/티어1 통로. 왜 필요한가 -- 실측(val 10명): 이 head 가 고르는 pair 집합의 Jaccard 가
        # 0.9896 이다. 즉 "어떤 연결이 존재하는가"에서 개인차가 이미 지워진다. 뒤의 count head 를
        # 아무리 잘 만들어도 여기서 걸러진 뒤다. 전역 a512 만 받던 것이 원인이다 (subject 성분 2.1%).
        # 전부 0-init 이라 켜도 시작은 bit-exact 다.
        self.local_dim, self.tier1_dim, self.tier1_n_roi = int(local_dim), int(tier1_dim), int(tier1_n_roi)
        self.local_proj = None
        if self.local_dim:
            self.local_proj = nn.Linear(self.local_dim, hidden)
            nn.init.zeros_(self.local_proj.weight); nn.init.zeros_(self.local_proj.bias)
        self.tier1_proj = None
        if self.tier1_dim:
            self.tier1_proj = nn.Sequential(nn.Linear(self.tier1_dim, hidden), nn.GELU(),
                                            nn.Linear(hidden, hidden))
            nn.init.zeros_(self.tier1_proj[-1].weight); nn.init.zeros_(self.tier1_proj[-1].bias)
        self.tier1_w = self.tier1_b = None
        if self.tier1_dim and self.tier1_n_roi:
            n_pair = self.tier1_n_roi * (self.tier1_n_roi - 1) // 2
            self.tier1_w = nn.Parameter(torch.zeros(n_pair, self.tier1_dim))
            self.tier1_b = nn.Parameter(torch.zeros(n_pair))

    def forward(self, anatomy: torch.Tensor, pair_vec: torch.Tensor,
                pairs: torch.Tensor | None = None,
                local: torch.Tensor | None = None,
                tier1: torch.Tensor | None = None) -> torch.Tensor:
        """anatomy [1,C] 또는 [N,C], pair_vec [N,E] -> logits [N].
        pairs [N,2] 는 템플릿 인수분해 모드에서만 필요하다."""
        n = pair_vec.shape[0]
        if anatomy.shape[0] == 1:
            anatomy = anatomy.expand(n, -1)
        h = self.net[0](torch.cat([anatomy, pair_vec], dim=-1))
        if self.local_proj is not None:
            assert local is not None, "local_dim > 0 인데 local feature 가 안 넘어왔다"
            assert local.shape == (n, self.local_dim), (local.shape, n, self.local_dim)
            h = h + self.local_proj(local)
        else:
            assert local is None, "local_dim = 0 인데 local feature 가 넘어왔다"
        if self.tier1_proj is not None:
            assert tier1 is not None, "tier1_dim > 0 인데 tier1 feature 가 안 넘어왔다"
            assert tier1.shape == (n, self.tier1_dim), (tier1.shape, n, self.tier1_dim)
            h = h + self.tier1_proj(tier1)
        else:
            assert tier1 is None, "tier1_dim = 0 인데 tier1 feature 가 넘어왔다"
        out = self.net[1:](h).squeeze(-1)
        if self.tier1_w is not None:
            assert pairs is not None, "pair 별 tier1 가중치는 pair 인덱스가 필요하다"
            idx = upper_index(pairs[:, 0].long(), pairs[:, 1].long(), self.tier1_n_roi)
            out = out + (self.tier1_w[idx] * tier1).sum(-1) + self.tier1_b[idx]
        if self.template_logit is not None:
            assert pairs is not None, "템플릿 인수분해 head 는 pair 인덱스가 필요하다 (forward(..., pairs=...))"
            assert pairs.ndim == 2 and pairs.shape == (n, 2), (pairs.shape, n)
            i, j = pairs[:, 0].long(), pairs[:, 1].long()
            r = self.template_logit.shape[0]
            assert int(i.min()) >= 0 and int(torch.maximum(i, j).max()) < r, (
                f"ROI 인덱스가 템플릿({r}x{r}) 범위 밖")
            out = out + self.template_logit.to(out.device)[i, j]
        return out
