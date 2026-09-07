"""Streamline weight head (pipeline §16).

pair 마다 같은 수의 streamline 을 만들면 absolute SC weight 를 재현할 수 없다.
각 생성 streamline 에 w_k >= 0 을 붙여 SC 기여도를 조절한다:
    SC(i,j) = sum_k w_k * P_i(start_k) * P_j(end_k)

입력은 cond(= anatomy + pair embedding, [N,512]) 와 latent z ([N,64]).
마지막 층을 0 으로, bias 를 softplus^-1(1) 로 초기화해 시작 시 w == 1 (= 순수 count).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamlineWeightHead(nn.Module):
    def __init__(self, cond_dim: int = 512, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + latent_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(math.e - 1.0))     # softplus -> 1.0

    def is_untrained(self) -> bool:
        """마지막 층이 0-init 그대로인가. True 면 출력은 입력과 무관하게 상수 softplus(bias) = 1.0 이다.

        0-init + gradient 0 -> 영원히 0 이므로, 이 값이 True 라는 것은 학습 중 이 head 의 출력이
        어떤 손실에도 닿지 않았다는 뜻이다 (trainer.weight_mode 가 'head'/'count_head' 가 아니면 그렇다).
        그 상태에서 w == 1.0 이라 weighted SC 는 가중 없는 count SC 와 수치적으로 같다.
        """
        return not bool(self.net[-1].weight.any())

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert cond.shape[0] == z.shape[0], (cond.shape, z.shape)
        w = F.softplus(self.net(torch.cat([cond, z], dim=-1))).squeeze(-1)
        assert w.shape == (cond.shape[0],)
        return w
