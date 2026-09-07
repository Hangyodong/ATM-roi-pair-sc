"""Edge count head (strategy §19-21).

EdgeHead 가 "이 pair 에 edge 가 있나" 를 풀었다면 여기서는 "몇 개 생성해야 하나" 를 푼다.
GT SC 가 곧 edge 를 구성하는 streamline/segment 수이므로 N_hat(i,j) 을 직접 회귀한다.

왜 필요한가: sc_magnitude_loss 는 normalize='sum' 으로 총합을 맞춘 뒤 비교하므로 절대
스케일 정보가 loss 에서 사라진다 (패턴 r=0.81 인데 CCC=0.02). count 를 직접 맞추는 head 가
있어야 절대값이 맞는다.

출력은 log count 다. count 분포가 heavy tail (1 ~ 수만) 이라 선형 회귀는 큰 edge 에만
끌려가고, log 면 상대오차가 균등해진다. 하한이 없으므로 exp() >= 0 이 구조적으로 보장된다.

마지막 층 0 초기화 + bias = init_log_count -> 시작 시 모든 edge 가 exp(init_log_count)
(GT median 근처의 상수) 를 예측한다. 0-init 은 EdgeHead 와 같은 이유이고, bias 만 옮겨서
"평균은 이미 맞고 편차만 학습" 하는 지점에서 출발시킨다.

템플릿 인수분해 (재학습 설계 §2 ③): `template` (train 평균 count [R,R]) 을 주면

    log_count(i,j) = log(template(i,j)) + f(anatomy, pair_vec)

가 되고 f 의 마지막 층이 0-init 이므로 **시작 시점 예측이 정확히 그룹 템플릿**이다
(그때 상수 bias 는 0 이어야 하므로 init_log_count 를 쓰지 않는다). 남은 학습량이 개인차뿐이라
기울기가 전부 거기로 간다. 지금 구조는 head 용량 대부분을 그룹 평균 암기에 쓰고 있었다
(추론에서 템플릿 배분 0.853 > 학습된 count_head_end 0.567).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def as_template(t, name: str = "template") -> torch.Tensor:
    """[R,R] 대칭 비음수 템플릿을 float32 텐서로. 비었거나 비대칭이면 조용히 넘어가지 않는다."""
    x = torch.as_tensor(np.asarray(t), dtype=torch.float32)
    assert x.ndim == 2 and x.shape[0] == x.shape[1], (name, x.shape)
    assert torch.isfinite(x).all(), f"{name} 에 NaN/Inf"
    assert bool((x >= 0).all()), f"{name} 에 음수"
    assert float(x.sum()) > 0, f"{name} 이 전부 0"
    assert torch.allclose(x, x.T, atol=1e-6), f"{name} 이 비대칭"
    return x


class EdgeCountHead(nn.Module):
    def __init__(self, anatomy_dim: int = 512, emb_dim: int = 64, hidden: int = 256,
                 init_log_count: float = 5.0, template=None, template_floor: float = 1e-2):
        """template: [R,R] train 평균 count. 주면 인수분해 모드 (없으면 기존 동작 그대로).
        template_floor: log(0) 을 피하는 바닥값. count < 1 은 "생성 0 개" 라 1e-2 는 0 과 같다."""
        super().__init__()
        self.anatomy_dim, self.emb_dim = anatomy_dim, emb_dim
        self.init_log_count = float(init_log_count)
        self.net = nn.Sequential(nn.Linear(anatomy_dim + emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        # 템플릿이 있으면 log(template) 이 상수항을 대신하므로 bias 는 0 이어야 한다.
        nn.init.constant_(self.net[-1].bias, 0.0 if template is not None else self.init_log_count)
        # 고정 buffer -- 학습하지 않는다 (train 144명 평균이고 개인차만 f 가 맞춘다).
        self.register_buffer("template_log", None if template is None else
                             as_template(template).clamp(min=template_floor).log())

    def _template_term(self, pairs, k: int, device) -> torch.Tensor:
        assert pairs is not None, "템플릿 인수분해 head 는 pair 인덱스가 필요하다 (forward(..., pairs=...))"
        assert pairs.ndim == 2 and pairs.shape == (k, 2), (pairs.shape, k)
        i, j = pairs[:, 0].long(), pairs[:, 1].long()
        r = self.template_log.shape[0]
        assert int(i.min()) >= 0 and int(torch.maximum(i, j).max()) < r, (
            f"ROI 인덱스가 템플릿({r}x{r}) 범위 밖")
        return self.template_log.to(device)[i, j]

    def forward(self, anatomy: torch.Tensor, pair_vec: torch.Tensor,
                pairs: torch.Tensor | None = None) -> torch.Tensor:
        """anatomy [1,C] 또는 [K,C], pair_vec [K,E] -> log_count [K] (자연로그).
        pairs [K,2] 는 템플릿 인수분해 모드에서만 필요하다."""
        assert pair_vec.ndim == 2 and pair_vec.shape[1] == self.emb_dim, pair_vec.shape
        k = pair_vec.shape[0]
        if anatomy.shape[0] == 1:
            anatomy = anatomy.expand(k, -1)
        assert anatomy.shape == (k, self.anatomy_dim), (anatomy.shape, k, self.anatomy_dim)
        out = self.net(torch.cat([anatomy, pair_vec], dim=-1)).squeeze(-1)
        if self.template_log is not None:
            out = out + self._template_term(pairs, k, out.device)
        return out

    def count(self, anatomy: torch.Tensor, pair_vec: torch.Tensor,
              pairs: torch.Tensor | None = None) -> torch.Tensor:
        """exp(log_count) [K]. 항상 >= 0."""
        return torch.exp(self.forward(anatomy, pair_vec, pairs))

    def matrix(self, anatomy: torch.Tensor, pair_vec: torch.Tensor, pairs: torch.Tensor,
               n_roi: int) -> torch.Tensor:
        """예측 count 를 대칭 [n_roi, n_roi] 로 흩뿌린다 (대각 0). 안 나온 pair 는 0 이다."""
        assert pairs.ndim == 2 and pairs.shape[1] == 2, pairs.shape
        assert pairs.shape[0] == pair_vec.shape[0], (pairs.shape, pair_vec.shape)
        i, j = pairs[:, 0].long(), pairs[:, 1].long()
        assert int(i.min()) >= 0 and int(torch.maximum(i, j).max()) < n_roi, "ROI 인덱스 범위 밖"
        assert bool((i != j).all()), "self-edge (i==j) 는 SC 에 없다 — 대각은 0 이어야 한다"
        c = self.count(anatomy, pair_vec, pairs)
        m = torch.zeros(n_roi, n_roi, dtype=c.dtype, device=c.device)
        # (i,j) 와 (j,i) 를 함께 넣어 대칭을 구조적으로 보장한다. 같은 pair 가 중복되면
        # accumulate 로 합쳐진다 (count 는 가산량).
        return m.index_put((torch.cat([i, j]), torch.cat([j, i])), torch.cat([c, c]),
                           accumulate=True)
