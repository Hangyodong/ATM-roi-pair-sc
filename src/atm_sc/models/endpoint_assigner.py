"""미분 가능한 soft ROI 할당 (Framework v2 §7.3).

ROI 별 거리맵을 미리 만들어 두고 streamline 점에서 grid_sample 로 보간한 뒤
softmax(-d/tau) 로 소속 확률을 만든다. hard atlas lookup 을 쓰지 않으므로
streamline 좌표까지 gradient 가 흐른다.

두 가지 집계를 제공한다.

  endpoint_probs   v2 §7.4/§8.1 의 정의. 양 끝점만 본다. (q_start, q_end)
  visit_probs      streamline 이 ROI 를 "통과"했는지. GT SC 가 pass 정의이므로
                   이쪽이 GT 를 재현한다.

실측 (sub-100001, 200k streamline, .mat 의 GT SC 대조):
    endpoint 규칙(hard)  r=0.673
    pass 규칙(hard)      r=0.9986
    pass 규칙(soft, tau=0.5 / max / d_bg=2.0)  r=0.9939, ccc=0.9854

d_bg (배경 클래스):
    None 이면 82개 ROI 에 대한 순수 softmax 라 확률 합이 1 이다.
    값을 주면 그 거리(mm)를 배경 로짓으로 넣는다. 이것이 없으면 모든 ROI 에서 먼
    심부 백질 점도 softmax 가 합 1 을 강제해 가장 가까운 피질 ROI 로 배정된다
    (hard 규칙은 label 0 을 준다). pass 모드에서 d_bg 를 끄면 r 이 0.99 -> 0.91 로 떨어진다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..spaces import mm_to_grid


def build_distance_maps(atlas: np.ndarray, n_roi: int, voxel_size) -> np.ndarray:
    """ROI 별 유클리드 거리맵 [R, X, Y, Z] (mm). ROI 내부는 0."""
    from scipy.ndimage import distance_transform_edt
    out = np.empty((n_roi,) + atlas.shape, np.float32)
    for r in range(n_roi):
        m = atlas == (r + 1)
        assert m.any(), f"ROI {r+1} 이 atlas 에 없음 (해상도 변환 중 소실?)"
        out[r] = distance_transform_edt(~m, sampling=voxel_size).astype(np.float32)
    assert np.isfinite(out).all()
    return out


class EndpointAssigner(torch.nn.Module):
    def __init__(self, dist_maps: np.ndarray, affine: np.ndarray, tau: float = 0.5,
                 device="cuda", dtype=torch.float32, aggregate: str = "max",
                 d_bg: float | None = None):
        super().__init__()
        d = torch.as_tensor(dist_maps, dtype=dtype)
        assert d.ndim == 4, d.shape
        self.register_buffer("dist", d.unsqueeze(0).to(device))       # [1,R,X,Y,Z]
        self.affine = np.asarray(affine, np.float64)
        self.shape = tuple(dist_maps.shape[1:])
        self.n_roi = int(dist_maps.shape[0])
        self.tau, self.aggregate, self.d_bg = float(tau), aggregate, d_bg

    # --- 공통 ---------------------------------------------------------------
    def point_probs(self, mm: torch.Tensor) -> torch.Tensor:
        """[N, T, 3] mm -> [N, T, R]. d_bg 가 None 이면 마지막 축 합이 1."""
        d = self.point_dist(mm)
        if self.d_bg is None:
            return torch.softmax(-d / self.tau, dim=-1)
        bg = torch.full_like(d[..., :1], float(self.d_bg))
        return torch.softmax(-torch.cat([d, bg], -1) / self.tau, dim=-1)[..., :self.n_roi]

    def point_dist(self, mm: torch.Tensor) -> torch.Tensor:
        """[N, T, 3] mm -> [N, T, R] 각 ROI 까지의 거리(mm). 보간된 거리맵."""
        assert mm.ndim == 3 and mm.shape[-1] == 3, mm.shape
        n, t, _ = mm.shape
        g = mm_to_grid(mm.reshape(-1, 3), self.affine, self.shape)
        g = g.reshape(1, 1, 1, -1, 3).to(self.dist.dtype)
        d = F.grid_sample(self.dist, g, mode="bilinear", align_corners=True, padding_mode="border")
        return d.squeeze(0).squeeze(1).squeeze(1).T.reshape(n, t, self.n_roi).float()

    def point_log_probs(self, mm: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """[N, T, 3] -> [N, T, R] log-확률. softmax 를 거치지 않고 log_softmax 로 바로 계산한다.

        끝점이 목표 ROI 에서 수십 mm 떨어져 있으면 확률이 exp(-d/tau) ~ 1e-40 으로 float32
        아래로 내려가 clamp 에 걸리고 **gradient 가 0** 이 된다 (실측: L_endpoint 가 34 에서
        고정 = 2*log(1e-8)). log 공간에서는 -d/tau - logsumexp 라 어떤 거리에서도 gradient 가 산다.
        """
        tau = self.tau if tau is None else float(tau)
        d = self.point_dist(mm)
        if self.d_bg is None:
            return torch.log_softmax(-d / tau, dim=-1)
        bg = torch.full_like(d[..., :1], float(self.d_bg))
        return torch.log_softmax(-torch.cat([d, bg], -1) / tau, dim=-1)[..., :self.n_roi]

    # --- v2 §7.4 endpoint ----------------------------------------------------
    def endpoint_probs(self, mm: torch.Tensor):
        """[N, T, 3] -> (q_start [N,R], q_end [N,R]). SC builder 용."""
        q = self.point_probs(mm[:, [0, -1], :])
        return q[:, 0], q[:, 1]

    def endpoint_log_probs(self, mm: torch.Tensor, tau: float | None = None):
        """(log q_start [N,R], log q_end [N,R]). endpoint loss 용 -- 반드시 이쪽을 쓴다."""
        lq = self.point_log_probs(mm[:, [0, -1], :], tau)
        return lq[:, 0], lq[:, 1]

    # --- pass ----------------------------------------------------------------
    def visit_probs(self, mm: torch.Tensor, aggregate: str | None = None,
                    beta: float = 20.0, eps: float = 1e-6) -> torch.Tensor:
        """[N, T, 3] -> [N, R] "ROI i 를 통과했는가".

        'max'      max_t q_t. 기본값. 의미상 옳고 max-pooling 처럼 gradient 가 흐른다.
        'lse'      beta 를 쓴 smooth max.
        'noisy_or' 1 - prod(1-q_t). streamline 위 연속한 점들이 강하게 상관되어 있어
                   128배로 과대계상된다 (실측 r 0.99 -> 0.89). 비교용으로만 남겨 둔다.
        """
        agg = aggregate or self.aggregate
        q = self.point_probs(mm)
        if agg == "max":
            return q.amax(dim=1)
        if agg == "lse":
            return torch.logsumexp(beta * q, dim=1) / beta - float(np.log(q.shape[1])) / beta
        if agg == "noisy_or":
            return -torch.expm1(torch.log1p(-q.clamp(max=1.0 - eps)).sum(dim=1))
        raise ValueError(agg)

    def visit_log_probs(self, mm: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """[N,T,3] -> [N,R] log P(ROI i 통과). max 집계를 log 공간에서 그대로 한다.

        확률 공간의 visit_probs 는 목표 ROI 가 멀면 exp(-d/tau) 가 float32 아래로 내려가
        BCE clamp 에 걸려 gradient 가 0 이 된다 (endpoint loss 와 같은 버그). route loss 는 이걸 쓴다.
        """
        return self.point_log_probs(mm, tau=tau).amax(dim=1)


RoiAssigner = EndpointAssigner        # 이전 이름 호환
