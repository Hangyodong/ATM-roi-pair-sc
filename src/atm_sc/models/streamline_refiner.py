"""동결 디코더 출력 위에 얹는 streamline 정련망 (PIPELINE_11 §D1-b).

upstream(`stable/`) 무수정 제약 때문에 ConvVAE 자체를 키울 수 없다. 대신 `decode_mm` 이
낸 [N,128,3] mm 좌표를 받아 **보정량만** 예측해 더한다.

  mm_refined = mm + delta(mm, cond)

설계 근거 (전부 실측):

* **마지막 conv 0-init** -> delta 가 정확히 0 이라 켜자마자는 기존 동작과 bit-exact 동일하다.
  사전학습 ConvVAE 를 건드리지 않고 용량만 더하는 유일한 방법이다.
* **BatchNorm 을 쓰지 않는다.** ConvVAE 의 BatchNorm1d 5개는 running stat 이 낡아
  held-out recon 을 8.46 -> 4.00 mm 로 두 배 부풀리고 있었다 (train/eval 모드 차이가
  곧 3.55 vs 9.17 mm 불일치의 절반이다). 정련망은 GroupNorm 만 써서 train/eval 이
  같은 함수가 되게 한다 -- running stat 이라는 상태를 아예 만들지 않는다.
* **1D conv + dilation.** 점마다 독립인 MLP 는 128점이 하나의 매끄러운 곡선이라는 구조를
  못 쓴다. dilation 을 1,2,4,... 로 키워 곡선 전체를 보는 수용영역을 만든다
  (kernel 5, layers L -> RF = 8*(2^L - 1) + 9 점, L=5 면 128점을 덮는다).
* **접선(tangent) 입력.** 좌표 차분은 곡률/방향 정보라 곡선의 매끄러움을 직접 준다.
* **조건은 선택.** 순수 복원 단계(D1)에는 조건이 필요 없다 -- 정답을 넣고 정답을 복원한다.
  `cond=None` 으로 동작하고, 조건화 단계에서 그대로 재사용할 수 있게 자리는 남긴다.

좌표 스케일: 내부 연산은 디코더와 같은 정규화 공간([-1,1])에서 하고 출력만 mm 로 되돌린다.
`coord_min`/`coord_scale` 을 주지 않으면 항등(= mm 공간에서 바로 연산)이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """GroupNorm -> GELU -> dilated Conv1d, 두 번. pre-norm 잔차."""

    def __init__(self, ch: int, kernel: int, dilation: int, groups: int):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.n1 = nn.GroupNorm(groups, ch)
        self.c1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.n2 = nn.GroupNorm(groups, ch)
        self.c2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.c1(nn.functional.gelu(self.n1(h)))
        x = self.c2(nn.functional.gelu(self.n2(x)))
        return h + x


class StreamlineRefiner(nn.Module):
    """[N,P,3] mm -> [N,P,3] mm. 시작 시 항등 (마지막 conv 0-init).

    Args:
        n_points:  곡선 점 개수 (ATM 은 128 고정).
        hidden:    conv 채널 폭.
        layers:    잔차 블록 수. dilation 은 1,2,4,... 로 커진다.
        kernel:    conv 커널 크기 (홀수).
        cond_dim:  조건 벡터 차원. None 이면 조건 입력을 아예 만들지 않는다.
        coord_min/coord_scale: 디코더와 같은 좌표 정규화 상수 [3]. `decode_mm` 의 역변환.
    """

    def __init__(self, n_points: int = 128, hidden: int = 128, layers: int = 3,
                 kernel: int = 5, cond_dim: int | None = 512,
                 coord_min=None, coord_scale=None):
        super().__init__()
        assert kernel % 2 == 1, f"kernel 은 홀수여야 한다 (받은 값: {kernel})"
        assert layers >= 1 and hidden >= 1, (layers, hidden)
        self.n_points, self.hidden, self.layers, self.kernel = n_points, hidden, layers, kernel
        self.cond_dim = cond_dim
        groups = max(1, min(8, hidden // 8))
        assert hidden % groups == 0, f"hidden({hidden}) 이 GroupNorm groups({groups}) 로 안 나뉜다"

        # 좌표 정규화 상수. 주지 않으면 항등 -> mm 공간에서 바로 연산.
        cmin = torch.zeros(3) if coord_min is None else torch.as_tensor(coord_min, dtype=torch.float32)
        cscl = torch.ones(3) if coord_scale is None else torch.as_tensor(coord_scale, dtype=torch.float32)
        assert cmin.shape == (3,) and cscl.shape == (3,), (cmin.shape, cscl.shape)
        assert (cscl > 0).all(), f"coord_scale 이 양수가 아니다: {cscl}"
        self.register_buffer("coord_min", cmin)
        self.register_buffer("coord_scale", cscl)

        # 입력: 정규화 좌표 3 + 접선 3
        self.stem = nn.Conv1d(6, hidden, kernel, padding=kernel // 2)
        # 곡선 위 위치 (양 끝 vs 중간은 역할이 다르다). 학습 가능한 점별 임베딩.
        self.pos = nn.Parameter(torch.zeros(1, hidden, n_points))
        self.cond_proj = nn.Linear(cond_dim, hidden) if cond_dim else None
        self.blocks = nn.ModuleList([_ResBlock(hidden, kernel, 2 ** i, groups) for i in range(layers)])
        self.out_norm = nn.GroupNorm(groups, hidden)
        self.out = nn.Conv1d(hidden, 3, kernel, padding=kernel // 2)
        # 0-init: 시작 시 delta == 0 (정확히). 이것이 bit-exact 동등성의 근거다.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def receptive_field(self) -> int:
        """점 단위 수용영역. n_points 이상이면 곡선 전체를 본다."""
        k = self.kernel - 1
        return 1 + 2 * k + sum(2 * k * (2 ** i) for i in range(self.layers))

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden}, layers={self.layers}, kernel={self.kernel}, "
                f"cond_dim={self.cond_dim}, RF={self.receptive_field}/{self.n_points}")

    def _features(self, mm: torch.Tensor) -> torch.Tensor:
        """[N,P,3] mm -> [N,6,P] (정규화 좌표 + 접선)."""
        s = (mm - self.coord_min) / self.coord_scale - 1.0          # decode_mm 의 역변환
        x = s.permute(0, 2, 1)                                       # [N,3,P]
        # 전방 차분. 마지막 점은 앞 값을 복제해 길이를 맞춘다 (곡선 끝의 접선 = 마지막 구간).
        d = x[:, :, 1:] - x[:, :, :-1]
        d = torch.cat([d, d[:, :, -1:]], dim=2)
        return torch.cat([x, d], dim=1)

    def delta(self, mm: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """보정량 [N,P,3] (mm). 0-init 상태에서는 정확히 0."""
        n, p, _ = mm.shape
        assert p == self.n_points, f"점 개수 {p} != {self.n_points}"
        h = self.stem(self._features(mm)) + self.pos
        if cond is not None:
            assert self.cond_proj is not None, "cond_dim=None 으로 만든 정련망에 조건을 줬다"
            if cond.shape[0] == 1:
                cond = cond.expand(n, -1)
            assert cond.shape == (n, self.cond_dim), (cond.shape, n, self.cond_dim)
            h = h + self.cond_proj(cond).unsqueeze(-1)
        for b in self.blocks:
            h = b(h)
        d = self.out(nn.functional.gelu(self.out_norm(h)))           # [N,3,P], 0-init -> 0
        return d.permute(0, 2, 1) * self.coord_scale                 # 정규화 단위 -> mm

    def forward(self, mm: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        assert mm.ndim == 3 and mm.shape[2] == 3, mm.shape
        out = mm + self.delta(mm, cond)
        assert out.shape == mm.shape, (out.shape, mm.shape)
        return out

    @torch.no_grad()
    def assert_identity(self, mm: torch.Tensor, cond: torch.Tensor | None = None) -> None:
        """0-init 인지 실제 출력으로 확인한다 (허용오차 0)."""
        d = self.delta(mm, cond)
        assert torch.isfinite(d).all(), "정련망 delta 에 NaN/Inf"
        assert float(d.abs().max()) == 0.0, f"delta 가 0 이 아니다 (max |d| = {float(d.abs().max()):g})"
