"""pair 앵커 재매개화 — 디코더가 절대 mm 대신 그 쌍의 그룹 평균 경로 주변 변위를 낸다.

왜 필요한가 (실측)
------------------
| | 크기 mm | 부피 mm^3 |
|---|---|---|
| 이 프로젝트 전역 뇌 박스 | 152 x 189 x 163 | 4.68e6 |
| upstream 번들 박스 (중앙, `supp/*_coords_*.npy`) | 83 x 121 x 100 | 1.17e6 |
| **실제 ROI 쌍의 GT 범위 (중앙)** | **43 x 71 x 61** | **1.67e5** |

전역 박스가 실제 pair 범위의 **28배 부피 = 선형 해상도 3.0배 손실**이다. 그런데 진짜 피해자는
디코더가 아니다 -- 디코더 복원은 3.08mm 로 이미 GT 잡음 바닥(pair 천장 MDF 3.92mm)에 닿아 있다.
피해자는 **prior** 다: 좌표가 전뇌 절대값이라 64-d z 가 "3,321 쌍 중 어느 것 + 뇌 어디쯤 + 모양"
을 전부 담아야 한다. upstream 은 번들 정체성이 모델 가중치(30개 모델)에 있어서 z 는 번들 *내부*
변이만 담으면 됐다. prior NLL 71.6 vs arch_additive 15.6, 생성 MDF 19.5 vs 오라클 3 의 구조적 이유다.

재매개화:  mm = anchor[pair] + half_range[pair] * raw      (raw = tanh 출력, [-1,1])

alpha 로 전역 박스와 섞는다. **alpha = 0 이면 기존 동작과 bit-exact** 이므로 기존 checkpoint 를
그대로 싣고 램프업하며 효과를 잴 수 있다 (재매개화는 출력의 의미를 바꾸므로 한 번에 켜면
디코더가 무너진다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = ROOT / "outputs" / "cache" / "pair_anchor_paths.npz"


def upper_index(i: torch.Tensor, j: torch.Tensor, n_roi: int) -> torch.Tensor:
    """canonical (i<j) 를 np.triu_indices(n_roi, 1) 순서의 위치로. 앵커 배열과 같은 순서여야 한다."""
    i, j = i.long(), j.long()
    assert bool((i < j).all()), "canonical_pairs 를 먼저 통과시켜야 한다 (i < j)"
    return i * n_roi - i * (i + 1) // 2 + (j - i - 1)


class PairAnchor(nn.Module):
    """[P,128,3] 앵커 경로 + [P,3] 반범위. 학습하지 않는 buffer 다 (train 144명 통계)."""

    def __init__(self, anchor: np.ndarray, half_range: np.ndarray, valid: np.ndarray,
                 n_roi: int = 82, alpha: float = 0.0):
        super().__init__()
        a = torch.as_tensor(np.asarray(anchor), dtype=torch.float32)
        h = torch.as_tensor(np.asarray(half_range), dtype=torch.float32)
        v = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
        p = n_roi * (n_roi - 1) // 2
        assert a.shape == (p, 128, 3), (a.shape, p)
        assert h.shape == (p, 3), h.shape
        assert v.shape == (p,), v.shape
        assert torch.isfinite(a).all(), "앵커에 NaN/Inf"
        assert float(h.min()) > 0, f"half_range 에 0 이하가 있다 ({float(h.min())}) -- 재매개화가 죽는다"
        self.n_roi = int(n_roi)
        self.register_buffer("anchor", a)
        self.register_buffer("half_range", h)
        self.register_buffer("valid", v)
        # alpha 는 학습 대상이 아니라 스케줄 값이다. buffer 라 checkpoint 에 같이 저장된다.
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def set_alpha(self, a: float) -> None:
        assert 0.0 <= a <= 1.0, a
        self.alpha.fill_(float(a))

    def forward(self, raw: torch.Tensor, pairs: torch.Tensor, mm_global: torch.Tensor) -> torch.Tensor:
        """raw [N,128,3] ([-1,1]), pairs [N,2] canonical, mm_global [N,128,3] = 전역 박스 결과.
        -> alpha 로 섞은 mm. alpha = 0 이면 mm_global 과 **bit-exact**."""
        if float(self.alpha) == 0.0:
            return mm_global
        assert raw.shape == mm_global.shape, (raw.shape, mm_global.shape)
        assert pairs.shape == (raw.shape[0], 2), (pairs.shape, raw.shape)
        idx = upper_index(pairs[:, 0], pairs[:, 1], self.n_roi)
        assert int(idx.max()) < self.anchor.shape[0], (int(idx.max()), self.anchor.shape)
        mm_anchor = self.anchor[idx] + self.half_range[idx].unsqueeze(1) * raw
        al = self.alpha
        return (1.0 - al) * mm_global + al * mm_anchor


def load(path=None, n_roi: int = 82, alpha: float = 0.0) -> PairAnchor:
    path = Path(path) if path is not None else DEFAULT_PATH
    assert path.exists(), f"앵커 파일이 없다: {path} (scripts/56_pair_anchors.py 로 생성)"
    z = np.load(path, allow_pickle=False)
    for k in ("anchor", "half_range", "valid", "pairs"):
        assert k in z.files, f"{path} 에 {k} 가 없다: {z.files}"
    ref = np.stack(np.triu_indices(n_roi, 1), 1).astype(z["pairs"].dtype)
    assert np.array_equal(np.asarray(z["pairs"]), ref), (
        "앵커의 pair 순서가 np.triu_indices 와 다르다 -- 인덱싱이 조용히 어긋난다")
    return PairAnchor(z["anchor"], z["half_range"], z["valid"], n_roi=n_roi, alpha=alpha)
