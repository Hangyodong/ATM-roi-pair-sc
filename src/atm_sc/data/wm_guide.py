"""백질 안내장 — WM 확률맵에서 만든 **기울기가 살아있는** 장.

왜: 원래 WM 확률맵으로 점유 손실을 걸면 백질에서 떨어진 가닥의 기울기가 정확히 0 이다
(확률이 그 바깥에서 평평하다). 손실은 크게 나오는데 밀어줄 방향이 없어 학습이 안 된다
(합성 검증: 바깥 가닥 손실 0.90, 기울기 노름 0.0).

거리 변환으로 램프를 만든다.
    guide = max(wm_prob, CAP * clip(1 - d/D, 0, 1)),  d = (wm > 0.5) 까지의 유클리드 거리 [mm]

램프에 상한 CAP=0.5 를 두는 이유: 상한이 1 이면 백질에서 5 mm 떨어진 점의 램프가 0.92 라
target 0.9 를 이미 넘겨 손실이 0 이 된다 (실측: 손실 0 인데 실제 점유 0.557, GT 0.825).
상한을 0.5 로 낮추면 먼 거리는 램프가 끌어당기고 **마지막 접근은 원 WM 확률**이 맡는다.
백질 안은 1, D mm 밖은 0, 그 사이는 기울기가 1/D 로 일정하다. D=60 은 실측 근거다: 생성 가닥 점의
최대 WM 거리가 62 mm 라 그보다 짧으면 멀리 나간 가닥에 기울기가 0 이 된다 (GT 는 최대 37 mm). W 격자가 1 mm 등방이라
voxel 거리 = mm 거리다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import CACHE
from .wm_segment import tissue_path

D_MM = 60.0
RAMP_CAP = 0.5


def guide_path(sub: str, source: str = "syn") -> Path:
    return CACHE / f"{sub}_wmguide_{source}_W.npy"


def build(sub: str, source: str = "syn", d_mm: float = D_MM) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    p = tissue_path(CACHE, sub, source)
    assert p.exists(), f"{sub}: {source} 조직맵이 없다 ({p.name}) -> scripts/69_tissue_extract.py --source {source}"
    wm = np.load(p)["wm"].astype(np.float32) / 255.0
    core = wm > 0.5
    assert core.sum() > 100_000, f"{sub}: WM core 가 너무 작다 ({int(core.sum())})"
    d = distance_transform_edt(~core).astype(np.float32)        # W 격자는 1 mm 등방
    g = np.maximum(wm, RAMP_CAP * np.clip(1.0 - d / float(d_mm), 0.0, 1.0)).astype(np.float32)
    assert np.isfinite(g).all() and g.max() > 0.99 and g.min() >= 0.0
    return g


def load(sub: str, source: str = "syn", d_mm: float = D_MM) -> np.ndarray:
    p = guide_path(sub, source)
    if not p.exists():
        g = build(sub, source, d_mm)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.round(g * 255).astype(np.uint8)); tmp.rename(p)
    a = np.load(p).astype(np.float32) / 255.0
    assert a.max() > 0.99, f"{sub}: 안내장이 비었다"
    return a
