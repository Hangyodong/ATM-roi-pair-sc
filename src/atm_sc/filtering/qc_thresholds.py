"""Synthetic QC 임계값을 TRAIN 실제 streamline 분포에서 뽑는다 (GESTA QC 문서 §11, §16).

임의 상수(예: 꺾임각 60°)를 쓰면 real 이 통과하는지조차 모른 채 synthetic 을 버리게 된다.
실제로 곡률 기준 하나가 생성물의 74 %를 탈락시키고 있었다. 모든 임계값은
"TRAIN real 이 이 정도는 통과한다"는 분위수로 정의한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .t1_streamline_filter import max_turn_angles_deg, streamline_lengths_mm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = ROOT / "outputs" / "stats" / "qc_thresholds.json"


def winding_deg(S: np.ndarray) -> np.ndarray:
    """[N,P,3] -> 총 회전량(도). loop/헤맴 탐지 (§11.5). 직선이면 0, 한 바퀴면 360."""
    d = np.diff(S, axis=1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.clip(n, 1e-9, None)
    cos = np.clip((u[:, 1:] * u[:, :-1]).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos)).sum(1)


def end_to_end_ratio(S: np.ndarray) -> np.ndarray:
    """직선거리 / 경로길이. 1 이면 직선, 0 에 가까우면 제자리를 맴돈다."""
    L = streamline_lengths_mm(S)
    d = np.linalg.norm(S[:, -1] - S[:, 0], axis=-1)
    return d / np.clip(L, 1e-9, None)


@dataclass
class QCThresholds:
    """TRAIN real 분포 분위수. 'real 의 q% 는 통과한다'는 뜻이라 해석 가능하다."""
    min_length_mm: float
    max_length_mm: float
    max_turn_deg: float
    max_winding_deg: float
    min_end_ratio: float
    brain_inside_min: float
    dedup_tol_mm: float = 0.1
    n_real: int = 0
    quantiles: tuple = (1.0, 99.0)

    def save(self, path: Path = DEFAULT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH):
        assert Path(path).exists(), f"{path} 없음 (scripts/27_qc_thresholds.py 먼저)"
        d = json.loads(Path(path).read_text())
        d["quantiles"] = tuple(d.get("quantiles", (1.0, 99.0)))
        return cls(**d)

    def to_filter_config(self):
        from .t1_streamline_filter import FilterConfig
        return FilterConfig(min_length_mm=self.min_length_mm, max_length_mm=self.max_length_mm,
                            max_turn_deg=self.max_turn_deg, brain_inside_min=self.brain_inside_min,
                            dedup_tol_mm=self.dedup_tol_mm)


def measure(S: np.ndarray, inside_frac: np.ndarray | None = None, q=(1.0, 99.0)) -> QCThresholds:
    """real streamline [N,P,3] 에서 임계값을 만든다. q=(하위, 상위) 분위수."""
    assert S.ndim == 3 and S.shape[0] >= 100, S.shape
    lo, hi = q
    L, A, W, R = streamline_lengths_mm(S), max_turn_angles_deg(S), winding_deg(S), end_to_end_ratio(S)
    return QCThresholds(min_length_mm=float(np.percentile(L, lo)), max_length_mm=float(np.percentile(L, hi)),
                        max_turn_deg=float(np.percentile(A, hi)), max_winding_deg=float(np.percentile(W, hi)),
                        min_end_ratio=float(np.percentile(R, lo)),
                        brain_inside_min=float(np.percentile(inside_frac, lo)) if inside_frac is not None else 0.9,
                        n_real=int(len(S)), quantiles=(lo, hi))
