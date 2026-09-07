"""T1 공간 정보만으로 합성 streamline 을 거르는 필터 (전략 문서 §28–30).

최종 inference 에는 dMRI 가 없으므로 FA/FODF 기준은 두지 않는다. atlas 라벨, brain mask,
기하(길이·꺾임각·근접 중복)만 쓴다. S 는 [N,128,3] mm (MNI152NLin6), pairs 는 [N,2] 0-based
ROI index (a<b), atlas 라벨은 1..R (0 = 배경, ROI index = 라벨-1).

기준은 각각 독립적으로 [N] bool 을 만들고 keep 은 켜진 기준 전부의 AND:
  finite    NaN/Inf 없음
  endpoint  양 끝점의 hard lookup 라벨이 {a,b} (순서 무관)
  length    min_length_mm <= 길이 <= max_length_mm
  curvature 연속 segment 사이 최대 꺾임각 <= max_turn_deg
  brain     brain mask 안 점 비율 >= brain_inside_min (mask 없으면 생략)
  dedup     같은 pair 에서 이미 keep 된 것과 점별 평균 거리 < dedup_tol_mm 이면 탈락
            (다른 기준을 전부 통과한 것들만 대상, index 가 앞선 것을 남김)

NaN/Inf 가 있는 streamline 은 기하 판정이 불가능하므로 모든 기준에서 실패로 센다. 꺼진 기준의
마스크는 전부 True 라서 stats 의 기준별 마스크의 AND 가 항상 keep 과 일치한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.tt_io import point_labels, segment_lengths


@dataclass
class FilterConfig:
    endpoint: bool = True            # 양 끝점이 {a,b} 에 (순서 무관) 떨어져야 함
    min_length_mm: float = 20.0
    max_length_mm: float = 220.0
    max_turn_deg: float = 60.0       # 연속 segment 사이 최대 허용 꺾임각
    brain_inside_min: float = 0.95   # brain mask 안 점 비율 하한 (mask None 이면 생략)
    max_winding_deg: float = 0.0     # 총 회전량 상한 (0 이면 생략). loop/헤맴 탐지 (GESTA QC §11.5)
    min_end_ratio: float = 0.0       # 직선거리/경로길이 하한 (0 이면 생략)
    dedup_tol_mm: float = 0.1        # 점별 평균 거리가 이보다 작으면 근접 중복. 0 이면 생략.
                                     # GT 로 보정(sub-000001, 15k streamline): 통과율 tol 0.1/0.2/0.5/1.0 에서
                                     # 0.999/0.981/0.761/0.482. 같은 bundle 의 진짜 GT 가 서로 비슷하므로
                                     # 1 mm 는 GT 의 절반을 지운다.
    require_finite: bool = True


def streamline_lengths_mm(S: np.ndarray) -> np.ndarray:
    """[N,P,3] -> [N] polyline 길이 (mm). NaN 이 있으면 그 streamline 만 NaN."""
    assert S.ndim == 3 and S.shape[2] == 3, S.shape
    N, P = S.shape[:2]
    return segment_lengths(S.reshape(-1, 3), np.full(N, P, np.int64))


def max_turn_angles_deg(S: np.ndarray) -> np.ndarray:
    """[N,P,3] -> [N] 연속 segment 사이 최대 꺾임각 (deg). NaN 이 있으면 NaN.

    길이 0 segment 는 방향이 없으므로 꺾임 0 으로 본다 (재샘플 후 겹친 점 때문에 탈락시키지 않는다).
    """
    assert S.ndim == 3 and S.shape[2] == 3 and S.shape[1] >= 3, S.shape
    d = np.diff(np.asarray(S, np.float64), axis=1)                    # [N,P-1,3]
    dot = (d[:, 1:] * d[:, :-1]).sum(-1)
    den = np.linalg.norm(d[:, 1:], axis=-1) * np.linalg.norm(d[:, :-1], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den == 0.0, 1.0, dot / den)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max(1)


_NBR = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _dedup(S64: np.ndarray, pairs: np.ndarray, cand: np.ndarray, tol: float) -> np.ndarray:
    """cand 인 streamline 을 index 순서로 보며 앞서 keep 된 같은 pair 와 근접 중복이면 False.

    후보 검색은 centroid 를 한 변 tol 인 격자에 넣고 같은 pair 의 이웃 27칸만 본다. 점별 평균
    거리 < tol 이면 삼각부등식으로 centroid 거리도 < tol 이라 이웃 밖은 중복일 수 없다 (누락 없음).
    좌표 여러 개를 반올림해 hash 하면 0.2 mm 만 밀려도 어느 한 좌표가 격자 경계를 넘어 후보를
    놓치므로 쓰지 않는다. 뒤집힌 방향(b->a)도 같은 곡선이므로 역순 거리도 본다.
    비용 O(N * k * P), k = 이웃 27칸 안의 같은 pair keep 수.
    """
    keep = np.ones(len(S64), bool)
    cell = np.floor(S64.mean(1) / tol).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(cand):
        a, b = int(pairs[i, 0]), int(pairs[i, 1])
        cx, cy, cz = (int(v) for v in cell[i])
        near = [j for o in _NBR for j in buckets.get((a, b, cx + o[0], cy + o[1], cz + o[2]), ())]
        if near:
            C = S64[near]
            d = np.minimum(np.linalg.norm(C - S64[i], axis=-1).mean(1),
                           np.linalg.norm(C - S64[i, ::-1], axis=-1).mean(1))
            if d.min() < tol:
                keep[i] = False
                continue
        buckets.setdefault((a, b, cx, cy, cz), []).append(int(i))
    return keep


def filter_streamlines(S: np.ndarray, pairs: np.ndarray, atlas: np.ndarray, affine: np.ndarray,
                       brain_mask: np.ndarray | None = None, mask_affine: np.ndarray | None = None,
                       cfg: FilterConfig = FilterConfig()) -> tuple[np.ndarray, dict]:
    """-> (keep [N] bool, stats). stats: n_in, n_keep, pass_rate, 기준별 pass rate (입력 전체 대비)."""
    assert isinstance(S, np.ndarray) and S.ndim == 3 and S.shape[2] == 3, getattr(S, "shape", type(S))
    assert np.issubdtype(S.dtype, np.floating), S.dtype
    N, P = S.shape[:2]
    assert N > 0 and P >= 3, S.shape                   # 빈 입력은 상류 오류; P<3 이면 꺾임각이 없다
    pairs = np.asarray(pairs)
    assert pairs.shape == (N, 2) and np.issubdtype(pairs.dtype, np.integer), (pairs.shape, pairs.dtype)
    assert atlas.ndim == 3 and np.issubdtype(atlas.dtype, np.integer), (atlas.shape, atlas.dtype)
    n_roi = int(atlas.max())
    assert n_roi >= 1, "atlas 에 ROI 가 없음"
    assert pairs.min() >= 0 and pairs.max() < n_roi and (pairs[:, 0] < pairs[:, 1]).all(), \
        "pairs 는 0-based ROI index, a<b, < atlas.max()"
    assert affine.shape == (4, 4) and np.isfinite(affine).all(), affine
    if brain_mask is not None:
        assert mask_affine is not None and mask_affine.shape == (4, 4), "brain_mask 에는 mask_affine 필요"
        assert brain_mask.ndim == 3 and (brain_mask.dtype == bool or np.issubdtype(brain_mask.dtype, np.integer)), \
            (brain_mask.shape, brain_mask.dtype)
        assert brain_mask.any(), "brain mask 가 비어 있음"
    assert 0 <= cfg.min_length_mm < cfg.max_length_mm and cfg.max_turn_deg > 0, cfg
    assert 0 <= cfg.brain_inside_min <= 1 and cfg.dedup_tol_mm >= 0, cfg

    ones = np.ones(N, bool)
    fin = np.isfinite(S).all(axis=(1, 2))
    S64 = S.astype(np.float64)
    S64[~fin] = 0.0                  # 비유한 좌표는 voxel index 로 못 쓴다; 판정은 fin 으로 실패시킨다

    lab = point_labels(S64[:, [0, -1]].reshape(-1, 3), atlas, affine).astype(np.int64).reshape(N, 2) - 1
    a, b = pairs[:, 0], pairs[:, 1]
    hit = ((lab[:, 0] == a) & (lab[:, 1] == b)) | ((lab[:, 0] == b) & (lab[:, 1] == a))
    endpoint = (hit & fin) if cfg.endpoint else ones
    L = streamline_lengths_mm(S64)
    length = (L >= cfg.min_length_mm) & (L <= cfg.max_length_mm) & fin
    curvature = (max_turn_angles_deg(S64) <= cfg.max_turn_deg) & fin
    if cfg.max_winding_deg > 0 or cfg.min_end_ratio > 0:
        from .qc_thresholds import end_to_end_ratio, winding_deg
        wind = ((winding_deg(S64) <= cfg.max_winding_deg) if cfg.max_winding_deg > 0 else ones) & \
               ((end_to_end_ratio(S64) >= cfg.min_end_ratio) if cfg.min_end_ratio > 0 else ones) & fin
    else:
        wind = ones
    if brain_mask is not None:       # mask 를 라벨 볼륨처럼 lookup: 격자 밖 = 0 = 밖
        inside = point_labels(S64.reshape(-1, 3), brain_mask, mask_affine) > 0
        brain = (inside.reshape(N, P).mean(1) >= cfg.brain_inside_min) & fin
    else:
        brain = ones
    finite = fin if cfg.require_finite else ones
    base = finite & endpoint & length & curvature & wind & brain
    dedup = _dedup(S64, pairs, base, cfg.dedup_tol_mm) if cfg.dedup_tol_mm > 0 else ones
    keep = base & dedup
    masks = {"finite": finite, "endpoint": endpoint, "length": length,
             "curvature": curvature, "winding": wind, "brain": brain, "dedup": dedup}
    stats = {"n_in": int(N), "n_keep": int(keep.sum()), "pass_rate": float(keep.mean()),
             **{k: float(m.mean()) for k, m in masks.items()}}
    return keep, stats
