"""Full streamline -> SC edge-aligned segment 분해 (ATM_SC_EDGE_ALIGNED_BUNDLE_DUAL_REPRESENTATION_STRATEGY.md §4-7, §30-35).

**GT 정의 실측 (sub-000001, 전체 1M streamline):**
    Case A 인접 transition만    r=0.781  log r=0.569  edge 700 / GT 2785
    Case B 같은 streamline 의 모든 ROI 쌍  r=0.9986 log r=0.9912  edge 2774 / GT 2785
따라서 이 데이터의 GT pass-SC 는 **Case B** 다. 문서 §30 의 "all-pass-pair 방식" 에 해당하므로
edge segment 는 인접 transition 이 아니라 **같은 streamline 안에서 ROI_i 구간과 ROI_j 구간을 잇는 부분경로**
로 정의한다 (인접만 쓰면 GT edge 의 25 % 밖에 못 만든다).

분해 규칙 (§31-35):
  1. 점별 atlas 라벨 -> 연속 중복 제거된 방문 순서, ROI 별 dwell 구간 [first, last]
  2. min_dwell 미만으로 스치는 구간은 버린다 (경계 jitter, §35)
  3. 방문한 ROI 쌍 (i,j) 마다 두 dwell 구간을 잇는 부분경로를 잘라 n_points 로 재샘플 (§32-33)
  4. 부분경로 길이가 min_length_mm 미만이면 버린다 (§34)
"""
from __future__ import annotations

import numpy as np

from .tt_io import point_labels

SEG_POINTS = 32          # segment 는 full streamline(128) 보다 짧다 (§33: validation 으로 결정)
MIN_DWELL = 2            # ROI 안에 최소 이만큼 연속한 점이 있어야 방문으로 인정 (§35)
MIN_LEN_MM = 4.0         # 너무 짧은 segment 는 geometry 학습 의미가 없다 (§34)


def dwell_intervals(lab: np.ndarray, min_dwell: int = MIN_DWELL):
    """[T] 점 라벨(0=배경) -> [(roi0based, first, last)] 방문 구간. 같은 ROI 를 여러 번 지나면 여러 구간."""
    out = []
    T = len(lab)
    i = 0
    while i < T:
        r = lab[i]
        j = i
        while j + 1 < T and lab[j + 1] == r:
            j += 1
        if r > 0 and (j - i + 1) >= min_dwell:
            out.append((int(r) - 1, i, j))
        i = j + 1
    return out


def _resample(seg: np.ndarray, n: int) -> np.ndarray:
    """[m,3] -> [n,3] 호길이 등간격. m==1 이면 그 점을 반복한다."""
    if len(seg) == 1:
        return np.repeat(seg, n, axis=0)
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.repeat(seg[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, seg[:, k]) for k in range(3)], 1)


def streamline_segments(mm: np.ndarray, lab: np.ndarray, n_points: int = SEG_POINTS,
                        min_dwell: int = MIN_DWELL, min_length_mm: float = MIN_LEN_MM):
    """streamline 하나 -> [(pair(a,b) a<b, segment [n_points,3], length_mm)].

    같은 ROI 쌍이 여러 번 나타나면(경로가 되돌아오면) 가장 긴 부분경로 하나만 쓴다:
    GT SC 가 streamline 당 pair 를 1회만 세므로(Case B) segment 도 1개여야 한다.
    """
    iv = dwell_intervals(lab, min_dwell)
    best = {}
    for x in range(len(iv)):
        for y in range(x + 1, len(iv)):
            ra, ia, ja = iv[x]
            rb, ib, jb = iv[y]
            if ra == rb:
                continue
            key = (min(ra, rb), max(ra, rb))
            lo, hi = ia, jb                       # ROI_i 진입 ~ ROI_j 이탈 (부분경로 전체)
            span = hi - lo
            if key not in best or span > best[key][1] - best[key][0]:
                best[key] = (lo, hi)
    out = []
    for (a, b), (lo, hi) in best.items():
        seg = mm[lo:hi + 1]
        L = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
        if L < min_length_mm:
            continue
        out.append(((a, b), _resample(seg, n_points).astype(np.float32), L))
    return out


def decompose_bundle(S: np.ndarray, atlas: np.ndarray, affine: np.ndarray, n_roi: int,
                     n_points: int = SEG_POINTS, min_dwell: int = MIN_DWELL,
                     min_length_mm: float = MIN_LEN_MM, chunk: int = 20000):
    """[N,128,3] mm -> (pair_index [M] int32 = a*n_roi+b, segments [M,n_points,3] float32, lengths [M] float32).

    M ≈ N × (streamline 당 방문 ROI 쌍 수). 실측 평균 6.98 쌍/streamline (전체 tractogram 기준).
    """
    assert S.ndim == 3 and S.shape[2] == 3, S.shape
    keys, segs, lens = [], [], []
    T = S.shape[1]
    for i in range(0, len(S), chunk):
        blk = S[i:i + chunk].astype(np.float64)
        lab = point_labels(blk.reshape(-1, 3), atlas, affine).astype(np.int32).reshape(len(blk), T)
        for t in range(len(blk)):
            for (a, b), seg, L in streamline_segments(blk[t], lab[t], n_points, min_dwell, min_length_mm):
                keys.append(a * n_roi + b); segs.append(seg); lens.append(L)
    if not segs:
        return (np.zeros(0, np.int32), np.zeros((0, n_points, 3), np.float32), np.zeros(0, np.float32))
    return (np.asarray(keys, np.int32), np.stack(segs).astype(np.float32), np.asarray(lens, np.float32))


def segment_sc(pair_index: np.ndarray, n_roi: int) -> np.ndarray:
    """분해 결과로 만든 대칭 SC (검증용). GT .mat 와 비교하면 분해가 GT 정의와 맞는지 알 수 있다."""
    M = np.zeros((n_roi, n_roi), np.int64)
    if len(pair_index):
        a, b = pair_index // n_roi, pair_index % n_roi
        np.add.at(M, (a, b), 1); np.add.at(M, (b, a), 1)
    return M
