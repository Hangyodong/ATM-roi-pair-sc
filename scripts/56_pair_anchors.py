"""ROI 쌍별 **anchor 경로**와 **변위 반범위**를 계산한다 (디코더 재매개화 상수).

디코더가 절대 mm 를 맨몸으로 예측하면 탐색 공간이 전뇌 박스(152x189x163mm, 4.68e6 mm^3)
전체다. 대신 pair 마다 그룹 평균 경로 `anchor` 와 그 주변 변위 반범위 `half_range` 를 두고

    mm = anchor + half_range * tanh(z)

로 재매개화하면 탐색 공간이 pair 국소 박스로 줄어든다. 이 스크립트는 그 두 상수를
**train subject 만으로** 만든다 (val/test 가 섞이면 누수다).

방향 문제. streamline 은 방향이 없어서 그냥 평균하면 경로가 뭉개진다. 두 단계로 잡는다.
  1. 참조 경로를 pair (i,j) 의 **ROI centroid 를 잇는 직선**(i -> j)으로 잡는다.
     subject 와 무관한 기준이라 전 subject 에서 끝점 방향이 같아진다 (i 쪽이 앞).
  2. 각 가닥을 참조에 대한 SSD 로 뒤집고(`_flip_to`), 평균을 새 참조로 삼아 반복한다.
     기본 2회 반복 후 마지막 pass 에서 변위 히스토그램을 모은다.

변위 분포는 전체를 메모리에 못 올린다 (train 전체 약 2e7 가닥 x 128점 x 3축). 축별
히스토그램(0.25mm bin)을 누적해 99 분위를 읽는다.

실행:
    PYTHONPATH=src python scripts/56_pair_anchors.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROI_PAIRS = ROOT / "outputs" / "roi_pairs"
SPLIT = ROOT / "outputs" / "splits" / "train.txt"
OUT_NPZ = ROOT / "outputs" / "cache" / "pair_anchor_paths.npz"
OUT_JSON = ROOT / "outputs" / "eval" / "pair_anchors.json"

N_ROI = 82
N_POINTS = 128
BIN_W = 0.25          # 변위 히스토그램 bin 폭 (mm)
N_BINS = 640          # 0 ~ 160 mm


# --------------------------------------------------------------------------- 준비
def triu_pairs(n_roi: int = N_ROI) -> np.ndarray:
    """[P,2] upper-triangle pair 목록. 저장 순서의 유일한 정의다."""
    return np.stack(np.triu_indices(n_roi, 1), 1).astype(np.int64)


def pair_lookup(pairs: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """[n_roi, n_roi] (i,j) -> pair index, 없으면 -1."""
    m = np.full((n_roi, n_roi), -1, np.int32)
    m[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs), dtype=np.int32)
    return m


def roi_centroids() -> np.ndarray:
    """[82,3] ROI 무게중심 (mm). 아틀라스를 W 격자(193x229x193 @1mm)로 올려서 계산."""
    from atm_sc.models.roi_pool import atlas_on_feature_grid, check_lateralization, roi_centroids_mm
    from atm_sc.spaces import W_AFFINE, W_SHAPE

    lab, aff = atlas_on_feature_grid(feat_shape=W_SHAPE, in_shape=W_SHAPE,
                                     in_affine=W_AFFINE, return_affine=True)
    lat = check_lateralization(lab, aff)                 # 좌우 반전이면 여기서 죽는다
    c = roi_centroids_mm(lab, aff)
    assert c.shape == (N_ROI, 3) and np.isfinite(c).all(), c.shape
    print(f"ROI centroid {c.shape}  좌우검사 L {lat['mean_x_left']:.1f} / R {lat['mean_x_right']:.1f} mm",
          flush=True)
    return c.astype(np.float64)


def straight_paths(cent: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """[P,128,3] centroid[i] -> centroid[j] 직선. 초기 참조이자 무효 pair 의 대체 anchor."""
    a, b = cent[pairs[:, 0]], cent[pairs[:, 1]]          # [P,3]
    t = np.linspace(0.0, 1.0, N_POINTS)[None, :, None]
    return a[:, None, :] * (1.0 - t) + b[:, None, :] * t


def train_subjects(limit: int | None) -> list[str]:
    subs = [s.strip() for s in open(SPLIT) if s.strip()]
    assert len(subs) == len(set(subs)), "train.txt 에 중복 subject"
    subs = [s for s in subs if (ROI_PAIRS / s / "bundles.npz").exists()]
    assert len(subs) >= 100, f"bundles.npz 가 있는 train subject 가 {len(subs)}명뿐"
    return subs[:limit] if limit else subs


# --------------------------------------------------------------------------- 정렬
def _flip_to(S: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """[n,128,3] 을 참조 곡선 방향으로 정렬 (scripts/52 와 동일 규약).

    뒤집힌 가닥을 섞어서 평균하면 anchor 가 두 방향의 평균으로 뭉개진다.
    """
    d0 = ((S - ref) ** 2).sum((1, 2))
    d1 = ((S[:, ::-1] - ref) ** 2).sum((1, 2))
    out = S.copy()
    out[d1 < d0] = S[d1 < d0][:, ::-1]
    return out


def load_subject(sub: str, lut: np.ndarray, min_n: int):
    """(pair index, streamlines float32) 목록. min_n 미만인 pair 는 버린다."""
    z = np.load(ROI_PAIRS / sub / "bundles.npz")
    pid, off = z["pair_ids"], z["pair_offsets"]
    S = z["streamlines"]
    assert pid.ndim == 2 and pid.shape[1] == 2, pid.shape
    assert len(off) == len(pid) + 1 and off[0] == 0 and off[-1] == len(S), (len(off), len(pid))
    assert S.ndim == 3 and S.shape[1:] == (N_POINTS, 3), S.shape
    assert pid.min() >= 0 and pid.max() < N_ROI, (pid.min(), pid.max())
    assert (pid[:, 0] < pid[:, 1]).all(), f"{sub}: pair_ids 가 upper-triangle 이 아니다"
    n = np.diff(off)
    keep = np.flatnonzero(n >= min_n)
    out = []
    for k in keep:
        p = int(lut[int(pid[k, 0]), int(pid[k, 1])])
        assert p >= 0, (sub, pid[k])
        out.append((p, np.asarray(S[off[k]:off[k + 1]], np.float32)))
    return out


# --------------------------------------------------------------------------- pass
def anchor_pass(subs, lut, ref, min_n, tag):
    """참조 ref 에 정렬해 pair 별 **subject 평균의 평균** 경로를 만든다.

    가닥 수가 많은 subject 가 anchor 를 지배하지 않도록 subject 단위로 균등 가중한다
    (scripts/52 의 build_paths 와 같은 규약).
    """
    P = len(ref)
    acc = np.zeros((P, N_POINTS, 3), np.float64)
    n_sub = np.zeros(P, np.int32)
    n_str = np.zeros(P, np.int64)
    t0 = time.time()
    for si, sub in enumerate(subs):
        for p, S in load_subject(sub, lut, min_n):
            acc[p] += _flip_to(S, ref[p].astype(np.float32)).mean(0)
            n_sub[p] += 1
            n_str[p] += len(S)
        if (si + 1) % 20 == 0 or si + 1 == len(subs):
            print(f"  [{tag}] {si + 1}/{len(subs)} subject  "
                  f"({time.time() - t0:.0f}s, pair {int((n_sub > 0).sum())}종)", flush=True)
    has = n_sub > 0
    out = ref.copy()
    out[has] = acc[has] / n_sub[has, None, None]
    return out, n_sub, n_str


def spread_pass(subs, lut, anchor, min_n, tag):
    """최종 anchor 기준 변위 |S - anchor| 의 축별 히스토그램을 누적한다."""
    P = len(anchor)
    hist = np.zeros((P, 3 * N_BINS), np.int64)
    ax_off = (np.arange(3, dtype=np.int32) * N_BINS)
    t0 = time.time()
    for si, sub in enumerate(subs):
        for p, S in load_subject(sub, lut, min_n):
            A = anchor[p].astype(np.float32)
            d = np.abs(_flip_to(S, A) - A)
            b = np.minimum((d * (1.0 / BIN_W)).astype(np.int32), N_BINS - 1) + ax_off
            hist[p] += np.bincount(b.ravel(), minlength=3 * N_BINS)
        if (si + 1) % 20 == 0 or si + 1 == len(subs):
            print(f"  [{tag}] {si + 1}/{len(subs)} subject ({time.time() - t0:.0f}s)", flush=True)
    return hist.reshape(P, 3, N_BINS)


def hist_quantile(hist: np.ndarray, q: float):
    """[P,3,N_BINS] 카운트 -> [P,3] q 분위 (bin 내 선형보간). 총 카운트 0 이면 NaN."""
    c = np.cumsum(hist, axis=2)
    tot = c[:, :, -1]
    target = q * tot
    k = np.array([[np.searchsorted(c[p, a], target[p, a]) if tot[p, a] > 0 else 0
                   for a in range(3)] for p in range(len(hist))], np.int64)
    k = np.minimum(k, N_BINS - 1)
    below = np.take_along_axis(c, k[:, :, None] - 1, 2)[:, :, 0]
    below = np.where(k == 0, 0.0, below)
    cnt = np.take_along_axis(hist, k[:, :, None], 2)[:, :, 0]
    frac = np.where(cnt > 0, (target - below) / np.maximum(cnt, 1), 0.0)
    val = (k + np.clip(frac, 0.0, 1.0)) * BIN_W
    return np.where(tot > 0, val, np.nan), k, tot


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-streamlines", type=int, default=4,
                    help="subject 당 이 미만이면 그 subject 는 해당 pair 에서 제외")
    ap.add_argument("--min-subjects", type=int, default=10,
                    help="기여 subject 가 이 미만이면 valid=False")
    ap.add_argument("--iters", type=int, default=2, help="anchor 정렬 반복 횟수")
    ap.add_argument("--pct", type=float, default=99.0, help="half_range 분위 (%%)")
    ap.add_argument("--min-half-range", type=float, default=5.0, help="half_range 하한 (mm)")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 subject 수 제한")
    a = ap.parse_args()
    assert 0 < a.pct < 100 and a.iters >= 1 and a.min_half_range > 0

    pairs = triu_pairs()
    P = len(pairs)
    assert P == 3321, P
    lut = pair_lookup(pairs)
    cent = roi_centroids()
    line = straight_paths(cent, pairs)
    subs = train_subjects(a.limit or None)
    print(f"train subject {len(subs)}명 / pair {P}개 / min_streamlines={a.min_streamlines} "
          f"min_subjects={a.min_subjects} iters={a.iters}", flush=True)

    ref = line
    prev = None
    shifts = []
    for it in range(a.iters):
        prev = ref
        ref, n_sub, n_str = anchor_pass(subs, lut, ref, a.min_streamlines, f"anchor {it + 1}/{a.iters}")
        moved = np.linalg.norm(ref - prev, axis=2)[n_sub > 0]
        shifts.append(float(np.median(moved)))
        print(f"  anchor pass {it + 1}: 점당 이동 중앙 {shifts[-1]:.2f}mm  최대 {moved.max():.1f}mm",
              flush=True)
    anchor = ref.astype(np.float64)

    hist = spread_pass(subs, lut, anchor, a.min_streamlines, "spread")
    hr_raw, kbin, tot = hist_quantile(hist, a.pct / 100.0)

    valid = n_sub >= a.min_subjects
    assert valid.sum() > 0, "valid pair 가 하나도 없다 -- min_subjects 확인"

    # 무효 pair: anchor 는 centroid 직선, half_range 는 유효 pair 의 축별 90 분위(넉넉하게).
    fallback = np.nanpercentile(hr_raw[valid], 90, axis=0)
    anchor[~valid] = line[~valid]
    half = np.where(np.isfinite(hr_raw), hr_raw, np.nan)
    half[~valid] = fallback
    assert np.isfinite(half).all(), f"half_range 결측 {int((~np.isfinite(half)).sum())}칸"
    half = np.maximum(half, a.min_half_range)

    # 99 분위가 오버플로 bin 에 걸리면 값이 잘린 것이다
    over = int((kbin[valid] >= N_BINS - 1).sum())
    assert over == 0, f"valid pair 의 {over}개 축에서 변위 분위가 {N_BINS * BIN_W:.0f}mm 를 넘었다"

    # --- 검증 --------------------------------------------------------------
    assert np.isfinite(anchor).all(), "anchor 에 NaN/Inf"
    seglen = np.linalg.norm(np.diff(anchor, axis=1), axis=2).sum(1)
    assert seglen.min() > 0, f"길이 0 인 anchor {int((seglen <= 0).sum())}개"
    assert half.min() > 0, half.min()
    assert np.array_equal(pairs, np.stack(np.triu_indices(N_ROI, 1), 1)), "pair 순서가 triu 와 다르다"
    assert anchor.shape == (P, N_POINTS, 3) and half.shape == (P, 3)

    # 끝점이 규약대로 i -> j 인지 (참조가 i->j 직선이므로 뒤집혔으면 정렬이 깨진 것)
    d_head_i = np.linalg.norm(anchor[:, 0] - cent[pairs[:, 0]], axis=1)
    d_tail_j = np.linalg.norm(anchor[:, -1] - cent[pairs[:, 1]], axis=1)
    d_head_j = np.linalg.norm(anchor[:, 0] - cent[pairs[:, 1]], axis=1)
    wrong = int(((d_head_i > d_head_j) & valid).sum())
    assert wrong <= 0.02 * valid.sum(), f"끝점 방향이 뒤집힌 valid pair {wrong}개"

    # 탐색 공간: pair 국소 박스(2*half_range) vs 전뇌 박스
    from atm_sc.models.roi_atm import brain_box_mm
    lo, hi = brain_box_mm()
    brain_vol = float(np.prod(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)))
    box_vol = np.prod(2.0 * half, axis=1)
    ratio = brain_vol / box_vol

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        anchor=anchor.astype(np.float32),
        half_range=half.astype(np.float32),
        valid=valid.astype(bool),
        n_subjects=n_sub.astype(np.int32),
        n_streamlines=n_str.astype(np.int64),
        pairs=pairs.astype(np.int16),
    )

    def q(x, ps=(5, 50, 95)):
        return [round(float(v), 3) for v in np.percentile(np.asarray(x, np.float64), ps)]

    rep = {
        "script": "scripts/56_pair_anchors.py",
        "npz": str(OUT_NPZ.relative_to(ROOT)),
        "split": "train only (outputs/splits/train.txt)",
        "n_subjects_used": len(subs),
        "params": {"min_streamlines": a.min_streamlines, "min_subjects": a.min_subjects,
                   "iters": a.iters, "pct": a.pct, "min_half_range_mm": a.min_half_range,
                   "hist_bin_mm": BIN_W, "hist_max_mm": BIN_W * N_BINS},
        "orientation": {
            "convention": "pair (i,j), i<j: anchor[0] 은 ROI i 쪽, anchor[127] 은 ROI j 쪽. "
                          "초기 참조 = centroid[i]->centroid[j] 직선, 이후 SSD flip 반복.",
            "flip_iters": a.iters,
            "median_point_shift_mm_per_iter": [round(s, 3) for s in shifts],
            "reversed_valid_pairs": wrong,
            "endpoint_dist_head_to_roi_i_mm": q(d_head_i[valid]),
            "endpoint_dist_tail_to_roi_j_mm": q(d_tail_j[valid]),
        },
        "coverage": {
            "n_pairs": int(P),
            "n_valid": int(valid.sum()),
            "frac_valid": round(float(valid.mean()), 4),
            "n_pairs_seen": int((n_sub > 0).sum()),
            "n_pairs_seen_but_invalid": int(((n_sub > 0) & ~valid).sum()),
            "n_subjects_valid": {"min": int(n_sub[valid].min()), "median": float(np.median(n_sub[valid])),
                                 "max": int(n_sub[valid].max()), "mean": round(float(n_sub[valid].mean()), 2)},
            "n_streamlines_total": int(n_str.sum()),
            "n_streamlines_per_valid_pair": q(n_str[valid]),
        },
        "half_range_mm": {
            "valid": {ax: q(half[valid, k]) for k, ax in enumerate("xyz")},
            "all": {ax: q(half[:, k]) for k, ax in enumerate("xyz")},
            "min": round(float(half.min()), 3), "max": round(float(half.max()), 3),
            "n_at_floor": int((half <= a.min_half_range + 1e-6).sum()),
            "fallback_for_invalid_mm": [round(float(v), 3) for v in np.maximum(fallback, a.min_half_range)],
        },
        "search_volume": {
            "brain_box_mm": [round(float(v), 1) for v in (np.asarray(hi) - np.asarray(lo))],
            "brain_box_volume_mm3": round(brain_vol, 1),
            "pair_box_volume_mm3_valid": q(box_vol[valid]),
            "reduction_ratio_valid": {"median": round(float(np.median(ratio[valid])), 2),
                                      "p5": round(float(np.percentile(ratio[valid], 5)), 2),
                                      "p95": round(float(np.percentile(ratio[valid], 95)), 2),
                                      "mean": round(float(ratio[valid].mean()), 2)},
            "reduction_ratio_at_median_box": round(brain_vol / float(np.median(box_vol[valid])), 2),
        },
        "anchor_length_mm": q(seglen[valid]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print(f"\n저장 {OUT_NPZ.relative_to(ROOT)}  ({OUT_NPZ.stat().st_size / 1e6:.1f} MB)")
    print(f"valid {int(valid.sum())}/{P} pair, 기여 subject 중앙 {np.median(n_sub[valid]):.0f}명 "
          f"({int(n_sub[valid].min())}~{int(n_sub[valid].max())})")
    print("half_range 중앙 (x,y,z) = " +
          ", ".join(f"{np.median(half[valid, k]):.1f}" for k in range(3)) + " mm")
    print(f"탐색 박스 축소 배수 중앙 {np.median(ratio[valid]):.1f}x "
          f"(전뇌 {brain_vol:.3g} mm^3 / pair 박스 중앙 {np.median(box_vol[valid]):.4g} mm^3)")
    print(f"리포트 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
