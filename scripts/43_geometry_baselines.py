#!/usr/bin/env python
"""S5-e: tractogram 수준의 기하 기준선 실측.

지금까지 유일한 기준선이 "train 144명 GT SC 의 그룹 평균 행렬(r=0.945)" 이었는데,
그 행렬은 streamline 을 하나도 만들지 못한다 -- 생성기의 기준선이 될 수 없다.
여기서는 **다른 사람의 진짜 tractogram** 을 예측으로 제출했을 때 몇 점을 받는지를 잰다.

기준선 4종
    cross_subject     다른 test subject 의 GT (가장 중요. 모델은 최소한 이걸 넘어야 한다)
    self              같은 subject GT 를 무작위 두 표본으로 -> 지표의 천장
    subsample         GT 를 29k 만 뽑아 채점 -> 개수 차이만의 효과
    group_tractogram  train subject 여러 명의 GT 를 합친 풀에서 추출 -> 그룹 템플릿의 tractogram 판

채점 규약은 scripts/29_final_evaluation.py 의 --trk-eval 과 같게 맞춘다:
    bundle_geometry_metrics(pred, gt, voxel_mm=2.0), gt = bundles.npz 에서 8,000 개
    (실측한 모델 JSON 의 trk_geometry.n_gen == n_gt == 8000), pred 도 8,000 개.
지표 절대값은 표본 개수에 강하게 의존하므로 이 개수를 바꾸면 비교가 깨진다.

두 계열을 낸다.
    family "pairs"  outputs/roi_pairs/<sub>/bundles.npz (ROI pair 당 cap 256 로 자른 GT 부분집합).
                    모델 평가가 GT 로 쓴 바로 그 집합이므로 모델 수치와 직접 비교 가능.
    family "raw"    <sub>_tract.tt.gz 원본 1,000,000 streamline 에서 무작위 추출.
                    endpoint_in_roi / 길이 분포처럼 "진짜 tractogram 의 성질" 은 이쪽이 정직하다.

GPU 를 쓰지 않는다. 모델을 돌리지 않는다.
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

from atm_sc.data.paths import ATLAS, SC_MAT, tt_path                              # noqa: E402
from atm_sc.data.roi_groups import BLOCKS, N_CTX, TIERS, block_masks              # noqa: E402
from atm_sc.data.tt_io import hard_sc, load_streamlines, point_labels, resample_128  # noqa: E402
from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics            # noqa: E402
from atm_sc.evaluation.reproduction_metrics import (                             # noqa: E402
    bundle_distance, endpoint_distance, length_distribution, stratified_corr,
    valid_connection_rate)

N_ROI = 82
BUNDLE_DIR = ROOT / "outputs" / "roi_pairs"
# 뇌(MNI) 크기 범위. 이 밖으로 나가면 좌표 단위가 mm 가 아니거나 공간 가정이 틀린 것이다.
BRAIN_MM = (-110.0, 110.0)


# --------------------------------------------------------------------------- 로드 + 검사
def load_atlas():
    import nibabel as nib
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    have = set(np.unique(atlas).tolist()) - {0}
    missing = sorted(set(range(1, N_ROI + 1)) - have)
    assert not missing, f"아틀라스에 없는 라벨 {missing} (기대 1..{N_ROI})"
    assert not (have - set(range(1, N_ROI + 1))), f"기대 밖 라벨 {sorted(have - set(range(1, N_ROI + 1)))}"
    return atlas, img.affine


def _stride(S, n=2000):
    """bundles.npz 는 pair 순으로 정렬되어 있어 앞 n 개만 보면 bbox 가 국소적이다."""
    return S[:: max(1, len(S) // n)].astype(np.float32)


def check_streamlines(name: str, S: np.ndarray) -> list:
    """streamline 배열의 기본 건전성. 조용히 틀리는 것을 막는 게 목적이다."""
    assert S.ndim == 3 and S.shape[1:] == (128, 3), f"{name}: shape {S.shape}"
    assert len(S) > 0, f"{name}: streamline 이 0 개"
    P = S.reshape(-1, 3).astype(np.float64)
    n_bad = int((~np.isfinite(P)).sum())
    assert n_bad == 0, f"{name}: 비유한 좌표 {n_bad} 개"
    assert np.abs(P).max() > 1.0, f"{name}: 좌표가 전부 0 근처 (빈 tractogram)"
    lo, hi = P.min(0), P.max(0)
    assert lo.min() > BRAIN_MM[0] and hi.max() < BRAIN_MM[1], \
        f"{name}: bbox {lo.tolist()}..{hi.tolist()} 가 뇌 크기(mm) 범위를 벗어남 -- 단위/공간 가정 오류"
    assert (hi - lo).min() > 30.0, f"{name}: bbox 가 너무 납작하다 {(hi - lo).tolist()}"
    return [lo.tolist(), hi.tolist()]


def assert_bbox_overlap(na, ba, nb, bb, min_iou: float = 0.5):
    """같은 QSDR 템플릿 공간이면 두 subject 의 bbox 는 거의 일치해야 한다."""
    lo = np.maximum(ba[0], bb[0]); hi = np.minimum(ba[1], bb[1])
    if (hi <= lo).any():
        raise AssertionError(f"{na} 와 {nb} 의 bbox 가 겹치지 않는다 -- 공간 가정이 틀렸다")
    inter = float(np.prod(hi - lo))
    va = float(np.prod(np.asarray(ba[1]) - np.asarray(ba[0])))
    vb = float(np.prod(np.asarray(bb[1]) - np.asarray(bb[0])))
    iou = inter / (va + vb - inter)
    assert iou > min_iou, f"{na} vs {nb}: bbox IoU {iou:.3f} 가 너무 낮다 -- 공간 정합을 의심하라"
    return iou


def load_bundles(sub: str):
    """bundles.npz -> (streamlines float16 [n,128,3], pair [n,2] int16, pair_ids, pair_offsets, count_full)."""
    z = np.load(BUNDLE_DIR / sub / "bundles.npz")
    S = z["streamlines"]
    off = z["pair_offsets"]
    ids = z["pair_ids"].astype(np.int64)
    assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
    assert off[0] == 0 and off[-1] == len(S) and len(ids) == len(off) - 1
    pair = np.repeat(ids, np.diff(off), axis=0).astype(np.int64)
    return S, pair, ids, off, z["pair_count_full"].astype(np.int64)


def load_raw_sample(sub: str, n: int, seed: int, chunk: int = 100_000):
    """.tt.gz 원본에서 무작위 n 개를 128 점으로 재샘플해 가져온다 (전체를 메모리에 올리지 않는다)."""
    hdr, gen = load_streamlines(str(tt_path(sub)), chunk=chunk)
    # 총 개수를 모르므로 1패스로 훑으면서 reservoir 대신 "먼저 총수를 세는" 대신
    # chunk 를 돌며 각 chunk 에서 균등 확률로 뽑는다: 전체 개수는 1M 로 알려져 있으나
    # 파일마다 다를 수 있으므로 reservoir sampling 으로 정확한 균등 추출을 보장한다.
    rng = np.random.default_rng(seed)
    res = np.zeros((n, 128, 3), np.float16)
    seen = 0
    for mm, npts in gen():
        starts = np.concatenate([[0], np.cumsum(npts)[:-1]])
        m = len(npts)
        if seen < n:                                    # 아직 저장소가 안 찼다
            take = min(n - seen, m)
            idx = np.arange(take)
            res[seen:seen + take] = resample_128(
                mm[np.concatenate([np.arange(starts[i], starts[i] + npts[i]) for i in idx])], npts[idx])
            rest = np.arange(take, m)
            seen += take
        else:
            rest = np.arange(m)
        if len(rest):
            # rest[k] 는 전역 순번 seen+k. 확률 n/(seen+k+1) 로 저장소의 무작위 자리를 대체한다.
            pos = seen + np.arange(len(rest))
            keep = rng.random(len(rest)) < (n / (pos + 1.0))
            if keep.any():
                sel = rest[keep]
                slot = rng.integers(0, n, size=int(keep.sum()))
                slot, uniq = np.unique(slot[::-1], return_index=True)   # 같은 자리 중복은 마지막 것만
                sel = sel[::-1][uniq]
                idx = np.concatenate([np.arange(starts[i], starts[i] + npts[i]) for i in sel])
                res[slot] = resample_128(mm[idx], npts[sel])
            seen += len(rest)
    assert seen > 0, f"{sub}: .tt.gz 에서 streamline 을 하나도 읽지 못함"
    assert seen >= n, f"{sub}: streamline {seen} 개 < 요구 {n} 개"
    return res, seen


# --------------------------------------------------------------------------- 채점
def _lengths(S):
    return np.linalg.norm(np.diff(np.asarray(S, np.float64), axis=1), axis=-1).sum(1)


def score_wb(pred, gt, voxel_mm=2.0, dist_n=32, seed=0):
    """whole-brain 채점. 모델 평가(29_final_evaluation --trk-eval)와 같은 함수/인자."""
    pred = np.ascontiguousarray(pred, np.float32)
    gt = np.ascontiguousarray(gt, np.float32)
    g = bundle_geometry_metrics(pred, gt, voxel_mm=voxel_mm)
    assert 0.0 < g["dice"] < 1.0, f"dice 가 {g['dice']} -- 완전 불일치/완전 일치는 계산 오류다"
    Lp, Lg = _lengths(pred), _lengths(gt)
    out = {"dice": g["dice"], "coverage": g["coverage"], "overreach": g["overreach"],
           "n_pred": int(len(pred)), "n_gt": int(len(gt)),
           "len_mean_mm": float(Lp.mean()), "len_median_mm": float(np.median(Lp)),
           "gt_len_mean_mm": float(Lg.mean()), "gt_len_median_mm": float(np.median(Lg)),
           "duplicate_ratio": g["duplicate_ratio"], "valid_ratio": g["valid_ratio"]}
    out.update(length_distribution(Lp, Lg))
    out.update(bundle_distance(pred, gt, max_n=dist_n, seed=seed, hausdorff=False))
    out["endpoint_dist_mm"] = endpoint_distance(pred, gt, max_n=dist_n, seed=seed)
    return out


def score_conn(pred, pairs, atlas, affine):
    """valid_connection_rate. pairs 는 예측 가닥이 '의도한' ROI 쌍 (0-based, 미정의는 -1)."""
    pred = np.ascontiguousarray(pred, np.float32)
    ok = (np.asarray(pairs) >= 0).all(1)
    # endpoint_in_roi 는 pair 정의와 무관하게 전체에서 잰다 (모델의 분모와 같다)
    allm = valid_connection_rate(pred, np.zeros((len(pred), 2), np.int64), atlas, affine, N_ROI)
    out = {"endpoint_in_roi": allm["endpoint_in_roi"], "pair_defined_frac": float(ok.mean())}
    if ok.any():
        d = valid_connection_rate(pred[ok], np.asarray(pairs)[ok], atlas, affine, N_ROI)
        # 모델은 모든 생성 가닥이 의도 pair 를 가지므로 분모가 전체다 -> 미정의는 invalid 로 센다
        out.update({"valid_conn": float(d["valid_conn"] * ok.mean()),
                    "partial_conn": float(d["partial_conn"] * ok.mean()),
                    "valid_conn_given_pair": d["valid_conn"]})
    return out


def edge_length_corr(pred, gt_len_mat, gt_cnt_mat, atlas, affine, mode="pass"):
    """예측 tractogram 의 edge 평균 길이 행렬 vs GT edge 평균 길이 (모델 length.r 과 같은 뜻)."""
    S = np.ascontiguousarray(pred, np.float32)
    mm = S.reshape(-1, 3)
    npts = np.full(len(S), S.shape[1], np.int64)
    W, L = hard_sc(mm, npts, atlas, affine, N_ROI, mode)
    iu = np.triu_indices(N_ROI, 1)
    pm = np.zeros_like(L); pm[W > 0] = L[W > 0] / W[W > 0]
    gm = np.zeros_like(gt_len_mat, np.float64)
    nz = gt_cnt_mat > 0
    gm[nz] = gt_len_mat[nz] / gt_cnt_mat[nz]
    a, b = pm[iu], gm[iu]
    m = (a > 0) & (b > 0)
    assert m.sum() >= 10, f"공통 edge 가 {m.sum()} 개뿐"
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return {"len_edge_r": r, "len_edge_n": int(m.sum()),
            "len_edge_mean_pred_mm": float(a[m].mean()), "len_edge_mean_gt_mm": float(b[m].mean()),
            "n_edges_pred": int((a > 0).sum()), "n_edges_gt": int((b > 0).sum())}


def pair_dice(pred_b, gt_b, voxel_mm=2.0):
    if len(pred_b) < 2 or len(gt_b) < 2:
        return None
    g = bundle_geometry_metrics(np.ascontiguousarray(pred_b, np.float32),
                                np.ascontiguousarray(gt_b, np.float32), voxel_mm=voxel_mm)
    return {"dice": g["dice"], "coverage": g["coverage"], "overreach": g["overreach"],
            "n_pred": int(len(pred_b)), "n_gt": int(len(gt_b))}


def take(S, n, rng, exclude=None):
    """S 에서 n 개 무작위 추출 (exclude 인덱스 제외). float32 로 승격."""
    idx = np.arange(len(S))
    if exclude is not None:
        keep = np.ones(len(S), bool); keep[exclude] = False
        idx = idx[keep]
    assert len(idx) >= n, f"표본 부족: {len(idx)} < {n}"
    sel = np.sort(rng.choice(idx, n, replace=False))
    return S[sel].astype(np.float32), sel


def model_gt_sample(S, n):
    """모델 평가와 **같은** GT 표본 (scripts/29: default_rng(0) -> sort(choice))."""
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(S), min(n, len(S)), replace=False))
    return S[sel].astype(np.float32), sel


def summarize(rows: list) -> dict:
    """subject 별 dict 리스트 -> 평균/표준편차."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], float)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = float(v.mean())
            out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n_obs"] = len(rows)
    return out


# --------------------------------------------------------------------------- W1-c: SC
# 기하 기준선(dice)과 SC 상관을 같은 tractogram 에서 재서 dice -> SC r 대응표를 만든다.
# 검증 대상 주장: "그룹 tractogram(dice 0.614) 이면 SC r 이 0.9 근처" 그리고 "0.713 -> 0.9x
# 의 격차는 개인차 부재가 아니라 기하 품질 때문". 0.945 는 그룹 평균 SC **행렬** 값이지
# 그룹 tractogram 에서 SC 를 센 값이 아니므로 둘이 같다는 보장이 없다 -- 그걸 직접 잰다.
EXPECT_BLOCK_N = {"ctx-ctx": 2145, "ctx-sub": 1056, "sub-sub": 120}


def load_gt_sc(subs) -> dict:
    """.mat pass-SC (GT 정의: 같은 streamline 이 지난 모든 ROI 쌍, Case B). 한 번만 파싱한다."""
    import scipy.io as sio
    m = sio.loadmat(SC_MAT, variable_names=["data"])
    tbl = {str(r["subject"][0]): np.asarray(r["SC_weight"], np.float64) for r in m["data"][0]}
    out = {}
    for s_ in subs:
        assert s_ in tbl, f"{s_} 가 {SC_MAT.name} 에 없다"
        w = tbl[s_]
        check_sc(f"gt/{s_}", w)
        out[s_] = w
    return out


def check_sc(name: str, W: np.ndarray):
    """SC 행렬 건전성. 조용히 빈/비대칭 커넥톰이 흘러가는 것을 막는다."""
    W = np.asarray(W)
    assert W.shape == (N_ROI, N_ROI), f"{name}: shape {W.shape} != ({N_ROI},{N_ROI})"
    n_bad = int((~np.isfinite(W)).sum())
    assert n_bad == 0, f"{name}: NaN/Inf {n_bad} 개"
    assert np.array_equal(W, W.T), f"{name}: 비대칭"
    assert not np.any(np.diag(W)), f"{name}: 대각이 0 이 아니다"
    nnz = int((np.triu(W, 1) > 0).sum())
    assert nnz > 0, f"{name}: 비영 요소 0 개 -- 빈 커넥톰"
    return nnz


def sc_pass(S: np.ndarray, atlas, affine) -> np.ndarray:
    """streamline [n,128,3] -> pass 규칙 SC count. 기존 hard_sc 경로를 그대로 쓴다."""
    S = np.ascontiguousarray(S, np.float32)
    W, _ = hard_sc(S.reshape(-1, 3), np.full(len(S), S.shape[1], np.int64), atlas, affine, N_ROI, "pass")
    return W.astype(np.float64)


def flat_corr(pred: np.ndarray, gt: np.ndarray) -> dict:
    """stratified_corr (읽기 전용) 결과를 평탄화. 상삼각 off-diagonal 만 쓰는 것은 그쪽이 보장한다."""
    st = stratified_corr(pred, gt, n_roi=N_ROI, n_ctx=N_CTX)
    iu = np.triu_indices(N_ROI, 1)
    assert st["n"]["all"] == len(iu[0]) == 3321, st["n"]["all"]
    for b, n_exp in EXPECT_BLOCK_N.items():
        assert st["n"][b] == n_exp, f"block {b}: {st['n'][b]} != {n_exp}"
    assert sum(st["n"][b] for b in BLOCKS) == 3321
    out = {"sc_r": st["all"], "sc_r_log": st["all_log"]}
    out.update({f"tier_{t}": st["tier"][t] for t in TIERS})
    out.update({f"tier_log_{t}": st["tier_log"][t] for t in TIERS})
    out.update({f"block_{b}": st["block"][b] for b in BLOCKS})
    out.update({f"n_{t}": st["n"][t] for t in TIERS})
    return out


def endpoint_pair_ids(S, atlas, affine):
    """raw streamline 의 양 끝점 ROI 쌍 (0-based, 정렬). 끝점이 배경/같은 ROI 면 ok=False."""
    lab = point_labels(S[:, [0, -1]].reshape(-1, 3).astype(np.float32), atlas, affine)
    p = np.sort(np.stack([lab[0::2].astype(np.int64) - 1, lab[1::2].astype(np.int64) - 1], 1), 1)
    ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] != p[:, 1])
    return p, ok


def uniform_pair_draw(pairs, ok, n, rng):
    """존재하는 endpoint pair 마다 같은 개수를 뽑는다 = '기하는 그대로, 개수 배분만 망친' tractogram.
    dice 는 거의 그대로인데 SC r 만 무너지면 두 지표가 분리된다는 직접 증거가 된다."""
    idx = np.flatnonzero(ok)
    assert len(idx) > 0
    key = pairs[idx, 0] * N_ROI + pairs[idx, 1]
    o = np.argsort(key, kind="stable")
    idx, key = idx[o], key[o]
    bnd = np.r_[np.flatnonzero(np.r_[True, key[1:] != key[:-1]]), len(key)]
    n_uniq = len(bnd) - 1
    per = max(1, int(np.ceil(n / n_uniq)))
    sel = [c if len(c) <= per else rng.choice(c, per, replace=False)
           for c in (idx[s0:s1] for s0, s1 in zip(bnd[:-1], bnd[1:]))]
    sel = np.concatenate(sel)
    if len(sel) > n:
        sel = rng.choice(sel, n, replace=False)
    return np.sort(sel), n_uniq


def sc_summarize(rows: list) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = float(v.mean())
            out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out[k + "_min"] = float(v.min()); out[k + "_max"] = float(v.max())
    out["n_obs"] = len(rows)
    return out


def run_sc(a, cmd, atlas, affine, pairs_ab, group_subs, train):
    """각 기하 기준선 tractogram 에서 SC 를 실제로 계산해 GT(.mat pass-SC) 와 상관을 낸다."""
    t0_all = time.time()
    targets = [b for _, b in pairs_ab]
    gt_sc = load_gt_sc(sorted(set(targets) | set(group_subs) | set(train)))
    print(f"GT pass-SC {len(gt_sc)}명 로드", flush=True)

    # --- 참고 기준선: 그룹 평균 SC '행렬' (tractogram 이 아니다) ------------------
    # group32 는 group_subs 를 포함하는 상위집합이어야 한다 (같은 permutation 의 앞부분).
    group32 = [train[i] for i in np.random.default_rng(a.seed + 1).permutation(len(train))[: a.sc_group_n2]]
    assert group32[: len(group_subs)] == group_subs, "group32 가 group_subs 의 상위집합이 아니다"
    tmpl = {"group_mat_144": [s for s in train if s not in set(targets)],
            "group_mat_32": group32, "group_mat_8": group_subs}
    ref = {}
    for k, subs in tmpl.items():
        T = np.mean(np.stack([gt_sc[s] for s in subs]), 0)
        check_sc(k, T)
        ref[k] = sc_summarize([flat_corr(T, gt_sc[b]) for b in targets])
        ref[k]["n_template_subjects"] = len(subs)
    print(f"참고: 그룹 평균 SC 행렬 144명 r={ref['group_mat_144']['sc_r']:.4f} · "
          f"{len(group32)}명 r={ref['group_mat_32']['sc_r']:.4f} · "
          f"8명(=group tractogram 과 같은 pool) r={ref['group_mat_8']['sc_r']:.4f}", flush=True)

    # --- family 'pairs': bundles.npz. 기하 실행과 **같은 rng/순서** 로 같은 가닥을 고른다 ---
    print("group pool (bundles.npz) 구축...", flush=True)
    per = int(np.ceil(a.n_sample * 3 / len(group_subs)))
    pool = []
    for gs in group_subs:
        S, _, _, _, _ = load_bundles(gs)
        s_, _ = take(S, min(per, len(S)), np.random.default_rng(a.seed + 7))
        pool.append(s_.astype(np.float16))
        del S
    pool = np.concatenate(pool)
    check_streamlines("group_pool", _stride(pool))
    print(f"  풀 {len(pool):,}", flush=True)

    rows = {f: {k: [] for k in ("cross_subject", "self", "subsample", "group_tractogram")}
            for f in ("pairs", "raw")}
    rows["raw"]["group_tractogram_pool8"] = []
    rows["raw"][f"group_tractogram_pool{len(group32)}"] = []
    rows["raw"]["uniform_pair_counts"] = []
    sweep = []

    # raw group pool 2종: 기하 실행과 같은 것(4명x8000) + 8명 전체를 쓴 큰 것
    print("raw group pool...", flush=True)
    g_geo = np.concatenate([load_raw_sample(gs, max(4000, a.n_sample), a.seed)[0]
                            for gs in group_subs[: max(2, a.n_group // 2)]])
    g_all = [load_raw_sample(gs, a.sc_group_raw, a.seed)[0] for gs in group32]
    n8 = sum(len(x) for x in g_all[: len(group_subs)])
    g_full = np.concatenate(g_all); del g_all
    g_pool8 = g_full[:n8]                       # 앞부분이 group_subs 8명 그대로다
    check_streamlines("raw_group_geo", _stride(g_geo)); check_streamlines("raw_group_full", _stride(g_full))
    print(f"  geo-matched {len(g_geo):,} · pool8 {len(g_pool8):,} · "
          f"pool{len(group32)} {len(g_full):,}  ({time.time() - t0_all:.0f}s)", flush=True)

    for pi, (sa, sb) in enumerate(pairs_ab):
        t0 = time.time()
        gt = gt_sc[sb]
        # ---- pairs family (dice 0.574 / 0.614 / 0.783 이 나온 바로 그 가닥들) ----
        SA, _, _, _, _ = load_bundles(sa)
        SB, _, _, _, _ = load_bundles(sb)
        _, gt_idx = model_gt_sample(SB, a.n_sample)
        r_ab = np.random.default_rng(a.seed + 100 + pi)        # 기하 실행과 동일 (같은 순서로 소비)
        sel = {"cross_subject": take(SA, a.n_sample, r_ab)[0],
               "self": take(SB, a.n_sample, r_ab, exclude=gt_idx)[0],
               "subsample": take(SB, min(a.n_model, len(SB) - len(gt_idx)), r_ab, exclude=gt_idx)[0],
               "group_tractogram": take(pool, a.n_sample, r_ab)[0]}
        for name, S in sel.items():
            W = sc_pass(S, atlas, affine)
            nnz = check_sc(f"pairs/{name}/{sb}", W)
            m = {"subject": sb, "pred_from": sa if name == "cross_subject" else
                 ("group" if name == "group_tractogram" else sb),
                 "n_streamlines": int(len(S)), "n_edges_pred": nnz,
                 "n_edges_gt": int((np.triu(gt, 1) > 0).sum()), "sum_pred": float(W.sum())}
            m.update(flat_corr(W, gt))
            rows["pairs"][name].append(m)
        del SA, SB

        # ---- raw family (.tt.gz 원본 추출 = 진짜 whole-brain tractogram) ----
        RA, _ = load_raw_sample(sa, a.raw_sample, a.seed)
        RB, _ = load_raw_sample(sb, a.raw_sample, a.seed)
        check_streamlines(f"{sa}/raw", _stride(RA)); check_streamlines(f"{sb}/raw", _stride(RB))
        _, gt_idx = model_gt_sample(RB, a.n_sample)
        r_ab = np.random.default_rng(a.seed + 400 + pi)        # 기하 실행 raw 계열과 동일
        sel = {"cross_subject": take(RA, a.n_sample, r_ab)[0],
               "self": take(RB, a.n_sample, r_ab, exclude=gt_idx)[0],
               "subsample": take(RB, min(a.n_model, len(RB) - len(gt_idx)), r_ab, exclude=gt_idx)[0],
               "group_tractogram": take(g_geo, a.n_sample, r_ab)[0]}
        sel["group_tractogram_pool8"] = take(g_pool8, a.n_model, np.random.default_rng(a.seed + 500 + pi))[0]
        sel[f"group_tractogram_pool{len(group32)}"] = take(
            g_full, a.n_model, np.random.default_rng(a.seed + 500 + pi))[0]
        for name, S in sel.items():
            W = sc_pass(S, atlas, affine)
            nnz = check_sc(f"raw/{name}/{sb}", W)
            m = {"subject": sb, "pred_from": sa if name == "cross_subject" else
                 ("group" if name.startswith("group") else sb),
                 "n_streamlines": int(len(S)), "n_edges_pred": nnz,
                 "n_edges_gt": int((np.triu(gt, 1) > 0).sum()), "sum_pred": float(W.sum())}
            m.update(flat_corr(W, gt))
            rows["raw"][name].append(m)

        # ---- 개수 배분만 망친 대조군: 기하는 자기 자신 것 그대로 ----
        pr, ok = endpoint_pair_ids(RB, atlas, affine)
        u_idx, n_uniq = uniform_pair_draw(pr, ok, a.n_model, np.random.default_rng(a.seed + 600 + pi))
        SU = RB[u_idx].astype(np.float32)
        W = sc_pass(SU, atlas, affine)
        nnz = check_sc(f"raw/uniform/{sb}", W)
        gtS, _ = model_gt_sample(RB, a.n_sample)
        d_uni = bundle_geometry_metrics(SU, np.ascontiguousarray(gtS, np.float32), voxel_mm=a.voxel_mm)
        m = {"subject": sb, "pred_from": sb, "n_streamlines": int(len(SU)), "n_edges_pred": nnz,
             "n_endpoint_pairs": int(n_uniq), "dice_vs_self": d_uni["dice"],
             "coverage_vs_self": d_uni["coverage"], "sum_pred": float(W.sum())}
        m.update(flat_corr(W, gt))
        rows["raw"]["uniform_pair_counts"].append(m)

        # ---- 개수 효과 곡선: self(=표본 잡음만) 와 group(=템플릿 한계) ----
        for n in (2000, a.n_sample, a.n_model, a.raw_sample):
            if n > len(RB):
                continue
            rs = np.random.default_rng(a.seed + 700 + pi)
            c = flat_corr(sc_pass(take(RB, n, rs)[0], atlas, affine), gt)
            sweep.append({"src": "self", "subject": sb, "n_pred": int(n), **c})
        for src, P in (("group_pool8", g_pool8), (f"group_pool{len(group32)}", g_full)):
            for n in (a.n_sample, a.n_model, min(len(P), 4 * a.n_model)):
                rs = np.random.default_rng(a.seed + 800 + pi)
                c = flat_corr(sc_pass(take(P, n, rs)[0], atlas, affine), gt)
                sweep.append({"src": src, "subject": sb, "n_pred": int(n), **c})
        del RA, RB
        print(f"[{pi + 1}/{len(pairs_ab)}] {sa} -> {sb}  "
              f"pairs: cross {rows['pairs']['cross_subject'][-1]['sc_r']:.3f} / "
              f"group {rows['pairs']['group_tractogram'][-1]['sc_r']:.3f} / "
              f"self {rows['pairs']['self'][-1]['sc_r']:.3f}   raw: "
              f"cross {rows['raw']['cross_subject'][-1]['sc_r']:.3f} / "
              f"group {rows['raw']['group_tractogram'][-1]['sc_r']:.3f} / "
              f"group8 {rows['raw']['group_tractogram_pool8'][-1]['sc_r']:.3f} / "
              f"self {rows['raw']['self'][-1]['sc_r']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    out = {"pairs_family": {k: sc_summarize(v) for k, v in rows["pairs"].items()},
           "raw_family": {k: sc_summarize(v) for k, v in rows["raw"].items()},
           "reference": ref}

    # --- 필수 검사: self 계열이 가장 높아야 한다 -----------------------------
    for fam in ("pairs_family", "raw_family"):
        d = out[fam]
        best_self = min(d["self"]["sc_r"], d["subsample"]["sc_r"])
        worst_other = max(d["cross_subject"]["sc_r"], d["group_tractogram"]["sc_r"])
        assert best_self > worst_other, \
            f"{fam}: self({best_self:.4f}) 가 cross/group({worst_other:.4f}) 보다 낮다 -- 채점이 틀렸다"

    out["count_sweep"] = [{"src": s_, "n_pred": int(n), **sc_summarize([r for r in sweep
                                                                       if r["src"] == s_ and r["n_pred"] == n])}
                          for s_ in ("self", "group_pool8", f"group_pool{len(group32)}")
                          for n in sorted({r["n_pred"] for r in sweep if r["src"] == s_})]

    # --- dice -> SC r 대응표 (dice 는 기하 실행 JSON 에서 가져온다: 같은 seed/쌍/가닥) ---
    geo_path = ROOT / a.sc_geometry_json
    dice = {}
    if geo_path.exists() and geo_path.stat().st_size > 0:
        gj = json.loads(geo_path.read_text())
        assert gj["meta"]["seed"] == a.seed and gj["meta"]["n_sample"] == a.n_sample, \
            "기하 JSON 의 seed/n_sample 이 다르다 -- dice 와 SC 가 다른 tractogram 이다"
        assert [p["target"] for p in gj["meta"]["pairs"]][:len(pairs_ab)] == [b for _, b in pairs_ab], \
            "기하 JSON 의 subject 쌍이 다르다"
        dice = {"pairs": {k: v.get("dice") for k, v in gj["wb_pairs"].items()},
                "raw": {k: v.get("dice") for k, v in gj.get("wb_raw", {}).items()}}
    table = []
    for fam, key in (("pairs", "pairs_family"), ("raw", "raw_family")):
        for name, d in out[key].items():
            table.append({"family": fam, "baseline": name,
                          "dice": dice.get(fam, {}).get(name),   # uniform 대조군 dice 는 raw_family.*.dice_vs_self
                          "sc_r": d["sc_r"], "sc_r_sd": d["sc_r_sd"], "sc_r_log": d["sc_r_log"],
                          "tier": {t: d[f"tier_{t}"] for t in TIERS},
                          "n_streamlines": d.get("n_streamlines")})
    mdl = {"baseline": "model (final_p4_joint_step3000, T1 단독 generated)", "dice": 0.5457,
           "sc_r": 0.7130, "tier": {"small": 0.164, "mid": 0.189, "large": 0.624},
           "n_streamlines": 28992}
    ref_row = {"baseline": "group mean SC matrix (train 144, tractogram 아님)", "dice": None,
               "sc_r": ref["group_mat_144"]["sc_r"],
               "tier": {t: ref["group_mat_144"][f"tier_{t}"] for t in TIERS}}
    out["table"] = {"model": mdl, "group_sc_matrix": ref_row, "baselines": table}

    gtr = out["pairs_family"]["group_tractogram"]["sc_r"]
    gtr_raw = out["raw_family"]["group_tractogram"]["sc_r"]
    out["verdict"] = {
        "threshold": 0.85,
        "group_tractogram_sc_r_pairs_family": gtr,
        "group_tractogram_sc_r_raw_family": gtr_raw,
        "group_tractogram_sc_r_raw_pool8": out["raw_family"]["group_tractogram_pool8"]["sc_r"],
        f"group_tractogram_sc_r_raw_pool{len(group32)}":
            out["raw_family"][f"group_tractogram_pool{len(group32)}"]["sc_r"],
        "pass": bool(min(gtr, gtr_raw) >= 0.85),
        "group_mat_8_sc_r": ref["group_mat_8"]["sc_r"],
        "group_mat_32_sc_r": ref["group_mat_32"]["sc_r"],
        "group_mat_144_sc_r": ref["group_mat_144"]["sc_r"],
        # 같은 subject pool 에서 tractogram 으로 센 SC 와 평균 SC 행렬의 차이 = '행렬 vs tractogram' 효과
        "gap_tractogram_vs_matrix_pool8":
            float(out["raw_family"]["group_tractogram_pool8"]["sc_r"] - ref["group_mat_8"]["sc_r"]),
        "gap_tractogram_vs_matrix_pool32":
            float(out["raw_family"][f"group_tractogram_pool{len(group32)}"]["sc_r"]
                  - ref["group_mat_32"]["sc_r"]),
        # 템플릿 subject 수 효과 = 0.945 와의 격차 대부분을 설명하는 항
        "gap_8_vs_144_subjects": float(ref["group_mat_144"]["sc_r"] - ref["group_mat_8"]["sc_r"]),
        "gap_32_vs_144_subjects": float(ref["group_mat_144"]["sc_r"] - ref["group_mat_32"]["sc_r"]),
        "self_ceiling_sc_r": out["raw_family"]["subsample"]["sc_r"],
    }
    out["meta"] = {
        "cmd": cmd, "seed": a.seed, "n_sample": a.n_sample, "n_model_streamlines": a.n_model,
        "raw_sample": a.raw_sample, "sc_group_raw": a.sc_group_raw, "voxel_mm": a.voxel_mm,
        "pairs": [{"pred": x, "target": y} for x, y in pairs_ab],
        "group_train_subjects": group_subs, "group_train_subjects_pool2": group32,
        "sc_mode": "pass", "n_roi": N_ROI, "n_ctx": N_CTX,
        "gt": "FC_DKPD25_82_ppmi_all_nomed_qc.mat SC_weight (pass count, 전체 1M streamline)",
        "sc_impl": "src/atm_sc/data/tt_io.py hard_sc(mode='pass') -- 새 구현 없음",
        "corr_impl": "src/atm_sc/evaluation/reproduction_metrics.py stratified_corr (상삼각 off-diagonal)",
        "geometry_json": a.sc_geometry_json,
        "notes": [
            "family 'pairs' 는 bundles.npz(ROI pair 당 cap 256) 에서 뽑은 가닥이다. 기하 실행의 "
            "dice 0.574/0.614/0.783 이 나온 바로 그 집합이지만, cap 때문에 pair 당 개수가 "
            "GT 와 다르다 -- SC 는 개수 지표이므로 이 계열의 SC r 은 cap 에 눌린 값이다.",
            "family 'raw' 는 .tt.gz 원본의 균등 표본이므로 pair 당 개수 분포가 GT 와 같다. "
            "'진짜 tractogram 을 제출하면 SC r 이 몇인가' 의 답은 이쪽이다.",
            "self/subsample 은 대상 subject 자신의 다른 표본이다 = 표본 잡음 + 128점 재샘플링만 "
            "반영한 천장. 이것이 tractogram 경로로 도달 가능한 SC r 의 상한이다.",
            "uniform_pair_counts 는 대상 자신의 가닥을 그대로 쓰되 endpoint pair 마다 같은 개수를 "
            "뽑은 것이다. 기하(dice)는 거의 그대로인데 SC r 만 무너진다면 dice 와 SC r 이 서로 "
            "다른 것을 재고 있다는 직접 증거다.",
            "group_tractogram 은 기하 실행과 같은 pool(bundles 8명 / raw 4명), "
            "group_tractogram_pool8 은 raw 8명 전체 pool 이다.",
        ]}
    out["elapsed_sec"] = time.time() - t0_all

    dst = ROOT / a.sc_out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    f = lambda x: "  -  " if x is None or not np.isfinite(x) else f"{x:.3f}"
    print("\n=== dice -> SC r 대응표 (GT = .mat pass-SC, 상삼각 3321 pair) ===")
    print(f"{'family':<7}{'baseline':<26}{'dice':>7}{'SC r':>8}{'sd':>7}"
          f"{'small':>8}{'mid':>8}{'large':>8}{'n_strm':>9}")
    print(f"{'-':<7}{mdl['baseline'][:25]:<26}{f(mdl['dice']):>7}{f(mdl['sc_r']):>8}{'':>7}"
          f"{f(mdl['tier']['small']):>8}{f(mdl['tier']['mid']):>8}{f(mdl['tier']['large']):>8}"
          f"{mdl['n_streamlines']:>9}")
    for r in table:
        print(f"{r['family']:<7}{r['baseline']:<26}{f(r['dice']):>7}{f(r['sc_r']):>8}{f(r['sc_r_sd']):>7}"
              f"{f(r['tier']['small']):>8}{f(r['tier']['mid']):>8}{f(r['tier']['large']):>8}"
              f"{r['n_streamlines'] or 0:>9.0f}")
    for k, lbl in (("group_mat_144", "group SC matrix (144)"), ("group_mat_32", "group SC matrix (32)"),
                   ("group_mat_8", "group SC matrix (8)")):
        d = ref[k]
        print(f"{'ref':<7}{lbl:<26}{'  -  ':>7}{f(d['sc_r']):>8}{f(d['sc_r_sd']):>7}"
              f"{f(d['tier_small']):>8}{f(d['tier_mid']):>8}{f(d['tier_large']):>8}{'  -  ':>9}")
    print("\n=== 개수 효과 (SC r vs 제출 streamline 수) ===")
    for s_ in out["count_sweep"]:
        print(f"  {s_['src']:<12} n={s_['n_pred']:>6}  r={s_['sc_r']:.4f}  "
              f"tier {s_['tier_small']:.3f}/{s_['tier_mid']:.3f}/{s_['tier_large']:.3f}")
    v = out["verdict"]
    print(f"\n판정: group_tractogram SC r = {gtr:.4f}(pairs) / {gtr_raw:.4f}(raw) "
          f"vs 기준 0.85 -> {'통과' if v['pass'] else '미달'}")
    print(f"  같은 pool 의 그룹 평균 SC '행렬' = {v['group_mat_8_sc_r']:.4f}(8명) / "
          f"{v['group_mat_32_sc_r']:.4f}({len(group32)}명) / {v['group_mat_144_sc_r']:.4f}(144명)")
    print(f"  tractogram - 행렬 (같은 pool) = {v['gap_tractogram_vs_matrix_pool8']:+.4f}(8명) / "
          f"{v['gap_tractogram_vs_matrix_pool32']:+.4f}({len(group32)}명)  <- 거의 0 이면 둘은 같은 것이다")
    print(f"  템플릿 subject 수 효과 = {v['gap_8_vs_144_subjects']:+.4f}(8->144) / "
          f"{v['gap_32_vs_144_subjects']:+.4f}({len(group32)}->144)")
    print(f"  self 천장(표본잡음만) = {v['self_ceiling_sc_r']:.4f}")
    print(f"저장: {dst}  ({out['elapsed_sec']:.0f}s)")
    return 0


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=10, help="subject 쌍 개수 (A=예측, B=정답)")
    ap.add_argument("--n-sample", type=int, default=8000,
                    help="whole-brain 채점 표본 크기. 모델 JSON 의 trk_geometry.n_gen 과 같아야 한다")
    ap.add_argument("--n-model", type=int, default=29000, help="모델이 실제로 생성한 streamline 수")
    ap.add_argument("--n-group", type=int, default=8, help="group tractogram 에 합칠 train subject 수")
    ap.add_argument("--trk-pairs", type=int, default=50, help="pair 단위 비교에 쓸 큰 연결 수")
    ap.add_argument("--n-per-pair", type=int, default=16, help="모델의 pair 당 생성 개수 (구 *_n16 행 전용)")
    ap.add_argument("--pair-n", type=int, default=64,
                    help="크기 일치 pair dice 의 한 쪽 가닥 수. scripts/29 --trk-pair-n 과 같아야 "
                         "모델과 비교 가능하다 (기본 64)")
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--dist-n", type=int, default=32)
    ap.add_argument("--raw-sample", type=int, default=60000, help="raw .tt.gz 계열에서 subject 당 뽑을 수")
    ap.add_argument("--no-raw", action="store_true", help="raw .tt.gz 계열을 건너뛴다")
    ap.add_argument("--seed", type=int, default=20250906)
    ap.add_argument("--limit-subjects", type=int, default=0, help="파일럿용: 쌍을 이 개수로 자른다")
    ap.add_argument("--out", default="outputs/eval/s5e_geometry_baselines.json")
    ap.add_argument("--sc", action="store_true",
                    help="W1-c: 각 기준선 tractogram 에서 SC 를 실제로 계산해 GT(.mat pass-SC) 와 상관을 낸다")
    ap.add_argument("--sc-out", default="outputs/eval/w1c_baseline_sc.json")
    ap.add_argument("--sc-geometry-json", default="outputs/eval/s5e_geometry_baselines.json",
                    help="dice 를 가져올 기하 실행 결과 (같은 seed/쌍이어야 한다)")
    ap.add_argument("--sc-group-raw", type=int, default=10000,
                    help="raw group pool 에서 train subject 당 뽑을 수")
    ap.add_argument("--sc-group-n2", type=int, default=32,
                    help="두 번째(더 큰) raw group pool 의 train subject 수. --n-group 을 포함하는 상위집합")
    a = ap.parse_args()

    t_start = time.time()
    cmd = "python " + " ".join([str(Path(sys.argv[0]).relative_to(ROOT) if Path(sys.argv[0]).is_absolute()
                                    else sys.argv[0])] + sys.argv[1:])
    atlas, affine = load_atlas()
    print(f"atlas {atlas.shape} labels=1..{N_ROI} OK", flush=True)

    test = [l.strip() for l in open(ROOT / "outputs/splits/test.txt") if l.strip()]
    train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(test))
    n_pairs = a.limit_subjects or a.n_pairs
    assert 2 * n_pairs <= len(test), f"test {len(test)}명으로 쌍 {n_pairs}개를 만들 수 없다"
    pairs_ab = [(test[order[2 * i]], test[order[2 * i + 1]]) for i in range(n_pairs)]
    group_subs = [train[i] for i in np.random.default_rng(a.seed + 1).permutation(len(train))[:a.n_group]]
    print(f"쌍 {len(pairs_ab)}개 (A=예측 -> B=정답), group={len(group_subs)}명 train, seed={a.seed}", flush=True)

    if a.sc:
        return run_sc(a, cmd, atlas, affine, pairs_ab, group_subs, train)

    meta = {"cmd": cmd, "seed": a.seed, "n_sample": a.n_sample, "n_model_streamlines": a.n_model,
            "voxel_mm": a.voxel_mm, "trk_pairs": a.trk_pairs, "n_per_pair": a.n_per_pair,
            "pairs": [{"pred": x, "target": y} for x, y in pairs_ab],
            "group_train_subjects": group_subs, "raw_sample": 0 if a.no_raw else a.raw_sample,
            "protocol": ("bundle_geometry_metrics(pred, gt, voxel_mm) 로 29_final_evaluation --trk-eval "
                         "과 동일. gt 는 bundles.npz 에서 default_rng(0) 로 뽑은 n_sample 개 "
                         "(모델 JSON 의 trk_geometry.n_gen==n_gt==8000 과 일치)."),
            "pair_n": a.pair_n,
            "protocol_ref": "outputs/eval/w3b_protocol.json",
            "model_reference": {"source": "규약이 같은 모델 실행에서만 가져올 것. 옛 하드코딩 값"
                                          "(trk_dice 0.5457 / pair_dice 0.0946) 은 규약 불일치라 삭제했다: "
                                          "wb 는 n_gen!=n_gt (모델 14.5k vs GT 20k, 기준선 8k vs 8k), "
                                          "pair 는 16 vs <=256 (천장은 128 vs 128) 이었다.",
                                "matched_runs": "outputs/eval/w3b_*.json (scripts/29 --trk-gen-sample "
                                                "== --trk-gt-sample == n_sample, --trk-pair-n == --pair-n)"},
            "notes": [
                "지표 절대값은 표본 개수에 강하게 의존한다 (count_sweep 참조). 모델 평가가 pred/gt 를 "
                "둘 다 8,000 으로 잘랐으므로 모든 기준선도 8,000 으로 맞췄다. subsample 만 pred=29,000.",
                "family 'pairs' 의 valid_conn / endpoint_in_roi 는 1.0 이 **구조적**이다: bundles.npz 는 "
                "양 끝점이 서로 다른 ROI 에 닿은 streamline 만 담고 있고 pair 라벨도 같은 아틀라스로 "
                "매겼기 때문이다. 즉 '진짜 tractogram 을 제출하면 이 지표는 자동으로 1.0' 이라는 뜻이고, "
                "이 지표로는 기준선끼리 구분되지 않는다.",
                "family 'raw' 의 endpoint_in_roi(~0.59) 가 진짜 whole-brain tractogram 의 실제 값이다. "
                "raw 의 valid_conn 은 pair 를 자기 끝점에서 유도했으므로 사실상 '양 끝이 서로 다른 ROI' "
                "비율과 같다.",
                "per_pair.self 는 모델 평가의 trk_per_pair_ceiling 과 같은 규약(번들 절반 vs 절반)이다. "
                "실측 0.704 vs 모델 JSON 0.6997 로 일치 -> 채점 규약이 같음을 확인한 것이다.",
                "per_pair 의 *_n16 은 옛 규약(pred=16, gt=<=256)이다. 크기가 안 맞아 해석이 어렵다 -- "
                "남겨둔 것은 과거 수치와의 연속성 때문이다.",
                "**보고에 쓸 pair dice 는 *_matched 행이다**: pred(n)=gtA(n)=gtB(n), n=--pair-n. "
                "ceiling_matched 가 같은 규약의 천장이고 scripts/29 --trk-pair-n 과 직접 비교된다."]}

    # --- group tractogram 풀 (train GT 합본) ---------------------------------
    print("group tractogram 풀 구축...", flush=True)
    need_pairs = set()
    tops = {}
    for _, b in pairs_ab:
        z = np.load(BUNDLE_DIR / b / "bundles.npz")
        ids, cnt = z["pair_ids"].astype(np.int64), z["pair_count_full"].astype(np.int64)
        top = ids[np.argsort(-cnt)[: a.trk_pairs]]
        tops[b] = top
        need_pairs |= {(int(i), int(j)) for i, j in top}
    per = int(np.ceil(a.n_sample * 3 / len(group_subs)))
    pool, pool_pair, bank = [], [], {p: [] for p in need_pairs}
    for gs in group_subs:
        S, pr, ids, off, _ = load_bundles(gs)
        g_rng = np.random.default_rng(a.seed + 7)
        s, sel = take(S, min(per, len(S)), g_rng)
        pool.append(s.astype(np.float16)); pool_pair.append(pr[sel])
        key = {(int(i), int(j)): k for k, (i, j) in enumerate(ids)}
        for p in need_pairs:
            k = key.get(p)
            if k is not None:
                bank[p].append(S[off[k]:off[k + 1]])
        del S
    pool = np.concatenate(pool); pool_pair = np.concatenate(pool_pair)
    bank = {p: (np.concatenate(v) if v else None) for p, v in bank.items()}
    bb_pool = check_streamlines("group_pool", _stride(pool))
    print(f"  풀 {len(pool):,} streamline, pair bank {sum(v is not None for v in bank.values())}/{len(bank)}",
          flush=True)

    res = {"wb_pairs": {}, "wb_raw": {}, "per_pair": {}}
    rows = {k: [] for k in ("cross_subject", "self", "subsample", "group_tractogram")}
    rows_raw = {k: [] for k in rows}
    rows_pair = {k: [] for k in ("cross_subject", "cross_subject_n16", "self", "subsample_n16",
                                "group_tractogram_n16",
                                "ceiling_matched", "cross_subject_matched", "subsample_matched",
                                "group_tractogram_matched")}
    matched_skip = {}
    sweep = []
    raw_cache_bbox = {}

    for pi, (sa, sb) in enumerate(pairs_ab):
        t0 = time.time()
        SA, PA, idsA, offA, _ = load_bundles(sa)
        SB, PB, idsB, offB, cntB = load_bundles(sb)
        bba = check_streamlines(f"{sa}/bundles", _stride(SA))
        bbb = check_streamlines(f"{sb}/bundles", _stride(SB))
        iou = assert_bbox_overlap(sa, bba, sb, bbb)

        gt, gt_idx = model_gt_sample(SB, a.n_sample)
        r_ab = np.random.default_rng(a.seed + 100 + pi)
        preds = {
            "cross_subject": take(SA, a.n_sample, r_ab)[0:2],
            "self": take(SB, a.n_sample, r_ab, exclude=gt_idx)[0:2],
            "subsample": take(SB, min(a.n_model, len(SB) - len(gt_idx)), r_ab, exclude=gt_idx)[0:2],
            "group_tractogram": take(pool, a.n_sample, r_ab)[0:2],
        }
        src_pair = {"cross_subject": PA, "self": PB, "subsample": PB, "group_tractogram": pool_pair}
        gtW, gtL = hard_sc(gt.reshape(-1, 3), np.full(len(gt), 128, np.int64), atlas, affine, N_ROI, "pass")

        for name, (S, sel) in preds.items():
            m = {"subject": sb, "pred_from": sa if name == "cross_subject" else
                 ("group" if name == "group_tractogram" else sb), "bbox_iou": iou}
            m.update(score_wb(S, gt, a.voxel_mm, a.dist_n, seed=pi))
            m.update(score_conn(S, src_pair[name][sel], atlas, affine))
            m.update(edge_length_corr(S, gtL, gtW, atlas, affine))
            rows[name].append(m)

        # 개수 효과 곡선 (self 를 pred 개수만 바꿔가며)
        for n in (1000, 4000, a.n_sample, 16000, a.n_model):
            if n > len(SB) - len(gt_idx):
                continue
            S, _ = take(SB, n, np.random.default_rng(a.seed + 200 + pi))
            g = bundle_geometry_metrics(S, gt, voxel_mm=a.voxel_mm)
            sweep.append({"subject": sb, "n_pred": n, "n_gt": len(gt),
                          "dice": g["dice"], "coverage": g["coverage"], "overreach": g["overreach"]})

        # --- pair 단위 -------------------------------------------------------
        keyA = {(int(i), int(j)): k for k, (i, j) in enumerate(idsA)}
        n_have = 0
        for (i, j) in tops[sb]:
            kb = int(np.flatnonzero((idsB[:, 0] == i) & (idsB[:, 1] == j))[0])
            gtb = SB[offB[kb]:offB[kb + 1]].astype(np.float32)
            if len(gtb) < 8:
                continue
            h = len(gtb) // 2
            pr_rng = np.random.default_rng(a.seed + 300 + pi)
            d = pair_dice(gtb[:h], gtb[h:], a.voxel_mm)
            if d:
                rows_pair["self"].append(d)
            n16 = min(a.n_per_pair, len(gtb))
            d = pair_dice(gtb[pr_rng.choice(len(gtb), n16, replace=False)], gtb, a.voxel_mm)
            if d:
                rows_pair["subsample_n16"].append(d)
            ka = keyA.get((int(i), int(j)))
            if ka is not None:
                ab = SA[offA[ka]:offA[ka + 1]].astype(np.float32)
                n_have += 1
                d = pair_dice(ab, gtb, a.voxel_mm)
                if d:
                    rows_pair["cross_subject"].append(d)
                d = pair_dice(ab[pr_rng.choice(len(ab), min(a.n_per_pair, len(ab)), replace=False)],
                              gtb, a.voxel_mm)
                if d:
                    rows_pair["cross_subject_n16"].append(d)
            gb = bank.get((int(i), int(j)))
            if gb is not None and len(gb) >= 2:
                d = pair_dice(gb[pr_rng.choice(len(gb), min(a.n_per_pair, len(gb)), replace=False)]
                              .astype(np.float32), gtb, a.voxel_mm)
                if d:
                    rows_pair["group_tractogram_n16"].append(d)
            # --- 크기 일치 규약 (C9). scripts/29 --trk-pair-n 과 같은 저울: pred(n) vs gtA(n),
            # 천장 = gtB(n) vs gtA(n). gtA/gtB 는 GT 를 섞어 나눈 겹치지 않는 두 조각이다.
            # 크기가 다르면 dice 가 흐릿한 생성기에 상을 주고 (n<=32 에서 변위에 비단조),
            # pred/천장 비율도 뜻을 잃는다.
            n = min(a.pair_n, len(gtb) // 2)
            if n >= 2:
                m_rng = np.random.default_rng(a.seed + 500 + pi + 1000 * int(i) + int(j))
                perm = m_rng.permutation(len(gtb))
                gtA, gtB = gtb[perm[:n]], gtb[perm[n:2 * n]]
                assert len(gtA) == len(gtB) == n, (len(gtA), len(gtB), n)
                d = pair_dice(gtB, gtA, a.voxel_mm)
                if d:
                    rows_pair["ceiling_matched"].append(d)
                srcs = {"subsample_matched": gtb,
                        "cross_subject_matched": (SA[offA[ka]:offA[ka + 1]].astype(np.float32)
                                                  if ka is not None else None),
                        "group_tractogram_matched": (np.asarray(gb, np.float32)
                                                     if gb is not None else None)}
                for nm, src in srcs.items():           # `pool`(group 풀) 을 가리지 않도록 이름 분리
                    if src is None or len(src) < n:
                        matched_skip[nm] = matched_skip.get(nm, 0) + 1
                        continue
                    if nm == "subsample_matched":       # 자기 자신에서 뽑되 gtA/gtB 와 겹치지 않게
                        idx = perm[2 * n:]
                        if len(idx) < n:
                            matched_skip[nm] = matched_skip.get(nm, 0) + 1
                            continue
                        pred = gtb[m_rng.choice(idx, n, replace=False)]
                    else:
                        pred = src[m_rng.choice(len(src), n, replace=False)]
                    assert len(pred) == len(gtA) == n, (len(pred), len(gtA), n)
                    d = pair_dice(pred, gtA, a.voxel_mm)
                    if d:
                        rows_pair[nm].append(d)
        res.setdefault("pair_availability", []).append(
            {"target": sb, "pred": sa, "n_top": len(tops[sb]), "n_in_pred": n_have,
             "frac": n_have / max(len(tops[sb]), 1),
             "frac_all_pairs": float(np.mean([(int(i), int(j)) in keyA for i, j in idsB]))})
        del SA, SB
        print(f"[{pi + 1}/{len(pairs_ab)}] {sa} -> {sb}  bboxIoU={iou:.3f}  "
              f"cross dice={rows['cross_subject'][-1]['dice']:.3f}  "
              f"self dice={rows['self'][-1]['dice']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    res["wb_pairs"] = {k: summarize(v) for k, v in rows.items() if v}
    res["wb_pairs_rows"] = rows
    res["per_pair"] = {k: summarize(v) for k, v in rows_pair.items() if v}
    res["per_pair_matched_skipped"] = matched_skip
    res["count_sweep"] = [{"n_pred": int(n), **summarize([r for r in sweep if r["n_pred"] == n])}
                          for n in sorted({r["n_pred"] for r in sweep})]

    # --- raw .tt.gz 계열 -----------------------------------------------------
    if not a.no_raw:
        print("raw .tt.gz 계열...", flush=True)
        need = sorted({s for p in pairs_ab for s in p})
        raw = {}
        for s in need:
            t0 = time.time()
            R, seen = load_raw_sample(s, a.raw_sample, a.seed)
            raw_cache_bbox[s] = check_streamlines(f"{s}/raw", _stride(R))
            raw[s] = R
            print(f"  {s}: {seen:,} streamline 중 {len(R):,} 추출 ({time.time() - t0:.0f}s)", flush=True)
        g_pool = []
        for gs in group_subs[: max(2, a.n_group // 2)]:
            R, _ = load_raw_sample(gs, max(4000, a.n_sample), a.seed)
            g_pool.append(R)
        g_pool = np.concatenate(g_pool)
        print(f"  raw group pool {len(g_pool):,}", flush=True)

        for pi, (sa, sb) in enumerate(pairs_ab):
            RA, RB = raw[sa], raw[sb]
            assert_bbox_overlap(sa, raw_cache_bbox[sa], sb, raw_cache_bbox[sb])
            gt, gt_idx = model_gt_sample(RB, a.n_sample)
            r_ab = np.random.default_rng(a.seed + 400 + pi)
            preds = {"cross_subject": take(RA, a.n_sample, r_ab)[0],
                     "self": take(RB, a.n_sample, r_ab, exclude=gt_idx)[0],
                     "subsample": take(RB, min(a.n_model, len(RB) - len(gt_idx)), r_ab,
                                       exclude=gt_idx)[0],
                     "group_tractogram": take(g_pool, a.n_sample, r_ab)[0]}
            for name, S in preds.items():
                m = {"subject": sb}
                m.update(score_wb(S, gt, a.voxel_mm, a.dist_n, seed=pi))
                lab = point_labels(S[:, [0, -1]].reshape(-1, 3).astype(np.float32), atlas, affine)
                pr = np.stack([lab[0::2].astype(np.int64) - 1, lab[1::2].astype(np.int64) - 1], 1)
                pr = np.sort(pr, 1)
                pr[pr[:, 0] == pr[:, 1]] = -1
                m.update(score_conn(S, pr, atlas, affine))
                rows_raw[name].append(m)
            print(f"  [{pi + 1}/{len(pairs_ab)}] {sa} -> {sb} raw cross dice="
                  f"{rows_raw['cross_subject'][-1]['dice']:.3f}", flush=True)
        res["wb_raw"] = {k: summarize(v) for k, v in rows_raw.items() if v}
        res["wb_raw_rows"] = rows_raw

    res["meta"] = meta
    res["elapsed_sec"] = time.time() - t_start
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n저장: {out}  ({res['elapsed_sec']:.0f}s)")

    fmt = lambda d, k: f"{d[k]:.3f}" if k in d else "  -  "
    print("\n=== whole-brain (bundles.npz, 모델 평가와 동일 규약) ===")
    print(f"{'baseline':<20}{'dice':>8}{'cover':>8}{'overr':>8}{'validc':>8}{'ep_roi':>8}{'len_mm':>9}{'len_r':>8}")
    for k, d in res["wb_pairs"].items():
        print(f"{k:<20}{fmt(d,'dice'):>8}{fmt(d,'coverage'):>8}{fmt(d,'overreach'):>8}"
              f"{fmt(d,'valid_conn'):>8}{fmt(d,'endpoint_in_roi'):>8}"
              f"{d.get('len_mean_mm',float('nan')):>9.1f}{fmt(d,'len_edge_r'):>8}")
    if res["wb_raw"]:
        print("\n=== whole-brain (raw .tt.gz 1M 에서 추출) ===")
        for k, d in res["wb_raw"].items():
            print(f"{k:<20}{fmt(d,'dice'):>8}{fmt(d,'coverage'):>8}{fmt(d,'overreach'):>8}"
                  f"{fmt(d,'valid_conn'):>8}{fmt(d,'endpoint_in_roi'):>8}"
                  f"{d.get('len_mean_mm',float('nan')):>9.1f}")
    print("\n=== pair 단위 dice ===")
    for k, d in res["per_pair"].items():
        print(f"{k:<24}dice={d['dice']:.3f}  cov={d['coverage']:.3f}  "
              f"n_pred={d['n_pred']:.0f} n_gt={d['n_gt']:.0f}  obs={d['n_obs']}")
    print("\n=== 개수 효과 (self, gt 고정) ===")
    for s in res["count_sweep"]:
        print(f"  n_pred={s['n_pred']:>6}  dice={s['dice']:.3f}  cov={s['coverage']:.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
