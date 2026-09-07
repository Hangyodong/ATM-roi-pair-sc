#!/usr/bin/env python
"""W2-c: 점유(occupancy) 진단 -- 모델이 "어느 ROI 쌍에" 가닥을 놓는가.

문제 (C10): 기준선 8점 회귀 sc_r = 0.622 + 0.445 x dice 는 모델 dice 0.546 에서 0.865 를
예측하는데 실측 pass-SC r 은 0.713 이다 (잔차 -0.15). 남의 진짜 tractogram(0.841~0.891)
보다도 낮다 -> 기하 품질만으로 설명되지 않는 결함이 따로 있다.

W1-c 의 결정적 대조군: 대상 **본인 가닥**을 쓰되 endpoint pair 당 개수를 균일화한
(일부러 망친) tractogram 도 SC r 0.945 다. 즉 pass-SC r 은 개수 배분이 아니라
**"어느 쌍을 지나갔나"(점유)** 에 지배된다.

여기서 재는 것
  1. 점유 혼동행렬  GT 양성 쌍 중 모델이 실제로 가닥을 통과시킨 쌍(TP)/놓친 쌍(FN)/
                    GT 음성인데 통과시킨 쌍(FP). tier x block 분해.
  2. FN 의 성격      GT 강도/평균 길이/block 과의 관계.
  3. 원인 분리       FN 이 (a) edge head 미선택인가 (b) 선택은 했는데 가닥이 두 ROI 에
                    실제로 닿지 않아서인가. 처방이 완전히 다르다.
  4. 2x2 상한        {기하: 모델/GT} x {쌍 집합: 모델/GT} 네 칸을 모두 같은 규약
                    (쌍 당 16 가닥, pass-SC) 으로 채워 -0.15 잔차의 점유 몫과 기하 몫을 분리.
  5. edge_thr 스윕   임계값을 낮추면 점유와 SC r 이 얼마나 오르는가 (재학습 없는 이득).

규약은 scripts/29_final_evaluation.py 의 generated 경로와 같다:
  pass-SC = tractogram_sc(...)["pass"]["sc"] (가중 없음), GT = .mat pass-SC (subj.sc_mat),
  n_per_pair = 16, seed = 0. 지표는 stratified_corr 로 tier/block 을 함께 낸다.

GPU 는 stage='gen' 에서만 쓰고 (subject 당 ~2초), 결과를 npz 로 캐시해 분석은 CPU 로 한다.

  python scripts/47_occupancy_diag.py --ckpt outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt
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

from atm_sc.data.paths import ATLAS, CACHE                                  # noqa: E402
from atm_sc.data.roi_groups import BLOCKS, N_CTX, TIERS, block_masks, tier_masks   # noqa: E402
from atm_sc.data.tt_io import point_labels, roi_visit_sets                  # noqa: E402
from atm_sc.evaluation.reproduction_metrics import stratified_corr          # noqa: E402

N_ROI = 82
N_PER_PAIR = 16
CACHE_DIR = ROOT / "outputs" / "eval" / "w2c_cache"
OUT_JSON = ROOT / "outputs" / "eval" / "w2c_occupancy.json"
LOCK = ROOT / "outputs" / "gpu.lock"
# 상삼각 pair 를 linear index 로 (i<j): lin = i*N_ROI + j
IU = np.triu_indices(N_ROI, 1)
IU_LIN = IU[0] * N_ROI + IU[1]
assert len(IU_LIN) == 3321, len(IU_LIN)


# --------------------------------------------------------------------- 공통 유틸
def check_sc(name: str, W: np.ndarray):
    """SC 행렬 건전성. 조용히 틀리는 것을 막는 게 목적이다."""
    W = np.asarray(W)
    assert W.shape == (N_ROI, N_ROI), f"{name}: shape {W.shape}"
    assert np.isfinite(W).all(), f"{name}: NaN/Inf {int((~np.isfinite(W)).sum())} 개"
    assert np.array_equal(W, W.T), f"{name}: 대칭이 아니다"
    assert np.abs(np.diag(W)).max() == 0, f"{name}: 대각이 0 이 아니다"
    return W


def visit_structure(S: np.ndarray, atlas, affine, n_roi: int = N_ROI):
    """streamline 별 통과 ROI 쌍을 한 번만 계산한다.

    -> (tid [M] int32, lin [M] int64)  lin = i*n_roi + j (i<j).
    hard_sc(mode='pass') 와 같은 정의이되 track 루프를 벡터화한 것이다 (아래에서 동치 검증).
    """
    S = np.ascontiguousarray(S, np.float32)
    assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
    assert len(S) > 0, "streamline 이 0 개"
    P = S.shape[1]
    npts = np.full(len(S), P, np.int64)
    lab = point_labels(S.reshape(-1, 3), atlas, affine)
    pk = roi_visit_sets(lab, npts)                       # (tid, roi0) 정렬
    if len(pk) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    cnt = np.bincount(pk[:, 0], minlength=len(S))
    off = np.concatenate([[0], np.cumsum(cnt)])
    pos = np.arange(len(pk)) - off[pk[:, 0]]
    kmax = int(cnt.max())
    dense = np.full((len(S), kmax), -1, np.int64)
    dense[pk[:, 0], pos] = pk[:, 1]
    tids, lins = [], []
    for k in np.unique(cnt[cnt >= 2]):
        tr = np.flatnonzero(cnt == k)
        ii, jj = np.triu_indices(int(k), 1)
        a = dense[np.ix_(tr, np.arange(int(k)))]
        u, v = a[:, ii], a[:, jj]                        # pk 가 정렬돼 있어 u < v
        assert (u < v).all(), "ROI 방문 집합이 정렬되어 있지 않다"
        tids.append(np.repeat(tr, len(ii)).astype(np.int32))
        lins.append((u * n_roi + v).ravel())
    if not tids:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    return np.concatenate(tids), np.concatenate(lins)


def sc_from_visits(tid: np.ndarray, lin: np.ndarray, keep: np.ndarray | None, n_roi: int = N_ROI):
    """streamline 부분집합 -> pass-SC [R,R] (대칭)."""
    if keep is None:
        sel = slice(None)
    else:
        sel = keep[tid]
    L = lin[sel]
    W = np.bincount(L, minlength=n_roi * n_roi).reshape(n_roi, n_roi).astype(np.float64)
    W = W + W.T
    return check_sc("pass-SC", W)


def gt_positive(sc_gt: np.ndarray) -> np.ndarray:
    """GT 양성 쌍 (상삼각 bool [3321])."""
    g = np.asarray(sc_gt, np.float64)
    m = g[IU] > 0
    n = int(m.sum())
    assert 2000 < n < 3321, f"GT 양성 쌍 {n} 개 -- 알려진 값(약 2,900)과 다르다"
    return m


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sc_scores(W: np.ndarray, gt: np.ndarray) -> dict:
    """29_final_evaluation 과 같은 stratified_corr + 점유 요약."""
    st = stratified_corr(W, gt, n_roi=N_ROI)
    return {"r": st["all"], "r_log": st["all_log"],
            "tier": st["tier"], "block": st["block"],
            "n_edges_pred": int((W[IU] > 0).sum()), "n_edges_gt": int((gt[IU] > 0).sum())}


# --------------------------------------------------------------------- stage 1: 생성
def generate_all(a):
    """test subject 마다 상삼각 3,321 쌍 전부 x 16 가닥을 생성해 방문 구조를 캐시한다.

    임계값 스윕도 2x2 상한도 이 한 번의 생성에서 부분집합으로 얻는다 (GPU 를 반복해 잡지 않는다).
    """
    import nibabel as nib
    import torch
    from atm_sc.inference.generate_sc import generate_tractogram
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.training.run import anatomy_feature, t1_input

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    t1_src = sd.get("t1_source", "syn")
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    have = set(np.unique(atlas).tolist()) - {0}
    assert not (set(range(1, N_ROI + 1)) - have), "아틀라스에 빠진 라벨이 있다"
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (CACHE / f"{s}_T1w_syn_W.npy").exists()]
    if a.limit:
        subs = subs[: a.limit]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = np.stack(IU, 1).astype(np.int64)                 # [3321,2]
    for i, sub in enumerate(subs):
        p = CACHE_DIR / f"{sub}.npz"
        if p.exists() and not a.overwrite:
            print(f"[{i+1}/{len(subs)}] {sub}: 캐시 재사용", flush=True)
            continue
        t0 = time.time()
        with torch.no_grad():
            feat = (anatomy_feature(m, sub, a.init_bundle, t1_src) if m.unet_level == "none"
                    else m.atm.encode_anatomy(t1_input(m, sub, t1_src)))
            P = torch.as_tensor(all_pairs, device=m.device)
            prob = torch.cat([torch.sigmoid(m.edge_logits(feat, P[k:k + 8192]))
                              for k in range(0, len(P), 8192)]).cpu().numpy()
            S, w, pr = generate_tractogram(m, feat, all_pairs, N_PER_PAIR, seed=0)
        assert len(S) == len(all_pairs) * N_PER_PAIR, (len(S), len(all_pairs))
        assert np.isfinite(S).all(), f"{sub}: 생성 좌표에 NaN/Inf"
        assert np.abs(S).max() > 1.0 and np.abs(S).max() < 200.0, f"{sub}: 좌표 범위 이상 {np.abs(S).max()}"
        # streamline -> 상삼각 pair index. canonical_pairs 가 순서를 바꿀 수 있어 pr 로부터 되짚는다.
        lo, hi = pr.min(1), pr.max(1)
        s_lin = lo * N_ROI + hi
        order = np.argsort(IU_LIN)
        s_pidx = order[np.searchsorted(IU_LIN[order], s_lin)]
        assert (IU_LIN[s_pidx] == s_lin).all(), f"{sub}: 생성 pair 를 상삼각에 매핑하지 못했다"
        tid, lin = visit_structure(S, atlas, img.affine)
        assert len(tid) > 0, f"{sub}: 생성 가닥이 어떤 ROI 쌍도 통과하지 않았다"
        np.savez_compressed(p, prob=prob.astype(np.float32), s_pidx=s_pidx.astype(np.int32),
                            tid=tid, lin=lin, n_stream=len(S))
        assert p.stat().st_size > 0, f"{p} 가 0 바이트"
        print(f"[{i+1}/{len(subs)}] {sub}: 3321 쌍 x{N_PER_PAIR} = {len(S):,} 가닥, "
              f"방문쌍 {len(tid):,} · {time.time()-t0:.1f}s", flush=True)
    return subs


def verify_vectorized_sc(sub: str):
    """visit_structure + sc_from_visits 가 hard_sc(mode='pass') 와 같은 값을 내는지 확인."""
    import nibabel as nib
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.data.tt_io import hard_sc
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    z = dict(np.load(ROIPairSubject(sub).dir / "bundles.npz"))
    S = np.ascontiguousarray(z["streamlines"][:3000], np.float32)
    tid, lin = visit_structure(S, atlas, img.affine)
    W1 = sc_from_visits(tid, lin, None)
    W2, _ = hard_sc(S.reshape(-1, 3), np.full(len(S), 128, np.int64), atlas, img.affine, N_ROI, "pass")
    assert np.array_equal(W1, W2.astype(np.float64)), "벡터화 pass-SC 가 hard_sc 와 다르다"
    return True


# --------------------------------------------------------------------- stage 2: 분석
def gt_geometry_sc(subj, pair_lin_keep: set, atlas, affine, seed: int = 0):
    """GT 가닥(bundles.npz)에서 지정 쌍마다 16 개만 뽑아 pass-SC 를 만든다.

    모델과 같은 규약(쌍 당 16 개)이라 2x2 의 다른 칸과 직접 비교된다. GT 가닥이 없는 쌍
    (모델이 잘못 고른 쌍)은 비워 둔다 -- 존재하지 않는 연결에 진짜 가닥을 놓을 수는 없다.
    """
    z = subj._bnd
    Sall, off, pid = z["streamlines"], z["pair_offsets"], np.asarray(z["pair_ids"], np.int64)
    lin = np.minimum(pid[:, 0], pid[:, 1]) * N_ROI + np.maximum(pid[:, 0], pid[:, 1])
    rng = np.random.default_rng(seed)
    idx = []
    n_hit = 0
    for k in range(len(pid)):
        if int(lin[k]) not in pair_lin_keep:
            continue
        n_hit += 1
        a, b = int(off[k]), int(off[k + 1])
        n = b - a
        take = rng.choice(n, N_PER_PAIR, replace=n < N_PER_PAIR) + a
        idx.append(take)
    assert idx, "GT 가닥으로 채울 쌍이 하나도 없다"
    idx = np.sort(np.concatenate(idx))
    S = np.ascontiguousarray(Sall[idx], np.float32)
    tid, ln = visit_structure(S, atlas, affine)
    nvis = np.bincount(tid, minlength=len(S))
    route = (float(nvis.mean()), float(((1.0 + np.sqrt(1.0 + 8.0 * nvis)) / 2.0).mean()))
    return sc_from_visits(tid, ln, None), len(S), n_hit, route


def analyse(subs, a):
    import nibabel as nib
    from atm_sc.data.dataset import ROIPairSubject
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    affine = img.affine
    bm = {b: v[IU] for b, v in block_masks(N_ROI, N_CTX).items()}
    assert [int(bm[b].sum()) for b in BLOCKS] == [2145, 1056, 120], \
        [int(bm[b].sum()) for b in BLOCKS]

    rows = []
    for i, sub in enumerate(subs):
        p = CACHE_DIR / f"{sub}.npz"
        assert p.exists(), f"{p} 없음 -- --stage gen 을 먼저"
        z = np.load(p)
        prob, s_pidx, tid, lin = z["prob"], z["s_pidx"].astype(np.int64), z["tid"].astype(np.int64), z["lin"]
        subj = ROIPairSubject(sub)
        gt = np.asarray(subj.sc_mat, np.float64)
        check_sc(f"{sub} GT", gt)
        gt_u = gt[IU]
        pos = gt_positive(gt)
        tm = {t: v[IU] for t, v in tier_masks(gt).items()}
        assert sum(int(v.sum()) for v in tm.values()) == int(pos.sum()), "tier 가 GT 양성을 분할하지 않는다"
        gt_len_u = np.asarray(subj.len_mat, np.float64)[IU]

        # --- 모델 운영점 (edge_thr = 0.5) -----------------------------------
        sel = prob > a.edge_thr                                  # [3321] edge head 의 "의도"
        assert sel.any(), f"{sub}: thr={a.edge_thr} 에서 고른 쌍이 없다"
        keep = sel[s_pidx]                                       # streamline 단위
        W = sc_from_visits(tid, lin, keep)
        occ = W[IU] > 0
        assert occ.sum() > 0, f"{sub}: 점유 집합이 비었다"

        TP = pos & occ; FN = pos & ~occ; FP = ~pos & occ
        assert int(TP.sum()) + int(FN.sum()) == int(pos.sum()), "TP+FN != GT 양성"

        # --- 원인 분리: FN 이 (a) 미선택인가 (b) 선택했으나 미도달인가 ------
        fn_a = FN & ~sel                                          # edge head 가 애초에 안 골랐다
        fn_b = FN & sel                                           # 골랐는데 16 가닥이 두 ROI 를 못 지났다
        assert int(fn_a.sum()) + int(fn_b.sum()) == int(FN.sum())
        # (a) 를 더 쪼갠다: edge head 의 학습 타깃은 endpoint pair (bundles.npz pair_ids) 다.
        # pass-양성이지만 endpoint-양성이 아닌 쌍은 애초에 타깃에 없었다.
        endpos = np.zeros(3321, bool)
        pid = np.asarray(subj.pair_ids, np.int64)
        plin = np.minimum(pid[:, 0], pid[:, 1]) * N_ROI + np.maximum(pid[:, 0], pid[:, 1])
        o = np.argsort(IU_LIN)
        endpos[o[np.searchsorted(IU_LIN[o], plin)]] = True
        fn_a_intarget = fn_a & endpos          # 타깃에 있었는데 edge head 가 놓쳤다 (진짜 미선택)
        fn_a_offtarget = fn_a & ~endpos        # 타깃에 없었다 (edge head 정의 문제)

        def split(mask):
            return {"n": int(mask.sum()),
                    "tier": {t: int((mask & tm[t]).sum()) for t in TIERS},
                    "block": {b: int((mask & bm[b]).sum()) for b in BLOCKS},
                    "gt_sum_frac": float(gt_u[mask].sum() / max(gt_u[pos].sum(), 1.0)),
                    "gt_mean": float(gt_u[mask].mean()) if mask.any() else 0.0,
                    "gt_median": float(np.median(gt_u[mask])) if mask.any() else 0.0,
                    "len_mean_mm": float(gt_len_u[mask & (gt_len_u > 0)].mean())
                                   if (mask & (gt_len_u > 0)).any() else 0.0}

        rec = {"subject": sub, "n_selected": int(sel.sum()), "n_streamlines": int(keep.sum()),
               "n_gt_pos": int(pos.sum()), "n_occ": int(occ.sum()),
               "confusion": {"TP": split(TP), "FN": split(FN), "FP": split(FP)},
               "cause": {"a_not_selected": split(fn_a),
                         "a_not_selected_in_edge_target": split(fn_a_intarget),
                         "a_not_selected_off_edge_target": split(fn_a_offtarget),
                         "b_selected_not_reached": split(fn_b)},
               "recall": float(TP.sum() / pos.sum()),
               "precision": float(TP.sum() / max(occ.sum(), 1)),
               "recall_tier": {t: float((TP & tm[t]).sum() / max(tm[t].sum(), 1)) for t in TIERS},
               "baseline": sc_scores(W, gt)}

        # --- 점유(support) 몫 vs 크기(magnitude) 몫: 마스크만으로 재는 값 (생성 불필요) ---
        # gt_on_model_support = GT 값을 모델 점유 집합으로만 가린 것. 1 에 가까우면
        # "점유는 사실상 완벽하고 남은 손실은 전부 값(=기하)" 이라는 뜻이다.
        gm = gt_u * occ
        mo = W[IU] * pos
        rec["support"] = {
            "r_gt_on_model_support": corr(gm, gt_u),          # 점유 결함만의 비용
            "r_model_on_gt_support": corr(mo, gt_u),          # FP 를 지운 모델 (FP 의 비용)
            "r_binary_model_support": corr(occ.astype(float), gt_u),
            "r_binary_gt_support": corr(pos.astype(float), gt_u),
            "gt_mass_missed_frac": float(gt_u[FN].sum() / gt_u[pos].sum()),   # 놓친 쌍이 가진 GT 총량 비율
            "pred_mass_on_fp_frac": float(W[IU][FP].sum() / max(W[IU].sum(), 1.0)),
        }

        # --- 가닥 하나가 몇 개의 ROI 를 지나는가 (기하 결함의 기전) ---------
        nvis_m = np.bincount(tid[keep[tid]], minlength=len(s_pidx))[keep]
        n_roi_m = (1.0 + np.sqrt(1.0 + 8.0 * nvis_m)) / 2.0
        rec["route"] = {"model_pairs_per_streamline": float(nvis_m.mean()),
                        "model_rois_per_streamline": float(n_roi_m.mean())}

        # FN 과 GT 속성의 관계: 놓친 쌍이 약한 연결에 몰려 있는가?
        rec["fn_correlates"] = {
            "pointbiserial_r_gt_strength": corr(FN[pos].astype(float), np.log1p(gt_u[pos])),
            "pointbiserial_r_len": corr(FN[pos].astype(float), gt_len_u[pos]),
            "gt_strength_median_TP": float(np.median(gt_u[TP])) if TP.any() else 0.0,
            "gt_strength_median_FN": float(np.median(gt_u[FN])) if FN.any() else 0.0,
        }

        # --- 2x2 상한: {기하 모델/GT} x {쌍 모델/GT}, 전부 쌍 당 16 가닥 -----
        z_b = subj._bnd
        sel_lin = set(IU_LIN[sel].tolist())
        gtpair_lin = set(plin.tolist())                    # GT 쌍 = bundles.npz(=endpoint 양성)
        keep_gtpairs = np.zeros(3321, bool)
        keep_gtpairs[o[np.searchsorted(IU_LIN[o], plin)]] = True
        W_m_gtpairs = sc_from_visits(tid, lin, keep_gtpairs[s_pidx])
        W_m_allpairs = sc_from_visits(tid, lin, None)      # 3321 쌍 전부 (점유 상한)
        W_g_mpairs, n1, h1, rt1 = gt_geometry_sc(subj, sel_lin, atlas, affine)
        W_g_gtpairs, n2, h2, rt2 = gt_geometry_sc(subj, gtpair_lin, atlas, affine)
        # 닻: 같은 GT 가닥을 "쌍 당 16 개" 가 아니라 자연 비율로 n2 개 뽑은 것.
        # W1-c pairs-family self 기준선(8,000 개에서 r=0.936)과 같은 규약이라 구현 검증도 된다.
        Sn = np.ascontiguousarray(
            z_b["streamlines"][np.sort(np.random.default_rng(0).choice(len(z_b["streamlines"]), n2, replace=False))],
            np.float32)
        tn, ln_ = visit_structure(Sn, atlas, affine)
        W_g_natural = sc_from_visits(tn, ln_, None)
        rec["route"]["gt_pairs_per_streamline"] = rt2[0]
        rec["route"]["gt_rois_per_streamline"] = rt2[1]
        rec["grid"] = {
            "model_geom_model_pairs": sc_scores(W, gt) | {"n_streamlines": int(keep.sum()), "n_pairs": int(sel.sum())},
            "model_geom_gt_pairs": sc_scores(W_m_gtpairs, gt) | {"n_streamlines": int(keep_gtpairs[s_pidx].sum()),
                                                                 "n_pairs": int(keep_gtpairs.sum())},
            "model_geom_all_pairs": sc_scores(W_m_allpairs, gt) | {"n_streamlines": int(len(s_pidx)), "n_pairs": 3321},
            "gt_geom_model_pairs": sc_scores(W_g_mpairs, gt) | {"n_streamlines": n1, "n_pairs": h1},
            "gt_geom_gt_pairs": sc_scores(W_g_gtpairs, gt) | {"n_streamlines": n2, "n_pairs": h2},
            "gt_geom_natural_counts": sc_scores(W_g_natural, gt) | {"n_streamlines": n2, "n_pairs": int(len(z_b["pair_ids"]))},
        }

        # --- edge_thr 스윕 ---------------------------------------------------
        sw = {}
        for thr in a.thr_sweep:
            s2 = prob > thr
            if not s2.any():
                continue
            W2 = sc_from_visits(tid, lin, s2[s_pidx])
            o2 = W2[IU] > 0
            sw[f"{thr:g}"] = {"n_pairs": int(s2.sum()), "n_streamlines": int(s2[s_pidx].sum()),
                              "recall": float((pos & o2).sum() / pos.sum()),
                              "precision": float((pos & o2).sum() / max(o2.sum(), 1)),
                              **sc_scores(W2, gt)}
        rec["thr_sweep"] = sw
        rows.append(rec)
        g = rec["grid"]
        print(f"[{i+1}/{len(subs)}] {sub}: sel {rec['n_selected']} · 점유 {rec['n_occ']} / GT {rec['n_gt_pos']} "
              f"· recall {rec['recall']:.3f} · FN {rec['confusion']['FN']['n']} "
              f"(a {rec['cause']['a_not_selected']['n']} / b {rec['cause']['b_selected_not_reached']['n']}) "
              f"· r MM {g['model_geom_model_pairs']['r']:.3f} MG {g['model_geom_gt_pairs']['r']:.3f} "
              f"GM {g['gt_geom_model_pairs']['r']:.3f} GG {g['gt_geom_gt_pairs']['r']:.3f}", flush=True)
    return rows


# --------------------------------------------------------------------- C9: pair dice 의 n_pred 결함
def displace_iid(S, rng, sigma_mm):
    """scripts/45_dice_displacement.py 의 displace(kind='iid') 와 같은 정의 (점 RMSE = sigma_mm)."""
    S = np.asarray(S, np.float32)
    if sigma_mm <= 0:
        return S.copy()
    return S + rng.standard_normal(S.shape).astype(np.float32) * np.float32(sigma_mm / np.sqrt(3.0))


def c9_npred_sweep(a):
    """C9: pair dice(n_pred=16) 가 iid 변위에서 단조 감소하지 않는 결함의 원인과 처방.

    W1-b 실측: pred=16 가닥 vs gt=번들 전체 규약에서 dice 0.498 -> 0.545(sigma=1.0) -> 하락.
    16 가닥의 복셀 footprint 가 GT 번들 footprint 의 부분집합(coverage 0.35, overreach 0)이라
    잡음이 pred 를 GT 복셀 안으로 부풀리면 coverage 가 overreach 보다 빨리 오른다.
    -> 흐릿한 생성기가 상을 받는다.

    여기서는 n_pred 를 바꿔 가며 (i) 언제 단조성이 회복되는지 (ii) 그때 sigma=0 천장이 얼마인지를
    실측한다. 두 규약을 모두 낸다:
      full  pred = 번들에서 n 개 (변위), gt = 번들 전체        <- 29_final_evaluation 의 현재 규약
      half  pred = 앞 절반에서 n 개 (변위), gt = 뒤 절반에서 n 개  <- 크기를 맞춘 규약
    """
    import nibabel as nib
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics
    from atm_sc.data.roi_groups import block_of_pairs, tier_of_strength
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()][: a.c9_subjects]
    rng = np.random.default_rng(20250906)
    ns, sigmas = a.c9_n, a.c9_sigma
    acc = {p: {n: {s: [] for s in sigmas} for n in ns} for p in ("full", "half")}
    cov = {n: {s: [] for s in sigmas} for n in ns}
    ovr = {n: {s: [] for s in sigmas} for n in ns}
    n_pairs_used = 0
    for sub in subs:
        subj = ROIPairSubject(sub)
        z = subj._bnd
        off, pid = z["pair_offsets"], np.asarray(z["pair_ids"], np.int64)
        size = np.diff(off)
        # 두 규약 모두를 같은 pair 에서 재려면 번들이 2*max(n) 이상이어야 한다
        ok = np.flatnonzero(size >= 2 * max(ns))
        assert len(ok) > 0, f"{sub}: 번들이 {2*max(ns)} 개 이상인 pair 가 없다"
        strength = np.asarray(subj.sc_mat)[pid[:, 0], pid[:, 1]]
        strat = tier_of_strength(strength) * 3 + block_of_pairs(pid)
        pick = []
        for g in np.unique(strat[ok]):
            c = ok[strat[ok] == g]
            pick.append(rng.choice(c, min(a.c9_pairs_per_stratum, len(c)), replace=False))
        pick = np.concatenate(pick)
        n_pairs_used += len(pick)
        for k in pick:
            B = np.ascontiguousarray(z["streamlines"][off[k]:off[k + 1]], np.float32)
            perm = rng.permutation(len(B))
            for n in ns:
                pf = B[perm[:n]]
                ph_a, ph_b = B[perm[:n]], B[perm[n:2 * n]]
                for s_ in sigmas:
                    r = bundle_geometry_metrics(displace_iid(pf, rng, s_), B, voxel_mm=a.c9_voxel_mm)
                    acc["full"][n][s_].append(r["dice"])
                    cov[n][s_].append(r["coverage"]); ovr[n][s_].append(r["overreach"])
                    r2 = bundle_geometry_metrics(displace_iid(ph_a, rng, s_), ph_b, voxel_mm=a.c9_voxel_mm)
                    acc["half"][n][s_].append(r2["dice"])
    out = {"meta": {"subjects": subs, "n_pairs_total": int(n_pairs_used), "n_pred": ns,
                    "sigmas_mm": sigmas, "voxel_mm": a.c9_voxel_mm, "displacement": "iid",
                    "protocols": {"full": "pred=n 가닥(변위) vs gt=번들 전체 (29_final_evaluation 현재 규약)",
                                  "half": "pred=n 가닥(변위) vs gt=서로 겹치지 않는 n 가닥 (크기 맞춘 규약)"}},
           "curves": {}, "verdict": {}}
    for prot in ("full", "half"):
        out["curves"][prot] = {}
        for n in ns:
            d = [float(np.mean(acc[prot][n][s_])) for s_ in sigmas]
            bump = float(max(d) - d[0])
            arg = float(sigmas[int(np.argmax(d))])
            out["curves"][prot][str(n)] = {
                "dice": d, "dice0": d[0], "max_bump": bump, "argmax_sigma_mm": arg,
                "monotone": bool(all(d[i + 1] <= d[i] + 1e-9 for i in range(len(d) - 1))),
                **({"coverage": [float(np.mean(cov[n][s_])) for s_ in sigmas],
                    "overreach": [float(np.mean(ovr[n][s_])) for s_ in sigmas]} if prot == "full" else {})}
    for prot in ("full", "half"):
        mono = [n for n in ns if out["curves"][prot][str(n)]["monotone"]]
        out["verdict"][prot] = {"smallest_monotone_n_pred": (min(mono) if mono else None),
                                "ceiling_at_smallest_monotone": (
                                    out["curves"][prot][str(min(mono))]["dice0"] if mono else None)}
    p = ROOT / "outputs" / "eval" / "w2c_c9_pair_dice_npred.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    assert p.stat().st_size > 0
    print(f"저장: {p.relative_to(ROOT)}")
    for prot in ("full", "half"):
        for n in ns:
            c = out["curves"][prot][str(n)]
            print(f"  {prot} n={n:<4d} dice0={c['dice0']:.4f} max_bump={c['max_bump']:+.4f} "
                  f"@sigma={c['argmax_sigma_mm']} monotone={c['monotone']}")
    return 0


# --------------------------------------------------------------------- 집계
def mean_of(rows, fn):
    v = [fn(r) for r in rows]
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def aggregate(rows, a):
    def m(path):
        def f(r):
            x = r
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
                if x is None:
                    return None
            return x if isinstance(x, (int, float)) else None
        return mean_of(rows, f)

    conf = {}
    for c in ("TP", "FN", "FP"):
        conf[c] = {"n": m(["confusion", c, "n"]),
                   "tier": {t: m(["confusion", c, "tier", t]) for t in TIERS},
                   "block": {b: m(["confusion", c, "block", b]) for b in BLOCKS},
                   "gt_sum_frac": m(["confusion", c, "gt_sum_frac"]),
                   "gt_median": m(["confusion", c, "gt_median"]),
                   "len_mean_mm": m(["confusion", c, "len_mean_mm"])}
    cause = {k: {"n": m(["cause", k, "n"]),
                 "tier": {t: m(["cause", k, "tier", t]) for t in TIERS},
                 "block": {b: m(["cause", k, "block", b]) for b in BLOCKS},
                 "gt_sum_frac": m(["cause", k, "gt_sum_frac"]),
                 "gt_median": m(["cause", k, "gt_median"])}
             for k in rows[0]["cause"]}
    grid = {k: {"r": m(["grid", k, "r"]), "r_log": m(["grid", k, "r_log"]),
                "tier": {t: m(["grid", k, "tier", t]) for t in TIERS},
                "block": {b: m(["grid", k, "block", b]) for b in BLOCKS},
                "n_edges_pred": m(["grid", k, "n_edges_pred"]),
                "n_pairs": m(["grid", k, "n_pairs"]),
                "n_streamlines": m(["grid", k, "n_streamlines"])}
            for k in rows[0]["grid"]}
    thrs = sorted({t for r in rows for t in r["thr_sweep"]}, key=float, reverse=True)
    sweep = {t: {k: m(["thr_sweep", t, k]) for k in
                 ("n_pairs", "n_streamlines", "recall", "precision", "r", "r_log", "n_edges_pred")}
             | {"tier": {x: m(["thr_sweep", t, "tier", x]) for x in TIERS}}
             for t in thrs}

    r_MM = grid["model_geom_model_pairs"]["r"]
    r_MG = grid["model_geom_gt_pairs"]["r"]
    r_GM = grid["gt_geom_model_pairs"]["r"]
    r_GG = grid["gt_geom_gt_pairs"]["r"]
    decomp = {
        "note": "같은 규약(쌍 당 16 가닥, pass-SC, GT=.mat)으로 채운 2x2. "
                "행=기하 출처, 열=쌍 집합 출처. 점유 몫은 열을 바꾼 이득, 기하 몫은 행을 바꾼 이득.",
        "r_model_geom_model_pairs": r_MM,
        "r_model_geom_gt_pairs": r_MG,
        "r_gt_geom_model_pairs": r_GM,
        "r_gt_geom_gt_pairs": r_GG,
        "occupancy_gain_model_geom": None if None in (r_MG, r_MM) else r_MG - r_MM,
        "geometry_gain_model_pairs": None if None in (r_GM, r_MM) else r_GM - r_MM,
        "occupancy_gain_gt_geom": None if None in (r_GG, r_GM) else r_GG - r_GM,
        "geometry_gain_gt_pairs": None if None in (r_GG, r_MG) else r_GG - r_MG,
        "total_gap_to_ceiling": None if None in (r_GG, r_MM) else r_GG - r_MM,
    }
    tot = decomp["total_gap_to_ceiling"]
    if tot and abs(tot) > 1e-9:
        occ_share = 0.5 * (decomp["occupancy_gain_model_geom"] + decomp["occupancy_gain_gt_geom"]) / tot
        geo_share = 0.5 * (decomp["geometry_gain_model_pairs"] + decomp["geometry_gain_gt_pairs"]) / tot
        decomp["occupancy_share"] = float(occ_share)      # 두 경로 평균 (상호작용은 반씩 나눠 가짐)
        decomp["geometry_share"] = float(geo_share)
    return {"confusion": conf, "cause": cause, "grid": grid, "decomposition": decomp, "thr_sweep": sweep,
            "recall": m(["recall"]), "precision": m(["precision"]),
            "recall_tier": {t: m(["recall_tier", t]) for t in TIERS},
            "n_gt_pos": m(["n_gt_pos"]), "n_occ": m(["n_occ"]), "n_selected": m(["n_selected"]),
            "fn_correlates": {k: m(["fn_correlates", k]) for k in rows[0]["fn_correlates"]},
            "support": {k: m(["support", k]) for k in rows[0]["support"]},
            "route": {k: m(["route", k]) for k in rows[0]["route"]}}


def main(a):
    if a.stage == "c9":
        return c9_npred_sweep(a)
    subs = None
    if a.stage in ("gen", "both"):
        if LOCK.exists() and not a.ignore_lock:
            print(f"GPU 잠금 {LOCK} 이 있다 -- 다른 작업(W2-a)이 임계 경로다. "
                  f"--ignore-lock 을 주거나 잠금이 풀린 뒤 다시 실행하라.", flush=True)
            return 1
        took = False
        if not LOCK.exists():
            LOCK.write_text(f"47_occupancy_diag {time.strftime('%F %T')}\n"); took = True
        try:
            subs = generate_all(a)
        finally:
            if took and LOCK.exists():
                LOCK.unlink()
    if a.stage == "gen":
        return 0
    if subs is None:
        subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
        subs = [s for s in subs if (CACHE_DIR / f"{s}.npz").exists()]
        if a.limit:
            subs = subs[: a.limit]
    assert subs, "분석할 subject 가 없다"
    assert verify_vectorized_sc(subs[0])
    rows = analyse(subs, a)
    out = {"summary": {"checkpoint": a.ckpt, "n_subjects": len(rows), "split": a.subjects,
                       "n_per_pair": N_PER_PAIR, "edge_thr": a.edge_thr, "time": time.strftime("%F %T"),
                       **aggregate(rows, a)},
           "per_subject": rows,
           "cmd": f"python scripts/47_occupancy_diag.py --ckpt {a.ckpt} --subjects {a.subjects}"}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    assert OUT_JSON.stat().st_size > 0
    print(f"\n저장: {OUT_JSON.relative_to(ROOT)}", flush=True)
    d = out["summary"]["decomposition"]
    print(f"2x2  MM {d['r_model_geom_model_pairs']:.3f} | MG {d['r_model_geom_gt_pairs']:.3f} | "
          f"GM {d['r_gt_geom_model_pairs']:.3f} | GG {d['r_gt_geom_gt_pairs']:.3f}", flush=True)
    print(f"점유 몫 {d.get('occupancy_share')} · 기하 몫 {d.get('geometry_share')}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt")
    ap.add_argument("--subjects", default="outputs/splits/test.txt")
    ap.add_argument("--init-bundle", default="AF_L")
    ap.add_argument("--edge-thr", type=float, default=0.5)
    ap.add_argument("--thr-sweep", type=float, nargs="*",
                    default=[0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0])
    ap.add_argument("--stage", choices=["gen", "analyse", "both", "c9"], default="both")
    ap.add_argument("--c9-subjects", type=int, default=5)
    ap.add_argument("--c9-pairs-per-stratum", type=int, default=8)
    ap.add_argument("--c9-n", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--c9-sigma", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.55, 5.0, 8.0])
    ap.add_argument("--c9-voxel-mm", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ignore-lock", action="store_true")
    sys.exit(main(ap.parse_args()))
