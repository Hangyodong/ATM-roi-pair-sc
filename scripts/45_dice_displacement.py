#!/usr/bin/env python
"""D0-b: GT streamline 을 **알려진 변위**로 흔들어 voxel dice 곡선을 만든다.

가설: 디코더의 복원 오차가 voxel dice 를 떨어뜨리는 주원인이다.
  - GT streamline 은 128 점 등간격, 길이 ~83 mm -> 점 간격 ~1.24 mm
  - 채점 복셀 2 mm -> 3.55 mm 는 1.8 칸
  - 오라클 latent(정답 z)로 디코딩해도 pair dice 0.167 (실제 표본끼리의 천장 0.704)

반증 가능한 예측: GT 를 복원 오차만큼 흔들면 pair dice 가 0.167 근처로 떨어져야 한다.
훨씬 높으면 가설 기각 -- dice 를 떨어뜨리는 다른 원인이 있다는 뜻이다.

**기준 오차가 하나가 아니다** (W1-a, 2026-09-06). ConvVAE 의 BatchNorm 5개 때문에
같은 복원이 train 모드(batch 통계) 3.70 mm, eval 모드(running 통계) 7.96~8.14 mm 다.
학습 로그의 3.55 mm 는 train 모드 값이고 **추론 시점의 실제 오차는 약 8 mm** 다.
따라서 sigma=3.55 와 sigma=8.0 둘 다에서 dice 를 보고한다 -- 어느 쪽이 0.167 을
설명하는지가 "학습 시점 오차가 문제냐 추론 시점 오차가 문제냐" 를 가른다.

채점 규약은 scripts/43_geometry_baselines.py / 29_final_evaluation.py --trk-eval 과 같다:
  bundle_geometry_metrics(pred, gt, voxel_mm=2.0). 복셀 격자는 두 다발 공통 원점(최소 좌표)
  기준 floor(mm / 2.0) -- 아틀라스 격자가 아니다. 아틀라스는 건전성 확인에만 쓴다.
  whole-brain 양쪽 8,000 가닥, pair 단위 16 가닥.

변위 세 종류 (셋 다 **같은** 점 변위 RMSE 를 갖도록 정규화한다):
  iid     점마다 독립 등방 가우시안              -- 가닥이 부풀어 오른다
  smooth  arclength 방향 저주파 4 모드(평행이동+완만한 굽힘) -- autoencoder 의 실제 오차에 가깝다
  shift   가닥 단위 평행이동 (저주파 1 모드)     -- 상관의 극단

RMSE 정의는 trainer.py 의 recon_rmse_mm 과 같다: sqrt(mean_points ||d||^2) (3D 점 거리의 RMS).
따라서 iid 의 축별 표준편차는 sigma/sqrt(3) 이다.

GPU 를 쓰지 않는다. 모델을 돌리지 않는다.

  python scripts/45_dice_displacement.py --n-subjects 4
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

from atm_sc.data.paths import ATLAS                                    # noqa: E402
from atm_sc.data.roi_groups import BLOCKS, TIERS, block_of_pairs, tier_of_strength  # noqa: E402
from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics  # noqa: E402

N_ROI = 82
BUNDLE_DIR = ROOT / "outputs" / "roi_pairs"
BRAIN_MM = (-110.0, 110.0)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.55, 5.0, 6.5, 8.0, 10.0, 14.0, 20.0)
KINDS = ("iid", "smooth", "shift")

# 비교 기준 (모두 실측치. outputs/eval/{final_p4_joint_step3000,s5e_geometry_baselines}.json)
REF = {"model_wb_dice": 0.5457, "cross_subject_wb_dice": 0.5738, "group_wb_dice": 0.6141,
       "self_wb_dice": 0.7831, "model_pair_dice": 0.0946, "oracle_pair_dice": 0.167,
       "bank_pair_dice": 0.137, "prior_pair_dice": 0.053, "pair_ceiling_half": 0.7039,
       "pair_n16_vs_full_ceiling": 0.3854, "cross_subject_pair_n16": 0.1395,
       "group_pair_n16": 0.1786,
       # 복원 오차는 BatchNorm 통계 때문에 모드마다 다르다 (W1-a 실측, 2026-09-06):
       #   train 모드(batch 통계) 3.70 mm  <-  학습 로그가 보고해 온 값 (3.55)
       #   eval  모드(running 통계) 7.96~8.14 mm  <-  **추론 시점의 실제 오차**
       # flip(방향 모호성)은 원인이 아니다: frac_flipped 0.0 %.
       "recon_rmse_mm_train_mode": 3.55, "recon_rmse_mm_eval_mode": 8.0}
KEY_SIGMAS = (3.55, 8.0)


# --------------------------------------------------------------------------- 건전성
def load_atlas():
    import nibabel as nib
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    have = set(np.unique(atlas).tolist()) - {0}
    missing = sorted(set(range(1, N_ROI + 1)) - have)
    assert not missing, f"아틀라스에 없는 라벨 {missing}"
    return atlas.shape


def check_streamlines(name: str, S: np.ndarray):
    assert S.ndim == 3 and S.shape[1:] == (128, 3), f"{name}: shape {S.shape}"
    assert len(S) > 0, f"{name}: streamline 이 0 개"
    P = S.reshape(-1, 3).astype(np.float64)
    n_bad = int((~np.isfinite(P)).sum())
    assert n_bad == 0, f"{name}: 비유한 좌표 {n_bad} 개"
    assert np.abs(P).max() > 1.0, f"{name}: 좌표가 전부 0 근처"
    lo, hi = P.min(0), P.max(0)
    assert lo.min() > BRAIN_MM[0] and hi.max() < BRAIN_MM[1], \
        f"{name}: bbox {lo.tolist()}..{hi.tolist()} 가 뇌 크기(mm) 범위 밖"
    assert (hi - lo).min() > 30.0, f"{name}: bbox 가 너무 납작 {(hi - lo).tolist()}"
    return [lo.tolist(), hi.tolist()]


# --------------------------------------------------------------------------- 채점 (43 과 동일)
def voxel_scores(pred, gt, voxel_mm=2.0):
    """bundle_geometry_metrics 의 voxel 부분만 뽑은 것. 값은 bit-exact 로 같다 (--verify 로 확인).

    ks_2samp / duplicate_mask 는 여기서 쓰지 않으므로 뺀다 (수만 번 호출하기 때문)."""
    pg = np.asarray(pred, np.float64).reshape(-1, 3)
    pt = np.asarray(gt, np.float64).reshape(-1, 3)
    origin = np.concatenate([pg, pt]).min(0)
    vg, vt = (np.floor((p - origin) / voxel_mm).astype(np.int64) for p in (pg, pt))
    dims = np.concatenate([vg, vt]).max(0) + 1
    lin = lambda v: np.unique((v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2])
    sg, st = lin(vg), lin(vt)
    inter = np.intersect1d(sg, st, assume_unique=True).size
    return {"dice": 2 * inter / (sg.size + st.size), "coverage": inter / st.size,
            "overreach": (sg.size - inter) / st.size}


def verify_scoring(S, voxel_mm):
    """자체 구현이 upstream bundle_geometry_metrics 와 정확히 같은 값을 내는지."""
    rng = np.random.default_rng(0)
    a = S[:200].astype(np.float32)
    b = (S[200:400].astype(np.float32) + rng.standard_normal((200, 128, 3)).astype(np.float32))
    ref = bundle_geometry_metrics(a, b, voxel_mm=voxel_mm)
    mine = voxel_scores(a, b, voxel_mm)
    for k in ("dice", "coverage", "overreach"):
        assert abs(ref[k] - mine[k]) < 1e-12, f"채점 불일치 {k}: {ref[k]} vs {mine[k]}"
    same = voxel_scores(a, a, voxel_mm)
    assert same["dice"] == 1.0, f"자기 자신 dice 가 {same['dice']}"
    return {"dice_ref": ref["dice"], "dice_mine": mine["dice"]}


# --------------------------------------------------------------------------- 변위
def displace(S, rng, kind, sigma_mm, n_modes=4):
    """S [n,128,3] -> 변위된 복사본. 목표 RMSE(점 거리 RMS) = sigma_mm.

    iid     d[n,p,c] ~ N(0, (s/sqrt(3))^2)                E||d||^2 = s^2
    smooth  d(t) = sum_{k<K} A_k cos(k pi t), A ~ N(0,a^2)
            mean_t E||d(t)||^2 = 3 a^2 (1 + (K-1)/2)  ->  a = s / sqrt(3(1+(K-1)/2))
    shift   smooth 의 K=1 (가닥마다 상수 평행이동)
    """
    S = np.asarray(S, np.float32)
    if sigma_mm <= 0:
        return S.copy()
    n, P, _ = S.shape
    if kind == "iid":
        d = rng.standard_normal((n, P, 3)).astype(np.float32) * np.float32(sigma_mm / np.sqrt(3.0))
    elif kind in ("smooth", "shift"):
        K = 1 if kind == "shift" else n_modes
        t = np.linspace(0.0, 1.0, P, dtype=np.float64)
        B = np.cos(np.pi * np.arange(K, dtype=np.float64)[None, :] * t[:, None])   # [P,K]
        amp = sigma_mm / np.sqrt(3.0 * (1.0 + (K - 1) / 2.0))
        A = rng.standard_normal((n, K, 3)) * amp
        d = np.einsum("pk,nkc->npc", B, A).astype(np.float32)
    else:
        raise ValueError(kind)
    return S + d


def rmse_mm(pred, base):
    """trainer.py recon_rmse_mm 과 같은 정의: sqrt(mean_points ||d||^2)."""
    d = np.asarray(pred, np.float64) - np.asarray(base, np.float64)
    return float(np.sqrt((d ** 2).sum(-1).mean()))


def rmse_flip_aware(pred, base):
    """stream_recon_loss 처럼 뒤집힘을 허용했을 때의 가닥별 RMSE 평균."""
    p, g = np.asarray(pred, np.float64), np.asarray(base, np.float64)
    a = ((p - g) ** 2).sum(-1).mean(-1)
    b = ((p - g[:, ::-1]) ** 2).sum(-1).mean(-1)
    return float(np.sqrt(np.minimum(a, b)).mean())


# --------------------------------------------------------------------------- 데이터
def load_bundles(sub):
    z = np.load(BUNDLE_DIR / sub / "bundles.npz")
    S = z["streamlines"]
    off = z["pair_offsets"].astype(np.int64)
    ids = z["pair_ids"].astype(np.int64)
    cnt = z["pair_count_full"].astype(np.int64)
    assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
    assert off[0] == 0 and off[-1] == len(S) and len(ids) == len(off) - 1
    return S, ids, off, cnt


def model_gt_sample(S, n):
    """모델 평가(scripts/29)와 **같은** GT 표본: default_rng(0) -> sort(choice)."""
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(S), min(n, len(S)), replace=False))
    return S[sel].astype(np.float32), sel


def take_excluding(S, n, rng, exclude):
    keep = np.ones(len(S), bool); keep[exclude] = False
    idx = np.flatnonzero(keep)
    assert len(idx) >= n, f"표본 부족 {len(idx)} < {n}"
    return S[np.sort(rng.choice(idx, n, replace=False))].astype(np.float32)


def select_pairs(ids, off, cnt, n_top, n_strat, min_bundle, rng):
    """top-N (모델 평가와 동일) + tier x block 층화 표본. -> [(k, stratum, is_top)]"""
    have = np.diff(off)
    tier = tier_of_strength(cnt.astype(np.float64))
    block = block_of_pairs(ids)
    top = np.argsort(-cnt)[:n_top]
    top = [k for k in top if have[k] >= 8]
    chosen = {int(k): True for k in top}
    elig = np.flatnonzero((have >= min_bundle))
    cells = {}
    for k in elig:
        cells.setdefault((int(tier[k]), int(block[k])), []).append(int(k))
    if cells:
        per = int(np.ceil(n_strat / len(cells)))
        for c in sorted(cells):
            pool = np.array(cells[c])
            sel = rng.choice(pool, min(per, len(pool)), replace=False)
            for k in sel:
                chosen.setdefault(int(k), False)
    out = []
    for k, is_top in sorted(chosen.items()):
        out.append((k, f"{TIERS[tier[k]]}|{BLOCKS[block[k]]}", bool(is_top)))
    return out


# --------------------------------------------------------------------------- 집계
def agg(rows, key="dice"):
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def inverse_table(curve, targets):
    """curve = [(rmse, dice)] rmse 오름차순. dice 목표 -> 필요한 RMSE 상한 (선형 보간)."""
    r = np.array([c[0] for c in curve], float)
    d = np.array([c[1] for c in curve], float)
    out = {}
    for t in targets:
        if d[0] < t:
            out[f"{t:g}"] = None                       # sigma=0 에서도 목표에 못 미친다
            continue
        if d[-1] >= t:
            out[f"{t:g}"] = f">{r[-1]:.2f}"            # 측정 범위 전체에서 목표 이상
            continue
        i = int(np.flatnonzero(d < t)[0])              # d[i-1] >= t > d[i]
        r0, r1, d0, d1 = r[i - 1], r[i], d[i - 1], d[i]
        out[f"{t:g}"] = round(float(r0 + (d0 - t) * (r1 - r0) / (d0 - d1)), 3)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--n-sample", type=int, default=8000, help="whole-brain 표본 (모델 평가와 동일)")
    ap.add_argument("--n-per-pair", type=int, default=16, help="pair 당 제출 가닥 수 (모델과 동일)")
    ap.add_argument("--n-top-pairs", type=int, default=50, help="count 상위 pair (43/29 와 동일)")
    ap.add_argument("--n-strat-pairs", type=int, default=60, help="tier x block 층화 pair")
    ap.add_argument("--min-bundle", type=int, default=32)
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--n-modes", type=int, default=4, help="smooth 변위의 저주파 모드 수")
    ap.add_argument("--sigmas", type=float, nargs="*", default=list(SIGMAS))
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--seed", type=int, default=20250906)
    ap.add_argument("--out", default="outputs/eval/w1b_dice_displacement.json")
    a = ap.parse_args()

    t_start = time.time()
    cmd = "python scripts/45_dice_displacement.py " + " ".join(sys.argv[1:])
    sigmas = sorted(set(float(s) for s in a.sigmas))
    assert sigmas[0] == 0.0, "sigma=0 이 있어야 자기 자신 dice=1.0 을 확인할 수 있다"

    atlas_shape = load_atlas()
    test = [l.strip() for l in open(ROOT / "outputs/splits/test.txt") if l.strip()]
    order = np.random.default_rng(a.seed).permutation(len(test))
    subs = [test[i] for i in order[: a.n_subjects]]
    print(f"atlas {atlas_shape} labels 1..{N_ROI} OK | subjects {subs} | seed {a.seed}", flush=True)

    # protocol -> (설명, sigma=0 기대 dice)
    PROTO = {
        "wb_self":          "pred=jitter(GT 8000), gt=같은 8000. 변위만의 순수 효과 (sigma=0 -> 1.0)",
        "wb_disjoint":      "pred=jitter(다른 8000), gt=GT 8000. 표본 차이 + 변위 (sigma=0 -> self 천장 0.783)",
        "pair_full_self":   "pred=jitter(번들 전체), gt=같은 번들. 오라클 latent 로 번들 전체를 복원한 경우",
        "pair_self16":      "pred=jitter(16 가닥), gt=같은 16 가닥. 변위만 (sigma=0 -> 1.0)",
        "pair_n16_vs_full": "pred=jitter(16 가닥), gt=번들 전체. **모델 pair dice 와 직접 비교** (sigma=0 -> 0.385)",
        "pair_half":        "pred=jitter(번들 앞 절반), gt=뒤 절반. 천장 규약 (sigma=0 -> 0.704)",
    }
    rows = {p: {k: {s: [] for s in sigmas} for k in a.kinds} for p in PROTO}
    rmses = {p: {k: {s: [] for s in sigmas} for k in a.kinds} for p in PROTO}
    pair_meta, wb_meta, verify = [], [], None

    for si, sub in enumerate(subs):
        t0 = time.time()
        S, ids, off, cnt = load_bundles(sub)
        bbox = check_streamlines(f"{sub}/bundles", S[:: max(1, len(S) // 2000)])
        if verify is None:
            verify = verify_scoring(S, a.voxel_mm)
            print(f"채점 자체구현 검증 OK (upstream 과 bit-exact): {verify}", flush=True)

        gt_wb, gt_idx = model_gt_sample(S, a.n_sample)
        other_wb = take_excluding(S, a.n_sample, np.random.default_rng(a.seed + 100 + si), gt_idx)
        wb_meta.append({"subject": sub, "n_streamlines": int(len(S)), "n_pairs": int(len(ids)),
                        "bbox": bbox, "gt_len_mean_mm": float(np.linalg.norm(
                            np.diff(gt_wb.astype(np.float64), axis=1), axis=-1).sum(1).mean())})

        sel = select_pairs(ids, off, cnt, a.n_top_pairs, a.n_strat_pairs, a.min_bundle,
                           np.random.default_rng(a.seed + 200 + si))
        # pair 별 고정 표본 (sigma/kind 에 상관없이 같은 밑바탕을 쓴다)
        pr_rng = np.random.default_rng(a.seed + 300 + si)
        pairs = []
        for k, stratum, is_top in sel:
            b = S[off[k]:off[k + 1]].astype(np.float32)
            if len(b) < 8:
                continue
            n16 = min(a.n_per_pair, len(b))
            i16 = pr_rng.choice(len(b), n16, replace=False)
            h = len(b) // 2
            pairs.append({"k": int(k), "stratum": stratum, "is_top": is_top, "n": int(len(b)),
                          "count_full": int(cnt[k]), "b": b, "s16": b[i16], "h": h})
        pair_meta.append({"subject": sub, "n_pairs_scored": len(pairs),
                          "n_top": sum(p["is_top"] for p in pairs),
                          "strata": {s: sum(p["stratum"] == s for p in pairs)
                                     for s in sorted({p["stratum"] for p in pairs})}})
        print(f"[{si+1}/{len(subs)}] {sub}: {len(S):,} 가닥, pair {len(pairs)} 개 "
              f"(top {pair_meta[-1]['n_top']})", flush=True)

        for kind in a.kinds:
            for sg in sigmas:
                # 시드는 (subject, kind, sigma) 로 결정된다. 문자열 hash 는 실행마다 달라지므로 쓰지 않는다.
                rng = np.random.default_rng([a.seed, si, KINDS.index(kind), int(round(sg * 100))])
                # --- whole-brain ---
                for proto, base, gt in (("wb_self", gt_wb, gt_wb), ("wb_disjoint", other_wb, gt_wb)):
                    p = displace(base, rng, kind, sg, a.n_modes)
                    m = voxel_scores(p, gt, a.voxel_mm)
                    m["rmse_mm"] = rmse_mm(p, base)
                    # 손실은 flip 을 허용한다 (stream_recon_loss). 보고 지표 3.55 mm 와 같은 뜻인지 확인용.
                    m["rmse_flip_mm"] = rmse_flip_aware(p, base)
                    rows[proto][kind][sg].append(m); rmses[proto][kind][sg].append(m["rmse_mm"])
                    if sg == 0.0 and proto == "wb_self":
                        assert m["dice"] == 1.0, f"{sub} wb_self sigma=0 dice={m['dice']} != 1.0"
                # --- pair ---
                for pd in pairs:
                    for proto, base, gt in (
                            ("pair_full_self",   pd["b"],   pd["b"]),
                            ("pair_self16",      pd["s16"], pd["s16"]),
                            ("pair_n16_vs_full", pd["s16"], pd["b"]),
                            ("pair_half",        pd["b"][:pd["h"]], pd["b"][pd["h"]:])):
                        p = displace(base, rng, kind, sg, a.n_modes)
                        m = voxel_scores(p, gt, a.voxel_mm)
                        m["rmse_mm"] = rmse_mm(p, base)
                        m["stratum"] = pd["stratum"]; m["is_top"] = pd["is_top"]
                        rows[proto][kind][sg].append(m); rmses[proto][kind][sg].append(m["rmse_mm"])
                        if sg == 0.0 and proto in ("pair_full_self", "pair_self16"):
                            assert m["dice"] == 1.0, f"{proto} sigma=0 dice={m['dice']}"
            print(f"    {kind}: {time.time() - t0:.0f}s", flush=True)
        del S

    # --- 집계 -------------------------------------------------------------
    curves, warn = {}, []
    for proto in PROTO:
        curves[proto] = {}
        for kind in a.kinds:
            cur = []
            for sg in sigmas:
                r = rows[proto][kind][sg]
                e = np.array(rmses[proto][kind][sg], float)
                d = agg(r); c = agg(r, "coverage"); o = agg(r, "overreach")
                if sg > 0:
                    ratio = e.mean() / sg
                    assert 0.9 < ratio < 1.1, f"{proto}/{kind} sigma={sg}: 실측 RMSE {e.mean():.3f} 가 지정과 어긋남"
                else:
                    assert e.max() == 0.0, f"{proto}/{kind} sigma=0 인데 RMSE {e.max()}"
                item = {"sigma_mm": sg, "rmse_mm": float(e.mean()), "rmse_sd": float(e.std()),
                        "dice": d["mean"], "dice_sd": d["sd"], "n_obs": d["n"],
                        "coverage": c["mean"], "overreach": o["mean"]}
                if "rmse_flip_mm" in r[0]:
                    item["rmse_flip_mm"] = agg(r, "rmse_flip_mm")["mean"]
                if proto.startswith("pair"):
                    top = [x for x in r if x["is_top"]]
                    if top:
                        item["dice_top50"] = agg(top)["mean"]
                    item["dice_by_stratum"] = {s: agg([x for x in r if x["stratum"] == s])["mean"]
                                               for s in sorted({x["stratum"] for x in r})}
                cur.append(item)
            dd = [x["dice"] for x in cur]
            if any(dd[i + 1] > dd[i] + 1e-9 for i in range(len(dd) - 1)):
                warn.append(f"{proto}/{kind}: dice 가 sigma 에 단조 감소하지 않는다 {['%.4f' % x for x in dd]}")
            curves[proto][kind] = cur

    # --- 역표 -------------------------------------------------------------
    wb_targets = [0.783, 0.70, 0.65, 0.614, 0.5738, 0.5457, 0.50]
    pair_targets = [0.7039, 0.5, 0.4, 0.3854, 0.3, 0.2, 0.167, 0.0946]
    inverse = {}
    for proto in PROTO:
        tg = wb_targets if proto.startswith("wb") else pair_targets
        inverse[proto] = {kind: inverse_table([(x["rmse_mm"], x["dice"]) for x in curves[proto][kind]], tg)
                          for kind in a.kinds}

    # 핵심 비교: train 모드 오차(3.55) 와 eval 모드 오차(8.0) 둘 다
    at_key = {}
    for ks in KEY_SIGMAS:
        at_key[f"{ks:g}"] = {}
        for proto in PROTO:
            at_key[f"{ks:g}"][proto] = {}
            for kind in a.kinds:
                c0 = curves[proto][kind][0]["dice"]
                hit = [x for x in curves[proto][kind] if abs(x["sigma_mm"] - ks) < 1e-9]
                if hit:
                    at_key[f"{ks:g}"][proto][kind] = {
                        "dice": hit[0]["dice"], "ceiling_sigma0": c0,
                        "frac_of_ceiling": hit[0]["dice"] / c0 if c0 else None}

    res = {"meta": {"cmd": cmd, "seed": a.seed, "subjects": subs, "n_sample": a.n_sample,
                    "n_per_pair": a.n_per_pair, "n_top_pairs": a.n_top_pairs,
                    "n_strat_pairs": a.n_strat_pairs, "min_bundle": a.min_bundle,
                    "voxel_mm": a.voxel_mm, "n_modes": a.n_modes, "sigmas": sigmas,
                    "kinds": list(a.kinds), "atlas": str(ATLAS), "atlas_shape": list(atlas_shape),
                    "protocols": PROTO, "reference": REF, "scoring_verified": verify,
                    "rmse_definition": "sqrt(mean_points ||pred-base||^2)  (trainer.py recon_rmse_mm 과 동일)",
                    "displacement": {
                        "iid": "점마다 독립 N(0,(s/sqrt3)^2 I3)",
                        "smooth": f"arclength 저주파 {a.n_modes} 모드 cos(k pi t), k=0..{a.n_modes-1}",
                        "shift": "가닥 단위 평행이동 (저주파 1 모드)"},
                    "subject_meta": wb_meta, "pair_meta": pair_meta},
            "curves": curves, "inverse_rmse_for_dice": inverse,
            "at_key_sigma": at_key, "key_sigmas": list(KEY_SIGMAS),
            "warnings": warn, "elapsed_sec": time.time() - t_start}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # --- 출력 -------------------------------------------------------------
    for proto in PROTO:
        print(f"\n=== {proto} === {PROTO[proto]}")
        hdr = "  sigma  rmse " + "".join(f"{k:>10}" for k in a.kinds)
        print(hdr)
        for i, sg in enumerate(sigmas):
            r0 = curves[proto][a.kinds[0]][i]["rmse_mm"]
            print(f"  {sg:5.2f} {r0:5.2f} " + "".join(
                f"{curves[proto][k][i]['dice']:>10.3f}" for k in a.kinds))
    for ks in KEY_SIGMAS:
        tag = "train 모드 복원오차" if ks == 3.55 else "eval 모드 복원오차 (추론 시점)"
        print(f"\n=== sigma={ks} mm ({tag}) vs 기준 ===")
        for proto in PROTO:
            for kind in a.kinds:
                v = at_key[f"{ks:g}"][proto].get(kind)
                if v:
                    print(f"  {proto:<18}{kind:<8}dice={v['dice']:.3f}  천장(sigma0)={v['ceiling_sigma0']:.3f}  "
                          f"천장대비={v['frac_of_ceiling']*100:.0f}%")
    print("\n=== 역표: dice X 를 원하면 RMSE <= Y mm ===")
    for proto in PROTO:
        print(f"  {proto}")
        for kind in a.kinds:
            print(f"    {kind:<8}" + "  ".join(f"{t}->{v}" for t, v in inverse[proto][kind].items()))
    if warn:
        print("\n경고:")
        for w in warn:
            print("  " + w)
    print(f"\n저장: {out}  ({res['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
