#!/usr/bin/env python
"""S1-b 공간 결정 probe — ROI 별 국소 풀링에 개인별 SC 편차를 설명하는 신호가 있는가.

  python scripts/42_space_probe.py --stage all          # GPU forward + probe
  python scripts/42_space_probe.py --stage extract      # GPU forward 만 (특징 캐시 생성)
  python scripts/42_space_probe.py --stage probe        # 캐시된 특징으로 probe 만 (GPU 불필요)

전뇌 global average pooling 이 개인차를 지운다는 것은 측정으로 확정됐다 (예측 SC 의
subject 간 상관 0.99997 vs GT 0.882). 해결안인 ROI 별 국소 풀링은 구조 변경 + 전 phase
재학습(수일)을 요구하므로, **학습 없이** forward + ridge probe 로 먼저 판정한다.

분석 방법은 docs/PIPELINE_10_RESOLUTION_PLAN.md "S1 분석 프로토콜 — 실험 전 사전등록"
에 실험 전에 못박혀 있다. 결과를 보고 방법을 바꾸면 판정이 성립하지 않는다.

조건
  R-ROI  rigid(체크포인트와 같은 분포) 입력 -> stage3 [1,256,49,58,49] -> 82-ROI 풀링 f[82,256]
  R-GAP  rigid -> 전뇌 pooling a[512] (현재 구조, 대조군)
  S-ROI  SyN 입력 -> 82-ROI 풀링          (2차. off-distribution 여부를 먼저 검사한다)
  S-GAP  SyN -> 전뇌 pooling

대조군 (없으면 결과를 믿지 않는다)
  1 subject 라벨 셔플 — r ~ 0 이 아니면 누수. 어떤 값이 나오든 판정 보류.
  2 전뇌 pooling (R-GAP)
  3 머리 크기 단일 회귀 (뇌 부피 / T1 총 강도)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEAT_DIR = ROOT / "outputs" / "cache" / "s1b_feats"
OUT_JSON = ROOT / "outputs" / "eval" / "s1b_space_probe.json"
CKPT = ROOT / "outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt"
GPU_LOCK = ROOT / "outputs" / "gpu.lock"
TRAIN_PIPELINE_LOCKS = [ROOT / "outputs/checkpoints/retrain/pipeline.lock",
                        ROOT / "outputs/checkpoints/route2/pipeline.lock",
                        ROOT / "outputs/checkpoints/pipeline.lock"]

SEED_SHUFFLE = 20260906          # subject 라벨 셔플 대조군
SEED_FOLD = 7                    # train 내부 5-fold CV 분할
SEED_BOOT = 11                   # subject 부트스트랩
N_PC = 32
N_FOLD = 5
N_BOOT = 10000
ALPHAS = np.logspace(-2, 6, 17)
WEAK_ROI_NAMES = ("R_subthalamic_nucleus", "L_subthalamic_nucleus", "L_red_nucleus")


# --- lock (scripts/19_train_pipeline.py:acquire_lock 과 같은 패턴) --------------
def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    """다른 프로세스가 살아 있으면 False. heartbeat(mtime) 가 stale_sec 이상 멈췄으면 빼앗는다."""
    me = f"{socket.gethostname()}:{os.getpid()}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        host, pid = lock.read_text().strip().split(":")
        alive = False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive = True
            except OSError:
                alive = False
        if alive or (host != socket.gethostname() and time.time() - lock.stat().st_mtime < stale_sec):
            print(f"lock 보유 중: {host}:{pid} (mtime {time.time() - lock.stat().st_mtime:.0f}s 전) -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def training_running(stale_sec: int = 900) -> Path | None:
    for p in TRAIN_PIPELINE_LOCKS:
        if p.exists() and time.time() - p.stat().st_mtime < stale_sec:
            return p
    return None


# --- 1) GPU forward: 특징 추출 -------------------------------------------------
def extract(ckpt: Path, subs: list[str], sources=("rigid", "syn"), force: bool = False) -> dict:
    import torch
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.models.roi_pool import (atlas_on_feature_grid, check_lateralization, global_pool,
                                        roi_names, roi_pool, roi_voxel_counts)
    from atm_sc.training.run import T1_SOURCES, t1_input
    from atm_sc.data.paths import CACHE

    assert torch.cuda.is_available(), "GPU 가 없다. nproc=1 이라 CPU forward 는 subject 당 51s -> 불가"
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    m, sd = from_checkpoint(ckpt, device="cuda")
    assert sd["t1_source"] == "rigid" and int(sd["in_channels"]) == 2 and bool(sd["template"]), sd
    labels, faff = atlas_on_feature_grid(return_affine=True)
    lat = check_lateralization(labels, faff)
    cnt = roi_voxel_counts(labels)
    names = roi_names()
    print(f"atlas on feature grid {labels.shape} | ROI voxel min {cnt.min()} max {cnt.max()} | "
          f"좌우 위반 {lat['n_bad']}", flush=True)

    u = m.atm.net.unet
    t_all = time.time()
    stats = {s: [] for s in sources}
    for n, sub in enumerate(subs):
        for src in sources:
            out = FEAT_DIR / f"{sub}_{src}.npz"
            if out.exists() and not force:
                continue
            x = t1_input(m, sub, src)                                        # [1,2,193,229,193] CPU
            assert x.shape == (1, 2, 193, 229, 193) and x.dtype == torch.float32, x.shape
            assert torch.isfinite(x).all(), f"{sub}/{src}: 입력에 NaN/Inf"
            assert float(x[0, 0].abs().max()) > 0 and float(x[0, 1].abs().max()) > 0, f"{sub}/{src}: 채널이 전부 0"
            stats[src].append((float(x[0, 0].min()), float(x[0, 0].max()), float(x[0, 0].mean()),
                               float(x[0, 1].mean())))
            with torch.no_grad():
                o3 = m.atm._unet_stage3(u, x.to(m.device))                   # [1,256,49,58,49]
                assert tuple(o3.shape) == (1, 256, 49, 58, 49), o3.shape
                f_roi = roi_pool(o3, labels).float().cpu().numpy()           # [82,256] (내부 assert 포함)
                g3 = global_pool(o3).float().cpu().numpy()                   # [256]
                a512 = m.atm.encode_anatomy(x).squeeze(0).float().cpu().numpy()   # conv4+GAP+fc
                del o3
            assert f_roi.shape == (82, 256) and np.isfinite(f_roi).all()
            assert (np.abs(f_roi).sum(1) > 0).all(), f"{sub}/{src}: 값이 전부 0 인 ROI 행이 있다"
            assert a512.shape == (512,) and np.isfinite(a512).all()

            raw = np.load(CACHE / f"{sub}{T1_SOURCES[src][0]}")              # 머리 크기 대조군용
            head = np.array([float((raw > 0).sum()),                         # 뇌 부피 (voxel)
                             float(x[0, 0].sum()),                           # 정규화 T1 총 강도
                             float((x[0, 1] > 0.5).sum())], np.float64)      # WM 부피
            assert np.isfinite(head).all() and (head > 0).all(), (sub, src, head)
            np.savez(out, f_roi=f_roi, g3=g3, a512=a512, head=head)
            del x, raw
            torch.cuda.empty_cache()
        if (n + 1) % 25 == 0 or n == len(subs) - 1:
            GPU_LOCK.touch()                                                 # heartbeat
            print(f"  [{n + 1}/{len(subs)}] {time.time() - t_all:.0f}s", flush=True)

    dist = {}
    for src, v in stats.items():
        if v:
            a = np.asarray(v)
            dist[src] = {"n": len(v), "ch0_min": round(float(a[:, 0].mean()), 4),
                         "ch0_max_mean": round(float(a[:, 1].mean()), 4),
                         "ch0_max_over1_frac": round(float((a[:, 1] > 1.0 + 1e-6).mean()), 4),
                         "ch0_mean": round(float(a[:, 2].mean()), 4),
                         "ch1_mean": round(float(a[:, 3].mean()), 4)}
    return {"lateralization": lat, "roi_voxel_count_min": int(cnt.min()),
            "roi_voxel_count_max": int(cnt.max()),
            "weak_rois": {n: int(cnt[i]) for i, n in enumerate(names) if n in WEAK_ROI_NAMES},
            "input_distribution": dist}


def syn_offdist_check(subs: list[str], n: int = 12) -> dict:
    """SyN 입력이 이 체크포인트와 같은 분포인가. **1차(rigid) 판정과 무관하게 사실만 기록한다.**

    두 가지가 다르다:
      (a) 강도 정규화 — training/run.py 의 T1_SOURCES 가 rigid 는 unit=True([0,1] clip),
          syn 은 unit=False(구 robust) 로 서로 다른 정규화를 건다.
      (b) WM 채널 — scripts/36 이 **rigid** T1 에서만 만든다 (36 번 docstring 명시).
          SyN T1 과 짝지으면 두 채널이 서로 다른 공간이다.
    """
    from atm_sc.data.paths import CACHE
    from atm_sc.data.wm_segment import wm_path
    rows = []
    for sub in subs[:n]:
        r = np.load(CACHE / f"{sub}_T1w_rigid_W.npy").astype(np.float32)
        y = np.load(CACHE / f"{sub}_T1w_syn_W.npy").astype(np.float32)
        w = np.load(wm_path(CACHE, sub)).astype(np.float32)
        rows.append([float((r > 0).sum()), float((y > 0).sum()),
                     float(np.corrcoef(r.ravel(), w.ravel())[0, 1]),
                     float(np.corrcoef(y.ravel(), w.ravel())[0, 1]),
                     float(np.corrcoef(r.ravel(), y.ravel())[0, 1])])
    a = np.asarray(rows)
    return {"n": len(rows),
            "brain_voxels_rigid": round(float(a[:, 0].mean())),
            "brain_voxels_syn": round(float(a[:, 1].mean())),
            "brain_voxel_sd_rigid": round(float(a[:, 0].std())),
            "brain_voxel_sd_syn": round(float(a[:, 1].std())),
            "corr_T1rigid_WM": round(float(a[:, 2].mean()), 4),
            "corr_T1syn_WM": round(float(a[:, 3].mean()), 4),
            "corr_T1rigid_T1syn": round(float(a[:, 4].mean()), 4),
            "note": ("WM 채널은 rigid T1 에서만 만들어진다 (scripts/36). syn 조건은 "
                     "정규화(unit=False)와 WM 채널 공간이 둘 다 체크포인트와 다르다 -> off-distribution")}


# --- 2) 특징 / 타깃 로드 -------------------------------------------------------
def load_feats(subs: list[str], src: str):
    F, A, G, H = [], [], [], []
    for s in subs:
        z = np.load(FEAT_DIR / f"{s}_{src}.npz")
        F.append(z["f_roi"]); A.append(z["a512"]); G.append(z["g3"]); H.append(z["head"])
    F, A, G, H = np.stack(F).astype(np.float64), np.stack(A).astype(np.float64), \
        np.stack(G).astype(np.float64), np.stack(H).astype(np.float64)
    assert F.shape == (len(subs), 82, 256) and np.isfinite(F).all()
    assert A.shape == (len(subs), 512) and G.shape == (len(subs), 256)
    assert H.shape == (len(subs), 3) and np.isfinite(H).all()
    return F, A, G, H


def load_sc(subs: list[str]) -> np.ndarray:
    import scipy.io as sio
    from atm_sc.data.paths import SC_MAT
    m = sio.loadmat(str(SC_MAT))
    by = {str(r["subject"][0]): np.asarray(r["SC_weight"], np.float64) for r in m["data"][0]}
    out = np.stack([by[s] for s in subs])
    assert out.shape == (len(subs), 82, 82) and np.isfinite(out).all()
    assert (out.sum((1, 2)) > 0).all(), "GT SC 가 빈 subject"
    assert np.allclose(out, out.transpose(0, 2, 1)), "GT SC 가 대칭이 아니다"
    return out


# --- 3) PCA (train 에서만 적합) ------------------------------------------------
def pca_train(Xtr: np.ndarray, k: int):
    """Xtr [N,D] (중심화 완료) -> [k,D] 주성분. train 샘플로만 부른다."""
    assert Xtr.ndim == 2 and Xtr.shape[0] > k, Xtr.shape
    _, sv, vt = np.linalg.svd(Xtr, full_matrices=False)
    assert np.isfinite(vt).all()
    return vt[:k], sv[:k]


def roi_scores(F: np.ndarray, tr: np.ndarray, k: int = N_PC):
    """F [S,82,C] -> u [S,82,k]. 중심/스케일/PC 를 **train subject 에서만** 적합한다.

    ROI 별 train 평균으로 중심화한다 (ROI 정체성이 아니라 subject 편차를 PC 가 잡도록).
    """
    mu = F[tr].mean(0)                                    # [82,C]
    sd = F[tr].std((0, 1)) + 1e-8                         # [C]  채널 스케일 (ROI 로 pool)
    Z = (F - mu) / sd
    P, _ = pca_train(Z[tr].reshape(-1, F.shape[2]), k)    # [k,C]
    u = Z @ P.T
    assert u.shape == (F.shape[0], F.shape[1], k) and np.isfinite(u).all()
    return u


def vec_scores(A: np.ndarray, tr: np.ndarray, k: int = N_PC):
    """A [S,D] -> u [S,k] (전뇌 pooling 대조군). train 에서만 적합."""
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-8
    Z = (A - mu) / sd
    k = min(k, len(tr) - 1)
    P, _ = pca_train(Z[tr], k)
    u = Z @ P.T
    assert u.shape == (A.shape[0], k) and np.isfinite(u).all()
    return u


# --- 4) pair 별 ridge (train 내부 5-fold CV 로 alpha 선택) ----------------------
def _design(u: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """ROI 조건: [K,S,3k] = [u[:,i], u[:,j], u[:,i]*u[:,j]]. **타깃을 받지 않는다.**"""
    ui, uj = u[:, pairs[:, 0], :], u[:, pairs[:, 1], :]          # [S,K,k]
    X = np.concatenate([ui, uj, ui * uj], -1).transpose(1, 0, 2)  # [K,S,3k]
    return np.ascontiguousarray(X)


def _fit_predict(Xa, ya, Xb, alphas):
    """Xa [k,n,P] ya [k,n] -> 각 alpha 의 Xb 예측 [n_alpha,k,m]. 절편은 벌하지 않는다."""
    mx, sx = Xa.mean(1, keepdims=True), Xa.std(1, keepdims=True)
    sx = np.where(sx < 1e-12, 1.0, sx)
    my = ya.mean(1, keepdims=True)
    A0 = (Xa - mx) / sx
    y0 = ya - my
    G = np.einsum("knp,knq->kpq", A0, A0)
    b = np.einsum("knp,kn->kp", A0, y0)
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)                                # eigh 의 미세 음수로 1/(w+alpha) 가 터지는 것 방지
    Vb = np.einsum("kpq,kp->kq", V, b)
    B0 = (Xb - mx) / sx
    out = np.empty((len(alphas), Xa.shape[0], Xb.shape[1]))
    for ai, al in enumerate(alphas):
        coef = np.einsum("kpq,kq->kp", V, Vb / (w + al))
        out[ai] = np.einsum("kmp,kp->km", B0, coef) + my
    return out


def _fit_predict_alpha(Xa, ya, Xb, alpha):
    """pair 마다 다른 alpha 로 최종 적합 -> [k,m]."""
    mx, sx = Xa.mean(1, keepdims=True), Xa.std(1, keepdims=True)
    sx = np.where(sx < 1e-12, 1.0, sx)
    my = ya.mean(1, keepdims=True)
    A0, y0 = (Xa - mx) / sx, ya - my
    G = np.einsum("knp,knq->kpq", A0, A0)
    b = np.einsum("knp,kn->kp", A0, y0)
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    Vb = np.einsum("kpq,kp->kq", V, b)
    coef = np.einsum("kpq,kq->kp", V, Vb / (w + alpha[:, None]))
    return np.einsum("kmp,kp->km", (Xb - mx) / sx, coef) + my


def ridge_pair(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               alphas=ALPHAS, n_fold=N_FOLD, seed=SEED_FOLD, chunk=512):
    """X [K,S,P] · Y [K,S] -> (test 예측 [K,len(te)], pair 별 alpha [K]).

    alpha 는 train 내부 5-fold CV 로만 고른다 (test 를 보지 않는다).
    """
    K, S, P = X.shape
    assert Y.shape == (K, S) and len(set(tr) & set(te)) == 0
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(tr)), n_fold)
    Yhat, best = np.empty((K, len(te))), np.empty(K)
    for c0 in range(0, K, chunk):
        sl = slice(c0, min(c0 + chunk, K))
        Xc, Yc = X[sl], Y[sl]
        sse = np.zeros((len(alphas), Xc.shape[0]))
        for f in folds:
            va = tr[f]
            fit = tr[np.setdiff1d(np.arange(len(tr)), f)]
            pred = _fit_predict(Xc[:, fit], Yc[:, fit], Xc[:, va], alphas)
            sse += ((pred - Yc[None, :, va]) ** 2).sum(-1)
        bi = sse.argmin(0)
        al = alphas[bi]
        best[sl] = al
        Yhat[sl] = _fit_predict_alpha(Xc[:, tr], Yc[:, tr], Xc[:, te], al)
    assert np.isfinite(Yhat).all()
    return Yhat, best


# --- 5) 판정 지표 --------------------------------------------------------------
def per_subject_r(Ytrue: np.ndarray, Yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[K,n_te] 두 잔차의 pair 축 상관 -> subject 별 r [n_te]."""
    a, b = Ytrue[mask], Yhat[mask]
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    r = np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1.0), 0.0)
    assert np.isfinite(r).all()
    return r


def boot_ci(r: np.ndarray, n=N_BOOT, seed=SEED_BOOT):
    rng = np.random.default_rng(seed)
    med = np.median(r[rng.integers(0, len(r), (n, len(r)))], axis=1)
    return [round(float(np.percentile(med, 2.5)), 4), round(float(np.percentile(med, 97.5)), 4)]


def summarize(Ytrue, Yhat, masks) -> dict:
    out = {}
    for name, mk in masks.items():
        if mk.sum() < 20:
            continue
        r = per_subject_r(Ytrue, Yhat, mk)
        out[name] = {"n_pairs": int(mk.sum()), "median_r": round(float(np.median(r)), 4),
                     "mean_r": round(float(r.mean()), 4), "ci95": boot_ci(r),
                     "frac_pos": round(float((r > 0).mean()), 3)}
        if name == "all":
            out[name]["per_subject_r"] = [round(float(v), 4) for v in r]
    return out


# --- 6) probe 전체 -------------------------------------------------------------
def run_probe(train: list[str], test: list[str], sources: list[str], extract_meta: dict) -> dict:
    from atm_sc.data.roi_groups import TIER_EDGES
    from atm_sc.models.roi_pool import roi_names

    subs = train + test
    assert len(set(train) & set(test)) == 0, "train/test 겹침"
    assert len(subs) == len(set(subs))
    S = len(subs)
    tr = np.arange(len(train)); te = np.arange(len(train), S)

    SC = load_sc(subs)
    iu = np.triu_indices(82, 1)
    pairs = np.stack(iu, 1)
    K = len(pairs)
    lin = SC[:, iu[0], iu[1]]                                   # [S,K]
    lg = np.log1p(lin)

    # 타깃: 잔차 = SC_s - template_train. template 은 **train 144명 평균 하나로 고정**.
    tmpl_lin, tmpl_log = lin[tr].mean(0), lg[tr].mean(0)
    assert not np.allclose(tmpl_lin, lin[te].mean(0)), "template 이 test 를 봤을 가능성"
    Y = {"log": (lg - tmpl_log).T.copy(), "linear": (lin - tmpl_lin).T.copy()}   # [K,S]
    for v in Y.values():
        assert v.shape == (K, S) and np.isfinite(v).all()
        assert abs(float(v[:, tr].mean())) < 1e-6 * float(np.abs(v).mean()), "train 잔차 평균이 0 이 아니다 (중심화 오류)"

    # pair 부분집합: tier 는 **train 평균** 으로 정의 (test 를 보지 않는다)
    t = tmpl_lin
    names = roi_names()
    weak = np.array([i for i, n in enumerate(names) if n in WEAK_ROI_NAMES])
    assert len(weak) == len(WEAK_ROI_NAMES), [names[i] for i in weak]
    no_weak = ~np.isin(pairs, weak).any(1)
    masks = {"all": np.ones(K, bool),
             "tier_zero": t <= 0,
             "tier_small": (t > 0) & (t <= TIER_EDGES[0]),
             "tier_mid": (t > TIER_EDGES[0]) & (t <= TIER_EDGES[1]),
             "tier_large": t > TIER_EDGES[1],
             "excl_weak_roi": no_weak,
             "excl_weak_roi_tier_mid": no_weak & (t > TIER_EDGES[0]) & (t <= TIER_EDGES[1]),
             "excl_weak_roi_tier_large": no_weak & (t > TIER_EDGES[1])}

    rng = np.random.default_rng(SEED_SHUFFLE)
    perm = np.concatenate([tr[rng.permutation(len(tr))], te[rng.permutation(len(te))]])
    n_fix = int((perm == np.arange(S)).sum())

    conds: dict[str, np.ndarray] = {}                          # 이름 -> X [K,S,P]
    heads = {}
    for src in sources:
        F, A, G, H = load_feats(subs, src)
        tag = {"rigid": "R", "syn": "S"}[src]
        conds[f"{tag}-ROI"] = _design(roi_scores(F, tr), pairs)
        conds[f"{tag}-GAP"] = np.repeat(vec_scores(A, tr)[None], K, 0)
        conds[f"{tag}-GAP-stage3"] = np.repeat(vec_scores(G, tr)[None], K, 0)
        if src == "rigid":
            conds["R-ROI-shuffled"] = _design(roi_scores(F[perm], tr), pairs)
            conds["R-GAP-shuffled"] = np.repeat(vec_scores(A[perm], tr)[None], K, 0)
            for j, hn in enumerate(("brain_volume", "t1_total_intensity")):
                v = ((H[:, j] - H[tr, j].mean()) / (H[tr, j].std() + 1e-12))[:, None]
                conds[f"head_{hn}"] = np.repeat(v[None], K, 0)
                heads[hn] = [round(float(H[tr, j].mean()), 1), round(float(H[tr, j].std()), 1)]

    # --- 필수 assert: 설계 행렬에 타깃이 들어가지 않았는지 -------------------------
    chk = np.random.default_rng(0).choice(K, 200, replace=False)
    for cn, X in conds.items():
        assert X.shape[0] == K and X.shape[1] == S and np.isfinite(X).all(), (cn, X.shape)
        for sp in ("log", "linear"):
            y = Y[sp][chk][:, tr]                              # [200,144]
            x = X[chk][:, tr, :]                               # [200,144,P]
            xc = x - x.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
            num = np.einsum("knp,kn->kp", xc, yc)
            den = np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1)[:, None]
            r = np.abs(np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0))
            assert r.max() < 0.999, f"{cn}/{sp}: 설계 행렬이 타깃과 |r|={r.max():.4f} — 라벨 누수 의심"

    res = {}
    for cn, X in conds.items():
        res[cn] = {"n_predictors": int(X.shape[2])}
        for sp in ("log", "linear"):
            t0 = time.time()
            Yh, al = ridge_pair(X, Y[sp], tr, te)
            res[cn][sp] = summarize(Y[sp][:, te], Yh, masks)
            res[cn][sp]["alpha_median"] = float(np.median(al))
            print(f"  {cn:22s} {sp:6s} median_r(all) {res[cn][sp]['all']['median_r']:+.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # 사전등록 게이트는 "R-GAP 대비 +0.05" 와 "머리 크기를 넘는가" 라는 **쌍대 비교**다.
    # 독립 CI 겹침이 아니라 subject 별 r 차이로 재야 한다 (같은 31명이므로 짝지어진다).
    deltas = {}
    for sp in ("log", "linear"):
        for a_, b_ in (("R-ROI", "R-GAP"), ("R-ROI", "head_brain_volume"),
                       ("R-ROI", "R-ROI-shuffled"), ("R-GAP", "head_brain_volume"),
                       ("S-ROI", "S-GAP")):
            if a_ not in res or b_ not in res:
                continue
            ra = np.asarray(res[a_][sp]["all"]["per_subject_r"])
            rb = np.asarray(res[b_][sp]["all"]["per_subject_r"])
            d = ra - rb
            deltas[f"{sp}:{a_}-{b_}"] = {"median_delta": round(float(np.median(d)), 4),
                                         "ci95": boot_ci(d), "frac_pos": round(float((d > 0).mean()), 3)}

    return {"conditions": res, "deltas": deltas, "shuffle_fixed_points": n_fix,
            "syn_offdist_check": syn_offdist_check(subs) if "syn" in sources else None,
            "head_train_stats": heads,
            "pair_subsets": {k: int(v.sum()) for k, v in masks.items()},
            "template_train_n": len(train), "n_pairs": K,
            "extract": extract_meta}


def verdict(res: dict) -> str:
    """사전등록 판정 (docs/PIPELINE_10_RESOLUTION_PLAN.md). 결과를 보고 바꾸지 않는다."""
    c = res["conditions"]
    sh = c["R-ROI-shuffled"]["log"]["all"]
    if abs(sh["median_r"]) > 0.02 or sh["ci95"][0] > 0 or sh["ci95"][1] < 0:
        return (f"판정 보류 — 셔플 대조군 r={sh['median_r']:+.4f} CI{sh['ci95']} 가 0 에서 벗어났다. "
                "누수를 먼저 잡아야 한다.")
    roi = c["R-ROI"]["log"]["all"]["median_r"]
    gap = c["R-GAP"]["log"]["all"]["median_r"]
    head = max(c[f"head_{h}"]["log"]["all"]["median_r"] for h in ("brain_volume", "t1_total_intensity"))
    dg = res["deltas"]["log:R-ROI-R-GAP"]
    dh = res["deltas"]["log:R-ROI-head_brain_volume"]
    band = ("(a) >=0.15" if roi >= 0.15 else "(b) 0.05~0.15 약한 신호" if roi >= 0.05 else "(c) <0.05")
    base = (f"R-ROI(log,all) r={roi:+.4f} -> 사전등록 눈금 {band}. "
            f"R-GAP r={gap:+.4f} (짝지은 차 {dg['median_delta']:+.4f} CI{dg['ci95']}), "
            f"머리 크기 단일 회귀 r={head:+.4f} (짝지은 차 {dh['median_delta']:+.4f} CI{dh['ci95']}). ")
    if roi >= 0.15 and roi - gap >= 0.05:
        return base + "게이트 통과 -> S2 진행."
    fails = []
    if roi - gap < 0.05:
        fails.append(f"R-GAP 대비 +0.05 미달({roi - gap:+.4f})")
    if roi <= head:
        fails.append(f"필수 대조군 3(머리 크기)을 넘지 못함({roi:+.4f} <= {head:+.4f}) -> 크기 효과")
    tail = " / ".join(fails)
    if roi >= 0.05:
        return base + (f"게이트 미통과: {tail}. 눈금상 (b) 약한 신호이나 그 신호가 **ROI 국소 풀링에서 온 것이 아니고**"
                       " 머리 크기를 넘지도 못한다 -> S1 이 물은 질문('국소 풀링이 전뇌 풀링이 지우는 신호를"
                       " 살리는가')의 답은 아니다.")
    return base + f"사전등록 기준: <0.05 -> A1/A2 중단, 축 B 에 집중. ({tail})"


def main(a):
    train = [s.strip() for s in open(ROOT / "outputs/splits/train.txt") if s.strip()]
    test = [s.strip() for s in open(ROOT / "outputs/splits/test.txt") if s.strip()]
    assert len(train) == 144 and len(test) == 31 and not set(train) & set(test)
    sources = a.sources.split(",")
    meta = {}
    if a.stage in ("all", "extract"):
        held = training_running()
        assert held is None, f"학습 파이프라인이 {held} lock 을 잡고 있다 -> GPU 충돌"
        if not acquire_lock(GPU_LOCK):
            return 3
        try:
            meta = extract(Path(a.ckpt), train + test, tuple(sources), force=a.force)
        finally:
            if GPU_LOCK.exists() and GPU_LOCK.read_text().strip().endswith(f":{os.getpid()}"):
                GPU_LOCK.unlink()
        (FEAT_DIR / "extract_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if a.stage in ("all", "probe"):
        if not meta:
            p = FEAT_DIR / "extract_meta.json"
            meta = json.loads(p.read_text()) if p.exists() else {}
        out = run_probe(train, test, sources, meta)
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        except OSError:
            sha = ""
        out["meta"] = {"ckpt": str(Path(a.ckpt).relative_to(ROOT)), "n_train": len(train),
                       "n_test": len(test), "train": train, "test": test, "sources": sources,
                       "n_pc": N_PC, "n_fold": N_FOLD, "alphas": [float(ALPHAS[0]), float(ALPHAS[-1]), len(ALPHAS)],
                       "seeds": {"shuffle": SEED_SHUFFLE, "fold": SEED_FOLD, "boot": SEED_BOOT},
                       "n_boot": N_BOOT, "weak_rois": list(WEAK_ROI_NAMES),
                       "git": sha, "date": time.strftime("%Y-%m-%d %H:%M"),
                       "cmd": "python scripts/42_space_probe.py --stage extract && python scripts/42_space_probe.py --stage probe"}
        out["verdict"] = verdict(out)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print("\n" + out["verdict"] + f"\n-> {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "extract", "probe"))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--sources", default="rigid,syn")
    ap.add_argument("--force", action="store_true", help="특징 캐시를 다시 계산")
    sys.exit(main(ap.parse_args()))
