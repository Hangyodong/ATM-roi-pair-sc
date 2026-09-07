#!/usr/bin/env python
"""티어 1 명시적 해부 feature 추출 + 프로브.

  python scripts/55_anat_tier1.py --stage extract          # 206명 (CPU)
  python scripts/55_anat_tier1.py --stage probe            # GT SC 잔차 예측력

학습된 feature 가 아니라 subject 볼륨에서 직접 계산한 양이다 (ROI 조직량, 경계 대비,
WM 가중 centroid 거리). 근거: 지금까지 유의했던 유일한 개인 신호가 머리 크기라는 명시적
스칼라 하나였다 (S1-b 0.102 / W4-a 0.101).

프로브는 W4 와 **같은 규약**이다 -- subject 단위 5-fold, pair 평균은 훈련 fold 로만,
순열검정 200회. 그래야 기존 표(전역 anatomy 0.101 / 끝점 0.053 / 경로 −0.007)에 나란히 놓을 수 있다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data import anat_tier1 as T1F                                # noqa: E402
from atm_sc.data.dataset import ROIPairSubject                           # noqa: E402
from atm_sc.data.paths import CACHE                                      # noqa: E402
from atm_sc.models.roi_pool import atlas_on_feature_grid                 # noqa: E402

EVAL = ROOT / "outputs/eval"
W_SHAPE = (193, 229, 193)


def subjects_all() -> list[str]:
    out = []
    for sp in ("train", "val", "test"):
        out += [l.strip() for l in (ROOT / f"outputs/splits/{sp}.txt").read_text().splitlines() if l.strip()]
    return out


def stage_extract(a):
    from atm_sc.spaces import W_AFFINE
    labels, aff = atlas_on_feature_grid(feat_shape=W_SHAPE, in_shape=W_SHAPE,
                                        in_affine=W_AFFINE, return_affine=True)
    assert labels.shape == W_SHAPE, labels.shape
    print(f"아틀라스 W 격자 {labels.shape}, ROI {len(np.unique(labels)) - 1}개", flush=True)
    T1F.OUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = [s for s in subjects_all() if (CACHE / f"{s}_T1w_{a.source}_W.npy").exists()
            and (CACHE / f"{s}_WM_W.npy").exists()]
    print(f"대상 {len(subs)}명", flush=True)
    t0, done = time.time(), []
    for k, s in enumerate(subs):
        p = T1F.OUT_DIR / f"{s}_{a.source}.npz"
        if p.exists() and not a.force:
            done.append(s); continue
        f = T1F.build_subject(s, labels, aff, a.source)
        np.savez_compressed(p, **f)
        done.append(s)
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(subs)}  {el:.0f}s  (남은 {el/(k+1)*(len(subs)-k-1):.0f}s)", flush=True)
    feats = [T1F.load(s, a.source) for s in done]
    rep = T1F.assert_subject_varying(feats)
    print("subject 변동 (CV):", json.dumps({k: round(v, 5) for k, v in rep.items()}, ensure_ascii=False), flush=True)
    (EVAL / "anat_tier1_extract.json").write_text(json.dumps(
        {"n": len(done), "source": a.source, "subject_cv": rep}, ensure_ascii=False, indent=2))
    return done


def _ridge_cv(X, Y, subs_idx=None, lams=(1e-1, 1, 10, 100, 1e3, 1e4), k=5, seed=0):
    """subject 단위 k-fold. pair 평균은 **훈련 fold 로만** 계산한다 (누수 방지). -> 잔차 상관.

    **쌍대(dual) 형태로 푼다.** 표본 206명 << 특징 29,889 이라 X^T X (29889^2) 를 만들면
    7GB 에 O(p^3) 이라 사실상 끝나지 않는다. n < p 일 때는
        W = X^T (X X^T + lam I)^-1 Y   -> 206x206 역행렬
    이 수학적으로 동일하고 비교가 안 되게 싸다.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % k
    best = (None, -9)
    for lam in lams:
        preds = np.zeros_like(Y)
        for f in range(k):
            tr, te = fold != f, fold == f
            mx, my = X[tr].mean(0), Y[tr].mean(0)            # pair 평균 = 훈련 fold 평균
            Xt, Yt = X[tr] - mx, Y[tr] - my
            K = Xt @ Xt.T + lam * np.eye(Xt.shape[0])        # [n_tr, n_tr] -- 쌍대
            A = np.linalg.solve(K, Yt)
            preds[te] = ((X[te] - mx) @ Xt.T) @ A
        # 채점: 훈련 fold 평균을 뺀 GT 잔차와의 상관
        gt = np.zeros_like(Y)
        for f in range(k):
            tr, te = fold != f, fold == f
            gt[te] = Y[te] - Y[tr].mean(0)
        r = float(np.corrcoef(preds.ravel(), gt.ravel())[0, 1])
        if r > best[1]:
            best = (lam, r)
    return best


def stage_probe(a):
    subs = [s for s in subjects_all() if (T1F.OUT_DIR / f"{s}_{a.source}.npz").exists()]
    print(f"프로브 대상 {len(subs)}명", flush=True)
    assert len(subs) >= 100, len(subs)
    iu = np.triu_indices(T1F.N_ROI, 1)
    pairs = np.stack(iu, 1)
    X, Y = [], []
    for s in subs:
        f = T1F.load(s, a.source)
        X.append(T1F.pair_features(f, pairs).ravel())         # [K*9]
        Y.append(np.log1p(np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu]))
    X = np.stack(X).astype(np.float64); Y = np.stack(Y)
    print("X", X.shape, "Y", Y.shape, flush=True)
    assert np.isfinite(X).all() and np.isfinite(Y).all()

    # pair 별로 9차 feature -> 그 pair 의 log1p SC. 모든 pair 를 공유 가중치로 (W4-b 와 같은 규약)
    K = pairs.shape[0]
    Xp = X.reshape(len(subs), K, T1F.PAIR_DIM)
    res = {}
    # (a) 공유 가중치 프로브: [N*K, 9] -> [N*K]  (pair 평균은 훈련 fold 로 제거)
    lam, r = _ridge_cv(Xp.reshape(len(subs), -1), Y, None)
    res["tier1_pair9_flat"] = {"lam": lam, "r": r}
    # (b) 전역 스칼라만 (머리 크기 재현 확인)
    G = np.stack([[float(T1F.load(s, a.source)["brain_vol"]),
                   float(T1F.load(s, a.source)["t1_total"]),
                   float(T1F.load(s, a.source)["wm_total"])] for s in subs])
    lam, r = _ridge_cv(G, Y, None)
    res["global_size3"] = {"lam": lam, "r": r}
    # (c) 순열검정: subject 라벨을 섞는다
    rng = np.random.default_rng(0)
    null = []
    Xf = Xp.reshape(len(subs), -1)
    for _ in range(a.n_perm):
        _, rn = _ridge_cv(Xf, Y[rng.permutation(len(subs))], None, lams=(res["tier1_pair9_flat"]["lam"],))
        null.append(rn)
    null = np.array(null)
    res["permutation"] = {"n": a.n_perm, "mean": float(null.mean()), "sd": float(null.std()),
                          "q95": float(np.percentile(null, 95)),
                          "p": float((null >= res["tier1_pair9_flat"]["r"]).mean())}
    res["n_subjects"] = len(subs)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    (EVAL / "anat_tier1_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="extract", choices=("extract", "probe", "all"))
    ap.add_argument("--source", default="rigid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-perm", type=int, default=50)
    a = ap.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    if a.stage in ("extract", "all"):
        stage_extract(a)
    if a.stage in ("probe", "all"):
        stage_probe(a)


if __name__ == "__main__":
    main()
