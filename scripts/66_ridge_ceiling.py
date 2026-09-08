"""T1 로 SC 개인차를 어디까지 예측할 수 있나 — 선형 상한 (학습 없이, CPU).

count head 의 val resid_r 은 0.086 이다. 그것이 모델 한계인지 데이터 한계인지 모르면
어디를 고쳐야 할지 알 수 없다. 여기서는 **캐시된 해부 feature + 능선회귀**로 상한을 잰다.
λ 는 train 144명 안의 K-fold 로만 고른다 (val 로 고르면 상한이 부풀려진다).

feature
  tier1  : pair 당 9개 자로 잰 값 (ROI 부피/centroid 거리/corridor). `data/anat_tier1.py`
  local  : ROI 국소 anatomy [82,512] 를 ROI 별 PCA 로 줄여 pair 마다 concat. `data/local_feats.py`
목표는 count head 와 같다: y = log1p(SC) - train 템플릿(log1p).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.anat_tier1 import load as load_tier1, pair_features
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.local_feats import roi_feature_path
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr


def per_pair_ridge(X, Y, lam):
    """pair 마다 독립 능선회귀. X [N,P,F] (절편 포함), Y [N,P] -> W [P,F]."""
    N, P, F = X.shape
    G = np.einsum("npf,npg->pfg", X, X) + lam * np.eye(F)[None]
    b = np.einsum("npf,np->pf", X, Y)
    return np.linalg.solve(G, b[..., None])[..., 0]


def cv_lambda(X, Y, lams, k=4, seed=0):
    """train 안에서만 K-fold. pair 전체의 잔차 상관(중심화 후)을 기준으로 고른다."""
    rng = np.random.default_rng(seed); N = len(X)
    fold = rng.permutation(N) % k
    best, best_r = None, -np.inf
    for lam in lams:
        pr = np.zeros_like(Y)
        for f in range(k):
            tr, te = fold != f, fold == f
            W = per_pair_ridge(X[tr], Y[tr], lam)
            pr[te] = np.einsum("npf,pf->np", X[te], W)
        r = float(np.mean([_corr(loo_center(pr)[i], loo_center(Y)[i]) for i in range(N)]))
        if r > best_r:
            best, best_r = lam, r
    return best, best_r


def main(a):
    tr = [l.strip() for l in (ROOT / "outputs/splits/train.txt").read_text().splitlines() if l.strip()]
    va = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()]
    tr = [s for s in tr if roi_feature_path(s, "rigid").exists()]
    va = [s for s in va if roi_feature_path(s, "rigid").exists()]
    assert len(tr) > 100 and len(va) > 20, (len(tr), len(va))
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); tmpl = np.asarray(z["template"], np.float64)
    assert int(z["n_subjects"]) == 144, "템플릿이 train 144명이 아니다"
    pairs = np.stack(iu, 1)

    def sc_of(subs):
        return np.stack([np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu] for s in subs])
    t0 = time.time()
    SC = {"tr": sc_of(tr), "va": sc_of(va)}
    Y = {k: np.log1p(v) - tmpl[None] for k, v in SC.items()}

    def tier1_of(subs):
        return np.stack([pair_features(load_tier1(s, "rigid"), pairs) for s in subs]).astype(np.float64)
    T1 = {"tr": tier1_of(tr), "va": tier1_of(va)}
    mu, sd = T1["tr"].mean(0), T1["tr"].std(0) + 1e-6
    T1 = {k: (v - mu) / sd for k, v in T1.items()}

    def roi_of(subs):
        return np.stack([np.asarray(np.load(roi_feature_path(s, "rigid"))["f_roi"], np.float64) for s in subs])
    Fr = {"tr": roi_of(tr), "va": roi_of(va)}
    rm = Fr["tr"].mean(0)                                    # ROI 별 평균 (train 만)
    Z = {}
    q = a.pca
    comp = np.zeros((R, Fr["tr"].shape[2], q))
    for r in range(R):
        M = Fr["tr"][:, r] - rm[r]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        comp[r] = Vt[:q].T / (S[:q] / np.sqrt(len(M)) + 1e-9)     # 표준화된 성분 점수
    for k in ("tr", "va"):
        Z[k] = np.einsum("nrd,rdq->nrq", Fr[k] - rm[None], comp)
    LOC = {k: np.concatenate([Z[k][:, pairs[:, 0]], Z[k][:, pairs[:, 1]]], -1) for k in ("tr", "va")}

    def loc_q(k, qq):
        return np.concatenate([Z[k][:, pairs[:, 0], :qq], Z[k][:, pairs[:, 1], :qq]], -1)
    sets = {"tier1": lambda k: T1[k]}
    for qq in a.pca_sweep:
        sets[f"local{qq}"] = (lambda qq: (lambda k: loc_q(k, qq)))(qq)
        sets[f"tier1+local{qq}"] = (lambda qq: (lambda k: np.concatenate([T1[k], loc_q(k, qq)], -1)))(qq)
    out = {"n_train": len(tr), "n_val": len(va), "pca": q, "load_sec": round(time.time() - t0, 1)}
    for name, fn in sets.items():
        Xtr, Xva = fn("tr"), fn("va")
        one = lambda X: np.concatenate([X, np.ones(X.shape[:2] + (1,))], -1)
        Xtr, Xva = one(Xtr), one(Xva)
        lam, cvr = cv_lambda(Xtr, Y["tr"], a.lams)
        W = per_pair_ridge(Xtr, Y["tr"], lam)
        pv = np.einsum("npf,pf->np", Xva, W)
        if a.shared:      # 공유(pair 무관) 가중치도 같이 -- pair 별 모델이 과적합인지 본다
            Gs = np.einsum("npf,npg->fg", Xtr, Xtr) + lam * np.eye(Xtr.shape[2])
            ws = np.linalg.solve(Gs, np.einsum("npf,np->f", Xtr, Y["tr"]))
            pv_s = np.einsum("npf,f->np", Xva, ws)
            out[name + "_shared"] = {"val_resid_r": round(subject_specificity(
                np.maximum(np.expm1(tmpl[None] + pv_s), 0.0), SC["va"])["resid_r"], 4)}
        pred_lin = np.expm1(tmpl[None] + pv)
        sp = subject_specificity(np.maximum(pred_lin, 0.0), SC["va"])
        out[name] = {"lam": float(lam), "cv_resid_r_train": round(cvr, 4),
                     "val_resid_r_log": round(float(np.mean([_corr(loo_center(pv)[i], loo_center(Y["va"])[i])
                                                             for i in range(len(va))])), 4),
                     "val_resid_r": round(sp["resid_r"], 4),
                     "val_inter_subj_r": round(sp["inter_subj_r_pred"], 4),
                     "val_abs_r": round(float(np.mean([_corr(pred_lin[i], SC["va"][i]) for i in range(len(va))])), 4),
                     "n_feat": int(Xtr.shape[2])}
        print(name, json.dumps(out[name]), flush=True)
    # 재현성 상한: GT 자신의 반쪽 (pair 를 반으로 갈라 서로 예측) 은 못 하므로, 대신
    # "템플릿만" 기준선을 같이 낸다 (개인차 0 -> resid_r 0).
    out["gt_inter_subj_r"] = round(float(subject_specificity(SC["va"], SC["va"])["inter_subj_r_gt"]), 4)
    (ROOT / "outputs/eval/ridge_ceiling.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--pca-sweep", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--shared", action="store_true", default=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    main(ap.parse_args())
