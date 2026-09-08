"""feature 블록별 T1->SC 잔차 예측 상한 (pair 별 능선회귀, train 적합 / val 보고).

블록을 켜고 끄며 "이 데이터를 추출해서 쓰면 얼마나 오르나" 를 숫자로 답한다.
λ 는 train 144명 안의 K-fold 로만 고른다.

  tier1     자로 잰 ROI feature 에서 만든 pair feature 9개        (기존)
  corridor  두 centroid 를 잇는 선 위의 WM 확률 7개               (신규, 기존 WM 맵만)
  surf      ROI 별 WM 경계 면적 3개                               (신규, marching cubes)
  tissue    GM/CSF 부피·위축 6개                                  (신규, Atropos 재실행)
  local{q}  인코더 ROI 국소 feature PCA q개 x 2                    (기존)
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data import anat_corridor as AC
from atm_sc.data.anat_tier1 import load as load_tier1, pair_features
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.local_feats import roi_feature_path
from atm_sc.data.paths import CACHE
from atm_sc.data.wm_segment import tissue_path
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr


def per_pair_ridge(X, Y, lam):
    F = X.shape[2]
    G = np.einsum("npf,npg->pfg", X, X) + lam * np.eye(F)[None]
    return np.linalg.solve(G, np.einsum("npf,np->pf", X, Y)[..., None])[..., 0]


def cv_lambda(X, Y, lams, k=4, seed=0):
    rng = np.random.default_rng(seed); fold = rng.permutation(len(X)) % k
    best, best_r = None, -np.inf
    for lam in lams:
        pr = np.zeros_like(Y)
        for f in range(k):
            m = fold != f
            pr[~m] = np.einsum("npf,pf->np", X[~m], per_pair_ridge(X[m], Y[m], lam))
        r = float(np.mean([_corr(loo_center(pr)[i], loo_center(Y)[i]) for i in range(len(X))]))
        if r > best_r:
            best, best_r = lam, r
    return best, best_r


def tissue_roi_feats(sub, lab_flat, nvox, n_roi, source="rigid"):
    """[R, 4] ROI 별 GM 부피(log)·GM 비중·CSF 비중·GM/WM 비. bincount 한 번으로 집계한다."""
    p = tissue_path(CACHE, sub)
    if not p.exists():
        return None
    z = np.load(p)
    g, c, w = (z[k].astype(np.float32).ravel() / 255.0 for k in ("gm", "csf", "wm"))
    agg = lambda v: np.bincount(lab_flat, weights=v, minlength=n_roi + 1)[1:]
    gs, cs, ws = agg(g), agg(c), agg(w)
    return np.stack([np.log1p(gs), gs / nvox, cs / nvox,
                     np.log((gs + 1.0) / (ws + 1.0))], 1).astype(np.float64)


def main(a):
    from atm_sc.models.roi_pool import atlas_on_feature_grid
    from atm_sc.spaces import W_AFFINE, W_SHAPE
    S = {k: [l.strip() for l in (ROOT / f"outputs/splits/{k}.txt").read_text().splitlines() if l.strip()]
         for k in ("train", "val")}
    S = {k: [s for s in v if roi_feature_path(s, a.source).exists()] for k, v in S.items()}
    assert len(S["train"]) == 144, len(S["train"])
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); tmpl = np.asarray(z["template"], np.float64)
    pairs = np.stack(iu, 1)
    t0 = time.time()
    SC = {k: np.stack([np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu] for s in v]) for k, v in S.items()}
    Y = {k: np.log1p(v) - tmpl[None] for k, v in SC.items()}
    B = {k: {} for k in S}
    for k, subs in S.items():
        B[k]["tier1"] = np.stack([pair_features(load_tier1(s, a.source), pairs) for s in subs]).astype(np.float64)
        B[k]["corridor"] = np.stack([AC.load(s, a.source)["corridor"] for s in subs]).astype(np.float64)
        ar = np.stack([AC.load(s, a.source)["wm_area"] for s in subs]).astype(np.float64)
        la = np.log1p(ar)
        B[k]["surf"] = np.stack([la[:, pairs[:, 0]], la[:, pairs[:, 1]],
                                 0.5 * (la[:, pairs[:, 0]] + la[:, pairs[:, 1]])], -1)
        B[k]["_roi"] = np.stack([np.asarray(np.load(roi_feature_path(s, a.source))["f_roi"], np.float64)
                                 for s in subs])
    miss = [s for v in S.values() for s in v if not tissue_path(CACHE, s).exists()]
    have_tissue = not miss
    if miss:
        print(f"tissue 없는 subject {len(miss)}명: {miss[:5]} -> tissue 블록 제외", flush=True)
    if have_tissue:
        labels, _ = atlas_on_feature_grid(feat_shape=W_SHAPE, in_shape=W_SHAPE,
                                          in_affine=W_AFFINE, return_affine=True)
        lab_flat = np.asarray(labels, np.int64).ravel()
        nvox = np.maximum(np.bincount(lab_flat, minlength=R + 1)[1:], 1.0)
        for k, subs in S.items():
            tr_ = np.stack([tissue_roi_feats(s, lab_flat, nvox, R, a.source) for s in subs])
            B[k]["tissue"] = np.concatenate([tr_[:, pairs[:, 0]], tr_[:, pairs[:, 1]]], -1)
    print(f"feature 적재 {time.time()-t0:.0f}s  tissue={'있음' if have_tissue else '없음'}", flush=True)
    # 표준화는 train 통계로만
    for name in [n for n in B["train"] if not n.startswith("_")]:
        mu, sd = B["train"][name].mean(0), B["train"][name].std(0) + 1e-6
        for k in B:
            B[k][name] = (B[k][name] - mu) / sd
    rm = B["train"]["_roi"].mean(0); q = a.pca
    comp = np.zeros((R, B["train"]["_roi"].shape[2], q))
    for r in range(R):
        M = B["train"]["_roi"][:, r] - rm[r]
        _, Sv, Vt = np.linalg.svd(M, full_matrices=False)
        comp[r] = Vt[:q].T / (Sv[:q] / np.sqrt(len(M)) + 1e-9)
    for k in B:
        Zc = np.einsum("nrd,rdq->nrq", B[k]["_roi"] - rm[None], comp)
        B[k][f"local{q}"] = np.concatenate([Zc[:, pairs[:, 0]], Zc[:, pairs[:, 1]]], -1)
        B[k].pop("_roi")
    names = [n for n in ["tier1", "corridor", "surf", "tissue", f"local{q}"] if n in B["train"]]
    combos = [(n,) for n in names] + a.extra_combos + [tuple(names)]
    seen, out = set(), {"n_train": len(S["train"]), "n_val": len(S["val"]), "pca": q,
                        "have_tissue": have_tissue}
    for cb in combos:
        cb = tuple(n for n in cb if n in B["train"])
        if not cb or cb in seen:
            continue
        seen.add(cb)
        X = {k: np.concatenate([B[k][n] for n in cb] + [np.ones(Y[k].shape + (1,))], -1) for k in B}
        lam, cvr = cv_lambda(X["train"], Y["train"], a.lams)
        W = per_pair_ridge(X["train"], Y["train"], lam)
        pv = np.einsum("npf,pf->np", X["val"], W)
        lin = np.maximum(np.expm1(tmpl[None] + pv), 0.0)
        sp = subject_specificity(lin, SC["val"])
        key = "+".join(cb)
        out[key] = {"lam": float(lam), "cv_train": round(cvr, 4), "val_resid_r": round(sp["resid_r"], 4),
                    "val_resid_r_log": round(float(np.mean([_corr(loo_center(pv)[i], loo_center(Y["val"])[i])
                                                            for i in range(len(pv))])), 4),
                    "n_feat": int(X["train"].shape[2])}
        print(f"{key:40s} {json.dumps(out[key])}", flush=True)
    (ROOT / f"outputs/eval/{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rigid"); ap.add_argument("--pca", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--tag", default="feature_ceiling")
    ap.add_argument("--extra-combos", nargs="*", default=[])
    a = ap.parse_args()
    a.extra_combos = [tuple(x.split("+")) for x in a.extra_combos] or [
        ("tier1", "corridor"), ("tier1", "corridor", "surf"),
        ("tier1", "corridor", "surf", "tissue"), ("tier1", "local16"), ("corridor", "local16")]
    main(a)
