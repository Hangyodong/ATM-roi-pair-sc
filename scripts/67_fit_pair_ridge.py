"""pair 별 능선 head 적합 (train 144명 전용) -> outputs/cache/pair_ridge_rigid.npz"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.anat_tier1 import load as load_tier1, pair_features
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.local_feats import roi_feature_path
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr
from atm_sc.models.pair_ridge import WEIGHTS, _pair_list, raw_blocks


def per_pair_ridge(X, Y, lam):
    F = X.shape[2]
    G = np.einsum("npf,npg->pfg", X, X) + lam * np.eye(F)[None]
    return np.linalg.solve(G, np.einsum("npf,np->pf", X, Y)[..., None])[..., 0]


def main(a):
    tr = [l.strip() for l in (ROOT / "outputs/splits/train.txt").read_text().splitlines() if l.strip()]
    tr = [s for s in tr if roi_feature_path(s, a.source).exists()]
    assert len(tr) == 144, f"train 이 144명이 아니다 ({len(tr)})"
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); tmpl = np.asarray(z["template"], np.float64)
    assert int(z["n_subjects"]) == 144
    pairs = _pair_list(R)
    Y = np.stack([np.log1p(np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu]) for s in tr]) - tmpl[None]
    blocks = [b for b in a.blocks if b]
    B = {}
    for b in blocks:
        B[b] = np.stack([raw_blocks(s, R, a.source)[b] for s in tr])
        print(f"블록 {b}: {B[b].shape[2]}개", flush=True)
    stats = {}
    for b in blocks:
        mu_ = B[b].mean(0)
        # 표준편차 하한을 **feature 전체 규모의 5 %** 로 둔다. pair 별로 표준화하는데 train 에서
        # 분산이 0 에 가까운 pair-feature 가 있으면(예: corridor 최솟값이 늘 0 인 pair) 1e-6 으로
        # 나뉘어 폭발한다 -- 실측: val 표준화 값 최대 36789, 예측 잔차 std 0.178(train) vs 1.897(val),
        # resid_r 0.147 -> 0.0002. 조용히 숫자만 이상해지는 종류의 실패다.
        sd_ = np.maximum(B[b].std(0), 0.05 * (B[b].std(axis=(0, 1)) + 1e-9)[None])
        stats[f"mu_{b}"], stats[f"sd_{b}"] = mu_, sd_
        B[b] = (B[b] - mu_) / sd_
        assert np.abs(B[b]).max() < 50, f"{b}: 표준화 후 값이 너무 크다 ({np.abs(B[b]).max():.1f})"
    F = np.stack([np.asarray(np.load(roi_feature_path(s, a.source))["f_roi"], np.float64) for s in tr])
    rm = F.mean(0)
    comp = np.zeros((R, F.shape[2], a.pca))
    for r in range(R):
        M = F[:, r] - rm[r]
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        comp[r] = Vt[:a.pca].T / (S[:a.pca] / np.sqrt(len(M)) + 1e-9)
    Zc = np.einsum("nrd,rdq->nrq", F - rm[None], comp)
    X = np.concatenate([B[b] for b in blocks] +
                       [Zc[:, pairs[:, 0]], Zc[:, pairs[:, 1]], np.ones((len(tr), len(pairs), 1))], -1)
    assert np.isfinite(X).all()
    rng = np.random.default_rng(0); fold = rng.permutation(len(tr)) % a.folds
    best, best_r, curve = None, -np.inf, {}
    for lam in a.lams:
        pr = np.zeros_like(Y)
        for f in range(a.folds):
            m = fold != f
            pr[~m] = np.einsum("npf,pf->np", X[~m], per_pair_ridge(X[m], Y[m], lam))
        r = float(np.mean([_corr(loo_center(pr)[i], loo_center(Y)[i]) for i in range(len(tr))]))
        curve[str(lam)] = round(r, 4)
        if r > best_r:
            best, best_r = lam, r
    W = per_pair_ridge(X, Y, best)
    # 잔차 진폭: 예측 잔차가 GT 잔차보다 얼마나 작은가 (alpha 기본값 근거)
    pr_tr = np.einsum("npf,pf->np", X, W)
    amp = float(loo_center(Y).std() / max(loo_center(pr_tr).std(), 1e-9))
    p = Path(str(WEIGHTS).format(source=a.source))
    np.savez_compressed(p, W=W, comp=comp, roi_mean=rm, template=tmpl,
                        blocks=np.array(blocks), n_roi=np.int64(R), n_train=np.int64(len(tr)),
                        lam=np.float64(best), pca=np.int64(a.pca), amp_train=np.float64(amp), **stats)
    assert p.stat().st_size > 0
    print(json.dumps({"lam": best, "cv_resid_r_train": round(best_r, 4), "curve": curve, "blocks": blocks,
                      "amp_train": round(amp, 2), "n_feat": int(X.shape[2]), "out": str(p)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rigid"); ap.add_argument("--pca", type=int, default=16)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--blocks", nargs="+", default=["tier1", "corridor", "surf", "tissue"])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    main(ap.parse_args())
