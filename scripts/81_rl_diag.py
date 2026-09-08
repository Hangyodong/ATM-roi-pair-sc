"""RL 역산이 목표의 개인차를 왜 못 전달하는지 — 생성 없이 역산만 스윕한다.

관측: GT 를 목표로 하면 전달률 0.43 인데 count head(0.097) / 능선(0.205) 목표는 0 또는 음수다.
역산 해 n 의 **예측 SC** A^T n 을 직접 채점하면 생성 잡음과 분리해서 원인을 볼 수 있다.
A 는 subject 당 한 번만 만들면 되므로 설정 수십 개를 몇 초에 훑는다.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import importlib
ev = importlib.import_module("49_eval_frozen")
rl = importlib.import_module("65_rl_alloc")
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.paths import ATLAS, CACHE
from atm_sc.evaluation.reproduction_metrics import subject_specificity, _corr
from atm_sc.inference.generate_sc import allocate_counts, pair_residual_log, select_pairs
from atm_sc.models import pair_ridge
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.training.run import anatomy_feature, roi_feats_if_needed, t1_input


def main(a):
    import nibabel as nib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    a.t1_src = sd.get("t1_source", "rigid"); a.init_bundle = "AF_L"
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()][:a.limit]
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); idx = rl.pair_index(R)
    tmpl = np.asarray(z["template"], np.float64)
    a.template_log1p = np.zeros((R, R)); a.template_log1p[iu] = tmpl; a.template_log1p.T[iu] = tmpl
    from atm_sc.data.anat_tier1 import STATS_PATH
    z1 = np.load(STATS_PATH, allow_pickle=False); a.tier1_stats = {k: z1[k] for k in z1.files}
    w = pair_ridge.load(a.t1_src)
    base_lin = np.expm1(tmpl)
    S = []
    for si, s in enumerate(subs):
        t0 = time.time(); subj = ROIPairSubject(s)
        with torch.no_grad():
            feat = (anatomy_feature(m, s, a.init_bundle, a.t1_src) if m.unet_level == "none"
                    else m.atm.encode_anatomy(t1_input(m, s, a.t1_src)))
            al, at = ev.aux_feats(m, s, R, a, dev)
            pairs, _ = select_pairs(m, feat, R, thr=0.5, local=al, tier1=at)
            roi = roi_feats_if_needed(m, s, a.t1_src)
            loc, t1f = ev.head_feats(m, s, pairs, a, dev)
            ai = ev._pair_rows(pairs, R)
            kw = dict(local=None if al is None else al[ai], tier1=None if at is None else at[ai])
            rlg = pair_residual_log(m, feat, pairs, a.template_log1p, loc, t1f)
            n_base = allocate_counts(m, feat, pairs, total=a.total, resid_log=rlg, resid_gain=1.0, **kw)
        A = rl.visit_matrix(m, feat, pairs, roi, atlas, aff, R, a.n_probe, seed=si, idx=idx)
        # 독립 표본으로 만든 두 번째 방문 행렬. A1 로 풀고 A2 로 채점하면 "Â 추정 잡음 때문에
        # 실제 생성에서 사라지는 몫" 을 생성 없이 잴 수 있다 (역산은 고주파를 증폭하므로
        # Â 의 잡음도 같이 증폭한다).
        A2 = rl.visit_matrix(m, feat, pairs, roi, atlas, aff, R, a.n_probe, seed=si + 7777, idx=idx)
        d = np.expm1(tmpl + pair_ridge.resid_log(s, w, a.t1_src)) - base_lin
        S.append({"A": A, "A2": A2, "n0": n_base.astype(np.float64),
                  "gt": np.asarray(subj.sc_mat, np.float64)[iu], "ridge_d": d})
        print(f"[{si+1}/{len(subs)}] {s} A 준비 {time.time()-t0:.0f}s", flush=True)
    G = np.stack([x["gt"] for x in S])
    out = {"n_subjects": len(S), "gt_inter": float(subject_specificity(G, G)["inter_subj_r_gt"])}
    rows = []
    for tgt, al, it, dp in itertools.product(a.targets, a.alphas, a.iters, a.damps):
        P, P2 = [], []
        for x in S:
            T = x["gt"] if tgt == "gt" else np.maximum(base_lin + al * x["ridge_d"], 0.0)
            n, _ = rl.rl_solve(x["A"], T, x["n0"], iters=it, damp=dp)
            P.append(x["A"].T @ n); P2.append(x["A2"].T @ n)
        P = np.stack(P); P2 = np.stack(P2)
        sp = subject_specificity(P, G); sp2 = subject_specificity(P2, G)
        Tk = np.stack([x["gt"] if tgt == "gt" else np.maximum(base_lin + al * x["ridge_d"], 0.0) for x in S])
        st = subject_specificity(Tk, G)
        r = {"target": tgt, "alpha": al, "iters": it, "damp": dp,
             "target_resid_r": round(st["resid_r"], 4), "solved_resid_r": round(sp["resid_r"], 4),
             "transmit": round(sp["resid_r"] / st["resid_r"], 3) if abs(st["resid_r"]) > 1e-6 else None,
             "inter": round(sp["inter_subj_r_pred"], 4),
             "abs_r": round(float(np.mean([_corr(P[i], G[i]) for i in range(len(P))])), 4),
             "fit_r": round(float(np.mean([_corr(P[i], Tk[i]) for i in range(len(P))])), 4),
             "holdout_resid_r": round(sp2["resid_r"], 4),
             "holdout_transmit": round(sp2["resid_r"] / st["resid_r"], 3) if abs(st["resid_r"]) > 1e-6 else None,
             "holdout_inter": round(sp2["inter_subj_r_pred"], 4)}
        rows.append(r)
        print(f"{tgt:5s} a={al:<4g} it={it:<4d} damp={dp:<4g} 목표 {r['target_resid_r']:+.4f} -> "
              f"실현 {r['solved_resid_r']:+.4f} (전달 {str(r['transmit']):>6s}) | 독립표본 "
              f"{r['holdout_resid_r']:+.4f} (전달 {str(r['holdout_transmit']):>6s}) inter {r['holdout_inter']:.3f}",
              flush=True)
    out["rows"] = rows
    (ROOT / f"outputs/eval/{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt")
    ap.add_argument("--limit", type=int, default=10); ap.add_argument("--total", type=int, default=460000)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=["gt", "ridge"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 8])
    ap.add_argument("--iters", type=int, nargs="+", default=[5, 20, 50, 200])
    ap.add_argument("--damps", type=float, nargs="+", default=[0.3, 1.0])
    ap.add_argument("--subjects", default="outputs/splits/val.txt")
    ap.add_argument("--tag", default="rl_diag")
    main(ap.parse_args())
