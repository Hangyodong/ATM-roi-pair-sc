"""RL 배분의 목표값을 무엇으로 할까 — count head / pair ridge / 둘의 결합 비교 (생성 없음).

결합 가중치는 **train 144명**에서만 최소제곱으로 구한다. val 은 보고에만 쓴다.
출력은 log1p 잔차 예측 [P] 이며, RL 목표 T = expm1(template + alpha * resid) 로 쓴다.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import importlib
ev = importlib.import_module("49_eval_frozen")
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.local_feats import roi_feature_path
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr
from atm_sc.models import pair_ridge
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.training.run import anatomy_feature, t1_input


@torch.no_grad()
def count_resid(m, sub, a, dev, pairs_all, tmpl):
    feat = (anatomy_feature(m, sub, "AF_L", a.t1_src) if m.unet_level == "none"
            else m.atm.encode_anatomy(t1_input(m, sub, a.t1_src)))
    loc, t1f = ev.head_feats(m, sub, pairs_all, a, dev)
    P = torch.as_tensor(pairs_all, device=dev)
    lp = torch.nn.functional.softplus(m.edge_log_counts(feat, P, loc, t1f)).double().cpu().numpy()
    assert np.isfinite(lp).all(), f"{sub}: count head 에 NaN/Inf"
    return lp - tmpl


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev); a.t1_src = sd.get("t1_source", "syn")
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); tmpl = np.asarray(z["template"], np.float64)
    a.template_log1p = np.zeros((R, R)); a.template_log1p[iu] = tmpl; a.template_log1p.T[iu] = tmpl
    from atm_sc.data.anat_tier1 import STATS_PATH
    z1 = np.load(STATS_PATH, allow_pickle=False); a.tier1_stats = {k: z1[k] for k in z1.files}
    pairs_all = np.stack(iu, 1)
    w = pair_ridge.load(a.t1_src)
    S = {k: [l.strip() for l in (ROOT / f"outputs/splits/{k}.txt").read_text().splitlines() if l.strip()]
         for k in ("train", "val")}
    S = {k: [s for s in v if roi_feature_path(s, a.t1_src).exists()] for k, v in S.items()}
    D = {}
    for k, subs in S.items():
        Y = np.stack([np.log1p(np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu]) for s in subs]) - tmpl[None]
        C = np.stack([count_resid(m, s, a, dev, pairs_all, tmpl) for s in subs])
        Rg = np.stack([pair_ridge.resid_log(s, w, a.t1_src) for s in subs])
        D[k] = {"Y": Y, "count": C, "ridge": Rg, "sc": np.stack(
            [np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu] for s in subs]), "subs": subs}
        print(k, len(subs), "done", flush=True)
    # 결합 가중치: train 에서 잔차 최소제곱 (중심화 후, pair 전체 동시)
    Ytr = loo_center(D["train"]["Y"])
    Xtr = np.stack([loo_center(D["train"]["count"]).ravel(), loo_center(D["train"]["ridge"]).ravel()], 1)
    beta = np.linalg.lstsq(Xtr, Ytr.ravel(), rcond=None)[0]
    out = {"beta_count": float(beta[0]), "beta_ridge": float(beta[1]),
           "corr_count_ridge_train": float(_corr(Xtr[:, 0], Xtr[:, 1])), "ckpt": a.ckpt}
    cand = {"count": lambda d: d["count"], "ridge": lambda d: d["ridge"],
            "stack": lambda d: beta[0] * d["count"] + beta[1] * d["ridge"]}
    for split in ("train", "val"):
        d = D[split]; out[split] = {}
        for name, fn in cand.items():
            pr = fn(d)
            amp = float(loo_center(d["Y"]).std() / max(loo_center(pr).std(), 1e-9))
            row = {"resid_r_log": round(float(np.mean([_corr(loo_center(pr)[i], loo_center(d["Y"])[i])
                                                       for i in range(len(pr))])), 4),
                   "amp_to_gt": round(amp, 2)}
            for al in a.alphas:
                lin = np.maximum(np.expm1(tmpl[None] + al * pr), 0.0)
                sp = subject_specificity(lin, d["sc"])
                row[f"a{al:g}"] = {"resid_r": round(sp["resid_r"], 4),
                                   "inter": round(sp["inter_subj_r_pred"], 4),
                                   "abs_r": round(float(np.mean([_corr(lin[i], d["sc"][i])
                                                                 for i in range(len(lin))])), 4)}
            out[split][name] = row
        out[split]["gt_inter"] = round(float(subject_specificity(d["sc"], d["sc"])["inter_subj_r_gt"]), 4)
    (ROOT / "outputs/eval/target_stack.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "outputs/eval/target_stack_val.npz", beta=beta,
                        **{k: D["val"][k] for k in ("Y", "count", "ridge")}, subs=np.array(D["val"]["subs"]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    main(ap.parse_args())
