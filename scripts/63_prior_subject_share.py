#!/usr/bin/env python
"""prior 가 subject 조건부인지 잰다 (D-f 판정 지표).
  python scripts/63_prior_subject_share.py --ckpt <path> [--n-subj 8]
prior mu 의 subject 성분(분산 지분)과, 국소 가지(r_p = mu_p - prior_mean)만의 subject 성분을 낸다.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.models.roi_pair_embedding import canonical_pairs            # noqa: E402
from atm_sc.data.local_feats import load_roi_feats, pair_local          # noqa: E402
from atm_sc.training.run import anatomy_feature                         # noqa: E402


def share(X): return float((X - X.mean(0)).var() / max(X.var(), 1e-12))


def main(a):
    m, sd = from_checkpoint(a.ckpt, device=a.device); src = sd.get("t1_source", "rigid"); pe = m.pair_emb
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()][:a.n_subj]
    n = int(m.n_roi)
    P = canonical_pairs(torch.as_tensor(np.stack(np.triu_indices(n, 1), 1), device=m.device))
    M, R = [], []
    with torch.no_grad():
        base = m.prior_mean(P).flatten().cpu().numpy()
        for s in subs:
            f = anatomy_feature(m, s, "AF_L", source=src)
            pl = pair_local(load_roi_feats(s, src, m.device, n_roi=n), P) if pe.prior_local is not None else None
            mu, _ = m.prior_params(P, anatomy=(f if pe.prior_use_anatomy else None), local=pl)
            v = mu.flatten().cpu().numpy(); M.append(v); R.append(v - base)
    M, R = np.stack(M), np.stack(R)
    C = np.corrcoef(M)
    out = {"ckpt": a.ckpt, "n_subj": len(subs),
           "prior_mu_inter_subj_r": float(C[np.triu_indices(len(subs), 1)].mean()),
           "prior_mu_subject_share": share(M),
           "branch_rms": float(np.sqrt((R ** 2).mean())), "base_rms": float(np.sqrt((base ** 2).mean())),
           "branch_subject_share": share(R),
           "centered": bool(getattr(pe, "prior_local_roi_mean", torch.zeros(1)).abs().sum() > 0)}
    print(json.dumps(out, indent=2)); return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--subjects", default="outputs/splits/val.txt"); ap.add_argument("--device", default="cuda")
    sys.exit(main(ap.parse_args()))
