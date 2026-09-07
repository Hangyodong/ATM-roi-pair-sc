#!/usr/bin/env python
"""A3 — edge_head / count_head_end 를 subject feature 위에서 재학습한다.

측정: 학습 전후로 val subject 들이 **서로 다른 pair 집합을 고르는지**를 Jaccard 로 잰다.
그게 이 단계의 존재 이유다 (E6 checkpoint 실측 0.9896 = 사실상 전원 동일).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc.data.anat_tier1 import pair_input                              # noqa: E402
from atm_sc.data.local_feats import load_roi_feats, pair_local             # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                          # noqa: E402
from atm_sc.inference.generate_sc import select_pairs                      # noqa: E402
from atm_sc.training import config as C                                    # noqa: E402
from atm_sc.training.run import anatomy_feature, run                       # noqa: E402


def edge_agreement(ckpt: str, subs: list[str], device: str, thr: float = 0.5) -> dict:
    """subject 들이 고르는 pair 집합의 일치도. 1.0 이면 전원 같은 답이다."""
    m, sd = from_checkpoint(ckpt, device=device)
    src = sd.get("t1_source", "rigid")
    n_roi = m.n_roi
    iu = np.stack(np.triu_indices(n_roi, 1), 1)
    P = torch.as_tensor(iu, device=m.device)
    h = m.edge_head
    stats = None
    if int(getattr(h, "tier1_dim", 0) or 0):
        z = np.load(ROOT / "outputs/cache/anat_tier1_pair_stats.npz", allow_pickle=False)
        stats = {k: z[k] for k in z.files}
    sel, probs = [], []
    for s in subs:
        with torch.no_grad():
            f = anatomy_feature(m, s, "AF_L", source=src)
            loc = (pair_local(load_roi_feats(s, src, m.device, n_roi=n_roi), P)
                   if int(getattr(h, "local_dim", 0) or 0) else None)
            t1f = (torch.as_tensor(pair_input(s, stats, src), device=m.device)
                   if stats is not None else None)
            p, pr = select_pairs(m, f, n_roi, thr=thr, local=loc, tier1=t1f)
        sel.append(set(map(tuple, np.asarray(p))))
        # 확률은 전체 3321 개로 다시 받아 subject 간 비교가 가능하게 한다
        with torch.no_grad():
            probs.append(torch.sigmoid(m.edge_logits(f, P, loc, t1f)).cpu().numpy())
    inter, union = set.intersection(*sel), set.union(*sel)
    Q = np.stack(probs)
    C_ = np.corrcoef(Q)
    n = [len(x) for x in sel]
    del m
    torch.cuda.empty_cache()
    return {"n_pairs_min": int(min(n)), "n_pairs_max": int(max(n)),
            "jaccard_all_subjects": len(inter) / max(len(union), 1),
            "prob_inter_subj_r": float(C_[np.triu_indices(len(subs), 1)].mean()),
            "prob_subject_share": float((Q - Q.mean(0)).var() / max(Q.var(), 1e-12)),
            "n_subjects": len(subs)}


def main(a):
    built = C.build(C.load(ROOT / a.config))
    if a.resume:
        built["resume"] = Path(a.resume)
    if a.steps:
        built["max_steps"] = a.steps
        built["out_dir"] = str(built["out_dir"]) + "_smoke"
    val = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()][:a.n_val]
    before = edge_agreement(str(built["resume"]), val, a.device)
    print("[a3] 학습 전 edge 일치도:", json.dumps(before, indent=2), flush=True)
    assert before["jaccard_all_subjects"] > 0.0, "edge head 가 아무 pair 도 안 골랐다"

    ck = run(phase=built["phase"], subjects=built["subjects"], max_steps=built["max_steps"],
             out_dir=built["out_dir"], cfg=built["cfg"], weights=built["weights"],
             trainable=built["trainable"], unet_level=built["unet_level"],
             t1_source=built["t1_source"], in_channels=built["in_channels"],
             template=built["template"], resume=built["resume"],
             log_every=built["log_every"], save_every=built["save_every"],
             init_bundle=built["init_bundle"], device=a.device)
    after = edge_agreement(str(ck), val, a.device)
    print("[a3] 학습 후 edge 일치도:", json.dumps(after, indent=2), flush=True)
    res = {"checkpoint": str(ck), "before": before, "after": after,
           "jaccard_drop": before["jaccard_all_subjects"] - after["jaccard_all_subjects"]}
    out = ROOT / "outputs/eval/a3_aux.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print(f"저장: {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain/a3_aux.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=0, help="배선 스모크용 step 수 덮어쓰기")
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    sys.exit(main(ap.parse_args()))
