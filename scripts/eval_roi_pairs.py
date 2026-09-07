#!/usr/bin/env python
"""checkpoint(들)의 ROI-pair 생성 품질 평가. --checkpoints 없이 부르면 pretrained 초기 상태만."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                    # noqa: E402
from atm_sc.data.paths import ATLAS                                # noqa: E402
from atm_sc.evaluation.roi_pair_eval import evaluate_pairs         # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                       # noqa: E402
from atm_sc.training.run import anatomy_feature                    # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--n-pairs", type=int, default=64, help="count 상위 pair 수")
    ap.add_argument("--n-per-pair", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subj = ROIPairSubject(a.sub)
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    top = np.argsort(-np.asarray(subj.pair_count_full))[: a.n_pairs]
    m = ROIPairATM(n_roi=subj.n_roi, device=dev); m.eval()
    feat = anatomy_feature(m, a.sub, "AF_L")
    rows = [("pretrained-init", evaluate_pairs(m, subj, feat, atlas, img.affine, top, a.n_per_pair))]
    for ck in a.checkpoints:
        sd = torch.load(ck, map_location=dev, weights_only=True)
        m.load_state_dict(sd["model"]); m.eval()
        rows.append((Path(ck).name, evaluate_pairs(m, subj, feat, atlas, img.affine, top, a.n_per_pair)))
    keys = ["start_roi_acc", "end_roi_acc", "pair_acc_unordered", "endpoint_in_background",
            "endpoint_dist_to_gt_mm", "length_mean_mm", "gt_length_mean_mm", "length_abs_err_mm", "w_mean"]
    print(f"{a.sub}: 상위 {a.n_pairs} pair x {a.n_per_pair} 생성")
    print(f"{'model':32s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for name, r in rows:
        print(f"{name:32s} " + " ".join(f"{r[k]:14.4f}" for k in keys))
