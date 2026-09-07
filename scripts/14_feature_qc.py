#!/usr/bin/env python
"""T1 feature QC (최종 전략 §11): frozen pretrained encoder vs fine-tuned encoder.

subject 별 anatomy feature a_s 에 대해 norm, 쌍별 cosine / Pearson, Euclidean, 차원별 분산,
PCA explained variance 를 내고 scanner/protocol(manifest 의 batch2/proto) 과 같이 저장한다.
핵심 질문: 모든 subject 가 같은 feature 로 collapse 했는가, subject-specific 변동이 있는가,
있다면 scanner 만 구분하는가.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import CACHE, subject_meta                   # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                        # noqa: E402
from atm_sc.training.run import t1_input                            # noqa: E402


def t1_tensor(model, sub):
    """모델이 받는 채널 구성 그대로 만든다 (2채널이면 rigid T1 + WM)."""
    return t1_input(model, sub)


@torch.no_grad()
def features(model, subs):
    F = []
    for s in subs:
        x = t1_tensor(model, s)
        a = model.atm.encode_anatomy(x) if model.unet_level == "none" else model.atm.encode_anatomy_grad(x, model.unet_level, use_checkpoint=False)
        F.append(a.float().cpu().numpy())
    return np.concatenate(F)


def qc(F, subs, meta):
    S, D = F.shape
    iu = np.triu_indices(S, 1)
    norm = np.linalg.norm(F, axis=1)
    Fn = F / (norm[:, None] + 1e-12)
    cos = (Fn @ Fn.T)[iu]
    Fc = F - F.mean(1, keepdims=True)
    corr = ((Fc / (np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-12)) @ (Fc / (np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-12)).T)[iu]
    euc = np.linalg.norm(F[:, None] - F[None], axis=-1)[iu]
    dimvar = F.var(0)
    centered = F - F.mean(0)
    sv = np.linalg.svd(centered, compute_uv=False) if S > 1 else np.zeros(1)
    ev = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
    out = {"n_subjects": S, "dim": D,
           "norm_mean": float(norm.mean()), "norm_std": float(norm.std()),
           "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
           "corr_mean": float(corr.mean()), "corr_min": float(corr.min()),
           "euclid_mean": float(euc.mean()), "euclid_over_norm": float(euc.mean() / (norm.mean() + 1e-12)),
           "dimvar_mean": float(dimvar.mean()), "dimvar_max": float(dimvar.max()),
           "n_dims_var_gt_1e-6": int((dimvar > 1e-6).sum()),
           "pca_explained_top5": [float(x) for x in ev[:5]],
           "per_subject_norm": {s: float(n) for s, n in zip(subs, norm)}}
    # scanner/protocol 만 구분하는가: 같은 batch2 쌍 vs 다른 batch2 쌍의 cosine
    b2 = [meta.get(s, {}).get("batch2", "?") for s in subs]
    same = np.array([b2[i] == b2[j] for i, j in zip(*iu)])
    if same.any() and (~same).any():
        out["cos_same_scanner"] = float(cos[same].mean()); out["cos_diff_scanner"] = float(cos[~same].mean())
    out["batch2"] = dict(zip(subs, b2))
    return out


def main(a):
    subs = sorted(p.name.replace("_T1w_syn_W.npy", "") for p in CACHE.glob("sub-*_T1w_syn_W.npy"))[: a.max_subjects]
    assert len(subs) >= 2, "SyN T1 캐시가 2명 이상 필요"
    meta = subject_meta()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    m = ROIPairATM(n_roi=82, trainable="vae", device=dev); m.eval()
    results["frozen_pretrained"] = qc(features(m, subs), subs, meta)
    for ck in a.checkpoints:
        sd = torch.load(ck, map_location=dev, weights_only=True)
        lv = sd.get("unet_level", {"vae+unet4": "stage4", "full": "full"}.get(sd.get("trainable", "vae"), "none"))
        m = ROIPairATM(n_roi=82, trainable="full" if lv == "full" else "vae", unet_level=lv, device=dev)
        m.load_checkpoint(sd["model"]); m.eval()   # 구 1채널 checkpoint 는 conv1_1 이 0-패딩된다
        results[Path(ck).stem + f"[{lv}]"] = qc(features(m, subs), subs, meta)
    keys = ["norm_mean", "cos_mean", "cos_min", "corr_mean", "euclid_mean", "euclid_over_norm",
            "dimvar_mean", "n_dims_var_gt_1e-6", "cos_same_scanner", "cos_diff_scanner"]
    print(f"subjects ({len(subs)}): {subs}")
    print(f"{'encoder':34s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for name, r in results.items():
        print(f"{name:34s} " + " ".join(f"{r.get(k, float('nan')):12.4f}" if isinstance(r.get(k), float) else f"{str(r.get(k,'-')):>12s}" for k in keys))
        print(f"{'':34s} PCA top5 explained: " + ", ".join(f"{x:.3f}" for x in r["pca_explained_top5"]))
    out = ROOT / "outputs" / "qc" / "feature_qc.json"; out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1, ensure_ascii=False); print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--max-subjects", type=int, default=10)
    main(ap.parse_args())
