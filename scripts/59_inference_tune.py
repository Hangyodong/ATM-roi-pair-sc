#!/usr/bin/env python
"""B 묶음 — 추론 캘리브레이션 3종을 **한 번의 생성으로** 동시에 잰다 (학습 불필요).

  python scripts/59_inference_tune.py --ckpt <path>

B2 prior 온도   z = mu + s*sigma*eps.  실측 prior sigma 1.0 vs GT 조건 내 sd 0.18 로 5.6배 과대
B3 후처리 필터  overreach 1.407 vs GT 0.212. **subject 자신의** WM 맵으로 거른다 (상류는 템플릿)
B1 가닥 수 보정 예측 총합이 GT 의 9.1%. 전역 상수 k 는 다시 그룹 수준이라 **subject 별** k 를 쓴다

같은 생성물에서 셋을 다 재므로 GPU 시간이 1/3 이다. 판정은 Pareto -- 한 지표만 좋아지고
다른 게 무너지면 채택하지 않는다 (전략 문서 §7.7, §6.2).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402
from atm_sc.data.paths import ATLAS, CACHE                              # noqa: E402
from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics   # noqa: E402
from atm_sc.inference.generate_sc import select_pairs, tractogram_sc    # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.models.roi_pair_embedding import canonical_pairs            # noqa: E402
from atm_sc.training.run import anatomy_feature                         # noqa: E402

EVAL = ROOT / "outputs/eval"
DEFAULT_CKPT = ROOT / "outputs/checkpoints/retrain/d3_joint/d3_joint_step3000.pt"
TEMPS = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)
WM_GRID = (0.0, 0.2, 0.3, 0.4, 0.5)


def gen_with_temp(model, a, pairs_t, n_per_pair, s, seed, local_roi=None):
    """z = mu + s*sigma*eps 로 생성. s=0 이면 prior 평균 결정론적 디코딩."""
    pr = canonical_pairs(pairs_t).repeat_interleave(n_per_pair, dim=0)
    g = torch.Generator(device=model.device); g.manual_seed(seed)
    with torch.inference_mode():
        mu, ls = model.prior_params(pr, anatomy=None if not model.pair_emb.prior_use_anatomy else a)
        eps = torch.randn(mu.shape, generator=g, device=model.device)
        z = mu + s * torch.exp(ls) * eps
        S, w, prr = model.generate(a, pairs_t, n_per_pair, generator=g, z=z, local_roi=local_roi)
    return S.cpu().numpy().astype(np.float32), w.cpu().numpy(), prr.cpu().numpy()


def wm_scores(S, wm, W_AFF):
    inv = np.linalg.inv(W_AFF)
    v = np.rint(S.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    ok = ((v >= 0) & (v < np.asarray(wm.shape))).all(1)
    val = np.zeros(len(v), np.float32)
    val[ok] = wm[v[ok, 0], v[ok, 1], v[ok, 2]]
    val = val.reshape(S.shape[0], S.shape[1])
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
    return val[:, 16:112].mean(1), seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--n-per-pair", type=int, default=16)
    ap.add_argument("--n-dice", type=int, default=8000)
    a = ap.parse_args()
    from atm_sc.spaces import W_AFFINE
    m, sd = from_checkpoint(a.ckpt, device="cuda"); m.eval()
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    subs = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()][:a.n_subj]
    src = sd.get("t1_source", "rigid")
    iu = np.triu_indices(int(m.n_roi), 1)
    rows = []
    for s_temp in TEMPS:
        per = []
        t0 = time.time()
        for sub in subs:
            feat = anatomy_feature(m, sub, "AF_L", source=src)
            lr = None
            if m.pair_emb.local_dim or m.pair_emb.prior_local is not None:
                from atm_sc.data.local_feats import load_roi_feats
                lr = load_roi_feats(sub, src, m.device, n_roi=m.n_roi)
            pairs, _ = select_pairs(m, feat, int(m.n_roi))
            S, w, pr = gen_with_temp(m, feat, torch.as_tensor(np.asarray(pairs), device=m.device),
                                     a.n_per_pair, s_temp, seed=0, local_roi=lr)
            gt = np.asarray(ROIPairSubject(sub).sc_mat, np.float64)
            sc = tractogram_sc(S, w, atlas, img.affine, int(m.n_roi))["pass"]["sc"]
            wm = np.load(CACHE / f"{sub}_WM_W.npy").astype(np.float32)
            mid, length = wm_scores(S.astype(np.float64), wm, W_AFFINE)
            z = np.load(ROIPairSubject(sub).dir / "bundles.npz")
            rng = np.random.default_rng(0)
            gt_S = z["streamlines"][np.sort(rng.choice(len(z["streamlines"]), a.n_dice, replace=False))]
            rec = {"subject": sub,
                   "sc_r": float(np.corrcoef(sc[iu], gt[iu])[0, 1]),
                   "sum_ratio": float(sc[iu].sum() / max(gt[iu].sum(), 1)),
                   "n_gen": int(len(S))}
            # B3 필터 스윕 + 그 조건에서의 전뇌 dice (규약: pred = gt = n_dice)
            for g in WM_GRID:
                keep = (mid > g) & (length >= 20.0)
                rec[f"retain@{g}"] = float(keep.mean())
                if keep.sum() >= a.n_dice:
                    idx = np.sort(rng.choice(np.flatnonzero(keep), a.n_dice, replace=False))
                    gm = bundle_geometry_metrics(S[idx], gt_S)
                    rec[f"dice@{g}"] = float(gm["dice"]); rec[f"over@{g}"] = float(gm["overreach"])
                    rec[f"cov@{g}"] = float(gm["coverage"])
            per.append(rec)
        agg = {"temp": s_temp, "sec": time.time() - t0, "n_subj": len(per)}
        for k in per[0]:
            if k == "subject":
                continue
            v = [p[k] for p in per if k in p]
            if v:
                agg[k] = float(np.mean(v))
        rows.append(agg)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()},
                         ensure_ascii=False), flush=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "b_tune.json").write_text(json.dumps({"ckpt": a.ckpt, "rows": rows},
                                                 ensure_ascii=False, indent=2))
    print("\n저장: outputs/eval/b_tune.json", flush=True)


if __name__ == "__main__":
    main()
