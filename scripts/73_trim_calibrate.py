"""trimming/filtering 임계값 보정 — GT 를 얼마나 지우는지 먼저 재고 정한다 (문서 §7.7).

지표만 좋아지고 진짜 연결이 사라지면 개선이 아니다. 그래서 GT 가닥에 같은 필터를 걸어
  retained  버려지지 않은 비율
  kept_len  남은 길이 / 원래 길이
  n_visit   가닥 하나가 지나는 ROI 수 (GT 4.4 / 생성 6.4 -- 이 격차를 좁히는 게 목적)
를 임계값 격자에서 잰다. 생성물에도 같은 격자를 걸어 양쪽을 나란히 본다.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import importlib
ev = importlib.import_module("49_eval_frozen")
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.paths import ATLAS, CACHE
from atm_sc.data.tt_io import point_labels, roi_visit_sets
from atm_sc.data.wm_segment import tissue_path
from atm_sc.filtering.wm_trim import TrimConfig, trim_and_filter
from atm_sc.inference.generate_sc import select_pairs
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.spaces import W_AFFINE
from atm_sc.training.run import anatomy_feature, roi_feats_if_needed, t1_input


def visit_stats(pts, npts, atlas, aff):
    """가닥당 방문 ROI 수와 pair 기여 수."""
    if len(npts) == 0:
        return {"n_visit": 0.0, "pair_per_stream": 0.0, "n_stream": 0}
    lab = point_labels(np.asarray(pts, np.float64), atlas, aff)
    pk = roi_visit_sets(lab, np.asarray(npts, np.int64))
    nv = np.bincount(pk[:, 0], minlength=len(npts))
    return {"n_visit": float(nv.mean()), "pair_per_stream": float((nv * (nv - 1) / 2).mean()),
            "n_stream": int(len(npts))}


def measure(S, wm, brain, atlas, aff, grid):
    T = S.shape[1]
    base_len = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
    out = {"raw": {**visit_stats(S.reshape(-1, 3), np.full(len(S), T, np.int64), atlas, aff),
                   "retained": 1.0, "kept_len": 1.0, "length_mm": float(base_len.mean())}}
    for thr, marg, mo in grid:
        cfg = TrimConfig(wm_thr=thr, gm_margin_pts=marg, max_outside_brain=mo)
        pts, npts, keep = trim_and_filter(S, wm, brain, W_AFFINE, cfg)
        st = visit_stats(pts, npts, atlas, aff)
        if len(npts):
            o = np.cumsum(np.concatenate([[0], npts]))
            L = np.array([np.linalg.norm(np.diff(pts[o[i]:o[i + 1]], axis=0), axis=1).sum()
                          for i in range(len(npts))])
            st["kept_len"] = float((L / np.maximum(base_len[keep], 1e-6)).mean())
            st["length_mm"] = float(L.mean())
        st["retained"] = float(keep.mean())
        out[f"thr{thr}_m{marg}_o{mo}"] = st
    return out


def main(a):
    import nibabel as nib
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
    subs = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()][:a.n_subj]
    grid = [(t, m, o) for t in a.thrs for m in a.margins for o in a.outs]
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_ = sd = None
    if a.ckpt:
        m_, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
        a.t1_src = sd.get("t1_source", "rigid"); a.init_bundle = "AF_L"
        from atm_sc.data.anat_tier1 import STATS_PATH
        z1 = np.load(STATS_PATH, allow_pickle=False); a.tier1_stats = {k: z1[k] for k in z1.files}
    agg = {"gt": [], "gen": []}
    for s in subs:
        t0 = time.time()
        # 필터/trimming 은 **SyN(NLin6)** 조직맵을 쓴다. GT 가닥이 그 공간에 있어서 rigid 를
        # 쓰면 조직 경계가 어긋난다 (GT 점이 GM+WM 안: rigid 0.820 / syn 0.907).
        p = tissue_path(CACHE, s, a.tissue_source)
        assert p.exists(), f"{s}: {a.tissue_source} tissue 없음 -> scripts/69_tissue_extract.py --source {a.tissue_source}"
        z = np.load(p); wm = z["wm"].astype(np.float32) / 255.0; brain = z["brain"]
        St = np.load(ROIPairSubject(s).dir / "bundles.npz")["streamlines"]
        sel = np.sort(rng.choice(len(St), min(a.n_stream, len(St)), replace=False))
        agg["gt"].append(measure(St[sel].astype(np.float64), wm, brain, atlas, aff, grid))
        if m_ is not None:
            with torch.no_grad():
                feat = (anatomy_feature(m_, s, a.init_bundle, a.t1_src) if m_.unet_level == "none"
                        else m_.atm.encode_anatomy(t1_input(m_, s, a.t1_src)))
                al, at = ev.aux_feats(m_, s, 82, a, dev)
                pairs, _ = select_pairs(m_, feat, 82, thr=0.5, local=al, tier1=at)
                roi = roi_feats_if_needed(m_, s, a.t1_src)
                k = rng.choice(len(pairs), min(a.n_stream // 8, len(pairs)), replace=False)
                P = torch.as_tensor(np.repeat(pairs[k], 8, axis=0), device=dev)
                g = torch.Generator(device=dev); g.manual_seed(0)
                Sg, _, _ = m_.generate(feat, P, 1, generator=g, local_roi=roi)
            agg["gen"].append(measure(Sg.cpu().numpy().astype(np.float64), wm, brain, atlas, aff, grid))
        print(f"{s} {time.time()-t0:.0f}s", flush=True)
    out = {}
    for src, rows in agg.items():
        if not rows:
            continue
        out[src] = {k: {kk: float(np.mean([r[k][kk] for r in rows])) for kk in rows[0][k]} for k in rows[0]}
    (ROOT / "outputs/eval/trim_calibration.json").write_text(json.dumps(out, indent=1))
    for src in out:
        print(f"== {src}")
        for k, v in out[src].items():
            print(f"  {k:16s} retained {v['retained']:.3f} kept_len {v.get('kept_len', 1):.3f} "
                  f"n_visit {v['n_visit']:.2f} pair/stream {v['pair_per_stream']:.1f} "
                  f"len {v.get('length_mm', 0):.0f}mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt")
    ap.add_argument("--n-subj", type=int, default=4); ap.add_argument("--n-stream", type=int, default=8000)
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--margins", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--outs", type=float, nargs="+", default=[0.05, 0.15, 0.30, 1.0],
                    help="뇌 마스크 밖 비율 상한. 1.0 = 이 기준 끔")
    ap.add_argument("--tissue-source", default="syn")
    main(ap.parse_args())
