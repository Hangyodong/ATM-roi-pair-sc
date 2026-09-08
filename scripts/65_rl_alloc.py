"""배분 역산(Richardson-Lucy): 생성 SC 가 count head SC 가 되도록 pair 별 가닥 수를 푼다.

왜 필요한가 (`outputs/eval/alloc_to_sc_val.json`, val 31명):
  count head resid_r 0.086 -> 배분 라벨 0.018 -> 끝점 SC 0.014 -> pass SC 0.004
  (1) count_head_end 가 배분을 지배하는데 잔차 목적으로 학습된 적이 없다 -> 79 % 손실
  (2) 가닥 하나가 ROI 6.4 개를 지나 pair 20 개에 count 를 더한다 -> 라벨->pass 전달률 0.27

pass SC 는 라벨의 선형 변환이다:  SC = A^T n,  A[k,p] = 라벨 k 가닥 1개가 pair p 에 더하는 기댓값.
A 는 subject 자신의 가닥 표본에서 측정한다 (GT/그룹 정보 없음). 그러면 n 을 "그냥 목표값"으로
두는 대신 A^T n = T 를 풀 수 있다. count 가 Poisson 이므로 EM(=Richardson-Lucy) 곱셈 갱신이
n >= 0 을 자동으로 지키고 KL 을 단조 감소시킨다.

목표 T 후보:
  count : count head 의 SC (추론 규칙 준수 -- T1 만 쓴다)
  gt    : GT SC (**오라클, 진단 전용**) -- 디컨볼루션이 개인차를 얼마나 전달할 수 있는지의 상한
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
from atm_sc.filtering.wm_trim import TrimConfig
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr
from atm_sc.inference.generate_sc import allocate_counts, pair_residual_log, generate_by_count, select_pairs
from atm_sc.models import pair_ridge
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.training.run import anatomy_feature, t1_input, roi_feats_if_needed


def pair_index(n_roi: int) -> np.ndarray:
    """IDX[i,j] = upper-triangle 순서의 pair 번호 (대각 -1)."""
    iu = np.triu_indices(n_roi, 1)
    idx = np.full((n_roi, n_roi), -1, np.int64)
    idx[iu] = np.arange(len(iu[0])); idx.T[iu] = np.arange(len(iu[0]))
    return idx


@torch.no_grad()
def visit_matrix(model, feat, pairs, roi, atlas, aff, n_roi, n_probe, seed, idx, batch=20000, trim=None):
    """A [K, n_pair]: 라벨 k 가닥 1개가 pair p 를 방문하는 기댓값. subject 자신의 표본에서만 나온다.

    trim 을 주면 **실제 생성과 같은** 후처리를 거친 뒤 세야 A 가 A^T n = 실현 SC 를 만족한다.
    (trimming 을 켜고 A 를 원본 가닥으로 재면 역산이 틀린 연산자를 뒤집는 셈이 된다.)"""
    K = len(pairs); n_pair = n_roi * (n_roi - 1) // 2
    A = np.zeros((K, n_pair), np.float32)
    rep_k = np.repeat(np.arange(K), n_probe)
    rep_p = np.repeat(np.asarray(pairs, np.int64), n_probe, axis=0)
    g = torch.Generator(device=model.device); g.manual_seed(seed)
    for i in range(0, len(rep_p), batch):
        blk = rep_p[i:i + batch]; kk = rep_k[i:i + batch]
        S, _, _ = model.generate(feat, torch.as_tensor(blk, device=model.device), 1,
                                 generator=g, local_roi=roi)
        S = S.cpu().numpy().astype(np.float32); N, T = S.shape[:2]
        if trim is None:
            pts = S.reshape(-1, 3); npts = np.full(N, T, np.int64); lk = kk
        else:
            from atm_sc.filtering.wm_trim import trim_and_filter
            from atm_sc.spaces import W_AFFINE
            pts, npts, keep_m = trim_and_filter(S, trim[0], trim[1], np.asarray(W_AFFINE), trim[2])
            lk = kk[keep_m]
            if len(npts) == 0:
                continue
        lab = point_labels(np.asarray(pts, np.float64), atlas, aff)
        pk = roi_visit_sets(lab, npts)
        bnd = np.searchsorted(pk[:, 0], np.arange(len(npts) + 1))
        for t in range(len(npts)):
            rs = pk[bnd[t]:bnd[t + 1], 1]
            if len(rs) < 2:
                continue
            ii, jj = np.triu_indices(len(rs), 1)
            np.add.at(A, (lk[t], idx[rs[ii], rs[jj]]), 1.0)
    A /= float(n_probe)
    rs_ = A.sum(1)
    assert np.isfinite(A).all(), "A 에 NaN/Inf"
    assert (rs_ > 0).mean() > 0.9, f"가닥이 pair 를 하나도 안 지나는 라벨이 {(rs_ == 0).mean():.1%}"
    return A


def rl_solve(A, T, n0, iters=200, damp=1.0, floor=1e-6):
    """min KL(T || A^T n) s.t. n >= 0. EM(Richardson-Lucy) 곱셈 갱신."""
    A = np.asarray(A, np.float64); T = np.maximum(np.asarray(T, np.float64), 0.0)
    n = np.maximum(np.asarray(n0, np.float64), floor)
    w = A.sum(1); w = np.maximum(w, 1e-9)
    hist = []
    for it in range(iters):
        pred = A.T @ n
        ratio = np.where(pred > 1e-9, T / np.maximum(pred, 1e-9), 1.0)
        upd = (A @ ratio) / w
        n = np.maximum(n * np.power(np.maximum(upd, 1e-9), damp), floor)
        if it % 20 == 0 or it == iters - 1:
            hist.append(float(_corr(A.T @ n, T)))
    assert np.isfinite(n).all(), "RL 해에 NaN/Inf"
    return n, hist


def main(a):
    import nibabel as nib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    a.t1_src = sd.get("t1_source", "syn"); a.init_bundle = "AF_L"
    if a.ridge is not None and not a.ridge_source:
        assert int(a.ridge["n_train"]) == 144, "능선이 train 144명으로 적합되지 않았다"
    assert "test" not in a.subjects, "test 는 1회 소진됨 (memory: test-split-used-once)"
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (CACHE / f"{s}_T1w_syn_W.npy").exists()][: a.limit or None]
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1); idx = pair_index(R)
    a.template_log1p = np.zeros((R, R)); a.template_log1p[iu] = z["template"]; a.template_log1p.T[iu] = z["template"]
    from atm_sc.data.anat_tier1 import STATS_PATH
    z1 = np.load(STATS_PATH, allow_pickle=False); a.tier1_stats = {k: z1[k] for k in z1.files}

    modes = [x for x in a.modes.split(",") if x]
    V = {x: [] for x in ["base"] + modes}; TG = {x: [] for x in modes}; G = []; info = []
    for si, s in enumerate(subs):
        t0 = time.time(); subj = ROIPairSubject(s); assert subj.n_roi == R
        gt = np.asarray(subj.sc_mat, np.float64)
        trim = None
        if a.trim:
            tp = tissue_path(CACHE, s, a.trim_source)
            assert tp.exists(), f"{s}: {a.trim_source} 조직맵 없음"
            tz = np.load(tp)
            trim = (tz["wm"].astype(np.float32) / 255.0, tz["brain"],
                    TrimConfig(wm_thr=a.wm_thr, gm_margin_pts=a.gm_margin,
                               max_outside_brain=a.max_outside, min_length_mm=a.min_len))
        with torch.no_grad():
            feat = (anatomy_feature(m, s, a.init_bundle, a.t1_src) if m.unet_level == "none"
                    else m.atm.encode_anatomy(t1_input(m, s, a.t1_src)))
            al, at = ev.aux_feats(m, s, R, a, dev)
            pairs, _ = select_pairs(m, feat, R, thr=a.edge_thr, local=al, tier1=at)
            roi = roi_feats_if_needed(m, s, a.t1_src)
            loc, t1f = ev.head_feats(m, s, pairs, a, dev)
            P = torch.as_tensor(pairs, device=dev)
            cnt = m.edge_count_matrix(feat, P, loc, t1f).cpu().numpy()
            rl_ = pair_residual_log(m, feat, pairs, a.template_log1p, loc, t1f)
            ai = ev._pair_rows(pairs, R)
            kw = dict(local=None if al is None else al[ai], tier1=None if at is None else at[ai])
            n_base = allocate_counts(m, feat, pairs, total=a.total, resid_log=rl_, resid_gain=1.0, **kw)
        A = visit_matrix(m, feat, pairs, roi, atlas, aff, R, a.n_probe, seed=si, idx=idx,
                         trim=trim)
        rec = {"subject": s, "n_pairs": int(len(pairs)), "A_rowsum": float(A.sum(1).mean())}
        # count head 는 선택된 pair 밖도 예측한다. 목표는 3321 개 전부로 둔다 (제약이 많을수록 해가 낫다).
        with torch.no_grad():
            Pall = torch.as_tensor(np.stack(iu, 1), device=dev)
            if int(getattr(m.count_head, "local_dim", 0) or 0) or int(getattr(m.count_head, "tier1_dim", 0) or 0):
                la2, ta2 = ev.head_feats(m, s, np.stack(iu, 1), a, dev)
            else:
                la2 = ta2 = None
            T_count = m.edge_count_matrix(feat, Pall, la2, ta2).cpu().numpy()[iu]
        targets = {"count": T_count, "gt": gt[iu]}
        if a.ridge is not None:
            # 능선 목표값: 선형 공간 진폭 alpha 로 "얼마나 서로 다른가" 를 GT 수준까지 올린다
            # (로그 공간 증폭은 exp 가 큰 edge 를 폭발시켜 resid_r 을 무너뜨린다 -- 실측 0.147 -> 0.018).
            targets["ridge"] = pair_ridge.predict_sc(s, a.ridge, a.t1_src, alpha=a.alpha)
            rec["ridge_alpha"] = a.alpha
        sc0, n0_gen, _ = generate_by_count(m, feat, pairs, n_base, atlas, aff, R, local_roi=roi,
                                           seed=1000 + si, trim=trim)
        if trim is not None:
            rec["trim_drop"] = float(sc0.get("trim_drop", 0.0))
        V["base"].append(np.asarray(sc0["pass"]["sc"], np.float64)[iu]); rec["base_gen"] = int(n0_gen)
        for md in modes:
            Tm = np.maximum(targets[md], 0.0)
            n_rl, hist = rl_solve(A, Tm, n_base.astype(np.float64), iters=a.iters, damp=a.damp)
            tot = n_rl.sum()
            if a.max_total and tot > a.max_total:
                n_rl = n_rl * (a.max_total / tot)
            nn = np.clip(np.round(n_rl), 1, 20000).astype(np.int64)
            scm, ng, _ = generate_by_count(m, feat, pairs, nn, atlas, aff, R, local_roi=roi,
                                           seed=2000 + si, trim=trim)
            V[md].append(np.asarray(scm["pass"]["sc"], np.float64)[iu])
            TG[md].append(Tm.copy())
            rec[f"{md}_gen"] = int(ng); rec[f"{md}_solve_r"] = hist[-1]; rec[f"{md}_total"] = float(tot)
        G.append(gt[iu]); info.append(rec)
        print(f"[{si+1}/{len(subs)}] {s}: pair {len(pairs)} A_rowsum {rec['A_rowsum']:.1f} " +
              " ".join(f"{md} gen {rec[f'{md}_gen']:,} solve_r {rec[f'{md}_solve_r']:.3f}" for md in modes) +
              f" {time.time()-t0:.0f}s", flush=True)

    G = np.stack(G); out = {"n_subjects": len(subs), "n_probe": a.n_probe, "iters": a.iters,
                            "damp": a.damp, "ckpt": a.ckpt, "per_subject": info}
    for k in ["base"] + modes:
        Pk = np.stack(V[k]); sp = subject_specificity(Pk, G)
        out[k] = {"resid_r": sp["resid_r"], "inter_subj_r": sp["inter_subj_r_pred"],
                  "abs_r": float(np.mean([_corr(Pk[i], G[i]) for i in range(len(G))]))}
    for md in modes:   # 목표 자체의 개인차 -> 실현 SC 개인차 (전달률을 분리해서 본다)
        Tk = np.stack(TG[md]); st = subject_specificity(Tk, G)
        out[f"target_{md}"] = {"resid_r": st["resid_r"], "inter_subj_r": st["inter_subj_r_pred"],
                               "abs_r": float(np.mean([_corr(Tk[i], G[i]) for i in range(len(G))]))}
        out[md]["transmit"] = (out[md]["resid_r"] / st["resid_r"]) if abs(st["resid_r"]) > 1e-6 else None
    out["gt_inter_subj_r"] = float(subject_specificity(G, G)["inter_subj_r_gt"])
    print(json.dumps({k: out[k] for k in ["base"] + modes + [f"target_{x}" for x in modes] + ["gt_inter_subj_r"]}, indent=1), flush=True)
    p = ROOT / f"outputs/eval/{a.tag}.json"; p.write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / f"outputs/eval/{a.tag}_vectors.npz", gt_sc=G, subjects=np.array(subs),
                        **{f"pred_{k}": np.stack(V[k]) for k in ["base"] + modes})
    assert p.stat().st_size > 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt")
    ap.add_argument("--subjects", default="outputs/splits/val.txt")
    ap.add_argument("--modes", default="count,gt")
    ap.add_argument("--total", type=int, default=460000)
    ap.add_argument("--max-total", type=int, default=600000)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--edge-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="rl_alloc_val")
    ap.add_argument("--ridge-source", default="")
    ap.add_argument("--alpha", type=float, default=8.0, help="선형 잔차 진폭. val 실측 alpha=8 에서 subject 간 상관 0.902 (GT 0.904)")
    ap.add_argument("--trim", action="store_true", help="백질 trimming + 뇌 마스크 filtering")
    ap.add_argument("--trim-source", default="syn")
    ap.add_argument("--wm-thr", type=float, default=0.3)
    ap.add_argument("--gm-margin", type=int, default=8)
    ap.add_argument("--max-outside", type=float, default=1.0)
    ap.add_argument("--min-len", type=float, default=20.0)
    _a = ap.parse_args()
    _a.ridge = None
    if "ridge" in _a.modes.split(","):
        _a.ridge = pair_ridge.load(_a.ridge_source or "rigid")
    main(_a)
