"""배분(라벨) -> 생성 trk -> SC 로 가면서 subject 개인차가 어디서 사라지는지 (val 전용 진단).

단계별 upper-tri 벡터의 LOO resid_r 을 같은 조건(edge thr, total, gain)으로 잰다:
  count   : pair count head 직접 예측
  label0  : count_head_end 배분 (gain 0) 을 pair 위치에 그대로 놓은 행렬
  label1  : count_head_end x exp(gain * count head 잔차) 배분  (= trk 의 pair 라벨)
  end     : 생성 trk 의 끝점 SC
  pass    : 생성 trk 의 pass-through SC  (49 가 보고하는 값)
추가로 가닥 표본에서 "배분한 pair 에 실제로 끝점이 닿는 비율" 과 가닥당 방문 ROI 수를 잰다.
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
from atm_sc.evaluation.reproduction_metrics import subject_specificity, loo_center, _corr
from atm_sc.inference.generate_sc import allocate_counts, pair_residual_log, generate_by_count, select_pairs
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.training.run import anatomy_feature, t1_input, roi_feats_if_needed


def main(a):
    import nibabel as nib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    a.t1_src = sd.get("t1_source", "syn"); a.init_bundle = "AF_L"
    assert m.count_head is not None and m.count_head_end is not None
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (CACHE / f"{s}_T1w_syn_W.npy").exists()][: a.limit or None]
    assert "test" not in a.subjects, "test 는 1회 소진됨 -- val 로만 진단"
    z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
    R = int(z["n_roi"]); iu = np.triu_indices(R, 1)
    a.template_log1p = np.zeros((R, R)); a.template_log1p[iu] = z["template"]; a.template_log1p.T[iu] = z["template"]
    from atm_sc.data.anat_tier1 import STATS_PATH
    z1 = np.load(STATS_PATH, allow_pickle=False); a.tier1_stats = {k: z1[k] for k in z1.files}
    keys = ["count", "label0", "label1", "end", "pass"]
    V = {k: [] for k in keys}; G = []; hits = []
    rs = np.random.default_rng(0)
    for si, s in enumerate(subs):
        t0 = time.time(); subj = ROIPairSubject(s); n_roi = subj.n_roi
        with torch.no_grad():
            # 49 와 같은 경로: unet_level none 이면 디스크 캐시 (= E8 이전 동결 인코더 출력)
            feat = (anatomy_feature(m, s, a.init_bundle, a.t1_src) if m.unet_level == "none"
                    else m.atm.encode_anatomy(t1_input(m, s, a.t1_src)))
            al, at = ev.aux_feats(m, s, n_roi, a, dev)
            pairs, prob = select_pairs(m, feat, n_roi, thr=a.edge_thr, local=al, tier1=at)
            roi = roi_feats_if_needed(m, s, a.t1_src)
            loc, t1f = ev.head_feats(m, s, pairs, a, dev)
            P = torch.as_tensor(pairs, device=dev)
            cnt = m.edge_count_matrix(feat, P, loc, t1f).cpu().numpy()
            rl = pair_residual_log(m, feat, pairs, a.template_log1p, loc, t1f)
            ai = ev._pair_rows(pairs, n_roi)
            kw = dict(local=None if al is None else al[ai], tier1=None if at is None else at[ai])
            n0 = allocate_counts(m, feat, pairs, total=a.total, **kw)
            n1 = allocate_counts(m, feat, pairs, total=a.total, resid_log=rl, resid_gain=a.gain, **kw)
        lab = {}
        for k, n in (("label0", n0), ("label1", n1)):
            M = np.zeros((n_roi, n_roi)); M[pairs[:, 0], pairs[:, 1]] = n; M.T[pairs[:, 0], pairs[:, 1]] = n; lab[k] = M
        sc, n_gen, _ = generate_by_count(m, feat, pairs, n1, atlas, aff, n_roi, local_roi=roi)
        # 가닥 표본: 배분한 pair 에 끝점이 닿는가, 몇 개 ROI 를 지나가는가
        rep = np.repeat(pairs, n1, axis=0); idx = rs.choice(len(rep), min(a.hit_sample, len(rep)), replace=False)
        blk = rep[idx]
        with torch.no_grad():
            g = torch.Generator(device=dev); g.manual_seed(1)
            S, _, _ = m.generate(feat, torch.as_tensor(blk, device=dev), 1, generator=g, local_roi=roi)
        S = S.cpu().numpy().astype(np.float32); N, T = S.shape[:2]
        pl = point_labels(S.reshape(-1, 3), atlas, aff).reshape(N, T)
        e0, e1 = pl[:, 0] - 1, pl[:, -1] - 1
        end_hit = ((e0 == blk[:, 0]) & (e1 == blk[:, 1])) | ((e0 == blk[:, 1]) & (e1 == blk[:, 0]))
        pk = roi_visit_sets(pl.reshape(-1), np.full(N, T, np.int64))
        n_visit = np.bincount(pk[:, 0], minlength=N)
        vis = np.zeros((N, n_roi), bool); vis[pk[:, 0], pk[:, 1]] = True
        pass_hit = vis[np.arange(N), blk[:, 0]] & vis[np.arange(N), blk[:, 1]]
        both_end_valid = (pl[:, 0] > 0) & (pl[:, -1] > 0) & (pl[:, 0] != pl[:, -1])
        h = dict(end_hit=float(end_hit.mean()), pass_hit=float(pass_hit.mean()), n_visit=float(n_visit.mean()),
                 end_valid=float(both_end_valid.mean()), n_pairs=int(len(pairs)), n_gen=int(n_gen),
                 resid_log_sd=float(np.std(rl)), pass_sum_ratio=float(sc["pass"]["sc"][iu].sum() / n_gen))
        hits.append(h)
        for k, M in (("count", cnt), ("label0", lab["label0"]), ("label1", lab["label1"]),
                     ("end", sc["end"]["sc"]), ("pass", sc["pass"]["sc"])):
            V[k].append(np.asarray(M, np.float64)[iu])
        G.append(np.asarray(subj.sc_mat, np.float64)[iu])
        print(f"[{si+1}/{len(subs)}] {s}: pair {len(pairs)} gen {n_gen:,} end_hit {h['end_hit']:.3f} "
              f"pass_hit {h['pass_hit']:.3f} visit {h['n_visit']:.1f} pass_sum/gen {h['pass_sum_ratio']:.2f} "
              f"{time.time()-t0:.0f}s", flush=True)
    G = np.stack(G); out = {"n_subjects": len(subs), "gain": a.gain, "total": a.total,
                            "hit": {k: float(np.mean([h[k] for h in hits])) for k in hits[0]}}
    Gc = loo_center(G)
    for k in keys:
        Pk = np.stack(V[k]); sp = subject_specificity(Pk, G)
        out[k] = {"resid_r": sp["resid_r"], "inter_subj_r": sp["inter_subj_r_pred"],
                  "abs_r": float(np.mean([_corr(Pk[i], G[i]) for i in range(len(G))]))}
    # 라벨 잔차가 end/pass 잔차로 얼마나 보존되는가 (subject 별 상관의 평균)
    Lc = loo_center(np.stack(V["label1"]))
    for k in ("end", "pass"):
        Kc = loo_center(np.stack(V[k]))
        out[k]["r_resid_vs_label1"] = float(np.nanmean([_corr(Lc[i], Kc[i]) for i in range(len(G))]))
    out["gt_inter_subj_r"] = float(subject_specificity(G, G)["inter_subj_r_gt"])
    p = ROOT / "outputs/eval/alloc_to_sc_val.json"; p.write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "outputs/eval/alloc_to_sc_val_vectors.npz", gt=G, subjects=np.array(subs),
                        **{k: np.stack(V[k]) for k in keys})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt")
    ap.add_argument("--subjects", default="outputs/splits/val.txt")
    ap.add_argument("--total", type=int, default=460000); ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--edge-thr", type=float, default=0.5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hit-sample", type=int, default=20000)
    main(ap.parse_args())
