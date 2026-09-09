"""checkpoint 별 조건 개인화 / prior 정렬 / 생성 기하 비교.

인자: 이름:경로 를 여러 개. 예) J1:outputs/.../a.pt E1:outputs/.../b.pt
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from atm_sc.data.dataset import ROIPairSubject
from atm_sc.data.local_feats import load_roi_feats, pair_local
from atm_sc.data.paths import CACHE
from atm_sc.data.wm_segment import tissue_path
from atm_sc.filtering.wm_trim import sample_at
from atm_sc.models.roi_atm import from_checkpoint
from atm_sc.spaces import W_AFFINE
from atm_sc.training.run import anatomy_feature


def report(name, ckpt, subs_val, subs_tr, dev="cuda"):
    m, sd = from_checkpoint(ROOT / ckpt, device=dev)
    R = int(m.n_roi); rng = np.random.default_rng(0)
    P = torch.as_tensor(np.stack(np.triu_indices(R, 1), 1)[rng.choice(3321, 256, replace=False)],
                        device=dev, dtype=torch.long)
    # 1) 디코더 조건의 개인 성분
    C = []
    with torch.no_grad():
        for s in subs_val:
            a = anatomy_feature(m, s, "AF_L", "rigid"); fr = load_roi_feats(s, "rigid", dev, n_roi=R)
            lo = pair_local(fr, P) if m.pair_emb.local_proj is not None else None
            C.append(m.condition(a, P, local=lo).double().cpu().numpy().ravel())
    X = np.stack(C); iu = np.triu_indices(len(X), 1)
    cond_share = 100 * ((X - X.mean(0)) ** 2).sum() / (X ** 2).sum()
    cond_corr = float(np.corrcoef(X)[iu].mean())
    # 2) prior 가 posterior pair 평균을 얼마나 설명하나 (train)
    o0 = ROIPairSubject(subs_tr[0]); ks = rng.choice(o0.n_pairs, 192, replace=False)
    acc = {}
    with torch.no_grad():
        for s in subs_tr:
            o = ROIPairSubject(s); a = anatomy_feature(m, s, "AF_L", "rigid")
            fr = load_roi_feats(s, "rigid", dev, n_roi=R)
            for k in ks:
                if k >= o.n_pairs:
                    continue
                S, _ = o.get_pair(int(k)); n = min(8, len(S))
                if n < 2:
                    continue
                pid = torch.as_tensor(np.asarray(o.pair_ids[k], np.int64), device=dev).repeat(n, 1)
                lo = pair_local(fr, pid) if m.pair_emb.local_proj is not None else None
                mu, _ = m.encode_streamlines(S[:n].to(dev).float(), m.condition(a, pid, local=lo))
                acc.setdefault(int(k), []).append(mu.double().cpu().numpy())
    keys = sorted(acc); Q = np.stack([np.concatenate(acc[k]).mean(0) for k in keys])
    Pk = torch.as_tensor(np.asarray([o0.pair_ids[k] for k in keys], np.int64), device=dev)
    with torch.no_grad():
        a = anatomy_feature(m, subs_tr[0], "AF_L", "rigid"); fr = load_roi_feats(subs_tr[0], "rigid", dev, n_roi=R)
        lo = pair_local(fr, Pk) if m.pair_emb.prior_local is not None else None
        M = m.prior_params(Pk, anatomy=None, local=lo)[0].double().cpu().numpy()
    Qc, Mc = Q - Q.mean(0), M - M.mean(0)
    prior_share = 1 - ((Qc - Mc) ** 2).sum() / (Qc ** 2).sum()
    prior_dist = float(np.linalg.norm(Q - M, axis=1).mean())
    # 3) 생성 기하 (백질 점유, 길이)
    s = subs_val[0]
    z = np.load(tissue_path(CACHE, s, "syn")); wm = z["wm"].astype(np.float32) / 255.0
    with torch.no_grad():
        a = anatomy_feature(m, s, "AF_L", "rigid"); fr = load_roi_feats(s, "rigid", dev, n_roi=R)
        roi = fr if m.pair_emb.local_proj is not None else None
        g = torch.Generator(device=dev); g.manual_seed(0)
        S, _, _ = m.generate(a, P.repeat_interleave(8, 0), 1, generator=g, local_roi=roi)
    N = S.cpu().numpy().astype(np.float64)
    occ = sample_at(N, wm, np.asarray(W_AFFINE))
    T = N.shape[1]; lo_, hi_ = T // 8, T - T // 8
    geo = {"wm_occ_mid": float(occ[:, lo_:hi_].mean()),
           "length_mm": float(np.linalg.norm(np.diff(N, axis=1), axis=2).sum(1).mean())}
    del m; torch.cuda.empty_cache()
    return {"name": name, "cond_share_pct": cond_share, "cond_corr": cond_corr,
            "prior_explained": float(prior_share), "prior_dist": prior_dist, **geo}


def main(args):
    val = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()][:12]
    tr = [l.strip() for l in (ROOT / "outputs/splits/train.txt").read_text().splitlines() if l.strip()][:5]
    rows = []
    for spec in args:
        name, ck = spec.split(":", 1)
        try:
            rows.append(report(name, ck, val, tr))
        except Exception as e:
            print(f"{name}: 실패 {type(e).__name__}: {e}", flush=True)
    hdr = ["cond_share_pct", "cond_corr", "prior_explained", "prior_dist", "wm_occ_mid", "length_mm"]
    lbl = {"cond_share_pct": "조건 개인성분%", "cond_corr": "조건 subj상관", "prior_explained": "prior 설명몫",
           "prior_dist": "prior-post 거리", "wm_occ_mid": "생성 WM점유", "length_mm": "생성 길이mm"}
    print(f"\n{'항목':16s}" + "".join(f"{r['name']:>12s}" for r in rows))
    for k in hdr:
        print(f"{lbl[k]:16s}" + "".join(f"{r[k]:12.4f}" for r in rows))
    print(f"{'GT 기준':16s}" + f"{'':>12s}" * max(len(rows) - 1, 0) + "  WM점유 0.825 / 길이 103mm")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
