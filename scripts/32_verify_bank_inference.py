#!/usr/bin/env python
"""추론 방식 검증 (재학습 없음): 배분=train 템플릿, latent=train latent bank.

측정 결과 현재 추론(prior + 균등 배분)이 SC 상관 0.67 인데, 배분을 train 평균 비율로 두고
latent 를 train streamline 의 KDE bank 에서 뽑으면 test 3명에서 0.89 가 나왔다. 다만 그 실험은
ROI pair 목록을 GT(bundles.npz) 에서 가져와 완전한 T1-only 가 아니었다. 여기서는 pair 선택도
T1 만으로(edge head) 또는 GT 무관하게(템플릿 비영) 하고 31명 전원에서 확인한다.

  python scripts/32_verify_bank_inference.py --n-bank-subj 12 --total 100000
결과: outputs/eval/bank_verify_<ckpt>.json
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
from atm_sc.data.dataset import ROIPairSubject                    # noqa: E402
from atm_sc.data.paths import ATLAS                               # noqa: E402
from atm_sc.data.roi_groups import block_masks, tier_masks        # noqa: E402
from atm_sc.data.tt_io import hard_sc                             # noqa: E402
from atm_sc.inference.generate_sc import select_pairs             # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM, from_checkpoint                      # noqa: E402
from atm_sc.training.run import t1_input                          # noqa: E402

IU = np.triu_indices(82, 1)


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ccc(p, g):
    mp, mg = p.mean(), g.mean()
    return float(2 * ((p - mp) * (g - mg)).mean() / (p.var() + g.var() + (mp - mg) ** 2 + 1e-8))


def build_template(subs, key):
    acc = np.zeros((82, 82))
    for s in subs:
        acc += np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")[key]
    return acc / len(subs)


def build_bank(m, subs, n_per_pair, dev, t1_src="syn"):
    """train subject 의 실제 streamline 을 그 subject 의 condition 으로 encode 해 pair 별로 모은다."""
    bank, rng = {}, np.random.default_rng(0)
    for s in subs:
        b = np.load(ROOT / f"outputs/roi_pairs/{s}/bundles.npz")
        pairs, off, SS = b["pair_ids"].astype(np.int64), b["pair_offsets"], b["streamlines"]
        with torch.no_grad():
            feat = m.atm.encode_anatomy(t1_input(m, s, t1_src))
        idx, pr = [], []
        for k in range(len(pairs)):
            avail = off[k + 1] - off[k]
            n = min(n_per_pair, avail)
            if n <= 0:
                continue
            idx.append(off[k] + rng.choice(avail, n, replace=False))
            pr.append(np.repeat(pairs[k][None], n, axis=0))
        idx, pr = np.concatenate(idx), np.concatenate(pr)
        for i in range(0, len(idx), 20000):
            S = torch.as_tensor(SS[idx[i:i + 20000]].astype(np.float32), device=dev)
            P = torch.as_tensor(pr[i:i + 20000], device=dev)
            with torch.no_grad():
                mu, _ = m.encode_streamlines(S, m.condition(feat, P))
            mu = mu.cpu().numpy().astype(np.float32)
            for t, (a_, b_) in enumerate(pr[i:i + 20000]):
                bank.setdefault((int(a_), int(b_)), []).append(mu[t])
        print(f"  bank += {s}  ({len(bank)} pair)", flush=True)
    out = {}
    for k, v in bank.items():
        z = np.stack(v)
        # Silverman 대역폭: 차원마다 std * n^(-1/(D+4))
        out[k] = (z, z.std(0) * len(z) ** (-1.0 / (z.shape[1] + 4)))
    return out


@torch.no_grad()
def generate(m, feat, pairs, counts, bank, atlas, affine, dev, D, batch=20000, seed=0):
    """-> (pass SC [82,82], edge 별 평균 길이 [82,82], 가닥당 방문 쌍, 총 가닥, bank 적중률)"""
    rep = np.repeat(np.asarray(pairs, np.int64), np.asarray(counts, np.int64), axis=0)
    assert len(rep) > 0, "생성할 가닥이 없음"
    g = torch.Generator(device=dev); g.manual_seed(seed)
    rs = np.random.default_rng(seed)
    W = np.zeros((82, 82), np.float64); Ssum = np.zeros((82, 82), np.float64)
    n_hit = 0
    for i in range(0, len(rep), batch):
        blk = rep[i:i + batch]
        P = torch.as_tensor(blk, device=dev)
        c = m.condition(feat, P)
        z = m.prior_mean(P) + torch.randn(len(blk), D, device=dev, generator=g)
        if bank is not None:
            zs = np.empty((len(blk), D), np.float32)
            hit = np.zeros(len(blk), bool)
            for t in range(len(blk)):
                e = bank.get((int(blk[t, 0]), int(blk[t, 1])))
                if e is None:
                    continue
                v, h = e
                zs[t] = v[rs.integers(0, len(v))] + h * rs.standard_normal(D).astype(np.float32)
                hit[t] = True
            if hit.any():
                zt = torch.as_tensor(zs, device=dev); ht = torch.as_tensor(hit, device=dev)
                z = torch.where(ht[:, None], zt, z)
            n_hit += int(hit.sum())
        S = m.decode(z, c).cpu().numpy().astype(np.float32)
        npts = np.full(len(S), S.shape[1], np.int64)
        w_, s_ = hard_sc(S.reshape(-1, 3), npts, atlas, affine, 82, "pass")
        W += w_; Ssum += s_
    L = np.divide(Ssum, W, out=np.zeros_like(Ssum), where=W > 0)
    assert np.isfinite(W).all() and W.sum() > 0, "생성 SC 가 비었음"
    return W, L, W[IU].sum() / len(rep), len(rep), n_hit / len(rep)


def metrics(W, L, subj, tmpl_pass, masks):
    gt = np.asarray(subj.sc_mat, np.float64)[IU]
    gl = np.asarray(subj.len_mat, np.float64)[IU]
    p, pl, t = W[IU], L[IU], tmpl_pass[IU]
    o = {"r": corr(p, gt), "r_log": corr(np.log1p(p), np.log1p(gt)), "ccc": ccc(p, gt),
         "residual_r": corr(p - t, gt - t)}
    ok = (gl > 0) & (pl > 0)
    o.update({"len_rmse": float(np.sqrt(((pl[ok] - gl[ok]) ** 2).mean())),
              "len_mae": float(np.abs(pl[ok] - gl[ok]).mean()),
              "len_r": corr(pl[ok], gl[ok]), "len_n": int(ok.sum()),
              "gen_len_mean": float(pl[pl > 0].mean())})
    for name, msk in masks.items():
        mm = msk[IU]
        if mm.sum() > 2:
            o[f"r_{name}"] = corr(p[mm], gt[mm])
    return o


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint 메타(in_channels / template / t1_source)로 모델을 만든다.
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    t1_src = sd.get("t1_source", "syn")
    img = nib.load(ATLAS); atlas = np.asarray(img.dataobj).astype(np.int16); affine = img.affine
    train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()]
    test = [l.strip() for l in open(ROOT / "outputs/splits/test.txt") if l.strip()]
    if a.limit:
        test = test[:a.limit]

    t_end, t_pass = build_template(train, "sc_end"), build_template(train, "sc_pass")
    assert t_end.sum() > 0 and t_pass.sum() > 0
    print(f"템플릿: sc_end 비영 {int((t_end[IU] > 0).sum())} pair, "
          f">= {a.tmpl_thr} 인 pair {int((t_end[IU] >= a.tmpl_thr).sum())}", flush=True)

    print(f"latent bank 구축 (train {a.n_bank_subj}명)...", flush=True)
    bank = build_bank(m, train[:a.n_bank_subj], a.bank_per_pair, dev, t1_src)
    D = next(iter(bank.values()))[0].shape[1]
    print(f"bank: {len(bank)} pair, 평균 {np.mean([len(v) for v, _ in bank.values()]):.0f}개, D={D}", flush=True)

    # 템플릿 비영 pair (GT/edge head 모두 불필요 -> 완전 배포 가능)
    ti, tj = IU[0][t_end[IU] >= a.tmpl_thr], IU[1][t_end[IU] >= a.tmpl_thr]
    pairs_tmpl = np.stack([ti, tj], 1)

    rows = []
    for n, s in enumerate(test, 1):
        subj = ROIPairSubject(s)
        masks = dict(block_masks(82)); masks.update(tier_masks(np.asarray(subj.sc_mat)))
        with torch.no_grad():
            feat = m.atm.encode_anatomy(t1_input(m, s, t1_src))
            pairs_eh, _ = select_pairs(m, feat, 82, thr=a.edge_thr)
        gt = np.asarray(subj.sc_mat, np.float64)[IU]
        o = {"subject": s, "n_pairs_edgehead": int(len(pairs_eh)), "n_pairs_tmpl": int(len(pairs_tmpl)),
             "template_r": corr(t_pass[IU], gt), "template_ccc": ccc(t_pass[IU], gt)}
        t0 = time.time()
        for tag, prs, bk in [("bank_edgehead", pairs_eh, bank), ("bank_tmplpairs", pairs_tmpl, bank),
                             ("prior_edgehead", pairs_eh, None)]:
            cnt = t_end[prs[:, 0], prs[:, 1]]
            if tag == "prior_edgehead":                     # 현재 코드 = prior + 균등 배분
                cnt = np.ones(len(prs))
            cnt = np.maximum(np.round(np.maximum(cnt, 0) * a.total / max(cnt.sum(), 1e-9)), 1).astype(np.int64)
            W, L, vis, ntot, hit = generate(m, feat, prs, cnt, bk, atlas, affine, dev, D, seed=a.seed)
            o[tag] = metrics(W, L, subj, t_pass, masks)
            o[tag].update({"visits_per_streamline": vis, "n_streamlines": ntot, "bank_hit": hit})
        o["gt_visits_per_streamline"] = float(np.asarray(subj.sc_mat)[IU].sum() / np.load(
            ROOT / f"outputs/roi_pairs/{s}/assignments.npz")["n_assigned"])
        o["sec"] = time.time() - t0
        rows.append(o)
        print(f"[{n}/{len(test)}] {s}  bank_eh r={o['bank_edgehead']['r']:.3f} "
              f"len_rmse={o['bank_edgehead']['len_rmse']:.1f}  "
              f"bank_tmpl r={o['bank_tmplpairs']['r']:.3f}  "
              f"prior r={o['prior_edgehead']['r']:.3f}  tmpl r={o['template_r']:.3f} "
              f"({o['sec']:.0f}s)", flush=True)

    summ = {"n_subjects": len(rows), "ckpt": a.ckpt, "total_streamlines": a.total,
            "n_bank_subj": a.n_bank_subj, "edge_thr": a.edge_thr, "tmpl_thr": a.tmpl_thr}
    for tag in ("bank_edgehead", "bank_tmplpairs", "prior_edgehead"):
        keys = [k for k in rows[0][tag] if isinstance(rows[0][tag][k], float)]
        summ[tag] = {k: float(np.nanmean([r[tag][k] for r in rows])) for k in keys}
    for k in ("template_r", "template_ccc", "gt_visits_per_streamline", "n_pairs_edgehead", "n_pairs_tmpl"):
        summ[k] = float(np.mean([r[k] for r in rows]))
    out = ROOT / f"outputs/eval/bank_verify_{Path(a.ckpt).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "per_subject": rows}, open(out, "w"), indent=1, ensure_ascii=False)

    print(f"\n==== 평균 (test {len(rows)}명, 총 {a.total:,} 가닥) ====")
    print(f"{'구성':18s} {'r':>7s} {'r_log':>7s} {'CCC':>7s} {'잔차r':>7s} {'길이RMSE':>9s} {'길이r':>7s} {'방문쌍':>7s}")
    for tag in ("prior_edgehead", "bank_edgehead", "bank_tmplpairs"):
        d = summ[tag]
        print(f"{tag:18s} {d['r']:7.3f} {d['r_log']:7.3f} {d['ccc']:7.3f} {d['residual_r']:7.3f} "
              f"{d['len_rmse']:9.1f} {d['len_r']:7.3f} {d['visits_per_streamline']:7.2f}")
    print(f"{'그룹 템플릿':18s} {summ['template_r']:7.3f} {'':7s} {summ['template_ccc']:7.3f}")
    print(f"GT 가닥당 방문 쌍 = {summ['gt_visits_per_streamline']:.2f}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt")
    ap.add_argument("--total", type=int, default=100_000)
    ap.add_argument("--n-bank-subj", type=int, default=12)
    ap.add_argument("--bank-per-pair", type=int, default=24)
    ap.add_argument("--edge-thr", type=float, default=0.5)
    ap.add_argument("--tmpl-thr", type=float, default=1.0, help="템플릿 sc_end 가 이 값 이상인 pair 만 생성")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
