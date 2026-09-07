#!/usr/bin/env python
"""학습된 조건부 prior p(z|pair) 를 checkpoint 단위로 채점한다 (W3-a).

  python scripts/50_prior_score.py --ckpt A.pt --ckpt B.pt

왜 48번을 그냥 못 쓰나
----------------------
`scripts/48_prior_ladder.py` 는 `outputs/eval/w1e_latent_cache.npz` (p4_joint 로 만든 캐시)를
읽고, 후보 prior 를 **오프라인 적합**해 사다리를 만든다. 거기서 'current' 는 언제나
N(mu_pair, I) 다 -- 즉 **학습된 sigma 를 반영하지 않는다**. 우리가 알고 싶은 것은
"학습이 끝난 모델이 실제로 쓰는 prior 가 몇 점인가" 라서, 같은 지표·같은 held-out 분할로
checkpoint 의 prior 를 직접 채점한다. 공유 캐시는 건드리지 않는다 (48번/W2-b 결과 보존).

같게 맞춘 것 (값이 직접 비교되도록)
  * pair 선택: 46번 `select_pairs` 그대로 (tier x block 라운드로빈, min_n=40)
  * 절반 분할: `np.random.default_rng(2000 + k)` -- 46/48 과 같은 규약
  * 채점: `evaluation.prior_metrics.evaluate_prior` (n_sample 256, n_perm 200), pair 별 중앙값
  * 'powered' = 그 pair 의 총 가닥 수 >= 200 (48번의 power_n)

후보
  current  N(mu_pair, I)                        지금까지의 prior. 학습 전 checkpoint 에서는 learned 와 같다.
  learned  N(mu_pair, diag(sigma_pair^2))       checkpoint 가 실제로 샘플링에 쓰는 분포.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                     # noqa: E402
from atm_sc.evaluation import prior_metrics as pm                  # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                  # noqa: E402
from atm_sc.models.roi_pair_embedding import canonical_pairs       # noqa: E402
from atm_sc.training.run import CACHE, T1_SOURCES, t1_input        # noqa: E402

MKEYS = ("nll", "nll_per_dim", "mmd_mmd2", "mmd_p", "precision_ratio", "recall_ratio", "d_gt_self")


def _select_pairs():
    """46번의 pair 선택 규약을 그대로 쓴다 (import 만; 46번은 수정하지 않는다)."""
    spec = importlib.util.spec_from_file_location("m46", ROOT / "scripts/46_latent_modality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.select_pairs


def encode(ckpt: Path, subs_all: list[str], a) -> list[dict]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ckpt, device=dev)
    t1_src = sd.get("t1_source", "syn")
    suffix = T1_SOURCES[t1_src][0]
    subs = [s for s in subs_all if (CACHE / f"{s}{suffix}").exists()][: a.n_subj]
    assert len(subs) >= 3, len(subs)
    select_pairs = _select_pairs()
    rng = np.random.default_rng(a.seed)
    conds = []
    with torch.no_grad():
        for si, sub in enumerate(subs):
            subj = ROIPairSubject(sub)
            ks, _ = select_pairs(subj, a.pairs_per_subject, a.min_n, rng)
            feat = m.atm.encode_anatomy(t1_input(m, sub, t1_src))
            assert feat.shape == (1, 512) and torch.isfinite(feat).all(), feat.shape
            for k in ks:
                S, _ = subj.get_pair(int(k))
                S = S.to(dev)
                pid = np.asarray(subj.pair_ids[k], np.int64)[None]
                pt = torch.as_tensor(pid, device=dev).repeat(S.shape[0], 1)
                mu, logvar = m.encode_streamlines(S, m.condition(feat, pt))
                mu_p, ls_p = m.pair_emb.prior_params(canonical_pairs(torch.as_tensor(pid, device=dev)))
                z = mu.float().cpu()
                assert z.shape == (S.shape[0], 64) and torch.isfinite(z).all(), z.shape
                assert float(z.abs().max()) > 0, f"{sub} pair {k}: latent 이 전부 0"
                n = z.shape[0]
                perm = np.random.default_rng(2000 + len(conds)).permutation(n)   # 46/48 과 같은 규약
                conds.append({"k": len(conds), "n": n, "z": z,
                              "ev": torch.as_tensor(perm[n // 2:].copy()),
                              "mu_p": mu_p[0].float().cpu(), "ls_p": ls_p[0].float().cpu(),
                              "sub": sub, "roi": (int(pid[0, 0]), int(pid[0, 1]))})
            print(f"  {sub}: pair {len(ks)}개 (누적 {len(conds)})", flush=True)
    ls_all = torch.stack([c["ls_p"] for c in conds])
    meta = {"ckpt": str(ckpt.relative_to(ROOT)), "phase": str(sd.get("phase")), "step": int(sd.get("step", 0)),
            "t1_source": t1_src, "n_subj": len(subs), "n_cond": len(conds),
            "prior_log_sigma_mean": float(ls_all.mean()), "prior_sigma_mean": float(ls_all.exp().mean()),
            "prior_sigma_min": float(ls_all.exp().min()), "prior_sigma_max": float(ls_all.exp().max()),
            "mean_offset_from_mu_pair": float(torch.stack(
                [(c["z"].mean(0) - c["mu_p"]).norm() for c in conds]).mean())}
    del m
    torch.cuda.empty_cache()
    return conds, meta


def agg(rows) -> dict:
    return {k: float(np.median([r[k] for r in rows])) for k in MKEYS} | {
        "frac_mmd_significant": float(np.mean([r["mmd_p"] <= 0.05 for r in rows])),
        "n_pairs": len(rows), "n_eval_median": float(np.median([r["n_gt"] for r in rows]))}


def score(conds, a) -> dict:
    rows = {"current": [], "learned": []}
    for c in conds:
        zev = c["z"][c["ev"]]
        assert zev.ndim == 2 and zev.shape[1] == 64 and torch.isfinite(zev).all()
        cand = {"current": pm.GaussianPrior(c["mu_p"], 0.0),
                "learned": pm.GaussianPrior(c["mu_p"], 2.0 * c["ls_p"])}
        for name, p in cand.items():
            r = pm.evaluate_prior(p, zev, n_sample=a.n_prior_sample, seed=c["k"], n_perm=a.n_perm)
            rows[name].append(r | {"k": c["k"], "n": c["n"]})
    out = {"all": {k: agg(v) for k, v in rows.items()}}
    pw = {k: [r for r in v if r["n"] >= a.power_n] for k, v in rows.items()}
    if len(pw["current"]) >= 20:
        out["powered"] = {k: agg(v) for k, v in pw.items()}
    return out


def main(a) -> int:
    subs_all = [s.strip() for s in (ROOT / a.split).read_text().splitlines() if s.strip()]
    res = {"cmd": "python scripts/50_prior_score.py " + " ".join(f"--ckpt {c}" for c in a.ckpt)
                  + f" --n-subj {a.n_subj} --pairs-per-subject {a.pairs_per_subject}"
                  + f" --power-n {a.power_n} --n-perm {a.n_perm} --seed {a.seed}",
           "protocol": {"split": a.split, "n_subj": a.n_subj, "pairs_per_subject": a.pairs_per_subject,
                        "min_n": a.min_n, "power_n": a.power_n, "n_prior_sample": a.n_prior_sample,
                        "n_perm": a.n_perm, "seed": a.seed,
                        "note": "46/48 과 같은 pair 선택·절반 분할·채점 함수. 공유 캐시는 쓰지 않는다."},
           "checkpoints": {}}
    for ck in a.ckpt:
        t0 = time.time()
        print(f"\n== {ck}", flush=True)
        conds, meta = encode(ROOT / ck, subs_all, a)
        s = score(conds, a)
        key = Path(ck).stem
        res["checkpoints"][key] = {"meta": meta, "ladder": s, "sec": time.time() - t0}
        for grp, v in s.items():
            print(f"  [{grp}] current prec={v['current']['precision_ratio']:.3f} · "
                  f"learned prec={v['learned']['precision_ratio']:.3f} "
                  f"(n_pairs={v['current']['n_pairs']})", flush=True)
    out = ROOT / "outputs/eval" / a.out
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n저장: {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--split", default="outputs/splits/val.txt")
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--pairs-per-subject", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--power-n", type=int, default=200)
    ap.add_argument("--n-prior-sample", type=int, default=256)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="w3a_prior_score.json")
    sys.exit(main(ap.parse_args()))
