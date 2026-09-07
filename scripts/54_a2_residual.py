#!/usr/bin/env python
"""A2 template+residual — 전략 문서 §1·§4·§5 의 Phase 2 최소 실험.

  python scripts/54_a2_residual.py --selfcheck-only
  python scripts/54_a2_residual.py --arm D          # 권장 조합 L_res + 0.2 L_corr + 0.2 L_diff
  python scripts/54_a2_residual.py --arm A          # L_res 만
  python scripts/54_a2_residual.py --arm shuffled   # 음성 대조 (T1 을 한 칸 밀어 넣는다)

A1(53번) 과 무엇이 다른가
------------------------
A1 은 절대 count 손실을 켠 채 EMA 잔차 상관만 얹었다 -- 전략 문서가 경계하는 구성이다
("초기 실험에서는 absolute SC RMSE, 전체 scale loss, whole SC magnitude loss를 끈다").
A2 는 문서대로 간다:
  * 절대 count 손실 **끔** (count weight 0)
  * 타깃 = train split 전용 log1p 템플릿을 빼고 edge 별 std 로 정규화한 잔차
  * L_res (SmoothL1) 를 주손실, L_corr / L_diff 를 ablation
  * L_diff 를 위해 한 step 에 subject 2명

판정 (문서 §8.1): val 31명의 residual r, self vs shuffled vs zero, 식별 정확도,
피험자 간 상관, variance ratio, difference correlation.
"""
import argparse
import copy
import json
import os
import socket
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
from atm_sc.data.local_feats import load_roi_feats, local_dim, pair_local  # noqa: E402
from atm_sc.data.sc_template import load_or_build                       # noqa: E402
from atm_sc.models.endpoint_assigner import EndpointAssigner            # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.training import config as C                                 # noqa: E402
from atm_sc.training.run import anatomy_feature, run, t1_input           # noqa: E402
from atm_sc.training.trainer import Trainer                             # noqa: E402

LOCK = ROOT / "outputs/gpu.lock"
EVAL = ROOT / "outputs/eval"
STATS = CACHE / "sc_template_stats.npz"

# 전략 문서 §4.5 의 ablation 표. count(절대 손실)는 전 arm 에서 0 이다.
ARMS = {
    "A": {"res": 1.0, "res_corr": 0.0, "diff": 0.0},     # residual target 자체의 학습 가능성
    "B": {"res": 1.0, "res_corr": 0.2, "diff": 0.0},     # + edge 별 증감 패턴
    "C": {"res": 1.0, "res_corr": 0.0, "diff": 0.2},     # + 동일 출력 collapse 억제
    "D": {"res": 1.0, "res_corr": 0.2, "diff": 0.2},     # 권장 조합
    # 문서 §4.4 의 E. diff_ratio 0.055 (예측 개인차가 GT 의 5.5%) 를 직격한다.
    # 정보를 늘리는 항이 아니라 **있는 정보를 출력 진폭으로 내보내는** 항이다.
    "V": {"res": 1.0, "res_corr": 0.2, "diff": 0.2, "var": 0.5},
    "V2": {"res": 0.5, "res_corr": 0.3, "diff": 0.3, "var": 1.0},   # 진폭 우선
}


def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    me = f"{socket.gethostname()}:{os.getpid()}"
    if lock.exists():
        txt = lock.read_text().strip()
        parts = txt.split(":")
        host, pid = (parts[0], parts[-1]) if len(parts) >= 2 else (txt, "-1")
        alive = False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive = True
            except (OSError, ValueError):
                alive = not pid.lstrip("-").isdigit()
        if alive or (host != socket.gethostname() and time.time() - lock.stat().st_mtime < stale_sec):
            print(f"lock 보유 중: {txt} -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def release_lock(lock: Path) -> None:
    if lock.exists() and lock.read_text().strip().endswith(f":{os.getpid()}"):
        lock.unlink()


# ─────────────────────────────────────────────── 평가 (전략 문서 §8.1)
def resid_eval(model, subs: list[str], stats: dict, t1_source: str, init_bundle: str = "AF_L",
               n_perm: int = 200, seed: int = 0) -> dict:
    """val subject 의 count head 예측으로 개인차 지표 일습. 생성 경로는 안 쓴다 (싸다)."""
    R = int(model.n_roi)
    iu = np.triu_indices(R, 1)
    Pt = torch.as_tensor(np.stack(iu, 1).astype(np.int64), device=model.device)
    tpl, sd, msk = stats["template"], stats["std"], stats["mask"].astype(bool)

    feats, G = [], []
    for s in subs:
        # run() 과 **같은 경로**로 anatomy 를 얻는다 (unet_level=none 이면 디스크 캐시).
        # 여기서 UNet 을 다시 돌리면 값은 같지만 메모리를 2 GB 씩 먹는다.
        a = anatomy_feature(model, s, init_bundle, source=t1_source)
        assert float(a.norm()) > 1e-3, f"{s}: anatomy feature 가 0 에 가깝다"
        feats.append(a)
        G.append(np.log1p(np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu]))
    G = np.stack(G)
    assert np.isfinite(G).all(), "GT 에 NaN"

    # count head 가 국소 feature 를 받도록 만들어졌으면 평가에서도 같은 입력을 준다.
    ld = int(getattr(model.count_head, "local_dim", 0) or 0)
    locs = [pair_local(load_roi_feats(s, t1_source, model.device, n_roi=R), Pt) if ld else None
            for s in subs]
    if ld:
        assert locs[0].shape == (Pt.shape[0], ld), (locs[0].shape, ld)

    def pred_log1p(a, loc=None):
        v = torch.nn.functional.softplus(model.edge_log_counts(a, Pt, loc)).double().cpu().numpy()
        assert np.isfinite(v).all(), "count head 예측에 NaN/Inf"
        return v

    P = np.stack([pred_log1p(a, l) for a, l in zip(feats, locs)])
    # shuffled: 전역과 국소를 **같은** 이웃 subject 것으로 바꾼다 (한쪽만 바꾸면 대조가 성립 안 함)
    P_shuf = np.stack([pred_log1p(feats[(i + 1) % len(subs)], locs[(i + 1) % len(subs)])
                       for i in range(len(subs))])
    # zero T1 은 캐시가 없으므로 UNet 을 한 번만 돈다 (subject 무관이라 1회면 충분).
    a_zero = model.atm.encode_anatomy(torch.zeros_like(t1_input(model, subs[0], t1_source)))
    l_zero = torch.zeros_like(locs[0]) if ld else None
    P_zero = np.broadcast_to(pred_log1p(a_zero, l_zero), P.shape)
    del a_zero
    torch.cuda.empty_cache()

    def rr(A, B):                                  # 행별 상관
        a = A - A.mean(1, keepdims=True); b = B - B.mean(1, keepdims=True)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    def resid_r(A):                                # 템플릿(고정 train 통계) 제거 후 상관
        dp = (A[:, msk] - tpl[msk]) / sd[msk]
        dg = (G[:, msk] - tpl[msk]) / sd[msk]
        dp = dp - dp.mean(0, keepdims=True); dg = dg - dg.mean(0, keepdims=True)   # LOO 근사
        return float(np.mean(rr(dp, dg)))

    dp = (P[:, msk] - tpl[msk]) / sd[msk]; dp -= dp.mean(0, keepdims=True)
    dg = (G[:, msk] - tpl[msk]) / sd[msk]; dg -= dg.mean(0, keepdims=True)
    n = len(subs)

    # 식별: 각 예측이 어느 GT 와 가장 닮았나 (잔차 공간)
    Sm = (dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-12)) @ \
         (dg / (np.linalg.norm(dg, axis=1, keepdims=True) + 1e-12)).T
    ident = float((Sm.argmax(1) == np.arange(n)).mean())
    rng = np.random.default_rng(seed)
    null = [float((Sm[rng.permutation(n)].argmax(1) == np.arange(n)).mean()) for _ in range(n_perm)]

    # 모든 subject 쌍의 차분 상관 (동일 출력 shortcut 직접 평가)
    i_, j_ = np.triu_indices(n, 1)
    Dp, Dg = dp[i_] - dp[j_], dg[i_] - dg[j_]
    diff_r = float(np.mean(rr(Dp, Dg)))

    def inter(A):
        Z = A - A.mean(1, keepdims=True)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        M = Z @ Z.T
        return float(M[np.triu_indices(n, 1)].mean())

    return {"n_subjects": n,
            "resid_r": resid_r(P), "resid_r_shuffled": resid_r(P_shuf), "resid_r_zero": resid_r(P_zero),
            "abs_r_self": float(np.mean(rr(P, G))), "abs_r_shuffled": float(np.mean(rr(P_shuf, G))),
            "identification": ident, "identification_chance": 1.0 / n,
            "identification_null_mean": float(np.mean(null)),
            "identification_p": float((np.array(null) >= ident).mean()),
            "diff_corr": diff_r,
            "variance_ratio": float(dp.std() / max(dg.std(), 1e-12)),
            "inter_subj_r_pred": inter(P), "inter_subj_r_gt": inter(G)}


# ─────────────────────────────────────────────── 자기검증
def selfcheck(built: dict, stats: dict, train_subs: list[str], device: str) -> dict:
    ld = int(getattr(built["cfg"], "count_local_dim", 0) or 0)
    out = {}
    # (1) 누수: 템플릿은 train 만으로 만들어졌는가
    used = set(stats["subjects"].tolist())
    for split in ("val", "test"):
        other = {l.strip() for l in (ROOT / f"outputs/splits/{split}.txt").read_text().splitlines() if l.strip()}
        assert not (used & other), f"템플릿에 {split} subject 가 섞였다: {sorted(used & other)[:3]}"
    out["template_train_only"] = True
    out["template_n"] = len(used)
    assert used <= set(train_subs), "템플릿 subject 가 train 목록 밖이다"

    cfg, w = built["cfg"], built["weights"]
    cfg.active = {"count"}
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), nib.load(ATLAS).affine, tau=0.5,
                          device=device, d_bg=None if cfg.sc_mode == "endpoint" else 2.0)
    s0, s1 = ROIPairSubject(built["subjects"][0]), ROIPairSubject(built["subjects"][1])
    m0, _ = from_checkpoint(built["resume"], device=device, count_local_dim=ld)
    ib = built["init_bundle"]
    a0 = anatomy_feature(m0, s0.sub, ib, source=built["t1_source"])
    a1 = anatomy_feature(m0, s1.sub, ib, source=built["t1_source"])
    del m0
    torch.cuda.empty_cache()

    def one(weights):
        mm, _ = from_checkpoint(built["resume"], device=device, count_local_dim=ld)
        cc = copy.deepcopy(cfg); cc.active = {"count"}
        tr = Trainer(mm, ea, cc, weights)
        o = tr.step(s0, a0, partner=(s1, a1) if (weights.diff > 0 or weights.var > 0) else None)
        has_stats = tr.rstats is not None
        del tr, mm
        torch.cuda.empty_cache()
        return has_stats, o

    # (2) 새 손실이 전부 0 이면 기존 경로와 같다
    w0 = copy.deepcopy(w); w0.res = w0.res_corr = w0.diff = w0.var = 0.0
    has0, o0 = one(w0)
    assert (not has0) and "L_res" not in o0, "가중치 0 인데 잔차 손실이 켜졌다"

    # (3) 세 손실이 각각 켜지고 기울기가 count head 에 닿는가
    for name, kw in (("res", {"res": 1.0}), ("res_corr", {"res_corr": 1.0}),
                     ("diff", {"diff": 1.0}), ("var", {"var": 1.0})):
        ww = copy.deepcopy(w); ww.res = ww.res_corr = ww.diff = ww.var = 0.0; ww.count = 0.0
        for k, v in kw.items():
            setattr(ww, k, v)
        _, o = one(ww)
        key = {"res": "L_res", "res_corr": "L_res_corr", "diff": "L_diff", "var": "L_var"}[name]
        assert key in o, f"{name} 손실이 안 켜졌다: {sorted(o)}"
        out[f"{key}_value"] = o[key]
        out[f"{key}_gradnorm"] = o["grad_norm_total"]
        assert o["grad_norm_total"] > 0, f"{name} 손실이 기울기를 안 흘린다"
    out["resid_var_ratio_at_init"] = o0.get("resid_var_ratio")

    # (4) 상수 예측이면 L_diff 를 피할 수 없다 (shortcut 차단이 실제로 작동하는가)
    from atm_sc.losses import subject_diff_loss
    e = int(stats["mask"].sum())
    z = torch.zeros(e)
    g1, g2 = torch.randn(e), torch.randn(e)
    l_const = float(subject_diff_loss(z, z, g1, g2))
    out["L_diff_when_constant"] = l_const
    assert l_const > 0.5, f"상수 예측인데 L_diff 가 {l_const:.3f} 로 작다 -- shortcut 차단이 안 된다"

    # (5) 국소 가지는 0-init -- 켜도 시작 예측이 전역 전용과 bit-exact 여야 한다
    if ld:
        R = int(stats["n_roi"])
        mg, _ = from_checkpoint(built["resume"], device=device)                    # 전역 전용
        ml, _ = from_checkpoint(built["resume"], device=device, count_local_dim=ld)  # 국소 가지 추가
        iu = np.triu_indices(R, 1)
        Pt = torch.as_tensor(np.stack(iu, 1).astype(np.int64), device=device)
        fr = load_roi_feats(s0.sub, built["t1_source"], device, n_roi=R)
        with torch.no_grad():
            v0 = mg.edge_log_counts(a0, Pt)
            v1 = ml.edge_log_counts(a0, Pt, pair_local(fr, Pt))
        d = float((v0 - v1).abs().max())
        out["local_zero_init_max_abs_diff"] = d
        assert d == 0.0, f"국소 가지 0-init 인데 예측이 바뀐다 (max|diff| = {d:.3e})"
        # 국소 feature 가 실제로 subject 마다 다른가
        fr2 = load_roi_feats(s1.sub, built["t1_source"], device, n_roi=R)
        out["local_feat_cosine_two_subj"] = float(
            torch.nn.functional.cosine_similarity(fr.flatten(), fr2.flatten(), dim=0))
        del mg, ml
        torch.cuda.empty_cache()

    # (6) GT 잔차 신호
    out["resid_share"] = float(stats["resid_share"])
    out["n_edges_used"] = e
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain/a2_residual.yaml")
    ap.add_argument("--arm", default="D", choices=sorted(ARMS))
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--local", action="store_true",
                    help="count head 에 ROI 국소 anatomy 를 넣는다 (전략 문서 §3.2). "
                         "실측: 전역 a512 는 subject 성분 2.1%%, ROI 국소는 13.4%%")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    built = C.build(C.load(ROOT / a.config))
    if a.steps:
        built["max_steps"] = a.steps
    for k, v in ARMS[a.arm].items():
        setattr(built["weights"], k, v)
    built["weights"].count = 0.0                    # 문서 §4.5: 절대 손실은 끈다
    tag = a.arm + ("_local" if a.local else "")
    if a.local:
        built["cfg"].count_local_dim = local_dim(built["t1_source"])
        print(f"[a2] ROI 국소 anatomy 사용: local_dim={built['cfg'].count_local_dim}", flush=True)
    built["out_dir"] = Path(str(built["out_dir"]) + f"_{tag}")
    stats = load_or_build(built["subjects"], STATS)
    built["cfg"].resid_stats = str(STATS)
    assert built["resume"] and built["resume"].exists(), built["resume"]

    EVAL.mkdir(parents=True, exist_ok=True)
    sc = selfcheck(built, stats, built["subjects"], a.device)
    (EVAL / "a2_residual_selfcheck.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2))
    if a.selfcheck_only:
        return

    val_subs = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()]
    trace = EVAL / f"a2_residual_trace_{tag}.jsonl"

    def hook(step, model):
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was = model.training
        t0 = time.time()
        try:
            with torch.no_grad():
                m = resid_eval(model, val_subs, stats, built["t1_source"],
                               init_bundle=built["init_bundle"])
        finally:
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
            model.train(was)
        row = {"arm": tag, "step": int(step), "sec": time.time() - t0, **m}
        with open(trace, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval/{tag}] step {step} resid_r={m['resid_r']:+.4f} "
              f"(shuf {m['resid_r_shuffled']:+.4f} zero {m['resid_r_zero']:+.4f}) "
              f"ident={m['identification']:.3f}/{m['identification_chance']:.3f} "
              f"diff_r={m['diff_corr']:+.4f} var={m['variance_ratio']:.3f} "
              f"inter={m['inter_subj_r_pred']:.5f} ({row['sec']:.0f}s)", flush=True)

    if not acquire_lock(LOCK):
        sys.exit(1)
    try:
        ck = run(phase=built["phase"], subjects=built["subjects"], max_steps=built["max_steps"],
                 out_dir=built["out_dir"], cfg=built["cfg"], weights=built["weights"],
                 init_bundle=built["init_bundle"], device=a.device, resume=built["resume"],
                 log_every=built["log_every"], trainable=built["trainable"],
                 unet_level=built["unet_level"], save_every=built["save_every"],
                 in_channels=built["in_channels"], template=built["template"],
                 t1_source=built["t1_source"], step_hook=hook, step_hook_every=a.eval_every)
    finally:
        release_lock(LOCK)

    rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    best = max(rows, key=lambda r: r["resid_r"]) if rows else None
    res = {"arm": tag, "weights": ARMS[a.arm], "local": bool(a.local), "config": a.config, "checkpoint": str(ck),
           "selfcheck": sc, "first": rows[0] if rows else None, "last": rows[-1] if rows else None,
           "best": best, "n_points": len(rows), "trace": str(trace)}
    (EVAL / f"a2_residual_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k] for k in ("arm", "weights", "first", "last", "best")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
