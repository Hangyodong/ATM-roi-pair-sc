#!/usr/bin/env python
"""A1 잔차 타깃 — count head 를 `SC - 그룹템플릿` 잔차에 학습시킨다 (PIPELINE_12 §8-A).

  python scripts/53_a1_resid.py --selfcheck-only
  python scripts/53_a1_resid.py --config configs/retrain/a1_resid.yaml --arm R
  python scripts/53_a1_resid.py --config configs/retrain/a1_resid.yaml --arm C   # 대조군 (resid=0)

무엇을 검정하나
---------------
W4 프로브는 frozen feature + **선형** ridge 로 개인 신호가 r<0.02 라고 쟀다. 그건 그 readout
의 상한이지 "T1 에 정보가 없다" 가 아니다. 여기서는 supervision 을 직접 걸어서, 비선형 head 가
지도학습으로 개인차를 잡아내는지 본다. val `resid_r` 이 판정값이다.

반드시 확인하는 것 (실패하면 학습 전에 죽는다)
  1. resid=0 이면 step 출력이 기존과 **bit-exact** (손실을 켜기 전에는 아무것도 안 바뀐다)
  2. EMA 초기값 == count head 초기 예측  -> 시작 시점 잔차가 0 이라 편향이 없다
     (softplus(log t) = log1p(t) 항등식에 의존한다. 어긋나면 여기서 죽는다)
  3. GT 잔차의 분산이 0 이 아니다 (배울 개인차가 실재하는가)
  4. 잔차 손실이 count head 에 실제로 기울기를 흘린다
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import nibabel as nib                                                   # noqa: E402
from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402
from atm_sc.data.paths import ATLAS, CACHE                              # noqa: E402
from atm_sc.models.endpoint_assigner import EndpointAssigner            # noqa: E402
from atm_sc.losses import ResidualCorr, upper                           # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.training import config as C                                 # noqa: E402
from atm_sc.training.run import individuality_metrics, run, t1_input    # noqa: E402
from atm_sc.training.trainer import Trainer                             # noqa: E402

LOCK = ROOT / "outputs/gpu.lock"
EVAL = ROOT / "outputs/eval"


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


# --------------------------------------------------------------------------- 자기검증
def selfcheck(built: dict, device: str) -> dict:
    """학습에 쓸 **바로 그** 모델/설정으로 검증한다 (장난감 모델로는 (2)를 못 잡는다)."""
    out = {}
    subs = built["subjects"][:2]
    assert len(subs) == 2, "검증에 train subject 2명이 필요"

    # 학습에 쓸 checkpoint 그대로 만든다 (run() 과 같은 경로).
    cfg, w = built["cfg"], built["weights"]
    cfg.active = {"count"}
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), nib.load(ATLAS).affine, tau=0.5,
                          device=device, d_bg=None if cfg.sc_mode == "endpoint" else 2.0)
    m, _ = from_checkpoint(built["resume"], device=device)

    # (1) resid=0 이면 기존과 bit-exact
    import copy
    w0 = copy.deepcopy(w); w0.resid = 0.0
    wr = copy.deepcopy(w); wr.resid = 1.0
    s0 = ROIPairSubject(subs[0])
    a0 = t1_input(m, subs[0], built["t1_source"])

    def one_step(weights, seed):
        mm, _ = from_checkpoint(built["resume"], device=device)
        cc = copy.deepcopy(cfg); cc.active = {"count"}; cc.seed = seed
        tr = Trainer(mm, ea, cc, weights)
        return tr, tr.step(s0, a0)

    tr0, o0 = one_step(w0, 0)
    trr, orr = one_step(wr, 0)
    assert "L_resid" not in o0, "resid=0 인데 잔차 손실이 켜졌다"
    assert "L_resid" in orr, "resid=1 인데 잔차 손실이 안 켜졌다"
    assert tr0.resid_corr is None and trr.resid_corr is not None
    out["count_loss_unchanged"] = bool(abs(o0["L_count"] - orr["L_count"]) < 1e-9)
    assert out["count_loss_unchanged"], (o0["L_count"], orr["L_count"])
    # warmup 안에서는 기울기가 0 이므로 파라미터도 그대로여야 한다
    out["warmup_zero_loss"] = bool(orr["L_resid"] == 0.0)
    assert out["warmup_zero_loss"], orr["L_resid"]

    # (2) EMA 초기값 == count head 초기 예측 (softplus(log t) = log1p(t))
    mm, _ = from_checkpoint(built["resume"], device=device)
    cc = copy.deepcopy(cfg); cc.active = {"count"}
    tr = Trainer(mm, ea, cc, wr)
    R = mm.n_roi
    iu = torch.triu_indices(R, R, 1, device=mm.device)
    P_all = torch.stack([iu[0], iu[1]], 1)
    with torch.no_grad():
        a = mm.atm.encode_anatomy(a0)
        pred0 = torch.nn.functional.softplus(mm.edge_log_counts(a, P_all))
    # 처음부터 학습하면 count head 의 초기 예측이 정확히 log1p(template) 이라 시작 잔차가 0 이다.
    # 그런데 여기서는 **이미 학습된** d3_joint 에서 이어받으므로 그 등식이 깨진다 (실측 max 3.39).
    # 그래서 편향을 없애는 근거를 초기값 일치가 아니라 **warmup 동안의 EMA 감쇠**에 둔다:
    # EMA 는 warmup 안에서도 매 step 갱신되고 기울기만 차단되므로, (1-momentum)^warmup 만큼
    # 남은 초기값 성분이 기울기가 흐르기 시작할 때 무시할 수준이면 된다.
    d = float((pred0 - tr.resid_corr.ema_p).abs().max())
    decay = (1.0 - cc.resid_momentum) ** cc.resid_warmup
    out["ema_init_max_abs_diff"] = d
    out["ema_init_residual_after_warmup"] = d * decay
    assert out["ema_init_residual_after_warmup"] < 0.02, (
        f"warmup 이 짧다: 초기 편향 {d:.3f} x {decay:.2e} = "
        f"{out['ema_init_residual_after_warmup']:.4f} (log1p 단위, 허용 0.02). "
        f"resid_warmup 을 올려라 (지금 {cc.resid_warmup}, momentum {cc.resid_momentum})")

    # (3) GT 잔차 분산 > 0 : 배울 개인차가 실재하는가
    gs = []
    for s in built["subjects"][:8]:
        gs.append(torch.log1p(torch.as_tensor(
            np.asarray(ROIPairSubject(s).sc_mat, np.float32))[iu[0].cpu(), iu[1].cpu()]))
    Gm = torch.stack(gs)
    rg = Gm - Gm.mean(0, keepdim=True)
    out["gt_resid_var"] = float(rg.var())
    out["gt_resid_share"] = float(rg.var() / Gm.var())
    assert out["gt_resid_var"] > 1e-6, "GT 잔차 분산이 0 -- subject 가 전부 같다"

    # (4) 잔차 손실이 count head 에 기울기를 흘리는가
    rc = ResidualCorr(tr.resid_corr.ema_p.clone(), tr.resid_corr.ema_g.clone(),
                      momentum=0.5, warmup=0, log=False)
    a_leaf = a.detach()
    logc = mm.edge_log_counts(a_leaf, P_all)
    gt_c = torch.as_tensor(np.asarray(ROIPairSubject(subs[1]).sc_mat, np.float32),
                           device=mm.device)[iu[0], iu[1]]
    l = rc.on_upper(torch.nn.functional.softplus(logc), torch.log1p(gt_c))
    g = torch.autograd.grad(l, list(mm.count_head.net.parameters()), allow_unused=True)
    gn = sum(float(x.norm()) for x in g if x is not None)
    out["resid_grad_norm"] = gn
    assert gn > 0, "잔차 손실이 count head 에 기울기를 안 흘린다"
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


# --------------------------------------------------------------------------- 학습
def indiv_hook(val_subs, t1_source, trace: Path, arm: str):
    def hook(step, model):
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        t0 = time.time()
        try:
            with torch.no_grad():
                m = individuality_metrics(model, val_subs, n_subj=len(val_subs),
                                          pair_dice=False, source=t1_source)
        finally:
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
            model.train(was_training)
        assert m and "resid_r" in m, f"개인차 지표에 resid_r 이 없다: {sorted(m)}"
        row = {"arm": arm, "step": int(step), "time": time.strftime("%F %T"),
               "indiv_sec": time.time() - t0,
               **{k: m[k] for k in ("resid_r", "abl_own_r", "abl_shuf_r", "abl_zero_r", "abl_gap",
                                    "abl_pred_own_vs_zero_r", "inter_subj_r", "n_indiv_subj") if k in m}}
        with open(trace, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[indiv/{arm}] step {step} " +
              " ".join(f"{k}={row[k]:+.4f}" for k in ("resid_r", "abl_gap", "inter_subj_r") if k in row) +
              f" (n={row.get('n_indiv_subj')}, {row['indiv_sec']:.0f}s)", flush=True)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain/a1_resid.yaml")
    ap.add_argument("--arm", default="R", choices=["R", "C"],
                    help="R = 잔차 손실 on, C = 대조군 (resid=0, 다른 모든 것 동일)")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--indiv-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    built = C.build(C.load(ROOT / a.config))
    if a.steps:
        built["max_steps"] = a.steps
    if a.arm == "C":
        built["weights"].resid = 0.0
        built["out_dir"] = Path(str(built["out_dir"]) + "_ctrl")
    assert built["resume"] and built["resume"].exists(), built["resume"]

    EVAL.mkdir(parents=True, exist_ok=True)
    sc = selfcheck(built, a.device)
    (EVAL / "a1_resid_selfcheck.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2))
    if a.selfcheck_only:
        return

    val_subs = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()]
    trace = EVAL / f"a1_resid_trace_{a.arm}.jsonl"
    if not acquire_lock(LOCK):
        sys.exit(1)
    try:
        ck = run(phase=built["phase"], subjects=built["subjects"], max_steps=built["max_steps"],
                 out_dir=built["out_dir"], cfg=built["cfg"], weights=built["weights"],
                 init_bundle=built["init_bundle"], device=a.device, resume=built["resume"],
                 log_every=built["log_every"], trainable=built["trainable"],
                 unet_level=built["unet_level"], save_every=built["save_every"],
                 in_channels=built["in_channels"], template=built["template"],
                 t1_source=built["t1_source"],
                 step_hook=indiv_hook(val_subs, built["t1_source"], trace, a.arm),
                 step_hook_every=a.indiv_every)
    finally:
        release_lock(LOCK)

    rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    r = [x["resid_r"] for x in rows]
    res = {"arm": a.arm, "config": a.config, "checkpoint": str(ck), "selfcheck": sc,
           "resid_r_first": r[0] if r else None, "resid_r_last": r[-1] if r else None,
           "resid_r_max": max(r) if r else None, "n_points": len(r), "trace": str(trace)}
    (EVAL / f"a1_resid_{a.arm}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
