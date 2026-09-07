#!/usr/bin/env python
"""디코더 기하 정확도가 용량 문제인가, 최적화 문제인가, **정규화 문제**인가.

  python scripts/38_decoder_capacity.py --steps 4000 --eval-every 500

복원 오차는 모든 phase 에서 평평했다 (s1 3.72 -> s5 3.55 mm). 상류 ATM 은 번들 하나당
모델 하나(30개)인데 우리는 **디코더 하나로 3,321쌍**을 커버한다 -- 용량이 병목일 수 있다.
UNet 을 동결하고(캐시된 anatomy 사용) VAE 만 학습해 조건별 복원 오차를 비교한다.

--- 2026-09-06 개정 (W1-d). 이전 실행의 `base` 9.17 mm 가 무엇이었는지 규명한 결과 ---

이 스크립트는 **route2/s5_joint** checkpoint 를 기본값으로 쓰고 있었다. 그 checkpoint 는
(a) `pair_emb.anatomy_norm` (재학습 때 도입된 LayerNorm) 이 없고, (b) conv1_1 이 1채널이며,
(c) `syn` + robust 정규화 T1 로 학습된 것이다. 그런데 스크립트는 `ROIPairATM(...)` 을 직접
만들어(=`in_channels=2` 기본, template 없음) `--t1-mode rigid` 입력을 먹였다. 그 조합의
시작 복원 오차는 **117 mm** 다 (실측). 1500 step 학습이 그것을 9.17 mm 까지 되돌린 것이지,
9.17 이 디코더의 성능이었던 적은 없다. -> `from_checkpoint()` 로 통일하고 기본
checkpoint 를 retrain/p4_joint 로 바꾼다.

두 번째 원인은 **ConvVAE 의 BatchNorm1d 5개**다. 같은 복원이
    train 모드(배치 통계)  4.0~4.6 mm      <- trainer 가 로그에 찍는 값 (3.55/3.79 mm)
    eval  모드(running 통계) 8.46 mm       <- 추론에서 실제로 나오는 값
로 2배 갈린다. 즉 **3.55 mm 는 추론 시점에 성립한 적이 없다.** 그런데 gradient 없이
running 통계만 다시 쌓으면 (BN 재보정) eval 모드가 8.46 -> 4.00 mm 로 떨어진다.
파라미터를 하나도 안 늘리고 오차의 53 %가 사라진다.
-> 모든 arm 을 **train 모드 / eval 모드 / BN 재보정 후 eval 모드** 세 값으로 채점한다.
   한쪽만 보면 이번 같은 착시가 반복된다.

flip(방향 모호성)은 원인이 아니다: flip-aware RMSE 와 그냥 RMSE 가 소수점 셋째 자리까지 같다.

조건 축:
  base/lr3x/lr10x   구조 고정, LR 만                 -> 최적화 가설
  bn_*              BN 을 학습 중에 어떻게 다루나     -> 정규화 가설
  refine*           동결 디코더 뒤 잔차 정련망        -> 용량 가설 (사전학습 가중치 보존)
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.losses.geometry import adjacency_loss, kl_loss, stream_recon_loss   # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                    # noqa: E402
from atm_sc.models.streamline_refiner import StreamlineRefiner       # noqa: E402
from atm_sc.training.run import t1_input                             # noqa: E402

LOCK = ROOT / "outputs/gpu.lock"


def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    """scripts/19_train_pipeline.py 와 같은 규약. 다른 에이전트가 같은 A10 을 쓴다."""
    me = f"{socket.gethostname()}:{os.getpid()}"
    if lock.exists():
        host, pid = lock.read_text().strip().split(":")
        alive = False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive = True
            except OSError:
                alive = False
        if alive or (host != socket.gethostname() and time.time() - lock.stat().st_mtime < stale_sec):
            print(f"lock 보유 중: {host}:{pid} "
                  f"(mtime {time.time() - lock.stat().st_mtime:.0f}s 전) -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def release_lock(lock: Path) -> None:
    if lock.exists() and lock.read_text().strip().endswith(f":{os.getpid()}"):
        lock.unlink()


# --- BatchNorm 취급 -------------------------------------------------------------
def ae_bns(m) -> list:
    return [b for b in m.atm.net.ae.modules() if isinstance(b, nn.BatchNorm1d)]


def bn_state(m) -> list:
    """running 통계 스냅샷. train 모드 forward 는 no_grad 여도 이 buffer 를 바꾼다."""
    return [(b.running_mean.clone(), b.running_var.clone(), b.num_batches_tracked.clone())
            for b in ae_bns(m)]


def bn_restore(m, st) -> None:
    for b, (mu, var, n) in zip(ae_bns(m), st):
        b.running_mean.copy_(mu); b.running_var.copy_(var); b.num_batches_tracked.copy_(n)


@torch.no_grad()
def bn_recalibrate(m, refiner, data, batch=2048) -> int:
    """gradient 없이 train 데이터로 running 통계를 다시 쌓는다 (momentum=None = 누적 평균).

    학습이 아니다 -- 파라미터는 그대로다. 이것만으로 eval 모드 오차가 절반으로 준다.
    """
    bns = ae_bns(m)
    saved = [b.momentum for b in bns]
    for b in bns:
        b.reset_running_stats(); b.momentum = None
    m.train()
    for S, P, feat, _sub in data:   # 배치 튜플에 subject 가 추가됐다 (국소 feature 용)
        for i in range(0, len(S), batch):
            s, p = S[i:i + batch].to(m.device), P[i:i + batch].to(m.device)
            c = m.condition(feat, p)
            mu, _ = m.encode_streamlines(s, c)
            rec = m.decode(mu, c)
            if refiner is not None:
                refiner(rec, c)
    for b, mo in zip(bns, saved):
        b.momentum = mo
    return len(bns)


def set_bn_eval(m) -> None:
    """ConvVAE 의 BN 만 eval 로 고정 (running 통계 사용 + 갱신 안 함)."""
    for b in ae_bns(m):
        b.eval()


# --- 데이터 ---------------------------------------------------------------------
def load_subject(sub, n_per_pair, rng, device):
    b = np.load(ROOT / f"outputs/roi_pairs/{sub}/bundles.npz")
    pairs, off, SS = b["pair_ids"].astype(np.int64), b["pair_offsets"], b["streamlines"]
    idx, pr = [], []
    for k in range(len(pairs)):
        avail = int(off[k + 1] - off[k])
        n = min(n_per_pair, avail)
        if n <= 0:
            continue
        idx.append(off[k] + rng.choice(avail, n, replace=False))
        pr.append(np.repeat(pairs[k][None], n, axis=0))
    idx, pr = np.concatenate(idx), np.concatenate(pr)
    S = torch.as_tensor(SS[idx].astype(np.float32))
    assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
    assert torch.isfinite(S).all(), f"{sub}: GT streamline 에 NaN/Inf"
    return S, torch.as_tensor(pr)


@torch.no_grad()
def _rmse(m, refiner, data, batch=2048, flip=False):
    """held-out 복원 RMSE (mm). 모드(train/eval)는 호출측이 정한다."""
    tot, n = 0.0, 0
    for S, P, feat, _sub in data:   # 배치 튜플에 subject 가 추가됐다 (국소 feature 용)
        for i in range(0, len(S), batch):
            s, p = S[i:i + batch].to(m.device), P[i:i + batch].to(m.device)
            c = m.condition(feat, p)
            mu, lv = m.encode_streamlines(s, c)
            rec = m.decode(mu, c)                       # 평가는 mu (샘플링 잡음 없이)
            if refiner is not None:
                rec = refiner(rec, c)
            assert rec.shape == s.shape and torch.isfinite(rec).all(), "복원에 NaN/Inf"
            a = ((rec - s) ** 2).sum(-1).mean(-1)
            v = torch.minimum(a, ((rec - s.flip(1)) ** 2).sum(-1).mean(-1)) if flip else a
            tot += float(v.mean()) * len(s); n += len(s)
    assert n > 0, "평가 데이터가 비었다"
    return float(np.sqrt(tot / n))


def evaluate(m, refiner, val_data, recal_data, batch=2048) -> dict:
    """세 렌즈로 같은 복원을 잰다. BN buffer 는 반드시 원상복구한다."""
    st = bn_state(m)
    m.eval()
    out = {"eval_rmse_mm": _rmse(m, refiner, val_data, batch),
           "eval_rmse_flip_mm": _rmse(m, refiner, val_data, batch, flip=True)}
    m.train()                                            # 배치 통계 (trainer 가 로그에 찍는 렌즈)
    out["train_rmse_mm"] = _rmse(m, refiner, val_data, batch)
    bn_restore(m, st)
    if recal_data:                                       # 재보정 후 eval 모드
        bn_recalibrate(m, refiner, recal_data, batch)
        m.eval()
        out["recal_rmse_mm"] = _rmse(m, refiner, val_data, batch)
        bn_restore(m, st)
    m.train()
    return out


def run_condition(name, cfg, sd, train_data, val_data, a, dev):
    torch.manual_seed(0)
    m, _ = from_checkpoint(ROOT / a.ckpt, device=dev,
                           use_refiner=False)            # 정련망은 아래에서 따로 만든다 (arm 축)
    m.train()
    refiner = None
    if cfg["refine"]:
        refiner = StreamlineRefiner(
            hidden=cfg["hidden"], layers=cfg["layers"], kernel=cfg.get("kernel", 5),
            cond_dim=None if cfg.get("nocond") else 512,
            coord_min=m.atm.coord_min.detach().cpu(),
            coord_scale=m.atm.coord_scale.detach().cpu()).to(dev)
        # 통과 기준: 시작 시 정련망은 항등이어야 한다 (허용오차 0).
        S0, P0, f0 = val_data[0]
        with torch.no_grad():
            c0 = m.condition(f0, P0[:64].to(dev))
            mm0 = m.decode(m.encode_streamlines(S0[:64].to(dev), c0)[0], c0)
        refiner.assert_identity(mm0, None if cfg.get("nocond") else c0)

    # nocond arm 은 조건을 안 받는 정련망이다. 평가/재보정 경로가 조건을 그대로 넘기지
    # 않도록 여기서 한 번만 감싼다 (안 감싸면 정련망의 cond 검사에 걸려 죽는다).
    apply = refiner
    if refiner is not None and cfg.get("nocond"):
        def apply(mm, _c, _r=refiner):
            return _r(mm, None)

    bn_mode = cfg.get("bn", "train")
    assert bn_mode in ("train", "eval", "recal_eval"), bn_mode
    if bn_mode == "recal_eval":                           # 먼저 재보정하고 그 통계로 고정
        bn_recalibrate(m, apply, train_data)

    ae = list(m.atm.net.ae.parameters())
    groups = ([] if cfg.get("freeze_ae") else
              [{"params": ae, "lr": cfg["lr"]},
               {"params": list(m.pair_emb.parameters()), "lr": cfg["lr"] * 3}])
    if cfg.get("freeze_ae"):
        for p in ae:
            p.requires_grad_(False)
    if refiner is not None:
        groups.append({"params": list(refiner.parameters()), "lr": cfg["lr"] * 3})
    assert groups, f"{name}: 학습할 파라미터가 없다"
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    rng = np.random.default_rng(0)
    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        m.train()
        if bn_mode != "train":
            set_bn_eval(m)
        S_all, P_all, feat = train_data[rng.integers(len(train_data))]
        sel = rng.choice(len(S_all), min(a.batch, len(S_all)), replace=False)
        s, p = S_all[sel].to(dev), P_all[sel].to(dev)
        amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.amp)
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            c = m.condition(feat, p)
            mu, logvar = m.encode_streamlines(s, c)
            rec = m.decode(m.reparameterize(mu, logvar), c)
            if refiner is not None:
                rec = apply(rec, c)
        rec, mu, logvar = rec.float(), mu.float(), logvar.float()   # 손실은 fp32 로 (mm 정밀도)
        loss = stream_recon_loss(rec, s) + 0.1 * kl_loss(mu, logvar, m.prior_mean(p)) \
            + 1.0 * adjacency_loss(rec)
        assert torch.isfinite(loss), f"{name} step {step}: loss 가 NaN/Inf"
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 50.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(m, apply, val_data, train_data if a.recal else None)
            hist.append({"step": step, "loss": float(loss), **r})
            print(f"  [{name}] step {step:5d}  eval {r['eval_rmse_mm']:6.3f}  "
                  f"train {r['train_rmse_mm']:6.3f}  "
                  f"recal {r.get('recal_rmse_mm', float('nan')):6.3f} mm "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert len(hist) >= 2, f"{name}: 평가점이 {len(hist)}개뿐이라 수렴을 볼 수 없다"
    n_extra = sum(q.numel() for q in refiner.parameters()) if refiner else 0
    best = {k: min(h[k] for h in hist) for k in hist[0] if k.endswith("_mm")}
    return {"name": name, "cfg": cfg, "hist": hist, "extra_params": n_extra,
            "best": best, "best_val_rmse_mm": best["eval_rmse_mm"],
            "rf": refiner.receptive_field if refiner else 0, "sec": time.time() - t0}


BASE_LR = 3e-5
CONDS = {
    # --- 최적화 가설 (기존 arm, 지우지 않는다) ---
    "base":        {"lr": BASE_LR,      "refine": False, "hidden": 0,   "layers": 0},
    "lr3x":        {"lr": BASE_LR * 3,  "refine": False, "hidden": 0,   "layers": 0},
    "lr10x":       {"lr": BASE_LR * 10, "refine": False, "hidden": 0,   "layers": 0},
    "refine":      {"lr": BASE_LR,      "refine": True,  "hidden": 128, "layers": 3},
    "refine_big":  {"lr": BASE_LR,      "refine": True,  "hidden": 256, "layers": 5},
    # --- 정규화 가설: BN 을 학습 중에 어떻게 다루나 ---
    "bn_eval":     {"lr": BASE_LR, "refine": False, "hidden": 0, "layers": 0, "bn": "eval"},
    "bn_recal":    {"lr": BASE_LR, "refine": False, "hidden": 0, "layers": 0, "bn": "recal_eval"},
    # --- 용량 가설: 정련망 크기 축 ---
    "refine_small": {"lr": BASE_LR, "refine": True, "hidden": 64,  "layers": 2},
    "refine_deep":  {"lr": BASE_LR, "refine": True, "hidden": 128, "layers": 5},   # RF 257 > 128
    "refine_wide":  {"lr": BASE_LR, "refine": True, "hidden": 256, "layers": 3},
    # --- 둘을 합친 구성 (D1 후보) ---
    "refine_bn_recal": {"lr": BASE_LR, "refine": True, "hidden": 128, "layers": 5, "bn": "recal_eval"},
    # 순수 복원에는 조건이 필요 없다. 조건 없이도 되는지 확인한다.
    "refine_nocond":   {"lr": BASE_LR, "refine": True, "hidden": 128, "layers": 5,
                        "bn": "recal_eval", "nocond": True},
    # AE 를 완전히 동결하고 정련망만 학습 -> 정련망 단독 용량
    "refine_only":     {"lr": BASE_LR, "refine": True, "hidden": 128, "layers": 5,
                        "bn": "recal_eval", "freeze_ae": True},
}


def save(out: Path, a, sd, res) -> None:
    cmd = (f"python scripts/38_decoder_capacity.py --ckpt {a.ckpt} --steps {a.steps} "
           f"--eval-every {a.eval_every} --n-train {a.n_train} --n-val {a.n_val} "
           f"--n-per-pair {a.n_per_pair} --batch {a.batch}"
           + ("" if a.recal else " --no-recal") + (f" --amp {a.amp}" if a.amp else "")
           + (" --tf32" if a.tf32 else "") + (" --cudnn-benchmark" if a.cudnn_benchmark else ""))
    json.dump({"args": vars(a), "cmd": cmd,
               "ckpt_meta": {k: str(sd.get(k)) for k in
                             ("unet_level", "in_channels", "template", "t1_source", "phase", "step")},
               "results": res}, open(out, "w"), indent=1, ensure_ascii=False, default=str)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    print(f"tf32={a.tf32} cudnn_benchmark={a.cudnn_benchmark} amp={a.amp}", flush=True)
    train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()][:a.n_train]
    val = [l.strip() for l in open(ROOT / "outputs/splits/val.txt") if l.strip()][:a.n_val]

    # checkpoint 메타로 모델을 만든다. 직접 ROIPairATM(...) 을 부르면 in_channels/template/
    # t1_source 가 어긋나 **조용히 틀린다** (이전 실행의 9.17 mm 가 그 사례다).
    m0, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    m0.eval()
    t1_mode = a.t1_mode or sd.get("t1_source", "syn")
    assert t1_mode == sd.get("t1_source", "syn"), (
        f"checkpoint 는 t1_source={sd.get('t1_source', 'syn')} 로 학습됐는데 {t1_mode} 를 먹이려 한다")

    def prep(subs, npp):
        out, rng = [], np.random.default_rng(0)
        for s in subs:
            with torch.no_grad():                        # UNet 동결 -> subject 당 1회
                feat = m0.atm.encode_anatomy(t1_input(m0, s, source=t1_mode))
            S, P = load_subject(s, npp, rng, dev)
            out.append((S, P, feat))
            print(f"  준비 {s}: {len(S):,} 가닥", flush=True)
        return out

    print(f"train {len(train)}명 / val {len(val)}명, ckpt={a.ckpt}, T1={t1_mode}, "
          f"in_channels={m0.in_channels}, template={m0.template is not None}", flush=True)
    train_data, val_data = prep(train, a.n_per_pair), prep(val, max(a.n_per_pair // 2, 2))
    del m0; torch.cuda.empty_cache()

    conds = {k: v for k, v in CONDS.items() if not a.only or k in a.only.split(",")}
    assert conds, f"--only 가 아무 arm 도 고르지 못했다 (가능: {sorted(CONDS)})"
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res = []
    for name, cfg in conds.items():
        print(f"\n=== {name}  {cfg} ===", flush=True)
        res.append(run_condition(name, cfg, sd, train_data, val_data, a, dev))
        torch.cuda.empty_cache()
        save(out, a, sd, res)          # arm 마다 부분 저장 -- 중간에 죽어도 앞선 arm 을 잃지 않는다
    print(f"\n{'조건':16s} {'eval':>8s} {'train':>8s} {'재보정':>8s} {'추가파라미터':>13s} "
          f"{'RF':>5s} {'시간':>7s}")
    for r in sorted(res, key=lambda x: x["best"].get("recal_rmse_mm", x["best"]["eval_rmse_mm"])):
        b = r["best"]
        print(f"{r['name']:16s} {b['eval_rmse_mm']:8.3f} {b['train_rmse_mm']:8.3f} "
              f"{b.get('recal_rmse_mm', float('nan')):8.3f} {r['extra_params']:13,d} "
              f"{r['rf']:5d} {r['sec']:6.0f}s")
    print("\n기준: trainer 로그(train 모드) 3.55 mm 는 추론 시점 값이 아니다. "
          "eval 모드 기본값이 8.46 mm, BN 재보정만으로 4.00 mm.")
    print("refine 계열만 뚜렷이 낮으면 용량, lr 계열이 낮으면 최적화, bn 계열이 낮으면 정규화 문제다.")
    print(f"저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt")
    ap.add_argument("--t1-mode", default="", choices=["", "rigid", "syn"],
                    help="비우면 checkpoint 의 t1_source 를 쓴다 (권장)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-per-pair", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="outputs/eval/decoder_capacity.json")
    ap.add_argument("--recal", action="store_true", default=True,
                    help="평가 시 BN 재보정 후 eval 모드 값도 남긴다")
    ap.add_argument("--no-recal", dest="recal", action="store_false")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmul 허용 (PyTorch 기본은 꺼짐)")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--amp", default="", choices=["", "bf16", "fp16"])
    args = ap.parse_args()
    if not acquire_lock(LOCK):
        sys.exit(3)
    try:
        main(args)
    finally:
        release_lock(LOCK)
