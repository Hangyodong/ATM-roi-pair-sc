#!/usr/bin/env python
"""D1 — 디코더 충실도 phase 드라이버 (W2-a).

  python scripts/48_d1_decoder.py --config configs/retrain/d1_decoder.yaml

고치는 것은 **ConvVAE BatchNorm 의 train/eval 격차** 하나다 (W1-d 실측):
    train 모드(배치 통계)   3.55~4.55 mm   <- trainer 로그가 찍던 값
    eval  모드(running)     8.458 mm       <- 추론에서 실제로 나오던 값
즉 3.55 mm 는 추론 시점에 성립한 적이 없다. 이 스크립트는 학습 **전후 모두** eval 모드와
train 모드 복원 RMSE 를 재고, 게이트는 반드시 eval 모드로 건다 (train 모드로 걸면 8.46 mm
짜리 모델이 3.55 로 통과한다).

산출: outputs/eval/w2a_d1_result.json (학습 전/후 표 + 게이트 판정 + 재현 명령)
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.evaluation.gates import check_gates                      # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                    # noqa: E402
from atm_sc.training import config as C                              # noqa: E402
from atm_sc.training.run import recon_rmse_metrics, run              # noqa: E402

LOCK = ROOT / "outputs/gpu.lock"


def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    """scripts/19_train_pipeline.py 와 같은 규약 (15분 heartbeat)."""
    me = f"{socket.gethostname()}:{os.getpid()}"
    if lock.exists():
        # 다른 에이전트가 "host:pid" 가 아닌 형식으로 쓸 수 있다 -> 파싱 실패를 점유로 본다
        # (형식을 모르는 lock 을 빼앗는 것보다 기다리는 쪽이 안전하다).
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
            print(f"lock 보유 중: {txt} "
                  f"(mtime {time.time() - lock.stat().st_mtime:.0f}s 전) -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def release_lock(lock: Path) -> None:
    if lock.exists() and lock.read_text().strip().endswith(f":{os.getpid()}"):
        lock.unlink()


def measure(ckpt: Path, val_subs, dev, n_subj, n_pairs, n_per_pair, seed=0) -> dict:
    """checkpoint 를 배포 그대로 열어 held-out 복원 RMSE 를 eval/train 두 모드로 잰다."""
    m, sd = from_checkpoint(ckpt, device=dev)
    src = sd.get("t1_source", "syn")
    meta = {k: str(sd.get(k)) for k in ("phase", "step", "unet_level", "in_channels", "template", "t1_source")}
    out = recon_rmse_metrics(m, val_subs, n_subj=n_subj, n_pairs=n_pairs, n_per_pair=n_per_pair,
                             seed=seed, source=src)
    del m
    torch.cuda.empty_cache()
    return {"ckpt": str(ckpt.relative_to(ROOT)), "meta": meta, **out}


def main(a) -> int:
    cfg_raw = C.load(ROOT / a.config)
    kw = C.build(cfg_raw)
    phase = kw["phase"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.max_steps:
        kw["max_steps"] = min(kw["max_steps"], a.max_steps)
    if a.subjects_limit:
        kw["subjects"] = kw["subjects"][: a.subjects_limit]
    val_subs = [l.strip() for l in (ROOT / a.val_subjects).read_text().splitlines() if l.strip()]
    assert val_subs, a.val_subjects

    resume = kw["resume"]
    assert resume is not None and resume.exists(), f"resume checkpoint 가 없다: {resume}"
    sd0 = torch.load(resume, map_location="cpu", weights_only=False)
    assert sd0.get("t1_source", "syn") == cfg_raw.get("t1_source", "rigid"), (
        f"t1_source 불일치: checkpoint {sd0.get('t1_source')} vs config {cfg_raw.get('t1_source')}. "
        "프로토콜이 어긋나면 오류 없이 결과만 나빠진다.")
    assert int(sd0.get("in_channels", 1)) == int(cfg_raw.get("in_channels", 2)), (
        sd0.get("in_channels"), cfg_raw.get("in_channels"))
    del sd0

    print(f"[d1] before: {resume}", flush=True)
    t0 = time.time()
    before = measure(resume, val_subs, dev, a.n_val, a.n_val_pairs, kw["cfg"].n_gt_per_pair)
    print(f"[d1] before  eval {before['recon_rmse_eval_mm']:.3f} mm / "
          f"train {before['recon_rmse_train_mm']:.3f} mm "
          f"({before['recon_n_subj']}명, {before['recon_n_streamlines']:,} 가닥, {time.time()-t0:.0f}s)",
          flush=True)

    print(f"[d1] 학습 시작: {kw['max_steps']} step, subject {len(kw['subjects'])}명, "
          f"bn_mode={kw['cfg'].bn_mode}", flush=True)
    ck = run(**kw, device=dev, heartbeat=LOCK)
    assert ck.exists() and ck.stat().st_size > 0, ck

    after = measure(ck, val_subs, dev, a.n_val, a.n_val_pairs, kw["cfg"].n_gt_per_pair)
    print(f"[d1] after   eval {after['recon_rmse_eval_mm']:.3f} mm / "
          f"train {after['recon_rmse_train_mm']:.3f} mm", flush=True)
    assert after["meta"]["t1_source"] == before["meta"]["t1_source"], (before["meta"], after["meta"])

    gates = (cfg_raw.get("gates") or {}).get(phase) or []
    metrics = dict(after)
    metrics["n_val"] = after["recon_n_subj"]
    ok, msgs = (True, ["게이트 없음"]) if not gates else check_gates(metrics, gates)
    for msg in msgs:
        print(f"[gate] {msg}", flush=True)

    res = {"cmd": f"python scripts/48_d1_decoder.py --config {a.config}"
                  + (f" --max-steps {a.max_steps}" if a.max_steps else "")
                  + f" --n-val {a.n_val} --n-val-pairs {a.n_val_pairs}",
           "config": a.config, "phase": phase, "checkpoint": str(ck.relative_to(ROOT)),
           "max_steps": kw["max_steps"], "n_train_subjects": len(kw["subjects"]),
           "bn_mode": kw["cfg"].bn_mode, "val_subjects": a.val_subjects,
           "recon": {"before": before, "after": after},
           "gates": {"passed": bool(ok), "messages": msgs},
           "elapsed_sec": time.time() - t0}
    out = ROOT / "outputs" / "eval" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    assert out.stat().st_size > 0, out
    print(f"\n저장: {out.relative_to(ROOT)}", flush=True)
    if not ok:
        print("!! 게이트 미달 -- 실패로 보고한다.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/retrain/d1_decoder.yaml")
    ap.add_argument("--val-subjects", default="outputs/splits/val.txt")
    ap.add_argument("--n-val", type=int, default=3, help="복원 RMSE 를 잴 held-out subject 수 (스윕과 동일)")
    ap.add_argument("--n-val-pairs", type=int, default=512, help="subject 당 표본 pair 수")
    ap.add_argument("--max-steps", type=int, default=0, help="config 보다 짧게 (스모크용)")
    ap.add_argument("--subjects-limit", type=int, default=0, help="학습 subject 수 제한 (스모크용)")
    ap.add_argument("--out", default="w2a_d1_result.json")
    args = ap.parse_args()
    if not acquire_lock(LOCK):
        sys.exit(3)
    try:
        sys.exit(main(args))
    finally:
        release_lock(LOCK)
