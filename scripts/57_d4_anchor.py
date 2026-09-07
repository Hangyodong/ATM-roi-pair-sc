#!/usr/bin/env python
"""D4 pair 앵커 재매개화 학습 (alpha 램프업).

  python scripts/57_d4_anchor.py --selfcheck-only
  python scripts/57_d4_anchor.py --config configs/retrain/d4_anchor.yaml

판정: **eval 모드** recon RMSE. D3 기준값과 비교하고 valid/invalid 앵커로 나눠 본다.

반드시 확인하는 것 (실패하면 학습 전에 죽는다)
  1. alpha=0 에서 앵커 경로가 기존 경로와 **bit-exact** (기존 checkpoint 를 그대로 쓴다)
  2. alpha=1 에서 출력이 실제로 바뀐다 (통로가 죽지 않았다)
  3. 앵커 pair 순서가 np.triu_indices(82,1) 와 일치 (어긋나면 조용히 틀린다)
  4. 램프업 스케줄이 step 에 따라 alpha 를 실제로 움직인다
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
from atm_sc.models.endpoint_assigner import EndpointAssigner            # noqa: E402
from atm_sc.models.pair_anchor import load as load_anchor               # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.training import config as C                                 # noqa: E402
from atm_sc.training.run import (anatomy_feature, recon_batches, recon_rmse_metrics,  # noqa: E402
                                 run)
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


def selfcheck(built: dict, device: str) -> dict:
    out = {}
    cfg = built["cfg"]
    sub = built["subjects"][0]

    # (3) 앵커 순서 -- load() 안에서 assert 하지만 값도 남긴다
    z = np.load(ROOT / "outputs/cache/pair_anchor_paths.npz")
    out["anchor_valid"] = int(z["valid"].sum())
    out["anchor_total"] = int(z["valid"].size)
    out["half_range_min"] = float(z["half_range"].min())
    vol = np.prod(2.0 * z["half_range"].astype(np.float64), 1)
    out["box_shrink_median"] = float(4.68e6 / np.median(vol))
    out["box_shrink_median_valid"] = float(4.68e6 / np.median(vol[z["valid"]]))

    m0, _ = from_checkpoint(built["resume"], device=device)                       # 앵커 없음
    ma, _ = from_checkpoint(built["resume"], device=device, pair_anchor=True, anchor_alpha=0.0)
    a = anatomy_feature(m0, sub, built["init_bundle"], source=built["t1_source"])
    s = ROIPairSubject(sub)
    zt = np.load(s.dir / "bundles.npz")
    pid, poff = zt["pair_ids"], zt["pair_offsets"]
    k = int(np.argmax(np.diff(poff)))                                            # 가장 가닥이 많은 쌍
    lo, hi = int(poff[k]), int(poff[k + 1])
    S = torch.as_tensor(zt["streamlines"][lo:lo + 64].astype(np.float32), device=device)
    P = torch.as_tensor(np.repeat(pid[k][None], S.shape[0], 0).astype(np.int64), device=device)

    with torch.no_grad():
        # cond FiLM 국소 통로(cond_local_dim)가 checkpoint 에서 상속되면 condition 은 local 이 필수다
        loc = None
        if m0.pair_emb.local_dim:
            from atm_sc.data.local_feats import load_roi_feats, pair_local
            from atm_sc.models.roi_pair_embedding import canonical_pairs
            loc = pair_local(load_roi_feats(sub, built["t1_source"], device, n_roi=m0.n_roi), canonical_pairs(P))
        c0, ca = m0.condition(a, P, local=loc), ma.condition(a, P, local=loc)
        mu0, _ = m0.encode_streamlines(S, c0)
        r0 = m0.decode(mu0, c0)
        ra = ma.decode(mu0, ca, P)
        out["alpha0_max_abs_diff_mm"] = float((r0 - ra).abs().max())
        assert out["alpha0_max_abs_diff_mm"] == 0.0, "alpha=0 인데 bit-exact 가 아니다"
        ma.anchor.set_alpha(1.0)
        r1 = ma.decode(mu0, ca, P)
        out["alpha1_shift_mm"] = float((r1 - r0).abs().max())
        assert out["alpha1_shift_mm"] > 1.0, "alpha=1 인데 출력이 안 바뀐다 -- 통로가 죽었다"
        out["alpha1_recon_rmse_mm"] = float(((r1 - S) ** 2).sum(-1).mean().sqrt())
        out["alpha0_recon_rmse_mm"] = float(((r0 - S) ** 2).sum(-1).mean().sqrt())

    # (4) 램프업 스케줄
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), nib.load(ATLAS).affine, tau=0.5,
                          device=device, d_bg=2.0)
    cc = copy.deepcopy(cfg); cc.active = {"recon"}
    mm, _ = from_checkpoint(built["resume"], device=device, pair_anchor=True, anchor_alpha=0.0)
    tr = Trainer(mm, ea, cc, built["weights"])
    seen = []
    for st in (0, cc.anchor_alpha_steps // 2, cc.anchor_alpha_steps, cc.anchor_alpha_steps * 2):
        tr.step(s, a, step=st)
        seen.append(round(float(mm.anchor.alpha), 4))
    out["alpha_schedule"] = seen
    assert seen[0] == 0.0 and seen[-1] == cc.anchor_alpha_end and seen[1] < seen[2], seen
    del m0, ma, mm, tr
    torch.cuda.empty_cache()
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain/d4_anchor.yaml")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    built = C.build(C.load(ROOT / a.config))
    if a.steps:
        built["max_steps"] = a.steps
    assert built["resume"] and built["resume"].exists(), built["resume"]
    EVAL.mkdir(parents=True, exist_ok=True)
    sc = selfcheck(built, a.device)
    (EVAL / "d4_anchor_selfcheck.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2))
    if a.selfcheck_only:
        return
    if not acquire_lock(LOCK):
        sys.exit(1)
    try:
        ck = run(phase=built["phase"], subjects=built["subjects"], max_steps=built["max_steps"],
                 out_dir=built["out_dir"], cfg=built["cfg"], weights=built["weights"],
                 init_bundle=built["init_bundle"], device=a.device, resume=built["resume"],
                 log_every=built["log_every"], trainable=built["trainable"],
                 unet_level=built["unet_level"], save_every=built["save_every"],
                 in_channels=built["in_channels"], template=built["template"],
                 t1_source=built["t1_source"])
    finally:
        release_lock(LOCK)
    res = {"config": a.config, "checkpoint": str(ck), "selfcheck": sc}
    (EVAL / "d4_anchor.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
