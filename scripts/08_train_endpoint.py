#!/usr/bin/env python
"""Phase 3 — + L_endpoint  (pipeline §21).

full training 금지 규칙에 따라 --max-steps 를 명시해야 한다. 기본값은 smoke 수준(5 step)이다.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc.training.run import run                     # noqa: E402
from atm_sc.training.trainer import TrainConfig         # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["sub-000001"])
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--sc-mode", default="endpoint", choices=["endpoint", "pass"])
    ap.add_argument("--n-gen", type=int, default=4, help="양성 pair 당 생성 수")
    ap.add_argument("--max-pairs", type=int, default=None, help="step 당 pair subset (None=전부)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init-bundle", default="AF_L")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--trainable", default="vae", choices=["decoder", "vae", "vae+unet4"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "checkpoints" / "endpoint"))
    a = ap.parse_args()
    cfg = TrainConfig(sc_mode=a.sc_mode, n_gen_per_pair=a.n_gen, max_pairs_per_step=a.max_pairs,
                      chunk=a.chunk, lr=a.lr)
    ck = run("endpoint", a.subjects, a.max_steps, Path(a.out), cfg, init_bundle=a.init_bundle,
             resume=Path(a.resume) if a.resume else None, trainable=a.trainable)
    print("checkpoint:", ck)
