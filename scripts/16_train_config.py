#!/usr/bin/env python
"""config 파일 하나로 phase 학습 실행 (PBS job 과 로컬 공용). 예: python scripts/16_train_config.py --config configs/phase2_geometry.yaml"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.training import config as C           # noqa: E402
from atm_sc.training.run import run               # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="config 값 덮어쓰기 (sanity 용)")
    a = ap.parse_args()
    kw = C.build(C.load(a.config))
    if a.max_steps:
        kw["max_steps"] = a.max_steps
    import torch
    torch.manual_seed(kw["cfg"].seed)
    print("checkpoint:", run(**kw))
