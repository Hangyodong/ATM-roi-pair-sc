"""YAML/JSON config -> (phase, subjects, steps, TrainConfig, LossWeights, ...). 최종 전략 §8: LR 은 config 에서."""
from __future__ import annotations

import json
from pathlib import Path

from .trainer import LossWeights, TrainConfig

ROOT = Path(__file__).resolve().parents[3]


def load(path) -> dict:
    p = Path(path)
    txt = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def build(cfg: dict):
    train = dict(cfg.get("train", {}))
    if isinstance(train.get("balance"), dict):
        from ..data.balanced_pair_sampler import BalanceConfig
        train["balance"] = BalanceConfig(**train["balance"])
    if isinstance(train.get("segment_balance"), dict):
        from ..data.segment_sampler import SegmentBalanceConfig
        train["segment_balance"] = SegmentBalanceConfig(**train["segment_balance"])
    tc = TrainConfig(**train)
    if tc.amp:                                   # yaml 은 torch.dtype 을 담을 수 없어 문자열로 받는다
        import torch
        m = {"bf16": torch.bfloat16, "fp16": torch.float16}
        assert tc.amp in m, f"amp 는 {sorted(m)} 중 하나여야 한다 (받은 값: {tc.amp})"
        tc.amp_dtype = m[tc.amp]
    lw = LossWeights(**cfg.get("loss", {}))
    subs = cfg["subjects"]
    if isinstance(subs, str):                       # 파일 경로 (subject 목록)
        subs = [l.strip() for l in (ROOT / subs).read_text().splitlines() if l.strip()]
    # 전처리(01/02/03)가 끝난 subject 만. 배치가 진행 중이면 그 시점의 부분집합으로 돈다.
    from ..data.paths import CACHE
    rp = ROOT / "outputs" / "roi_pairs"
    t1_suffix = {"rigid": "_T1w_rigid_W.npy", "syn": "_T1w_syn_W.npy"}[cfg.get("t1_source", "rigid")]
    ready = [s for s in subs if (CACHE / f"{s}{t1_suffix}").exists()
             and (rp / s / "assignments.npz").exists() and (rp / s / "bundles.npz").exists()]
    if len(ready) < len(subs):
        print(f"[config] 전처리 완료 subject {len(ready)}/{len(subs)} 만 사용 (나머지는 배치 진행 중)", flush=True)
    assert ready, "전처리된 subject 가 없음"
    subs = ready
    return dict(phase=cfg["phase"], subjects=subs, max_steps=int(cfg["max_steps"]),
                out_dir=ROOT / cfg["out_dir"], cfg=tc, weights=lw,
                init_bundle=cfg.get("init_bundle", "AF_L"), trainable=cfg.get("trainable", "vae"),
                unet_level=cfg.get("unet_level"), resume=(ROOT / cfg["resume"]) if cfg.get("resume") else None,
                log_every=int(cfg.get("log_every", 10)), save_every=int(cfg.get("save_every", 200)),
                # 전처리 프로토콜 (재학습 설계 §2). 이 셋이 run() 까지 안 가면 템플릿 인수분해가
                # 꺼진 채로 돌고, 입력 채널/프로토콜이 checkpoint 메타와 어긋난다.
                in_channels=int(cfg.get("in_channels", 2)),
                template=cfg.get("template", None),
                t1_source=cfg.get("t1_source", "rigid"))

