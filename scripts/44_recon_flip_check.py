#!/usr/bin/env python
"""복원 RMSE 지표가 streamline 방향 모호성 때문에 과대평가인지 확인 (W1-a).

  python scripts/44_recon_flip_check.py

streamline 은 방향이 임의다 (A->B 와 B->A 는 같은 가닥). 그래서
`losses/geometry.stream_recon_loss` 는 정방향과 역방향(`gt.flip(1)`) 중 작은 쪽을 쓴다.
그런데 `training/trainer.py` 의 `recon_rmse_mm` 지표에는 flip 처리가 없다:

    float(((rec - S_gt) ** 2).sum(-1).mean().sqrt())

손실이 뒤집힌 쪽을 목표로 학습시키면 모델은 뒤집힌 방향을 출력하고, 지표는 그것을 전부
오차로 셈한다 -> 보고값이 부풀려진다. 같은 forward pass 에서 두 값을 나란히 재고 뒤집힌
가닥의 비율까지 낸다. 값은 나온 그대로 보고하고 조정하지 않는다.

경로:
  - 오라클 posterior (GT 인코딩 -> 디코딩). scripts/38_decoder_capacity.py 의 evaluate() 와 같다.
  - batch 는 trainer 의 [R] 분기와 **같은 sampler** (checkpoint 에 저장된 BalanceConfig 로 만든
    BalancedPairSampler). 균등 추출은 학습이 거의 보지 않은 희소 pair 를 과대표집해 지표가
    로그와 어긋난다 (실측: 균등 8.8 mm vs 로그 3.8 mm) -> --sampling uniform 으로 비교 가능.
  - ConvVAE 에 BatchNorm 이 있다 (roi_atm.ROIPairATM.train 주석). 로그의 recon_rmse_mm 은
    **train 모드(batch 통계)** 로 계산된 값이므로 eval 모드(running 통계)로 재면 값이 다르다
    (실측: 8.0 vs 4.0 mm). 지표 정의를 따지는 게 목적이므로 두 모드를 모두 낸다.
    BN batch 구성까지 맞추려고 sampler batch 하나를 forward chunk 하나로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.balanced_pair_sampler import BalanceConfig, BalancedPairSampler   # noqa: E402
from atm_sc.data.dataset import ROIPairSubject                       # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                    # noqa: E402
from atm_sc.training.run import t1_input                             # noqa: E402


# --- GPU lock (scripts/19_train_pipeline.py 와 같은 규약) ---------------------------------
def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    """다른 프로세스가 살아 있으면 False. heartbeat(mtime) 가 stale_sec 이상 멈췄으면 빼앗는다."""
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
            print(f"lock 보유 중: {host}:{pid} (mtime {time.time()-lock.stat().st_mtime:.0f}s 전) -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def start_heartbeat(lock: Path, period: int = 60) -> threading.Event:
    """15분 stale 기준보다 훨씬 짧게 mtime 갱신."""
    stop = threading.Event()

    def beat():
        while not stop.wait(period):
            if lock.exists():
                lock.touch()

    threading.Thread(target=beat, daemon=True).start()
    return stop


# --- 데이터 ------------------------------------------------------------------------------
def sample_balanced(sub: str, bal: BalanceConfig, n_batches: int, n_pairs: int,
                    n_per_pair: int, rng: np.random.Generator):
    """trainer [R] 분기와 동일한 batch 구성 -> [(S [n,128,3], P [n,2], syn [n] bool)] * n_batches."""
    bs = BalancedPairSampler(ROIPairSubject(sub), bal)
    out = []
    for _ in range(n_batches):
        s, p, _, syn, _ = bs.sample_batch(rng, n_pairs, n_per_pair)
        out.append((s.float(), p, syn))
    return out


def sample_uniform(sub: str, n_per_pair: int, rng: np.random.Generator, chunk: int = 0):
    """pair 균등 추출 (GT 번들만). 학습 분포가 아니라 참고용."""
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
    S, P = torch.as_tensor(SS[idx].astype(np.float32)), torch.as_tensor(pr)
    n = chunk or len(S)
    return [(S[i:i + n], P[i:i + n], np.zeros(len(S[i:i + n]), bool)) for i in range(0, len(S), n)]


# --- 측정 --------------------------------------------------------------------------------
def freeze_bn_stats(m):
    """BatchNorm running 통계를 갱신하지 않게 한다 (momentum=0).

    train 모드에서 정규화는 어차피 batch 통계로 하므로 계산은 그대로고, 여러 번 forward 해도
    buffer 가 변하지 않아 뒤이은 eval 모드 측정이 오염되지 않는다.
    """
    n = 0
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.0; n += 1
    return n


@torch.no_grad()
def batch_sq(m, S, P, feat, posterior: str):
    """가닥별 평균 제곱거리 (정방향 a, 역방향 b), mm^2. batch 하나 = forward 하나."""
    s, p = S.to(m.device), P.to(m.device)
    c = m.condition(feat, p)
    mu, logvar = m.encode_streamlines(s, c)
    z = mu if posterior == "mu" else m.reparameterize(mu, logvar)
    rec = m.decode(z, c).float()
    assert rec.shape == s.shape, (rec.shape, s.shape)
    assert torch.isfinite(rec).all(), "디코더 출력에 NaN/Inf"
    # stream_recon_loss 와 동일한 두 항
    a = (((rec - s) ** 2).sum(-1).mean(-1)).double().cpu().numpy()
    b = (((rec - s.flip(1)) ** 2).sum(-1).mean(-1)).double().cpu().numpy()
    return a, b


def summarize(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> dict:
    """trainer.py:350 의 지표식과, 같은 집계의 flip-aware 대응식."""
    mn = np.minimum(a, b)
    # trainer.py:350 -> ((rec-S)**2).sum(-1).mean().sqrt() = sqrt(모든 가닥/점 평균 제곱거리)
    noflip, flipaware = float(np.sqrt(a.mean())), float(np.sqrt(mn.mean()))
    out = {
        "n_streamlines": int(a.size),
        "rmse_noflip_mm": noflip,
        "rmse_flipaware_mm": flipaware,
        "delta_mm": noflip - flipaware,
        "frac_flipped": float((b < a).mean()),
        # stream_recon_loss 자체의 집계 (가닥별 RMSE 의 평균) — 로그의 L_recon 과 대응
        "loss_style_flipaware_mm": float(np.sqrt(mn + 1e-6).mean()),
        "loss_style_noflip_mm": float(np.sqrt(a + 1e-6).mean()),
    }
    if w is not None:                       # L_recon 은 synthetic 에 lambda_syn 가중 (§47)
        out["loss_style_flipaware_weighted_mm"] = float((np.sqrt(mn + 1e-6) * w).sum() / w.sum())
    return out


def self_test():
    """flip 검출이 실제로 동작하는지 확인. 0% 라는 결과가 조용한 버그가 아님을 보증한다.

    이 파이프라인의 전형적 실패는 "오류 없이 늘 0" 이다. rec 를 GT 와 GT.flip 으로 각각
    강제해 두 극단이 나오는지 본다.
    """
    g = torch.Generator().manual_seed(0)
    S = torch.randn(64, 128, 3, generator=g) * 10.0

    def ab(rec, gt):
        return ((((rec - gt) ** 2).sum(-1).mean(-1)).double().numpy(),
                (((rec - gt.flip(1)) ** 2).sum(-1).mean(-1)).double().numpy())

    a, b = ab(S, S)                      # 완벽 복원 -> 뒤집힘 0%, 두 RMSE 모두 0
    r = summarize(a, b)
    assert r["frac_flipped"] == 0.0 and r["rmse_noflip_mm"] < 1e-6, r
    a, b = ab(S.flip(1), S)              # 뒤집힌 복원 -> 뒤집힘 100%, flip-aware 만 0
    r = summarize(a, b)
    assert r["frac_flipped"] == 1.0, r
    assert r["rmse_flipaware_mm"] < 1e-6 < r["rmse_noflip_mm"], r
    assert r["delta_mm"] > 0, r


MODES = ("train", "eval")
POSTS = ("mu", "sample")


def main(a):
    lock = ROOT / a.lock
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not acquire_lock(lock, stale_sec=900):
        sys.exit(1)
    stop = start_heartbeat(lock)
    t0 = time.time()
    self_test()
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if a.tf32:
            torch.set_float32_matmul_precision("high"); torch.backends.cudnn.allow_tf32 = True
        m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
        n_bn = freeze_bn_stats(m)
        cfg = sd.get("cfg", {})
        t1_src = sd.get("t1_source", "rigid")        # checkpoint 가 학습에 쓴 입력 프로토콜
        bal = cfg.get("balance", BalanceConfig(enabled=True))
        n_pairs = int(a.pairs_per_batch or cfg.get("gt_pairs_per_step", 128))
        n_pp = int(a.n_per_pair or cfg.get("n_gt_per_pair", 8))
        print(f"ckpt={a.ckpt} step={sd.get('step')} unet={sd.get('unet_level')} "
              f"in_ch={sd.get('in_channels')} t1={t1_src} BN={n_bn}개(momentum=0)", flush=True)
        print(f"sampling={a.sampling} pairs/batch={n_pairs} n/pair={n_pp} "
              f"batches={a.n_batches} lambda_syn={bal.lambda_syn}", flush=True)

        subs = [l.strip() for l in open(ROOT / a.subjects) if l.strip()][:a.n_subjects]
        assert subs, f"{a.subjects}: subject 없음"
        rng = np.random.default_rng(a.seed)
        torch.manual_seed(a.seed)

        keys = [(md, po) for md in MODES for po in POSTS]
        acc = {k: [[], []] for k in keys}
        syn_all, per_sub = [], []
        for sub in subs:
            batches = (sample_balanced(sub, bal, a.n_batches, n_pairs, n_pp, rng)
                       if a.sampling == "balanced" else sample_uniform(sub, n_pp, rng, a.batch))
            m.eval()
            with torch.no_grad():
                feat = m.atm.encode_anatomy(t1_input(m, sub, t1_src))
            assert float(feat.norm()) > 1e-3, f"{sub}: anatomy feature 가 0 에 가까움"
            loc = {k: [[], []] for k in keys}
            nseen = 0
            syn_sub = []
            for S, P, syn in batches:
                # 필수 검증: 가닥 수, shape, 좌표 유한성
                assert len(S) > 0, f"{sub}: 가닥 0개"
                assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
                assert torch.isfinite(S).all(), f"{sub}: GT 좌표에 NaN/Inf"
                nseen += len(S); syn_sub.append(syn)
                for md, po in keys:
                    m.train() if md == "train" else m.eval()
                    torch.manual_seed(a.seed + nseen)      # mu/sample 간 z 잡음 재현성
                    x, y = batch_sq(m, S, P, feat, po)
                    loc[(md, po)][0].append(x); loc[(md, po)][1].append(y)
            syn_sub = np.concatenate(syn_sub)
            w_sub = np.where(syn_sub, bal.lambda_syn, 1.0)
            row = {"sub": sub, "n_streamlines": int(nseen), "synth_frac": float(syn_sub.mean())}
            for k in keys:
                x, y = np.concatenate(loc[k][0]), np.concatenate(loc[k][1])
                acc[k][0].append(x); acc[k][1].append(y)
                row[f"{k[0]}_{k[1]}"] = summarize(x, y, w_sub)
            syn_all.append(syn_sub); per_sub.append(row)
            r = row[f"{a.mode}_{a.posterior}"]
            print(f"  {sub}: {nseen:,} 가닥 (synth {syn_sub.mean():.0%})  noflip {r['rmse_noflip_mm']:.3f}  "
                  f"flip-aware {r['rmse_flipaware_mm']:.3f}  뒤집힘 {r['frac_flipped']:.2%}", flush=True)

        syn = np.concatenate(syn_all)
        w = np.where(syn, bal.lambda_syn, 1.0)
        overall, real_only = {}, {}
        for k in keys:
            x, y = np.concatenate(acc[k][0]), np.concatenate(acc[k][1])
            name = f"{k[0]}_{k[1]}"
            overall[name] = summarize(x, y, w)
            real_only[name] = summarize(x[~syn], y[~syn]) if (~syn).any() else None

        # --- 필수 검증 -------------------------------------------------------------------
        for name, r in overall.items():
            assert r["n_streamlines"] > 0, name
            # min(a,b) <= a 이므로 수학적으로 반드시 성립. 깨지면 구현 오류다.
            assert r["rmse_flipaware_mm"] <= r["rmse_noflip_mm"] + 1e-9, (name, r)
            assert r["loss_style_flipaware_mm"] <= r["loss_style_noflip_mm"] + 1e-9, (name, r)
            assert 0.0 <= r["frac_flipped"] <= 1.0, (name, r)
            assert np.isfinite([r["rmse_noflip_mm"], r["rmse_flipaware_mm"]]).all(), (name, r)

        prim_key = f"{a.mode}_{a.posterior}"
        prim = overall[prim_key]
        inflated = prim["frac_flipped"] > 0.05 and prim["delta_mm"] > 0.5
        out = {
            "cmd": f"python scripts/44_recon_flip_check.py --ckpt {a.ckpt} --subjects {a.subjects}"
                   f" --n-subjects {a.n_subjects} --n-batches {a.n_batches} --sampling {a.sampling}"
                   f" --mode {a.mode} --posterior {a.posterior} --seed {a.seed}"
                   + (" --tf32" if a.tf32 else ""),
            "args": vars(a), "ckpt_step": sd.get("step"), "t1_source": t1_src,
            "subjects": subs, "sampling": a.sampling, "primary": prim_key,
            "primary_note": "trainer.py:350 이 계산되는 조건 (BN train 모드 + posterior 샘플링)",
            "rmse_noflip": prim["rmse_noflip_mm"],
            "rmse_flipaware": prim["rmse_flipaware_mm"],
            "frac_flipped": prim["frac_flipped"],
            "verdict": "METRIC_INFLATED" if inflated else "METRIC_VALID",
            "criterion": "frac_flipped > 0.05 and delta_mm > 0.5 -> trainer.py 지표를 flip-aware 로 수정",
            "overall": overall, "real_only": real_only, "per_subject": per_sub,
            "sec": time.time() - t0,
        }
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)

        print(f"\n{'BN/posterior':16s} {'noflip':>9s} {'flip-aware':>11s} {'차이':>7s} "
              f"{'뒤집힘':>8s} {'L_recon식':>10s}")
        for name, r in overall.items():
            print(f"{name:16s} {r['rmse_noflip_mm']:9.3f} {r['rmse_flipaware_mm']:11.3f} "
                  f"{r['delta_mm']:7.3f} {r['frac_flipped']:8.2%} "
                  f"{r['loss_style_flipaware_weighted_mm']:10.3f}")
        print(f"\n로그 대조 (step {sd.get('step')}): recon_rmse_mm={prim['rmse_noflip_mm']:.3f}, "
              f"L_recon={prim['loss_style_flipaware_weighted_mm']:.3f}  [primary={prim_key}]")
        print(f"판정: {out['verdict']}  ({out['criterion']})")
        print(f"저장: {p}")
    finally:
        stop.set()
        if lock.exists() and lock.read_text().strip() == f"{socket.gethostname()}:{os.getpid()}":
            lock.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt")
    ap.add_argument("--subjects", default="outputs/splits/train.txt",
                    help="보고된 3.55 mm 는 학습 중 지표라 기본값은 train")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--sampling", default="balanced", choices=["balanced", "uniform"],
                    help="balanced = trainer [R] 분기와 동일, uniform = pair 균등 (참고용)")
    ap.add_argument("--n-batches", type=int, default=8, help="subject 당 sampler batch 수")
    ap.add_argument("--pairs-per-batch", type=int, default=0, help="0 이면 checkpoint cfg")
    ap.add_argument("--n-per-pair", type=int, default=0, help="0 이면 checkpoint cfg")
    ap.add_argument("--batch", type=int, default=1024, help="--sampling uniform 의 forward chunk")
    ap.add_argument("--mode", default="train", choices=["train", "eval"],
                    help="ConvVAE BatchNorm: train = batch 통계 (로그의 지표와 같은 조건)")
    ap.add_argument("--posterior", default="sample", choices=["mu", "sample"],
                    help="sample = trainer 와 동일 (reparameterize), mu = 잡음 없이")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--lock", default="outputs/gpu.lock")
    ap.add_argument("--out", default="outputs/eval/w1a_flip_check.json")
    main(ap.parse_args())
