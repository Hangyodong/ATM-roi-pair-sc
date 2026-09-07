#!/usr/bin/env python
"""§30 생성 throughput benchmark.

streamline batch(chunk) 크기와 정밀도별로 ATMBundle.generate / decode 처리량과 peak VRAM 을 잰다.
T1 encoder 는 subject 당 1회이므로 따로 1회만 잰다. .trk 저장은 CPU 작업이라 따로 잰다.

bf16 은 일부러 제외한다: 같은 latent 로 fp32 대비 좌표 오차 max 57 mm (p99 4.6 mm) 가 측정되었다.
decoder 가 k=127 Conv1d + BatchNorm 이라 8비트 가수로는 부족하다. fp16 은 max 6.4 mm / p99 0.6 mm.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

from atm_sc.models.atm_adapter import ATMBundle, BundleNorm, load_kde, sample_latents   # noqa: E402
from atm_sc.data.paths import CACHE                                                      # noqa: E402
from atm_sc.spaces import W_SHAPE                                                        # noqa: E402

OUT = ROOT / "outputs" / "benchmarks"
DT = {"fp32": None, "fp16": torch.float16}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeat):
    ts, peak = [], 0.0
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.time(); out = fn(); sync(); ts.append(time.time() - t0)
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
    return statistics.median(ts), peak, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="AF_L")
    ap.add_argument("--batches", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=list(DT))
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    print(f"device {dev}  {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}  torch {torch.__version__}")

    norm = BundleNorm.from_upstream(a.bundle)
    atm = ATMBundle(a.bundle, norm, device=dev); atm.freeze_unet()

    cache = CACHE / "sub-100001_T1w_syn_W.npy"
    if cache.exists():
        x = torch.tensor(norm.normalize_t1(np.load(cache)).reshape(1, 1, *W_SHAPE), dtype=torch.float32)
        t_enc, v_enc, feat = timed(lambda: atm.encode_anatomy(x), 1)
        print(f"T1 encoder (sub-100001, 1회): {t_enc:.2f}s  peak VRAM {v_enc:.2f} GB")
    else:
        feat = torch.randn(1, 512, device=dev); t_enc = float("nan")
        print("T1 캐시 없음 -> 무작위 anatomy feature 사용")

    t0 = time.time(); load_kde(a.bundle); t_kde = time.time() - t0
    t0 = time.time(); z = torch.from_numpy(sample_latents(a.bundle, a.n, seed=0)).to(dev); t_samp = time.time() - t0
    print(f"KDE load {t_kde:.1f}s | KDE sample({a.n}) {t_samp:.2f}s (CPU)")

    rows = []
    print(f"\n{'batch':>6} {'prec':>5} | {'generate s':>10} {'stream/s':>10} | {'decode-only s':>13} {'stream/s':>10} | {'peak VRAM GB':>12}")
    for prec in a.precisions:
        for b in a.batches:
            amp = DT[prec]
            # generate = KDE 샘플링(CPU) + decode(GPU) end-to-end
            t_gen, v_gen, mm = timed(lambda: atm.generate(a.n, feat, chunk=b, seed=0, amp_dtype=amp), a.repeat)
            # decode-only = 미리 뽑은 z 로 GPU decode 만

            def dec():
                with torch.inference_mode():
                    return torch.cat([c for c in atm.decode_chunks(z, feat, chunk=b, amp_dtype=amp)])
            t_dec, v_dec, mm2 = timed(dec, a.repeat)
            assert mm.shape == mm2.shape == (a.n, 128, 3), (mm.shape, mm2.shape)
            assert torch.isfinite(mm).all() and torch.isfinite(mm2).all()
            peak = max(v_gen, v_dec)
            print(f"{b:>6} {prec:>5} | {t_gen:10.3f} {a.n/t_gen:10,.0f} | {t_dec:13.3f} {a.n/t_dec:10,.0f} | {peak:12.2f}")
            rows.append({"bundle": a.bundle, "n": a.n, "batch": b, "precision": prec, "generate_s": t_gen,
                         "generate_streamlines_per_s": a.n / t_gen, "decode_s": t_dec,
                         "decode_streamlines_per_s": a.n / t_dec, "peak_vram_gb": peak})

    # .trk 저장 (CPU). 좌표는 전부 GPU 에서 만든 뒤 마지막에 내린다.
    import nibabel as nib
    arr = mm.cpu().numpy().astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    trk = OUT / f"bench_{a.bundle}_{a.n}.trk"
    t0 = time.time()
    tg = nib.streamlines.Tractogram(list(arr), affine_to_rasmm=np.eye(4))
    hdr = nib.streamlines.TrkFile.create_empty_header()
    hdr["voxel_to_rasmm"] = np.eye(4, dtype=np.float32); hdr["dimensions"] = np.array(W_SHAPE, np.int16)
    hdr["voxel_sizes"] = np.ones(3, np.float32); hdr["voxel_order"] = b"RAS"
    nib.streamlines.TrkFile(tg, header=hdr).save(str(trk))
    t_trk = time.time() - t0
    print(f"\n.trk write ({a.n} streamlines, CPU, nibabel): {t_trk:.2f}s -> {trk}  ({trk.stat().st_size/1e6:.1f} MB)")

    csvp = OUT / f"generation_{a.bundle}.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["t1_encoder_s", "trk_write_s", "kde_load_s"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "t1_encoder_s": t_enc, "trk_write_s": t_trk, "kde_load_s": t_kde})
    print(f"CSV -> {csvp}")


if __name__ == "__main__":
    main()
