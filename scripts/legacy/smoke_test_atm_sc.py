#!/usr/bin/env python
"""ATM -> SC-aware fine-tuning smoke test.

두 단계를 돈다.
  [A] synthetic  : 작은 합성 atlas/streamline 으로 endpoint assigner + SC builder + loss +
                   backward 를 검증한다. GPU/모델이 없어도 반드시 통과해야 한다.
  [B] integration: 실제 pretrained ATM 으로 T1 -> anatomy feature -> latent batch ->
                   decoder -> streamline -> SC -> loss -> backward 를 끝까지 돈다.
                   자산이 없으면 이유를 남기고 SKIP 한다.

full training 은 하지 않는다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc import losses as L                                     # noqa: E402
from atm_sc.models.endpoint_assigner import EndpointAssigner, build_distance_maps   # noqa: E402
from atm_sc.models.sc_builder import ChunkedSC, SCBuilder, streamline_lengths       # noqa: E402
from atm_sc.spaces import W_SHAPE, voxel_to_mm                     # noqa: E402

P_POINTS = 128
LAM = dict(atm=1.0, endpoint=0.1, corr=0.05, mag=0.05, length=0.02)
results: list[tuple[str, bool]] = []


def rec(name, ok):
    results.append((name, bool(ok)))
    print(f"  {name:34s} {'PASS' if ok else 'FAIL'}")
    return ok


def hdr(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


# ---------------------------------------------------------------- synthetic
def synthetic(device, mode):
    hdr(f"[A] synthetic  (mode={mode})")
    R, N = 6, 12
    shape = (24, 20, 18)
    atlas = np.zeros(shape, np.int16)
    for r in range(R):
        atlas[2 + r * 3: 4 + r * 3, 4:16, 4:14] = r + 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-24.0, -20.0, -18.0]
    dist = build_distance_maps(atlas, R, (2.0, 2.0, 2.0))

    rng = np.random.default_rng(0)
    pairs = [(0, 5), (1, 4), (2, 3), (0, 3), (1, 5), (2, 4)] * 2
    S = []
    for a, b in pairs:
        pa = voxel_to_mm(np.array([3.0 + a * 3, 10.0, 9.0]), affine)
        pb = voxel_to_mm(np.array([3.0 + b * 3, 10.0, 9.0]), affine)
        t = np.linspace(0, 1, P_POINTS)[:, None]
        S.append(pa * (1 - t) + pb * t + rng.normal(0, 0.3, (P_POINTS, 3)))
    mm = torch.tensor(np.stack(S), dtype=torch.float32, device=device).requires_grad_(True)

    ea = EndpointAssigner(dist, affine, tau=0.5, device=device,
                          d_bg=None if mode == "endpoint" else 2.0)
    builder = SCBuilder(ea, mode=mode)
    qs, qe = ea.endpoint_probs(mm)
    sc, num = builder(mm)

    print(f"  streamline shape                   {tuple(mm.shape)}")
    print(f"  endpoint probability shape         {tuple(qs.shape)}")
    print(f"  SC shape                           {tuple(sc.shape)}")
    sym = float((sc - sc.T).abs().max())
    print(f"  SC symmetry error                  {sym:.3e}")

    rec("streamline shape [N,P,3]", mm.shape == (N, P_POINTS, 3))
    rec("endpoint prob shape [N,R]", qs.shape == (N, R))
    if mode == "endpoint":
        rec("ROI probability sum == 1", torch.allclose(qs.sum(-1), torch.ones(N, device=device), atol=1e-5))
    rec("SC shape [R,R]", sc.shape == (R, R))
    rec("SC symmetric", sym < 1e-5)
    rec("SC diagonal == 0", float(sc.diagonal().abs().max()) == 0.0)
    rec("SC finite & nonzero", bool(torch.isfinite(sc).all()) and float(sc.sum()) > 0)

    # chunk 로 나눈 부분 SC 의 합 == 전체 SC (v2 §18: loss 는 합친 뒤 한 번만)
    acc = ChunkedSC(R, device=device)
    for i in range(0, N, 5):
        acc += builder(mm[i:i + 5])
    rec("chunked partial SC == full SC", torch.allclose(acc.sc, sc, atol=1e-4))

    gt_w = sc.detach() * 1.3 + 0.5
    gt_l = (num / (sc + 1e-8)).detach() * 1.1 + 1.0
    i_gt = torch.tensor([p[0] for p in pairs], device=device)
    j_gt = torch.tensor([p[1] for p in pairs], device=device)

    l_atm = L.adjacency_loss(mm)
    l_end = L.endpoint_loss(qs, qe, i_gt, j_gt)
    l_corr = L.sc_corr_loss(sc, gt_w)
    l_mag = L.sc_magnitude_loss(sc, gt_w)
    l_len = L.tract_length_loss(num, sc, gt_l, gt_w)
    total = (LAM["atm"] * l_atm + LAM["endpoint"] * l_end + LAM["corr"] * l_corr
             + LAM["mag"] * l_mag + LAM["length"] * l_len)

    print(f"  endpoint loss                      {float(l_end):.6f}")
    print(f"  SC corr loss                       {float(l_corr):.6f}")
    print(f"  SC magnitude loss                  {float(l_mag):.6f}")
    print(f"  tract length loss                  {float(l_len):.6f}")
    print(f"  L_ATM (adjacency)                  {float(l_atm):.6f}")
    print(f"  total loss                         {float(total):.6f}")
    rec("all losses finite", all(torch.isfinite(x) for x in (l_atm, l_end, l_corr, l_mag, l_len, total)))

    total.backward()
    g = mm.grad
    ok = g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0
    print(f"  streamline gradient                {'PASS' if ok else 'FAIL'}"
          f"  (max |grad| = {float(g.abs().max()):.3e})")
    rec("streamline gradient", ok)
    return all(o for _, o in results)


# -------------------------------------------------------------- integration
def integration(device, mode, n_streamlines, chunk, amp):
    hdr(f"[B] ATM integration  (mode={mode})")
    from atm_sc.models.atm_adapter import ATMBundle, BundleNorm, UPSTREAM, sample_latents
    from atm_sc.data.paths import ATLAS, CACHE, subjects, t1_path

    bundle = "AF_L"
    ckpt = UPSTREAM / "models" / bundle / f"atmvae_{bundle}.pth"
    kde = UPSTREAM / "kde_models" / bundle / "kde_model.joblib"
    if not (ckpt.exists() and kde.exists() and ATLAS.exists()):
        print(f"  SKIP: 자산 없음 (ckpt={ckpt.exists()} kde={kde.exists()} atlas={ATLAS.exists()})")
        return None
    import nibabel as nib
    img = nib.load(ATLAS)
    atlas = np.asanyarray(img.dataobj).astype(np.int16)
    R = int(atlas.max())

    dmp = CACHE / "dist_maps.npy"
    if dmp.exists():
        dist = np.load(dmp)
    else:
        print("  ROI 거리맵 생성 중 ...", flush=True)
        dist = build_distance_maps(atlas, R, img.header.get_zooms()[:3])
        dmp.parent.mkdir(parents=True, exist_ok=True); np.save(dmp, dist)

    subs = subjects()
    t1w = None
    for s in subs[:1]:
        c = CACHE / f"{s}_T1w_syn_W.npy"
        if c.exists():
            t1w = np.load(c); sub = s; break
    if t1w is None:
        sub = subs[0]
        print(f"  {sub} T1 을 W 격자로 정합 중 (최초 1회, 수 분) ...", flush=True)
        from atm_sc.data.prepare_t1 import prepare_subject
        t1w = prepare_subject(t1_path(sub), mode="syn", out_dir=CACHE)
        np.save(CACHE / f"{sub}_T1w_syn_W.npy", t1w)

    norm = BundleNorm.from_upstream(bundle)
    atm = ATMBundle(bundle, norm, device=device)
    atm.freeze_unet()
    dec = atm.decoder_parameters()

    x = torch.tensor(norm.normalize_t1(t1w).reshape(1, 1, *W_SHAPE), dtype=torch.float32)
    t0 = time.time()
    a = atm.encode_anatomy(x)                      # subject x bundle 당 1회
    t_enc = time.time() - t0
    a2 = atm.encode_anatomy(x)
    rec("T1 encoder 결정적 (.eval 강제)", float((a - a2).abs().max()) == 0.0)
    print(f"  subject                            {sub}")
    print(f"  anatomy feature                    {tuple(a.shape)}  ({t_enc:.2f}s, 1회만 호출)")

    ea = EndpointAssigner(dist, img.affine, tau=0.5, device=device,
                          d_bg=None if mode == "endpoint" else 2.0)
    builder = SCBuilder(ea, mode=mode)

    z = torch.from_numpy(sample_latents(bundle, n_streamlines, seed=0)).to(device)
    acc = ChunkedSC(R, device=device)
    n_pts = 0
    amp_dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    for c in atm.decode_chunks(z, a, chunk=chunk, amp_dtype=amp_dtype):
        acc += builder(c, streamline_lengths(c))
        n_pts += c.shape[0]
    sc, num = acc.sc, acc.num
    print(f"  streamline shape                   ({n_pts}, {P_POINTS}, 3)   "
          f"chunk={chunk}  amp={amp}")
    print(f"  SC shape                           {tuple(sc.shape)}")
    sym = float((sc - sc.T).abs().max())
    print(f"  SC symmetry error                  {sym:.3e}")
    rec("SC 대칭 / 유한 / 비영", sym < 1e-3 and bool(torch.isfinite(sc).all()) and float(sc.sum()) > 0)

    gt = CACHE / f"{sub}_hardsc.npz"
    if gt.exists():
        z_ = np.load(gt); gt_w = torch.from_numpy(z_["GT_W"]).float().to(device)
        gt_l = torch.from_numpy(z_["GT_L"]).float().to(device)
    else:
        import scipy.io as sio
        from atm_sc.data.paths import SC_MAT
        row = [r for r in sio.loadmat(SC_MAT)["data"][0] if str(r["subject"][0]) == sub][0]
        gt_w = torch.tensor(np.asarray(row["SC_weight"], np.float32), device=device)
        gt_l = torch.tensor(np.asarray(row["SC_length"], np.float32), device=device)

    # gradient 는 chunk 를 다시 돌면서 흘린다 (전체 그래프를 들고 있지 않기 위해)
    for p in dec:
        p.grad = None
    l_corr = L.sc_corr_loss(sc, gt_w); l_mag = L.sc_magnitude_loss(sc, gt_w)
    l_len = L.tract_length_loss(num, sc, gt_l, gt_w)
    total = LAM["corr"] * l_corr + LAM["mag"] * l_mag + LAM["length"] * l_len
    print(f"  SC corr loss                       {float(l_corr):.6f}")
    print(f"  SC magnitude loss                  {float(l_mag):.6f}")
    print(f"  tract length loss                  {float(l_len):.6f}")
    print(f"  total loss                         {float(total):.6f}")
    rec("실제 GT SC 로 loss 유한", bool(torch.isfinite(total)))

    total.backward()
    gd = sum(1 for p in dec if p.grad is not None and float(p.grad.abs().max()) > 0)
    gu = sum(1 for p in atm.net.unet.parameters() if p.grad is not None)
    print(f"  model gradient                     decoder {gd}/{len(dec)}, UNet {gu} (0 이어야 함)")
    rec("decoder gradient (모든 파라미터)", gd == len(dec))
    rec("UNet 동결", gu == 0)

    m = L.sc_metrics(sc, gt_w)
    print(f"  (참고) ATM baseline vs GT SC       r={m['r']:.4f}  ccc={m['ccc']:.4f}  "
          f"F1={m['edge_f1']:.4f}   * AF_L 한 bundle 만, fine-tuning 이전")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="endpoint", choices=["endpoint", "pass"])
    ap.add_argument("--n-streamlines", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"])
    ap.add_argument("--skip-integration", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hdr("environment")
    print(f"  device                             {dev}")
    print(f"  GPU                                "
          f"{torch.cuda.get_device_name(0) if dev == 'cuda' else '-'}")
    if dev == "cuda":
        print(f"  VRAM total                         "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  cudnn                              {torch.backends.cudnn.version()}")
        torch.cuda.reset_peak_memory_stats()
    print(f"  torch                              {torch.__version__}")

    t0 = time.time()
    synthetic(dev, a.mode)
    integ = None if a.skip_integration else integration(
        dev, a.mode, a.n_streamlines, a.chunk, a.amp)
    el = time.time() - t0

    hdr("summary")
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if dev == "cuda":
        print(f"\n  peak VRAM                          "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  elapsed time                       {el:.1f}s")
    if integ is None and not a.skip_integration:
        print("  integration                        SKIPPED (자산 없음)")
    ok = all(o for _, o in results)
    print(f"\nFINAL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
