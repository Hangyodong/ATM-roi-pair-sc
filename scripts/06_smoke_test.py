#!/usr/bin/env python
"""ROI-pair ATM smoke test (pipeline §27 synthetic, §28 real-data).

full training 은 하지 않는다. 각 단계가 실제로 동작하고 gradient 가 끝까지 흐르는지만 본다.

  [A] synthetic : ROI=6, 양성 pair 3, pair 당 8 streamline, P=128. pretrained ConvVAE 를
                  6-ROI 합성 atlas 박스에서 그대로 쓴다 (UNet 은 쓰지 않고 anatomy feature 는 난수).
  [B] real      : subject 1명, 양성 pair 8개 subset. T1 -> UNet -> ROI-pair ATM -> soft SC
                  -> loss -> backward. outputs/roi_pairs/{sub}/{assignments,bundles}.npz 필요.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc import losses as L                                             # noqa: E402
from atm_sc.models.endpoint_assigner import EndpointAssigner               # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                               # noqa: E402
from atm_sc.models.sc_builder import SCBuilder, streamline_lengths         # noqa: E402
from atm_sc.training.synthetic import make_synthetic                       # noqa: E402
from atm_sc.training.trainer import LossWeights, TrainConfig, Trainer      # noqa: E402

results: list[tuple[str, bool]] = []


def rec(name, ok, extra=""):
    results.append((name, bool(ok)))
    print(f"  {name:38s} {'PASS' if ok else 'FAIL'}  {extra}")
    return ok


def hdr(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


def synthetic(device, sc_mode):
    hdr(f"[A] synthetic  (sc_mode={sc_mode})")
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8)
    R = subj.n_roi
    m = ROIPairATM(n_roi=R, coord_min=lo, coord_max=hi, device=device)
    ea = EndpointAssigner(dist, affine, tau=0.5, device=device,
                          d_bg=None if sc_mode == "endpoint" else 2.0)
    # 실측: pretrained UNet 의 anatomy feature 는 |a|~0.09, std~0.004 (PPMI sub-100001).
    # std 1 난수를 넣으면 FiLM 이 발산해 encoder 가 inf/nan 을 낸다.
    a = torch.randn(1, 512, device=device) * 0.004
    pairs = torch.as_tensor(subj.pair_ids, device=device).repeat_interleave(8, 0)
    print(f"  ROI={R}  positive pairs={len(subj.pair_ids)} {subj.pair_ids.tolist()}  N={pairs.shape[0]}  P=128")

    # --- forward ------------------------------------------------------------
    cond = m.condition(a, pairs)
    rec("ROI-pair embedding [N,512]", cond.shape == (pairs.shape[0], 512) and torch.isfinite(cond).all())
    rev = m.condition(a, pairs.flip(1))
    rec("pair order invariance (a,b)==(b,a)", torch.allclose(cond, rev))
    z = m.sample_z(pairs.shape[0])
    S = m.decode(z, cond)
    rec("generated streamline shape [N,128,3]", S.shape == (pairs.shape[0], 128, 3) and torch.isfinite(S).all(),
        f"{tuple(S.shape)}")
    wk = m.weights(cond, z)
    rec("streamline weight w>=0, init==1", (wk >= 0).all() and torch.allclose(wk, torch.ones_like(wk)))
    qs, qe = ea.endpoint_probs(S)
    rec("endpoint probability [N,R]", qs.shape == (pairs.shape[0], R) and torch.isfinite(qs).all(), f"{tuple(qs.shape)}")
    if sc_mode == "endpoint":
        rec("endpoint prob sum == 1", torch.allclose(qs.sum(-1), torch.ones_like(qs.sum(-1)), atol=1e-5))
    builder = SCBuilder(ea, mode=sc_mode)
    sc, num = builder(S, streamline_lengths(S), wk)
    sym = float((sc - sc.T).abs().max())
    rec("SC shape [R,R]", sc.shape == (R, R), f"{tuple(sc.shape)}")
    rec("SC symmetry", sym < 1e-5, f"err={sym:.2e}")
    rec("SC finite", torch.isfinite(sc).all() and torch.isfinite(num).all())
    logits = m.edge_logits(a, pairs)
    rec("edge head logits [N]", logits.shape == (pairs.shape[0],) and torch.isfinite(logits).all())

    # --- losses -------------------------------------------------------------
    gt_w = torch.as_tensor(subj.sc_end, dtype=torch.float32, device=device)
    gt_l = torch.as_tensor(subj.len_end, dtype=torch.float32, device=device)
    S_gt = torch.cat([subj.get_pair(k)[0] for k in range(len(subj.pair_ids))]).to(device)
    mu, logvar = m.encode_streamlines(S_gt, m.condition(a, pairs))
    recon = m.decode(m.reparameterize(mu, logvar), m.condition(a, pairs))
    terms = {
        "L_ATM(recon)": L.stream_recon_loss(recon, S_gt),
        "L_kl": L.kl_loss(mu, logvar),
        "L_adj": L.adjacency_loss(S),
        "L_endpoint": L.endpoint_loss(qs, qe, pairs[:, 0], pairs[:, 1]),
        "L_edge": L.edge_loss(logits, torch.ones_like(logits)),
        "L_SC_corr": L.sc_corr_loss(sc, gt_w),
        "L_SC_mag": L.sc_magnitude_loss(sc, gt_w),
        "L_length": L.tract_length_loss(num, sc, gt_l, gt_w),
    }
    for k, v in terms.items():
        print(f"  {k:38s} {float(v):.6f}")
    rec("finite loss (모든 항)", all(torch.isfinite(v) for v in terms.values()))

    # --- backward: streamline 좌표 + 모델 파라미터 -----------------------------
    S2 = S.detach().clone().requires_grad_(True)
    qs2, qe2 = ea.endpoint_probs(S2); sc2, num2 = builder(S2, streamline_lengths(S2), wk.detach())
    (L.sc_corr_loss(sc2, gt_w) + L.sc_magnitude_loss(sc2, gt_w) + L.endpoint_loss(qs2, qe2, pairs[:, 0], pairs[:, 1])
     + L.tract_length_loss(num2, sc2, gt_l, gt_w)).backward()
    g = S2.grad
    rec("backward", g is not None)
    rec("streamline gradient", g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0,
        f"max|grad|={float(g.abs().max()):.2e}")

    # --- 학습 step 하나 (trainer 전체 경로) ----------------------------------
    tr = Trainer(m, ea, TrainConfig(sc_mode=sc_mode, n_gen_per_pair=8, n_gt_per_pair=8,
                                    gt_pairs_per_step=3, neg_pairs_per_step=3, chunk=16))
    before = [p.detach().clone() for p in m.trainable_parameters()]
    o = tr.step(subj, a)
    groups = {"pair_emb": m.pair_emb, "weight_head": m.weight_head, "edge_head": m.edge_head,
              "convvae": m.atm.net.ae}
    got = {k: any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in v.parameters())
           for k, v in groups.items()}
    rec("model parameter gradient (전 그룹)", all(got.values()), str(got))
    changed = sum(1 for p0, p1 in zip(before, m.trainable_parameters()) if not torch.equal(p0, p1))
    rec("optimizer step 이 파라미터를 바꿈", changed > 0, f"{changed}/{len(before)} tensors")
    unet_grads = sum(1 for p in m.atm.net.unet.parameters() if p.grad is not None)
    rec("UNet 동결", unet_grads == 0)
    print("  trainer step:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()
                              if k.startswith(("L_", "grad_norm", "endpoint_pair", "sc_r", "sc_ccc", "w_mean"))})


def real(device, sub, sc_mode, n_pairs, n_gen):
    hdr(f"[B] real-data  {sub}  (sc_mode={sc_mode}, {n_pairs} pairs x {n_gen})")
    from atm_sc.data.paths import ATLAS, CACHE
    import nibabel as nib
    try:
        from atm_sc.data.dataset import ROIPairSubject
    except ImportError as e:
        print(f"  SKIP: data/dataset.py 없음 ({e})"); return None
    rp = ROOT / "outputs" / "roi_pairs" / sub
    need = [rp / "assignments.npz", rp / "bundles.npz", CACHE / "dist_maps.npy", CACHE / f"{sub}_T1w_syn_W.npy"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print("  SKIP: 필요한 파일 없음 ->", missing); return None

    subj = ROIPairSubject(sub)
    img = nib.load(ATLAS)
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), img.affine, tau=0.5, device=device,
                          d_bg=None if sc_mode == "endpoint" else 2.0)
    m = ROIPairATM(n_roi=subj.n_roi, device=device)
    from atm_sc.training.run import t1_input
    x = t1_input(m, sub)                       # 모델 채널 구성대로 (2채널이면 rigid T1 + WM)
    t0 = time.time(); a = m.encode_anatomy(x); t_enc = time.time() - t0
    rec("T1 encoder 1회 -> [1,512]", a.shape == (1, 512) and torch.isfinite(a).all(), f"{t_enc:.2f}s")
    print(f"  positive pairs K={len(subj.pair_ids)}  positive ratio={subj.positive_ratio:.3f}  "
          f"sc_end sum={int(np.asarray(subj.sc_end).sum())}")

    tr = Trainer(m, ea, TrainConfig(sc_mode=sc_mode, n_gen_per_pair=n_gen, max_pairs_per_step=n_pairs,
                                    n_gt_per_pair=8, gt_pairs_per_step=n_pairs, neg_pairs_per_step=n_pairs,
                                    chunk=2048))
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    o = tr.step(subj, a)
    keys = ["L_recon", "L_kl", "L_adj", "L_endpoint", "L_edge", "L_corr", "L_mag", "L_length",
            "endpoint_pair_acc", "edge_acc", "sc_r", "sc_ccc", "w_mean", "grad_norm_total", "n_generated", "step_sec"]
    for k in keys:
        if k in o:
            print(f"  {k:38s} {o[k]:.5f}" if isinstance(o[k], float) else f"  {k:38s} {o[k]}")
    rec("real step: 모든 loss 유한", all(np.isfinite(v) for k, v in o.items() if k.startswith("L_")))
    rec("real step: gradient 유한", np.isfinite(o["grad_norm_total"]) and o["grad_norm_total"] > 0)
    # 두 번째 step 도 돌아가는지 (BatchNorm 통계/optimizer 상태)
    o2 = tr.step(subj, a)
    rec("real step x2", all(np.isfinite(v) for k, v in o2.items() if k.startswith("L_")))
    return True


def trainable_encoder(device, sub, sc_mode, n_pairs, n_gen):
    """최종 전략 §2/§6: T1 encoder 전체 unfreeze. gradient 가 UNet 까지, 특히 SC loss 에서 오는지."""
    hdr(f"[C] trainable T1 encoder  {sub}  (unet_level=full, sc_mode={sc_mode})")
    from atm_sc.data.paths import ATLAS, CACHE
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.training.run import t1_input
    import nibabel as nib
    rp = ROOT / "outputs" / "roi_pairs" / sub
    if not all(p.exists() for p in (rp / "assignments.npz", rp / "bundles.npz", CACHE / "dist_maps.npy", CACHE / f"{sub}_T1w_syn_W.npy")):
        print("  SKIP: 전처리 산출물 없음"); return None
    subj = ROIPairSubject(sub)
    img = nib.load(ATLAS)
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), img.affine, tau=0.5, device=device,
                          d_bg=None if sc_mode == "endpoint" else 2.0)
    m = ROIPairATM(n_roi=subj.n_roi, trainable="full", device=device)
    pc = m.param_counts()
    print(f"  params: trainable {pc['trainable']/1e6:.2f}M (t1_enc {pc['t1_encoder']/1e6:.2f}M, vae_enc {pc['vae_encoder']/1e6:.2f}M, "
          f"dec {pc['decoder']/1e6:.2f}M, heads {pc['heads']/1e6:.2f}M) | frozen {pc['frozen']/1e6:.2f}M")
    x = t1_input(m, sub)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    u = m.atm.net.unet
    snap = {n: p.detach().clone() for n, p in [("conv1_1", u.conv1_1.weight), ("conv2_1", u.conv2_1.weight),
                                                 ("conv4_1", u.conv4_1.weight), ("fc", u.fc.weight)]}
    vae_snap = m.atm.net.ae.conv1.weight.detach().clone(); dec_snap = m.atm.net.ae.deconv1.weight.detach().clone()

    # (1) SC loss 만 -> T1 encoder gradient 가 오는가
    tr = Trainer(m, ea, TrainConfig(sc_mode=sc_mode, n_gen_per_pair=n_gen, max_pairs_per_step=n_pairs,
                                    gt_pairs_per_step=n_pairs, neg_pairs_per_step=n_pairs, chunk=2048,
                                    active={"corr", "mag"}))
    o = tr.step(subj, x)
    g_sc = {n: float(p.grad.abs().max()) if p.grad is not None else 0.0
            for n, p in [("conv1_1", u.conv1_1.weight), ("conv4_1", u.conv4_1.weight)]}
    print(f"  SC-only step: L_corr {o.get('L_corr', float('nan')):.4f} L_mag {o.get('L_mag', float('nan')):.4f} "
          f"dL/da(SC) {o.get('dLda_G', 0):.3e} | UNet grad conv1_1 {g_sc['conv1_1']:.2e} conv4_1 {g_sc['conv4_1']:.2e} "
          f"| {o['step_sec']:.1f}s peak {o.get('peak_vram_gb', 0):.2f} GB")
    rec("SC loss -> T1 encoder gradient", o.get("dLda_G", 0) > 0 and g_sc["conv1_1"] > 0 and g_sc["conv4_1"] > 0)

    # (2) 전체 objective 한 step: 모든 그룹 gradient + 파라미터 실제 변경
    tr = Trainer(m, ea, TrainConfig(sc_mode=sc_mode, n_gen_per_pair=n_gen, max_pairs_per_step=n_pairs,
                                    gt_pairs_per_step=n_pairs, neg_pairs_per_step=n_pairs, chunk=2048))
    o = tr.step(subj, x)
    for k in ("L_recon", "L_kl", "L_geom", "L_endpoint", "L_edge", "L_corr", "L_mag", "L_length"):
        if k in o: print(f"  {k:38s} {o[k]:.5f}")
    for k in ("dLda_G", "dLda_R", "dLda_E", "dLda_total", "gnorm_t1_encoder", "gnorm_vae_encoder", "gnorm_decoder", "gnorm_heads", "grad_norm_total"):
        if k in o: print(f"  {k:38s} {o[k]:.4e}")
    rec("loss finite (전체 objective)", all(np.isfinite(v) for k, v in o.items() if k.startswith("L_")))
    rec("backward / gradient finite", np.isfinite(o["grad_norm_total"]) and o["grad_norm_total"] > 0)
    rec("T1 encoder gradient", o.get("gnorm_t1_encoder", 0) > 0)
    rec("VAE encoder gradient", o.get("gnorm_vae_encoder", 0) > 0)
    rec("decoder gradient", o.get("gnorm_decoder", 0) > 0)
    hg = {n: any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in mod.parameters())
          for n, mod in (("pair_emb", m.pair_emb), ("edge_head", m.edge_head), ("weight_head", m.weight_head))}
    rec("ROI-pair embedding gradient", hg["pair_emb"])
    rec("edge/weight head gradient", hg["edge_head"] and hg["weight_head"], str(hg))
    ch = {n: float((p.detach() - snap[n]).abs().max()) for n, p in [("conv1_1", u.conv1_1.weight), ("conv2_1", u.conv2_1.weight),
                                                                      ("conv4_1", u.conv4_1.weight), ("fc", u.fc.weight)]}
    rec("T1 encoder 파라미터 실제 변경 (optimizer step)", all(v > 0 for v in ch.values()),
        " ".join(f"{k}={v:.1e}" for k, v in ch.items()))
    rec("VAE enc / decoder 파라미터 변경", float((m.atm.net.ae.conv1.weight - vae_snap).abs().max()) > 0
        and float((m.atm.net.ae.deconv1.weight - dec_snap).abs().max()) > 0)
    rec("UNet segmentation 가지 동결", not any(p.grad is not None for p in u.final_conv.parameters()))
    print(f"  step {o['step_sec']:.1f}s | peak VRAM {o.get('peak_vram_gb', 0):.2f} GB | anat_norm {o['anat_norm']:.4f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-mode", default="endpoint", choices=["endpoint", "pass"])
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-trainable", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hdr("environment")
    print(f"  device {dev} | GPU {torch.cuda.get_device_name(0) if dev == 'cuda' else '-'} | torch {torch.__version__}")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    synthetic(dev, a.sc_mode)
    r = None if a.skip_real else real(dev, a.sub, a.sc_mode, a.n_pairs, a.n_gen)
    if not a.skip_real and not a.skip_trainable:
        trainable_encoder(dev, a.sub, a.sc_mode, a.n_pairs, a.n_gen)
    hdr("summary")
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if dev == "cuda":
        print(f"\n  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  elapsed   {time.time()-t0:.1f}s")
    if r is None and not a.skip_real:
        print("  real-data: SKIPPED")
    ok = all(o for _, o in results)
    print(f"\nFINAL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
