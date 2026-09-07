#!/usr/bin/env python
"""통합 스모크 테스트: T1 -> anatomy feature -> KDE latent -> streamline -> soft SC -> gradient.

확인 항목
  1. T1 encoder 가 subject x bundle 당 1회만 호출되고 결과를 캐시할 수 있는가
  2. 64-D latent N개 -> [N,128,3] mm streamline
  3. upstream 의 3000 하드코딩(D3) 없이 임의 N 이 동작하는가
  4. .eval() 강제(D2) 로 anatomy feature 가 결정적인가
  5. SC loss 의 gradient 가 decoder 까지 흐르는가 (UNet 은 동결)
"""
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atm_sc.models.atm_adapter import ATMBundle, BundleNorm, sample_latents   # noqa: E402
from atm_sc.models.endpoint_assigner import RoiAssigner                            # noqa: E402
from atm_sc.models.sc_builder import pass_sc, streamline_lengths              # noqa: E402
from atm_sc import losses as L                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUNDLE, SUB, N_ROI = "AF_L", "sub-000001", 82

t1 = np.load(ROOT / "outputs" / "cache" / f"{SUB}_T1w_syn_W.npy")
norm = BundleNorm.from_upstream(BUNDLE)
t1n = norm.normalize_t1(t1)
print(f"T1 W {t1.shape}  정규화 후 [{t1n.min():.3f}, {t1n.max():.3f}]")

atm = ATMBundle(BUNDLE, norm, device=DEV)
atm.freeze_unet()
dec = atm.decoder_parameters()
print(f"{BUNDLE}: 학습 파라미터 {sum(p.numel() for p in dec):,} / 전체 "
      f"{sum(p.numel() for p in atm.net.parameters()):,}")

x = torch.tensor(t1n.reshape(1, 1, *t1.shape), dtype=torch.float32)
t0 = time.time(); a1 = atm.encode_anatomy(x); t_enc = time.time() - t0
a2 = atm.encode_anatomy(x)
det = float((a1 - a2).abs().max())
print(f"anatomy feature {tuple(a1.shape)} in {t_enc:.2f}s  두 번 호출 차이 = {det:.3e}")
assert det == 0.0, "eval() 을 강제했는데도 비결정적 (D2 우회 실패)"

for n in (777, 3000):                                     # D3: 3000 하드코딩 우회 확인
    z = torch.from_numpy(sample_latents(BUNDLE, n, seed=0)).to(DEV)
    mm = atm.decode_mm(z, a1)
    assert mm.shape == (n, 128, 3), mm.shape
    assert torch.isfinite(mm).all()
    print(f"  N={n:>5}: streamlines {tuple(mm.shape)}  "
          f"mm bbox [{mm.amin((0,1)).tolist()}] .. [{mm.amax((0,1)).tolist()}]")

img = nib.load(ROOT / "DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz")
ra = RoiAssigner(np.load(ROOT / "outputs" / "cache" / "dist_maps.npy"), img.affine, device=DEV)

z = torch.from_numpy(sample_latents(BUNDLE, 3000, seed=0)).to(DEV)
mm = atm.decode_mm(z, a1)
u = ra.visit_probs(mm)
L = streamline_lengths(mm)
sc, num = pass_sc(u, L)
print(f"soft SC {tuple(sc.shape)} sum={float(sc.sum()):.1f} nnz={int((sc>1e-3).sum())} "
      f"| streamline 길이 평균 {float(L.mean()):.1f}mm")
assert float(sc.sum()) > 0, "SC 가 전부 0 — streamline 이 atlas 밖"

gt = torch.from_numpy(np.load(ROOT / "outputs" / "cache" / f"{SUB}_hardsc.npz")["GT_W"]).float().to(DEV)
loss = Lsc.sc_corr_loss(sc, gt) + 0.05 * Lsc.sc_magnitude_loss(sc, gt) \
       + 1.0 * Lloc.adjacency_loss(mm)
loss.backward()
gn = [(p.grad is not None and float(p.grad.abs().max()) > 0) for p in dec]
un = [p.grad for p in atm.net.unet.parameters() if p.grad is not None]
print(f"loss={float(loss):.4f}  decoder 파라미터 {sum(gn)}/{len(dec)} 에 gradient 도달, "
      f"UNet gradient {len(un)} (0 이어야 함)")
assert all(gn), "decoder 일부에 gradient 가 안 옴"
assert not un, "UNet 이 동결되지 않았음"
print("\nINTEGRATION OK")
