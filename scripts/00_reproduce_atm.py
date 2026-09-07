#!/usr/bin/env python
"""Phase 0 — 공식 ATM inference 재현 (docs/ATM_ROI_PAIR_SC_FINETUNING_PIPELINE.md §21).

이 서버에는 ANTs / MRtrix / MATLAB / scilpy 가 없으므로 infer.py 의 후처리
(filtering / trimming / to_native) 는 재현할 수 없다. 재현 가능한 부분은 **생성 단계**뿐이고,
그것도 upstream 이 배포한 예제 subject(sub-1135, 이미 MNI 193x229x193 로 정합된 T1) 로만 가능하다.

upstream 결함 우회 (원본 무수정):
  D2  infer.py 는 .eval() 을 호출하지 않아 UNet Dropout3d 가 살아 있다 -> 여기서는 eval 을
      기본으로 쓰고, train mode 3회 실행으로 upstream 의 비결정성을 정량화해 같이 출력한다.
  D3  repeat(3000, 1) 하드코딩 -> 임의 N.
  D4  좌표 상수 경로 (data/ vs supp/).
전체 UNet forward 는 23 GB 에서 OOM 이라 인코더 가지만 실행한다 (CPU 전체 forward 와 bit-exact 확인됨).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

from atm_sc.models.atm_adapter import (ATMBundle, BundleNorm, BUNDLES, UPSTREAM,   # noqa: E402
                                       load_kde)
from atm_sc.spaces import W_AFFINE, W_SHAPE, apply_affine                        # noqa: E402

OUT = ROOT / "outputs" / "reproduce"


def load_example_t1(sub: str) -> np.ndarray:
    p = UPSTREAM / "data" / sub / "anat" / f"{sub}__T12mniWarped.nii.gz"
    assert p.exists(), f"예제 T1 없음: {p}"
    img = nib.load(p)
    raw = img.get_fdata()
    assert raw.shape == W_SHAPE, f"예제 T1 shape {raw.shape} != {W_SHAPE}"
    assert np.isfinite(raw).all() and raw.max() > 0
    return raw


def save_trk_tck(mm: np.ndarray, trk_path: Path, tck_path: Path) -> None:
    """nibabel 만으로 저장. 좌표는 이미 RASMM 이므로 affine_to_rasmm = I."""
    ref = UPSTREAM / "data" / "sub-1135" / "tractography" / "sub-1135__AF_L_mni_1mm.trk"
    assert ref.exists(), ref
    ref_hdr = nib.streamlines.load(str(ref), lazy_load=True).header
    hdr = nib.streamlines.TrkFile.create_empty_header()
    for k in ("voxel_to_rasmm", "dimensions", "voxel_sizes", "voxel_order"):
        hdr[k] = ref_hdr[k]
    assert tuple(int(x) for x in hdr["dimensions"]) == W_SHAPE, hdr["dimensions"]
    tg = nib.streamlines.Tractogram(list(mm.astype(np.float32)), affine_to_rasmm=np.eye(4))
    trk_path.parent.mkdir(parents=True, exist_ok=True)
    nib.streamlines.TrkFile(tg, header=hdr).save(str(trk_path))
    nib.streamlines.TckFile(tg).save(str(tck_path))
    for p in (trk_path, tck_path):
        assert p.exists() and p.stat().st_size > 0, p
    back = nib.streamlines.load(str(trk_path)).tractogram.streamlines
    assert len(back) == len(mm), (len(back), len(mm))
    assert np.allclose(back[0], mm[0], atol=1e-3), "trk 왕복 후 좌표 불일치"


def lengths_of(mm: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(mm, axis=1), axis=-1).sum(1)


def compare_with_gt(mm: np.ndarray, sub: str, bundle: str) -> dict | None:
    """GT 와 비교. 그 디렉터리의 trk 는 _mni_1mm 붙은 것만 MNI 공간이다 (나머지는 native)."""
    gt = UPSTREAM / "data" / sub / "tractography" / f"{sub}__{bundle}_mni_1mm.trk"
    if not gt.exists():
        print(f"  GT 비교      SKIP — {gt.name} 없음 (native 공간 trk 는 비교 불가)")
        return None
    from scipy.ndimage import distance_transform_edt
    gts = nib.streamlines.load(str(gt)).tractogram.streamlines          # RASMM
    gt_pts = np.concatenate([np.asarray(s) for s in gts])
    gt_len = np.array([np.linalg.norm(np.diff(np.asarray(s), axis=0), axis=1).sum() for s in gts])
    inv = np.linalg.inv(W_AFFINE)

    def vox(p):
        v = np.rint(apply_affine(inv, p.astype(np.float64))).astype(np.int64)
        ok = np.all((v >= 0) & (v < np.array(W_SHAPE)), 1)
        return v[ok], ok

    def within(src_pts, dst_pts, r=3.0):
        mask = np.zeros(W_SHAPE, bool)
        v, _ = vox(dst_pts); mask[v[:, 0], v[:, 1], v[:, 2]] = True
        d = distance_transform_edt(~mask)                          # W 는 1 mm 등방
        v2, ok = vox(src_pts)
        hit = np.zeros(len(src_pts), bool)
        hit[ok] = d[v2[:, 0], v2[:, 1], v2[:, 2]] <= r
        return float(hit.mean())

    gen_pts = mm.reshape(-1, 3)
    res = {"gt_n": len(gts), "gt_len_mean": float(gt_len.mean()),
           "overlap_gen_in_gt3mm": within(gen_pts, gt_pts),
           "coverage_gt_in_gen3mm": within(gt_pts, gen_pts)}
    print(f"  GT 비교      GT {res['gt_n']} streamlines, GT 길이 평균 {res['gt_len_mean']:.1f} mm")
    print(f"               생성점이 GT 3 mm 이내: {res['overlap_gen_in_gt3mm']:.3f}   "
          f"GT 점이 생성 3 mm 이내: {res['coverage_gt_in_gen3mm']:.3f}")
    return res


def run_bundle(bundle: str, sub: str, n: int, raw_t1: np.ndarray, device: str) -> dict:
    print(f"\n=== {bundle} ===")
    norm = BundleNorm.from_upstream(bundle)
    x = torch.tensor(norm.normalize_t1(raw_t1).reshape(1, 1, *W_SHAPE), dtype=torch.float32)

    t0 = time.time(); load_kde(bundle); t_kde = time.time() - t0
    t0 = time.time(); atm = ATMBundle(bundle, norm, device=device); t_model = time.time() - t0
    atm.freeze_unet()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time(); a = atm.encode_anatomy(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t_enc = time.time() - t0
    vram_enc = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")

    # D2: upstream 실제 동작(train mode, Dropout3d 활성) 의 비결정성
    atm.net.unet.train()
    feats = torch.stack([atm.encode_anatomy(x)[0] for _ in range(3)])
    atm.net.unet.eval()
    nd_std = float(feats.std(0).mean())
    nd_dist = float((feats - a).norm(dim=1).mean())
    a_norm = float(a.norm())
    print(f"  anatomy feature {tuple(a.shape)}  |a|={a_norm:.2f}   encoder {t_enc:.2f}s  "
          f"(peak VRAM {vram_enc:.2f} GB)")
    print(f"  D2 upstream train-mode 비결정성: 3회 실행 feature 표준편차(평균) {nd_std:.4f}, "
          f"eval 결과와 거리(평균) {nd_dist:.3f}  (|a|={a_norm:.2f})")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time(); mm_t = atm.generate(n, a, seed=0)
    if device == "cuda":
        torch.cuda.synchronize()
    t_gen = time.time() - t0
    vram_gen = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    mm = mm_t.cpu().numpy()

    assert mm.shape == (n, 128, 3), mm.shape
    n_nan = int(np.isnan(mm).sum()); assert n_nan == 0, f"NaN {n_nan}"
    L = lengths_of(mm)
    lo, hi = mm.reshape(-1, 3).min(0), mm.reshape(-1, 3).max(0)
    print(f"  streamlines {mm.shape}  NaN {n_nan}")
    print(f"  mm bbox     [{lo[0]:.1f},{lo[1]:.1f},{lo[2]:.1f}] .. [{hi[0]:.1f},{hi[1]:.1f},{hi[2]:.1f}]")
    print(f"  length      mean {L.mean():.1f}  p5 {np.percentile(L,5):.1f}  p95 {np.percentile(L,95):.1f} mm")
    print(f"  time        KDE load {t_kde:.1f}s | model load {t_model:.1f}s | encoder {t_enc:.2f}s | "
          f"generate {t_gen:.2f}s -> {n/t_gen:,.0f} streamlines/s | peak VRAM gen {vram_gen:.2f} GB")

    trk = OUT / sub / f"{bundle}_inferred.trk"; tck = OUT / sub / f"{bundle}_inferred.tck"
    t0 = time.time(); save_trk_tck(mm, trk, tck); t_save = time.time() - t0
    print(f"  saved       {trk}  ({t_save:.1f}s)  + .tck")

    gtres = compare_with_gt(mm, sub, bundle) or {}
    return {"bundle": bundle, "n": n, "len_mean": float(L.mean()), "len_p5": float(np.percentile(L, 5)),
            "len_p95": float(np.percentile(L, 95)), "t_kde": t_kde, "t_model": t_model, "t_enc": t_enc,
            "t_gen": t_gen, "streamlines_per_s": n / t_gen, "vram_enc_gb": vram_enc, "vram_gen_gb": vram_gen,
            "d2_train_std": nd_std, "d2_train_dist": nd_dist, "a_norm": a_norm, **gtres}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="AF_L")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--sub", default="sub-1135")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device {args.device}  torch {torch.__version__}  upstream {UPSTREAM}")
    raw = load_example_t1(args.sub)
    print(f"예제 T1 {args.sub}: {raw.shape}  intensity [{raw.min():.0f}, {raw.max():.0f}]")

    rows = [run_bundle(b, args.sub, args.n, raw, args.device)
            for b in (BUNDLES if args.all else [args.bundle])]
    (OUT / args.sub).mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / args.sub / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nsummary -> {OUT / args.sub / 'summary.csv'}")


if __name__ == "__main__":
    main()
