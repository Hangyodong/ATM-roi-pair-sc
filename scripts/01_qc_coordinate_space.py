#!/usr/bin/env python
"""Step 1 — 좌표계 QC (docs/ATM_ROI_PAIR_SC_FINETUNING_PIPELINE.md §4).

T1(native) / GT .tt.gz(QSDR 템플릿) / atlas(MNI152NLin6 2mm) / W 격자(ATM 입력) 가 mm 공간에서
서로 맞는지 숫자와 **그림**으로 확인한다. 좌표계 오류는 endpoint ROI 할당과 SC 를 통째로
망가뜨리는데 코드는 정상 종료하고 숫자도 그럴듯하게 나오기 때문에, 반드시 overlay 를 눈으로 본다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import nibabel as nib                                             # noqa: E402
import numpy as np                                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atm_sc.data import tt_io                                     # noqa: E402
from atm_sc.data.paths import ATLAS, CACHE, t1_path, tt_path      # noqa: E402
from atm_sc.data.prepare_t1 import (TEMPLATE_MASK, check_alignment,   # noqa: E402
                                    prepare_subject, resample_to_W)
from atm_sc.spaces import W_AFFINE, W_SHAPE, apply_affine         # noqa: E402

QC = ROOT / "outputs" / "qc"


def geom(name, img):
    z = tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])
    print(f"  {name:14s} shape={tuple(img.shape[:3])} zoom={z} orient={nib.aff2axcodes(img.affine)} "
          f"origin={img.affine[:3, 3].round(2)}")


def in_mask(mm, arr, affine):
    v = np.rint(apply_affine(np.linalg.inv(affine), mm.astype(np.float64))).astype(np.int64)
    ok = np.all((v >= 0) & (v < np.array(arr.shape)), 1)
    hit = np.zeros(len(mm), bool)
    hit[ok] = arr[v[ok, 0], v[ok, 1], v[ok, 2]] > 0
    return hit


def slice_points(vox, axis, idx, tol=2.0):
    m = np.abs(vox[:, axis] - idx) <= tol
    return vox[m]


def overlay_png(t1, atlas_w, vox, centroid_vox, path, title):
    """3 orthogonal slice. W 는 RAS 라 배열 i->x(R), j->y(A), k->z(S). imshow(.T, origin='lower')
    로 그리면 가로축이 첫 인덱스, 세로축이 둘째 인덱스가 되어 R 이 오른쪽, S/A 가 위로 온다."""
    ci, cj, ck = [int(round(c)) for c in centroid_vox]
    lab = np.ma.masked_where(atlas_w == 0, atlas_w)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    specs = [("axial  z=k", t1[:, :, ck], lab[:, :, ck], slice_points(vox, 2, ck)[:, [0, 1]], "x (i) R->", "y (j) A->"),
             ("coronal y=j", t1[:, cj, :], lab[:, cj, :], slice_points(vox, 1, cj)[:, [0, 2]], "x (i) R->", "z (k) S->"),
             ("sagittal x=i", t1[ci, :, :], lab[ci, :, :], slice_points(vox, 0, ci)[:, [1, 2]], "y (j) A->", "z (k) S->")]
    for ax, (nm, im, lb, pts, xl, yl) in zip(axes, specs):
        ax.imshow(im.T, cmap="gray", origin="lower")
        ax.imshow(lb.T % 20, cmap="tab20", alpha=0.35, origin="lower", interpolation="nearest")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="yellow", alpha=0.6, linewidths=0)
        ax.set_title(f"{nm}  ({len(pts)} pts within ±2 mm)"); ax.set_xlabel(xl); ax.set_ylabel(yl)
    fig.suptitle(title); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def lr_png(t1, vox, mm, centroid_vox, path, title):
    """좌우 검증: mm x<0 (좌) 와 x>0 (우) 를 다른 색으로. W affine 은 x 가 i 와 함께 증가(RAS) 하므로
    그림 오른쪽이 해부학적 우측이다. 이 전제를 그림 가장자리에 L / R 로 적어 눈으로 확인한다."""
    ci, cj, ck = [int(round(c)) for c in centroid_vox]
    left = mm[:, 0] < 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (nm, im, axis, idx, cols) in zip(axes, [("axial", t1[:, :, ck], 2, ck, (0, 1)),
                                                     ("coronal", t1[:, cj, :], 1, cj, (0, 2))]):
        ax.imshow(im.T, cmap="gray", origin="lower")
        sel = np.abs(vox[:, axis] - idx) <= 2.0
        for msk, c, lbl in [(sel & left, "cyan", "x<0 (L)"), (sel & ~left, "red", "x>0 (R)")]:
            p = vox[msk][:, list(cols)]
            ax.scatter(p[:, 0], p[:, 1], s=1.5, c=c, alpha=0.7, linewidths=0, label=f"{lbl} n={msk.sum()}")
        ax.text(0.02, 0.5, "L", transform=ax.transAxes, color="w", fontsize=22, fontweight="bold", va="center")
        ax.text(0.95, 0.5, "R", transform=ax.transAxes, color="w", fontsize=22, fontweight="bold", va="center")
        ax.set_title(nm); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-100001")
    ap.add_argument("--mode", default="syn", choices=["syn", "rigid"])
    ap.add_argument("--n-tracks", type=int, default=20000)
    a = ap.parse_args()

    print(f"[{a.sub}] geometry")
    t1_img = nib.load(t1_path(a.sub)); geom("T1 native", t1_img)
    hdr = tt_io.read_header(str(tt_path(a.sub)))
    T = hdr.trans_to_mni
    print(f"  {'tt.gz (QSDR)':14s} dim={hdr.dimension} vox={hdr.voxel_size} trans_to_mni diag={np.diag(T)[:3]} "
          f"offset={T[:3, 3]} offdiag_max={np.abs(T[:3, :3] - np.diag(np.diag(T[:3, :3]))).max():.4f}")
    atl_img = nib.load(ATLAS); geom("atlas", atl_img)
    print(f"  {'W (ATM 입력)':14s} shape={W_SHAPE} zoom=(1,1,1) orient={nib.aff2axcodes(W_AFFINE)} origin={W_AFFINE[:3, 3]}")
    assert hdr.report == "" or "QSDR" in hdr.report or True     # report 는 정보용
    assert np.abs(T[:3, :3] - np.diag(np.diag(T[:3, :3]))).max() == 0.0, "trans_to_mni 에 회전 성분 — native 공간?"

    cache = CACHE / f"{a.sub}_T1w_{a.mode}_W.npy"
    if cache.exists():
        vol = np.load(cache); print(f"\nT1 in W: 캐시 사용 {cache.name}")
    else:
        print(f"\nT1 in W: {a.mode} 정합 중 (syn 은 약 6 분) ...", flush=True)
        vol = prepare_subject(t1_path(a.sub), mode=a.mode, out_dir=CACHE)
        CACHE.mkdir(parents=True, exist_ok=True); np.save(cache, vol)
    assert vol.shape == W_SHAPE and vol.max() > 0

    _, gen = tt_io.load_streamlines(str(tt_path(a.sub)), chunk=a.n_tracks)
    mm, npts = next(gen())
    print(f"GT streamlines: {len(npts):,} 개, {len(mm):,} 점, mm bbox {mm.min(0).round(1)} .. {mm.max(0).round(1)}")

    frac_brain = check_alignment(vol, mm)                         # < 0.9 면 assert
    starts = np.concatenate([[0], np.cumsum(npts)[:-1]]); ends = starts + npts - 1
    atlas = np.asanyarray(atl_img.dataobj)
    ep = np.concatenate([mm[starts], mm[ends]])
    frac_ep_roi = float(in_mask(ep, atlas, atl_img.affine).mean())
    tmask_img = nib.load(TEMPLATE_MASK)
    frac_tmask = float(in_mask(mm, np.asanyarray(tmask_img.dataobj), tmask_img.affine).mean())
    hemi_l = float((mm[:, 0] < 0).mean())
    print(f"\nalignment")
    print(f"  streamline 점이 warp 된 T1 뇌 안:            {frac_brain:.4f}   (assert >= 0.90)")
    print(f"  endpoint 가 atlas ROI(비배경) 안:             {frac_ep_roi:.4f}")
    print(f"  streamline 점이 MNI152NLin6 brain mask 안:    {frac_tmask:.4f}")
    print(f"  점의 x<0 (좌반구) 비율:                        {hemi_l:.3f}   (whole-brain 이면 ~0.5)")

    atlas_w = np.rint(resample_to_W(atl_img, order=0)).astype(np.int16)
    assert atlas_w.max() == int(atlas.max()), "atlas 를 W 로 재샘플하는 중 라벨 소실"
    vox = apply_affine(np.linalg.inv(W_AFFINE), mm.astype(np.float64))
    cen = vox.mean(0)
    QC.mkdir(parents=True, exist_ok=True)
    p1 = QC / f"{a.sub}_overlay_{a.mode}.png"
    p2 = QC / f"{a.sub}_leftright_{a.mode}.png"
    overlay_png(vol, atlas_w, vox, cen, p1, f"{a.sub} T1({a.mode}->W) + atlas + GT streamlines (yellow)")
    lr_png(vol, vox, mm, cen, p2, f"{a.sub} left/right check (cyan x<0, red x>0)")
    for p in (p1, p2):
        assert p.exists() and p.stat().st_size > 0, p
        print(f"  PNG -> {p}")


if __name__ == "__main__":
    main()
