#!/usr/bin/env python
"""S1-a: ROI 풀링 유틸 + 아틀라스 정렬 검증 (CPU 전용, GPU 금지).

  python scripts/41_check_roi_pool.py --n 4

검사 항목 (전부 assert, 실패하면 시끄럽게 죽는다):
  1  feature 격자 affine 유도 — conv 산술을 autograd receptive field 로 **실측** 확인
  2  아틀라스 -> feature 격자 nearest 재표본화, ROI 82개 voxel 개수 전부
  3  좌우 검사 (L_*/R_* 무게중심 x 부호). 1mm/2mm 사이에서 뒤집힌 적이 있다
  4  f[82,256] 풀링 결과의 shape/min/max/mean/NaN
  5  subject 간 상관: stage3 공간맵 / ROI 풀링 / 전뇌 풀링 / 최종 anatomy(=0.9996 재현)
  6  조건 (b)(rigid feature -> 템플릿 warp) 에 필요한 ANTs 변환이 디스크에 있는지

결과: outputs/eval/s1a_atlas_pool_check.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # GPU 금지 (다른 작업이 쓴다)
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import ATLAS, CACHE                                   # noqa: E402
from atm_sc.models.roi_pool import (atlas_on_feature_grid, check_lateralization,  # noqa: E402
                                    feature_grid, global_pool, roi_names, roi_pool,
                                    roi_voxel_counts)
from atm_sc.spaces import W_AFFINE, W_SHAPE                                  # noqa: E402

CMD = "python scripts/41_check_roi_pool.py --n {n}"


def mean_off(M) -> float:
    j = np.triu_indices(len(M), 1)
    return float(np.asarray(M)[j].mean())


def corr_rows(X: np.ndarray) -> float:
    """행(subject) 끼리 Pearson 상관의 off-diagonal 평균."""
    X = np.asarray(X, np.float64).reshape(len(X), -1)
    return mean_off(np.corrcoef(X))


# --- 1. 격자 affine 실측 -----------------------------------------------------
def measure_receptive_field(in_shape=W_SHAPE):
    """stage1~3 과 같은 k/s/p 로 1채널 망을 만들어 출력 voxel 의 receptive field 중심을 잰다.

    가중치는 전부 양수(1/27)라 gradient 가 상쇄되지 않는다. 즉 grad>0 인 입력 voxel 집합이
    정확히 receptive field 이고, 그 무게중심이 대응 입력 index 다.
    기하는 가중치와 무관하므로 실제 pretrained 가중치를 쓸 필요가 없다.
    """
    def c(stride):
        m = torch.nn.Conv3d(1, 1, 3, stride=stride, padding=1, bias=False)
        torch.nn.init.constant_(m.weight, 1.0 / 27.0)
        return m

    c11, c12, c13 = c(1), c(1), c(1)
    c21, c22, c23 = c(2), c(1), c(1)
    c31, c32, c33 = c(2), c(1), c(1)

    x = torch.zeros(1, 1, *in_shape, requires_grad=True)
    a11 = c11(x); o1 = a11 + c13(c12(a11))                     # stage1 (s=1, 크기 불변)
    a21 = c21(o1); o2 = a21 + c23(c22(a21))                    # stage2 (첫 conv s=2)
    a31 = c31(o2); o3 = a31 + c33(c32(a31))                    # stage3 (첫 conv s=2)
    shape = tuple(o3.shape[2:])

    probes = [(12, 14, 12), (24, 29, 24), (36, 43, 36), (48, 57, 48), (0, 0, 0)]
    got = []
    for p in probes:
        if x.grad is not None:
            x.grad = None
        o3[0, 0, p[0], p[1], p[2]].backward(retain_graph=True)
        g = x.grad[0, 0].detach().numpy()
        idx = np.argwhere(g > 0).astype(np.float64)
        w = g[g > 0]
        centre = (idx * w[:, None]).sum(0) / w.sum()
        lo, hi = idx.min(0).astype(int), idx.max(0).astype(int)
        got.append({"feat_idx": list(p), "rf_centre_in_idx": [round(float(v), 4) for v in centre],
                    "expected_4x": [4 * v for v in p], "rf_lo": lo.tolist(), "rf_hi": hi.tolist(),
                    "rf_size": (hi - lo + 1).tolist(),
                    "clipped": bool(((lo == 0) | (hi == np.array(in_shape) - 1)).any())})
        del g, idx, w
    return shape, got


def check_receptive_field(rf, in_shape=W_SHAPE):
    """중심 = 4j 인가. 볼륨 가장자리는 padding 으로 receptive field 가 잘리므로
    무게중심이 안쪽으로 밀린다 -- 잘린 probe 는 support 구간 [4j-h, 4j+h] 의
    clip 결과로 검사한다 (중심 대신 경계를 본다)."""
    interior = [r for r in rf if not r["clipped"]]
    assert interior, "잘리지 않은 probe 가 없다"
    half = (np.array(interior[0]["rf_size"]) - 1) // 2
    assert (half == half[0]).all(), interior[0]
    n = np.asarray(in_shape) - 1
    for r in rf:
        e = 4 * np.asarray(r["feat_idx"])
        assert np.array_equal(r["rf_lo"], np.clip(e - half, 0, n)), r
        assert np.array_equal(r["rf_hi"], np.clip(e + half, 0, n)), r
        if not r["clipped"]:
            assert np.allclose(r["rf_centre_in_idx"], e, atol=1e-3), r
    return {"half_width_in_voxels": int(half[0]), "n_probes": len(rf),
            "n_interior_probes_centre_exact": len(interior)}


# --- 2~5. 모델 경로 ----------------------------------------------------------
def stage3_maps(ckpt: Path, subs, labels):
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.training.run import t1_input

    m, sd = from_checkpoint(ckpt, device="cpu")
    src = sd.get("t1_source", "rigid")
    u = m.atm.net.unet
    out = {"ckpt": ckpt.name, "unet_level": sd.get("unet_level"),
           "in_channels": int(sd.get("in_channels", 1)), "t1_source": src, "device": "cpu"}

    O3, ROI, GLB, ANAT, per = [], [], [], [], []
    with torch.no_grad():
        for s in subs:
            t0 = time.time()
            x = t1_input(m, s, src)
            assert tuple(x.shape) == (1, m.in_channels) + W_SHAPE, x.shape
            o3 = m.atm._unet_stage3(u, x)                         # [1,256,49,58,49]
            assert torch.isfinite(o3).all(), f"{s}: stage3 에 NaN/Inf"
            f = roi_pool(o3, labels)                              # [82,256]
            g = global_pool(o3)                                   # [256]
            c41 = F.relu(u.conv4_1(o3))
            o4 = c41 + F.relu(u.conv4_3(u.dropout4(F.relu(u.conv4_2(c41)))))
            a = u.fc(u.global_avg_pool(o4).view(1, -1))           # 현재 경로 [1,512]
            fn = f.numpy()
            per.append({"sub": s, "sec": round(time.time() - t0, 1),
                        "f_shape": list(f.shape),
                        "f_min": round(float(fn.min()), 6), "f_max": round(float(fn.max()), 6),
                        "f_mean": round(float(fn.mean()), 6), "f_std": round(float(fn.std()), 6),
                        "f_nan": int(np.isnan(fn).sum()), "f_inf": int(np.isinf(fn).sum()),
                        "f_zero_rows": int((np.abs(fn).sum(1) == 0).sum()),
                        "stage3_absmax": round(float(o3.abs().max()), 6),
                        "anat_norm": round(float(a.norm()), 6)})
            O3.append(o3.numpy().ravel().astype(np.float32))
            ROI.append(fn.ravel().copy())
            GLB.append(g.numpy().copy())
            ANAT.append(a.numpy().ravel().copy())
            del x, o3, c41, o4, a, f, g
    return out, per, np.stack(O3), np.stack(ROI), np.stack(GLB), np.stack(ANAT)


# --- 6. 조건 (b) 가능성 -------------------------------------------------------
def check_transforms(subs):
    """(b) rigid feature -> 템플릿 warp 에 필요한 subject 별 변환이 디스크에 있는가."""
    pats = ("*0GenericAffine.mat", "*1Warp.nii.gz", "*1InverseWarp.nii.gz", "*Composite.h5")
    found = [str(p.relative_to(ROOT)) for pat in pats for p in (ROOT / "outputs").rglob(pat)]
    cache = {"rigid_W": sum((CACHE / f"{s}_T1w_rigid_W.npy").exists() for s in subs),
             "syn_W": sum((CACHE / f"{s}_T1w_syn_W.npy").exists() for s in subs),
             "torigid_nii": sum((CACHE / f"{s}_T1w__torigid.nii.gz").exists() for s in subs),
             "tosyn_nii": sum((CACHE / f"{s}_T1w__tosyn.nii.gz").exists() for s in subs)}
    return {
        "ants_transform_files_under_outputs": found,
        "n_transform_files": len(found),
        "cache_present_for_probe_subjects": cache,
        "feasible_now": len(found) > 0,
        "reason": ("src/atm_sc/data/prepare_t1.py:register_to_template 은 ants.registration 의 "
                   "reg['warpedmovout'] 만 저장하고 reg['fwdtransforms'] 를 버린다. "
                   "scripts/37_build_rigid_t1.py 의 docstring 도 '워프 필드는 저장되지 않았다' 고 명시. "
                   "따라서 rigid -> 템플릿 비선형 변환은 디스크에 없다."),
        "recompute": {"how": ("outputs/cache/{sub}_T1w__torigid.nii.gz (206명 전부 존재, 이미 템플릿 "
                              "격자) 를 fixed=templates/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz 에 "
                              "SyN 정합하고 reg['fwdtransforms'] 를 저장하면 rigid->템플릿 잔여 "
                              "비선형 변환을 얻는다. rigid 는 이미 반영돼 있으므로 추가 단계 없음."),
                      "cost_per_subject_sec": 130,
                      "basis": "outputs/preprocess_logs/summary.csv step 01(SyN) 중앙값 129.5s, n=204",
                      "n_subjects": 206, "total_core_hours": round(206 * 130 / 3600, 1),
                      "note": "CPU 코어 1개 환경 -> 직렬 7.4h. 병렬 6워커면 ~1.2h."},
    }


def main(a):
    subs = [s.strip() for s in open(a.split) if s.strip()][: a.n]
    assert len(subs) >= 2, f"subject 2명 이상 필요 (now {len(subs)})"
    torch.manual_seed(0)

    # 1 -----------------------------------------------------------------
    shape_th, rf = measure_receptive_field()
    shape, aff = feature_grid()
    assert shape_th == shape, (shape_th, shape)
    rf_summary = check_receptive_field(rf)
    grid = {
        "unet_in_shape": list(W_SHAPE), "unet_in_affine": W_AFFINE.tolist(),
        "stage_strides": {"stage1": 1, "stage2": 2, "stage3": 2, "total": 4},
        "conv_source": "stable/stable/model/model.py:148-166 (rigid_UNet, conv2_1/conv3_1 stride=2, k=3, p=1)",
        "derivation": ("k=3,s=2,p=1 은 출력 j 가 입력 [2j-1,2j,2j+1] 을 보므로 중심이 2j. "
                       "두 번 거치면 feature index k <-> UNet 입력 index 4k. "
                       "k=0..48 -> 0..192 로 193 축을 정확히 덮는다 ((193-1)/4+1 = 49). "
                       "=> feature affine = 입력 affine 의 3x3 x 4, translation 불변."),
        "feature_shape": list(shape), "feature_affine": aff.tolist(),
        "torch_measured_shape": list(shape_th),
        "receptive_field_check": rf_summary,
        "receptive_field_probes": rf,
    }

    # 2 -----------------------------------------------------------------
    labels, aff2 = atlas_on_feature_grid(ATLAS, shape, return_affine=True)
    assert np.allclose(aff2, aff)
    names = roi_names()
    cnt = roi_voxel_counts(labels)
    import nibabel as nib
    atl = nib.load(str(ATLAS))
    cnt2 = roi_voxel_counts(np.rint(np.asanyarray(atl.dataobj)).astype(np.int16))
    atlas = {
        "atlas": ATLAS.name, "atlas_shape": list(atl.shape), "atlas_affine": atl.affine.tolist(),
        "resample": "nearest neighbour (mm 경유). 라벨 데이터라 선형보간 금지",
        "n_roi_present": int((cnt > 0).sum()), "n_roi_missing": int((cnt == 0).sum()),
        "labeled_voxels_feature_grid": int((labels > 0).sum()),
        "labeled_voxels_atlas_2mm": int(cnt2.sum()),
        "roi_voxel_counts": {str(i + 1): int(cnt[i]) for i in range(len(cnt))},
        "roi_voxel_counts_atlas_2mm": {str(i + 1): int(cnt2[i]) for i in range(len(cnt2))},
        "smallest_rois": [{"roi": int(i + 1), "name": names[i], "feat_vox": int(cnt[i]),
                           "atlas2mm_vox": int(cnt2[i])} for i in np.argsort(cnt)[:8]],
        "fragile_rois_lt5_voxels": [{"roi": int(i + 1), "name": names[i], "feat_vox": int(cnt[i])}
                                    for i in np.flatnonzero(cnt < 5)],
        "fragile_note": ("stage3 격자는 4mm 라 PD25 소핵이 1~2 voxel 로 줄어든다. 통과 기준(>=1)은 "
                         "만족하지만 그 ROI 의 f 는 사실상 단일 voxel 값이다. S1-b 에서 "
                         "해당 ROI 결과를 따로 봐야 하고, 필요하면 stage2(2mm, 97x115x97) 로 "
                         "풀링하거나 아틀라스 부피 비율 soft 가중을 쓴다."),
    }
    assert atlas["n_roi_missing"] == 0

    # 3 -----------------------------------------------------------------
    lat = check_lateralization(labels, aff, names)
    lat["note"] = ("W 격자 x 는 +1mm/voxel, 아틀라스 x 는 -2mm/voxel 이라 voxel index 를 직접 "
                   "쓰면 좌우가 뒤집힌다. mm 경유 재표본화가 필수.")

    # 4~5 ---------------------------------------------------------------
    meta, per, O3, ROI, GLB, ANAT = stage3_maps(Path(a.ckpt), subs, labels)
    corr = {
        "n_subjects": len(subs), "subjects": subs,
        "stage3_spatial_map": round(corr_rows(O3), 6),
        "roi_pool_f82x256": round(corr_rows(ROI), 6),
        "global_pool_stage3_256": round(corr_rows(GLB), 6),
        "anatomy_conv4_gap_fc_512": round(corr_rows(ANAT), 6),
        "reference_findings_B6": {"stage3_spatial_map": 0.575, "global_avg_pool": 0.9996,
                                  "fc": 0.9996, "pred_sc": 0.99997, "gt_sc": 0.882},
    }
    corr["roi_pool_lower_than_global"] = bool(corr["roi_pool_f82x256"] < corr["global_pool_stage3_256"])
    corr["roi_pool_lower_than_0.9996"] = bool(corr["roi_pool_f82x256"] < 0.9996)
    # ROI 마다 개인차가 얼마나 살아 있는가 (ROI 별 subject 간 상관)
    R = ROI.reshape(len(subs), 82, -1)
    per_roi = np.array([corr_rows(R[:, i, :]) for i in range(82)])
    corr["per_roi_subject_corr"] = {"mean": round(float(per_roi.mean()), 6),
                                    "min": round(float(per_roi.min()), 6),
                                    "max": round(float(per_roi.max()), 6),
                                    "argmin_roi": int(per_roi.argmin() + 1),
                                    "argmin_name": names[int(per_roi.argmin())]}

    res = {"cmd": CMD.format(n=a.n), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": meta, "grid": grid, "atlas": atlas, "lateralization": lat,
           "pooled_features": per, "subject_correlation": corr,
           "condition_b_template_warp": check_transforms(subs),
           "pass": {"all_82_rois_have_voxels": atlas["n_roi_missing"] == 0,
                    "lateralization_ok": lat["n_bad"] == 0,
                    "no_nan_inf": all(p["f_nan"] == 0 and p["f_inf"] == 0 for p in per),
                    "features_nonzero": all(p["stage3_absmax"] > 0 for p in per)}}
    res["pass"]["all"] = all(res["pass"].values())

    out = ROOT / "outputs" / "eval" / "s1a_atlas_pool_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("pass", "subject_correlation")}, indent=1, ensure_ascii=False))
    print(f"ROI voxel: min {cnt.min()} (ROI {int(cnt.argmin()) + 1} {names[int(cnt.argmin())]}) "
          f"max {cnt.max()} / 82개 전부 >=1: {bool((cnt > 0).all())}")
    print(f"좌우: L 평균 x {lat['mean_x_left']} / R {lat['mean_x_right']} / 위반 {lat['n_bad']}")
    print(f"(b) 템플릿 warp 변환 디스크 존재: {res['condition_b_template_warp']['feasible_now']}")
    print(f"-> {out}")
    assert res["pass"]["all"], res["pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt"))
    ap.add_argument("--split", default=str(ROOT / "outputs/splits/test.txt"))
    ap.add_argument("--n", type=int, default=4, help="subject 수 (CPU 라 1명당 ~1분)")
    main(ap.parse_args())
