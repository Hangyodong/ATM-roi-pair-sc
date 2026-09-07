"""티어 1 명시적 해부 feature — 학습 없이, subject 볼륨에서 직접 계산한다.

왜 필요한가
-----------
지금까지 이 프로젝트에서 유의했던 유일한 개인 신호는 **학습된 feature 가 아니라 머리 크기라는
명시적 스칼라 하나**였다 (S1-b r=0.102, W4-a 가 0.101 로 독립 재현). 분할(segmentation)로
학습된 upstream UNet 이 연결 세기를 알아서 배우길 기대하는 것보다, SC 를 실제로 결정한다고
알려진 해부 량을 직접 계산하는 편이 근거가 있다.

**중요 — 아틀라스는 모두에게 같다.** rigid 공간이라 ROI 마스크 voxel 집합이 subject 마다
동일하므로 "ROI voxel 개수" 같은 값은 **상수**다. 여기서 계산하는 값은 전부 subject 자신의
볼륨(T1, WM 확률)에서 나오고, 아틀라스는 공간 참조로만 쓴다. 상수가 되어버린 feature 는
`assert_subject_varying` 이 시끄럽게 잡는다.

per-ROI (i = 0..81)
  t1_sum, t1_mean, t1_std   ROI 안 T1 강도 (조직량·대비의 대리)
  wm_sum, wm_mean           Σ WM 확률 = **조직 가중 부피** (ROI 가 크면 가닥이 많다)
  shell_t1_mean             ROI 바깥 껍질의 T1 평균 -> 경계 대비 (수초화 대리)
  centroid (3)              **WM 가중** centroid, mm. rigid 공간이라 subject 마다 다르다
per-subject
  brain_vol, t1_total, wm_total, head(3)   전역 크기

pair feature 는 `pair_features()` 가 위 값에서 만든다 (거리·corridor). 캐시는 ROI 단위로만
저장한다 -- 3,321 쌍을 subject 마다 저장하면 디스크가 낭비다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import CACHE

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = CACHE / "anat_tier1"
N_ROI = 82


def _shell(mask: np.ndarray) -> np.ndarray:
    """3D 이진 마스크의 1-voxel 바깥 껍질 (6-이웃 팽창 - 원본)."""
    d = mask.copy()
    for ax in range(3):
        d |= np.roll(mask, 1, ax) | np.roll(mask, -1, ax)
    return d & ~mask


def build_subject(sub: str, labels: np.ndarray, aff: np.ndarray, source: str = "rigid") -> dict:
    """subject 하나의 ROI 단위 티어 1 feature. labels 는 W 격자의 아틀라스 [X,Y,Z] (라벨 1..82)."""
    t1 = np.load(CACHE / f"{sub}_T1w_{source}_W.npy").astype(np.float32)
    wm = np.load(CACHE / f"{sub}_WM_W.npy").astype(np.float32)
    assert t1.shape == wm.shape == labels.shape, (t1.shape, wm.shape, labels.shape)
    assert np.isfinite(t1).all() and np.isfinite(wm).all(), f"{sub}: T1/WM 에 NaN/Inf"
    assert t1.max() > 0 and wm.max() > 0, f"{sub}: T1 또는 WM 이 전부 0"
    assert 0.0 <= wm.min() and wm.max() <= 1.0 + 1e-5, f"{sub}: WM 이 확률이 아니다"

    # ── 강도 정규화 (필수) ─────────────────────────────────────────────────────
    # 캐시의 rigid T1 은 **raw 강도**다. 실측 40명: 뇌 내부 중앙값이 163 ~ 63,824 (390배,
    # subject 간 CV 1.635). 이건 스캐너/프로토콜 스케일이지 해부가 아니다. 정규화 없이
    # t1_mean 같은 값을 쓰면 ridge 가 해부 대신 **획득 조건**을 학습하고, 사이트 효과가
    # SC 와 상관되면 가짜 양성이 나온다. 뇌 내부 중앙값으로 나눠 상대 대비만 남긴다.
    # 스케일 자체는 버리지 않고 t1_scale 로 따로 남겨 교란변수로 검사할 수 있게 한다.
    brain = t1 > (0.05 * float(t1.max()))
    assert brain.sum() > 1e5, f"{sub}: 뇌 마스크가 {int(brain.sum())} voxel 뿐이다"
    scale = float(np.median(t1[brain]))
    assert scale > 0, f"{sub}: 뇌 내부 중앙 강도가 0"
    t1 = t1 / scale

    out = {k: np.zeros(N_ROI, np.float32) for k in
           ("t1_sum", "t1_mean", "t1_std", "wm_sum", "wm_mean", "shell_t1_mean", "n_vox")}
    cent = np.zeros((N_ROI, 3), np.float32)

    for i in range(N_ROI):
        m = labels == (i + 1)
        n = int(m.sum())
        assert n > 0, f"{sub}: ROI {i+1} 이 아틀라스에서 비었다"
        tv, wv = t1[m], wm[m]
        out["n_vox"][i] = n
        out["t1_sum"][i] = tv.sum()
        out["t1_mean"][i] = tv.mean()
        out["t1_std"][i] = tv.std()
        out["wm_sum"][i] = wv.sum()
        out["wm_mean"][i] = wv.mean()
        sh = _shell(m)
        out["shell_t1_mean"][i] = t1[sh].mean() if sh.any() else 0.0
        # WM 가중 centroid: 같은 마스크라도 subject 의 조직 분포에 따라 움직인다
        idx = np.stack(np.nonzero(m), 1).astype(np.float64)
        w = wv.astype(np.float64) + 1e-6
        c_vox = (idx * w[:, None]).sum(0) / w.sum()
        cent[i] = (aff[:3, :3] @ c_vox + aff[:3, 3]).astype(np.float32)

    return {**out, "centroid": cent,
            "brain_vol": np.float32(brain.sum()),
            "t1_total": np.float32(t1.sum()),
            "wm_total": np.float32(wm.sum()),
            "t1_scale": np.float32(scale),      # 교란변수 (스캐너 강도 스케일). feature 가 아니다
            "sub": np.array(sub)}


def pair_features(f: dict, pairs: np.ndarray) -> np.ndarray:
    """ROI 단위 feature -> [K, 9] pair feature. pairs [K,2] (0-기반, canonical i<j).

      0 dist_mm            centroid 거리. 문헌상 SC 의 최강 단일 예측자이고 rigid 공간이라 개인차가 있다
      1,2 wm_sum_i/j       조직 가중 ROI 부피 (log)
      3 wm_sum_geom        두 ROI 부피의 기하평균 (log) -- 가닥 수는 양끝 크기에 함께 좌우된다
      4,5 wm_mean_i/j      ROI 안 평균 WM 확률 (백질 비중)
      6,7 contrast_i/j     (ROI T1 평균 - 껍질 T1 평균) / ROI T1 평균. 경계 대비
      8 t1_ratio           두 ROI 의 T1 평균 비 (log)
    """
    assert pairs.ndim == 2 and pairs.shape[1] == 2, pairs.shape
    i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    c = f["centroid"]
    d = np.linalg.norm(c[i] - c[j], axis=1)
    ws, wm_, t1m, sh = f["wm_sum"], f["wm_mean"], f["t1_mean"], f["shell_t1_mean"]
    lg = lambda x: np.log1p(np.maximum(x, 0.0))
    con = (t1m - sh) / np.maximum(t1m, 1e-6)
    X = np.stack([d, lg(ws[i]), lg(ws[j]), 0.5 * (lg(ws[i]) + lg(ws[j])),
                  wm_[i], wm_[j], con[i], con[j],
                  np.log(np.maximum(t1m[i], 1e-6) / np.maximum(t1m[j], 1e-6))], 1)
    assert np.isfinite(X).all(), "pair feature 에 NaN/Inf"
    return X.astype(np.float32)


PAIR_DIM = 9


def load(sub: str, source: str = "rigid") -> dict:
    p = OUT_DIR / f"{sub}_{source}.npz"
    assert p.exists(), f"티어 1 feature 가 없다: {p} (scripts/55_anat_tier1.py 로 생성)"
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "sub"}


def assert_subject_varying(feats: list[dict], min_cv: float = 1e-3) -> dict:
    """아틀라스가 공유라 상수가 되어버린 feature 를 시끄럽게 잡는다."""
    rep = {}
    for k in ("t1_sum", "t1_mean", "t1_std", "wm_sum", "wm_mean", "shell_t1_mean", "n_vox"):
        V = np.stack([f[k] for f in feats])                    # [N, 82]
        cv = float(np.mean(V.std(0) / (np.abs(V).mean(0) + 1e-12)))
        rep[k] = cv
    C = np.stack([f["centroid"] for f in feats])
    rep["centroid_sd_mm"] = float(C.std(0).mean())
    for k, v in rep.items():
        if k == "n_vox":
            assert v < 1e-9, f"n_vox 가 subject 마다 다르다 ({v:.3e}) -- 아틀라스가 공유가 아니다"
            continue
        assert v > min_cv, f"{k} 가 사실상 상수다 (subject CV {v:.2e}) -- 개인 정보가 없다"
    return rep
