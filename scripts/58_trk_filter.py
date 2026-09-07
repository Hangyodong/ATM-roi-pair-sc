#!/usr/bin/env python
"""B3 후처리 필터 — 해부학적으로 잘못된 가닥을 거른다 (전략 문서 §7).

  python scripts/58_trk_filter.py --stage calibrate    # GT 로 임계값 보정 (CPU)
  python scripts/58_trk_filter.py --stage apply --ckpt ...   # 생성물에 적용 (GPU 필요)

왜 필요한가: 생성 tractogram 의 overreach 가 **1.407** 인데 GT 는 **0.212** 다. 상류 ATM 은
`tckedit` GM/WM 마스크 + `minlength 20` + MATLAB trimming 을 돌리는데 이 프로젝트는 아무것도
안 한다. MRtrix3 3.0.8 이 설치돼 있다.

**우리 목적에 맞춘 차이점**: 상류는 템플릿 마스크를 쓰지만 우리는 subject 자신의 WM 확률맵
(`outputs/cache/{sub}_WM_W.npy`, 206명)이 있다. 같은 필터를 **개인 마스크**로 돌리면 후처리
자체가 개인화 통로가 된다.

**calibrate 단계가 먼저인 이유**: GT 가닥이 이 기준을 얼마나 통과하는지 모르고 임계값을 정하면
진짜 연결까지 지운다. 문서 §7.7 -- retained rate 가 급락하는데 지표만 좋아지는 건 개선이 아니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402
from atm_sc.data.paths import CACHE                                     # noqa: E402
from atm_sc.spaces import W_AFFINE, W_SHAPE                             # noqa: E402

EVAL = ROOT / "outputs/eval"


def mm_to_vox(mm: np.ndarray) -> np.ndarray:
    """mm [.., 3] -> W 격자 voxel 인덱스 (nearest). 범위 밖은 -1 로 표시한다."""
    inv = np.linalg.inv(W_AFFINE)
    v = mm @ inv[:3, :3].T + inv[:3, 3]
    ijk = np.rint(v).astype(np.int64)
    bad = ((ijk < 0) | (ijk >= np.asarray(W_SHAPE))).any(-1)
    ijk[bad] = -1
    return ijk, bad


def streamline_scores(S: np.ndarray, wm: np.ndarray, brain: np.ndarray) -> dict:
    """[n,128,3] mm -> 가닥별 점수. 문서 §7.3 의 soft score 를 구성하는 성분들."""
    ijk, out_of_grid = mm_to_vox(S)
    n, t, _ = S.shape
    flat = ijk.reshape(-1, 3)
    ok = flat[:, 0] >= 0
    wmv = np.zeros(flat.shape[0], np.float32)
    brv = np.zeros(flat.shape[0], bool)
    wmv[ok] = wm[flat[ok, 0], flat[ok, 1], flat[ok, 2]]
    brv[ok] = brain[flat[ok, 0], flat[ok, 1], flat[ok, 2]]
    wmv = wmv.reshape(n, t); brv = brv.reshape(n, t)
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    return {
        "wm_occupancy": wmv.mean(1),                       # 백질 통과 비율
        "wm_mid": wmv[:, 16:112].mean(1),                  # 중간 구간은 더 엄격히 봐야 한다
        "outside_brain": 1.0 - brv.mean(1),                # 뇌 밖 비율
        "length_mm": seg.sum(1),
        "max_step_mm": seg.max(1),                         # 급격한 점프 = 깨진 가닥
        "endpoint_wm": 0.5 * (wmv[:, 0] + wmv[:, -1]),     # 끝점은 GM 이라 WM 이 낮아야 정상
    }


def stage_calibrate(a):
    subs = [l.strip() for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip()][:a.n_subj]
    rows = []
    for s in subs:
        wm = np.load(CACHE / f"{s}_WM_W.npy").astype(np.float32)
        t1 = np.load(CACHE / f"{s}_T1w_rigid_W.npy").astype(np.float32)
        brain = t1 > 0.05 * float(t1.max())
        assert wm.shape == W_SHAPE and brain.sum() > 1e5, (wm.shape, int(brain.sum()))
        z = np.load(ROIPairSubject(s).dir / "bundles.npz")
        St = z["streamlines"]
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(St), min(a.n_stream, len(St)), replace=False))
        sc = streamline_scores(St[sel].astype(np.float64), wm, brain)
        # 대조군: 뇌 안 무작위 점의 WM 값. 가닥이 이보다 확실히 높아야 마스크가 정렬된 것이다.
        idx = np.argwhere(brain)
        rp = idx[rng.choice(len(idx), 100_000, replace=False)]
        sc["_rand_wm"] = wm[rp[:, 0], rp[:, 1], rp[:, 2]]
        rows.append(sc)
        print(f"  {s}: {len(sel)} 가닥", flush=True)
    agg = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    rand_wm = agg.pop("_rand_wm")
    rep = {"_random_wm": float(rand_wm.mean())}
    for k, v in agg.items():
        rep[k] = {q: float(np.percentile(v, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    n = len(agg["length_mm"])
    rep["n_streamlines"] = int(n)
    rep["n_subjects"] = len(subs)
    # 문서 §7.7: 각 기준 단독으로 GT 를 얼마나 지우는가 (retained rate)
    rep["gt_retained"] = {
        "wm_occupancy>0.5": float((agg["wm_occupancy"] > 0.5).mean()),
        "wm_mid>0.5": float((agg["wm_mid"] > 0.5).mean()),
        "wm_mid>0.7": float((agg["wm_mid"] > 0.7).mean()),
        "outside_brain<0.02": float((agg["outside_brain"] < 0.02).mean()),
        "length>=20mm": float((agg["length_mm"] >= 20).mean()),
        "length>=30mm": float((agg["length_mm"] >= 30).mean()),
        "max_step<=5mm": float((agg["max_step_mm"] <= 5).mean()),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "trk_filter_calibration.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    # ── 임계값을 여기서 고정하지 않는다 (실측 근거) ──────────────────────────────
    # 처음엔 "GT 를 95% 이상 남기는 임계값"을 쓰려 했는데 GT 분위가 그걸 무의미하게 만든다:
    #   wm_mid p5 = 0.009, wm_occupancy p5 = 0.031
    # 즉 **GT 가닥의 5% 는 백질을 거의 안 지난다**. 100만 가닥 결정론적 tractography 의
    # 산물이라 당연하고, GT 의 쓰레기까지 재현할 이유가 없다. 그래서 문서 §7.3 대로
    # 임계값을 **스윕**하고 GT/생성물 잔존율을 나란히 봐서 operating point 를 고른다.
    # 필터의 가치는 "생성물을 GT 보다 훨씬 많이 지우는가" 에 있다.
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    rep["sweep_wm_mid"] = {f"{g:.1f}": float((agg["wm_mid"] > g).mean()) for g in grid}
    rep["sweep_length"] = {f"{g}": float((agg["length_mm"] >= g).mean()) for g in (0, 20, 25, 30, 40)}
    print("\nwm_mid 임계값별 GT 잔존율 (생성물과 나란히 봐야 의미가 있다):", flush=True)
    print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in rep["sweep_wm_mid"].items()), flush=True)
    # 마스크가 정렬돼 있고 정보가 있는지 -- 이게 아니면 필터 자체가 무의미하다
    print("\n[검증] 가닥 위 WM 평균 %.3f vs 뇌 안 무작위 %.3f (2배 이상이어야 정렬된 것)"
          % (float(agg["wm_occupancy"].mean()), rep.get("_random_wm", float("nan"))), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="calibrate", choices=("calibrate",))
    ap.add_argument("--n-subj", type=int, default=6)
    ap.add_argument("--n-stream", type=int, default=20000)
    a = ap.parse_args()
    stage_calibrate(a)


if __name__ == "__main__":
    main()
