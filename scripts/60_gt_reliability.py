#!/usr/bin/env python
"""GT SC 잔차의 **분할반분 신뢰도** — resid_r 의 이론적 상한을 잰다 (CPU).

왜 필요한가
-----------
"31명이 눈에 띄게 다른 SC"를 정직하게 얻으려면 resid_r 이 얼마나 필요한지 시뮬레이션했더니
0.3 이었다. 그런데 **그 값에 도달 가능한지는 GT 잔차가 얼마나 재현 가능한 신호인가**로 정해진다.

  GT 잔차 d = s(진짜 개인 신호) + n(잡음),  신뢰도 rho = Var(s)/Var(d)
  완벽한 모델이라도  max resid_r = sqrt(rho),  그때 var_ratio 도 sqrt(rho) 다.

test-retest 는 못 잰다(재스캔 2명, tract 미생성). 대신 **한 subject 의 streamline 을 반으로
나눠 SC 를 두 개 만들고** 그 잔차끼리의 상관을 재면 **tractography 표본 잡음** 성분의 신뢰도가
나온다. 정합/스캔 변동은 안 잡히므로 이 값은 **상한의 상한**이다 -- 실제 rho 는 이보다 낮다.

주의: 반쪽 SC 는 가닥 수가 절반이라 count 잡음이 커진다. Spearman-Brown 으로 전체 길이
기준으로 보정한 값도 같이 낸다 (rho_full = 2*rho_half / (1 + rho_half)).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402

EVAL = ROOT / "outputs/eval"
N_ROI = 82


def sc_from_streamlines(pair_ids, pair_offsets, idx_keep, n_roi=N_ROI) -> np.ndarray:
    """선택된 가닥 인덱스로 pass-SC 를 다시 센다. bundles.npz 의 pair 단위 구조를 그대로 쓴다."""
    sc = np.zeros((n_roi, n_roi), np.float64)
    for k in range(len(pair_ids)):
        lo, hi = int(pair_offsets[k]), int(pair_offsets[k + 1])
        c = int(idx_keep[lo:hi].sum())
        if c:
            i, j = int(pair_ids[k][0]), int(pair_ids[k][1])
            sc[i, j] += c; sc[j, i] += c
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=31)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    subs = [l.strip() for l in (ROOT / f"outputs/splits/{a.split}.txt").read_text().splitlines() if l.strip()][:a.n_subj]
    iu = np.triu_indices(N_ROI, 1)
    A, B = [], []
    rng = np.random.default_rng(a.seed)
    for s in subs:
        z = np.load(ROIPairSubject(s).dir / "bundles.npz")
        pid, poff = z["pair_ids"], z["pair_offsets"]
        n = int(poff[-1])
        assert n > 1000, (s, n)
        m = rng.random(n) < 0.5                      # 가닥을 무작위 반으로
        A.append(np.log1p(sc_from_streamlines(pid, poff, m)[iu]))
        B.append(np.log1p(sc_from_streamlines(pid, poff, ~m)[iu]))
        print(f"  {s}: {n} 가닥 -> {int(m.sum())}/{int((~m).sum())}", flush=True)
    A, B = np.stack(A), np.stack(B)
    assert np.isfinite(A).all() and np.isfinite(B).all()

    def resid(X):
        d = X - X.mean(0, keepdims=True)              # subject 평균(=이 표본의 템플릿) 제거
        return d

    dA, dB = resid(A), resid(B)
    # 잔차의 반쪽-반쪽 상관 = tractography 표본 잡음 기준 신뢰도
    def rowr(P, Q):
        p = P - P.mean(1, keepdims=True); q = Q - Q.mean(1, keepdims=True)
        return (p * q).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12)
    r_half = float(np.mean(rowr(dA, dB)))
    rho_full = 2 * r_half / (1 + r_half)              # Spearman-Brown (전체 길이 기준으로 보정)
    # 원래 SC(잔차 아님)의 반쪽 상관도 참고로
    r_raw = float(np.mean(rowr(A, B)))
    out = {"n_subjects": len(subs), "split": a.split,
           "resid_split_half_r": r_half,
           "resid_reliability_spearman_brown": rho_full,
           "max_resid_r_sampling_noise_only": float(np.sqrt(max(rho_full, 0.0))),
           "raw_sc_split_half_r": r_raw,
           "note": "정합/스캔 변동은 포함되지 않는다 -> 실제 상한은 이보다 낮다"}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "gt_reliability.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
