#!/usr/bin/env python
"""p2 관문(`resid_r >= 0.10`) 이 val 8명에서 통과한 것이 통계적으로 성립하는지 부트스트랩으로 잰다.

val 8명 0.1235 -> test 31명 -0.0107. 둘 중 하나는 표본 잡음이다. subject 단위 부트스트랩으로
n=8 에서 이 지표의 구간 폭이 얼마나 되는지, 그 폭에서 0.10 관문이 우연히 통과할 확률이
얼마인지 재고, 관문에 붙일 최소 표본 수(min_n) 를 그 수치로 정한다.

  python scripts/40_gate_ci.py            # -> outputs/eval/s0b_gate_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.evaluation.gates import bootstrap_ci  # noqa: E402

CMD = "python scripts/40_gate_ci.py"
FINAL = ROOT / "outputs" / "eval" / "final_p4_joint_step3000.json"
VAL = ROOT / "outputs" / "checkpoints" / "retrain" / "val_metrics.jsonl"
OUT = ROOT / "outputs" / "eval" / "s0b_gate_ci.json"
THR = 0.10                      # p2 관문
N_BOOT = 10000
SEED = 0


def subsample_means(x, n, n_boot=N_BOOT, seed=SEED):
    """n 명을 복원추출해 평균을 n_boot 번. n=8 짜리 추정량이 실제로 얼마나 흔들리는지."""
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, x.size, size=(n_boot, n))].mean(1)


def main():
    d = json.loads(FINAL.read_text())
    ps = d["per_subject"]
    x = np.asarray([p["residual_r"] for p in ps], float)
    assert x.size == d["summary"]["n_subjects"] == 31, x.size
    assert np.isfinite(x).all()

    vals = [json.loads(l) for l in VAL.read_text().splitlines() if l.strip()]
    p2 = next(v for v in vals if v["phase"] == "p2_count")
    val_hist = {v["phase"]: (v["resid_r"], v["n_indiv_subj"]) for v in vals}

    test_loo = d["summary"]["subject_specificity"]["resid_r"]      # LOO 중심화 (관문이 쓰는 정의)
    ci31 = bootstrap_ci(x, n_boot=N_BOOT, seed=SEED)

    # val 8명은 subject 별 값이 기록돼 있지 않다 (지금 이 문제의 한 원인이다). 그래서 test 의
    # subject 별 분포로 n=8 추정량의 흔들림을 재고, 그 폭을 val 점추정 0.1235 에 붙인다.
    m8 = subsample_means(x, 8)
    hw8 = float((np.quantile(m8, 0.975) - np.quantile(m8, 0.025)) / 2)
    v8, n8 = val_hist["p2_count"]
    assert n8 == 8, n8
    val_ci = [v8 - hw8, v8 + hw8]
    overlap = [max(val_ci[0], ci31["lo"]), min(val_ci[1], ci31["hi"])]
    overlaps = overlap[0] <= overlap[1]

    # min_n: 개인차가 실제로는 0 인 분포(test 실측)에서 관문 0.10 이 우연히 통과할 확률과
    # 95% 구간의 반폭을 n 별로 본다. 반폭 <= 0.05 (관문값의 절반) 이면 "0.10 이상" 이
    # "0 보다 확실히 크다" 를 함의한다.
    scan = []
    for n in list(range(3, 41)) + [50, 60]:
        m = subsample_means(x, n, seed=SEED + n)
        lo, hi = np.quantile(m, [0.025, 0.975])
        scan.append({"n": n, "half_width": float((hi - lo) / 2), "lo": float(lo), "hi": float(hi),
                     "p_false_pass_thr": float((m >= THR).mean()),
                     "p_at_least_val_point": float((m >= v8).mean())})
    need_hw = next(s["n"] for s in scan if s["half_width"] <= THR / 2)
    need_fp = next(s["n"] for s in scan if s["p_false_pass_thr"] <= 0.05)
    n_val_avail = sum(1 for l in (ROOT / "outputs/splits/val.txt").read_text().splitlines() if l.strip())
    min_n = min(max(need_hw, need_fp), n_val_avail)      # val split 이 상한이다

    # abl_gap (= own - zero). 게이트 지표로 같이 건다. subject 별 own/zero 상관이 기록돼 있지
    # 않아 부트스트랩 CI 를 만들 수 없다 -- 대신 관측된 부호를 그대로 남긴다.
    abl = {v["phase"]: {"abl_gap_own_minus_zero": v["abl_gap"], "own_minus_shuf": v["abl_own_r"] - v["abl_shuf_r"],
                        "abl_own_r": v["abl_own_r"], "abl_shuf_r": v["abl_shuf_r"], "abl_zero_r": v["abl_zero_r"],
                        "abl_ordered": v.get("abl_ordered")} for v in vals}
    for k, r in abl.items():
        assert abs(r["abl_gap_own_minus_zero"] - (r["abl_own_r"] - r["abl_zero_r"])) < 1e-12, k

    out = {
        "cmd": CMD,
        "purpose": "p2 관문 resid_r >= 0.10 의 val 8명 통과가 통계적으로 성립하는지 판정하고 min_n 을 정한다",
        "sources": {"per_subject": str(FINAL.relative_to(ROOT)), "val_metrics": str(VAL.relative_to(ROOT)),
                    "checkpoint": d["summary"]["checkpoint"], "split": d["summary"]["split"]},
        "caveat": ("subject 별로 남아 있는 잔차 상관은 템플릿 차감 버전(per_subject.residual_r)이고, "
                   "관문이 쓰는 resid_r 은 LOO 중심화 버전이다. test 31명에서 두 값은 "
                   f"{float(x.mean()):+.4f} vs {test_loo:+.4f} 로 사실상 같으므로 subject 단위 "
                   "잡음의 크기 추정에는 전자를 쓴다. val 8명은 subject 별 값이 기록돼 있지 "
                   "않아(현재 로깅의 구멍) 구간 폭을 test 분포에서 빌려 붙였다 -- 근사다."),
        "per_subject_residual_r": {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                                   "min": float(x.min()), "max": float(x.max()),
                                   "values": [round(float(t), 6) for t in x]},
        "test_n31": {"resid_r_loo": test_loo, "bootstrap": ci31},
        "val_n8": {"resid_r_loo": v8, "n": n8, "half_width_from_test_dist": hw8,
                   "ci_approx": val_ci, "note": "구간 폭은 test subject 분포에서 n=8 로 재표본해 얻었다"},
        "overlap": {"overlaps": bool(overlaps), "range": [float(overlap[0]), float(overlap[1])] if overlaps else None},
        "n8_null": {"p_mean8_ge_0.10": float((m8 >= THR).mean()),
                    "p_mean8_ge_val_point": float((m8 >= v8).mean()),
                    "note": "실제 개인차가 test 수준(평균 ~0)일 때 n=8 표본이 관문을 넘길 확률"},
        "val_history_resid_r": {k: {"resid_r": vv, "n": nn} for k, (vv, nn) in val_hist.items()},
        "abl_gap": {"definition": "abl_gap = abl_own_r - abl_zero_r (own-shuf 가 아니다)",
                    "per_phase": abl,
                    "finding": ("p0·p1 은 own/shuf/zero 가 16자리까지 같아 gap 이 정확히 0 이다 "
                                "(T1 입력이 출력에 준 영향 0). p2~p4 는 gap 이 음수 -- T1 을 지운 쪽이 "
                                "SC 상관이 더 좋다. own-shuf 도 1e-4 수준이다."),
                    "ci_note": ("subject 별 own/zero 상관이 로그에 없어 부트스트랩 CI 를 만들 수 없다. "
                                "부호가 모든 phase 에서 <= 0 이라 '0 초과' 판정에는 통계가 필요 없다. "
                                "CI 가 필요하면 individuality_metrics 가 subject 별 값을 남겨야 한다.")},
        "min_n": {"chosen": int(min_n), "by_half_width<=0.05": int(need_hw),
                  "by_false_pass<=0.05": int(need_fp), "val_subjects_available": int(n_val_avail),
                  "threshold": THR, "criterion": ("반폭 <= 관문값의 절반(0.05) 이어야 '0.10 이상' 이 "
                                                  "'0 보다 확실히 크다' 를 함의한다. 엄밀히는 n=34 가 "
                                                  "필요하지만 val split 이 31명이라 31 로 둔다 (n=31 반폭 0.052)."),
                  "scan": scan},
        "verdict": ("p2 통과 판정은 성립하지 않는다. n=8 구간이 test 구간과 겹치고, 개인차가 0 인 "
                    f"분포에서도 n=8 이면 {float((m8 >= THR).mean())*100:.1f}% 확률로 0.10 을 넘는다."
                    if overlaps else "n=8 구간과 test 구간이 겹치지 않는다"),
        "n_boot": N_BOOT, "alpha": 0.05, "seed": SEED,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"test n=31 resid_r(LOO)={test_loo:+.4f} | per-subject mean={x.mean():+.4f} "
          f"CI[{ci31['lo']:+.4f}, {ci31['hi']:+.4f}] 폭={ci31['width']:.4f}")
    print(f"val  n=8  resid_r(LOO)={v8:+.4f} CI≈[{val_ci[0]:+.4f}, {val_ci[1]:+.4f}] 폭={2*hw8:.4f} "
          f"({2*hw8/ci31['width']:.2f}배)")
    print(f"겹침: {overlaps} {out['overlap']['range']} | 개인차 0 일 때 n=8 이 0.10 넘을 확률 "
          f"{out['n8_null']['p_mean8_ge_0.10']*100:.1f}%")
    print(f"min_n = {min_n} (반폭<=0.05: {need_hw}, 위양성<=5%: {need_fp}) -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
