"""두 RL 실행 결과를 나란히 놓고 판정한다.

판정 기준 (우선순위)
  1) resid_r 가 유의하게 높은 쪽. subject 별 상관의 부트스트랩 95 % CI 가 겹치지 않아야 "유의".
  2) 비기면 subject 간 상관이 GT 에 가까운 쪽 (0.96 과 0.8 은 다르다).
  3) 그래도 비기면 절대 r 이 높은 쪽.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.evaluation.reproduction_metrics import loo_center, _corr


def load(tag):
    j = json.loads((ROOT / f"outputs/eval/{tag}.json").read_text())
    z = np.load(ROOT / f"outputs/eval/{tag}_vectors.npz", allow_pickle=False)
    key = "pred_ridge" if "pred_ridge" in z.files else "pred_base"
    return j, z["gt_sc"], z[key], key


def per_subj_resid(P, G):
    Pc, Gc = loo_center(P), loo_center(G)
    return np.array([_corr(Pc[i], Gc[i]) for i in range(len(P))])


def main(a, b):
    ja, Ga, Pa, ka = load(a)
    jb, Gb, Pb, kb = load(b)
    assert Ga.shape == Gb.shape, (Ga.shape, Gb.shape)
    ra, rb = per_subj_resid(Pa, Ga), per_subj_resid(Pb, Gb)
    rng = np.random.default_rng(0); n = len(ra)
    d = np.array([np.nanmean(ra[i] - rb[i]) for i in [rng.integers(0, n, n) for _ in range(2000)]])
    lo, hi = np.percentile(d, [2.5, 97.5])
    gt_inter = ja["gt_inter_subj_r"]
    rows = []
    for tag, j, k, r in ((a, ja, ka, ra), (b, jb, kb, rb)):
        m = j[k.replace("pred_", "")]
        rows.append({"tag": tag, "resid_r": float(np.nanmean(r)), "inter": m["inter_subj_r"],
                     "abs_r": m["abs_r"], "d_inter": abs(m["inter_subj_r"] - gt_inter)})
    print(f"{'실행':22s} {'resid_r':>9s} {'inter':>8s} {'|inter-GT|':>11s} {'abs_r':>8s}")
    for r in rows:
        print(f"{r['tag']:22s} {r['resid_r']:9.4f} {r['inter']:8.4f} {r['d_inter']:11.4f} {r['abs_r']:8.4f}")
    print(f"GT inter {gt_inter:.4f} | resid_r 차({a}-{b}) 평균 {np.mean(ra)-np.mean(rb):+.4f} "
          f"95%CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        win, why = a, "resid_r 우세 (CI 가 0 을 안 포함)"
    elif hi < 0:
        win, why = b, "resid_r 우세 (CI 가 0 을 안 포함)"
    else:
        cand = sorted(rows, key=lambda r: (r["d_inter"], -r["abs_r"]))
        win, why = cand[0]["tag"], "resid_r 는 무승부 -> subject 간 상관이 GT 에 더 가까움"
    print(f"판정: {win} ({why})")
    print(f"WINNER={win}")
    (ROOT / "outputs/eval/compare_last.json").write_text(
        json.dumps({"a": a, "b": b, "rows": rows, "ci": [float(lo), float(hi)], "winner": win}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
