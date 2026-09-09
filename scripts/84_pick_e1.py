"""E1 조건 개인화 스윕 판정 — 개인차를 가장 많이 싣되 생성이 안 무너지는 gain.

게이트
  recon_rmse_eval_mm <= 4.6      복원이 무너지면 기하가 GT 를 안 닮는다 (J1 실측 4.412)
  endpoint_pair_acc >= 0.9 x J1  조건이 바뀌어 끝점이 더 못 닿으면 안 된다
  valid_conn >= 0.9 x J1
목적함수: gain 이 클수록 조건의 개인 성분이 크다 (0.3 -> 0.69 %, 1.0 -> 4.44 %, 3.0 -> 11.14 %).
게이트를 통과한 것 중 **가장 큰 gain** 을 고른다.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAINS = ["1.0", "3.0"]
COND_SHARE = {"0.3": 0.69, "1.0": 4.44, "3.0": 11.14}     # 재학습 전 실측 (%)


def main(prefix="e1r_g"):
    global PREFIX
    PREFIX = prefix
    base = json.loads((ROOT / "outputs/eval/w1_smoke.json").read_text())["val"]["before"] \
        if (ROOT / "outputs/eval/w1_smoke.json").exists() else None
    ref_ep, ref_vc = (base["endpoint_in_roi"], base["valid_conn"]) if base else (0.6445, 0.2217)
    rows = []
    for g in GAINS:
        rj = ROOT / f"outputs/eval/{PREFIX}{g}.json"
        lj = ROOT / f"outputs/checkpoints/retrain/{PREFIX}{g}/log.jsonl"
        if not (rj.exists() and lj.exists()):
            print(f"gain {g}: 결과 없음", flush=True); continue
        d = json.loads(rj.read_text()); a = d["val"]["after"]
        r = [json.loads(l) for l in lj.read_text().splitlines() if l.strip()]
        ep = float(np.mean([x["endpoint_pair_acc"] for x in r[-20:]]))
        ok = (a["recon_rmse_eval_mm"] <= 4.6 and a["endpoint_in_roi"] >= 0.9 * ref_ep
              and a["valid_conn"] >= 0.9 * ref_vc)
        rows.append({"gain": g, "cond_share": COND_SHARE[g], "recon": a["recon_rmse_eval_mm"],
                     "endpoint_in_roi": a["endpoint_in_roi"], "valid_conn": a["valid_conn"],
                     "endpoint_pair_acc": ep, "pass": ok})
        print(f"gain {g}: 조건 개인성분 {COND_SHARE[g]:5.2f}% | recon {a['recon_rmse_eval_mm']:.3f} "
              f"| endpoint_in_roi {a['endpoint_in_roi']:.4f} | valid_conn {a['valid_conn']:.4f} "
              f"| pair_acc {ep:.3f} | {'통과' if ok else '탈락'}", flush=True)
    ok = [r for r in rows if r["pass"]]
    assert ok, "통과한 gain 이 없다 -- 램프업(gain 을 0 에서 서서히 올리기)이 필요하다"
    win = max(ok, key=lambda r: float(r["gain"]))
    (ROOT / "outputs/eval/e1_pick.json").write_text(json.dumps({"rows": rows, "winner": win}, indent=1))
    print(f"승자: gain {win['gain']} (조건 개인 성분 {win['cond_share']:.2f}%)", flush=True)
    print(f"GAIN={win['gain']}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="e1r_g")
    sys.exit(main(ap.parse_args().prefix))
