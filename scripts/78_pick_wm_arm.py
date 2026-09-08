"""W1 스윕 판정: 백질 점유를 가장 많이 올리면서 복원/연결을 안 망가뜨린 팔을 고른다.

게이트 (하나라도 어기면 탈락)
  recon_rmse_eval_mm <= 4.6     복원이 무너지면 기하가 GT 를 안 닮는다 (J1 실측 4.41)
  valid_conn >= 0.9 * before    백질로 뭉치느라 끝점이 피질에 못 닿으면 소용없다
  wm_occ_gen 상승 > 0.01        올라가지 않으면 손실이 안 걸린 것이다
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A": (10, 1.0e-05), "B": (10, 1.0e-04), "C": (30, 1.0e-04)}


def main():
    rows = []
    for n, (wg, lr) in ARMS.items():
        lj = ROOT / f"outputs/checkpoints/retrain/w1_sweep_{n}/log.jsonl"
        rj = ROOT / f"outputs/eval/w1_sweep_{n}.json"
        if not (lj.exists() and rj.exists()):
            print(f"arm {n}: 결과 없음 (건너뜀)", flush=True); continue
        r = [json.loads(l) for l in lj.read_text().splitlines() if l.strip()]
        d = json.loads(rj.read_text())
        b, a = d["val"]["before"], d["val"]["after"]
        occ0 = float(np.mean([x["wm_occ_gen"] for x in r[:20]]))
        occ1 = float(np.mean([x["wm_occ_gen"] for x in r[-20:]]))
        ok = (a["recon_rmse_eval_mm"] <= 4.6 and a["valid_conn"] >= 0.9 * b["valid_conn"]
              and occ1 - occ0 > 0.01)
        rows.append({"arm": n, "wm_gen": wg, "lr_dec": lr, "occ0": round(occ0, 4), "occ1": round(occ1, 4),
                     "d_occ": round(occ1 - occ0, 4), "recon": round(a["recon_rmse_eval_mm"], 3),
                     "valid_conn": round(a["valid_conn"], 4), "pass": ok})
        print(f"arm {n} wm_gen={wg} lr_dec={lr:g}: occ {occ0:.3f} -> {occ1:.3f} (+{occ1-occ0:.3f}) "
              f"recon {a['recon_rmse_eval_mm']:.3f} valid_conn {a['valid_conn']:.4f} "
              f"{'통과' if ok else '탈락'}", flush=True)
    ok = [r for r in rows if r["pass"]]
    assert ok, "통과한 팔이 없다 -- 손실 가중치를 더 올리거나 prior 를 손봐야 한다"
    win = max(ok, key=lambda r: r["d_occ"])
    out = {"rows": rows, "winner": win}
    (ROOT / "outputs/eval/w1_arm_pick.json").write_text(json.dumps(out, indent=1))
    print(f"승자: arm {win['arm']} (wm_gen={win['wm_gen']} lr_dec={win['lr_dec']:g})", flush=True)
    print(f"WM_GEN={win['wm_gen']}\nLR_DEC={win['lr_dec']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
