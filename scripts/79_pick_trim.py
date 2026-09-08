"""trimming 임계값 선택: GT 를 지우지 않으면서 생성 가닥의 초과 방문을 가장 많이 줄이는 설정.

게이트: GT retained >= --gt-keep (기본 0.9). 지표만 좋아지고 진짜 연결이 사라지면 개선이 아니다.
목적함수: 생성 가닥의 pair/stream 을 GT 의 pair/stream 에 가깝게.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(a):
    d = json.loads((ROOT / "outputs/eval/trim_calibration.json").read_text())
    gt, gen = d["gt"], d.get("gen")
    assert gen, "생성물 측정이 없다 (73 을 --ckpt 와 함께 돌려라)"
    tgt = gt["raw"]["pair_per_stream"]
    rows = []
    for k in gt:
        if k == "raw" or k not in gen:
            continue
        rows.append({"cfg": k, "gt_retained": gt[k]["retained"], "gt_kept_len": gt[k]["kept_len"],
                     "gen_retained": gen[k]["retained"], "gen_pair_per_stream": gen[k]["pair_per_stream"],
                     "gen_n_visit": gen[k]["n_visit"], "gap": abs(gen[k]["pair_per_stream"] - tgt)})
    ok = [r for r in rows if r["gt_retained"] >= a.gt_keep]
    for r in sorted(rows, key=lambda x: x["gap"])[:8]:
        print(f"{r['cfg']:22s} GT유지 {r['gt_retained']:.3f} 생성유지 {r['gen_retained']:.3f} "
              f"pair/stream {r['gen_pair_per_stream']:.1f} (GT {tgt:.1f}) "
              f"{'OK' if r['gt_retained']>=a.gt_keep else '탈락'}", flush=True)
    if not ok:
        best = max(rows, key=lambda r: r["gt_retained"])
        print(f"\nGT 유지율 {a.gt_keep} 를 넘는 설정이 없다 (최대 {best['gt_retained']:.3f}). "
              f"trimming 을 끄고 진행한다 -- 진짜 연결을 지우는 필터는 개선이 아니다.", flush=True)
        (ROOT / "outputs/eval/trim_pick.json").write_text(json.dumps({"rows": rows, "winner": None}, indent=1))
        print("TRIM=0")
        return 0
    win = min(ok, key=lambda r: r["gap"])
    thr, marg, out = win["cfg"].replace("thr", "").replace("m", "").replace("o", "").split("_")
    (ROOT / "outputs/eval/trim_pick.json").write_text(
        json.dumps({"rows": rows, "winner": win, "gt_pair_per_stream": tgt}, indent=1))
    print(f"\n승자: {win['cfg']} (GT 유지 {win['gt_retained']:.3f}, "
          f"pair/stream {win['gen_pair_per_stream']:.1f} vs GT {tgt:.1f})", flush=True)
    print(f"TRIM=1\nWM_THR={thr}\nGM_MARGIN={marg}\nMAX_OUTSIDE={out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-keep", type=float, default=0.9)
    sys.exit(main(ap.parse_args()))
