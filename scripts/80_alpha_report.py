"""진폭 스윕 요약: 생성 SC 의 subject 간 상관이 GT 에 가장 가까운 alpha 를 고른다."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = []
for p in sorted(ROOT.glob("outputs/eval/w_rl_*.json")):
    d = json.loads(p.read_text())
    if "ridge" not in d:
        continue
    al = d["per_subject"][0].get("ridge_alpha") if d.get("per_subject") else None
    rows.append((al, d["ridge"]["resid_r"], d["ridge"]["inter_subj_r"], d["ridge"]["abs_r"],
                 d["gt_inter_subj_r"], p.name))
print(f"{'alpha':>6s} {'resid_r':>8s} {'inter':>8s} {'abs_r':>8s}  (GT inter {rows[0][4]:.4f})" if rows else "결과 없음")
for r in sorted(rows, key=lambda x: (x[0] is None, x[0])):
    print(f"{str(r[0]):>6s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f}   {r[5]}")
if rows:
    best = min(rows, key=lambda r: abs(r[2] - r[4]))
    print(f"\nGT inter 에 가장 가까운 alpha={best[0]}: inter {best[2]:.4f} (GT {best[4]:.4f}), resid_r {best[1]:.4f}")
