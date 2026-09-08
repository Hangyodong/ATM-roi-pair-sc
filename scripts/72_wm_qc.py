"""기존 WM 맵 vs 새 3-class 분할 QC — 조용히 틀린 subject 를 잡는다.

한 subject(train) 는 기존 WM 비율이 0.540 (코호트 0.372+-0.020, 8 sigma 밖) 이었다. 그 맵으로
tier1/corridor feature 와 2채널 인코더 입력이 만들어졌으므로 그 subject 의 값은 전부 틀렸다.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import CACHE
from atm_sc.data.wm_segment import brain_mask_W, tissue_path

m = brain_mask_W(4); nb = float(m.sum())
subs = sorted({p.name.split("_WM_W")[0] for p in CACHE.glob("*_WM_W.npy")})
rows = []
for i, s in enumerate(subs):
    w0 = np.load(CACHE / f"{s}_WM_W.npy").astype(np.float32)
    r = {"sub": s, "old_frac": float((w0[m] > 0.5).sum() / nb)}
    p = tissue_path(CACHE, s)
    if p.exists():
        z = np.load(p); w1 = z["wm"].astype(np.float32) / 255.0
        r["new_frac"] = float((w1[m] > 0.5).sum() / nb)
        r["corr"] = float(np.corrcoef(w0[m], w1[m])[0, 1])
        r["gm_frac"] = float((z["gm"].astype(np.float32)[m] / 255.0 > 0.5).sum() / nb)
        r["csf_frac"] = float((z["csf"].astype(np.float32)[m] / 255.0 > 0.5).sum() / nb)
        r["brain_vox"] = int(z["brain"].sum())
    rows.append(r)
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(subs)}", flush=True)
of = np.array([r["old_frac"] for r in rows])
med, mad = float(np.median(of)), float(np.median(np.abs(of - np.median(of))) * 1.4826)
bad = [r for r in rows if abs(r["old_frac"] - med) > 4 * mad]
cor = [r["corr"] for r in rows if "corr" in r]
out = {"n": len(rows), "old_frac_median": round(med, 4), "old_frac_robust_sd": round(mad, 4),
       "corr_old_new": {"min": round(float(np.min(cor)), 4), "p05": round(float(np.percentile(cor, 5)), 4),
                        "median": round(float(np.median(cor)), 4)},
       "n_low_corr": int(sum(c < 0.95 for c in cor)),
       "low_corr_subs": [r["sub"] for r in rows if r.get("corr", 1) < 0.95],
       "outliers_old_frac": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in bad]}
(ROOT / "outputs/eval/wm_qc.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
print(json.dumps(out, indent=1))
