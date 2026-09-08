"""CSF/GM/WM 확률 + subject 뇌 마스크 추출 (Atropos 3-class 재실행).

기존 `wm_segment` 는 3-class 를 돌리고도 WM 하나만 저장하고 GM/CSF 를 버렸다. ROI 별 GM 부피,
CSF(위축), subject 고유 뇌 마스크는 전부 **자로 잰** 개인차 feature 라 버릴 이유가 없다
(측정: 자로 잰 tier1 능선 0.079 / 학습된 count head 0.086 -- 같은 자릿수다).

기존 `{sub}_WM_W.npy` 는 건드리지 않는다 (동결 인코더 캐시가 그것으로 만들어졌다).
새 산출물은 `{sub}_tissue_W.npz` 에만 쓴다. uint8 (1/255 단위) 로 저장해 subject 당 27 MB.
"""
import argparse, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import CACHE
from atm_sc.data.wm_segment import brain_mask_W, tissue_path, wm_probability


def one(sub: str, mask: np.ndarray, source: str = "rigid") -> str:
    p = tissue_path(CACHE, sub, source)
    if p.exists() and p.stat().st_size > 1_000_000:
        return "skip"
    t1p = CACHE / f"{sub}_T1w_{source}_W.npy"
    assert t1p.exists(), f"{sub}: T1 캐시 없음 ({t1p.name})"
    t1 = np.load(t1p)
    out = wm_probability(t1, mask=mask, return_all=True)
    # 기존 WM 과 어긋나면 둘 중 하나가 틀린 것이다. 코호트 비율(중앙값 0.373, robust SD 0.038)로
    # 어느 쪽이 이상한지 판정한다 -- 한 subject(train) 는 기존이 0.540 (8 sigma) 이고 새 것이 정상이었다.
    frac = float((out["wm"][mask] > 0.5).sum()) / float(mask.sum())
    assert 0.15 < frac < 0.65, f"{sub}: 새 WM 비율이 이상하다 ({frac:.3f})"
    old = CACHE / f"{sub}_WM_W.npy" if source == "rigid" else None
    if old is not None and old.exists():
        w0 = np.load(old).astype(np.float32)
        r = float(np.corrcoef(w0[mask], out["wm"][mask])[0, 1])
        f0 = float((w0[mask] > 0.5).sum()) / float(mask.sum())
        if r <= 0.95:
            print(f"  [QC] {sub}: 새 WM 이 기존과 다르다 (r={r:.3f}, 기존 비율 {f0:.3f} -> "
                  f"새 {frac:.3f}). 코호트 중앙값 0.373 에 가까운 쪽이 옳다.", flush=True)
    q = {k: np.round(np.clip(out[k], 0, 1) * 255).astype(np.uint8) for k in ("wm", "gm", "csf")}
    tmp = p.with_suffix(".npz.tmp.npz")
    np.savez_compressed(tmp, brain=out["brain"], **q)
    tmp.rename(p)
    assert p.stat().st_size > 1_000_000, f"{p} 가 너무 작다"
    return "ok"


def main(a):
    subs = sorted({p.name.split("_T1w_")[0] for p in CACHE.glob(f"*_T1w_{a.source}_W.npy")})
    assert len(subs) > 200, f"subject 가 {len(subs)}명뿐이다"
    if a.reverse:
        subs = subs[::-1]
    subs = subs[a.start::a.stride]
    mask = brain_mask_W(4)
    n_ok = n_skip = n_err = 0
    for i, s in enumerate(subs):
        t0 = time.time()
        try:
            r = one(s, mask, a.source)
            n_ok += r == "ok"; n_skip += r == "skip"
            print(f"[{i+1}/{len(subs)}] {s} {r} {time.time()-t0:.0f}s", flush=True)
        except Exception:
            n_err += 1
            print(f"[{i+1}/{len(subs)}] {s} FAIL\n{traceback.format_exc()}", flush=True)
    print(f"TISSUE DONE ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    assert n_err == 0, f"{n_err}명 실패"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rigid")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    main(ap.parse_args())
