#!/usr/bin/env python
"""전 subject 의 white matter 확률 맵 생성 (FreeSurfer white surface 대체).

  python scripts/36_build_wm_maps.py --workers 6

FreeSurfer recon-all 은 subject 당 6~10시간(206명이면 10~14일)이고 이 머신에 설치되어 있지
않다. 인코더가 3D UNet 이라 표면 메시를 받아도 볼륨으로 되돌려야 하므로 ANTs Atropos 3-class
분할의 WM 확률 볼륨으로 대체한다 (subject 당 ~35초).

입력은 rigid 로 MNI152 에 정합된 W 격자 T1 이다 (전처리 프로토콜). 모델 입력과 같은 공간이어야
하므로 SyN 이 아니라 rigid 를 쓴다.
결과: outputs/cache/{sub}_WM_W.npy  (float32 [193,229,193], 0~1)
"""
import os
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def one(sub):
    from atm_sc.data.paths import CACHE
    from atm_sc.data.wm_segment import brain_mask_W, wm_path, wm_probability
    out = wm_path(CACHE, sub)
    if out.exists():
        return sub, "cached", 0.0
    t1p = CACHE / f"{sub}_T1w_rigid_W.npy"
    if not t1p.exists():
        return sub, "FAIL rigid T1 캐시 없음 (37 번 먼저)", 0.0
    try:
        t0 = time.time()
        wm = wm_probability(np.load(t1p).astype(np.float32), brain_mask_W())
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, wm); tmp.replace(out)          # 부분 저장 파일이 남지 않게
        return sub, f"ok frac={float((wm > 0.5).mean()):.3f}", time.time() - t0
    except Exception as e:
        return sub, f"FAIL {type(e).__name__}: {e}", 0.0


def main(a):
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    if a.limit:
        subs = subs[:a.limit]
    print(f"WM 분할 {len(subs)}명, 병렬 {a.workers}", flush=True)
    done, fail, secs = 0, [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for sub, st, dt in ex.map(one, subs):
            done += 1
            if dt:
                secs.append(dt)
            if st.startswith("FAIL"):
                fail.append((sub, st)); print(f"  [{done}/{len(subs)}] {sub} {st}", flush=True)
            elif done % 20 == 0:
                print(f"  [{done}/{len(subs)}] {sub} {st}", flush=True)
    print(f"완료. 실패 {len(fail)}명" + (f": {fail[:3]}" if fail else "")
          + (f", 평균 {np.mean(secs):.1f}초" if secs else ""), flush=True)
    assert len(fail) < max(len(subs) * 0.05, 1), f"실패가 너무 많다: {len(fail)}/{len(subs)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="outputs/subjects_train_eval_206.txt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
