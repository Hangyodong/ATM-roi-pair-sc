#!/usr/bin/env python
"""T1 -> MNI152 **rigid** 정합 (전처리 프로토콜 확정판).

  python scripts/37_build_rigid_t1.py --workers 6

기존 파이프라인은 SyN(비선형) 정합을 썼다. 비선형은 각 뇌를 템플릿 모양으로 구부리므로
개인의 뇌 크기/모양이 워프 필드로 빠져나가고, 그 워프 필드는 저장되지 않았다.
rigid 는 위치/방향만 맞추고 개인 형태를 보존한다 (8명 실측: subject 간 영상 상관
SyN 0.900 -> Rigid 0.731, 개인 변동 1.89배 유지).

주의: streamline/아틀라스는 MNI 에 있고 rigid T1 은 복셀 단위로 그것과 정확히 대응하지 않는다.
인코더는 전역 특징 벡터 하나를 뽑으므로 복셀 대응이 필수는 아니지만, 이 선택은 Methods 에
명시해야 한다.
결과: outputs/cache/{sub}_T1w_rigid_W.npy  (float32 [193,229,193], 정규화 전)
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
    from atm_sc.data.paths import CACHE, t1_path
    from atm_sc.data.prepare_t1 import prepare_subject
    out = CACHE / f"{sub}_T1w_rigid_W.npy"
    if out.exists():
        return sub, "cached", 0.0
    try:
        t0 = time.time()
        v = prepare_subject(t1_path(sub), mode="rigid", out_dir=CACHE)
        assert v.shape == (193, 229, 193), v.shape
        assert np.isfinite(v).all() and v.max() > 0, "정합 결과가 비었거나 NaN"
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, v); tmp.replace(out)
        return sub, "ok", time.time() - t0
    except Exception as e:
        return sub, f"FAIL {type(e).__name__}: {e}", 0.0


def main(a):
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    if a.limit:
        subs = subs[:a.limit]
    print(f"Rigid 정합 {len(subs)}명, 병렬 {a.workers}", flush=True)
    done, fail, secs = 0, [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for sub, st, dt in ex.map(one, subs):
            done += 1
            if dt:
                secs.append(dt)
            if st.startswith("FAIL"):
                fail.append((sub, st)); print(f"  [{done}/{len(subs)}] {sub} {st}", flush=True)
            elif done % 20 == 0:
                print(f"  [{done}/{len(subs)}] {sub} {st} ({np.mean(secs) if secs else 0:.0f}s/명)", flush=True)
    print(f"완료. 실패 {len(fail)}명" + (f": {fail[:3]}" if fail else ""), flush=True)
    assert len(fail) < max(len(subs) * 0.05, 1), f"실패가 너무 많다: {len(fail)}/{len(subs)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="outputs/subjects_train_eval_206.txt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
