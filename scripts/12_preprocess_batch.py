#!/usr/bin/env python
"""Phase 1 전처리 배치: subject 마다 01(SyN→W) → 02(ROI-pair 할당) → 03(bundle npz).

이미 산출물이 있으면 그 단계는 건너뛴다. 한 subject 가 실패해도 다음으로 넘어가며
outputs/preprocess_logs/summary.csv 와 subject 별 로그에 기록한다.
이 머신은 코어 1개(nproc=1)라 병렬화하지 않는다. SyN ~6 min, 02 ~90 s, 03 ~20 s / subject.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import CACHE, subjects          # noqa: E402

LOGS = ROOT / "outputs" / "preprocess_logs"
RP = ROOT / "outputs" / "roi_pairs"
ENV = dict(os.environ, OMP_NUM_THREADS="1", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",
           PYTHONWARNINGS="ignore")


def outputs_of(sub):
    return {"01": CACHE / f"{sub}_T1w_syn_W.npy",
            "02": RP / sub / "assignments.npz",
            "03": RP / sub / "bundles.npz"}


def run_step(sub, step, timeout):
    cmd = {"01": ["scripts/01_qc_coordinate_space.py", "--sub", sub, "--mode", "syn"],
           "02": ["scripts/02_assign_roi_pairs.py", "--sub", sub],
           "03": ["scripts/03_build_roi_pair_bundles.py", "--sub", sub]}[step]
    t0 = time.time()
    with open(LOGS / f"{sub}.log", "a") as log:
        log.write(f"\n===== {step} {time.strftime('%F %T')} =====\n"); log.flush()
        r = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=ENV, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout)
    ok = r.returncode == 0 and outputs_of(sub)[step].exists()
    return ok, time.time() - t0


def process_subject(sub):
    """subject 하나: 01 -> 02 -> 03. (subject, [(step, ok, sec), ...])"""
    outs = outputs_of(sub)
    res = []
    for step, timeout in (("01", 3600), ("02", 1800), ("03", 900)):
        if outs[step].exists():
            continue
        try:
            ok, sec = run_step(sub, step, timeout)
        except subprocess.TimeoutExpired:
            ok, sec = False, timeout
        res.append((step, ok, sec))
        if not ok:
            break                               # 뒤 단계는 앞 단계 산출물이 필요하다
    return sub, res


def main(subs, limit, workers):
    LOGS.mkdir(parents=True, exist_ok=True)
    summ = LOGS / "summary.csv"
    new = not summ.exists()
    todo = [s for s in subs if not all(p.exists() for p in outputs_of(s).values())]
    if limit:
        todo = todo[:limit]
    print(f"남은 subject {len(todo)} / workers {workers}", flush=True)
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["subject", "step", "ok", "seconds", "time"])

        def record(sub, res, i):
            print(f"[{i}/{len(todo)}] {sub}: " + " ".join(f"{st} {'ok' if ok else 'FAIL'} {sec:.0f}s" for st, ok, sec in res), flush=True)
            for st, ok, sec in res:
                w.writerow([sub, st, int(ok), f"{sec:.0f}", time.strftime("%F %T")]); f.flush()

        if workers <= 1:
            for i, sub in enumerate(todo, 1):
                record(*process_subject(sub)[:2], i) if False else record(sub, process_subject(sub)[1], i)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed   # 워커는 subprocess 라 스레드로 충분
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_subject, s): s for s in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    sub, res = fut.result()
                    record(sub, res, i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 subject 수 (0=전부)")
    ap.add_argument("--workers", type=int, default=1, help="동시에 처리할 subject 수 (CPU job 용)")
    ap.add_argument("--threads-per-worker", type=int, default=1, help="ITK/OMP 스레드 수 (워커당)")
    ap.add_argument("--reverse", action="store_true", help="목록 뒤에서부터 (로컬 배치와 반대 방향으로 만나기)")
    a = ap.parse_args()
    subs = a.subjects or subjects()          # .mat ∩ 폴더 = 206
    assert subs, "subject 없음"
    if a.reverse:
        subs = subs[::-1]
    ENV.update(OMP_NUM_THREADS=str(a.threads_per_worker), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(a.threads_per_worker))
    main(subs, a.limit, a.workers)
    print("BATCH DONE", flush=True)
