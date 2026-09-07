#!/usr/bin/env python
"""9-phase fine-tuning 모니터. 모든 phase 의 설정 · 진행률 · loss/gradient 추이 · val 결과 · ETA 를 한 화면에.

  python scripts/20_monitor.py                 # 1회 출력
  python scripts/20_monitor.py --watch 60      # 60 s 마다 갱신 (Ctrl-C 종료)
  python scripts/20_monitor.py --json          # 수집 데이터를 JSON 으로
  python scripts/20_monitor.py --no-color      # 로그 파일로 남길 때

읽는 것: configs/pipeline.yaml → 각 phase yaml, <out_dir>/log.jsonl, *_latest.pt / *_step{N}.pt,
outputs/checkpoints/pipeline_state.json, val_metrics.jsonl, pipeline.lock(heartbeat), outputs/pbs/*.txt,
outputs/preprocess_logs/summary.csv, outputs/cache, outputs/roi_pairs, nvidia-smi, qstat.
"""
import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.training.run import PHASES            # noqa: E402  (phase → 활성 loss, torch CPU import 만)
from atm_sc.training.trainer import LossWeights   # noqa: E402

# ----------------------------------------------------------------------------- 상수
ROLE = {                                              # 최종 전략 §7
    "phase2_geometry": "pretrained VAE/decoder 를 GT streamline 분포에 적응 (T1 encoder 동결)",
    "phase3_t1_encoder": "UNet T1 encoder unfreeze → PPMI 도메인·subject 특징 학습 (recon 만)",
    "phase4_endpoint": "+ L_endpoint: 생성 streamline 양 끝이 지정 ROI 쌍에 닿게",
    "phase5_edge": "+ L_edge: ROI 쌍 연결 유무 판별 (edge head)",
    "phase6_sc_corr": "+ L_SC_corr: whole-brain SC 패턴, gradient 가 T1 encoder 까지",
    "phase7_sc_mag": "+ L_SC_mag: SC 절대 크기 (streamline weight head)",
    "phase8_length": "+ L_length: tract-length 행렬",
    "phase9_joint": "작은 LR 로 전체 loss joint fine-tuning",
    "s1_route": "baseline sc_corr 에 route(중간 통과 ROI) + ROI-pair 균형 노출 + GESTA synthetic",
    "s2_presence": "+ SUB-SUB pass-edge 존재 여부 (크기와 분리)",
    "s3_seg_count": "+ SC edge-aligned segment 분기 + Edge Count Head",
    "s4_mag": "+ log magnitude (Weight Head 는 route/segment 뒤에)",
    "s5_length": "+ tract-length 행렬",
    "s6_joint": "작은 LR 로 전체 loss joint fine-tuning",
}
SHORT = {"recon": "recon", "kl": "kl", "geom": "geom", "endpoint": "endpt", "edge": "edge", "corr": "corr",
         "mag": "mag", "length": "len", "route": "route", "route_gen": "routeG", "presence": "presn",
         "seg_recon": "seg", "seg_endpoint": "segEnd", "count": "count", "scale": "scale", "rmse": "rmse"}
LOSS_KEYS = ["L_recon", "L_kl", "L_geom", "L_endpoint", "L_route", "L_route_gen", "L_edge", "L_corr",
             "L_mag", "L_length", "L_presence", "L_seg_recon", "L_seg_kl", "L_seg_geom", "L_seg_endpoint",
             "L_count", "L_scale", "L_rmse"]
GNORM_KEYS = ["gnorm_t1_encoder", "gnorm_vae_encoder", "gnorm_decoder", "gnorm_heads", "grad_norm_total"]
AUX_KEYS = ["recon_rmse_train_mm", "endpoint_pair_acc", "edge_f1", "edge_acc", "w_mean", "sc_pred_sum", "n_generated",
            "anat_norm", "dLda_G", "dLda_R", "dLda_E", "dLda_total",
            "corr_r_ctx-ctx", "corr_r_ctx-sub", "corr_r_sub-sub", "recon_tier_small", "recon_tier_mid", "recon_tier_large",
            "sc_rlog_ctx-ctx", "sc_rlog_ctx-sub", "sc_rlog_sub-sub", "route_f1", "route_recall", "route_precision",
            "route_gt_visits", "route_pred_visits", "gen_presence_recall", "gen_presence_f1",
            "recon_real_frac", "recon_synth_frac", "seg_endpoint_pair_acc", "seg_edges", "seg_mean_length_mm",
            "count_r", "count_r_log", "count_ccc", "count_log_mae", "count_recall", "count_zero_specificity", "sc_sum_ratio",
            "seg_block_ctx-ctx", "seg_block_ctx-sub", "seg_block_sub-sub"]
# 아직 실측이 없는 phase 의 step 시간 추정: 기준 phase 실측 + 새로 켜지는 loss 의 생성/SC 비용 (파일럿 실측, 초)
EST_EXTRA = {"endpoint": 0.25, "edge": 0.0, "corr": 0.05, "mag": 0.0, "length": 0.45,
             "route": 0.30, "presence": 0.0, "segment": 0.35, "count": 0.05, "scale": 0.0, "rmse": 0.0}
FALLBACK_STEP_SEC = {"full": 1.7, "none": 0.03}
FALLBACK_VAL_SEC = 240.0
HEARTBEAT_STALE = 180.0


# ----------------------------------------------------------------------------- 유틸
class _Color:
    on = True

    def __call__(self, s, *codes):
        return f"\033[{';'.join(codes)}m{s}\033[0m" if (self.on and codes) else str(s)


C = _Color()
GREEN, YELLOW, RED, CYAN, DIM, BOLD, MAG = "32", "33", "31", "36", "90", "1", "35"


def sh(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def fnum(v, nd=3, width=None):
    if v is None:
        s = "-"
    elif isinstance(v, float) and (abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0)):
        s = f"{v:.2e}"
    elif isinstance(v, float):
        s = f"{v:.{nd}f}"
    else:
        s = str(v)
    return s.rjust(width) if width else s


def hms(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def clock(ts):
    return time.strftime("%m/%d %H:%M", time.localtime(ts)) if ts else "-"


def bar(frac, width):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def spark(vals, bins=40):
    vals = [v for v in vals if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if len(vals) < 2:
        return ""
    n = min(bins, len(vals))
    edges = [round(i * len(vals) / n) for i in range(n + 1)]
    b = [mean(vals[edges[i]:edges[i + 1]]) for i in range(n)]
    b = [x for x in b if x is not None]
    lo, hi = min(b), max(b)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[int((v - lo) / (hi - lo + 1e-12) * 7.999)] for v in b)


def fmt_lr(x):
    return "-" if x is None else f"{x:.0e}".replace("e-0", "e-")


def dw(s):
    """터미널 표시 폭 (한글/전각 = 2칸)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s, w, right=False):
    gap = max(0, w - dw(s))
    return (" " * gap + str(s)) if right else (str(s) + " " * gap)


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                                  # 쓰는 중인 마지막 줄
    return out


def qstat(job_id):
    if not job_id or not shutil.which("qstat"):
        return None
    txt = sh(["qstat", "-f", job_id])
    if not txt.strip():
        return {"id": job_id, "state": "없음/종료"}
    kv = {}
    cur = None
    for line in txt.splitlines():
        if " = " in line:
            k, v = line.strip().split(" = ", 1)
            kv[k] = v
            cur = k
        elif cur and line.startswith("\t"):
            kv[cur] += line.strip()                       # 줄바꿈된 값
    return {"id": job_id, "state": kv.get("job_state", "?"), "queue": kv.get("queue", "?"),
            "walltime": kv.get("resources_used.walltime", "-"), "comment": kv.get("comment", "")}


def gpu_status():
    if not shutil.which("nvidia-smi"):
        return None
    q = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]).strip()
    if not q:
        return None
    used, total, util = [x.strip() for x in q.splitlines()[0].split(",")]
    apps = []
    for line in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"]).splitlines():
        if line.strip():
            pid, mem, name = [x.strip() for x in line.split(",", 2)]
            apps.append({"pid": int(pid), "mem_mb": int(mem), "name": Path(name).name})
    return {"used_gb": int(used) / 1024, "total_gb": int(total) / 1024, "util": int(util), "apps": apps}


# ----------------------------------------------------------------------------- 수집
def collect(pipeline_cfg, window, bins):
    now = time.time()
    P = yaml.safe_load((ROOT / pipeline_cfg).read_text())
    state_p, lock_p = ROOT / P["state_file"], ROOT / P["lock_file"]
    state = json.loads(state_p.read_text()) if state_p.exists() else {"phase_idx": 0, "done": [], "history": []}
    val_hist = read_jsonl(state_p.parent / "val_metrics.jsonl")

    # lock / heartbeat
    lock = None
    if lock_p.exists():
        host, pid = lock_p.read_text().strip().split(":")
        age = now - lock_p.stat().st_mtime
        alive_pid = None
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive_pid = True
            except OSError:
                alive_pid = False
        lock = {"host": host, "pid": int(pid), "heartbeat_age": age,
                "alive": bool(alive_pid) if alive_pid is not None else age < HEARTBEAT_STALE, "same_host": alive_pid is not None}

    phases = []
    for idx, cfg_path in enumerate(P["phases"]):
        cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        name, out_dir, max_steps = cfg["phase"], ROOT / cfg["out_dir"], int(cfg["max_steps"])
        tr = cfg.get("train", {})
        lw = {**asdict(LossWeights()), **cfg.get("loss", {})}
        ACT = ["recon", "endpoint", "route", "edge", "corr", "presence", "mag", "scale", "rmse", "length", "segment", "count"]
        active = [k for k in ACT if k in PHASES[name]]
        LOSS_OF_ACT = {"endpoint": ["endpoint"], "route": ["route", "route_gen"], "edge": ["edge"],
                       "corr": ["corr"], "presence": ["presence"], "mag": ["mag"], "length": ["length"],
                       "segment": ["seg_recon", "seg_endpoint"], "count": ["count"],
                       "scale": ["scale"], "rmse": ["rmse"]}
        unet_level = cfg.get("unet_level") or ("full" if cfg.get("trainable") == "full" else "none")
        final, latest = out_dir / f"{name}_step{max_steps}.pt", out_dir / f"{name}_latest.pt"
        rows = read_jsonl(out_dir / "log.jsonl")
        step = int(rows[-1]["step"]) if rows else 0
        if final.exists():
            status, step = "done", max_steps
        elif idx < state["phase_idx"]:
            status = "done"
        elif idx == state["phase_idx"]:
            status = "running" if (lock and lock["alive"]) else ("stalled" if rows else "pending")
        else:
            status = "pending"
        recent, first = rows[-window:], rows[:window]
        resumes = sum(1 for a, b in zip(rows, rows[1:]) if b["step"] <= a["step"])
        vals = [v for v in val_hist if v.get("checkpoint", "").startswith(str(out_dir.relative_to(ROOT)))]
        phases.append({
            # baseline(configs/pipeline.yaml) 은 phase 2 부터 시작한다 (phase 0/1 은 전처리)
            "idx": idx, "n": idx + (2 if Path(pipeline_cfg).name == "pipeline.yaml" else 1),
            "key": Path(cfg_path).stem, "name": name, "cfg_path": cfg_path,
            "out_dir": str(out_dir.relative_to(ROOT)), "max_steps": max_steps, "step": step, "status": status,
            "trainable": cfg.get("trainable", "vae"), "unet_level": unet_level,
            "lr": {k: tr.get(f"lr_{k}") for k in ("t1", "vae_enc", "dec", "heads")},
            "sc_mode": tr.get("sc_mode", "pass"), "active": active,
            "weights": {k: lw[k] for k in (["recon", "kl", "geom"] if "recon" in active else [])
                        + [w for a in active for w in LOSS_OF_ACT.get(a, [])]},
            "n_rows": len(rows), "resumes": resumes,
            "train_sec": sum((r.get("step_sec") or 0) for r in rows),
            "step_sec": mean([r.get("step_sec") for r in recent]) if rows else None,
            "peak_vram_gb": max((r.get("peak_vram_gb") or 0) for r in rows) if rows else None,
            "last": rows[-1] if rows else None,
            "first_mean": {k: mean([r.get(k) for r in first]) for k in LOSS_KEYS + GNORM_KEYS + AUX_KEYS},
            "last_mean": {k: mean([r.get(k) for r in recent]) for k in LOSS_KEYS + GNORM_KEYS + AUX_KEYS},
            "series": {k: [r.get(k) for r in rows] for k in LOSS_KEYS + ["grad_norm_total", "endpoint_pair_acc", "edge_f1"]},
            "final_mtime": final.stat().st_mtime if final.exists() else None,
            "latest_mtime": latest.stat().st_mtime if latest.exists() else None,
            "val": vals[-1] if vals else None,
        })

    # val 소요 시간 (phase 최종 checkpoint mtime → val 기록 시각)
    val_secs = []
    for ph in phases:
        if ph["val"] and ph["final_mtime"]:
            t = time.mktime(time.strptime(ph["val"]["time"], "%Y-%m-%d %H:%M:%S"))
            if 0 < t - ph["final_mtime"] < 3600:
                val_secs.append(t - ph["final_mtime"])
    val_sec = mean(val_secs) or FALLBACK_VAL_SEC

    # ETA: 진행 중 phase 는 실측, 대기 phase 는 같은 unet_level 의 최근 실측 + 새 loss 비용
    t = now
    base = None                                           # (unet_level, active, step_sec)
    for ph in phases:
        if ph["status"] == "done":
            ph["eta_end"], ph["eta_est"], ph["remaining_sec"] = ph["final_mtime"] or ph["latest_mtime"], False, 0
            if ph["step_sec"] and ph["n_rows"] >= 20:
                base = (ph["unet_level"], set(ph["active"]), ph["step_sec"])
            continue
        measured = ph["step_sec"] if ph["n_rows"] >= 20 else None
        if measured:
            sec, est = measured, False
            base = (ph["unet_level"], set(ph["active"]), measured)
        elif base and base[0] == ph["unet_level"]:
            sec, est = base[2] + sum(EST_EXTRA.get(l, 0) for l in set(ph["active"]) - base[1]), True
        else:
            sec, est = FALLBACK_STEP_SEC.get(ph["unet_level"], 1.7) + sum(EST_EXTRA.get(l, 0) for l in ph["active"]), True
        ph["est_step_sec"] = sec
        remaining = (ph["max_steps"] - ph["step"]) * sec + val_sec
        if ph["status"] in ("running", "stalled"):
            t = now + remaining
        else:
            t = t + remaining
        ph["eta_end"], ph["eta_est"], ph["remaining_sec"] = t, est, remaining
    total_eta = phases[-1]["eta_end"] if phases else None

    # 전처리 (Phase 1)
    subs_p = ROOT / "outputs" / "subjects_train_eval_206.txt"
    subs = [l.strip() for l in subs_p.read_text().splitlines() if l.strip()] if subs_p.exists() else []
    cache, rp = ROOT / "outputs" / "cache", ROOT / "outputs" / "roi_pairs"

    def ready(s):
        return (cache / f"{s}_T1w_syn_W.npy").exists() and (rp / s / "assignments.npz").exists() and (rp / s / "bundles.npz").exists()

    pre = {"n": len(subs),
           "t1": sum((cache / f"{s}_T1w_syn_W.npy").exists() for s in subs),
           "assign": sum((rp / s / "assignments.npz").exists() for s in subs),
           "bundles": sum((rp / s / "bundles.npz").exists() for s in subs),
           "visit": sum((rp / s / "visit.npz").exists() for s in subs),
           "segments": sum((rp / s / "edge_segments.npz").exists() for s in subs),
           "synthetic": sum((ROOT / "outputs" / "synthetic" / s / "synthetic.npz").exists() for s in subs),
           "ready": sum(ready(s) for s in subs), "failed": [], "splits": {}}
    sp = ROOT / "outputs" / "preprocess_logs" / "summary.csv"
    if sp.exists():
        bad = {}
        for r in csv.DictReader(open(sp)):
            if r.get("ok") == "0":
                bad[r["subject"]] = r["step"]
        pre["failed"] = [f"{s}(step {st})" for s, st in sorted(bad.items()) if not ready(s)]
    for split in ("train", "val", "test"):
        f = ROOT / "outputs" / "splits" / f"{split}.txt"
        if f.exists():
            ss = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            pre["splits"][split] = (sum(ready(s) for s in ss), len(ss))

    jobs = {}
    for tag, f in (("h100", "h100_job_id.txt"), ("train", "last_job_id.txt"), ("preproc", "preproc_job_id.txt")):
        p = ROOT / "outputs" / "pbs" / f
        jobs[tag] = qstat(p.read_text().strip()) if p.exists() else None

    pre_f = ROOT / P["preempt_file"] if P.get("preempt_file") else None      # 'pre' 는 전처리 통계 변수라 이름 분리
    preempt = ({"path": str(pre_f.relative_to(ROOT)), "age": now - pre_f.stat().st_mtime}
               if pre_f and pre_f.exists() else None)
    return {"preempt": preempt,
            "now": now, "host": socket.gethostname(), "pipeline": pipeline_cfg, "state": state, "lock": lock,
            "phases": phases, "val_sec": val_sec, "total_eta": total_eta, "pre": pre, "jobs": jobs,
            "gpu": gpu_status(), "window": window, "bins": bins}


# ----------------------------------------------------------------------------- 렌더
STATUS_TXT = {"done": ("✔ 완료", GREEN), "running": ("▶ 진행", YELLOW), "stalled": ("‼ 중단", RED), "pending": ("· 대기", DIM)}


def render(d, width):
    L = []
    sep = C("━" * width, DIM)
    now = d["now"]
    L.append(C(f" ATM ROI-pair SC fine-tuning 모니터 ", BOLD) + C(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}  host {d['host']}  pipeline {d['pipeline']}", DIM))
    L.append(sep)

    # 실행 상태
    lk = d["lock"]
    if lk:
        who = f"{lk['host']}:{lk['pid']}"
        s = C(f"lock {who} 살아있음, heartbeat {lk['heartbeat_age']:.0f}s 전", GREEN) if lk["alive"] else C(f"lock {who} 응답 없음 (heartbeat {hms(lk['heartbeat_age'])} 전) — 죽었거나 멈춤", RED)
    else:
        s = C("lock 없음 — 실행 중인 driver 없음", RED if d["state"]["phase_idx"] < len(d["phases"]) else GREEN)
    L.append(" 실행   " + s)
    if d.get("preempt"):
        L.append(" 양보   " + C(f"preempt 파일 있음 ({d['preempt']['path']}, {hms(d['preempt']['age'])} 전) — "
                                f"학습이 GPU 를 비우고 대기 중", YELLOW))
    g = d["gpu"]
    if g:
        apps = ", ".join(f"{a['name']}[{a['pid']}] {a['mem_mb']/1024:.1f}GB" for a in g["apps"]) or "없음"
        L.append(f" GPU    {g['used_gb']:.1f}/{g['total_gb']:.1f} GB · util {g['util']}% · 프로세스: {apps}")
    for tag, label in (("h100", "H100 job "), ("train", "이전 job "), ("preproc", "전처리 job")):
        j = d["jobs"].get(tag)
        if j:
            st = j["state"]
            col = GREEN if st == "R" else (YELLOW if st == "Q" else DIM)
            extra = f" · {j.get('queue')} · walltime {j.get('walltime')}" if "queue" in j else ""
            cm = f" · {j['comment'][:70]}" if j.get("comment") and st == "Q" else ""
            L.append(f" PBS    {label} {j['id']} " + C(st, col) + extra + cm)
    p = d["pre"]
    fails = f" · 실패 {len(p['failed'])}: {', '.join(p['failed'])}" if p["failed"] else " · 실패 0"
    spl = " · ".join(f"{k} {a}/{b}" for k, (a, b) in p["splits"].items())
    L.append(f" 전처리 T1 {p['t1']}/{p['n']} · bundles {p['bundles']}/{p['n']} · 통과ROI {p['visit']}/{p['n']} · "
             f"edge segment {p['segments']}/{p['n']} · synthetic {p['synthetic']}/144 · 학습가능 {p['ready']}/{p['n']}{fails}")
    L.append(f"        split 준비: {spl}")
    L.append(sep)

    # phase 진행 표
    bw = 20 if width >= 110 else 12
    L.append(C(" Phase 진행", BOLD) + C(f"   (s/step 는 최근 {d['window']} step 평균, * = 추정, ETA 에 phase 별 val {hms(d['val_sec'])} 포함)", DIM))
    L.append(C(f" {'#':>2} {'phase':16} {pad('상태', 8)} {pad('진행', bw)} {'':>4} {'step':>11} {'s/step':>7} {pad('학습시간', 8, True)} {pad('남음', 7, True)} {pad('종료(예상)', 12, True)}  {pad('재개', 4, True)} {'VRAM':>6}", DIM))
    for ph in d["phases"]:
        txt, col = STATUS_TXT[ph["status"]]
        frac = ph["step"] / ph["max_steps"]
        ss = ph.get("step_sec") if ph["status"] != "pending" or ph["n_rows"] >= 20 else ph.get("est_step_sec")
        ss_txt = "-" if ss is None else (f"{ss:.2f}*" if ph.get("eta_est") else f"{ss:.2f}")
        end = clock(ph["eta_end"]) + ("*" if ph.get("eta_est") else "")
        rem = "-" if ph["status"] == "done" else hms(ph["remaining_sec"])
        line = (f" {ph['n']:>2} {ph['key'][:16]:16} {C(pad(txt, 8), col)} {C(bar(frac, bw), col)} {frac*100:3.0f}% "
                f"{ph['step']:>5}/{ph['max_steps']:<5} {ss_txt:>7} {hms(ph['train_sec']) if ph['n_rows'] else '-':>8} {rem:>7} {end:>12}  "
                f"{ph['resumes'] if ph['n_rows'] else '-':>4} {fnum(ph['peak_vram_gb'], 1) if ph['peak_vram_gb'] else '-':>6}")
        L.append(line)
    if d["total_eta"]:
        done_n = sum(ph["status"] == "done" for ph in d["phases"])
        L.append(C(f"   전체: {done_n}/{len(d['phases'])} phase 완료 · 전체 종료 예상 {clock(d['total_eta'])} (남은 시간 {hms(d['total_eta'] - now)})", BOLD))
    L.append(sep)

    # phase 설정 표
    L.append(C(" Phase 설정", BOLD) + C("   (trainable / UNet unfreeze / 그룹별 LR / 활성 loss 와 λ)", DIM))
    L.append(C(f" {'#':>2} {'phase':16} {'trainable':10} {'UNet':5} {'LR t1/vae/dec/heads':22} {'loss(λ)':70} 역할", DIM))
    for ph in d["phases"]:
        lr = "/".join(fmt_lr(ph["lr"][k]) if (k != "t1" or ph["trainable"] == "full") else "-" for k in ("t1", "vae_enc", "dec", "heads"))
        ws = " ".join(f"{SHORT[k]}:{v:g}" for k, v in ph["weights"].items())
        L.append(f" {ph['n']:>2} {ph['key'][:16]:16} {ph['trainable']:10} {ph['unet_level']:5} {lr:22} {ws:70} {C(ROLE.get(ph['key'], ''), DIM)}")
    L.append(sep)

    # 현재 phase 상세
    cur = next((ph for ph in d["phases"] if ph["status"] in ("running", "stalled")), None) or \
        next((ph for ph in reversed(d["phases"]) if ph["n_rows"]), None)
    if cur and cur["last"]:
        last = cur["last"]
        L.append(C(f" 현재 phase 상세: #{cur['n']} {cur['key']} ({cur['name']})", BOLD) + C(f"   step {last.get('step')} · subject {last.get('subject')} · out {cur['out_dir']} · 첫 {d['window']} step 평균 → 최근 {d['window']} step 평균 · 추이 {d['bins']} bin", DIM))
        L.append(C(f"   {'loss':12} {'λ':>5} {'first':>9} {'last':>9} {'Δ%':>7}  추이", DIM))
        for k in LOSS_KEYS:
            f0, f1 = cur["first_mean"].get(k), cur["last_mean"].get(k)
            if f1 is None:
                continue
            lam = cur["weights"].get(k[2:], None)
            dpct = f"{(f1 - f0) / abs(f0) * 100:+.1f}%" if f0 else "-"
            col = GREEN if (f0 and f1 < f0) else (RED if (f0 and f1 > f0 * 1.05) else "")
            L.append(f"   {k:12} {fnum(lam, 2, 5) if lam is not None else '    -'} {fnum(f0, 3, 9)} {C(fnum(f1, 3, 9), col)} {dpct:>7}  {C(spark(cur['series'][k], d['bins']), CYAN)}")
        gn = [f"{k.replace('gnorm_', '').replace('grad_norm_', '')} {fnum(cur['first_mean'][k], 1)}→{fnum(cur['last_mean'][k], 1)}" for k in GNORM_KEYS if cur["last_mean"].get(k) is not None]
        L.append(f"   grad norm  {' · '.join(gn)}   " + C(spark(cur['series']['grad_norm_total'], d['bins']), MAG))
        dl = [f"{k[5:]} {fnum(cur['last_mean'][k], 2)}" for k in ("dLda_G", "dLda_R", "dLda_E", "dLda_total") if cur["last_mean"].get(k) is not None]
        aux = [f"{k} {fnum(cur['last_mean'][k], 3)}" for k in ("recon_rmse_train_mm", "endpoint_pair_acc", "edge_f1", "w_mean", "sc_pred_sum", "anat_norm") if cur["last_mean"].get(k) is not None]
        L.append(f"   dL/da(anatomy) {' · '.join(dl) or '-'}   |   {' · '.join(aux)}")
        br = [f"{b} {fnum(cur['last_mean'].get(f'sc_rlog_{b}'), 3)}" for b in ("ctx-ctx", "ctx-sub", "sub-sub") if cur["last_mean"].get(f"sc_rlog_{b}") is not None]
        ts = [f"{t} {fnum(cur['last_mean'].get(f'recon_tier_{t}'), 2)}" for t in ("small", "mid", "large") if cur["last_mean"].get(f"recon_tier_{t}") is not None]
        if br or ts:
            L.append(f"   block SC r_log {' · '.join(br) or '-'}   |   recon pair tier share(소/중/대) {' · '.join(ts) or '-'}")
        rt = [f"{k.replace('route_', '')} {fnum(cur['last_mean'].get(k), 3)}" for k in
              ("route_f1", "route_recall", "route_gt_visits", "route_pred_visits") if cur["last_mean"].get(k) is not None]
        pr = [f"{k.replace('gen_presence_', '')} {fnum(cur['last_mean'].get(k), 3)}" for k in
              ("gen_presence_recall", "gen_presence_f1") if cur["last_mean"].get(k) is not None]
        mx = [f"{k.replace('recon_', '')} {fnum(cur['last_mean'].get(k), 2)}" for k in
              ("recon_real_frac", "recon_synth_frac") if cur["last_mean"].get(k) is not None]
        if rt or pr or mx:
            L.append(f"   route {' · '.join(rt) or '-'}   |   SS presence {' · '.join(pr) or '-'}   |   recon mix {' · '.join(mx) or '-'}")
        sg = [f"{k.replace('seg_', '')} {fnum(cur['last_mean'].get(k), 3)}" for k in
              ("seg_edges", "seg_endpoint_pair_acc", "seg_mean_length_mm") if cur["last_mean"].get(k) is not None]
        ct = [f"{k.replace('count_', '')} {fnum(cur['last_mean'].get(k), 3)}" for k in
              ("count_r", "count_ccc", "count_log_mae", "count_recall") if cur["last_mean"].get(k) is not None]
        if sg or ct:
            L.append(f"   segment {' · '.join(sg) or '-'}   |   edge count {' · '.join(ct) or '-'}")
        for k, lab in (("endpoint_pair_acc", "endpoint pair acc"), ("edge_f1", "edge F1")):
            if cur["last_mean"].get(k) is not None:
                L.append(f"   {lab:18} {fnum(cur['first_mean'][k], 3)} → {fnum(cur['last_mean'][k], 3)}   " + C(spark(cur['series'][k], d['bins']), CYAN))
        L.append(f"   step {fnum(cur['step_sec'], 2)} s · 생성 {last.get('n_generated', 0)} streamline/step · peak VRAM {fnum(cur['peak_vram_gb'], 1)} GB · 순수 학습시간 {hms(cur['train_sec'])} · 재개 {cur['resumes']}회 · latest ckpt {clock(cur['latest_mtime'])}")
        L.append(sep)

    # val 결과
    L.append(C(" Val 결과", BOLD) + C("   (phase 종료마다 unseen val subject · prior-z 생성 · hard atlas 끝점 / whole-brain pass-SC vs GT .mat, weighted)", DIM))
    L.append(C(f" {'#':>2} {'phase':16} {'n':>2} {'pair_acc':>8} {'endROI':>7} {pad('끝점거리', 8, True)} {'SC r':>6} {'r_log':>6} {'CCC':>6} {'RMSE':>8} {'합비':>6} {'routeF1':>7}  기록시각", DIM))
    prev = None
    for ph in d["phases"]:
        v = ph["val"]
        if not v:
            L.append(f" {ph['n']:>2} {ph['key'][:16]:16} " + C("-" * 12 + ("  (진행 중)" if ph["status"] == "running" else ""), DIM))
            continue

        def cell(k, w, nd=3, better="up"):
            x = v.get(k)
            s = fnum(x, nd, w)
            if prev and prev.get(k) is not None and x is not None:
                good = x > prev[k] if better == "up" else x < prev[k]
                return C(s, GREEN if good else RED)
            return s
        L.append(f" {ph['n']:>2} {ph['key'][:16]:16} {v.get('n_val', 0):>2} {cell('pair_acc', 8)} {cell('end_roi_acc', 7)} "
                 f"{cell('endpoint_dist_mm', 8, 1, 'down')} {cell('sc_pass_r_w', 6)} {cell('sc_pass_rlog_w', 6)} {cell('sc_pass_ccc_w', 6)} "
                 f"{cell('sc_pass_rmse', 8, 0, 'down')} {cell('sc_sum_ratio', 6, 3)} {cell('route_f1', 7)}  {v.get('time', '')[5:16]}")
        prev = v
    L.append(C("   색: 직전 phase 대비 개선(초록)/악화(빨강). endpoint/SC 지표는 phase 4/6 부터 학습되므로 그 전 값은 baseline.", DIM))
    L.append(sep)
    # 개인차 지표 (재학습 설계안 §2 ⑤). 이번 재학습의 주 목표 지표다.
    rows_id = [ph for ph in d["phases"] if ph["val"] and any(
        ph["val"].get(k) is not None for k in ("abl_own_r", "resid_r", "inter_subj_r", "pair_dice"))]
    if rows_id:
        L.append(C(" 개인차 지표", BOLD) + C("   T1 ablation(자기/남/0) · 잔차 r(LOO 중심화) · "
                  "예측 subject 간 상관(GT 0.897) · pair dice(천장 0.597)", DIM))
        L.append(C(f" {'#':>2} {'phase':16} {'자기T1':>7} {'남T1':>7} {'T1=0':>7} {pad('격차', 7, True)} "
                   f"{'잔차r':>7} {'subj간':>7} {'(GT)':>7} {'dice':>7} {'천장':>7}", DIM))
        prev_id = None
        for ph in rows_id:
            v = ph["val"]

            def c2(k, w, nd=3, better="up"):
                x = v.get(k)
                sx = fnum(x, nd, w)
                if prev_id and prev_id.get(k) is not None and x is not None:
                    good = x > prev_id[k] if better == "up" else x < prev_id[k]
                    return C(sx, GREEN if good else RED)
                return sx
            L.append(f" {ph['n']:>2} {ph['key'][:16]:16} {c2('abl_own_r', 7)} {c2('abl_shuf_r', 7)} "
                     f"{c2('abl_zero_r', 7)} {c2('abl_gap', 7)} {c2('resid_r', 7)} "
                     f"{c2('inter_subj_r', 7, 4, 'down')} {fnum(v.get('inter_subj_r_gt'), 4, 7)} "
                     f"{c2('pair_dice', 7)} {fnum(v.get('pair_dice_ceiling'), 3, 7)}")
            prev_id = v
        L.append(C("   목표: 자기T1 > 남T1 > T1=0 격차가 명확 · 잔차r 0.026 -> 0.20 · "
                   "subj간 0.9995 -> 0.90 쪽으로 · dice 0.059 -> 상승", DIM))
        L.append(sep)

    rows_bt = [ph for ph in d["phases"] if ph["val"] and ph["val"].get("blocks")]
    if rows_bt:
        L.append(C(" Val block / tier 별", BOLD) + C("   block: pair_acc / SC r_log / CCC   ·   tier(GT edge ≤100 / ≤1000 / >1000): pair_acc / SC log-MAE", DIM))
        L.append(C(f" {'#':>2} {'phase':16} {'ctx-ctx':>16} {'ctx-sub':>16} {'sub-sub':>16} {'small':>11} {'mid':>11} {'large':>11}", DIM))
        for ph in rows_bt:
            v = ph["val"]; bl, ti = v["blocks"], v["tiers"]

            def f2(x):
                return "-" if x is None or not math.isfinite(x) else f"{x:.2f}"

            def bcell(b):
                x = bl.get(b, {}); return f"{f2(x.get('pair_acc'))}/{f2(x.get('sc_rlog'))}/{f2(x.get('sc_ccc'))}"

            def tcell(t):
                x = ti.get(t, {}); return f"{f2(x.get('pair_acc'))}/{f2(x.get('sc_log_mae'))}"
            L.append(f" {ph['n']:>2} {ph['key'][:16]:16} {bcell('ctx-ctx'):>16} {bcell('ctx-sub'):>16} {bcell('sub-sub'):>16} {tcell('small'):>11} {tcell('mid'):>11} {tcell('large'):>11}" + (C("  (reval)", DIM) if v.get("reval") else ""))
        L.append(sep)

    # 완료 phase 요약
    done = [ph for ph in d["phases"] if ph["status"] == "done" and ph["n_rows"]]
    if done:
        L.append(C(" 완료 phase 요약", BOLD) + C(f"   (첫 {d['window']} → 마지막 {d['window']} step 평균)", DIM))
        for ph in done:
            parts = [f"{k} {fnum(ph['first_mean'][k], 2)}→{fnum(ph['last_mean'][k], 2)}" for k in LOSS_KEYS if ph["last_mean"].get(k) is not None]
            parts.append(f"gnorm {fnum(ph['first_mean']['grad_norm_total'], 0)}→{fnum(ph['last_mean']['grad_norm_total'], 0)}")
            L.append(f" {ph['n']:>2} {ph['key'][:16]:16} {' · '.join(parts)} · {hms(ph['train_sec'])} · VRAM {fnum(ph['peak_vram_gb'], 1)} GB · 완료 {clock(ph['final_mtime'])}")
        L.append(sep)
    return "\n".join(L)


def pick_pipeline() -> str:
    """lock 이 살아 있는 파이프라인 우선, 없으면 state 가 가장 최근인 것."""
    cands = sorted(ROOT.glob("configs/pipeline*.yaml"))
    best, best_t = None, -1.0
    for c in cands:
        P = yaml.safe_load(c.read_text())
        lock = ROOT / P.get("lock_file", "")
        st = ROOT / P.get("state_file", "")
        if lock.exists() and time.time() - lock.stat().st_mtime < HEARTBEAT_STALE:
            return str(c.relative_to(ROOT))
        t = st.stat().st_mtime if st.exists() else -1.0
        if t > best_t:
            best, best_t = c, t
    return str((best or cands[0]).relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="auto", help="auto = 지금 돌고 있는(또는 가장 최근) 파이프라인 자동 선택")
    ap.add_argument("--watch", type=float, default=0, help="갱신 주기(초). 0 이면 1회")
    ap.add_argument("--window", type=int, default=50, help="first/last 평균 창")
    ap.add_argument("--bins", type=int, default=40, help="추이 sparkline bin 수")
    ap.add_argument("--width", type=int, default=0, help="출력 폭 (0 = 터미널 자동)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    if a.config == "auto":
        a.config = pick_pipeline()
    C.on = not a.no_color and sys.stdout.isatty()
    width = a.width or shutil.get_terminal_size((130, 40)).columns
    while True:
        d = collect(a.config, a.window, a.bins)
        if a.json:
            slim = {k: v for k, v in d.items() if k != "phases"}
            slim["phases"] = [{k: v for k, v in ph.items() if k not in ("series", "last")} for ph in d["phases"]]
            print(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
        else:
            out = render(d, width)
            if a.watch:
                sys.stdout.write("\033[2J\033[H")
            print(out, flush=True)
        if not a.watch:
            break
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
