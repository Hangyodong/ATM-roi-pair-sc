#!/usr/bin/env python
"""9-phase 학습 드라이버. 어디서 실행하든(로컬 A10, PBS job) pipeline_state.json 과 각 phase 의
`*_latest.pt`(모델+optimizer+step+RNG) 에서 이어서 돈다. 두 프로세스가 동시에 돌지 않도록 lock.

  python scripts/19_train_pipeline.py --config configs/pipeline.yaml
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.evaluation.gates import check_gates    # noqa: E402
from atm_sc.training import config as C            # noqa: E402
from atm_sc.training.run import (Preempted, individuality_metrics, recon_rmse_metrics,   # noqa: E402
                                 run, t1_input)


def load_state(p):
    return json.loads(p.read_text()) if p.exists() else {"phase_idx": 0, "done": [], "history": []}


def save_state(p, st):
    p.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def wait_for_preempt_release(preempt: Path, lock: Path, poll: int = 30):
    """preempt 파일이 사라질 때까지 GPU 를 비우고 기다린다 (A10 <-> H100 인계)."""
    import torch as _t
    _t.cuda.empty_cache()
    t0 = time.time()
    while preempt.exists():
        if int(time.time() - t0) % 600 < poll:
            print(f"[pipeline] preempt 대기 {int(time.time() - t0) // 60}분 ({preempt})", flush=True)
        time.sleep(poll)
    print(f"[pipeline] preempt 해제, {int(time.time() - t0) // 60}분 만에 재개", flush=True)


def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    """다른 프로세스가 살아 있으면 False. heartbeat(mtime) 가 stale_sec 이상 멈췄으면 빼앗는다."""
    me = f"{socket.gethostname()}:{os.getpid()}"
    if lock.exists():
        host, pid = lock.read_text().strip().split(":")
        alive = False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive = True
            except OSError:
                alive = False
        if alive or (host != socket.gethostname() and time.time() - lock.stat().st_mtime < stale_sec):
            print(f"lock 보유 중: {host}:{pid} (mtime {time.time()-lock.stat().st_mtime:.0f}s 전) -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


EV_KEYS = ("pair_acc_unordered", "start_roi_acc", "end_roi_acc", "endpoint_dist_to_gt_mm",
           "length_mean_mm", "gt_length_mean_mm", "route_f1", "route_recall", "route_precision")


def _mean_nested(rows):
    """[{k: float | {k2: ...}}] -> key 별 평균 (None/nan 무시)."""
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[k] = _mean_nested(vals)
        elif isinstance(vals[0], (int, float)):
            v = [float(x) for x in vals if np.isfinite(x)]
            out[k] = float(np.mean(v)) if v else float("nan")
    return out


@torch.no_grad()
def validate(ck: Path, val_subs, n_cell, n_per_pair, device, indiv_subs=None, n_indiv=8,
             pair_dice=False, n_dice_pairs=30):
    """phase 종료 시 val subject 몇 명에서
    (1) block(ctx-ctx/ctx-sub/sub-sub) × tier(소/중/대) cell 마다 n_cell pair 를 뽑아 prior-z 생성 정확도,
    (2) whole-brain pass-SC vs GT .mat 를 전체 / block 별 / tier 별로 잰다.
    (이전 'count 상위 64 pair' 는 ctx-ctx 큰 bundle 만 평가했다: 평균 61.7 / 2.2 / 0.0.)
    (3) 개인차 지표 (T1 ablation / 잔차 r / subject 간 상관 / pair dice). 잔차 r 은 subject 가
        여러 명이어야 의미가 있어 indiv_subs(기본 val 전체) 에서 n_indiv 명으로 따로 잰다."""
    import nibabel as nib
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.data.paths import ATLAS, CACHE
    from atm_sc.data.roi_groups import BLOCKS, TIERS, block_masks, tier_masks
    from atm_sc.evaluation.balance_metrics import sc_metrics_extended
    from atm_sc.evaluation.roi_pair_eval import evaluate_pairs
    from atm_sc.inference.generate_sc import generate_tractogram, tractogram_sc
    from atm_sc.losses import sc_group_metrics, sc_metrics
    from atm_sc.models.roi_atm import ROIPairATM, from_checkpoint
    rp = ROOT / "outputs" / "roi_pairs"
    ready = [s for s in val_subs if (CACHE / f"{s}_T1w_syn_W.npy").exists() and (rp / s / "bundles.npz").exists()]
    if not ready:
        return {"note": "전처리된 val subject 없음"}
    # checkpoint 메타(in_channels / template / t1_source)로 모델을 만든다. 직접 ROIPairATM(...)
    # 을 만들면 템플릿 buffer 가 없어 로딩이 죽고(실측), 프로토콜이 어긋나면 오류 없이
    # 결과만 나빠진다 (own r 0.81 -> 0.72).
    m, sd = from_checkpoint(ck, device=device)
    src = sd.get("t1_source", "syn")
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    res = []
    for s in ready:
        subj = ROIPairSubject(s)
        a = m.atm.encode_anatomy(t1_input(m, s, src))
        rng = np.random.default_rng(0)
        cells = {}
        for bi, b in enumerate(BLOCKS):
            for ti, t in enumerate(TIERS):
                idx = np.flatnonzero((subj.pair_block == bi) & (subj.pair_tier == ti))
                if len(idx):
                    cells[(b, t)] = rng.choice(idx, size=min(n_cell, len(idx)), replace=False)
        ev = {c: evaluate_pairs(m, subj, a, atlas, img.affine, idx, n_per_pair) for c, idx in cells.items()}

        def agg(keys):                                    # pair 수 가중 평균
            sub = [ev[c] for c in keys if c in ev]
            n = sum(e["n_pairs"] for e in sub)
            return {k: float(sum(e[k] * e["n_pairs"] for e in sub) / n) for k in EV_KEYS} if n else None
        e_all = agg(list(ev))
        S, w, _ = generate_tractogram(m, a, np.asarray(subj.pair_ids), 8)
        sc = tractogram_sc(S, w, atlas, img.affine, subj.n_roi)
        pred = torch.as_tensor(sc["pass"]["sc_w"], dtype=torch.float32)
        gt = torch.as_tensor(np.asarray(subj.sc_mat, np.float32))
        mm = sc_metrics(pred, gt)
        bm = block_masks(subj.n_roi)
        # SUB-SUB 을 endpoint bundle 이 있는 edge 와 통과로만 생기는 edge 로 나눈다 (ROUTE 전략 §59)
        ss, e_sc, p_sc = bm["sub-sub"], np.asarray(subj.sc_end), np.asarray(subj.sc_mat)
        extra = {"ss_endpoint": ss & (e_sc > 0), "ss_pass_only": ss & (e_sc == 0) & (p_sc > 0)}
        gm = sc_group_metrics(pred, gt, {**bm, **tier_masks(subj.sc_mat)})
        gm.update({k: sc_metrics_extended(pred, gt, torch.as_tensor(v)) for k, v in extra.items() if v.any()})
        gm["all"] = sc_metrics_extended(pred, gt)

        def part(keys, g):
            e = agg(keys) or {}
            d = {"pair_acc": e.get("pair_acc_unordered"), "endpoint_dist_mm": e.get("endpoint_dist_to_gt_mm"),
                 "route_f1": e.get("route_f1"), "route_recall": e.get("route_recall"),
                 "n_pairs": int(sum(len(cells[c]) for c in keys if c in cells)),
                 "sc_r": g["r"], "sc_rlog": g["r_log"], "sc_ccc": g["ccc"], "sc_log_mae": g["log_mae"],
                 "sc_rmse": g.get("rmse"), "sc_mae": g.get("mae"), "n_edges": g.get("n_edges")}
            for k in ("spearman", "weak_edge_recall"):
                if k in g:
                    d[k] = g[k]
            return d
        res.append({"subject": s, "pair_acc": e_all["pair_acc_unordered"], "start_roi_acc": e_all["start_roi_acc"],
                    "end_roi_acc": e_all["end_roi_acc"], "endpoint_dist_mm": e_all["endpoint_dist_to_gt_mm"],
                    "length_mm": e_all["length_mean_mm"], "gt_length_mm": e_all["gt_length_mean_mm"],
                    "sc_pass_r_w": mm["r"], "sc_pass_rlog_w": mm["r_log"], "sc_pass_ccc_w": mm["ccc"],
                    "n_val_pairs": int(sum(len(v) for v in cells.values())),
                    "route_f1": e_all.get("route_f1"), "route_recall": e_all.get("route_recall"),
                    "sc_pass_spearman": gm["all"]["spearman"], "weak_edge_recall": gm["all"]["weak_edge_recall"],
                    "sc_pass_rmse": gm["all"]["rmse"], "sc_pass_mae": gm["all"]["mae"],
                    "sc_sum_ratio": float(pred.sum() / gt.sum().clamp(min=1)),
                    "blocks": {b: part([(b, t) for t in TIERS], gm[b]) for b in BLOCKS},
                    "tiers": {t: part([(b, t) for b in BLOCKS], gm[t]) for t in TIERS},
                    "ss_split": {k: part([], gm[k]) for k in ("ss_endpoint", "ss_pass_only") if k in gm}})
    agg_all = _mean_nested([{k: v for k, v in r.items() if k != "subject"} for r in res])
    t_ind = time.time()
    # subject 사이를 비교하는 지표라 per-subject 평균(_mean_nested)으로 만들 수 없다 -> 따로 넣는다.
    agg_all.update(individuality_metrics(m, indiv_subs or val_subs, n_subj=n_indiv,
                                         pair_dice=pair_dice, n_dice_pairs=n_dice_pairs))
    agg_all["indiv_sec"] = time.time() - t_ind
    # P4: held-out 복원 RMSE 를 **eval 모드와 train 모드 둘 다**. 게이트는 eval 로 건다.
    # 이게 없으면 `recon_rmse_eval_mm` 게이트가 발화할 수 없다 (지표가 아예 안 만들어진다).
    agg_all.update(recon_rmse_metrics(m, ready, n_subj=len(ready), n_per_pair=n_per_pair, source=src))
    agg_all["n_val"] = len(res); agg_all["per_subject"] = res
    return agg_all


def _val_lists(P):
    """(val 전체, block×tier 평가용 상위 val_max_subjects 명)."""
    all_ = [l.strip() for l in (ROOT / P["val_subjects"]).read_text().splitlines() if l.strip()]
    assert all_, f"{P['val_subjects']} 가 비었음"
    return all_, all_[: P.get("val_max_subjects", 3)]


def _indiv_kw(P, val_all):
    """개인차 지표 인자. pair dice 는 실제 생성이 필요해 기본 꺼져 있다."""
    return {"indiv_subs": val_all, "n_indiv": P.get("val_indiv_subjects", 8),
            "pair_dice": bool(P.get("val_pair_dice", False)), "n_dice_pairs": P.get("val_dice_pairs", 30)}


def _num(d, k):
    return isinstance(d.get(k), (int, float)) and not isinstance(d[k], bool) and np.isfinite(d[k])


def _prev_hist(st, phase):
    """직전 phase 의 val 지표 (같은 phase 재평가분은 건너뛴다)."""
    return next((h for h in reversed(st.get("history", [])[:-1]) if h.get("phase") != phase), None)


def _delta_prev(v, st):
    """`<metric>_delta_prev` = 직전 phase 대비 변화량. 회귀 감지 게이트(p4)가 이 키를 본다.
    직전 phase 가 없으면 키도 없다 -> 그런 게이트를 첫 phase 에 걸면 check_gates 가 죽는다 (의도)."""
    prev = _prev_hist(st, v.get("phase"))
    if prev is None:
        return {}
    return {f"{k}_delta_prev": float(v[k]) - float(prev[k]) for k in v if _num(v, k) and _num(prev, k)}


def _trend_line(st, trace, v, names):
    """게이트 지표가 오르는 중인지 내리는 중인지. 한 점만 보면 판정이 반쪽이다
    (resid_r 이 p2 0.124 -> p4 -0.031 로 무너졌는데 phase 내부 궤적이 없어 시점을 모른다)."""
    parts, phase = [], v.get("phase")
    prev = _prev_hist(st, phase)
    rows = []
    if trace.exists():
        rows = [r for r in (json.loads(l) for l in trace.read_text().splitlines() if l.strip())
                if r.get("phase") == phase]
    for k in dict.fromkeys(names):
        if not _num(v, k):
            continue
        if prev is not None and _num(prev, k):
            d = float(v[k]) - float(prev[k])
            parts.append(f"{k} {prev['phase']} {prev[k]:+.4f} -> {v[k]:+.4f} "
                         f"({d:+.4f} {'상승' if d > 0 else '하강' if d < 0 else '동일'})")
        hit = [r for r in rows if _num(r, k)]
        if len(hit) >= 2:
            d = float(hit[-1][k]) - float(hit[0][k])
            parts.append(f"{k} phase 내부 step {hit[0]['step']}->{hit[-1]['step']} "
                         f"{hit[0][k]:+.4f}->{hit[-1][k]:+.4f} ({d:+.4f})")
    return "[GATE] 추세: " + " | ".join(parts) if parts else ""


def run_gates(P, phase, v, st, trace):
    """phase 게이트 판정 + 추세 한 줄. 지표 키가 없거나 NaN 이면 check_gates 가 즉시 죽는다
    (조용히 건너뛰는 것이 지금 고치는 버그다)."""
    gates = (P.get("gates") or {}).get(phase)
    if not gates:
        print(f"[GATE] {phase}: 설정된 게이트 없음", flush=True)
        return True, []
    ok, msgs = check_gates({**v, **_delta_prev(v, st)}, gates)
    tl = _trend_line(st, trace, v, [str(g["metric"]).replace("_delta_prev", "") for g in gates])
    for s in msgs:
        print(s, flush=True)
    if tl:
        print(tl, flush=True)
        msgs = msgs + [tl]
    return ok, msgs


def _failed(msgs):
    """미달한 halt 게이트만. 판정은 메시지 첫 줄에 있다 (둘째 줄부터는 근거 note)."""
    head = [(m, m.split("\n")[0]) for m in msgs]
    return [m for m, h in head if h.startswith("[GATE:halt]") and "-> PASS" not in h]


def indiv_hook(P, val_all, trace, phase, t1_source):
    """(hook, every). 학습 중 every step 마다 개인차 지표만 따로 기록한다.

    phase 끝 한 점(5 phase = 5점/15,000 step)으로는 미학습과 평탄역을 구분할 수 없다.
    `indiv_sec` 이 10초라 500 step 간격이면 phase(2h10m) 대비 0.8% 다. 계측 전용이라
    no_grad + RNG 상태 복원 + train/eval 모드 복원으로 학습 상태를 바꾸지 않는다.
    출력은 `val_metrics.jsonl` 과 분리한다 (다른 소비자가 그 스키마를 읽는다)."""
    every = int(P.get("indiv_log_every", 0) or 0)
    if every <= 0:
        return None, 0
    kw = _indiv_kw(P, val_all)
    subs, n_indiv = kw["indiv_subs"], kw["n_indiv"]

    def hook(step, model):
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        t0 = time.time()
        try:
            with torch.no_grad():
                m = individuality_metrics(model, subs, n_subj=n_indiv, pair_dice=False, source=t1_source)
        finally:                                  # 학습 상태 원복 (계측이 다음 step 을 바꾸면 안 된다)
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
            model.train(was_training)
        if not m:                                 # count head 가 없는 phase (p0) -- 지표 자체가 없다
            print(f"[indiv] {phase} step {step}: count head 없음 -> 개인차 지표 없음", flush=True)
            return
        assert "resid_r" in m, f"개인차 지표에 resid_r 이 없다: {sorted(m)}"
        row = {"phase": phase, "step": int(step), "time": time.strftime("%F %T"),
               "indiv_sec": time.time() - t0,
               **{k: m[k] for k in ("resid_r", "abl_own_r", "abl_shuf_r", "abl_zero_r", "abl_gap",
                                    "inter_subj_r", "n_indiv_subj") if k in m}}
        with open(trace, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[indiv] {phase} step {step} " +
              " ".join(f"{k}={row[k]:+.4f}" for k in ("resid_r", "abl_gap", "inter_subj_r") if k in row) +
              f" (n={row.get('n_indiv_subj')}, {row['indiv_sec']:.0f}s)", flush=True)

    return hook, every


def val_line(v):
    s = f"[pipeline] val ({v.get('n_val', 0)}명): " + " ".join(f"{k}={v[k]:.4f}" for k in ("pair_acc", "sc_pass_r_w", "sc_pass_rlog_w", "sc_pass_ccc_w", "length_mm") if k in v)
    if v.get("blocks"):
        s += " | block r_log " + " ".join(f"{b}={x.get('sc_rlog', float('nan')):.2f}" for b, x in v["blocks"].items())
        s += " | tier pair_acc " + " ".join(f"{t}={x.get('pair_acc') if x.get('pair_acc') is None else round(x['pair_acc'], 3)}" for t, x in v["tiers"].items())
    if v.get("route_f1") is not None:
        s += f" | route F1={v['route_f1']:.3f}"
    if v.get("sc_pass_rmse") is not None:
        s += f" | RMSE={v['sc_pass_rmse']:.0f} 합비={v.get('sc_sum_ratio', float('nan')):.3f}"
    if v.get("ss_split"):
        s += " | SS " + " ".join(f"{k.replace('ss_', '')} r_log={x['sc_rlog']:.2f}" for k, x in v["ss_split"].items())
    if v.get("abl_own_r") is not None:
        # own > shuf > zero 가 뚜렷해야 T1 을 읽고 있는 것이다 (PIPELINE_08_RETRAIN_DESIGN.md §2 ⑤)
        s += (f"\n[pipeline] 개인차 ({v.get('n_indiv_subj', 0)}명): T1 abl own={v['abl_own_r']:.4f} "
              f"shuf={v['abl_shuf_r']:.4f} zero={v['abl_zero_r']:.4f} gap={v['abl_gap']:+.4f}"
              f" | resid_r={v['resid_r']:+.4f} | inter_subj r={v['inter_subj_r']:.4f} "
              f"(GT {v['inter_subj_r_gt']:.4f})")
        if v.get("pair_dice") is not None:
            s += f" | pair dice={v['pair_dice']:.3f} / 천장 {v['pair_dice_ceiling']:.3f}"
        s += f" ({v.get('indiv_sec', float('nan')):.0f}s)"
    return s


def main(a):
    P = C.load(a.config)
    state_p, lock = ROOT / P["state_file"], ROOT / P["lock_file"]
    state_p.parent.mkdir(parents=True, exist_ok=True)
    trace = state_p.parent / "indiv_trace.jsonl"
    if a.reval is not None:                                   # 기존 checkpoint 를 새 val 로 다시 평가 (lock/state 불변)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        val_all, val_subs = _val_lists(P)
        st = load_state(state_p)                              # 게이트 판정용(추세). 저장하지 않는다.
        for ck in a.reval:
            ck = ROOT / ck
            v = validate(ck, val_subs, P.get("val_pairs_per_cell", 8), P.get("val_n_per_pair", 16), dev,
                         **_indiv_kw(P, val_all))
            v.update(phase=ck.stem.split("_step")[0].split("_latest")[0], checkpoint=str(ck.relative_to(ROOT)),
                     time=time.strftime("%F %T"), reval=True)
            with open(state_p.parent / "val_metrics.jsonl", "a") as f:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
            print(f"[reval] {ck.name} " + val_line(v), flush=True)
            try:                                              # reval 은 판정 결과를 출력만 한다 (중단 없음)
                ok, _ = run_gates(P, v["phase"], v, st, trace)
                print(f"[reval] 게이트 {'통과' if ok else '미달'} (reval 이라 진행/중단에는 영향 없음)", flush=True)
            except (AssertionError, KeyError) as e:
                print(f"[reval] 게이트 판정 불가: {type(e).__name__}: {e}", flush=True)
        return 0
    if not acquire_lock(lock):
        return 3
    try:
        st = load_state(state_p)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        val_all, val_subs = _val_lists(P)
        phases = P["phases"]
        preempt = ROOT / P["preempt_file"] if P.get("preempt_file") else None
        assert not (a.no_val and P.get("gates") and not a.ignore_gates), (
            "게이트가 설정된 config 를 --no-val 로 돌릴 수 없다 (판정 지표가 안 만들어진다). "
            "정말 게이트 없이 돌리려면 --ignore-gates 를 같이 준다.")
        idx = st["phase_idx"]
        while idx < len(phases):
            kw = C.build(C.load(ROOT / phases[idx]))
            if a.max_steps:
                kw["max_steps"] = min(kw["max_steps"], a.max_steps)
            latest = kw["out_dir"] / f"{kw['phase']}_latest.pt"
            final = kw["out_dir"] / f"{kw['phase']}_step{kw['max_steps']}.pt"
            if final.exists():
                ck = final; print(f"[pipeline] phase {idx} ({kw['phase']}) 이미 완료: {ck.name}", flush=True)
            else:
                if latest.exists():
                    kw["resume"] = latest                          # 같은 phase 이어받기
                elif idx > 0 and st.get("last_ckpt"):
                    kw["resume"] = ROOT / st["last_ckpt"]          # 이전 phase 최종 가중치
                kw["heartbeat"] = lock
                kw["preempt_file"] = preempt
                kw["step_hook"], kw["step_hook_every"] = indiv_hook(P, val_all, trace, kw["phase"],
                                                                    kw.get("t1_source", "rigid"))
                torch.manual_seed(kw["cfg"].seed)
                print(f"[pipeline] phase {idx} ({kw['phase']}) 시작 {socket.gethostname()} resume={kw.get('resume')}", flush=True)
                try:
                    ck = run(**kw)
                except Preempted:
                    save_state(state_p, st)
                    if lock.exists() and lock.read_text().strip() == f"{socket.gethostname()}:{os.getpid()}":
                        lock.unlink()
                    wait_for_preempt_release(preempt, lock)
                    if not acquire_lock(lock):          # 인계받은 쪽이 아직 돌고 있으면 종료
                        return 3
                    continue                            # 같은 phase 를 latest 에서 다시 이어받는다
                
            st["last_ckpt"] = str(ck.relative_to(ROOT))
            if not a.no_val:
                v = validate(ck, val_subs, P.get("val_pairs_per_cell", 8), P.get("val_n_per_pair", 16), dev,
                             **_indiv_kw(P, val_all))
                v.update(phase=kw["phase"], checkpoint=st["last_ckpt"], time=time.strftime("%F %T"))
                st["history"].append({k: v[k] for k in v if k != "per_subject"})
                with open(state_p.parent / "val_metrics.jsonl", "a") as f:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
                print(val_line(v), flush=True)
                ok, msgs = run_gates(P, kw["phase"], v, st, trace)
                st["history"][-1]["gates"] = msgs
                if not ok and not a.ignore_gates:
                    # phase_idx 를 올리지 않는다: 재시작해도 같은 지점에서 다시 막힌다.
                    st["gate_halt"] = {"phase": kw["phase"], "checkpoint": st["last_ckpt"],
                                       "time": time.strftime("%F %T"), "failed": _failed(msgs)}
                    save_state(state_p, st)
                    print(f"[pipeline] 게이트 미달 -> phase {idx} ({kw['phase']}) 에서 중단한다. 다음 phase 로 넘어가지 않는다.",
                          flush=True)
                    for m in _failed(msgs):
                        print(f"[pipeline]   {m}", flush=True)
                    print(f"[pipeline] 기록: {state_p.relative_to(ROOT)} 의 gate_halt. "
                          f"기준을 알면서 강행하려면 --ignore-gates.", flush=True)
                    return 4
                if not ok:
                    print("!" * 78 + f"\n[pipeline] !! --ignore-gates: {kw['phase']} 의 게이트가 미달인데 강행한다.", flush=True)
                    for m in _failed(msgs):
                        print(f"[pipeline] !!   {m}", flush=True)
                    print("[pipeline] !! 이후 phase 의 결과는 기준 미달 위에서 나온 값이다.\n" + "!" * 78, flush=True)
                    st["gate_ignored"] = st.get("gate_ignored", []) + [{"phase": kw["phase"], "failed": _failed(msgs)}]
            st.pop("gate_halt", None)
            st["phase_idx"] = idx + 1
            if kw["phase"] not in st["done"]:
                st["done"].append(kw["phase"])
            idx += 1
            save_state(state_p, st)
        print("[pipeline] ALL PHASES DONE", flush=True)
        if P.get("final_eval") and st.get("last_ckpt"):
            # 마지막 단계가 끝나면 test set 을 T1 만으로 1회 평가한다 (§15 필수 evaluation).
            import subprocess
            cmd = [sys.executable, str(ROOT / "scripts" / "29_final_evaluation.py"),
                   "--ckpt", st["last_ckpt"], "--subjects", P.get("test_subjects", "outputs/splits/test.txt")]
            print(f"[pipeline] 최종 test 평가: {' '.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=False)
        return 0
    finally:
        if lock.exists() and lock.read_text().strip() == f"{socket.gethostname()}:{os.getpid()}":
            lock.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="phase 별 step 상한 (resume 테스트용)")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--ignore-gates", action="store_true",
                    help="게이트 미달에도 다음 phase 로 진행 (기본은 강제 중단). 로그에 크게 경고가 남는다")
    ap.add_argument("--reval", nargs="*", default=None, help="checkpoint 들을 새 val(block×tier) 로 다시 평가해 val_metrics.jsonl 에 추가")
    sys.exit(main(ap.parse_args()))
