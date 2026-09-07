#!/usr/bin/env python
"""D3 joint — 복원 + 생성 제약 + 조건부 prior 를 동시에 학습 (W3-a).

  python scripts/49_d3_joint.py --selfcheck-only              # patch 검증만 (GPU 잠깐)
  python scripts/49_d3_joint.py --sweep A,B,C --sweep-steps 800   # 손실 가중치 탐색
  python scripts/49_d3_joint.py --config configs/retrain/d3_joint.yaml

무엇을 고치나
-------------
D1 이 복원만 학습해 디코더를 고쳤지만(eval recon 7.721 -> 2.240 mm) **생성 경로가 무너졌다**
(C13: valid_conn 0.367 -> 0.042). posterior drift 가 아니라(오프셋 5.27 -> 3.51 로 감소),
prior 가 뽑는 영역이 더 이상 디코더가 정확한 영역이 아니게 된 것이다. 그래서 복원과 생성
제약을 **동시에** 돌리고, 조건부 prior 를 posterior 평균 쪽으로 따로 적합시킨다.

이 스크립트가 반드시 확인하는 것 (실패하면 학습 전에 죽는다)
  1. prior 0-init 에서 `sample_prior` == `prior_mean + randn` (구 checkpoint bit-exact)
  2. `kl_loss(logvar_prior=None)` 이 기존 식과 일치, `logvar_prior=0` 도 일치
  3. KL 에 들어가는 prior 파라미터가 detach 되어 있다 (C11 재발 방지)
  4. recon RMSE 를 train/eval 두 모드 다 기록 (8.46 vs 3.55 착시 방지)
"""
import argparse
import inspect
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
from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402
from atm_sc.data.paths import ATLAS                                     # noqa: E402
from atm_sc.evaluation.gates import check_gates                         # noqa: E402
from atm_sc.evaluation.reproduction_metrics import valid_connection_rate  # noqa: E402
from atm_sc.losses import geometry as G                                 # noqa: E402
from atm_sc.models import latent_prior as lp                            # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                       # noqa: E402
from atm_sc.models.roi_pair_embedding import ROIPairEmbedding, canonical_pairs  # noqa: E402
from atm_sc.training import config as C                                 # noqa: E402
from atm_sc.training.run import anatomy_feature, recon_rmse_metrics, roi_feats_if_needed, run  # noqa: E402
from atm_sc.training.trainer import Trainer                             # noqa: E402

LOCK = ROOT / "outputs/gpu.lock"


# ------------------------------------------------------------------ GPU lock (19번과 같은 규약)
def acquire_lock(lock: Path, stale_sec: int = 900) -> bool:
    me = f"{socket.gethostname()}:{os.getpid()}"
    if lock.exists():
        txt = lock.read_text().strip()
        parts = txt.split(":")
        host, pid = (parts[0], parts[-1]) if len(parts) >= 2 else (txt, "-1")
        alive = False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0); alive = True
            except (OSError, ValueError):
                alive = not pid.lstrip("-").isdigit()
        if alive or (host != socket.gethostname() and time.time() - lock.stat().st_mtime < stale_sec):
            print(f"lock 보유 중: {txt} -> 종료", flush=True)
            return False
        print(f"stale lock 인수: {host}:{pid}", flush=True)
    lock.write_text(me)
    return True


def release_lock(lock: Path) -> None:
    if lock.exists() and lock.read_text().strip().endswith(f":{os.getpid()}"):
        lock.unlink()


# --------------------------------------------------------------------------- 필수 자기검증
def selfcheck(ckpt: Path | None, device: str) -> dict:
    out = {}

    # (1) prior 0-init bit-exact -- 새 모듈만으로
    torch.manual_seed(0)
    e = ROIPairEmbedding(82, 64, 512, latent_dim=64)
    with torch.no_grad():
        e.prior_mu.weight.normal_(std=0.2); e.prior_mu.bias.normal_(std=0.2)
    _iu0 = torch.as_tensor(np.stack(np.triu_indices(82, 1), 1))   # i == j (self-edge) 를 피한다
    p = canonical_pairs(_iu0[torch.randperm(len(_iu0))[:128]])
    mu, ls = e.prior_params(p)
    assert torch.equal(ls, torch.zeros_like(ls)), "prior_log_sigma 가 0-init 이 아니다"
    ga = torch.Generator().manual_seed(7); gb = torch.Generator().manual_seed(7)
    old = e.prior_mean(p) + torch.randn(128, 64, generator=ga)
    new = e.sample_prior(p, generator=gb)
    assert torch.equal(old, new), "0-init 인데 sample_prior 가 bit-exact 하지 않다"
    out["module_bit_exact"] = True

    # (2) kl_loss 등가성
    mu_q, lv_q = torch.randn(64, 64), torch.randn(64, 64) * 0.5
    mu_p = torch.randn(64, 64)
    ref = (-0.5 * (1 + lv_q - (mu_q - mu_p).pow(2) - lv_q.exp()).sum(1)).mean()
    a1 = G.kl_loss(mu_q, lv_q, mu_p)                       # logvar_prior=None -> 기존 경로
    a2 = G.kl_loss(mu_q, lv_q, mu_p, torch.zeros_like(lv_q))
    assert torch.equal(ref, a1), (float(ref), float(a1))   # 같은 코드 경로 -> bit-exact
    assert float((ref - a2).abs()) < 1e-3, (float(ref), float(a2))
    ref0 = (-0.5 * (1 + lv_q - mu_q.pow(2) - lv_q.exp()).sum(1)).mean()
    assert torch.equal(ref0, G.kl_loss(mu_q, lv_q)), "mu_prior=None 경로가 바뀌었다"
    out.update(kl_old=float(ref), kl_new_logvar_prior0=float(a2), kl_none_branch_bit_exact=True)

    # (3) KL 이 prior 를 학습시키지 않는가 (C11). trainer 가 쓰는 식 그대로 재현한다.
    src = inspect.getsource(Trainer.step)
    assert "L.kl_loss(mu, logvar, mu_p.detach(), 2.0 * ls_p.detach())" in src, \
        "trainer 의 KL 이 detach 된 prior 를 쓰지 않는다 (C11 재발)"
    assert "L.kl_loss(mu, logvar, mu_ps.detach(), 2.0 * ls_ps.detach())" in src, "segment KL 도 detach 필요"
    mu_pp, ls_pp = e.prior_params(p)
    zq = torch.randn(128, 64, requires_grad=True)           # posterior 쪽으로는 gradient 가 흘러야 한다
    l_kl = G.kl_loss(zq, torch.zeros(128, 64), mu_pp.detach(), 2.0 * ls_pp.detach())
    g_kl = torch.autograd.grad(l_kl, [zq, e.prior_mu.weight, e.prior_log_sigma.weight],
                               allow_unused=True)
    assert g_kl[0] is not None and float(g_kl[0].abs().sum()) > 0, "KL gradient 자체가 죽었다"
    assert all(g is None for g in g_kl[1:]), "KL 에서 prior 파라미터로 gradient 가 샌다"
    # 같은 자리에서 적합항은 gradient 가 흘러야 한다 (그래야 prior 가 학습된다)
    l_fit = -lp.diag_log_prob(torch.randn(128, 64), mu_pp, ls_pp).mean()
    g_fit = torch.autograd.grad(l_fit, [e.prior_mu.weight, e.prior_log_sigma.weight], allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0 for g in g_fit), \
        "prior 적합항이 prior 파라미터를 학습시키지 못한다"
    out["kl_prior_detached"] = True

    # (4) 실제 checkpoint 에서 sample_z 가 예전 식과 bit-exact 인가
    if ckpt is not None:
        m, _ = from_checkpoint(ckpt, device=device)
        pe = m.pair_emb
        assert not pe.prior_use_anatomy, "prior_use_anatomy 가 켜져 있다 (0-init 로 꺼 두기로 했다)"
        # randint 두 번은 i == j 를 만든다 (self-edge). upper triangle 에서 뽑아야 한다.
        _iu = torch.as_tensor(np.stack(np.triu_indices(m.n_roi, 1), 1), device=m.device)
        pp = canonical_pairs(_iu[torch.randperm(len(_iu), device=m.device)[:256]])
        # 진단이라 국소 입력은 0 으로 둔다. 아래 모든 비교가 **같은** 입력을 쓰게 하려면
        # 여기서 한 번 만들어 전부에 넘겨야 한다 (한쪽만 다르면 비교 자체가 성립 안 한다).
        pl0 = (torch.zeros(pp.shape[0], pe.prior_local_dim, device=m.device)
               if pe.prior_local is not None else None)
        ls_ck = pe.prior_log_std(pp, local=pl0)
        g1 = torch.Generator(device=m.device); g1.manual_seed(11)
        g2 = torch.Generator(device=m.device); g2.manual_seed(11)
        z_old = m.prior_mean(pp) + torch.randn(256, 64, device=m.device, generator=g1)
        z_new = m.sample_z(pp, g2, local=pl0)
        # 학습 중 생성 경로가 쓰는 식도 같은지 (trainer.gen_chunk 의 zc)
        e_ = torch.randn(256, 64, device=m.device)
        mu_ck, ls2 = m.prior_params(pp, local=pl0)
        z_gen_old = m.prior_mean(pp) + e_
        z_gen_new = mu_ck + torch.exp(ls2.detach()) * e_
        out["ckpt_prior_log_sigma_absmax"] = float(ls_ck.abs().max())
        # log_sigma 가 0 이면 구 checkpoint 라 `mu + eps` 와 bit-exact 여야 한다. 하지만 D3 가
        # 실제로 돌면 prior scale 이 학습돼 0 이 아니게 된다 -- 그때 bit-exact 를 요구하면
        # "D3 를 돌렸다는 이유로 D3 후속이 못 돌아가는" 낡은 검증이 된다. 실측: d3_joint_step3000
        # 의 |log_sigma|max = 4.30 (평균 sigma 0.36). 그래서 조건을 나눈다.
        untrained = out["ckpt_prior_log_sigma_absmax"] == 0.0
        out["ckpt_prior_untrained"] = bool(untrained)
        out["ckpt_gen_z_bit_exact"] = bool(torch.equal(z_gen_old, z_gen_new))
        out["ckpt_sample_z_bit_exact"] = bool(torch.equal(z_old, z_new))
        if untrained:
            assert out["ckpt_gen_z_bit_exact"], "0-init 인데 생성 경로 z 가 달라졌다"
            assert out["ckpt_sample_z_bit_exact"], "구 checkpoint 에서 sample_z 가 달라졌다"
        else:
            # 학습된 prior 에서는 sample_z 가 **현행 식**과 일치하는지를 본다 (이게 진짜 불변식).
            g3 = torch.Generator(device=m.device); g3.manual_seed(11)
            g4 = torch.Generator(device=m.device); g4.manual_seed(11)
            z_ref = mu_ck + torch.exp(ls2) * torch.randn(256, 64, device=m.device, generator=g3)
            out["ckpt_sample_z_matches_formula"] = bool(torch.equal(
                z_ref, m.sample_z(pp, g4, local=pl0)))
            assert out["ckpt_sample_z_matches_formula"], \
                "학습된 prior 인데 sample_z 가 mu + exp(log_sigma)*eps 와 다르다"
        del m
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------ 생성 경로 빠른 진단
def gen_probe(model, subs: list[str], atlas, affine, source: str, n_pairs: int = 256,
              n_per_pair: int = 8, seed: int = 0) -> dict:
    """생성 가닥이 의도한 ROI 쌍을 실제로 잇는가 (valid_conn) — 탐색용 축약판.

    최종 판정은 `scripts/29_final_evaluation.py` 로 한다. 여기서는 pair 선택을 GT 양성 pair 의
    tier 균등 표본으로 고정해 빠르게(수 초) 상대 비교만 한다.
    """
    was = model.training
    model.eval()
    acc = []
    for sub in subs:
        s = ROIPairSubject(sub)
        a = anatomy_feature(model, sub, "AF_L", source=source)
        rng = np.random.default_rng(seed)
        ks = s.sample_pairs(min(n_pairs, len(s.pair_ids)), rng, mode="tier")
        P = torch.as_tensor(np.asarray(s.pair_ids)[ks], device=model.device)
        g = torch.Generator(device=model.device); g.manual_seed(seed)
        S, _, pr = model.generate(a, P, n_per_pair, generator=g,
                                  local_roi=roi_feats_if_needed(model, sub, source))
        Sn = S.detach().cpu().numpy()
        assert np.isfinite(Sn).all(), "생성 좌표에 NaN/Inf"
        acc.append(valid_connection_rate(Sn, pr.detach().cpu().numpy(), atlas, affine, s.n_roi))
    model.train(was)
    return {k: float(np.mean([r[k] for r in acc])) for k in acc[0]} | {"probe_n_subj": len(acc)}


def measure(ckpt: Path, val_subs, dev, atlas, affine, n_val, n_val_pairs, n_per_pair,
            probe_subs, probe_pairs) -> dict:
    m, sd = from_checkpoint(ckpt, device=dev)
    src = sd.get("t1_source", "syn")
    out = {"ckpt": str(ckpt.relative_to(ROOT)),
           "meta": {k: str(sd.get(k)) for k in ("phase", "step", "unet_level", "t1_source")}}
    out.update(recon_rmse_metrics(m, val_subs, n_subj=n_val, n_pairs=n_val_pairs,
                                  n_per_pair=n_per_pair, seed=0, source=src))
    out.update(gen_probe(m, probe_subs, atlas, affine, src, n_pairs=probe_pairs))
    gp = torch.Generator(device="cpu"); gp.manual_seed(0)
    _iu1 = torch.as_tensor(np.stack(np.triu_indices(m.n_roi, 1), 1))
    pp = canonical_pairs(_iu1[torch.randperm(len(_iu1), generator=gp)[:512]].to(m.device))
    with torch.no_grad():
        ls = m.pair_emb.prior_log_std(pp)
    out["prior_log_sigma_mean"] = float(ls.mean())
    out["prior_sigma_mean"] = float(ls.exp().mean())
    del m
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------------- 최종 판정 표
# `scripts/29_final_evaluation.py` 는 W3-b 담당이라 **읽기만** 한다. 전/후 두 checkpoint 를
# 같은 규약으로 돌린 뒤 그 JSON 두 개를 여기서 표로 합친다.
TABLE = [
    ("recon RMSE (eval, test 31)", ("oracle", "recon_rmse_eval_mm"), "down"),
    ("오라클 wb dice (wb_disjoint)", ("oracle", "wb_disjoint", "dice"), "up"),
    ("오라클 pair dice (pair_half)", ("oracle", "pair_half", "dice"), "up"),
    ("생성 wb dice", ("trk_geometry", "dice"), "up"),
    ("생성 wb coverage", ("trk_geometry", "coverage"), "up"),
    ("생성 pair dice", ("trk_per_pair", "dice"), "up"),
    ("생성 pair mdf_mm", ("trk_per_pair", "mdf_mm"), "down"),
    ("valid_conn", ("connection", "valid_conn"), "up"),
    ("endpoint_in_roi", ("connection", "endpoint_in_roi"), "up"),
    ("생성 SC pass r", ("generated", "all", "r"), "up"),
    ("생성 SC r_log", ("generated", "all", "r_log"), "up"),
    ("SC tier r small", ("primary", "tier_r", "small"), "up"),
    ("SC tier r mid", ("primary", "tier_r", "mid"), "up"),
    ("SC tier r large", ("primary", "tier_r", "large"), "up"),
    ("생성 길이 mean_mm", ("trk", "length_mean_mm"), "flat"),
    ("resid_r (generated)", ("primary", "residual_r_generated"), "up"),
]
GATE_KEYS = {"recon_rmse_eval_mm": ("oracle", "recon_rmse_eval_mm"),
             "valid_conn": ("connection", "valid_conn"),
             "gen_pair_dice": ("trk_per_pair", "dice"),
             "gen_wb_dice": ("trk_geometry", "dice")}


def dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def assemble(a) -> int:
    cfg_raw = C.load(ROOT / a.config)
    before = json.loads((ROOT / "outputs/eval" / a.before_json).read_text())["summary"]
    after = json.loads((ROOT / "outputs/eval" / a.after_json).read_text())["summary"]
    assert before["n_subjects"] == after["n_subjects"], (before["n_subjects"], after["n_subjects"])
    assert before["split"] == after["split"], "전후가 같은 split 이어야 한다"
    assert before["n_per_pair"] == after["n_per_pair"], "전후가 같은 생성 규약이어야 한다"
    rows = []
    for name, path, want in TABLE:
        b, f = dig(before, path), dig(after, path)
        d = None if (b is None or f is None) else f - b
        rows.append({"metric": name, "before": b, "after": f, "delta": d,
                     "better": want, "improved": None if d is None or want == "flat"
                     else bool(d > 0) == (want == "up")})
        print(f"  {name:34s} {b if b is None else round(b, 4)!s:>10} -> "
              f"{f if f is None else round(f, 4)!s:>10}", flush=True)
    # prior 지표 한 줄 (별도 스크립트 결과가 있으면). 낮을수록 좋다.
    ps = ROOT / "outputs/eval/w3a_prior_score_w1e.json"
    if ps.exists():
        pj = json.loads(ps.read_text())["checkpoints"]
        bs, fs = Path(before["checkpoint"]).stem, Path(after["checkpoint"]).stem
        if bs in pj and fs in pj:
            b = dig(pj[bs], ("ladder", "powered", "learned", "precision_ratio"))
            f = dig(pj[fs], ("ladder", "powered", "learned", "precision_ratio"))
            rows.append({"metric": "prior precision_ratio (powered, 학습된 prior)", "before": b,
                         "after": f, "delta": f - b, "better": "down", "improved": bool(f < b),
                         "note": ("같은 스크립트로 잰 p4_joint = "
                                  f"{dig(pj.get('p4_joint_step3000', {}), ('ladder', 'powered', 'learned', 'precision_ratio'))}"
                                  " (W2-b 의 36.61 재현). 오라클 arch_additive 목표는 15.60.")})
            print(f"  {rows[-1]['metric']:34s} {round(b, 4)!s:>10} -> {round(f, 4)!s:>10}", flush=True)
    metrics = {k: dig(after, p) for k, p in GATE_KEYS.items()}
    metrics["n_val"] = after["n_subjects"]
    gates = (cfg_raw.get("gates") or {}).get(cfg_raw["phase"]) or []
    ok, msgs = check_gates(metrics, gates)
    for m in msgs:
        print(f"[gate] {m}", flush=True)
    res = {"cmd": f"python scripts/49_d3_joint.py --assemble --before-json {a.before_json} "
                  f"--after-json {a.after_json}",
           "reproduce": [
               "python scripts/49_d3_joint.py --sweep A,B,C --sweep-steps 500 --sweep-only",
               "python scripts/49_d3_joint.py --resume-ckpt outputs/checkpoints/retrain/d3_sweep_A/"
               "d3_joint_step500.pt --out w3a_joint_train.json",
               "python scripts/49_d3_joint.py --resume-ckpt outputs/checkpoints/retrain/d3_joint/"
               "d3_joint_step3000.pt --max-steps 6000 --out w3a_joint_train6k.json",
               "python scripts/49_eval_frozen.py --ckpt outputs/checkpoints/retrain/d3_joint/"
               "d3_joint_step3000.pt --trk-eval --oracle",
               "python scripts/50_prior_score.py --n-subj 14 --pairs-per-subject 20 "
               "--ckpt outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt "
               "--ckpt outputs/checkpoints/retrain/d1_decoder/d1_decoder_step6000.pt "
               "--ckpt outputs/checkpoints/retrain/d3_joint/d3_joint_step3000.pt "
               "--out w3a_prior_score_w1e.json",
               "python scripts/49_d3_joint.py --assemble --after-json final_d3_joint_step3000.json "
               "--out w3a_joint_result.json"],
           "before_json": a.before_json, "after_json": a.after_json,
           "before_ckpt": before["checkpoint"], "after_ckpt": after["checkpoint"],
           "eval_protocol": {k: before[k] for k in ("split", "n_subjects", "n_per_pair", "edge_thr")} | {
               "script": "scripts/49_eval_frozen.py",
               "note": ("29_final_evaluation.py 의 2026-09-07 00:19 스냅샷(md5 2e8c5169...). 그 버전이 "
                        "before(final_d1_decoder_step6000.json)를 만들었고 그 뒤 W3-b 가 pair-dice "
                        "규약(C9 크기 일치)을 바꿨다. 전/후를 같은 규약으로 재려고 얼린 것이다.")},
           "table": rows, "gate_metrics": metrics,
           "gates": {"passed": bool(ok), "messages": msgs}}
    for extra, key in (("w3a_joint_train.json", "train_3000"), ("w3a_joint_train6k.json", "train_6000"),
                       ("w3a_sweep.json", "sweep"), ("w3a_prior_score.json", "prior_val8"),
                       ("w3a_prior_score_w1e.json", "prior_w1e_protocol"),
                       ("w3a_joint_result_6k.json", "after_6000_table")):
        fp = ROOT / "outputs/eval" / extra
        if fp.exists():
            res[key] = json.loads(fp.read_text())
    out = ROOT / "outputs/eval" / a.out
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n저장: {out.relative_to(ROOT)}", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------ 탐색 arm
# 축: 복원 대 생성 제약. p4_joint 가중치(A)가 valid_conn 0.367 을 냈던 구성이다.
# 20 step 스모크에서 본 것: eval recon 이 2.946 -> 4.155 mm 로 곧장 나빠졌고 valid_conn 은
# 0.0344 -> 0.0359 로 거의 안 움직였다. 그 시점 L_seg_recon 이 9.60 (mode-1 segment 분기)로
# recon(2.93)의 3배였다 -- D1 이 mode 0 만 6,000 step 학습해 mode-1 이 멀어져 있기 때문이다.
# 그래서 축을 (복원 보호) 와 (segment 분기 하향) 로 잡는다.
ARMS = {
    "A": {"note": "p4_joint 그대로 + prior 0.1 (기준)", "loss": {}},
    "B": {"note": "복원 보호: recon 3.0", "loss": {"recon": 3.0}},
    "C": {"note": "segment 분기 하향: seg_* 0.25 (보고 지표는 mode 0 생성이다)",
          "loss": {"seg_recon": 0.25, "seg_kl": 0.025, "seg_geom": 0.25, "seg_endpoint": 0.125}},
}


def main(a) -> int:
    if a.assemble:
        return assemble(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import nibabel as nib
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    cfg_raw = C.load(ROOT / a.config)
    kw0 = C.build(cfg_raw)
    resume = kw0["resume"]
    assert resume is not None and resume.exists(), f"resume checkpoint 가 없다: {resume}"
    val_subs = [l.strip() for l in (ROOT / a.val_subjects).read_text().splitlines() if l.strip()]
    assert val_subs, a.val_subjects
    probe_subs = val_subs[: a.n_probe]

    print("[d3] selfcheck ...", flush=True)
    sc = selfcheck(resume, dev)
    print(f"[d3] selfcheck OK: {json.dumps(sc)}", flush=True)
    if a.selfcheck_only:
        (ROOT / "outputs/eval/w3a_selfcheck.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False))
        return 0

    t_all = time.time()
    before = measure(resume, val_subs, dev, atlas, img.affine, a.n_val, a.n_val_pairs,
                     kw0["cfg"].n_gt_per_pair, probe_subs, a.probe_pairs)
    print(f"[d3] before  recon eval {before['recon_rmse_eval_mm']:.3f} / train "
          f"{before['recon_rmse_train_mm']:.3f} mm · valid_conn {before['valid_conn']:.4f} · "
          f"endpoint_in_roi {before['endpoint_in_roi']:.4f}", flush=True)

    # ------------------------------------------------------------------ 손실 가중치 탐색
    sweep = []
    if a.sweep:
        for name in a.sweep.split(","):
            arm = ARMS[name]
            kw = C.build(cfg_raw)
            kw["max_steps"] = a.sweep_steps
            kw["out_dir"] = ROOT / f"outputs/checkpoints/retrain/d3_sweep_{name}"
            kw["save_every"] = a.sweep_steps
            if a.resume_ckpt:
                kw["resume"] = ROOT / a.resume_ckpt
            for k, v in arm["loss"].items():
                setattr(kw["weights"], k, v)
            if a.subjects_limit:
                kw["subjects"] = kw["subjects"][: a.subjects_limit]
            t0 = time.time()
            print(f"\n[sweep {name}] {arm['note']} — {a.sweep_steps} step", flush=True)
            ck = run(**kw, device=dev, heartbeat=LOCK)
            r = measure(ck, val_subs, dev, atlas, img.affine, a.n_val, a.n_val_pairs,
                        kw["cfg"].n_gt_per_pair, probe_subs, a.probe_pairs)
            r |= {"arm": name, "note": arm["note"], "loss_override": arm["loss"],
                  "steps": a.sweep_steps, "sec": time.time() - t0}
            sweep.append(r)
            print(f"[sweep {name}] recon eval {r['recon_rmse_eval_mm']:.3f} mm · "
                  f"valid_conn {r['valid_conn']:.4f} · endpoint_in_roi {r['endpoint_in_roi']:.4f} · "
                  f"{r['sec']:.0f}s", flush=True)
        (ROOT / "outputs/eval/w3a_sweep.json").write_text(
            json.dumps({"before": before, "sweep": sweep}, indent=1, ensure_ascii=False))
        if a.sweep_only:
            return 0

    # ------------------------------------------------------------------------- 본 학습
    kw = C.build(cfg_raw)
    if a.max_steps:
        kw["max_steps"] = a.max_steps
    if a.subjects_limit:
        kw["subjects"] = kw["subjects"][: a.subjects_limit]
    if a.resume_ckpt:
        # 탐색 arm 을 그대로 이어 돈다 (같은 phase 이름이면 optimizer/step/RNG 까지 복원된다).
        kw["resume"] = ROOT / a.resume_ckpt
    if a.out_dir:
        kw["out_dir"] = ROOT / a.out_dir
    for kv in a.loss_override.split(",") if a.loss_override else []:
        k, v = kv.split("="); setattr(kw["weights"], k, float(v))
    for kv in a.lr_override.split(",") if a.lr_override else []:
        k, v = kv.split("=")
        assert hasattr(kw["cfg"], k), k
        setattr(kw["cfg"], k, float(v))
    print(f"\n[d3] 본 학습: {kw['max_steps']} step, subject {len(kw['subjects'])}명, "
          f"bn_mode={kw['cfg'].bn_mode}, loss={vars(kw['weights'])}", flush=True)
    t0 = time.time()
    ck = run(**kw, device=dev, heartbeat=LOCK)
    assert ck.exists() and ck.stat().st_size > 0, ck
    after = measure(ck, val_subs, dev, atlas, img.affine, a.n_val, a.n_val_pairs,
                    kw["cfg"].n_gt_per_pair, probe_subs, a.probe_pairs)
    print(f"[d3] after   recon eval {after['recon_rmse_eval_mm']:.3f} / train "
          f"{after['recon_rmse_train_mm']:.3f} mm · valid_conn {after['valid_conn']:.4f} · "
          f"endpoint_in_roi {after['endpoint_in_roi']:.4f}", flush=True)

    res = {"cmd": f"python scripts/49_d3_joint.py --config {a.config}"
                  + (f" --sweep {a.sweep} --sweep-steps {a.sweep_steps}" if a.sweep else "")
                  + (f" --loss-override {a.loss_override}" if a.loss_override else ""),
           "config": a.config, "phase": kw["phase"], "checkpoint": str(ck.relative_to(ROOT)),
           "resume_from": str(resume.relative_to(ROOT)), "max_steps": kw["max_steps"],
           "n_train_subjects": len(kw["subjects"]), "loss_weights": vars(kw["weights"]),
           "lr": {k: getattr(kw["cfg"], k) for k in
                  ("lr_t1", "lr_vae_enc", "lr_dec", "lr_heads", "lr_prior")},
           "lr_override": a.lr_override, "resume_ckpt": a.resume_ckpt,
           "selfcheck": sc, "sweep": sweep,
           "val": {"before": before, "after": after},
           "train_sec": time.time() - t0, "elapsed_sec": time.time() - t_all}
    out = ROOT / "outputs/eval" / a.out
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n저장: {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/retrain/d3_joint.yaml")
    ap.add_argument("--val-subjects", default="outputs/splits/val.txt")
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-val-pairs", type=int, default=512)
    ap.add_argument("--n-probe", type=int, default=2, help="생성 진단에 쓸 val subject 수")
    ap.add_argument("--probe-pairs", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--subjects-limit", type=int, default=0)
    ap.add_argument("--no-lock", action="store_true",
                    help="GPU lock 우회. prior 만 학습하는 D-f 는 ~2GB 라 병렬이 가능하다")
    ap.add_argument("--sweep", default="", help="예: A,B,C")
    ap.add_argument("--sweep-steps", type=int, default=800)
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--loss-override", default="", help="예: recon=3.0,endpoint=2.0")
    ap.add_argument("--resume-ckpt", default="", help="config 의 resume 대신 이 checkpoint 에서 시작")
    ap.add_argument("--lr-override", default="", help="예: lr_dec=3e-5,lr_vae_enc=3e-5,lr_heads=1e-4")
    ap.add_argument("--out-dir", default="", help="config 의 out_dir 대신 (arm 별 분리)")
    ap.add_argument("--out", default="w3a_joint_train.json")
    ap.add_argument("--assemble", action="store_true",
                    help="학습 없이 전/후 final_*.json 두 개를 표로 합치고 게이트를 건다 (GPU 불필요)")
    ap.add_argument("--before-json", default="final_d1_decoder_step6000.json")
    ap.add_argument("--after-json", default="final_d3_joint_step3000.json")
    args = ap.parse_args()
    if args.assemble:
        sys.exit(main(args))
    if args.no_lock:
        import torch as _t
        free = (_t.cuda.mem_get_info()[0] / 1e9) if _t.cuda.is_available() else 0.0
        print(f"[d3] lock 우회 (--no-lock). GPU 여유 {free:.1f} GB", flush=True)
        assert free > 4.0, f"GPU 여유가 {free:.1f} GB 뿐이다 -- 병렬로 돌리면 죽는다"
    elif not acquire_lock(LOCK):
        sys.exit(3)
    try:
        sys.exit(main(args))
    finally:
        release_lock(LOCK)
