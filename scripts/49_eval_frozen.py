#!/usr/bin/env python
# W3-a 전용: `scripts/29_final_evaluation.py` 의 **2026-09-07 00:19 시점 그대로**의 복사본이다
# (md5 2e8c516978493bf1a2dd9ddce36ce378). 그 버전이 `outputs/eval/final_d1_decoder_step6000.json`
# 을 만들었고, 그 뒤 W3-b 가 29번의 pair-dice 규약(C9 크기 일치)을 바꿨다. 전/후 비교는 **양쪽에
# 같은 규약**을 쓰는 것이 핵심이라, joint 학습 후 checkpoint 도 이 얼린 규약으로 잰다.
# 29번은 W3-b 담당이라 수정하지 않는다. 새 규약 결과가 필요하면 29번을 그대로 부르면 된다.
#!/usr/bin/env python
"""최종 test-set 평가: T1 만 넣어 whole-brain TRK + SC 를 만들고 GT 와 비교한다.

  python scripts/29_final_evaluation.py --ckpt outputs/checkpoints/route/s6_joint/seg_full_step4000.pt

- 입력은 T1 뿐이다 (GT tractogram/SC 는 지표 계산에만 쓴다).
- pair 선택은 edge head (학습된 것). GT pair 목록은 쓰지 않는다.
- SC 두 가지를 모두 보고한다:
    generated : 생성 streamline 을 GT 와 같은 pass 규칙으로 센 SC (weight head 가중 포함)
    count     : Edge Count Head 가 직접 예측한 SC (절대 스케일)
- 지표: 전체/CTX-CTX/CTX-SUB/SUB-SUB(+endpoint-supported/pass-only) x Pearson/Spearman/CCC/MAE/RMSE/log-MAE,
  강도 구간(소/중/대), tract length, TRK 기하(길이 분포·끝점 정확도).
결과: outputs/eval/final_<checkpoint>.json (+ --trk 로 subject 별 .trk 저장)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                                      # noqa: E402
from atm_sc.data.paths import ATLAS, CACHE                                          # noqa: E402
from atm_sc.data.roi_groups import BLOCKS, TIERS, block_masks, tier_masks           # noqa: E402
from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics, sc_metrics_extended   # noqa: E402
from atm_sc.evaluation.reproduction_metrics import (ablation_gap, bundle_distance,   # noqa: E402
                                                    endpoint_distance, length_distribution,
                                                    stratified_corr, subject_specificity,
                                                    valid_connection_rate)
from atm_sc.inference.generate_sc import (allocate_counts, pair_residual_log, generate_by_count, generate_tractogram,  # noqa: E402
                                          save_trk, select_pairs, template_counts, tractogram_sc)
from atm_sc.inference.latent_bank import LatentBank                                 # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM, from_checkpoint                                        # noqa: E402
from atm_sc.training.run import anatomy_feature, t1_input, roi_feats_if_needed                           # noqa: E402


def group_template(subjects) -> np.ndarray:
    """학습셋 GT SC 평균 = 개인을 전혀 보지 않는 기준선. 실측: test 31명에서 r=0.945, CCC=0.936, RMSE=1,844.
    모델이 이걸 못 넘으면 T1 에서 개인차를 전혀 못 읽었다는 뜻이다."""
    from atm_sc.data.dataset import load_mat_gt
    acc = None
    for s in subjects:
        w, _ = load_mat_gt(s)
        acc = w.astype(np.float64) if acc is None else acc + w
    return acc / len(subjects)


def split_primary_diagnostic(summary: dict) -> dict:
    """지표를 성능(primary)과 절대 스케일 진단(diagnostic)으로 명시 분리한다 (문제 C5).

    전체 pair r 은 tier 간 배율 차이가 만드는 허수다: 실측 test 31명에서 전체 0.713 인데
    tier 안으로 들어가면 small 0.164 / mid 0.189 / large 0.624 로 붕괴한다. CCC·sum_ratio·RMSE
    도 마찬가지로 "예측이 GT 와 같은 절대 스케일인가" 를 말할 뿐 개인차 재현 성능이 아니다.
    보고 지표는 T1 단독 generated 경로 하나만 쓴다 (alloc/template 은 기준선).
    """
    g = summary["generated"]
    st = summary["stratified"]
    for t in TIERS:                       # 두 경로(sc_metrics 마스크 / stratified_corr)가 같은 값을 내야 한다
        a_, b_ = g[t]["r"], st["tier"][t]
        assert a_ is None or b_ is None or np.isclose(a_, b_, atol=1e-6), (t, a_, b_)
    sg = summary.get("subject_specificity_generated") or {}
    sc_ = summary.get("subject_specificity_count") or {}
    sa = summary.get("subject_specificity") or {}
    summary["primary"] = {
        "note": "T1 단독 generated 경로만. tier 내부 r(특히 small/mid) 이 주지표다.",
        "tier_r": {t: st["tier"][t] for t in TIERS},
        "tier_r_log": {t: st["tier_log"][t] for t in TIERS},
        "tier_n": {t: st["n"][t] for t in TIERS},
        "block_r": {b: st["block"][b] for b in BLOCKS},
        "block_n": {b: st["n"][b] for b in BLOCKS},
        "residual_r_generated": summary["residual_r"],          # 그룹 템플릿 제거 후 잔차 상관
        "resid_r_loo_generated": sg.get("resid_r"),             # LOO 중심화 잔차 상관
        "pred_degenerate_generated": sg.get("pred_degenerate"),
        "inter_subj_r_generated": sg.get("inter_subj_r_pred"),
        "inter_subj_r_gt": sg.get("inter_subj_r_gt"),
        # training/run.py 의 val resid_r 은 count head 경로다. val 게이트와 비교 가능한 test 값은 이것뿐이다.
        "resid_r_loo_count": sc_.get("resid_r"),
        "inter_subj_r_count": sc_.get("inter_subj_r_pred"),
        "edge_f1": g["all"]["edge_f1"],
        "tier_edge_f1": {t: g[t]["edge_f1"] for t in TIERS},
    }
    deprecated = "폐기된 그룹정보 경로(--use-bank/템플릿 배분). 성능 지표 아님."
    summary["diagnostic"] = {
        "note": "절대 스케일 진단값. 성능 지표가 아니다 -- 주지표로 쓰지 말 것.",
        "all_r": st["all"], "all_r_log": st["all_log"], "all_n": st["n"]["all"],
        "all_spearman": g["all"]["spearman"],
        "ccc": g["all"]["ccc"], "rmse": g["all"]["rmse"], "mae": g["all"]["mae"],
        "log_mae": g["all"]["log_mae"], "sum_ratio": summary["sum_ratio"],
        "template_r": sa.get("template_r"),
        "alloc_baseline_r": (summary["alloc"]["all"]["r"] if summary.get("alloc") else None),
        "alloc_baseline_r_note": deprecated,
        "resid_r_loo_alloc": sa.get("resid_r"),
        "resid_r_loo_alloc_note": deprecated,
    }
    return summary


@torch.no_grad()
def oracle_reconstruct(m, feat, S_np, P_np, dev, batch=4096):
    """GT streamline 을 인코딩해 얻은 posterior mu 로 복원한다 (= 오라클 latent).

    prior/조건화의 latent 오류를 0 으로 두고 **디코더 오차만** 남긴 경로다. 생성 경로와의
    차이가 곧 prior 결함의 크기다. 모델은 eval 모드 (from_checkpoint 가 그렇게 만든다) --
    배포되는 함수 그대로 재야 한다.
    """
    out = []
    for i in range(0, len(S_np), batch):
        s = torch.as_tensor(np.ascontiguousarray(S_np[i:i + batch]), dtype=torch.float32, device=dev)
        pr = torch.as_tensor(np.ascontiguousarray(P_np[i:i + batch]), dtype=torch.int64, device=dev)
        c = m.condition(feat, pr)
        mu, _ = m.encode_streamlines(s, c)
        rec = m.decode(mu, c)
        assert rec.shape == s.shape, (rec.shape, s.shape)
        assert torch.isfinite(rec).all(), "오라클 복원에 NaN/Inf"
        out.append(rec.cpu().numpy().astype(np.float32))
    R = np.concatenate(out)
    assert R.shape == S_np.shape, (R.shape, S_np.shape)
    return R


def oracle_metrics(m, feat, subj, a, dev):
    """오라클 latent 복원의 기하. W1-b 보정곡선(`scripts/45`)의 규약을 그대로 쓴다.

    **두 규약을 모두 낸다.** 규약을 안 밝히면 숫자를 비교할 수 없다 (실측: 같은 모델·같은
    subject 인데 self 규약 0.38, half 규약 0.15 로 2.5배 갈린다):

      wb_self       pred = 복원(N),        gt = 같은 N           (sigma=0 -> 1.000) 변위만
      wb_disjoint   pred = 복원(N),        gt = 겹치지 않는 N    (sigma=0 -> 0.783) 표본차 + 변위
                    -> **생성 경로의 trk_geometry.dice (0.546) 와 같은 저울**
      pair_self     pred = 번들 전체 복원,  gt = 같은 번들 전체   (sigma=0 -> 1.000) 변위만
      pair_half     pred = 앞 절반 복원,    gt = 뒤 절반          (sigma=0 -> 0.704 = 천장)
                    -> **생성 경로의 trk_per_pair.dice (0.095) / 천장 (0.700) 과 같은 저울**

    latent 오류가 0 이므로 이것이 디코더 충실도의 상한이다. 생성 경로와의 차이가 prior 결함.
    """
    z = np.load(subj.dir / "bundles.npz")
    St, off = z["streamlines"], z["pair_offsets"]
    pid = np.asarray(subj.pair_ids, np.int64)
    assert len(St) > 0 and len(off) == len(pid) + 1, (len(St), len(off), len(pid))
    rng = np.random.default_rng(0)
    n_wb = min(a.oracle_wb_sample, len(St) // 2)
    assert n_wb >= 100, f"{subj.sub}: whole-brain 표본이 너무 적다 ({len(St)} 가닥)"
    sel = rng.choice(len(St), 2 * n_wb, replace=False)
    iA, iB = np.sort(sel[:n_wb]), np.sort(sel[n_wb:])
    labA = np.searchsorted(off, iA, side="right") - 1        # 가닥 -> 소속 pair
    assert labA.min() >= 0 and labA.max() < len(pid), (int(labA.min()), int(labA.max()), len(pid))
    A, B = St[iA].astype(np.float32), St[iB].astype(np.float32)
    recA = oracle_reconstruct(m, feat, A, pid[labA], dev)
    out = {"n_wb": int(n_wb),
           "recon_rmse_eval_mm": float(np.sqrt(((recA - A) ** 2).sum(-1).mean())),
           "wb_self": bundle_geometry_metrics(recA, A, voxel_mm=a.trk_voxel_mm),
           "wb_disjoint": bundle_geometry_metrics(recA, B, voxel_mm=a.trk_voxel_mm),
           "wb_gt_ceiling": bundle_geometry_metrics(A, B, voxel_mm=a.trk_voxel_mm)}
    order = np.argsort(-np.asarray(subj.pair_count_full))[: a.trk_pairs]
    self_, half_, ceil_ = [], [], []
    for k in order:
        gt_b = subj.get_pair(int(k))[0].numpy().astype(np.float32)
        if len(gt_b) < 2:
            continue
        gb = gt_b[: a.oracle_pair_max]
        pk = np.repeat(pid[k][None], len(gb), 0)
        self_.append(bundle_geometry_metrics(oracle_reconstruct(m, feat, gb, pk, dev), gb,
                                             voxel_mm=a.trk_voxel_mm))
        h = len(gt_b) // 2
        if h >= 2:                                   # 천장 규약과 같은 반반 분할
            first, second = gt_b[:h], gt_b[h:2 * h]
            rec1 = oracle_reconstruct(m, feat, first, np.repeat(pid[k][None], h, 0), dev)
            half_.append(bundle_geometry_metrics(rec1, second, voxel_mm=a.trk_voxel_mm))
            ceil_.append(bundle_geometry_metrics(first, second, voxel_mm=a.trk_voxel_mm))
    assert self_, f"{subj.sub}: 오라클 pair dice 를 잴 pair 가 없다"
    ks = ("coverage", "overreach", "dice")
    out["pair_self"] = {k: float(np.nanmean([x[k] for x in self_])) for k in ks}
    out["pair_self"]["n_pairs"] = len(self_)
    if half_:
        out["pair_half"] = {k: float(np.nanmean([x[k] for x in half_])) for k in ks}
        out["pair_half"]["n_pairs"] = len(half_)
        out["pair_half_ceiling"] = {k: float(np.nanmean([x[k] for x in ceil_])) for k in ks}
    return out


def _masks(subj):
    bm = block_masks(subj.n_roi)
    e_sc, p_sc = np.asarray(subj.sc_end), np.asarray(subj.sc_mat)
    ss = bm["sub-sub"]
    m = dict(bm)
    m.update(tier_masks(p_sc))
    m["ss_endpoint"] = ss & (e_sc > 0)
    m["ss_pass_only"] = ss & (e_sc == 0) & (p_sc > 0)
    return {k: v for k, v in m.items() if v.any()}


def aux_feats(m, sub, n_roi, a, dev):
    """edge_head / count_head_end 용 [3321, D]. 전체 upper pair 순서다."""
    h = m.edge_head
    if h is None or not (getattr(h, "local_dim", 0) or getattr(h, "tier1_dim", 0)):
        return None, None
    iu = np.stack(np.triu_indices(n_roi, 1), 1)
    P = torch.as_tensor(iu, device=dev)
    loc = t1f = None
    if int(getattr(h, "local_dim", 0) or 0):
        from atm_sc.data.local_feats import load_roi_feats, pair_local
        loc = pair_local(load_roi_feats(sub, a.t1_src, dev, n_roi=n_roi), P)
    if int(getattr(h, "tier1_dim", 0) or 0):
        from atm_sc.data.anat_tier1 import pair_input
        assert a.tier1_stats is not None, "aux tier1 head 인데 통계가 안 실렸다"
        t1f = torch.as_tensor(pair_input(sub, a.tier1_stats, a.t1_src), device=dev)
    return loc, t1f


def head_feats(m, sub, pairs, a, dev):
    """count head 가 국소/tier1 통로로 만들어졌으면 그 입력을 만든다. 아니면 (None, None).

    예전에는 여기서 안 만들어 넘겨서, 국소 통로를 켠 checkpoint 로 추론하면 head 의 assert 에
    걸렸다 -- 즉 개인차를 학습한 head 를 추론이 아예 못 쓰고 있었다."""
    h = m.count_head
    if h is None:
        return None, None
    P = torch.as_tensor(np.asarray(pairs, np.int64), device=dev)
    loc = t1f = None
    if int(getattr(h, "local_dim", 0) or 0):
        from atm_sc.data.local_feats import load_roi_feats, pair_local
        loc = pair_local(load_roi_feats(sub, a.t1_src, dev, n_roi=m.n_roi), P)
    if int(getattr(h, "tier1_dim", 0) or 0):
        from atm_sc.data.anat_tier1 import pair_input
        assert a.tier1_stats is not None, "tier1 head 인데 통계가 안 실렸다"
        t1f = torch.as_tensor(pair_input(sub, a.tier1_stats, a.t1_src), device=dev)[
            _pair_rows(pairs, m.n_roi)]
    return loc, t1f


def _pair_rows(pairs, n_roi):
    """pair_input 은 3321 개 전체 upper pair 순서다. 고른 pair 만 뽑는다."""
    from atm_sc.models.pair_anchor import upper_index
    pr = torch.as_tensor(np.asarray(pairs, np.int64))
    lo = torch.minimum(pr[:, 0], pr[:, 1]); hi = torch.maximum(pr[:, 0], pr[:, 1])
    return upper_index(lo, hi, n_roi)


def evaluate_subject(m, sub, atlas, affine, a, dev):
    subj = ROIPairSubject(sub)
    t0 = time.time()
    with torch.no_grad():
        feat = (anatomy_feature(m, sub, a.init_bundle) if m.unet_level == "none"
                else m.atm.encode_anatomy(t1_input(m, sub, a.t1_src)))          # T1 -> anatomy (입력은 T1 뿐)
        al, at = aux_feats(m, sub, subj.n_roi, a, dev)
        pairs, prob = select_pairs(m, feat, subj.n_roi, thr=a.edge_thr, local=al, tier1=at)
        assert len(pairs) > 0, f"{sub}: edge head 가 고른 pair 가 없음 (thr={a.edge_thr})"
        roi = roi_feats_if_needed(m, sub, a.t1_src)
        S, w, pr = generate_tractogram(m, feat, pairs, a.n_per_pair, seed=0, local_roi=roi)
        loc, t1f = head_feats(m, sub, pairs, a, dev)
        cnt_mat = (m.edge_count_matrix(feat, torch.as_tensor(pairs, device=dev),
                                       loc, t1f).cpu().numpy()
                   if m.count_head is not None else None)
    t_gen = time.time() - t0
    sc = tractogram_sc(S, w, atlas, affine, subj.n_roi)
    gt = torch.as_tensor(np.asarray(subj.sc_mat, np.float32))
    gt_len = torch.as_tensor(np.asarray(subj.len_mat, np.float32))
    masks = _masks(subj)
    out = {"subject": sub, "n_pairs_selected": int(len(pairs)), "n_streamlines": int(len(S)),
           "gen_sec": t_gen, "edge_prob_mean": float(np.mean(prob))}
    if a.by_count and (a.alloc_template is not None or m.count_head_end is not None):
        # GT 처럼 pair 마다 다른 개수를 생성한다 -> 생성 가닥 수 자체가 SC 값이 된다 (weight head 불필요)
        # 배분: train 평균 sc_end (실측 r 0.88) > count_head_end (0.57, 균등 0.71 보다도 낮다)
        if a.alloc_template is not None:
            nall = template_counts(a.alloc_template, pairs, a.total_streamlines or 460_000)
        else:
            rl = (pair_residual_log(m, feat, pairs, a.template_log1p, loc, t1f)
                  if a.resid_alloc else None)
            ai = _pair_rows(pairs, m.n_roi)
            nall = allocate_counts(m, feat, pairs, total=a.total_streamlines,
                                   resid_log=rl, resid_gain=a.resid_gain,
                                   local=None if al is None else al[ai],
                                   tier1=None if at is None else at[ai])
            if rl is not None:
                out["resid_log_sd"] = float(np.std(rl))
        sc_a, n_gen, S2 = generate_by_count(m, feat, pairs, nall, atlas, affine, subj.n_roi,
                                            keep=a.trk, bank=a.bank, local_roi=roi)
        if a.bank is not None:
            out["alloc_bank_hit"] = float(sc_a.get("bank_hit", 0.0))
        out["alloc"] = {"all": sc_metrics_extended(torch.as_tensor(sc_a["pass"]["sc"], dtype=torch.float32), gt)}
        out["alloc"].update({k: sc_metrics_extended(torch.as_tensor(sc_a["pass"]["sc"], dtype=torch.float32), gt,
                                                    torch.as_tensor(v)) for k, v in _masks(subj).items()})
        out["alloc_n_streamlines"] = n_gen
        out["alloc_sum_ratio"] = float(sc_a["pass"]["sc"][np.triu_indices(subj.n_roi, 1)].sum()
                                       / max(float(np.asarray(subj.sc_mat)[np.triu_indices(subj.n_roi, 1)].sum()), 1))
        if a.trk and S2 is not None:
            S = S2
    for name, pred in (("generated", torch.as_tensor(sc["pass"]["sc"], dtype=torch.float32)),
                       ("count", None if cnt_mat is None else torch.as_tensor(cnt_mat, dtype=torch.float32))):
        if pred is None:
            continue
        out[name] = {"all": sc_metrics_extended(pred, gt)}
        out[name].update({k: sc_metrics_extended(pred, gt, torch.as_tensor(v)) for k, v in masks.items()})
    # 주지표: tier/block 내부 상관 (전체 pair r 은 tier 간 배율차의 허수라 진단값이다)
    pw_np = np.asarray(sc["pass"]["sc"], np.float64)
    gt_np = np.asarray(subj.sc_mat, np.float64)
    out["stratified"] = stratified_corr(pw_np, gt_np, n_roi=subj.n_roi)
    iu0 = np.triu_indices(subj.n_roi, 1)
    out["sum_ratio"] = float(pw_np[iu0].sum() / max(gt_np[iu0].sum(), 1.0))     # 절대 스케일 진단
    if a.oracle:
        # 오라클 latent (GT posterior mu) 경로. 생성 경로와 분리해서 낸다 --
        # 디코더 오차와 prior 오차를 한 숫자에 섞으면 무엇이 고쳐졌는지 알 수 없다.
        # 주의: 아래 trk_eval 블록이 지역 변수 `m` 을 모델이 아닌 dict 로 덮어쓴다 -> 그 전에 부른다.
        out["oracle"] = oracle_metrics(m, feat, subj, a, dev)
    if a.template is not None:
        tm = torch.as_tensor(a.template, dtype=torch.float32)
        out["template"] = sc_metrics_extended(tm, gt)                     # 기준선 자체 성적
        # 개인차를 읽었는가: 그룹 평균을 뺀 잔차끼리의 상관. 0 이면 템플릿 이상의 정보가 없다.
        pw = torch.as_tensor(sc["pass"]["sc"], dtype=torch.float32)
        k = float((pw * gt).sum() / (pw * pw).sum().clamp(min=1e-9))       # 배율은 보정하고 비교
        iu = torch.triu_indices(gt.shape[0], gt.shape[0], 1)
        dp, dg = (pw * k - tm)[iu[0], iu[1]], (gt - tm)[iu[0], iu[1]]
        out["residual_r"] = float(torch.corrcoef(torch.stack([dp.double(), dg.double()]))[0, 1])
        out["beats_template_r"] = bool(out["generated"]["all"]["r"] > out["template"]["r"])
    if a.trk_eval:
        # 생성 tractogram 이 GT tractogram 의 기하를 재현하는가 (GESTA QC 문서 §12, §60)
        z = np.load(subj.dir / "bundles.npz")
        St = z["streamlines"]
        rng = np.random.default_rng(0)
        gt_S = St[np.sort(rng.choice(len(St), min(a.trk_gt_sample, len(St)), replace=False))].astype(np.float32)
        gen_S = S if len(S) <= a.trk_gen_sample else S[np.sort(rng.choice(len(S), a.trk_gen_sample, replace=False))]
        out["trk_geometry"] = bundle_geometry_metrics(gen_S, gt_S, voxel_mm=a.trk_voxel_mm)
        # pair 단위: 가장 큰 연결 몇 개에서 bundle 이 GT bundle 과 겹치는가.
        # 주의: 생성 수가 GT bundle(최대 256개) 보다 적으면 coverage 는 구조적으로 낮게 나온다.
        # 공정한 비교는 --by-count (연결마다 예측 개수만큼 생성) 와 함께 볼 것.
        pid = np.asarray(subj.pair_ids); order = np.argsort(-np.asarray(subj.pair_count_full))[: a.trk_pairs]
        per, ctrl = [], []
        for k in order:
            gt_b = subj.get_pair(int(k))[0].numpy()
            sel = (pr[:, 0] == pid[k][0]) & (pr[:, 1] == pid[k][1])
            if sel.sum() >= 2 and len(gt_b) >= 2:
                m = bundle_geometry_metrics(S[sel], gt_b, voxel_mm=a.trk_voxel_mm)
                m.update(bundle_distance(S[sel], gt_b, max_n=a.trk_dist_n, hausdorff=a.hausdorff))
                m["endpoint_dist_mm"] = endpoint_distance(S[sel], gt_b, max_n=a.trk_dist_n)
                per.append(m)
                # 대조군: 같은 GT 번들을 둘로 나눠 잰 값. 이 지표들의 **천장**이라
                # 없으면 절대값을 해석할 수 없다 (실측 pair dice 천장 0.597).
                if len(gt_b) >= 8:
                    h = len(gt_b) // 2
                    c = bundle_geometry_metrics(gt_b[:h], gt_b[h:], voxel_mm=a.trk_voxel_mm)
                    c.update(bundle_distance(gt_b[:h], gt_b[h:], max_n=a.trk_dist_n, hausdorff=a.hausdorff))
                    c["endpoint_dist_mm"] = endpoint_distance(gt_b[:h], gt_b[h:], max_n=a.trk_dist_n)
                    ctrl.append(c)
        if per:
            keys = ("coverage", "overreach", "dice", "length_err_mm", "length_ks",
                    "mdf_mm", "endpoint_dist_mm") + (("hausdorff_mm",) if a.hausdorff else ())
            out["trk_per_pair"] = {k: float(np.nanmean([x[k] for x in per])) for k in keys}
            out["trk_per_pair"]["n_pairs"] = len(per)
            if ctrl:
                out["trk_per_pair_ceiling"] = {k: float(np.nanmean([x[k] for x in ctrl])) for k in keys}
        # 의도한 ROI 쌍을 실제로 연결했는가 (§1-8)
        out["connection"] = valid_connection_rate(S, pr, atlas, affine, subj.n_roi)
        # 길이 분포 (§4-4): Wasserstein 은 KS 와 달리 mm 단위라 크기를 읽을 수 있다
        gt_len_all = np.linalg.norm(np.diff(gt_S, axis=1), axis=-1).sum(1)
        out["length_dist"] = length_distribution(
            np.linalg.norm(np.diff(gen_S, axis=1), axis=-1).sum(1), gt_len_all)
    lp = torch.as_tensor(sc["pass"]["len"], dtype=torch.float32)
    out["length"] = sc_metrics_extended(lp, gt_len)
    Lg = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)
    gt_L = np.asarray(subj.len_mat)[np.triu_indices(subj.n_roi, 1)]
    out["trk"] = {"length_mean_mm": float(Lg.mean()), "length_p5_p95": [float(np.percentile(Lg, 5)), float(np.percentile(Lg, 95))],
                  "gt_edge_length_mean_mm": float(gt_L[gt_L > 0].mean()),
                  "valid_ratio": float(np.mean(np.isfinite(Lg) & (Lg > 20) & (Lg < 220)))}
    if a.trk:
        d = ROOT / "outputs" / "eval" / "trk"; d.mkdir(parents=True, exist_ok=True)
        save_trk(S, d / f"{sub}.trk")
        out["trk_file"] = str((d / f"{sub}.trk").relative_to(ROOT))
    # subject 특이성(§3-1, §3-2)은 여러 subject 를 모아야 계산된다. main 에서 쓰고 JSON 저장 전에 지운다.
    iu2 = np.triu_indices(subj.n_roi, 1)
    best = sc_a["pass"]["sc"] if (a.by_count and "alloc" in out) else sc["pass"]["sc"]
    out["_pred_vec"] = np.asarray(best, np.float64)[iu2]
    # 보고 지표는 T1 단독 generated 경로 하나다. best 는 --by-count 시 alloc(추론에 그룹 정보를
    # 쓰는 폐기된 구성) 이 될 수 있으므로 generated 벡터를 따로 남긴다.
    out["_pred_vec_gen"] = np.asarray(sc["pass"]["sc"], np.float64)[iu2]
    # count head 직접 예측. training/run.py 의 val resid_r 이 이 경로라 val 과 비교 가능한 유일한 값이다.
    out["_pred_vec_count"] = None if cnt_mat is None else np.asarray(cnt_mat, np.float64)[iu2]
    out["_gt_vec"] = np.asarray(subj.sc_mat, np.float64)[iu2]
    return out


def main(a):
    import nibabel as nib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint 메타(in_channels / template / t1_source)로 모델을 만든다.
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    a.t1_src = sd.get("t1_source", "syn")   # checkpoint 가 학습에 쓴 입력 프로토콜
    img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()]
    subs = [s for s in subs if (CACHE / f"{s}_T1w_syn_W.npy").exists()]
    a.total_streamlines = a.total_streamlines or None
    a.bank = a.alloc_template = None
    a.template_log1p = a.tier1_stats = None
    if a.resid_alloc or any(int(getattr(h, "tier1_dim", 0) or 0) for h in (m.count_head, m.edge_head, m.count_head_end) if h is not None):
        z = np.load(ROOT / "outputs/cache/sc_template_stats.npz", allow_pickle=False)
        R = int(z["n_roi"]); iu = np.triu_indices(R, 1)
        a.template_log1p = np.zeros((R, R), np.float64)
        a.template_log1p[iu] = z["template"]; a.template_log1p.T[iu] = z["template"]
        assert int(z["n_subjects"]) == 144, f"템플릿이 train 144명이 아니다 ({int(z['n_subjects'])})"
    if any(int(getattr(h, "tier1_dim", 0) or 0) for h in (m.count_head, m.edge_head, m.count_head_end) if h is not None):
        from atm_sc.data.anat_tier1 import STATS_PATH
        assert STATS_PATH.exists(), "scripts/55_anat_tier1.py 를 먼저 실행"
        z1 = np.load(STATS_PATH, allow_pickle=False)
        a.tier1_stats = {k: z1[k] for k in z1.files}
        tr = [l.strip() for l in (ROOT / "outputs/splits/train.txt").read_text().splitlines() if l.strip()]
        assert list(z1["subjects"]) == tr, "tier1 통계가 train split 으로 만들어지지 않았다 (누수)"
    if getattr(a, "use_bank", False):   # 폐기됨: 추론에 train latent bank 를 쓰는 경로 (그룹 정보)
        bp, tp = ROOT / "outputs/inference/latent_bank.npz", ROOT / "outputs/inference/template.npz"
        assert bp.exists() and tp.exists(), "scripts/34_build_inference_prior.py 를 먼저 실행"
        a.bank = LatentBank.load(bp)
        a.alloc_template = np.load(tp)["sc_end"]
        print(f"추론 사전정보: latent bank {a.bank.n_pairs} pair + 배분 템플릿 (train 전용)", flush=True)
    a.template = None
    if a.template_subjects:
        tsubs = [l.strip() for l in (ROOT / a.template_subjects).read_text().splitlines() if l.strip()]
        a.template = group_template(tsubs)
        print(f"그룹 템플릿: 학습셋 {len(tsubs)}명 GT 평균 (개인 무시 기준선)", flush=True)
    if a.limit:
        subs = subs[: a.limit]
    res = []
    for i, s in enumerate(subs):
        r = evaluate_subject(m, s, atlas, img.affine, a, dev)
        res.append(r)
        g = r["generated"]["all"]
        print(f"[{i+1}/{len(subs)}] {s}: pair {r['n_pairs_selected']} · streamline {r['n_streamlines']:,} · "
              f"SC r={g['r']:.3f} r_log={g['r_log']:.3f} ccc={g['ccc']:.3f} · {r['gen_sec']:.1f}s", flush=True)

    def agg(path):
        vals = []
        for r in res:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None
    # --- subject 특이성 (§3-1 잔차 상관, §3-2 subject 간 유사도) --------------
    spec = spec_gen = spec_cnt = None
    if len(res) >= 3:
        P = np.stack([r.pop("_pred_vec") for r in res])
        Pg = np.stack([r.pop("_pred_vec_gen") for r in res])
        Pc_ = [r.pop("_pred_vec_count") for r in res]
        G = np.stack([r.pop("_gt_vec") for r in res])
        spec = subject_specificity(P, G)
        spec_gen = subject_specificity(Pg, G)          # T1 단독 generated 경로 (주지표는 이쪽)
        Pc = np.stack(Pc_) if all(v is not None for v in Pc_) else None
        if Pc is not None:
            spec_cnt = subject_specificity(Pc, G)      # count head 경로 = run.py val resid_r 과 같은 추정량
        # LOO 잔차 상관은 subject 벡터가 있어야 재계산된다. JSON 에는 안 들어가므로 npz 로 남긴다
        # (없으면 나중에 모델을 다시 돌려야만 얻을 수 있다 -- 실제로 그 일이 있었다).
        vdir = ROOT / "outputs" / "eval"; vdir.mkdir(parents=True, exist_ok=True)
        vp = vdir / f"final_{Path(a.ckpt).stem}_vectors.npz"
        np.savez_compressed(vp, subjects=np.array([r["subject"] for r in res]), gt=G,
                            pred_generated=Pg, pred_best=P,
                            **({"pred_count": Pc} if Pc is not None else {}))
        assert vp.stat().st_size > 0, f"{vp} 가 0 바이트"
        print(f"subject 벡터 저장: {vp.relative_to(ROOT)}  (LOO resid_r 재계산용)", flush=True)
        if a.template is not None:
            t = np.asarray(a.template, np.float64)[np.triu_indices(a.template.shape[0], 1)]
            spec["template_r"] = float(np.mean([np.corrcoef(t, G[i])[0, 1] for i in range(len(G))]))
            # 문서 §3-4: 모델이 population template 을 못 넘으면 T1 개인 정보를 못 쓴 것이다
            spec["beats_template"] = float(np.mean(
                [np.corrcoef(P[i], G[i])[0, 1] > np.corrcoef(t, G[i])[0, 1] for i in range(len(G))]))
    else:
        for r in res:
            for k in ("_pred_vec", "_pred_vec_gen", "_pred_vec_count", "_gt_vec"):
                r.pop(k, None)

    groups = ["all"] + list(BLOCKS) + list(TIERS) + ["ss_endpoint", "ss_pass_only"]
    summary = {"checkpoint": a.ckpt, "n_subjects": len(res), "split": a.subjects,
               "n_per_pair": a.n_per_pair, "edge_thr": a.edge_thr, "time": time.strftime("%F %T"),
               "generated": {g: {k: agg(["generated", g, k]) for k in
                                 ("r", "r_log", "spearman", "ccc", "mae", "rmse", "log_mae", "edge_f1",
                                  "weak_edge_recall", "n_edges")} for g in groups},
               "length": {k: agg(["length", k]) for k in ("r", "r_log", "ccc", "mae", "rmse")},
               "trk": {k: agg(["trk", k]) for k in ("length_mean_mm", "gt_edge_length_mean_mm", "valid_ratio")},
               "trk_geometry": ({k: agg(["trk_geometry", k]) for k in
                                 ("coverage", "overreach", "dice", "length_err_mm", "length_ks",
                                  "valid_ratio", "duplicate_ratio")} if any("trk_geometry" in r for r in res) else None),
               "trk_per_pair": ({k: agg(["trk_per_pair", k]) for k in
                                 ("coverage", "overreach", "dice", "length_err_mm", "length_ks",
                                  "mdf_mm", "endpoint_dist_mm", "hausdorff_mm")}
                                if any("trk_per_pair" in r for r in res) else None),
               "trk_per_pair_ceiling": ({k: agg(["trk_per_pair_ceiling", k]) for k in
                                         ("coverage", "overreach", "dice", "mdf_mm",
                                          "endpoint_dist_mm", "hausdorff_mm")}
                                        if any("trk_per_pair_ceiling" in r for r in res) else None),
               "connection": ({k: agg(["connection", k]) for k in
                               ("valid_conn", "partial_conn", "invalid_conn", "endpoint_in_roi")}
                              if any("connection" in r for r in res) else None),
               "length_dist": ({k: agg(["length_dist", k]) for k in
                                ("len_mean_diff_mm", "len_median_diff_mm", "len_wasserstein_mm", "len_ks")}
                               if any("length_dist" in r for r in res) else None),
               "oracle": ({g: {k: agg(["oracle", g, k]) for k in
                               ("coverage", "overreach", "dice", "n_pairs")}
                           for g in ("wb_self", "wb_disjoint", "wb_gt_ceiling",
                                     "pair_self", "pair_half", "pair_half_ceiling")}
                          | {"recon_rmse_eval_mm": agg(["oracle", "recon_rmse_eval_mm"]),
                             "n_wb": agg(["oracle", "n_wb"])}
                          if any("oracle" in r for r in res) else None),
               "subject_specificity": spec, "subject_specificity_generated": spec_gen,
               "subject_specificity_count": spec_cnt,
               "sum_ratio": agg(["sum_ratio"]),
               "stratified": {"all": agg(["stratified", "all"]), "all_log": agg(["stratified", "all_log"]),
                              "tier": {t: agg(["stratified", "tier", t]) for t in TIERS},
                              "tier_log": {t: agg(["stratified", "tier_log", t]) for t in TIERS},
                              "block": {b: agg(["stratified", "block", b]) for b in BLOCKS},
                              "block_log": {b: agg(["stratified", "block_log", b]) for b in BLOCKS},
                              "n": {k: agg(["stratified", "n", k]) for k in ("all", *TIERS, *BLOCKS)}},
               "template": {k: agg(["template", k]) for k in ("r", "r_log", "ccc", "rmse", "mae")},
               "alloc": ({g: {k: agg(["alloc", g, k]) for k in ("r", "r_log", "spearman", "ccc", "rmse", "log_mae")}
                          for g in groups} if any("alloc" in r for r in res) else None),
               "alloc_n_streamlines": agg(["alloc_n_streamlines"]), "alloc_sum_ratio": agg(["alloc_sum_ratio"]),
               "residual_r": agg(["residual_r"]),
               "beats_template": (float(np.mean([r.get("beats_template_r", False) for r in res]))
                                  if any("beats_template_r" in r for r in res) else None),
               "gen_sec_mean": agg(["gen_sec"]), "n_streamlines_mean": agg(["n_streamlines"]),
               "n_pairs_selected_mean": agg(["n_pairs_selected"])}
    if any("count" in r for r in res):
        summary["count"] = {g: {k: agg(["count", g, k]) for k in ("r", "r_log", "spearman", "ccc", "log_mae")} for g in groups}
    split_primary_diagnostic(summary)
    out = ROOT / "outputs" / "eval"; out.mkdir(parents=True, exist_ok=True)
    name = Path(a.ckpt).stem
    (out / f"final_{name}.json").write_text(json.dumps({"summary": summary, "per_subject": res}, indent=1))
    print("\n== 주지표 (primary) — T1 단독 generated ==")
    print(json.dumps(summary["primary"], indent=1, ensure_ascii=False))
    print("\n== 진단값 (diagnostic) — 절대 스케일용, 성능 지표 아님 ==")
    print(json.dumps(summary["diagnostic"], indent=1, ensure_ascii=False))
    print("\n== 최종 요약 (전체) ==")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n저장: outputs/eval/final_{name}.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", default="outputs/splits/test.txt")
    ap.add_argument("--n-per-pair", type=int, default=16)
    ap.add_argument("--edge-thr", type=float, default=0.5)
    ap.add_argument("--init-bundle", default="AF_L")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trk", action="store_true", help="subject 별 .trk 저장")
    ap.add_argument("--trk-eval", action="store_true", help="생성 tractogram 의 기하를 GT 와 비교")
    ap.add_argument("--trk-gt-sample", type=int, default=20000)
    ap.add_argument("--trk-gen-sample", type=int, default=20000)
    ap.add_argument("--trk-voxel-mm", type=float, default=2.0)
    ap.add_argument("--trk-pairs", type=int, default=50, help="pair 단위 기하 비교에 쓸 큰 연결 수")
    ap.add_argument("--trk-dist-n", type=int, default=32,
                    help="MDF/Hausdorff 표본 수 (N x M 쌍거리라 비싸다)")
    ap.add_argument("--hausdorff", action="store_true", help="Hausdorff 도 계산 (느림)")
    ap.add_argument("--oracle", action="store_true",
                    help="오라클 latent(GT posterior mu) 복원의 기하도 잰다. 디코더 충실도의 상한 "
                         "-- 생성 경로와의 차이가 prior/조건화 오류의 크기다")
    ap.add_argument("--oracle-wb-sample", type=int, default=8000,
                    help="오라클 whole-brain dice 표본 가닥 수 (W1-b 보정곡선과 같은 8000)")
    ap.add_argument("--oracle-pair-max", type=int, default=256,
                    help="오라클 pair dice 에서 번들당 최대 가닥 수")
    ap.add_argument("--resid-alloc", action="store_true",
                    help="pair count head 의 subject 잔차로 가닥 배분을 변조한다. 끝점 head 는 "
                         "잔차 목적으로 학습된 적이 없어 그 개인차는 잡음이다")
    ap.add_argument("--resid-gain", type=float, default=1.0)
    ap.add_argument("--by-count", action="store_true",
                    help="pair 마다 예측 개수만큼 생성 (GT 밀도 재현). 생성 개수가 곧 SC 값")
    ap.add_argument("--total-streamlines", type=int, default=0,
                    help="by-count 총 가닥 수 고정 (0 이면 예측값 그대로). GT 는 끝점 할당 기준 약 489,000")
    ap.add_argument("--template-subjects", default="outputs/splits/train.txt",
                    help="그룹 템플릿(개인 무시 기준선)을 만들 subject 목록. 빈 문자열이면 생략")
    sys.exit(main(ap.parse_args()))
