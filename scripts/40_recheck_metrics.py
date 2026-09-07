#!/usr/bin/env python
"""S0-a: 기존 평가 JSON 을 읽어 계층별(tier/block) 지표를 재계산한다.

  python scripts/40_recheck_metrics.py --eval outputs/eval/final_p4_joint_step3000.json

왜: 전체 pair 상관 r=0.713 은 tier 간 배율 차이가 만드는 허수다. tier 안으로 나누면
붕괴하는지 (small/mid 가 0.2 아래인지) 를 독립적으로 확인한다.

보고 대상은 T1 단독 `generated` 경로 하나뿐이다. alloc(추론에 그룹 정보를 쓰는 폐기된 구성)
과 group template 은 기준선으로만 싣는다.

세 가지를 낸다:
  1. per-subject 재집계   기존 JSON 의 subject 별 r 을 다시 평균/중앙값/IQR 로 요약
  2. 마스크 검증          .mat GT 로 tier/block pair 수를 다시 세어 JSON n_edges 와 대조
  3. 템플릿 기준선        train 평균 GT 를 예측으로 넣고 stratified_corr 실행 (모델 없이도
                          전체 r 이 높게 나오는지 = 전체 r 이 성능이 아님을 보이는 대조)
결과: outputs/eval/s0a_metrics_recheck.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import SC_MAT                                          # noqa: E402
from atm_sc.data.roi_groups import BLOCKS, TIERS, block_masks, tier_masks     # noqa: E402
from atm_sc.evaluation.reproduction_metrics import (stratified_corr,          # noqa: E402
                                                    subject_specificity)

GROUPS = ("all", *TIERS, *BLOCKS)
DEPRECATED = "폐기된 그룹정보 경로(--use-bank/템플릿 배분). 성능 지표 아님."
NO_VEC = ("subject 별 예측 벡터가 없어 재계산 불가. scripts/29_final_evaluation.py 가 "
          "_pred_vec* 를 json.dumps 전에 pop 해서 저장하지 않았다 (지금은 "
          "final_<ckpt>_vectors.npz 로 저장하도록 고쳤으므로 다음 평가 실행부터 채워진다).")


def loo_resid(vec_path: Path, subs: list[str]) -> dict:
    """경로별 LOO 중심화 resid_r. 벡터 npz 가 있어야 계산된다.

    training/run.py 의 val resid_r 은 count head 경로이므로 val 게이트와 비교 가능한 test 값은
    'count' 다. 'generated' 는 T1 단독 보고 경로, 'best'(=alloc) 는 폐기된 그룹정보 경로다.
    """
    if not vec_path.exists():
        return {k: None for k in ("generated", "count", "alloc")} | {"available": False, "reason": NO_VEC}
    z = np.load(vec_path, allow_pickle=False)
    got = [str(x) for x in z["subjects"]]
    assert got == subs, f"벡터 npz subject 순서 불일치: {got[:3]} vs {subs[:3]}"
    G = z["gt"]
    assert G.ndim == 2 and G.shape[0] == len(subs) and np.isfinite(G).all(), G.shape
    out = {"available": True, "n_subjects": len(subs), "n_edges": int(G.shape[1])}
    for name, key in (("generated", "pred_generated"), ("count", "pred_count"), ("alloc", "pred_best")):
        if key not in z.files:
            out[name] = None
            continue
        P = z[key]
        assert P.shape == G.shape and np.isfinite(P).all(), (name, P.shape)
        sp = subject_specificity(P, G)
        out[name] = {k: sp[k] for k in ("resid_r", "pred_degenerate", "inter_subj_r_pred",
                                        "inter_subj_r_gt", "n_subjects")}
    return out


def val_reference(path: Path) -> dict | None:
    """학습 중 val resid_r 궤적. run.py 는 count head 예측을 쓰므로 generated 경로와 다른 추정량이다."""
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {"source": str(path).split("ATM/")[-1],
            "estimator": "training/run.py individual_metrics: count head(edge_log_counts) + LOO 중심화",
            "by_phase": [{k: r.get(k) for k in ("phase", "checkpoint", "n_indiv_subj", "resid_r",
                                                "pred_degenerate", "inter_subj_r",
                                                "abl_own_r", "abl_zero_r", "abl_ordered")} for r in rows]}


def _dist(v: list[float]) -> dict:
    """subject 별 분포. 평균만 보면 한두 명이 끄는 경우를 못 본다."""
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
    assert a.size > 0, "유한한 subject 값이 하나도 없다"
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(a.min()), "max": float(a.max()), "n_subjects": int(a.size)}


def load_gt(subs: list[str]) -> dict[str, np.ndarray]:
    """.mat 을 한 번만 읽어 subject -> pass-SC [R,R]. load_mat_gt 는 호출마다 재파싱한다."""
    import scipy.io as sio
    m = sio.loadmat(SC_MAT, variable_names=["data"])
    tbl = {str(r["subject"][0]): np.asarray(r["SC_weight"], np.float64) for r in m["data"][0]}
    out = {}
    for s in subs:
        assert s in tbl, f"{s} 가 {SC_MAT.name} 에 없다"
        w = tbl[s]
        assert w.ndim == 2 and w.shape[0] == w.shape[1], w.shape
        assert np.isfinite(w).all() and w.sum() > 0, f"{s}: 빈/비유한 GT SC"
        out[s] = w
    return out


def main(a):
    ev = ROOT / a.eval
    assert ev.exists() and ev.stat().st_size > 0, f"{ev} 없음/0바이트"
    d = json.loads(ev.read_text())
    res, summ = d["per_subject"], d["summary"]
    subs = [r["subject"] for r in res]
    assert len(res) >= 3, f"subject {len(res)}명 (3명 이상 필요)"
    assert len(set(subs)) == len(subs), "subject 중복"

    # --- 1. per-subject 재집계 ------------------------------------------------
    per = {}
    for g in GROUPS:
        for k in ("r", "r_log"):
            vals = [r["generated"][g][k] for r in res]
            assert any(np.isfinite(x) for x in vals), f"generated/{g}/{k} 가 전부 비유한"
            per.setdefault(g, {})[k] = _dist(vals)
        per[g]["n_edges"] = _dist([r["generated"][g]["n_edges"] for r in res])

    # 기존 summary(평균) 를 그대로 재현하는지 -- 재집계가 다른 것을 세고 있으면 여기서 터진다
    repro = {}
    for g in GROUPS:
        ref, got = summ["generated"][g]["r"], per[g]["r"]["mean"]
        assert np.isclose(ref, got, atol=1e-9), (g, ref, got)
        repro[g] = {"reported": float(ref), "recomputed": float(got), "abs_diff": abs(float(ref - got))}

    # --- 2. 마스크 검증 (.mat GT 로 pair 수 재계산) ---------------------------
    gt = load_gt(subs)
    n_roi = gt[subs[0]].shape[0]
    bm = block_masks(n_roi)
    iu = np.triu_indices(n_roi, 1)
    mask_check = {}
    for g in BLOCKS:
        n_re = int(bm[g][iu].sum())
        n_js = int(round(per[g]["n_edges"]["mean"]))
        assert n_re == n_js, f"block {g}: 재계산 {n_re} vs JSON {n_js}"
        mask_check[g] = {"recomputed": n_re, "json": n_js}
    for g in TIERS:
        n_re = [int(tier_masks(gt[s])[g][iu].sum()) for s in subs]
        n_js = [int(r["generated"][g]["n_edges"]) for r in res]
        assert n_re == n_js, f"tier {g}: 첫 불일치 {[(x, y) for x, y in zip(n_re, n_js) if x != y][:3]}"
        mask_check[g] = {"recomputed_mean": float(np.mean(n_re)), "json_mean": float(np.mean(n_js)),
                         "exact_match_subjects": len(subs)}
    assert sum(mask_check[b]["recomputed"] for b in BLOCKS) == len(iu[0])

    # --- 3. 템플릿 기준선 (모델 없음) ----------------------------------------
    # train 평균 GT 를 '예측' 으로 넣는다. 개인을 전혀 안 봐도 전체 r 이 높게 나오면
    # 전체 r 은 성능 지표가 아니라는 뜻이다.
    tsubs = [l.strip() for l in (ROOT / a.template_subjects).read_text().splitlines() if l.strip()]
    tgt = load_gt([s for s in tsubs if s not in set(subs)])
    assert len(tgt) >= 3, f"템플릿 subject {len(tgt)}명"
    tmpl = np.mean(np.stack(list(tgt.values())), 0)
    assert tmpl.shape == (n_roi, n_roi) and np.isfinite(tmpl).all()
    tb = [stratified_corr(tmpl, gt[s], n_roi=n_roi) for s in subs]
    base = {"all": _dist([x["all"] for x in tb]), "all_log": _dist([x["all_log"] for x in tb]),
            "tier": {t: _dist([x["tier"][t] for x in tb]) for t in TIERS},
            "block": {b: _dist([x["block"][b] for x in tb]) for b in BLOCKS},
            "n": {k: float(np.mean([x["n"][k] for x in tb])) for k in GROUPS},
            "n_template_subjects": len(tgt)}

    # --- 4. LOO 잔차 상관 (경로별) + val 기준값 ------------------------------
    loo = loo_resid(ROOT / a.vectors, subs)
    valref = val_reference(ROOT / a.val_metrics)
    res_tpl = _dist([r["residual_r"] for r in res])          # generated, 템플릿 중심화 (LOO 아님)

    out = {
        "cmd": f"python scripts/40_recheck_metrics.py --eval {a.eval} "
               f"--template-subjects {a.template_subjects} --vectors {a.vectors} "
               f"--val-metrics {a.val_metrics}",
        "source_eval": a.eval, "checkpoint": summ["checkpoint"], "split": summ["split"],
        "n_subjects": len(res), "n_roi": n_roi,
        "note": ("전체 pair r 은 tier 간 배율차가 만드는 허수라 diagnostic 이다. "
                 "주지표는 tier 내부 r (특히 small/mid). 수치는 T1 단독 generated 경로."),
        "primary": {"note": "T1 단독 generated 경로만.",
                    "tier_r": {t: per[t]["r"]["mean"] for t in TIERS},
                    "tier_r_log": {t: per[t]["r_log"]["mean"] for t in TIERS},
                    "tier_n": {t: per[t]["n_edges"]["mean"] for t in TIERS},
                    "block_r": {b: per[b]["r"]["mean"] for b in BLOCKS},
                    "block_n": {b: per[b]["n_edges"]["mean"] for b in BLOCKS},
                    # 세 resid 값. 정의가 서로 달라 그대로 비교하면 안 된다.
                    "residual_r_generated": summ.get("residual_r"),          # 템플릿 중심화, generated
                    "residual_r_generated_dist": res_tpl,
                    "resid_r_loo_generated": (loo.get("generated") or {}).get("resid_r")
                                             if isinstance(loo.get("generated"), dict) else None,
                    "resid_r_loo_generated_detail": loo.get("generated"),
                    "resid_r_loo_generated_reason": None if loo["available"] else loo["reason"],
                    "resid_r_loo_count": (loo.get("count") or {}).get("resid_r")
                                         if isinstance(loo.get("count"), dict) else None,
                    "resid_r_loo_count_note": ("training/run.py 의 val resid_r 과 같은 추정량"
                                               "(count head + LOO). val 게이트와 비교 가능한 유일한 test 값."),
                    "edge_f1": summ["generated"]["all"]["edge_f1"]},
        "diagnostic": {"note": "절대 스케일 진단값 + 폐기 경로. 성능 지표 아님.",
                       "all_r": per["all"]["r"]["mean"], "all_r_log": per["all"]["r_log"]["mean"],
                       "all_n": per["all"]["n_edges"]["mean"],
                       "ccc": summ["generated"]["all"]["ccc"], "rmse": summ["generated"]["all"]["rmse"],
                       "mae": summ["generated"]["all"]["mae"],
                       "alloc_baseline_r": (summ["alloc"]["all"]["r"] if summ.get("alloc") else None),
                       "alloc_baseline_r_note": DEPRECATED,
                       "alloc_sum_ratio": summ.get("alloc_sum_ratio"),
                       "resid_r_loo_alloc": (summ.get("subject_specificity") or {}).get("resid_r"),
                       "resid_r_loo_alloc_note": DEPRECATED},
        "val_reference": valref,
        "per_group_distribution": per,
        "reproduction_vs_reported": repro,
        "mask_check": mask_check,
        "group_template_baseline": base,
    }
    dst = ROOT / "outputs" / "eval" / "s0a_metrics_recheck.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"소스: {a.eval}  ({len(res)}명, {n_roi} ROI)")
    print(f"[diagnostic] 전체 pair r = {per['all']['r']['mean']:.4f} "
          f"(median {per['all']['r']['median']:.4f}, IQR {per['all']['r']['iqr']:.4f}, n={per['all']['n_edges']['mean']:.0f})")
    for t in TIERS:
        p_ = per[t]
        print(f"[primary] tier {t:6s} r = {p_['r']['mean']:.4f} "
              f"(median {p_['r']['median']:.4f}, IQR {p_['r']['iqr']:.4f}) · r_log = {p_['r_log']['mean']:.4f} · n = {p_['n_edges']['mean']:.1f}")
    for b in BLOCKS:
        p_ = per[b]
        print(f"[primary] block {b:8s} r = {p_['r']['mean']:.4f} "
              f"(median {p_['r']['median']:.4f}, IQR {p_['r']['iqr']:.4f}) · n = {p_['n_edges']['mean']:.0f}")
    lg = out["primary"]["resid_r_loo_generated"]
    print(f"[primary] residual_r(generated, 템플릿 중심화) = {res_tpl['mean']:.4f} "
          f"(median {res_tpl['median']:.4f}, IQR {res_tpl['iqr']:.4f})")
    print(f"[primary] resid_r_loo(generated) = {lg if lg is None else f'{lg:.4f}'}"
          + ("" if loo["available"] else f"  <- {loo['reason']}"))
    print(f"[primary] resid_r_loo(count, val 과 같은 추정량) = {out['primary']['resid_r_loo_count']}")
    print(f"[diagnostic] resid_r_loo(alloc, 폐기 경로) = {out['diagnostic']['resid_r_loo_alloc']:.4f}")
    if valref:
        print("[val 기준] " + " · ".join(f"{r['phase']} {r['resid_r']:.4f}" for r in valref["by_phase"]))
    print(f"[baseline] group template(train {base['n_template_subjects']}명) 전체 r = {base['all']['mean']:.4f} · "
          + " · ".join(f"{t} {base['tier'][t]['mean']:.4f}" for t in TIERS))
    print(f"저장: {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="outputs/eval/final_p4_joint_step3000.json")
    ap.add_argument("--template-subjects", default="outputs/splits/train.txt")
    ap.add_argument("--vectors", default="outputs/eval/final_p4_joint_step3000_vectors.npz",
                    help="subject 별 예측/GT 벡터 npz (29_final_evaluation.py 가 저장). LOO resid_r 재계산에 필요")
    ap.add_argument("--val-metrics", default="outputs/checkpoints/retrain/val_metrics.jsonl")
    sys.exit(main(ap.parse_args()))
