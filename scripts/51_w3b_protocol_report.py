#!/usr/bin/env python
"""W3-b: 평가 규약을 못박고, 모델과 기준선을 **같은 규약**으로 맞춰 표를 만든다.

배경 -- 이 프로젝트의 핵심 판정("모델이 남의 뇌보다 못하다")은 규약이 다른 두 숫자를 비교한
것이었다. 모델 wb dice 0.546 은 pred(≈14,500) vs gt(20,000) 이고, cross_subject 0.574 는
8,000 vs 8,000 이다. pair dice 0.095 는 16 vs ≤256 인데 그 "천장" 0.700 은 128 vs 128 이다.
크기가 다르면 dice 는 비교 불가일 뿐 아니라 흐릿한(변위된) 생성기에 상을 준다
(outputs/eval/w2c_c9_pair_dice_npred.json: n=16 은 σ=1mm 에서 dice +0.066).

여기서는 실행 결과만 읽어(생성 없음) 규약 정의(w3b_protocol.json)와 대응표
(w3b_baselines_matched.json)를 만든다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "outputs" / "eval"

# 기하 지표는 표본 개수에 강하게 의존한다. 아래 세 축이 "규약" 이다.
PROTOCOL = {
    "version": "w3b-2026-09-07",
    "why": ("규약 없는 dice 는 인용 금지. 같은 체크포인트가 규약에 따라 wb 0.55~0.61, "
            "pair 0.13~0.37 로 갈린다. 모든 수치는 이 파일의 규약 id 를 달고 보고한다."),
    "common": {
        "voxel_mm": 2.0,
        "dice": ("bundle_geometry_metrics(pred, gt, voxel_mm): 두 bundle 공통 원점 기준 "
                 "floor(mm/voxel_mm) 복셀 집합의 2|A∩B|/(|A|+|B|). "
                 "coverage=|A∩B|/|B|, overreach=|A\\B|/|B| (Tractometer OR)."),
        "gt_source": "outputs/roi_pairs/<sub>/bundles.npz (ROI pair 당 cap 256 가닥, 128점 재샘플)",
        "space": "DSI Studio QSDR 템플릿 (≈MNI152NLin6), 좌표 단위 mm",
        "atlas": "DK-PD25 82 ROI, NLin6 2mm. block 2145/1056/120 = 3321 (N_CTX=66)",
        "self_check": "bundle_geometry_metrics(X, X).dice == 1.0 (실측 확인)",
    },
    "wb_dice": {
        "id": "wb@N",
        "rule": "n_pred == n_gt == N. 둘 중 하나라도 다르면 다른 규약이다.",
        "model_cmd": ("scripts/29_final_evaluation.py --trk-eval --trk-gen-sample N "
                      "--trk-gt-sample N  (gen/gt 모두 default_rng(0) 표본)"),
        "baseline_cmd": "scripts/43_geometry_baselines.py --n-sample N (gt 도 default_rng(0))",
        "N_reported": [8000, 20000],
        "N_note": ("N=8000 은 n_per_pair 8 과 16 이 둘 다 낼 수 있는 유일한 공통점이다 "
                   "(n8 은 subject 당 ≈14,500 가닥밖에 안 만든다). N=20000 은 n16 전용."),
        "retired": {"0.546": "pred≈14,500 vs gt=20,000 (크기 불일치). 인용 금지.",
                    "0.608": "20,000 vs 20,000 = wb@20000. 8,000 규약 기준선과 비교 금지."},
    },
    "pair_dice": {
        "id": "pair@n",
        "rule": ("pred(n) vs gtA(n), 천장 = gtB(n) vs gtA(n). gtA/gtB 는 GT 번들을 섞어 나눈 "
                 "겹치지 않는 두 조각. n = min(n_req, 예측 가닥, len(gt)//2)."),
        "n_req": 64,
        "n_req_why": ("iid 변위 실측에서 dice 가 변위에 단조 감소하는 최소 n 이다 "
                      "(n=8/16/32 는 σ=0.5~1.5mm 에서 dice 가 올라간다 = 흐릿한 생성기에 상). "
                      "천장 분해능도 n=64 가 낫다. outputs/eval/w2c_c9_pair_dice_npred.json"),
        "model_cmd": "scripts/29_final_evaluation.py --trk-pair-n 64 (--trk-pair-seed 1 로 별도 생성)",
        "baseline_cmd": "scripts/43_geometry_baselines.py --pair-n 64 -> per_pair.*_matched 행",
        "pairs": "GT pair_count_full 상위 --trk-pairs(50) 중 edge head 가 실제로 고른 pair",
        "retired": {"0.0946 / 0.095": "pred=16 vs gt≤256 (천장은 128 vs 128). 규약 불일치, 인용 금지.",
                    "0.700 / 0.711": "128 vs 128 천장. 위 pred 와 저울이 다르다.",
                    "0.167": "규약 기록 없음. 같은 체크포인트에서 pair_half 0.328 / pair_self 0.370."},
    },
    "sc_r": {
        "id": "sc@nstr",
        "rule": ("pass-SC = 같은 streamline 이 지난 모든 ROI 쌍 (Case B, hard_sc(mode='pass')). "
                 "상삼각 3321 성분의 Pearson r. SC 는 개수 지표라 제출 가닥 수에 직접 의존한다."),
        "model": ("생성된 **전체** 가닥에서 계산한다 (whole-brain dice 처럼 잘라내지 않는다). "
                  "n_streamlines = n_per_pair x 선택 pair 수 ≈ 14,500(n8) / 29,000(n16)."),
        "baseline": "scripts/43_geometry_baselines.py --sc --n-sample N -> N 가닥으로 계산",
        "warning": ("모델 0.713(≈29,000 가닥) 과 기준선 0.841(8,000 가닥) 은 가닥 수가 다르다. "
                    "다만 실측상 SC r 은 가닥 수에 거의 무관하다: cross_subject 0.8408(8,000) -> "
                    "0.8434(20,000), Δ=+0.0026. dice 와 달리 SC 비교는 이 축에 안 흔들린다."),
        "n_sample_sensitivity": {"cross_subject": {"8000": 0.8408, "20000": 0.8434},
                                 "group_tractogram": {"8000": 0.9142, "20000": 0.9181},
                                 "self": {"8000": 0.9360, "20000": 0.9361}},
        "checks": "SC 82x82, 대칭, 대각 0, NaN/Inf 0, block 2145/1056/120",
    },
    "n_per_pair": {
        "value_frozen": 16,
        "why": ("전역 생성 개수. 16 -> 64 로 올리면 whole-brain 가닥이 29k -> 116k 가 되어 "
                "generated SC 전부가 과거 실행·기준선과 비교 불가능해진다. pair 단위 dice 는 "
                "--trk-pair-n 으로 따로 올린다 (평가 대상 ≤50 쌍만, +3,200 가닥)."),
        "n8_note": "0.546 계열 과거 수치와의 연속성 확인용으로만 병기한다.",
    },
    "subjects": {
        "model": "outputs/splits/test.txt 31명 전원",
        "baseline": ("test 31명에서 seed 20250906 순열로 만든 10개 (A=예측 -> B=정답) 쌍. "
                     "target 10명이 모델의 31명 부분집합이라 subject 집합은 완전히 같지 않다 "
                     "-- 규약 일치는 ceiling(천장) 값이 양쪽에서 같은지로 확인한다."),
        "group_tractogram": "train.txt 에서 seed+1 로 뽑은 8명 GT 합본 풀",
    },
}


def load(p: Path):
    assert p.exists(), f"{p} 없음 -- 먼저 실행할 것"
    assert p.stat().st_size > 0, f"{p} 가 0 바이트"
    return json.loads(p.read_text())


def model_row(tag: str) -> dict:
    s = load(EV / f"w3b_{tag}.json")["summary"]
    pr = s["protocol"]
    assert pr["trk_gen_sample"] == pr["trk_gt_sample"], ("wb 규약 불일치", pr)
    g, pp, ce = s["trk_geometry"], s["trk_per_pair"], s["trk_per_pair_ceiling"]
    assert abs(pp["n_per_side"] - ce["n_per_side"]) < 1e-9, "pair pred/천장 크기 불일치"
    return {"tag": tag, "checkpoint": s["checkpoint"], "n_per_pair": pr["n_per_pair"],
            "N_wb": pr["trk_gen_sample"], "pair_n": pr["trk_pair_n"],
            "n_streamlines": s["n_streamlines_mean"], "n_subjects": s["n_subjects"],
            "wb_dice": g["dice"], "wb_coverage": g["coverage"], "wb_overreach": g["overreach"],
            "pair_dice": pp["dice"], "pair_coverage": pp["coverage"], "pair_overreach": pp["overreach"],
            "pair_ceiling": ce["dice"], "pair_n_per_side": pp["n_per_side"], "pair_n_pairs": pp["n_pairs"],
            "sc_r": s["generated"]["all"]["r"], "sc_r_log": s["generated"]["all"]["r_log"],
            "sc_tier_r": {t: s["stratified"]["tier"][t] for t in ("small", "mid", "large")},
            "valid_conn": (s["connection"] or {}).get("valid_conn"),
            "endpoint_in_roi": (s["connection"] or {}).get("endpoint_in_roi")}


def baseline_rows(geom_path: Path) -> dict:
    d = load(geom_path)
    m = d["meta"]
    wb = {k: {"dice": v["dice"], "dice_sd": v.get("dice_sd"), "coverage": v["coverage"],
              "overreach": v["overreach"], "n_obs": v["n_obs"]} for k, v in d["wb_pairs"].items()}
    assert _self_is_top(wb), "self 기준선이 최고가 아니다: " + \
        str({k: round(v["dice"], 3) for k, v in wb.items()})
    pp = {k: {"dice": v["dice"], "coverage": v["coverage"], "n_pred": v["n_pred"],
              "n_gt": v["n_gt"], "n_obs": v["n_obs"]}
          for k, v in d["per_pair"].items() if k.endswith("_matched")}
    for k, v in pp.items():
        assert v["n_pred"] == v["n_gt"], (k, v)          # 크기 일치 규약의 핵심
    return {"N_wb": m["n_sample"], "pair_n": m.get("pair_n"), "seed": m["seed"],
            "n_subject_pairs": len(m["pairs"]), "wb": wb, "pair_matched": pp,
            "wb_raw": ({k: {"dice": v["dice"], "coverage": v["coverage"]}
                        for k, v in d["wb_raw"].items()} if d.get("wb_raw") else None),
            "count_sweep": d.get("count_sweep")}


def _self_is_top(wb):
    """self 가 천장이어야 한다. subsample 은 같은 subject 의 더 큰 표본이라 근소하게 넘을 수 있다."""
    top = max(wb, key=lambda k: wb[k]["dice"])
    return top in ("self", "subsample") and wb["self"]["dice"] > wb["cross_subject"]["dice"]


def run_checks() -> dict:
    """필수 assert. 조용히 틀리는 것을 막는 게 목적이다 (숫자를 표에 남긴다)."""
    sys.path.insert(0, str(ROOT / "src"))
    from atm_sc.data.dataset import ROIPairSubject                          # noqa: E402
    from atm_sc.data.roi_groups import BLOCKS, N_CTX, block_masks           # noqa: E402
    from atm_sc.evaluation.balance_metrics import bundle_geometry_metrics   # noqa: E402

    sub = [l.strip() for l in (ROOT / "outputs/splits/test.txt").read_text().splitlines() if l.strip()][0]
    z = np.load(ROOT / "outputs/roi_pairs" / sub / "bundles.npz")
    X = z["streamlines"][:2000].astype(np.float32)
    assert X.shape[1:] == (128, 3) and len(X) == 2000, X.shape
    g = bundle_geometry_metrics(X, X, voxel_mm=PROTOCOL["common"]["voxel_mm"])
    assert g["dice"] == 1.0 and g["coverage"] == 1.0 and g["overreach"] == 0.0, g   # sigma=0 -> 1.0

    subj = ROIPairSubject(sub)
    W = np.asarray(subj.sc_mat, np.float64)
    assert W.shape == (82, 82), W.shape
    assert np.isfinite(W).all(), int((~np.isfinite(W)).sum())
    assert np.array_equal(W, W.T), "SC 가 대칭이 아니다"
    assert np.abs(np.diag(W)).max() == 0, "SC 대각이 0 이 아니다"
    iu = np.triu_indices(82, 1)
    bm = {b: v[iu] for b, v in block_masks(82).items()}     # 상삼각만 세야 2145/1056/120
    nblk = [int(bm[b].sum()) for b in BLOCKS]
    assert nblk == [2145, 1056, 120] and sum(nblk) == 3321 and N_CTX == 66, (nblk, N_CTX)

    npz = {}
    for p_ in sorted(EV.glob("w3b_*_vectors.npz")):
        v = np.load(p_)
        assert v["gt"].shape[1] == 3321, (p_.name, v["gt"].shape)
        assert np.isfinite(v["pred_generated"]).all(), f"{p_.name}: 예측에 NaN/Inf"
        npz[p_.name] = list(v["pred_generated"].shape)
    return {"self_dice_sigma0": g["dice"], "sc_shape": list(W.shape), "sc_symmetric": True,
            "sc_diag_zero": True, "sc_nan_inf": 0, "block_n": dict(zip(BLOCKS, nblk)),
            "block_total": sum(nblk), "n_ctx": N_CTX, "pred_vector_shapes": npz,
            "subject_checked": sub}


def sc_table() -> dict:
    """SC r 은 제출 가닥 수에 직접 의존한다. 모델(≈29,000) 과 기준선의 가닥 수를 나란히 둔다."""
    out = {}
    for n in (8000, 20000, 29000):
        p = EV / f"w3b_sc_n{n}.json"
        if not p.exists():
            continue
        d = load(p)
        rows = {}
        for b in d["table"]["baselines"]:
            if b["family"] == "pairs":
                rows[b["baseline"]] = {"sc_r": b["sc_r"], "sc_r_sd": b.get("sc_r_sd"),
                                       "sc_r_log": b.get("sc_r_log"), "tier": b["tier"],
                                       "n_streamlines": b.get("n_streamlines")}
        out[str(n)] = {"n_sample": d["meta"]["n_sample"], "pairs_family": rows,
                       "group_sc_matrix_r": d["table"]["group_sc_matrix"]["sc_r"]}
    return out


def verdict(table) -> dict:
    """핵심 판정: 규약을 맞춘 뒤에도 모델이 cross_subject 를 넘는가."""
    beats = [r for r in table if r["beats_cross_subject"]]
    return {"question": "모델 whole-brain dice > cross_subject 기준선?",
            "answer": ("모든 규약에서 넘지 못한다 (기존 판정 유지)" if not beats else
                       "일부 규약에서 넘는다: " + ", ".join(r["model"] for r in beats)),
            "n_configs": len(table), "n_beating": len(beats),
            "margins": {r["model"]: round(r["margin"], 4) for r in table},
            "note": ("N_wb 를 8,000 -> 20,000 으로 올리면 모델 0.546 -> 0.608 이지만 "
                     "cross_subject 도 0.574 -> 0.629 로 같이 오른다. 격차는 그대로다.")}


def main():
    (EV / "w3b_protocol.json").write_text(json.dumps(PROTOCOL, indent=1, ensure_ascii=False))
    print(f"규약 저장: outputs/eval/w3b_protocol.json")

    tags = [t for t in ("p4_n8_wb8k", "p4_n16_wb8k", "d1_n8_wb8k", "d1_n16_wb8k",
                        "p4_n16_wb20k", "d1_n16_wb20k") if (EV / f"w3b_{t}.json").exists()]
    tags += sorted(p.stem[4:] for p in EV.glob("w3b_w3a_*.json"))     # W3-a 체크포인트가 있으면 자동 포함
    models = [model_row(t) for t in tags]
    bases = {}
    for n in (8000, 20000):
        p = EV / f"w3b_geom_n{n}.json"
        if p.exists():
            bases[str(n)] = baseline_rows(p)
    scs = {}
    for n in (8000, 20000):
        p = EV / f"w3b_sc_n{n}.json"
        if p.exists():
            d = load(p)
            scs[str(n)] = {"table": d["table"], "n_sample": d["meta"]["n_sample"]}

    # --- 같은 규약 대응표 -----------------------------------------------------
    table = []
    for m in models:
        b = bases.get(str(m["N_wb"]))
        if b is None:
            continue
        cs = b["wb"]["cross_subject"]["dice"]
        table.append({"model": m["tag"], "N_wb": m["N_wb"], "n_per_pair": m["n_per_pair"],
                      "model_wb_dice": m["wb_dice"],
                      "cross_subject": cs, "cross_subject_sd": b["wb"]["cross_subject"]["dice_sd"],
                      "group_tractogram": b["wb"]["group_tractogram"]["dice"],
                      "self_ceiling": b["wb"]["self"]["dice"],
                      "beats_cross_subject": bool(m["wb_dice"] > cs),
                      "margin": m["wb_dice"] - cs,
                      "model_pair_dice": m["pair_dice"], "model_pair_ceiling": m["pair_ceiling"],
                      "baseline_pair_cross_subject": (b["pair_matched"].get("cross_subject_matched") or {}).get("dice"),
                      "baseline_pair_ceiling": (b["pair_matched"].get("ceiling_matched") or {}).get("dice")})

    checks = run_checks()
    # 규약 일치의 직접 증거: 같은 pair@n 천장을 두 스크립트가 각자 계산했는데 값이 같아야 한다.
    for m in models:
        for n, b in bases.items():
            ce = (b["pair_matched"].get("ceiling_matched") or {}).get("dice")
            if ce is not None:
                d = abs(m["pair_ceiling"] - ce)
                assert d < 0.05, (f"pair@{m['pair_n']} 천장이 두 스크립트에서 다르다 "
                                  f"(29: {m['pair_ceiling']:.4f} vs 43: {ce:.4f}) -- 규약 불일치")
    checks["pair_ceiling_cross_check"] = {
        "scripts_29": models[0]["pair_ceiling"] if models else None,
        "scripts_43": (bases.get("8000", {}).get("pair_matched", {}).get("ceiling_matched") or {}).get("dice"),
        "note": ("두 스크립트가 독립적으로 계산한 같은 규약의 천장. 값이 같으면 채점 규약이 "
                 "같다는 직접 증거다 (subject 집합이 31명 vs 10명이라 정확히 같지는 않다).")}
    out = {"protocol_ref": "outputs/eval/w3b_protocol.json", "protocol_version": PROTOCOL["version"],
           "checks": checks, "models": models, "baselines": bases, "baselines_sc": scs,
           "sc_matched_table": sc_table(), "matched_table": table,
           "verdict": verdict(table)}
    p = EV / "w3b_baselines_matched.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    assert p.stat().st_size > 0
    print(f"대응표 저장: outputs/eval/w3b_baselines_matched.json\n")

    print(f"{'모델':<14}{'npp':>4}{'N_wb':>7}{'wb dice':>9}{'cross':>8}{'group':>8}{'self':>8}"
          f"{'넘는가':>8}{'pair':>7}{'천장':>7}")
    for r in table:
        print(f"{r['model']:<14}{r['n_per_pair']:>4}{r['N_wb']:>7}{r['model_wb_dice']:>9.3f}"
              f"{r['cross_subject']:>8.3f}{r['group_tractogram']:>8.3f}{r['self_ceiling']:>8.3f}"
              f"{('YES' if r['beats_cross_subject'] else 'NO'):>8}"
              f"{r['model_pair_dice']:>7.3f}{r['model_pair_ceiling']:>7.3f}")
    print("\n[SC r -- 가닥 수를 나란히]")
    for n, t in out["sc_matched_table"].items():
        line = "  ".join(f"{k}={v['sc_r']:.3f}" for k, v in t["pairs_family"].items())
        ref = next((m for m in models if m["tag"] == "p4_n16_wb8k"), models[0])
        print(f"  기준선 {int(n):>6,} 가닥: {line}   "
              f"(모델 {ref['tag']} {ref['n_streamlines']:,.0f} 가닥: {ref['sc_r']:.3f})")
    print("\n[판정] " + out["verdict"]["answer"])
    for n, b in bases.items():
        print(f"\n[pair@{b['pair_n']} 기준선, N_wb={n}]  " +
              "  ".join(f"{k.replace('_matched',''):}={v['dice']:.3f}(n={v['n_pred']:.0f})"
                        for k, v in b["pair_matched"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
