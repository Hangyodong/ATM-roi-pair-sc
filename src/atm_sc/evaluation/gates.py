"""Phase 관문 판정 (`docs/PIPELINE_08_RETRAIN_DESIGN.md` §4).

문서에 기준만 적고 드라이버가 강제하지 않으면 기준은 없는 것과 같다. 실측 피해:

- p1 의 검증 `route_f1` 0.4956 (< 0.50) 이 그대로 통과해 p2 로 넘어갔다.
- p2 의 `resid_r` 0.1235 는 **val 8명**에서 나온 값이다. 같은 지표가 test 31명에서는
  -0.0107 이었다. 값만 보고 표본 수를 안 보면 이런 위양성을 못 막는다.

그래서 `check_gates` 는 값과 **표본 수를 함께** 본다. 표본이 모자라면 "통과" 가 아니라
"표본 부족" 이다. 필요한 최소 n 은 `scripts/40_gate_ci.py` 의 부트스트랩으로 정한다.

지표가 없거나 NaN 이면 조용히 건너뛰지 않고 즉시 실패한다 — "지표가 None 이라 게이트를
건너뛴다" 가 바로 지금 고치는 버그다.
"""
from __future__ import annotations

import operator

import numpy as np

OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}
ACTIONS = ("halt", "warn")


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """subject 단위 부트스트랩 신뢰구간.

    values 는 **subject 별 값** 이어야 한다 (edge 별이 아니다). n 이 작을 때 지표가 얼마나
    못 미더운지를 구간 폭으로 보여 준다: 같은 지표라도 n=8 이면 폭이 n=31 의 약 2배다.
    """
    x = np.asarray(values, dtype=float).ravel()
    assert x.size > 0, "부트스트랩할 값이 없다"
    assert np.isfinite(x).all(), f"NaN/Inf 가 {int((~np.isfinite(x)).sum())}개 있다"
    assert 0.0 < alpha < 1.0, alpha
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    boot = np.asarray([float(stat(x[i])) for i in idx])
    lo, hi = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    return {"stat": float(stat(x)), "lo": lo, "hi": hi, "width": hi - lo, "half_width": (hi - lo) / 2,
            "se": float(boot.std(ddof=1)), "n": int(x.size), "n_boot": int(n_boot),
            "alpha": float(alpha), "seed": int(seed)}


def _number(metrics: dict, key: str, gate: dict):
    """지표를 꺼낸다. 없거나 None/NaN 이면 통과시키지 않고 여기서 죽는다."""
    assert key in metrics, (f"게이트 지표 '{key}' 가 val 지표에 없다 (gate={gate}). "
                            f"있는 키: {sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))}")
    v = metrics[key]
    assert isinstance(v, (int, float)) and not isinstance(v, bool), f"지표 '{key}' 가 숫자가 아니다: {v!r}"
    assert np.isfinite(v), f"지표 '{key}' 가 NaN/Inf 다 ({v}). 게이트를 건너뛰지 않는다."
    return float(v)


def check_gates(metrics: dict, gates: list[dict]) -> tuple[bool, list[str]]:
    """gates = [{"metric": "resid_r", "op": ">=", "value": 0.10, "min_n": 25,
                 "n_key": "n_indiv_subj", "action": "halt"|"warn", "note": "..."}]

    반환: (halt 게이트가 전부 통과했는가, 판정 메시지들). 메시지에는 실제값·기준값·n·판정이
    모두 들어간다. `min_n` 미만이면 값이 기준을 넘어도 통과가 아니라 "표본 부족" 이다.
    """
    assert isinstance(gates, list), f"gates 는 리스트여야 한다: {type(gates)}"
    ok, msgs = True, []
    for g in gates:
        assert isinstance(g, dict), f"게이트 항목이 dict 가 아니다: {g!r}"
        op, action = g["op"], g.get("action", "halt")
        assert op in OPS, f"알 수 없는 비교 연산 '{op}' (가능: {sorted(OPS)})"
        assert action in ACTIONS, f"알 수 없는 action '{action}' (가능: {ACTIONS})"
        name, thr = g["metric"], float(g["value"])
        val = _number(metrics, name, g)
        min_n = int(g.get("min_n", 0))
        n_key = g.get("n_key", "n_val")
        n = int(_number(metrics, n_key, g)) if min_n > 0 else None
        value_ok = bool(OPS[op](val, thr))
        enough = n is None or n >= min_n
        if not enough:
            verdict = "표본 부족" + ("(값은 기준 이상)" if value_ok else "(값도 미달)")
        else:
            verdict = "PASS" if value_ok else "FAIL"
        passed = value_ok and enough
        n_txt = "n=-" if n is None else (f"n={n}" if min_n == 0 else f"n={n}/min_n={min_n}")
        msg = f"[GATE:{action}] {name} = {val:+.4f} (기준 {op} {thr:+.4f}) {n_txt} -> {verdict}"
        if g.get("note"):                     # 왜 이 기준인지를 판정 옆에 같이 남긴다
            note = " ".join(str(g["note"]).split())
            msg += (f"\n{' ' * 13}# {note}" if len(note) > 60 else f"  # {note}")
        msgs.append(msg)
        if not passed and action == "halt":
            ok = False
    return ok, msgs
