"""관문이 실제로 멈추는지. 문서에 기준만 있고 드라이버가 강제하지 않아 p1 의 route_f1
0.4956 (< 0.50) 이 그냥 통과했다 -- 그 재발을 막는 테스트다.

  pytest tests/test_gates.py -q
"""
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.evaluation.gates import bootstrap_ci, check_gates  # noqa: E402


def driver():
    """scripts/19_train_pipeline.py 는 숫자로 시작해 일반 import 가 안 된다."""
    spec = importlib.util.spec_from_file_location("train_pipeline", ROOT / "scripts" / "19_train_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G_HALT = {"metric": "resid_r", "op": ">=", "value": 0.10, "n_key": "n_indiv_subj", "min_n": 31, "action": "halt"}


# --- check_gates: 통과 / 미달 / 표본 부족 -------------------------------------------------
def test_gate_pass():
    ok, msgs = check_gates({"resid_r": 0.25, "n_indiv_subj": 31}, [G_HALT])
    assert ok and "-> PASS" in msgs[0]
    assert "+0.2500" in msgs[0] and "+0.1000" in msgs[0] and "n=31/min_n=31" in msgs[0]   # 값·기준·n


def test_gate_fail_halts():
    ok, msgs = check_gates({"resid_r": 0.05, "n_indiv_subj": 31}, [G_HALT])
    assert not ok and "-> FAIL" in msgs[0]


def test_gate_sample_shortage_is_not_a_pass():
    """p2 위양성의 직접 원인: val 8명 0.1235 는 기준을 넘지만 판정할 수 없는 표본이다."""
    ok, msgs = check_gates({"resid_r": 0.1235, "n_indiv_subj": 8}, [G_HALT])
    assert not ok, "표본 부족을 통과시키면 안 된다"
    assert "표본 부족" in msgs[0] and "값은 기준 이상" in msgs[0] and "n=8/min_n=31" in msgs[0]


def test_gate_warn_does_not_halt():
    ok, msgs = check_gates({"abl_gap": -0.0159}, [{"metric": "abl_gap", "op": ">", "value": 0.0, "action": "warn"}])
    assert ok and "-> FAIL" in msgs[0] and "[GATE:warn]" in msgs[0]


@pytest.mark.parametrize("metrics", [{"n_indiv_subj": 31},                       # 키 없음
                                     {"resid_r": None, "n_indiv_subj": 31},      # None
                                     {"resid_r": float("nan"), "n_indiv_subj": 31},
                                     {"resid_r": 0.2}])                          # n_key 없음
def test_missing_metric_fails_loudly(metrics):
    """'지표가 None 이라 게이트를 건너뛴다' 가 지금 고치는 버그다 -- 조용히 통과시키지 않는다."""
    with pytest.raises((AssertionError, KeyError)):
        check_gates(metrics, [G_HALT])


def test_bad_gate_spec():
    with pytest.raises(AssertionError):
        check_gates({"resid_r": 0.2}, [{"metric": "resid_r", "op": "~=", "value": 0.1}])
    with pytest.raises(AssertionError):
        check_gates({"resid_r": 0.2}, [{"metric": "resid_r", "op": ">=", "value": 0.1, "action": "email"}])


def test_config_gates_are_loadable():
    """configs/pipeline_retrain.yaml 의 게이트가 실제로 check_gates 에 먹는지."""
    P = yaml.safe_load((ROOT / "configs" / "pipeline_retrain.yaml").read_text())
    assert P["val_indiv_subjects"] >= 31, "resid_r 관문 min_n=31 을 만족할 수 없는 설정"
    fake = {"resid_r": 0.5, "route_f1": 0.9, "length_mm": 100.0, "abl_gap": 0.1,
            "sc_pass_r_w_delta_prev": 0.0, "n_val": 3, "n_indiv_subj": 31}
    for phase, gates in P["gates"].items():
        ok, msgs = check_gates(fake, gates)
        assert ok and len(msgs) == len(gates), phase
    assert not any(g["metric"] == "sc_pass_r_w" and g["value"] >= 0.88          # 폐기된 그룹정보 수치
                   for gs in P["gates"].values() for g in gs)


# --- bootstrap_ci -----------------------------------------------------------------------
def test_bootstrap_ci_narrows_with_n():
    x = np.random.default_rng(0).normal(0, 0.15, 400)       # resid_r 의 subject 별 산포와 비슷한 규모
    c31 = bootstrap_ci(x[:31], n_boot=2000)
    w8 = [bootstrap_ci(x[i * 8:(i + 1) * 8], n_boot=2000, seed=i)["width"] for i in range(10)]
    assert np.median(w8) > 1.8 * c31["width"], (np.median(w8), c31["width"])   # 이론 비 sqrt(31/8)=1.97
    assert min(w8) > c31["width"]
    assert c31["lo"] < c31["stat"] < c31["hi"] and c31["n"] == 31


def test_bootstrap_ci_rejects_nan():
    with pytest.raises(AssertionError):
        bootstrap_ci([0.1, float("nan"), 0.2])


# --- 드라이버: 미달이면 다음 phase 로 가지 않는다 -----------------------------------------
def _fake_pipeline(tmp: Path, resid: float, n_indiv: int, indiv_every: int = 0):
    """2 phase 짜리 가짜 파이프라인. run/validate 는 stub 이라 GPU 를 쓰지 않는다."""
    out = tmp / "ck"
    out.mkdir(parents=True, exist_ok=True)
    for i in (0, 1):
        (tmp / f"ph{i}.yaml").write_text("phase: stub\n")
    cfg = {"phases": [str((tmp / f"ph{i}.yaml").relative_to(ROOT)) for i in (0, 1)],
           "state_file": str((out / "pipeline_state.json").relative_to(ROOT)),
           "lock_file": str((out / "pipeline.lock").relative_to(ROOT)),
           "val_subjects": "outputs/splits/val.txt", "final_eval": False,
           "indiv_log_every": indiv_every,
           "gates": {"pA": [dict(G_HALT)]}}
    (tmp / "pipeline.yaml").write_text(yaml.safe_dump(cfg))
    m = driver()
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        p = kw["out_dir"] / f"{kw['phase']}_step{kw['max_steps']}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stub")
        return p

    def fake_build(cfg_dict):
        i = len(calls)
        return {"phase": ["pA", "pB"][i], "max_steps": 1, "out_dir": out / ["pA", "pB"][i],
                "cfg": types.SimpleNamespace(seed=0), "t1_source": "rigid"}

    m.run = fake_run
    m.C.build = fake_build
    m.validate = lambda *a, **k: {"resid_r": resid, "n_indiv_subj": n_indiv, "n_val": 3}
    return m, calls, out, tmp / "pipeline.yaml"


def _args(cfg, **kw):
    return types.SimpleNamespace(**{"config": str(cfg), "max_steps": None, "no_val": False,
                                    "reval": None, "ignore_gates": False, **kw})


@pytest.fixture
def tmp_under_root():
    d = ROOT / "outputs" / "_gate_test"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_driver_halts_on_failed_gate(tmp_under_root, capsys):
    m, calls, out, cfg = _fake_pipeline(tmp_under_root, resid=0.05, n_indiv=31)
    rc = m.main(_args(cfg))
    st = json.loads((out / "pipeline_state.json").read_text())
    assert rc == 4, "미달인데 0 으로 끝났다"
    assert len(calls) == 1, "게이트 미달 뒤에도 다음 phase 가 돌았다"
    assert st["phase_idx"] == 0 and st["done"] == [], "미달인데 phase 가 진행된 것으로 기록됐다"
    assert st["gate_halt"]["phase"] == "pA" and st["gate_halt"]["failed"], st["gate_halt"]
    assert "게이트 미달" in capsys.readouterr().out


def test_driver_halts_on_sample_shortage(tmp_under_root):
    """p2 재현: 값(0.1235)은 기준을 넘지만 n=8 -> 통과시키면 안 된다."""
    m, calls, out, cfg = _fake_pipeline(tmp_under_root, resid=0.1235, n_indiv=8)
    assert m.main(_args(cfg)) == 4 and len(calls) == 1
    st = json.loads((out / "pipeline_state.json").read_text())
    assert "표본 부족" in st["gate_halt"]["failed"][0]


def test_driver_passes_gate(tmp_under_root):
    m, calls, out, cfg = _fake_pipeline(tmp_under_root, resid=0.25, n_indiv=31, indiv_every=500)
    assert m.main(_args(cfg)) == 0
    st = json.loads((out / "pipeline_state.json").read_text())
    assert len(calls) == 2 and st["phase_idx"] == 2 and st["done"] == ["pA", "pB"]
    assert "gate_halt" not in st
    assert calls[0]["step_hook_every"] == 500 and callable(calls[0]["step_hook"])   # C7 배선


def test_driver_ignore_gates_warns_but_continues(tmp_under_root, capsys):
    m, calls, out, cfg = _fake_pipeline(tmp_under_root, resid=0.05, n_indiv=31)
    assert m.main(_args(cfg, ignore_gates=True)) == 0
    txt = capsys.readouterr().out
    assert len(calls) == 2 and "--ignore-gates" in txt and "!!!!" in txt
    assert json.loads((out / "pipeline_state.json").read_text())["gate_ignored"][0]["phase"] == "pA"


def test_driver_rejects_no_val_with_gates(tmp_under_root):
    m, _, _, cfg = _fake_pipeline(tmp_under_root, resid=0.25, n_indiv=31)
    with pytest.raises(AssertionError, match="no-val"):
        m.main(_args(cfg, no_val=True))


# --- C7: 학습 중 개인차 기록 훅 ------------------------------------------------------------
def test_indiv_hook_records_and_leaves_training_state_alone(tmp_under_root):
    m = driver()
    trace = tmp_under_root / "indiv_trace.jsonl"
    P = {"indiv_log_every": 500, "val_indiv_subjects": 31}
    hook, every = m.indiv_hook(P, ["s1", "s2"], trace, "p2_count", "rigid")
    assert every == 500 and hook is not None

    model = torch.nn.Linear(3, 3)
    model.train()

    def fake_indiv(mo, subs, n_subj=8, pair_dice=False, source="rigid", **kw):
        mo.eval()                                    # 모드/RNG 를 어지럽히는 계측을 흉내낸다
        torch.rand(5)
        assert not torch.is_grad_enabled(), "훅이 no_grad 밖에서 돌았다"
        return {"resid_r": 0.12, "abl_own_r": 0.9, "abl_shuf_r": 0.9, "abl_zero_r": 0.91,
                "abl_gap": -0.01, "inter_subj_r": 0.9999, "n_indiv_subj": n_subj}

    m.individuality_metrics = fake_indiv
    torch.manual_seed(0)
    before = torch.get_rng_state()
    w0 = model.weight.detach().clone()
    hook(500, model)
    assert torch.equal(torch.get_rng_state(), before), "훅이 RNG 를 소비했다 (학습 재현성 파괴)"
    assert model.training, "훅이 train/eval 모드를 되돌리지 않았다"
    assert torch.equal(model.weight.detach(), w0)
    row = json.loads(trace.read_text().splitlines()[0])
    assert row["phase"] == "p2_count" and row["step"] == 500 and row["n_indiv_subj"] == 31
    assert set(row) >= {"resid_r", "abl_own_r", "abl_shuf_r", "abl_zero_r", "abl_gap", "inter_subj_r"}

    m.individuality_metrics = lambda *a, **k: {}      # count head 없는 phase -> 기록 없음
    hook(1000, model)
    assert len(trace.read_text().splitlines()) == 1


def test_indiv_hook_off_by_default():
    m = driver()
    assert m.indiv_hook({}, ["s1"], Path("/dev/null"), "p0", "rigid") == (None, 0)


def test_run_accepts_step_hook():
    """드라이버가 넘기는 인자를 run() 이 실제로 받는지 (배선 확인)."""
    import inspect

    from atm_sc.training.run import run
    p = inspect.signature(run).parameters
    assert "step_hook" in p and "step_hook_every" in p and p["step_hook_every"].default == 0


def test_trend_line_reports_direction(tmp_under_root):
    m = driver()
    trace = tmp_under_root / "t.jsonl"
    trace.write_text("\n".join(json.dumps({"phase": "p4_joint", "step": s, "resid_r": r})
                               for s, r in [(500, 0.09), (3000, -0.03)]))
    st = {"history": [{"phase": "p3_mag", "resid_r": 0.009}, {"phase": "p4_joint", "resid_r": -0.0307}]}
    line = m._trend_line(st, trace, {"phase": "p4_joint", "resid_r": -0.0307}, ["resid_r"])
    assert "하강" in line and "p3_mag" in line and "step 500->3000" in line
