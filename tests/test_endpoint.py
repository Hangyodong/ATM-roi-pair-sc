import numpy as np
import torch

from atm_sc.models.endpoint_assigner import EndpointAssigner, build_distance_maps


def test_distance_maps_valid(synth):
    dm = synth["dist"]
    assert dm.shape == (synth["n_roi"],) + synth["shape"]
    assert np.isfinite(dm).all()
    for r in range(synth["n_roi"]):                       # ROI 내부 거리는 0
        assert dm[r][synth["atlas"] == r + 1].max() == 0.0


def test_missing_roi_is_loud(synth):
    """ROI 가 소실되면 조용히 넘어가지 않고 즉시 실패해야 한다."""
    a = synth["atlas"].copy()
    a[a == 3] = 0
    try:
        build_distance_maps(a, synth["n_roi"], (2.0, 2.0, 2.0))
    except AssertionError:
        return
    raise AssertionError("소실된 ROI 를 감지하지 못함")


def test_probabilities_sum_to_one(synth, streamlines):
    mm, _ = streamlines
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu", d_bg=None)
    q = ea.point_probs(mm)
    assert q.shape == (mm.shape[0], mm.shape[1], synth["n_roi"])
    assert torch.isfinite(q).all()
    assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-5)
    assert (q >= 0).all()


def test_background_class_reduces_mass(synth, streamlines):
    mm, _ = streamlines
    kw = dict(affine=synth["affine"], tau=0.5, device="cpu")
    q0 = EndpointAssigner(synth["dist"], d_bg=None, **kw).point_probs(mm)
    q1 = EndpointAssigner(synth["dist"], d_bg=2.0, **kw).point_probs(mm)
    assert float(q1.sum(-1).max()) <= float(q0.sum(-1).min()) + 1e-5


def test_endpoint_probs_pick_right_roi(synth, streamlines):
    mm, pairs = streamlines
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.2, device="cpu", d_bg=None)
    qs, qe = ea.endpoint_probs(mm)
    assert qs.shape == qe.shape == (mm.shape[0], synth["n_roi"])
    i = torch.tensor([p[0] for p in pairs]); j = torch.tensor([p[1] for p in pairs])
    assert (qs.argmax(1) == i).float().mean() > 0.9
    assert (qe.argmax(1) == j).float().mean() > 0.9


def test_visit_probs_modes(synth, streamlines):
    mm, pairs = streamlines
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu", d_bg=2.0)
    for agg in ("max", "lse", "noisy_or"):
        u = ea.visit_probs(mm, aggregate=agg)
        assert u.shape == (mm.shape[0], synth["n_roi"])
        assert torch.isfinite(u).all() and (u >= 0).all() and (u <= 1 + 1e-5).all()
    u = ea.visit_probs(mm, aggregate="max")
    # 직선이 시작/끝 ROI 는 반드시 통과한다
    for k, (a, b) in enumerate(pairs):
        assert u[k, a] > 0.5 and u[k, b] > 0.5


def test_gradient_reaches_coordinates(synth, streamlines):
    mm, _ = streamlines
    mm = mm.clone().requires_grad_(True)
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu", d_bg=None)
    ea.endpoint_probs(mm)[0].sum().backward()
    assert mm.grad is not None and float(mm.grad.abs().max()) > 0


def test_far_endpoint_still_has_gradient(synth):
    """끝점이 목표 ROI 에서 멀어도 (확률 < 1e-8) log-확률 경로는 gradient 를 준다."""
    import torch
    from atm_sc.losses import endpoint_loss
    from atm_sc.spaces import voxel_to_mm
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu", d_bg=None)
    # ROI 0 근처에서 시작해 ROI 0 근처에서 끝나는 streamline; 목표는 반대편 ROI 5
    p0 = voxel_to_mm(np.array([3.0, 10.0, 9.0]), synth["affine"])
    mm = torch.tensor(np.repeat(p0[None, None], 128, axis=1), dtype=torch.float32).requires_grad_(True)
    tgt = torch.tensor([5]); src = torch.tensor([0])
    qs, qe = ea.endpoint_probs(mm)
    assert float(qs[0, 5]) < 1e-8                       # 확률은 언더플로
    l_prob = endpoint_loss(qs, qe, tgt, tgt)            # clamp 경로
    l_prob.backward(); g_prob = mm.grad.abs().max().item(); mm.grad = None
    lqs, lqe = ea.endpoint_log_probs(mm)
    l_log = endpoint_loss(lqs, lqe, tgt, tgt, log_input=True)
    l_log.backward(); g_log = mm.grad.abs().max().item()
    assert g_prob == 0.0, g_prob                        # 버그 재현: 확률 경로는 gradient 0
    assert g_log > 0.0 and torch.isfinite(l_log), g_log # 수정: log 경로는 살아있다
    assert float(l_log) > float(l_prob)                 # 실제 CE 가 clamp 값보다 크다
