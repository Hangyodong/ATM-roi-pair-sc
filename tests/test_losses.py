import torch

from atm_sc import losses as L
from atm_sc.models.endpoint_assigner import EndpointAssigner
from atm_sc.models.sc_builder import SCBuilder


def _gt(R=6, seed=0):
    g = torch.rand(R, R, generator=torch.Generator().manual_seed(seed)) * 100
    g = (g + g.T) / 2
    g.fill_diagonal_(0)
    g[g < 20] = 0
    return g


def test_corr_loss_is_scale_invariant():
    g = _gt()
    assert float(L.sc_corr_loss(g * 7.0, g)) < 1e-5
    assert 0.0 <= float(L.sc_corr_loss(torch.rand_like(g) * 100, g)) <= 2.0


def test_magnitude_loss_catches_scale_that_corr_misses():
    """v2 §11: 상관만 쓰면 스케일이 10배 틀려도 r=1 이다."""
    g = _gt()
    p = g * 10.0
    assert float(L.sc_corr_loss(p, g)) < 1e-5
    assert float(L.sc_magnitude_loss(p, g, normalize="none")) > 1.0
    assert float(L.sc_magnitude_loss(p, g, normalize="sum")) < 1e-4


def test_losses_finite_and_nonneg():
    g = _gt(); p = torch.rand_like(g) * 50; p = (p + p.T) / 2; p.fill_diagonal_(0)
    for v in (L.sc_corr_loss(p, g), L.sc_magnitude_loss(p, g)):
        assert torch.isfinite(v) and v.ndim == 0
    num = p * 40
    tl = L.tract_length_loss(num, p, _gt(seed=3) + 1.0, g)
    assert torch.isfinite(tl) and float(tl) >= 0


def test_zero_variance_gt_is_rejected():
    p = torch.rand(6, 6)
    try:
        L.sc_corr_loss(p, torch.zeros(6, 6))
    except AssertionError:
        return
    raise AssertionError("분산 0 인 GT 를 걸러내지 못함")


def test_endpoint_loss_symmetry():
    torch.manual_seed(0)
    qs, qe = torch.softmax(torch.randn(8, 6) * 3, 1), torch.softmax(torch.randn(8, 6) * 3, 1)
    i, j = torch.randint(0, 6, (8,)), torch.randint(0, 6, (8,))
    a = L.endpoint_loss(qs, qe, i, j)
    b = L.endpoint_loss(qe, qs, i, j)          # 방향을 뒤집어도 같아야 한다
    assert torch.isfinite(a) and abs(float(a) - float(b)) < 1e-5


def test_roi_visit_loss():
    u = torch.rand(8, 6).clamp(0.01, 0.99)
    y = (torch.rand(8, 6) > 0.5).float()
    v = L.roi_visit_loss(u, y)
    assert torch.isfinite(v) and float(v) > 0
    assert float(L.roi_visit_loss(y.clamp(0.001, 0.999), y)) < float(v)


def test_geometry_losses():
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 128)[None, :, None]
    even = t.repeat(3, 1, 3) * 50
    assert float(L.adjacency_loss(even)) < 1e-8            # 완전 등간격 -> 0
    jag = even.clone(); jag[:, ::2] += 5.0
    assert float(L.adjacency_loss(jag)) > float(L.adjacency_loss(even))
    assert float(L.anchor_loss(even, even)) == 0.0
    assert float(L.stream_recon_loss(even, even.flip(1))) < 2e-3   # 방향 무관 (sqrt 의 eps=1e-6 -> 1e-3 mm)


def test_sc_metrics():
    g = _gt()
    m = L.sc_metrics(g.clone(), g)
    assert abs(m["r"] - 1.0) < 1e-6 and abs(m["ccc"] - 1.0) < 1e-6
    assert abs(m["edge_f1"] - 1.0) < 1e-6 and m["mae"] < 1e-6


def test_full_objective_backward(synth, streamlines):
    """L = L_ATM + λ_endpoint L_endpoint + λ_corr L_corr + λ_mag L_mag + λ_len L_len"""
    mm, pairs = streamlines
    mm = mm.clone().requires_grad_(True)
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu", d_bg=None)
    sc, num = SCBuilder(ea, "endpoint")(mm)
    qs, qe = ea.endpoint_probs(mm)
    i = torch.tensor([p[0] for p in pairs]); j = torch.tensor([p[1] for p in pairs])
    g = sc.detach() * 1.3 + 0.5
    gl = (num / (sc + 1e-8)).detach() * 1.1 + 1.0
    total = (1.0 * L.adjacency_loss(mm)
             + 0.1 * L.endpoint_loss(qs, qe, i, j)
             + 0.05 * L.sc_corr_loss(sc, g)
             + 0.05 * L.sc_magnitude_loss(sc, g)
             + 0.02 * L.tract_length_loss(num, sc, gl, g))
    assert torch.isfinite(total)
    total.backward()
    assert mm.grad is not None and float(mm.grad.abs().max()) > 0
