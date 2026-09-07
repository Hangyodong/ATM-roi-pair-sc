import torch

from atm_sc.models.endpoint_assigner import EndpointAssigner
from atm_sc.models.sc_builder import (BundleAccumulator, ChunkedSC, SCBuilder,
                               endpoint_sc, pass_sc, streamline_lengths)


def _builder(synth, mode):
    ea = EndpointAssigner(synth["dist"], synth["affine"], tau=0.5, device="cpu",
                          d_bg=None if mode == "endpoint" else 2.0)
    return SCBuilder(ea, mode=mode)


def test_shape_symmetry_finite(synth, streamlines):
    mm, _ = streamlines
    R = synth["n_roi"]
    for mode in ("endpoint", "pass"):
        sc, num = _builder(synth, mode)(mm)
        assert sc.shape == (R, R) and num.shape == (R, R)
        assert torch.isfinite(sc).all() and torch.isfinite(num).all()
        assert float((sc - sc.T).abs().max()) < 1e-5, mode
        assert float(sc.diagonal().abs().max()) == 0.0
        assert float(sc.sum()) > 0


def test_endpoint_sc_matches_gt_pairs(synth, streamlines):
    mm, pairs = streamlines
    sc, _ = _builder(synth, "endpoint")(mm)
    for a, b in set(pairs):
        assert sc[a, b] > 0.5, (a, b, float(sc[a, b]))


def test_lengths(streamlines):
    mm, _ = streamlines
    L = streamline_lengths(mm)
    assert L.shape == (mm.shape[0],) and (L > 0).all()
    direct = torch.linalg.norm(mm[:, 0] - mm[:, -1], dim=-1)
    assert (L >= direct - 1e-3).all()          # 경로 길이 >= 직선 거리


def test_predicted_length_reasonable(synth, streamlines):
    mm, _ = streamlines
    sc, num = _builder(synth, "endpoint")(mm)
    lp = num / (sc + 1e-8)
    m = sc > 1e-3
    assert torch.isfinite(lp[m]).all()
    assert float(lp[m].min()) > 0 and float(lp[m].max()) < float(streamline_lengths(mm).max()) * 1.5


def test_chunked_equals_full(synth, streamlines):
    mm, _ = streamlines
    b = _builder(synth, "endpoint")
    full_sc, full_num = b(mm)
    acc = ChunkedSC(synth["n_roi"], device="cpu")
    for i in range(0, mm.shape[0], 5):
        acc += b(mm[i:i + 5])
    assert torch.allclose(acc.sc, full_sc, atol=1e-5)
    assert torch.allclose(acc.num, full_num, atol=1e-3)


def test_two_pass_bundle_gradient_is_exact():
    """30 bundle 을 한 그래프에 못 올릴 때 쓰는 2-pass 누적이 정확한지."""
    torch.manual_seed(0)
    R, N, B = 6, 20, 4
    Ws = [torch.randn(N, R, requires_grad=True) for _ in range(B)]
    Ls = [torch.rand(N) * 50 + 10 for _ in range(B)]

    def loss_fn(sc, num):
        return (torch.log1p(sc) - 1.0).pow(2).mean() + 1e-3 * num.sum()

    tot_sc = tot_num = 0
    for W, L in zip(Ws, Ls):
        s, n = pass_sc(torch.sigmoid(W), L); tot_sc = tot_sc + s; tot_num = tot_num + n
    loss_fn(tot_sc, tot_num).backward()
    ref = [W.grad.clone() for W in Ws]
    for W in Ws:
        W.grad = None

    acc = BundleAccumulator(R, device="cpu")
    with torch.no_grad():
        for W, L in zip(Ws, Ls):
            acc.add(*pass_sc(torch.sigmoid(W), L))
    _, gs, gn = acc.loss_grads(loss_fn)
    for W, L in zip(Ws, Ls):
        s, n = pass_sc(torch.sigmoid(W), L)
        BundleAccumulator.backward_bundle(s, n, gs, gn)
    for W, r in zip(Ws, ref):
        assert torch.allclose(W.grad, r, atol=1e-6), float((W.grad - r).abs().max())


def test_endpoint_sc_is_symmetric_by_construction():
    torch.manual_seed(1)
    qs, qe = torch.softmax(torch.randn(9, 5), 1), torch.softmax(torch.randn(9, 5), 1)
    sc, _ = endpoint_sc(qs, qe, torch.rand(9) * 10)
    assert float((sc - sc.T).abs().max()) < 1e-6
