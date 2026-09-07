"""Route / ROI-visitation loss (docs/ATM_ROUTE_LOSS_PASS_SC_GESTA_DETAILED_STRATEGY.md §12–21, §54)."""
import numpy as np
import torch

from atm_sc import losses as L
from atm_sc.data.roi_groups import block_masks
from atm_sc.models.endpoint_assigner import EndpointAssigner
from atm_sc.training.synthetic import make_synthetic


def _log(u):
    return torch.log(torch.as_tensor(u, dtype=torch.float32).clamp(min=1e-30))


def test_route_bce_perfect_and_wrong():
    y = torch.zeros(2, 6); y[0, [0, 1, 2]] = 1; y[1, [3, 4]] = 1
    perfect = _log(y * (1 - 1e-6) + (1 - y) * 1e-6)
    assert float(L.route_bce(perfect, y)) < 1e-3
    wrong = _log((1 - y) * (1 - 1e-6) + y * 1e-6)          # 정확히 반대로 통과
    assert float(L.route_bce(wrong, y)) > 10
    assert float(L.route_dice(perfect, y)) < 1e-3 and float(L.route_dice(wrong, y)) > 0.9


def test_route_gradient_survives_far_roi():
    """확률 공간이면 exp(-d/tau) 가 float32 밑으로 내려가 gradient 가 0 이 된다 (endpoint loss 버그와 동일)."""
    log_u = torch.full((1, 6), -80.0, requires_grad=True)   # u ~ 1e-35
    y = torch.zeros(1, 6); y[0, 0] = 1
    L.route_bce(log_u, y).backward()
    assert torch.isfinite(log_u.grad).all() and float(log_u.grad[0, 0].abs()) > 1e-3


def test_pos_weight_and_modes():
    y = torch.zeros(4, 82); y[:, :5] = 1
    log_u = _log(torch.full((4, 82), 0.02))
    a = float(L.route_loss(log_u, y, "bce", pos_weight=1.0))
    b = float(L.route_loss(log_u, y, "bce", pos_weight=5.0))
    assert b > a                                            # 양성(놓친 통과 ROI) 에 더 큰 벌점
    assert float(L.route_loss(log_u, y, "both")) > float(L.route_loss(log_u, y, "dice"))


def test_endpoint_correct_but_route_wrong_is_penalised():
    """§18 도로 비유: 끝점은 맞고 중간 경로만 다른 streamline 은 endpoint loss 로는 잡히지 않는다."""
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8, seed=0)
    subj.set_visitation(atlas, affine)
    ea = EndpointAssigner(dist, affine, tau=0.5, d_bg=2.0, device="cpu")
    S = torch.from_numpy(subj._bundles[0]).float()
    y = subj.visitation(0)
    a, b = subj.pair_ids[0]
    tau = 1.0                                               # toy atlas 는 ROI 간격이 6 mm (실제 atlas 는 5.0 사용)
    ends = lambda X: L.endpoint_loss(*ea.endpoint_log_probs(X, tau=tau), torch.tensor([a] * 8),
                                     torch.tensor([b] * 8), log_input=True)
    route = lambda X: L.route_loss(ea.visit_log_probs(X, tau=tau), y)
    detour = S.clone(); detour[:, 20:108, 1] += 14.0        # 중간만 ROI 밖으로 우회 (끝점은 그대로)
    assert abs(float(ends(detour)) - float(ends(S))) < 1e-4          # endpoint loss 는 변화 없음
    assert float(route(detour)) > float(route(S)) * 2                # route loss 는 4배 이상 나빠진다
    inside = S.clone(); inside[:, 20:108, 1] += 7.0         # 여전히 같은 ROI 안 -> 통과 집합 동일
    assert abs(float(route(inside)) - float(route(S))) < 1e-4


def test_route_gradient_reaches_streamline_coordinates():
    subj, atlas, affine, dist, lo, hi = make_synthetic(seed=1)
    subj.set_visitation(atlas, affine)
    ea = EndpointAssigner(dist, affine, tau=0.5, d_bg=2.0, device="cpu")
    S = torch.from_numpy(subj._bundles[0]).float().requires_grad_(True)
    L.route_loss(ea.visit_log_probs(S, tau=5.0), subj.visitation(0)).backward()
    assert torch.isfinite(S.grad).all() and float(S.grad.abs().max()) > 0


def test_presence_loss_separates_existence_from_magnitude():
    R = 82
    g = torch.zeros(R, R)
    iu = torch.triu_indices(R, R, 1)
    rng = np.random.default_rng(0)
    val = torch.as_tensor(rng.lognormal(3, 2, iu.shape[1]), dtype=torch.float32) * (torch.as_tensor(rng.random(iu.shape[1])) > 0.3)
    g[iu[0], iu[1]] = val; g = g + g.T
    ss = torch.as_tensor(block_masks(R)["sub-sub"])
    perfect = L.pass_presence_loss(g, g, ss)
    missing = g.clone(); missing[ss] = 0.0                   # SUB-SUB edge 를 전부 놓친 예측
    assert float(L.pass_presence_loss(missing, g, ss)) > float(perfect) * 5
    weak = g.clone(); weak[ss] = weak[ss] * 0.01             # 존재하지만 크기만 작다
    assert float(L.pass_presence_loss(weak, g, ss)) < float(L.pass_presence_loss(missing, g, ss))
    m = L.presence_metrics(missing, g, ss)
    assert m["presence_recall"] == 0.0 and L.presence_metrics(g, g, ss)["presence_recall"] == 1.0


def test_presence_gradient():
    R = 12
    p = torch.rand(R, R); p = (p + p.T) / 2 * 5; p.requires_grad_(True)
    g = torch.zeros(R, R); g[0, 1] = g[1, 0] = 7.0
    L.pass_presence_loss(p, g).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
