"""Edge Count Head + count loss (strategy §19-22).

핵심 성질: sc_magnitude_loss 와 달리 절대 스케일에 민감해야 한다 (10배 예측 = 큰 벌).
"""
import numpy as np
import pytest
import torch

from atm_sc.data.roi_groups import block_masks
from atm_sc.losses.edge_count import edge_count_loss, edge_count_matrix_loss, edge_count_metrics
from atm_sc.models.edge_count_head import EdgeCountHead
from atm_sc.models.roi_pair_embedding import ROIPairEmbedding

C, E, K, R = 512, 64, 7, 82


def _head(init_log_count=5.0):
    torch.manual_seed(0)
    return EdgeCountHead(anatomy_dim=C, emb_dim=E, hidden=32, init_log_count=init_log_count)


def _inputs(k=K, n_roi=16, seed=0):
    torch.manual_seed(seed)
    emb = ROIPairEmbedding(n_roi=n_roi, emb_dim=E, cond_dim=C)
    g = torch.Generator().manual_seed(seed)
    pairs = torch.stack([torch.randint(0, n_roi // 2, (k,), generator=g),
                         torch.randint(n_roi // 2, n_roi, (k,), generator=g)], dim=1)
    return emb.pair_vec(pairs).detach(), pairs


def test_shape_and_constant_init():
    head = _head()
    pv, _ = _inputs()
    for anatomy in (torch.randn(1, C), torch.randn(K, C)):
        lp = head(anatomy, pv)
        assert lp.shape == (K,) and torch.isfinite(lp).all()
        c = head.count(anatomy, pv)
        # 마지막 층 0-init + bias -> pair/anatomy 와 무관하게 상수 exp(init_log_count)
        assert torch.allclose(c, torch.full((K,), float(np.exp(5.0))), atol=1e-4)


def test_gradient_flows_to_params_and_anatomy():
    head = _head()
    pv, _ = _inputs()
    # 0-init 이라 step 0 에서는 입력 쪽 gradient 가 정확히 0 이다. 학습이 시작된 뒤 상태를
    # 흉내내려고 마지막 층만 작게 흔든다.
    torch.nn.init.normal_(head.net[-1].weight, std=0.01)
    anatomy = torch.randn(1, C, requires_grad=True)
    loss = edge_count_loss(head(anatomy, pv), torch.full((K,), 100.0))
    loss.backward()
    assert anatomy.grad is not None and float(anatomy.grad.abs().sum()) > 0
    for n, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert float(p.grad.abs().sum()) > 0, n


def test_matrix_symmetric_zero_diag_and_placement():
    head = _head()
    torch.nn.init.normal_(head.net[-1].weight, std=0.05)   # 상수 예측이면 배치 검증이 무의미
    n_roi = 16
    pv, pairs = _inputs(n_roi=n_roi)
    anatomy = torch.randn(1, C)
    m = head.matrix(anatomy, pv, pairs, n_roi)
    c = head.count(anatomy, pv)
    assert m.shape == (n_roi, n_roi)
    assert torch.allclose(m, m.T)
    assert float(m.diagonal().abs().max()) == 0.0
    for k in range(K):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        assert torch.allclose(m[i, j], c[k], atol=1e-5)
        assert torch.allclose(m[j, i], c[k], atol=1e-5)
    assert float(m.sum()) == pytest.approx(float(2 * c.sum()), rel=1e-5)


def test_loss_zero_at_truth_and_scale_sensitive():
    gt = torch.tensor([10.0, 100.0, 1000.0, 5000.0], dtype=torch.float64)
    exact = edge_count_loss(torch.log(gt), gt)
    assert float(exact) < 1e-6
    small = edge_count_loss(torch.log(gt * 1.2), gt)
    big = edge_count_loss(torch.log(gt * 10.0), gt)
    assert 0 < float(small) < float(big)
    # 스케일 민감성: 10배는 log 로 ~2.3 만큼 벌을 받는다. sum-normalise 하는
    # sc_magnitude_loss 는 여기서 0 이 된다 (그게 CCC=0.02 의 원인).
    assert float(big) > 2.0


def test_zero_target_relaxes_gt_zero_edges():
    gt = torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64)
    lp = torch.log(torch.tensor([0.5, 0.5, 100.0], dtype=torch.float64))
    assert float(edge_count_loss(lp, gt, zero_target=0.5)) < 1e-6
    assert float(edge_count_loss(lp, gt)) > 0.2          # 목표 0 이면 아직 벌이 남는다


def test_block_masks_prevent_sub_sub_from_drowning():
    rng = np.random.default_rng(0)
    gt = np.triu(rng.integers(1, 2000, size=(R, R)).astype(np.float64), 1)
    gt = gt + gt.T
    masks = block_masks(R)
    pred = gt.copy()
    pred[masks["sub-sub"]] *= 100.0                      # sub-sub 만 100배 틀리게
    np.fill_diagonal(pred, 1.0)                          # log(0) 회피 (대각은 어차피 안 쓴다)
    log_pred = torch.tensor(np.log(pred))
    gt_t = torch.tensor(gt)
    plain = float(edge_count_matrix_loss(log_pred, gt_t))
    balanced = float(edge_count_matrix_loss(log_pred, gt_t, masks))
    # sub-sub 는 3321 edge 중 120 개뿐이라 단순 평균에서는 오차가 희석된다.
    assert balanced > 5 * plain
    assert balanced == pytest.approx(np.log(100.0) / 3, abs=0.2)   # (0 + 0 + ~log100) / 3


def test_metrics_perfect_and_degenerate():
    gt = torch.tensor([0.0, 0.0, 5.0, 50.0, 500.0, 5000.0])
    m = edge_count_metrics(gt.clone(), gt)
    assert m["r"] == pytest.approx(1.0, abs=1e-6)
    assert m["ccc"] == pytest.approx(1.0, abs=1e-6)
    assert m["log_mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["zero_specificity"] == 1.0 and m["recall"] == 1.0
    assert m["n_edges"] == 6

    z = edge_count_metrics(torch.zeros_like(gt), gt)
    assert z["recall"] == 0.0                 # 하나도 못 만든다
    assert z["zero_specificity"] == 1.0       # 0 은 맞췄지만 무의미


def test_template_factorization_starts_exactly_at_group_template():
    """③ 템플릿 인수분해: 시작 시점 예측이 그룹 템플릿 그 자체여야 한다 (재학습 설계 §2 ③)."""
    rng = np.random.default_rng(0)
    n_roi = 16
    t = np.triu(rng.integers(0, 500, size=(n_roi, n_roi)).astype(np.float64), 1)
    t = t + t.T
    head = EdgeCountHead(anatomy_dim=C, emb_dim=E, hidden=32, template=t)
    pv, pairs = _inputs(n_roi=n_roi)                       # _inputs 는 (낮은 ROI, 높은 ROI) 로 만든다
    ref = torch.tensor([t[int(i), int(j)] for i, j in pairs], dtype=torch.float32)
    for anatomy in (torch.randn(1, C), torch.randn(K, C)):
        c = head.count(anatomy, pv, pairs)                 # 0-init -> anatomy 와 무관
        nz = ref > 0
        assert torch.allclose(c[nz], ref[nz], rtol=1e-4), (c[nz], ref[nz])
        if bool((~nz).any()):
            assert float(c[~nz].max()) < 1.0               # 템플릿 0 -> "생성 0 개"
    # 템플릿은 고정 buffer 다 (학습하지 않는다)
    assert "template_log" in head.state_dict()
    assert not any("template" in n for n, _ in head.named_parameters())
    # pair 를 안 주면 조용히 틀린 값을 내지 않고 멈춘다
    with pytest.raises(AssertionError):
        head.count(torch.randn(1, C), pv)
    # 템플릿을 안 주면 기존 동작 그대로
    plain = EdgeCountHead(anatomy_dim=C, emb_dim=E, hidden=32)
    assert torch.allclose(plain.count(torch.randn(1, C), pv),
                          torch.full((K,), float(np.exp(5.0))), atol=1e-4)
