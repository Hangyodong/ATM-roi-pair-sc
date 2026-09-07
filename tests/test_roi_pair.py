"""ROI-pair 조건화 모듈 (pipeline §9, §10, §16) 단위 테스트. pretrained 체크포인트 불필요."""
import numpy as np
import torch

from atm_sc.models.edge_head import EdgeHead
from atm_sc.models.roi_pair_embedding import ROIPairEmbedding, canonical_pairs
from atm_sc.models.sc_builder import endpoint_sc, pass_sc
from atm_sc.models.streamline_weight_head import StreamlineWeightHead
from atm_sc.losses import edge_loss, edge_metrics
from atm_sc.training.synthetic import make_synthetic


def test_canonical_pairs():
    p = torch.tensor([[5, 2], [1, 7], [3, 3]])
    c = canonical_pairs(p)
    assert torch.equal(c, torch.tensor([[2, 5], [1, 7], [3, 3]]))


def test_embedding_symmetric_and_anatomy_only_at_init():
    torch.manual_seed(0)
    e = ROIPairEmbedding(n_roi=10, emb_dim=8, cond_dim=16)
    a = torch.randn(1, 16) * 0.01
    p = torch.tensor([[1, 4], [4, 1], [2, 9]])
    c = e(a, p)
    assert c.shape == (3, 16)
    assert torch.allclose(c[0], c[1])                       # (a,b) == (b,a)
    # proj 0-init -> 시작 cond 는 anatomy 항뿐이다 (재학습 설계 §2 ②: LayerNorm 을 거친다)
    assert torch.allclose(c, e.anatomy_term(a).expand(3, -1), atol=1e-6)
    # LayerNorm + 1/sqrt(C) 가중치 -> 입력 크기와 무관하게 L2 ~ 1 (pair 항과 대등한 자릿수)
    for scale in (1.0, 100.0):
        assert abs(float(e.anatomy_term(a * scale).norm()) - 1.0) < 1e-3, scale
    e.proj[-1].weight.data.normal_()
    c2 = e(a, p)
    assert not torch.allclose(c2[0], c2[2])                 # 다른 pair -> 다른 cond
    assert torch.allclose(c2[0], c2[1])


def test_embedding_rejects_out_of_range():
    e = ROIPairEmbedding(n_roi=5, emb_dim=4, cond_dim=8)
    try:
        e(torch.zeros(1, 8), torch.tensor([[0, 5]]))
    except AssertionError:
        return
    raise AssertionError("범위 밖 ROI 를 거르지 못함")


def test_weight_head_init_one_and_positive():
    h = StreamlineWeightHead(cond_dim=16, latent_dim=4)
    w = h(torch.randn(20, 16), torch.randn(20, 4))
    assert w.shape == (20,)
    assert torch.allclose(w, torch.ones(20), atol=1e-6)
    h.net[-1].weight.data.normal_()
    assert (h(torch.randn(20, 16), torch.randn(20, 4)) > 0).all()


def test_edge_head_init_half_and_loss():
    h = EdgeHead(cond_dim=16, emb_dim=4)
    lg = h(torch.randn(1, 16), torch.randn(6, 4))
    assert lg.shape == (6,) and torch.allclose(torch.sigmoid(lg), torch.full((6,), 0.5))
    y = torch.tensor([1., 1., 0., 0., 1., 0.])
    l = edge_loss(lg, y)
    assert abs(float(l) - float(np.log(2))) < 1e-5           # p=0.5 -> ln2
    m = edge_metrics(lg, y)
    assert abs(m["pos_ratio"] - 0.5) < 1e-6
    lw = edge_loss(lg, y, pos_weight=3.0)
    assert torch.isfinite(lw)


def test_weighted_sc_scales_linearly_and_stays_symmetric():
    torch.manual_seed(0)
    qs = torch.softmax(torch.randn(30, 7) * 3, 1); qe = torch.softmax(torch.randn(30, 7) * 3, 1)
    L = torch.rand(30) * 80 + 20
    sc1, n1 = endpoint_sc(qs, qe, L)
    sc2, n2 = endpoint_sc(qs, qe, L, weights=torch.full((30,), 2.5))
    assert torch.allclose(sc2, 2.5 * sc1, atol=1e-5) and torch.allclose(n2, 2.5 * n1, atol=1e-3)
    assert float((sc2 - sc2.T).abs().max()) < 1e-5
    w = torch.rand(30)
    sc3, _ = endpoint_sc(qs, qe, L, weights=w)
    ref = torch.zeros(7, 7)
    for k in range(30):
        ref += w[k] * 0.5 * (torch.outer(qs[k], qe[k]) + torch.outer(qe[k], qs[k]))
    ref.fill_diagonal_(0)
    assert torch.allclose(sc3, ref, atol=1e-5)
    u = torch.rand(30, 7)
    p3, _ = pass_sc(u, L, weights=w)
    assert float((p3 - p3.T).abs().max()) < 1e-5


def test_weight_gradient_reaches_head():
    h = StreamlineWeightHead(cond_dim=16, latent_dim=4)
    qs = torch.softmax(torch.randn(10, 5), 1); qe = torch.softmax(torch.randn(10, 5), 1)
    w = h(torch.randn(10, 16), torch.randn(10, 4))
    sc, _ = endpoint_sc(qs, qe, torch.rand(10) * 10, weights=w)
    (sc.sum()).backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in h.parameters())


def test_synthetic_subject_consistency():
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8)
    assert len(subj.pair_ids) == 3 and subj.n_roi == 6
    assert (subj.pair_ids[:, 0] < subj.pair_ids[:, 1]).all()
    for k, (a, b) in enumerate(subj.pair_ids):
        S, L = subj.get_pair(k)
        assert S.shape == (8, 128, 3) and L.shape == (8,)
        assert subj.sc_end[a, b] == 8 and subj.sc_end[b, a] == 8
    rng = np.random.default_rng(0)
    neg = subj.negative_pairs(5, rng)
    assert all(subj.sc_end[a, b] == 0 for a, b in neg)
    assert np.array_equal(subj.sc_end, subj.sc_end.T)
    assert (hi > lo).all()


def test_robust_t1_normalization_is_scale_invariant():
    """PPMI T1 은 subject 마다 강도 스케일이 230배까지 다르다. robust 정규화는 스케일에 불변해야 한다."""
    from atm_sc.models.atm_adapter import BundleNorm
    rng = np.random.default_rng(0)
    vol = np.zeros((20, 20, 20), np.float32)
    vol[4:16, 4:16, 4:16] = rng.gamma(4.0, 200.0, (12, 12, 12)).astype(np.float32)   # 뇌
    norm = BundleNorm(0.0, 8330.0, [0, 0, 0], [1, 1, 1])
    a = norm.normalize_t1(vol)
    b = norm.normalize_t1(vol * 137.0)                     # 다른 스캐너 스케일
    assert np.allclose(a, b, atol=1e-5)
    assert abs(np.percentile(a[vol > 0], 99.5) - 0.6) < 1e-3   # p99.5 -> target 0.6
    c = norm.normalize_t1(vol, robust=False); d = norm.normalize_t1(vol * 137.0, robust=False)
    assert not np.allclose(c, d)                           # upstream 식 그대로는 스케일 의존


def test_conditional_prior_zero_init_and_kl():
    import torch
    from atm_sc.losses import kl_loss
    e = ROIPairEmbedding(n_roi=10, emb_dim=8, cond_dim=16, latent_dim=6)
    p = torch.tensor([[1, 4], [4, 1], [2, 9]])
    mu = e.prior_mean(p)
    assert mu.shape == (3, 6) and torch.allclose(mu, torch.zeros(3, 6))    # 0-init -> N(0,I)
    assert torch.allclose(mu[0], mu[1])                                     # 순서 불변
    m, lv = torch.randn(3, 6), torch.zeros(3, 6)
    assert torch.allclose(kl_loss(m, lv), kl_loss(m, lv, torch.zeros(3, 6)))
    assert float(kl_loss(m, lv, m)) < float(kl_loss(m, lv))                 # prior 평균이 posterior 와 같으면 KL 최소


def test_edge_head_template_starts_at_group_frequency():
    """③ edge head 인수분해: 시작 확률 = train 그룹 빈도 (재학습 설계 §2 ③)."""
    rng = np.random.default_rng(0)
    n_roi = 8
    pr = np.triu(rng.random((n_roi, n_roi)), 1)
    pr = pr + pr.T
    h = EdgeHead(cond_dim=16, emb_dim=4, template_prob=pr)
    pairs = torch.tensor([[0, 3], [1, 5], [2, 7]])
    p = torch.sigmoid(h(torch.randn(1, 16), torch.randn(3, 4), pairs))
    ref = torch.tensor([pr[int(i), int(j)] for i, j in pairs], dtype=p.dtype)
    assert torch.allclose(p, ref, atol=1e-6), (p, ref)
    assert not any("template" in n for n, _ in h.named_parameters())
    # 템플릿 없이 만들면 기존대로 p = 0.5
    plain = EdgeHead(cond_dim=16, emb_dim=4)
    assert torch.allclose(torch.sigmoid(plain(torch.randn(1, 16), torch.randn(3, 4))),
                          torch.full((3,), 0.5))
