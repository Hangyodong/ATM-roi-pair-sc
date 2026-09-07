"""Trainer 통합: route + presence 가 실제 학습 loop 에서 동작하는지 (ROUTE 전략 §54 체크리스트)."""
import numpy as np
import torch

from atm_sc.models.endpoint_assigner import EndpointAssigner
from atm_sc.models.roi_atm import ROIPairATM
from atm_sc.training.synthetic import make_synthetic
from atm_sc.training.trainer import LossWeights, TrainConfig, Trainer


def _setup(active, seed=0, **cfg_kw):
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8, seed=seed)
    subj.set_visitation(atlas, affine)
    m = ROIPairATM(n_roi=6, coord_min=lo, coord_max=hi, device="cpu")
    ea = EndpointAssigner(dist, affine, tau=0.5, device="cpu", d_bg=2.0)
    cfg = TrainConfig(sc_mode="pass", n_gen_per_pair=4, n_gt_per_pair=4, gt_pairs_per_step=3,
                      neg_pairs_per_step=4, chunk=64, route_tau=1.0, active=set(active),
                      lr_dec=1e-3, lr_heads=1e-3, seed=seed, **cfg_kw)
    return subj, Trainer(m, ea, cfg, LossWeights())


def test_route_and_presence_appear_in_step_output():
    subj, tr = _setup({"recon", "endpoint", "route", "corr", "presence"})
    a = torch.randn(1, 512) * 0.004
    o = tr.step(subj, a)
    for k in ("L_recon", "L_endpoint", "L_route", "L_route_gen", "L_corr", "L_presence",
              "route_f1", "route_recall", "gen_presence_recall"):
        assert k in o, (k, sorted(o))
    assert all(np.isfinite(o[k]) for k in o if isinstance(o[k], float))
    assert o["route_gt_visits"] >= 2                      # 합성 streamline 은 여러 ROI 를 지난다


def test_route_loss_decreases_with_training():
    subj, tr = _setup({"recon", "route"}, seed=1)
    a = torch.randn(1, 512) * 0.004
    first = [tr.step(subj, a)["L_route"] for _ in range(3)]
    last = [tr.step(subj, a)["L_route"] for _ in range(12)][-3:]
    assert np.mean(last) < np.mean(first), (first, last)


def test_route_gradient_reaches_decoder_and_heads():
    """route 만 켰을 때도 decoder/embedding 에 gradient 가 간다 (§54: gradient 가 DS 까지 도달)."""
    subj, tr = _setup({"recon", "route"}, seed=2)
    tr.w.recon = tr.w.kl = tr.w.geom = 0.0                # route 만 남긴다
    a = torch.randn(1, 512) * 0.004
    o = tr.step(subj, a)
    assert o["L_route"] > 0
    assert o["gnorm_decoder"] > 0 and o["gnorm_heads"] > 0


def test_route_off_by_default_weight_zero():
    subj, tr = _setup({"recon", "route"}, seed=3)
    tr.w.route = tr.w.route_gen = 0.0
    o = tr.step(subj, torch.randn(1, 512) * 0.004)
    assert "L_route" not in o and "L_route_gen" not in o


def test_lambda_syn_downweights_synthetic_recon():
    from atm_sc.losses import stream_recon_loss
    gt = torch.zeros(4, 8, 3); pred = torch.ones(4, 8, 3)
    w = torch.tensor([1.0, 1.0, 0.5, 0.5])
    plain = float(stream_recon_loss(pred, gt))
    weighted = float(stream_recon_loss(pred, gt, weights=w))
    assert abs(plain - weighted) < 1e-5                    # 오차가 같으면 가중해도 값은 같다
    pred2 = pred.clone(); pred2[2:] *= 10                  # synthetic 쪽만 크게 틀리게
    assert float(stream_recon_loss(pred2, gt, weights=w)) < float(stream_recon_loss(pred2, gt))
