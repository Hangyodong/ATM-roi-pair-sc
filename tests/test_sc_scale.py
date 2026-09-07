"""SC 절대 스케일: 총합 로그 항 + RMSE 항 (실측 근거: 배율 하나로 CCC 0.024 -> 0.817)."""
import numpy as np
import torch

from atm_sc import losses as L
from atm_sc.data.roi_groups import block_masks
from atm_sc.evaluation.balance_metrics import sc_metrics_extended


def _gt(R=82, seed=0):
    rng = np.random.default_rng(seed)
    g = np.triu(rng.lognormal(4, 2.5, (R, R)), 1)
    return torch.as_tensor(g + g.T, dtype=torch.float64)


def test_sum_normalised_magnitude_is_blind_to_scale():
    """지금까지 절대 크기를 못 배운 이유: 총합 정규화 항은 배율에 완전히 무감각하다."""
    g = _gt()
    for k in (0.02, 1.0, 50.0):
        assert abs(float(L.sc_magnitude_loss(g * k, g, normalize="sum"))) < 1e-9
    assert float(L.sc_magnitude_loss(g * 0.02, g, normalize="none")) > 2.5   # 정규화를 끄면 보인다


def test_scale_loss_measures_global_factor():
    g = _gt()
    assert float(L.sc_scale_loss(g, g)) < 1e-9
    for k in (1 / 48.0, 48.0):
        v = float(L.sc_scale_loss(g * k, g))
        assert abs(v - abs(np.log(k))) < 1e-6                              # 정확히 log 배율
    p = g.clone().requires_grad_(True)
    L.sc_scale_loss(p * 0.02, g).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0


def test_rmse_loss_normalisations():
    g = _gt()
    assert float(L.sc_rmse_loss(g, g)) < 1e-9
    raw = float(L.sc_rmse_loss(g * 0.5, g, normalize="none"))
    std = float(L.sc_rmse_loss(g * 0.5, g, normalize="gt_std"))
    lg = float(L.sc_rmse_loss(g * 0.5, g, normalize="log"))
    assert raw > 100 and 0.1 < std < 2 and lg < 1                          # 무차원화가 자릿수를 맞춘다
    m = {k: torch.as_tensor(v) for k, v in block_masks(82).items()}
    bad = g.clone(); ss = torch.as_tensor(block_masks(82)["sub-sub"]); bad[ss] = bad[ss] * 20
    assert float(L.sc_rmse_loss(bad, g, m)) > 3 * float(L.sc_rmse_loss(bad, g))   # 작은 block 이 묻히지 않는다


def test_scale_correction_recovers_ccc():
    """CCC 는 배율에 민감하고 상관은 둔감하다 — 그래서 절대 항이 필요하다."""
    g = _gt(seed=3)
    p = g / 48.0
    before, after = sc_metrics_extended(p, g), sc_metrics_extended(p * 48.0, g)
    assert before["ccc"] < 0.1 and after["ccc"] > 0.99
    assert abs(before["r"] - after["r"]) < 1e-6


def test_losses_agree_on_the_direction():
    g = _gt(seed=4)
    scales = [1 / 48, 1 / 8, 1, 8, 48]
    for name, fn in (("scale", L.sc_scale_loss),
                     ("mag_none", lambda a, b: L.sc_magnitude_loss(a, b, normalize="none")),
                     ("rmse", lambda a, b: L.sc_rmse_loss(a, b))):
        v = [float(fn(g * k, g)) for k in scales]
        assert v[2] == min(v), (name, v)                                   # 배율 1 에서 최소


def test_count_weight_mode_gives_gt_scale():
    """weight_mode='count': 균등 생성 + 개수 가중 = 전체를 만든 것과 같은 스케일 (100만 가닥 반영)."""
    import numpy as np
    from atm_sc.models.endpoint_assigner import EndpointAssigner
    from atm_sc.models.roi_atm import ROIPairATM
    from atm_sc.training.synthetic import make_synthetic
    from atm_sc.training.trainer import LossWeights, TrainConfig, Trainer
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8, seed=0)
    m = ROIPairATM(n_roi=6, coord_min=lo, coord_max=hi, device="cpu")
    ea = EndpointAssigner(dist, affine, tau=0.5, device="cpu", d_bg=2.0)
    out = {}
    for mode in ("head", "count"):
        cfg = TrainConfig(sc_mode="pass", n_gen_per_pair=4, chunk=64, active={"corr"},
                          weight_mode=mode, seed=0)
        tr = Trainer(m, ea, cfg, LossWeights())
        out[mode] = tr.step(subj, torch.randn(1, 512) * 0.004)["sc_pred_sum"]
    # count 모드는 N_hat(초기 exp(3)=20.1)/n_gen=4 배 -> 약 5배 큰 SC
    assert 4.0 < out["count"] / out["head"] < 6.5, out
