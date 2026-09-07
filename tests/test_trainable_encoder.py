"""T1 encoder unfreeze (최종 전략 §2, §19-2/3). CPU, 작은 입력 (인코더 경로는 격자 크기에 무관)."""
import torch
import pytest

from atm_sc.models.atm_adapter import UPSTREAM, ATMBundle, BundleNorm
from atm_sc.training.trainer import LossWeights, TrainConfig

pytestmark = pytest.mark.skipif(not (UPSTREAM / "models" / "AF_L" / "atmvae_AF_L.pth").exists(),
                                reason="pretrained 체크포인트 없음")


@pytest.fixture(scope="module")
def atm():
    return ATMBundle("AF_L", BundleNorm.from_upstream("AF_L"), device="cpu")


def test_levels_set_requires_grad(atm):
    for level, n_stages in [("none", 0), ("stage4", 1), ("stage3", 2), ("stage2", 3), ("full", 4)]:
        atm.set_unet_trainable(level)
        assert len(atm.trainable_unet_stages(level)) == n_stages
        ps = atm.unet_trainable_parameters()
        assert all(p.requires_grad for p in ps)
        frozen = [p for st in atm._STAGES if st not in atm.trainable_unet_stages(level) for p in atm.unet_stage_parameters(st)]
        assert not any(p.requires_grad for p in frozen)
        # 디코더(segmentation) 가지는 어느 level 에서도 동결
        assert not any(p.requires_grad for p in atm.net.unet.final_conv.parameters())


def test_grad_encoder_matches_frozen_path_and_checkpoint(atm):
    torch.manual_seed(0)
    x = torch.rand(1, 1, 24, 28, 24) * 0.5
    ref = atm._unet_encoder_only(atm.net.unet, x)
    atm.set_unet_trainable("full")
    a1 = atm.encode_anatomy_grad(x, "full", use_checkpoint=True)
    a2 = atm.encode_anatomy_grad(x, "full", use_checkpoint=False)
    assert torch.allclose(a1, ref, atol=1e-5) and torch.allclose(a2, ref, atol=1e-5)
    assert a1.requires_grad


def test_gradient_reaches_stage1_only_when_full(atm):
    torch.manual_seed(0)
    x = torch.rand(1, 1, 24, 28, 24) * 0.5
    for level, expect_s1 in [("stage4", False), ("full", True)]:
        atm.set_unet_trainable(level)
        for p in atm.net.unet.parameters():
            p.grad = None
        a = atm.encode_anatomy_grad(x, level)
        a.pow(2).sum().backward()
        g1 = atm.net.unet.conv1_1.weight.grad
        g4 = atm.net.unet.conv4_1.weight.grad
        assert g4 is not None and float(g4.abs().max()) > 0
        assert (g1 is not None and float(g1.abs().max()) > 0) == expect_s1
    atm.set_unet_trainable("none")


def test_param_groups_disjoint_and_complete():
    from atm_sc.models.roi_atm import ROIPairATM
    m = ROIPairATM(n_roi=6, trainable="full", device="cpu")
    g = m.param_groups()
    ids = [id(p) for ps in g.values() for p in ps]
    assert len(ids) == len(set(ids)), "그룹이 겹침"
    assert set(ids) == {id(p) for p in m.parameters() if p.requires_grad}, "requires_grad 파라미터와 불일치"
    c = m.param_counts()
    assert c["t1_encoder"] > 20e6 and c["heads"] > 0 and c["trainable"] + c["frozen"] == c["total"]
    m2 = ROIPairATM(n_roi=6, trainable="vae", device="cpu")
    assert m2.param_counts()["t1_encoder"] == 0            # frozen baseline 보존


def test_config_defaults_match_strategy():
    tc, lw = TrainConfig(), LossWeights()
    assert tc.lr_t1 == 1e-5 and tc.lr_heads == 1e-4 and 1e-5 <= tc.lr_dec <= 5e-5
    assert {"recon", "kl", "geom", "endpoint", "edge", "corr", "mag", "length"} <= set(vars(lw))
