"""external/tractolearn 소스를 현재 환경(torch 2.6 / numpy 2 / dipy 1.12, scilpy 없음)에서 쓰기 위한 compat 확인."""
import numpy as np
import torch

from atm_sc.compat import tractolearn_env  # noqa: F401  (sys.path + dipy/scilpy shim)


def test_finta_autoencoder_forward_backward():
    from tractolearn.models.track_ae_cnn1d_incr_feat_strided_conv_fc_upsamp_reflect_pad_pytorch import (
        IncrFeatStridedConvFCUpsampReflectPadAE as AE)
    ae = AE(latent_space_dims=32)
    x = torch.randn(4, 3, 256)                      # FINTA 입력: [N, 3, 256 점]
    out = ae(x)
    assert out.shape == x.shape and ae.encode(x).shape == (4, 32)
    ((out - x) ** 2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in ae.parameters() if p.grad is not None)


def test_dipy_downsample_shim_matches_set_number_of_points():
    from dipy.tracking.metrics import downsample
    from dipy.tracking.streamline import set_number_of_points
    s = np.cumsum(np.random.default_rng(0).random((50, 3)), 0).astype(np.float32)
    assert np.allclose(downsample(s, 256), set_number_of_points(s, nb_points=256))
    from tractolearn.transformation import streamline_transformation   # downsample 을 import 하는 모듈
    assert hasattr(streamline_transformation, "resample_streamlines")


def test_scilpy_streamlines_in_mask_shim():
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    import nibabel as nib
    from scilpy.segment.streamlines import streamlines_in_mask
    ref = nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4))
    mask = np.zeros((10, 10, 10), bool); mask[5:8, 5:8, 5:8] = True
    inside = np.array([[5.5, 5.5, 5.5], [6.5, 6.5, 6.5]]); outside = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    partial = np.array([[0.5, 0.5, 0.5], [6.5, 6.5, 6.5]])
    sft = StatefulTractogram([inside, outside, partial], ref, Space.VOX)
    assert streamlines_in_mask(sft, mask).tolist() == [True, False, True]
    assert streamlines_in_mask(sft, mask, all_in=True).tolist() == [True, False, False]
