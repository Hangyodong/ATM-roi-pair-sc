"""latent sampler adapter (GESTA 전략 §22–27, §71, smoke test 1 §75)."""
import numpy as np
import pytest

from atm_sc.generative.latent_sampler import sample_latents, silverman_bandwidth

D = 64


def _seeds(n=100, d=D, seed=0, scale=1.0):
    return np.random.default_rng(seed).standard_normal((n, d)) * scale


def test_silverman_bandwidth_scale():
    h = silverman_bandwidth(_seeds(1000, D))
    assert 0.05 < h < 2.0
    assert silverman_bandwidth(_seeds(1000, D, scale=3.0)) > 2.5 * h    # σ 에 비례


@pytest.mark.parametrize("method", ["kde", "gaussian"])
def test_shape_finite_reproducible(method):
    z = _seeds()
    a, ia = sample_latents(z, 250, method=method, seed=7)
    b, ib = sample_latents(z, 250, method=method, seed=7)
    c, _ = sample_latents(z, 250, method=method, seed=8)
    assert a.shape == (250, D) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert ia["acceptance_rate"] > 0 and ia["n_trials"] >= 250 and ia["elapsed_sec"] >= 0
    assert ia["n_seeds"] == 100
    assert {k: ia[k] for k in ia if k != "elapsed_sec"} == {k: ib[k] for k in ib if k != "elapsed_sec"}


def test_kde_samples_stay_near_seeds():
    z = _seeds()
    out, info = sample_latents(z, 500, method="kde", seed=0)
    d = np.linalg.norm(out[:, None, :] - z[None], axis=-1).min(1)
    assert 0 < d.mean() < 2 * info["bandwidth"] * np.sqrt(D)


def test_tiny_seed_sets_work():
    z = _seeds(3)
    for method in ("kde", "gaussian"):
        out, _ = sample_latents(z, 20, method=method, seed=0)
        assert out.shape == (20, D) and np.isfinite(out).all()


def test_bad_arguments_rejected():
    z = _seeds(10)
    for bad in (lambda: sample_latents(z, 0), lambda: sample_latents(z, 5, method="nope"),
                lambda: sample_latents(z[0], 5), lambda: sample_latents(z, 5, bandwidth="scott")):
        with pytest.raises((AssertionError, ValueError)):
            bad()


def test_rejection_low_dim_and_high_dim_guard():
    """64-D KDE 는 proposal acceptance 가 ~1e-11 이라 sampling 이 끝나지 않는다 -> 조용한 fallback 대신 assert."""
    out, info = sample_latents(_seeds(200, 4), 100, method="rejection", seed=0)   # 저차원: 정상 동작
    assert out.shape == (100, 4) and np.isfinite(out).all() and 0 < info["acceptance_rate"] <= 1
    same, _ = sample_latents(_seeds(200, 4), 100, method="rejection", seed=0)
    assert np.array_equal(out, same)
    with pytest.raises(AssertionError, match="acceptance"):
        sample_latents(_seeds(100, D), 100, method="rejection", seed=0)
