"""ATM latent seed(N x 64) 로부터 새 latent 를 뽑는다. 전략 문서 §22–27, §71 의 TractoLearn RejectionSampler adapter.

method
  kde       : Gaussian KDE 에서 직접 샘플링 (seed 균등 선택 + N(0, h²I)). rejection 이 없어 항상 빠르다. 기본값.
  gaussian  : seed 의 평균/공분산(+ridge) multivariate normal.
  rejection : tractolearn RejectionSampler (target = KDE, proposal = Gaussian | GMM).
              고차원 KDE 는 seed 마다 떨어진 blob 이라 fitted proposal 의 acceptance 가 exp(-c·D) 로 떨어지고
              (64-D 등방 seed 100개, Silverman h: ~1e-11) 그러면 tractolearn 의 while 루프가 끝나지 않는다.
              그래서 sampling 전에 acceptance 를 추정해 min_acceptance 미만이면 AssertionError 를 낸다.
              fallback 여부는 호출자가 결정한다 (§27). 조용한 fallback 없음.

tractolearn 은 pip 설치가 아니라 external/ 소스이고 import 에 ~40 s(umap) 걸리므로 rejection 에서만 지연 import.
numpy in / numpy out. 같은 seed 면 같은 출력.
"""
from __future__ import annotations

import time

import numpy as np

# 우리 proposal 이름 -> tractolearn RejectionSampler 의 proposal_distribution_name
_PROPOSALS = {"multivariate_normal": "multivariate_normal", "gmm": "GMM"}
_RIDGE = 1e-6      # gaussian: N < D 여도 공분산이 PD 가 되게 하는 대각 ridge
_N_PROBE = 2048    # rejection: acceptance 추정에 쓰는 proposal 표본 수


def silverman_bandwidth(z: np.ndarray) -> float:
    """Silverman(1986) rule, D 차원: h = (n(d+2)/4)^(-1/(d+4)) · σ, σ = 차원별 표준편차(ddof=1)의 평균. n<2 면 σ=1."""
    z = np.asarray(z, dtype=np.float64)
    assert z.ndim == 2, z.shape
    n, d = z.shape
    sigma = float(np.std(z, axis=0, ddof=1).mean()) if n >= 2 else 1.0
    return float((n * (d + 2) / 4) ** (-1.0 / (d + 4)) * sigma)


def sample_latents(seed_z: np.ndarray, n: int, method: str = "kde", bandwidth: float | str = "silverman",
                   bw_factor: float = 1.0, seed: int = 0, proposal: str = "multivariate_normal",
                   gmm_components: int = 4, batch_size: int | None = None,
                   min_acceptance: float = 1e-3) -> tuple[np.ndarray, dict]:
    """seed_z [N, D] 에서 n 개의 새 latent 를 뽑아 (z_new [n, D] float32, info) 반환.

    info: method, bandwidth (실제 사용 h = base · bw_factor), acceptance_rate, n_trials, elapsed_sec, n_seeds.
    min_acceptance 는 rejection 전용: 추정 acceptance 가 이보다 낮으면 sampling 하지 않고 AssertionError.
    """
    z = np.asarray(seed_z, dtype=np.float64)
    assert z.ndim == 2 and z.shape[0] >= 1, f"seed_z must be [N>=1, D], got {z.shape}"
    assert np.isfinite(z).all(), "seed_z has NaN/Inf"
    assert isinstance(n, (int, np.integer)) and n > 0, f"n must be a positive int, got {n!r}"
    assert bw_factor > 0, f"bw_factor must be > 0, got {bw_factor}"
    if isinstance(bandwidth, str):
        assert bandwidth == "silverman", f"bandwidth must be a float or 'silverman', got {bandwidth!r}"
        h = silverman_bandwidth(z) * bw_factor
    else:
        h = float(bandwidth) * bw_factor
    assert h > 0, f"bandwidth must be > 0 (all seeds identical?), got {h}"

    n_seeds, d = z.shape
    rng = np.random.default_rng(seed)
    if method == "rejection":
        RejectionSampler = _import_rejection_sampler()   # timer 밖: 최초 1회 ~40 s 의 import 를 sampling 시간에 넣지 않는다
    t0 = time.perf_counter()
    if method == "kde":
        out = z[rng.integers(0, n_seeds, size=n)] + h * rng.standard_normal((n, d))
        n_trials = n
    elif method == "gaussian":
        cov = np.cov(z, rowvar=False) if n_seeds >= 2 else np.zeros((d, d))
        out = rng.multivariate_normal(z.mean(0), cov + _RIDGE * np.eye(d), size=n)
        n_trials = n
    elif method == "rejection":
        out, n_trials = _rejection(RejectionSampler, z, n, h, seed, proposal, gmm_components, batch_size,
                                   min_acceptance)
    else:
        raise ValueError(f"method must be 'kde' | 'gaussian' | 'rejection', got {method!r}")
    elapsed = time.perf_counter() - t0

    out = np.asarray(out, dtype=np.float32)
    assert out.shape == (n, d), f"sampler returned {out.shape}, expected {(n, d)}"
    assert np.isfinite(out).all(), f"{method}: sampled latents contain NaN/Inf"
    info = {"method": method, "bandwidth": float(h), "acceptance_rate": n / n_trials, "n_trials": int(n_trials),
            "elapsed_sec": elapsed, "n_seeds": int(n_seeds)}
    return out, info


def _import_rejection_sampler():
    """compat shim(sys.path + dipy/scilpy/fury 대체) 을 먼저 import 해야 tractolearn 소스가 import 된다."""
    from atm_sc.compat import tractolearn_env  # noqa: F401  (부수효과)
    from tractolearn.generative.generate_points import RejectionSampler
    return RejectionSampler


def _rejection(RejectionSampler, z, n, h, seed, proposal, gmm_components, batch_size, min_acceptance):
    assert proposal in _PROPOSALS, f"proposal must be one of {list(_PROPOSALS)}, got {proposal!r}"
    name = _PROPOSALS[proposal]
    ctx = f"N={len(z)}, D={z.shape[1]}, h={h:.4g}, proposal={proposal}"
    try:
        if name == "GMM":
            np.random.seed(seed)   # tractolearn 의 GaussianMixture(...).fit 은 random_state 를 받지 않는다: 전역 seed 로 재현
        # allow_singular: N < D 면 proposal 공분산이 특이행렬. scipy 가 pseudo-det/pinv 로 seed 가 놓인 부분공간에서 처리한다.
        sampler = RejectionSampler(
            data=z, kde_bw=h, kde_bw_factor=1, scaling_mode="max", allow_singular=True,
            proposal_distribution_name=name,
            proposal_distribution_params={"n_components": gmm_components} if name == "GMM" else None)
    except Exception as e:
        raise AssertionError(f"rejection: RejectionSampler init failed ({ctx}): {e!r}") from e

    # tractolearn 의 acceptance 판정 P(accept | x) = min(1, kde(x) / (M·q(x))) 을 proposal 표본으로 미리 평균낸다.
    M = float(sampler.scaling_factor)
    assert np.isfinite(M) and M > 0, f"rejection: degenerate scaling factor {M} (density under/overflow; {ctx})"
    rng = np.random.default_rng(seed)
    q = sampler.proposal_distribution
    if name == "GMM":
        q.set_params(random_state=int(rng.integers(2**31 - 1)))
        x, _ = q.sample(_N_PROBE)
        log_q = q.score_samples(x)
    else:
        x = q.rvs(size=_N_PROBE, random_state=rng)
        log_q = q.logpdf(x)
    est = float(np.exp(np.minimum(0.0, sampler.kde.score_samples(x) - np.log(M) - log_q)).mean())
    assert est >= min_acceptance, (
        f"rejection: expected acceptance {est:.2e} < min_acceptance {min_acceptance:.0e} "
        f"(~{n / max(est, 1e-300):.1e} trials for n={n}; {ctx}). Use method='kde' or a larger bw_factor.")

    try:
        out, n_trials, _ = sampler.sample(nb_samples=n, batch_size=batch_size, entropy=seed)
    except Exception as e:
        raise AssertionError(f"rejection: RejectionSampler.sample failed ({ctx}): {e!r}") from e
    assert len(out) == n and n_trials >= n, f"rejection: got {len(out)} samples in {n_trials} trials, wanted {n}"
    return out, int(n_trials)
