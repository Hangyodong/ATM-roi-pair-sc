"""조건부 잠재 prior p(z | pair) (W2-b).

왜 필요한가 (전부 `outputs/eval/w1e_latent_modality.json` 실측)
--------------------------------------------------------------
지금까지 prior 는 `z ~ N(mu_pair, I)` 였다 (`roi_pair_embedding.prior_mean`,
`roi_atm.sample_z`). 분산이 항등행렬로 **고정**이고 다봉도 표현하지 못한다.

  pair 안 GT latent sd    0.180   <- prior sd 1.0 은 차원당 5.6배 과대
  ||mean_gt - mu_pair||   4.25    (||z|| 8.0 대비)
  다봉 pair 비율          98.7%   (검정력 확인 구간 n>=200), BIC 최소 K 중앙값 7

오라클 사다리 (`k_ladder`, precision_ratio 낮을수록 좋음):
  N(mu_pair, I) 36.6 -> pair별 대각 가우시안 7.32 -> mix2 5.08 -> mix3 4.27 -> mix5 3.36

이 모듈은 그 사다리를 코드로 올라가기 위한 **분포 수학 한 벌**이다. 같은 함수를
(1) 학습되는 nn.Module 헤드와 (2) 오프라인 채점(`scripts/48_prior_ladder.py`)이 함께 쓴다.
채점이 재는 것이 실제로 모델이 쓰는 수학과 같아야 사다리 값이 의미를 갖는다.

단계
----
1단계 대각 가우시안: `roi_pair_embedding.prior_log_sigma` (0-init) 로 이미 들어가 있다.
2단계 K-혼합:       `PairMixturePrior` (여기). K=1 이면 1단계와 완전히 같다.

0-init 규약
-----------
K=1 은 log_sigma=0 -> sigma=1 -> `mu + 1.0*eps == mu + eps` 로 **bit-exact** 기존 동작.
K>1 은 성분이 전부 겹치면 gradient 가 대칭이라 영원히 갈라지지 않는다. 그래서 평균 offset
에만 `init_std` 크기의 대칭 깨기 잡음을 준다: 분포는 `N(mu_pair, I)` 에서 그만큼만 벗어나고
(기본 0.05, GT sd 0.18 의 1/4) RNG 소비 순서가 달라 bit-exact 는 K=1 에서만 성립한다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["LOG2PI", "diag_log_prob", "diag_sample", "mixture_log_prob", "mixture_sample",
           "fit_diag", "fit_mixture", "PairMixturePrior", "as_metric_prior"]

LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------- 분포 수학
def diag_log_prob(z: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """log N(z; mu, diag(exp(2*log_sigma))).  z [n,D], mu/log_sigma [n,D] 또는 [D] -> [n]."""
    d = (z - mu) * torch.exp(-log_sigma)
    return -0.5 * (d.pow(2) + 2.0 * log_sigma + LOG2PI).sum(-1)


def diag_sample(mu: torch.Tensor, log_sigma: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """mu + sigma * eps.  log_sigma == 0 이면 `mu + eps` 와 bit-exact 동일."""
    assert eps.shape[-1] == mu.shape[-1], (eps.shape, mu.shape)
    return mu + torch.exp(log_sigma) * eps


def mixture_log_prob(z: torch.Tensor, means: torch.Tensor, log_sigmas: torch.Tensor,
                     logits: torch.Tensor) -> torch.Tensor:
    """log sum_k w_k N(z; mu_k, diag(exp(2 log_sigma_k))).  means [K,D] -> [n]."""
    assert means.ndim == 2 and means.shape == log_sigmas.shape, (means.shape, log_sigmas.shape)
    assert logits.shape == (means.shape[0],), (logits.shape, means.shape)
    d = (z[:, None, :] - means[None]) * torch.exp(-log_sigmas[None])      # [n,K,D]
    comp = -0.5 * (d.pow(2) + 2.0 * log_sigmas[None] + LOG2PI).sum(-1)    # [n,K]
    return torch.logsumexp(torch.log_softmax(logits, 0)[None] + comp, dim=1)


def mixture_sample(n: int, means: torch.Tensor, log_sigmas: torch.Tensor, logits: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    idx = torch.multinomial(torch.softmax(logits, 0), n, replacement=True, generator=generator)
    eps = torch.randn(n, means.shape[1], device=means.device, dtype=means.dtype,
                      generator=generator)
    return diag_sample(means[idx], log_sigmas[idx], eps)


# --------------------------------------------------- 조건 하나짜리 적합 (채점 · 오라클용)
def fit_diag(z: torch.Tensor, min_log_sigma: float = -6.0, max_log_sigma: float = 3.0,
             mu: torch.Tensor | None = None):
    """대각 가우시안의 닫힌 형태 MLE. `mu` 를 주면 평균은 고정하고 분산만 적합한다.

    평균 고정 분기가 곧 "prior_mu 는 그대로 두고 분산만 학습" 의 상한이다.
    """
    assert z.ndim == 2 and z.shape[0] >= 2, z.shape
    m = z.mean(0) if mu is None else mu.reshape(-1)
    var = (z - m).pow(2).mean(0)                     # 평균 고정 시에도 MLE 는 2차 적률
    ls = (0.5 * var.clamp_min(1e-12).log()).clamp(min_log_sigma, max_log_sigma)
    return m, ls


def _kmeanspp(z: torch.Tensor, k: int, g: torch.Generator) -> torch.Tensor:
    c = z[torch.randint(z.shape[0], (1,), generator=g, device=z.device)]
    for _ in range(k - 1):
        d2 = torch.cdist(z, c).pow(2).min(1).values.clamp_min(0)
        p = d2 / d2.sum() if float(d2.sum()) > 0 else torch.ones_like(d2) / d2.numel()
        c = torch.cat([c, z[torch.multinomial(p, 1, generator=g)]], 0)
    return c


def fit_mixture(z: torch.Tensor, k: int, steps: int = 400, lr: float = 0.05, seed: int = 0,
                min_log_sigma: float = -6.0, max_log_sigma: float = 3.0):
    """K-혼합을 **이 모듈의 log_prob 그대로** 최대우도 적합 (kmeans++ 초기화 + Adam).

    sklearn EM 이 아니라 우리 수학으로 적합해야, 채점에서 나온 사다리 값이 학습된 헤드가
    실제로 도달할 수 있는 값이라는 뜻이 된다.
    """
    assert z.ndim == 2 and z.shape[0] > k, (z.shape, k)
    if k == 1:
        m, ls = fit_diag(z, min_log_sigma, max_log_sigma)
        return m[None], ls[None], torch.zeros(1, device=z.device)
    g = torch.Generator(device=z.device).manual_seed(seed)
    means = _kmeanspp(z, k, g).clone().requires_grad_(True)
    _, ls0 = fit_diag(z, min_log_sigma, max_log_sigma)
    # 성분 분산은 전체 분산 / k^(2/D) 가 아니라 그냥 전체 분산에서 출발한다 (안전한 과대 초기화).
    log_sigmas = ls0[None].repeat(k, 1).clone().requires_grad_(True)
    logits = torch.zeros(k, device=z.device, requires_grad=True)
    opt = torch.optim.Adam([means, log_sigmas, logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = -mixture_log_prob(z, means, log_sigmas.clamp(min_log_sigma, max_log_sigma),
                                 logits).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = (means.detach(), log_sigmas.detach().clamp(min_log_sigma, max_log_sigma),
               logits.detach())
    assert all(torch.isfinite(t).all() for t in out), "혼합 적합이 발산했다"
    return out


# ------------------------------------------------------------------ 학습되는 헤드 (2단계)
class PairMixturePrior(nn.Module):
    """p(z | pair) = sum_k w_k(pair) N(mu_pair + d_k(pair), diag(exp(2*ls_k(pair)))).

    `ROIPairEmbedding` 의 1단계(대각 가우시안) 위에 얹는다. `base_mu`/`base_log_sigma` 로
    1단계 출력을 받아 성분별 **offset** 만 예측하므로, offset 이 0 이면 1단계와 같아진다.
    """

    def __init__(self, emb_dim: int, latent_dim: int, k: int = 5, init_std: float = 0.05,
                 log_sigma_range=(-6.0, 3.0), seed: int = 0):
        super().__init__()
        assert k >= 1, k
        self.k, self.latent_dim, self.log_sigma_range = int(k), int(latent_dim), log_sigma_range
        self.head = nn.Linear(emb_dim, k * (2 * latent_dim + 1))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        # 대칭 깨기: 성분 평균 offset 의 bias 에만 잡음. K=1 이면 넣지 않는다 -> bit-exact 유지.
        if k > 1 and init_std > 0:
            g = torch.Generator().manual_seed(seed)
            b = self.head.bias.data.view(k, 2 * latent_dim + 1)
            b[:, :latent_dim] = torch.randn(k, latent_dim, generator=g) * init_std

    def params(self, pair_vec: torch.Tensor, base_mu: torch.Tensor,
               base_log_sigma: torch.Tensor):
        """pair_vec [N,E], base_* [N,D] -> means [N,K,D], log_sigmas [N,K,D], logits [N,K]."""
        n, d, k = pair_vec.shape[0], self.latent_dim, self.k
        assert base_mu.shape == (n, d) and base_log_sigma.shape == (n, d), base_mu.shape
        h = self.head(pair_vec).view(n, k, 2 * d + 1)
        lo, hi = self.log_sigma_range
        return (base_mu[:, None] + h[..., :d],
                (base_log_sigma[:, None] + h[..., d:2 * d]).clamp(lo, hi),
                h[..., -1])

    def log_prob(self, z: torch.Tensor, pair_vec, base_mu, base_log_sigma) -> torch.Tensor:
        """z [N,D] 와 조건 [N,...] 이 1:1 대응 (샘플마다 자기 조건). -> [N]."""
        mm, ll, lg = self.params(pair_vec, base_mu, base_log_sigma)
        dd = (z[:, None, :] - mm) * torch.exp(-ll)
        comp = -0.5 * (dd.pow(2) + 2.0 * ll + LOG2PI).sum(-1)
        return torch.logsumexp(torch.log_softmax(lg, -1) + comp, dim=-1)

    def sample(self, pair_vec, base_mu, base_log_sigma,
               generator: torch.Generator | None = None) -> torch.Tensor:
        mm, ll, lg = self.params(pair_vec, base_mu, base_log_sigma)
        idx = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).squeeze(-1)
        r = torch.arange(idx.shape[0], device=idx.device)
        mu, ls = mm[r, idx], ll[r, idx]
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return diag_sample(mu, ls, eps)


# ------------------------------------------------------------------------- 채점 어댑터
def as_metric_prior(mu, log_sigma, logits=None):
    """`evaluation/prior_metrics` 의 LatentPrior 로 감싼다 (그쪽은 logvar = 2*log_sigma 규약)."""
    from ..evaluation import prior_metrics as pm
    if logits is None:
        return pm.GaussianPrior(mu, 2.0 * torch.as_tensor(log_sigma))
    return pm.MixturePrior(mu, 2.0 * torch.as_tensor(log_sigma), logits)
