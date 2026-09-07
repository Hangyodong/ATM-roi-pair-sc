"""잠재 prior p(z | cond) 채점 지표 (W1-e).

왜 z 공간에서 재는가
--------------------
prior 를 바꿀 때마다 tractogram 을 **생성**해 voxel dice 로 채점하면 pair 하나에 수 초가 든다.
`training/run.py` 의 `pair_dice` 가 기본으로 꺼져 있는 이유가 그것이다. 그런데 우리가 바꾸는
대상은 `p(z | cond)` 하나뿐이므로, decoder 를 고정해 두는 한 **z (64차원) 에서 직접** 비교하면
같은 질문에 밀리초로 답할 수 있다. GT posterior latent 집합 `z_gt` 를 한 번 캐시해 두면
그 뒤 prior 후보는 GPU 없이 채점된다.

지표 3종
--------
`latent_nll`      GT latent 의 prior 하 음의 로그우도 (nats/샘플). prior 가 `log_prob` 만
                  제공하면 가우시안·혼합·flow 어디서든 **정확히** 계산된다 (근사·샘플링 없음).
                  단점: 질량이 GT 를 덮기만 하면 좋게 나온다 (너무 넓은 prior 를 벌하지 않는다).
`latent_mmd`      prior 샘플과 GT latent 사이의 MMD^2 (RBF 다중 대역폭). 양방향 불일치를 잡고
                  순열검정으로 p 값을 낸다. NLL 이 못 보는 "너무 넓다" 를 여기서 본다.
`latent_nn_dist`  양방향 최근접거리. sample->GT 가 precision(헛것을 만드는가),
                  GT->sample 이 recall(빠뜨린 갈래가 있는가). GT 자체의 최근접거리로 나눠
                  차원·스케일에 무관한 비율로 보고한다.

세 지표는 서로 다른 실패를 잡는다. 하나만 보지 말 것.

대역폭 선택 근거 (`latent_mmd`)
-------------------------------
RBF 커널 폭은 **pooled(샘플+GT) 쌍거리의 median heuristic** 으로 잡는다:
`sigma^2 = median(||x-y||^2) / 2` -> `k(x,y) = exp(-||x-y||^2 / median)`.
- pooled 로 잡는 이유: 한쪽 집합만 쓰면 대역폭이 그 집합의 퍼짐에 따라 달라져 두 집합을
  비대칭으로 다루게 된다. pooled 통계는 순열에 불변이라 순열검정도 그대로 유효하다.
- 한 폭에 의존하지 않도록 `MMD_SCALES` 배수(1/4~4배) 커널의 합을 쓴다. 분포 차이가
  대역폭보다 훨씬 크거나 작으면 단일 RBF 는 눈이 먼다 -- 다중 스케일이 그 실패를 막는다.

주의
----
- 이 지표들은 decoder 를 통과시키지 않는다. z 가 좋아졌다는 것이 곧 dice 가 좋아졌다는
  뜻은 아니다. **선별용**으로 쓰고, 최종 판정은 여전히 생성 dice 로 한다.
- `z_gt` 는 posterior 평균(mu)을 쓰는 것을 전제로 한다. 샘플을 쓰면 등방 잡음이 더해져
  모드가 흐려진다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "LatentPrior", "GaussianPrior", "MixturePrior", "CallablePrior",
    "current_prior", "fit_gaussian", "fit_mixture",
    "latent_nll", "latent_mmd", "mmd_permutation_test", "latent_nn_dist",
    "evaluate_prior", "validate_metrics", "MMD_SCALES", "BASELINE",
]

MMD_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)   # median heuristic 대비 배수 (다중 대역폭)

# 현재 prior `N(mu_pair, I)` 의 실측 기준값. 앞으로의 개선은 이 값 대비로 판정한다.
# 출처: `scripts/46_latent_modality.py` (ckpt retrain/p4_joint step3000, val 14명 x pair 20개
#       = 280 pair, pair 당 GT 가닥 40~256개, latent 절반으로 채점).
#       전체 결과 `outputs/eval/w1e_latent_modality.json`, 재현 명령은 그 안의 "cmd".
# 읽는 법: precision/recall_ratio 는 "GT latent 끼리의 최근접거리" 배수다. 1.0 이 목표,
#          현재 38.5 는 prior 샘플이 실제 latent 로부터 GT 간격의 38배 떨어져 있다는 뜻이다.
BASELINE = {
    "current_N(mu_pair,I)": {"nll": 71.62, "mmd2": 1.8249, "mmd_p": 0.005,
                             "precision_ratio": 38.50, "recall_ratio": 31.80,
                             "frac_mmd_significant": 1.00},
    # 참고선 1: 조건부 평균을 아예 빼면 (mu_pair 가 실제로 얼마나 기여하나 -- 거의 안 한다)
    "N(0,I)": {"nll": 91.12, "mmd2": 2.5528, "precision_ratio": 45.58, "recall_ratio": 38.63},
    # 참고선 2: pair 별 대각 가우시안을 GT 절반으로 적합 (분산만 고쳐도 여기까지)
    "oracle_gauss_fit": {"nll": -44.65, "mmd2": 0.4225, "precision_ratio": 7.85,
                         "recall_ratio": 5.77},
    # 참고선 3: pair 별 2-혼합 (다봉까지 반영)
    "oracle_mixture_k2": {"nll": -64.80, "mmd2": 0.1783, "precision_ratio": 4.50,
                          "recall_ratio": 3.88},
}
# GT latent 자체의 기하 (같은 실측): pair 안 표준편차 0.180 / 가닥별 posterior 표준편차 0.810 /
# ||mean_gt - mu_pair|| 4.25 / ||z|| 8.00. 즉 prior 는 GT 보다 차원당 5.6배 넓고 평균도 어긋나 있다.
_LOG2PI = math.log(2.0 * math.pi)


def _as2d(x, ref: torch.Tensor | None = None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32)
    if ref is not None:
        t = t.to(ref.device)
    if t.ndim == 1:
        t = t[None]
    assert t.ndim == 2, t.shape
    assert torch.isfinite(t).all(), "latent 에 NaN/Inf"
    return t


# --------------------------------------------------------------------------- prior
class LatentPrior:
    """채점에 필요한 최소 인터페이스. 이 둘만 있으면 flow/diffusion 도 그대로 들어온다.

    한 인스턴스는 **조건 하나** (한 (subject, pair)) 의 p(z | cond) 를 뜻한다.
    `cond` 인자는 조건을 prior 안에 넣지 않고 밖에서 주는 구현(예: 공유 flow)을 위한 통로다.
    """

    dim: int

    def log_prob(self, z: torch.Tensor, cond=None) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, n: int, cond=None, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError


class GaussianPrior(LatentPrior):
    """N(mean, diag(exp(logvar))). logvar=0 이면 현재 구현과 같은 등방 단위분산."""

    def __init__(self, mean, logvar=0.0):
        self.mean = _as2d(mean).reshape(-1)
        self.dim = int(self.mean.numel())
        lv = torch.as_tensor(logvar, dtype=torch.float32)
        lv = lv.expand(self.dim).clone() if lv.ndim == 0 else lv.reshape(-1)
        assert lv.numel() == self.dim, (lv.shape, self.dim)
        self.logvar = lv.to(self.mean.device)
        assert torch.isfinite(self.mean).all() and torch.isfinite(self.logvar).all()

    def log_prob(self, z, cond=None) -> torch.Tensor:
        z = _as2d(z, self.mean)
        assert z.shape[1] == self.dim, (z.shape, self.dim)
        d = z - self.mean
        return -0.5 * ((d * d) * torch.exp(-self.logvar) + self.logvar + _LOG2PI).sum(-1)

    def sample(self, n, cond=None, generator=None) -> torch.Tensor:
        eps = torch.randn(n, self.dim, generator=generator, device=self.mean.device,
                          dtype=self.mean.dtype)
        return self.mean + eps * torch.exp(0.5 * self.logvar)


class MixturePrior(LatentPrior):
    """sum_k w_k N(mu_k, diag(exp(logvar_k))). 다봉 prior 후보(= pair 별 K-혼합)의 채점용."""

    def __init__(self, means, logvars=0.0, logits=None):
        self.means = _as2d(means)                                   # [K, D]
        self.k, self.dim = int(self.means.shape[0]), int(self.means.shape[1])
        lv = torch.as_tensor(logvars, dtype=torch.float32)
        if lv.ndim == 0:
            lv = lv.expand(self.k, self.dim).clone()
        elif lv.ndim == 1:
            lv = lv[:, None].expand(self.k, self.dim).clone() if lv.numel() == self.k else \
                lv[None].expand(self.k, self.dim).clone()
        assert lv.shape == self.means.shape, (lv.shape, self.means.shape)
        self.logvars = lv.to(self.means.device)
        lg = torch.zeros(self.k) if logits is None else torch.as_tensor(logits, dtype=torch.float32)
        assert lg.numel() == self.k, (lg.shape, self.k)
        self.log_w = torch.log_softmax(lg.reshape(-1).to(self.means.device), 0)
        assert torch.isfinite(self.means).all() and torch.isfinite(self.logvars).all()

    def _component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        d = z[:, None, :] - self.means[None]                        # [n, K, D]
        return -0.5 * ((d * d) * torch.exp(-self.logvars) + self.logvars + _LOG2PI).sum(-1)

    def log_prob(self, z, cond=None) -> torch.Tensor:
        z = _as2d(z, self.means)
        assert z.shape[1] == self.dim, (z.shape, self.dim)
        return torch.logsumexp(self.log_w[None] + self._component_log_prob(z), dim=1)

    def sample(self, n, cond=None, generator=None) -> torch.Tensor:
        idx = torch.multinomial(self.log_w.exp(), n, replacement=True, generator=generator)
        eps = torch.randn(n, self.dim, generator=generator, device=self.means.device,
                          dtype=self.means.dtype)
        return self.means[idx] + eps * torch.exp(0.5 * self.logvars[idx])


class CallablePrior(LatentPrior):
    """flow / diffusion 처럼 log_prob 과 sample 을 함수로만 주는 경우."""

    def __init__(self, dim: int, log_prob_fn, sample_fn):
        self.dim, self._lp, self._sm = int(dim), log_prob_fn, sample_fn

    def log_prob(self, z, cond=None) -> torch.Tensor:
        out = self._lp(_as2d(z), cond)
        assert out.ndim == 1, out.shape
        return out

    def sample(self, n, cond=None, generator=None) -> torch.Tensor:
        out = _as2d(self._sm(n, cond, generator))
        assert out.shape == (n, self.dim), (out.shape, n, self.dim)
        return out


def current_prior(mu_pair) -> GaussianPrior:
    """현재 구현 그대로의 prior: `z ~ N(mu_pair, I)` (roi_atm.sample_z, roi_pair_embedding.prior_mean).

    분산이 학습되지 않고 anatomy 도 들어가지 않는다 -- 그것이 지금 재려는 기준값이다.
    """
    return GaussianPrior(mu_pair, logvar=0.0)


def fit_gaussian(z, diag: bool = True, min_var: float = 1e-6) -> GaussianPrior:
    """그 조건의 GT latent 로 적합한 대각 가우시안. '단봉 prior 로 가능한 최선' 상한."""
    z = _as2d(z)
    assert z.shape[0] >= 2, z.shape
    assert diag, "full covariance 는 아직 필요 없다 (D=64, n<=256)"
    return GaussianPrior(z.mean(0), torch.log(z.var(0, unbiased=True).clamp_min(min_var)))


def fit_mixture(z, k: int, seed: int = 0, reg: float = 1e-6) -> MixturePrior:
    """GT latent 로 적합한 대각 K-혼합. '다봉 prior 로 가능한 최선' 상한 (오라클)."""
    from sklearn.mixture import GaussianMixture
    z = _as2d(z)
    assert z.shape[0] > k, (z.shape, k)
    g = GaussianMixture(k, covariance_type="diag", reg_covar=reg, n_init=5,
                        random_state=seed).fit(z.numpy())
    return MixturePrior(g.means_, np.log(np.maximum(g.covariances_, reg)), np.log(g.weights_ + 1e-30))


# --------------------------------------------------------------------------- 지표
def latent_nll(prior: LatentPrior, z_gt, cond=None) -> float:
    """GT latent 의 prior 하 평균 음의 로그우도 (nats / 샘플). 낮을수록 좋다.

    가우시안·혼합은 닫힌 형태, flow 는 change-of-variables 로 정확히 나온다. 근사 없음.
    """
    z = _as2d(z_gt)
    lp = prior.log_prob(z, cond)
    assert lp.shape == (z.shape[0],), (lp.shape, z.shape)
    assert torch.isfinite(lp).all(), "log_prob 에 NaN/Inf"
    return float(-lp.mean())


def _rbf_sum(d2: torch.Tensor, med2: torch.Tensor, scales=MMD_SCALES) -> torch.Tensor:
    """sum_s exp(-d2 / (s * med2)). med2 = pooled 쌍거리 제곱의 중앙값."""
    out = torch.zeros_like(d2)
    for s in scales:
        out = out + torch.exp(-d2 / (s * med2))
    return out


def _pooled_kernel(z_sample: torch.Tensor, z_gt: torch.Tensor, scales):
    """pooled median heuristic 로 만든 (m+n)x(m+n) 커널 행렬. 라벨 순열에 불변."""
    pool = torch.cat([z_sample, z_gt], 0)
    d2 = torch.cdist(pool, pool).pow(2)
    n = pool.shape[0]
    iu = torch.triu_indices(n, n, 1, device=pool.device)
    med2 = d2[iu[0], iu[1]].median().clamp_min(1e-12)
    return _rbf_sum(d2, med2, scales), float(med2)


def _mmd2_from_kernel(K: torch.Tensor, m: int) -> float:
    """비편향 MMD^2 (대각 제외). K 의 앞 m 행이 sample, 나머지가 GT."""
    n = K.shape[0] - m
    assert m > 1 and n > 1, (m, n)
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    sxx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
    syy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())


def latent_mmd(z_sample, z_gt, scales=MMD_SCALES) -> float:
    """비편향 MMD^2 (다중 대역폭 RBF 합). 같은 분포면 0 근처(음수 가능), 다르면 양수.

    대역폭 근거는 모듈 docstring 참고 (pooled median heuristic x MMD_SCALES).
    """
    a, b = _as2d(z_sample), _as2d(z_gt)
    assert a.shape[1] == b.shape[1], (a.shape, b.shape)
    K, _ = _pooled_kernel(a, b, scales)
    return _mmd2_from_kernel(K, a.shape[0])


def mmd_permutation_test(z_sample, z_gt, scales=MMD_SCALES, n_perm: int = 200,
                         seed: int = 0) -> dict:
    """MMD^2 + 순열검정. 커널은 pooled 통계로 만들어 순열에 불변이므로 검정이 정확하다."""
    a, b = _as2d(z_sample), _as2d(z_gt)
    K, med2 = _pooled_kernel(a, b, scales)
    m, N = a.shape[0], K.shape[0]
    obs = _mmd2_from_kernel(K, m)
    g = torch.Generator(device=K.device).manual_seed(seed)
    null = []
    for _ in range(n_perm):
        p = torch.randperm(N, generator=g, device=K.device)
        null.append(_mmd2_from_kernel(K[p][:, p], m))
    null = np.asarray(null, np.float64)
    q95 = float(np.quantile(null, 0.95)) if n_perm else float("nan")
    return {"mmd2": obs, "p": float((null >= obs).sum() + 1) / (n_perm + 1),
            "null_q95": q95, "mmd2_over_null_q95": obs / q95 if q95 > 0 else float("inf"),
            "median_sq_dist": med2, "n_perm": n_perm}


def latent_nn_dist(z_sample, z_gt) -> dict:
    """양방향 최근접거리 (precision / recall 형태).

      d_sample_to_gt  샘플이 실제 latent 에서 얼마나 떨어져 있나  -> precision (헛것)
      d_gt_to_sample  실제 latent 가 샘플에서 얼마나 떨어져 있나  -> recall (놓친 갈래)
      d_gt_self       GT 안의 leave-one-out 최근접거리 = 자연 스케일

    비율(`*_ratio`) 이 1 근처면 "GT 끼리 떨어진 만큼" 이라는 뜻이라 차원·스케일에 무관하다.
    """
    a, b = _as2d(z_sample), _as2d(z_gt)
    assert a.shape[1] == b.shape[1], (a.shape, b.shape)
    assert b.shape[0] >= 2, b.shape
    d = torch.cdist(a, b)
    dgg = torch.cdist(b, b).fill_diagonal_(float("inf"))
    s2g, g2s, gg = d.min(1).values, d.min(0).values, dgg.min(1).values
    scale = max(float(gg.median()), 1e-12)
    out = {"d_sample_to_gt": float(s2g.median()), "d_gt_to_sample": float(g2s.median()),
           "d_gt_self": float(gg.median()),
           "d_sample_to_gt_mean": float(s2g.mean()), "d_gt_to_sample_mean": float(g2s.mean())}
    out["precision_ratio"] = out["d_sample_to_gt"] / scale
    out["recall_ratio"] = out["d_gt_to_sample"] / scale
    return out


def evaluate_prior(prior: LatentPrior, z_gt, cond=None, n_sample: int | None = None,
                   seed: int = 0, n_perm: int = 200, scales=MMD_SCALES) -> dict:
    """한 조건의 prior 를 세 지표로 한 번에 채점한다. GPU 불필요, D=64 에서 밀리초."""
    z = _as2d(z_gt)
    n, D = z.shape
    assert n >= 4, f"조건당 latent 가 너무 적다 (n={n})"
    m = int(n_sample or max(n, 128))
    g = torch.Generator(device=z.device).manual_seed(seed)
    s = _as2d(prior.sample(m, cond, g), z)
    assert s.shape == (m, D), (s.shape, m, D)
    assert torch.isfinite(s).all(), "prior 샘플에 NaN/Inf"
    out = {"n_gt": n, "n_sample": m, "dim": D,
           "nll": latent_nll(prior, z, cond)}
    out["nll_per_dim"] = out["nll"] / D
    out.update({f"mmd_{k}": v for k, v in
                mmd_permutation_test(s, z, scales, n_perm, seed).items()})
    out.update(latent_nn_dist(s, z))
    return out


# --------------------------------------------------------------------------- 자기검증
def _gauss(dim, n, gen, mean=None, sd=1.0):
    z = torch.randn(n, dim, generator=gen) * sd
    return z if mean is None else z + mean


def validate_metrics(dim: int = 64, n: int = 256, seed: int = 0, n_perm: int = 200) -> dict:
    """정답을 아는 합성 데이터로 세 지표가 실제로 분포 차이에 반응하는지 확인한다.

    같은 분포에서 뽑으면 좋은 점수, 다른 분포면 나쁜 점수여야 한다. assert 로 순서를 강제한다.
    GT 생성 seed 와 prior 샘플링 seed 를 반드시 분리한다 (같으면 샘플이 GT 와 글자 그대로
    같아져서 모든 지표가 0 이 나오고, 그것을 '완벽' 으로 오독하게 된다 -- 실제로 겪었다).

    고차원 주의: D=64 에서는 거리 집중 때문에 최근접거리 비율이 눌린다. 그래서 nn 지표의
    직관적 거동(다봉 GT + 단봉 prior -> recall 붕괴)은 저차원 블록에서 따로 확인한다.
    MMD 와 NLL 은 64차원에서도 그대로 예민하다.
    """
    gg = torch.Generator().manual_seed(seed + 1000)      # GT 전용 seed (prior 샘플링과 분리)
    z = _gauss(dim, n, gg)                               # GT ~ N(0, I)
    e = torch.zeros(dim); e[0] = 1.0
    far = torch.zeros(dim); far[0] = 30.0

    cases = {
        "same":       GaussianPrior(torch.zeros(dim), 0.0),               # 정답
        "shift_1":    GaussianPrior(e * 1.0, 0.0),                        # 1축 1 sigma 이동 (미세)
        "shift_3":    GaussianPrior(e * 3.0, 0.0),                        # 1축 3 sigma 이동
        "too_wide":   GaussianPrior(torch.zeros(dim), math.log(4.0)),     # 분산 4배
        "too_narrow": GaussianPrior(torch.zeros(dim), math.log(0.25)),    # 분산 1/4
        # 질량의 절반을 엉뚱한 곳에 두는 prior. GT 를 '덮기는' 하므로 NLL 은 거의 안 오른다
        # (~log 2 = 0.69 nats) -- NLL 만 보면 안 되는 이유를 보여주는 사례다.
        "half_wasted": MixturePrior(torch.stack([torch.zeros(dim), far]), 0.0,
                                    logits=torch.log(torch.tensor([0.35, 0.65]))),
    }
    res = {k: evaluate_prior(p, z, n_sample=n, seed=seed, n_perm=n_perm) for k, p in cases.items()}

    # 다봉 GT vs 단봉/혼합 prior (64차원)
    sep = 20.0        # 64차원 거리 집중 때문에 sep=12 면 최적 단봉 적합과 MMD 로 거의 구분되지 않는다
    zb = torch.cat([_gauss(dim, n // 2, gg) - sep / 2 * e, _gauss(dim, n - n // 2, gg) + sep / 2 * e], 0)
    res["bimodal_gt_uni_prior"] = evaluate_prior(
        fit_gaussian(zb), zb, n_sample=n, seed=seed, n_perm=n_perm)       # 최선의 단봉 적합
    res["bimodal_gt_mix_prior"] = evaluate_prior(
        MixturePrior(torch.stack([-sep / 2 * e, sep / 2 * e]), 0.0), zb,
        n_sample=n, seed=seed, n_perm=n_perm)                            # 올바른 혼합

    # 저차원 블록: 거리 집중이 없는 곳에서 nn 지표가 제 방향으로 움직이는지.
    d4 = 4
    e4 = torch.zeros(d4); e4[0] = 1.0
    z4 = torch.cat([_gauss(d4, n // 2, gg) - 4 * e4, _gauss(d4, n - n // 2, gg) + 4 * e4], 0)
    low = {
        # 최적 단봉 적합: 넓어서 recall 은 버티지만 모드 사이 빈 곳을 채운다 -> precision 악화
        "uni_fit": evaluate_prior(fit_gaussian(z4), z4, n_sample=n, seed=seed, n_perm=n_perm),
        # 현재 구현과 같은 형태 N(mean, I): 분산이 고정이라 가운데에만 뭉친다 -> recall 붕괴
        "uni_unit": evaluate_prior(GaussianPrior(z4.mean(0), 0.0), z4, n_sample=n, seed=seed,
                                   n_perm=n_perm),
        "mix": evaluate_prior(MixturePrior(torch.stack([-4 * e4, 4 * e4]), 0.0), z4,
                              n_sample=n, seed=seed, n_perm=n_perm)}

    s, ok = res["same"], {}
    # (1) MMD 순열검정: 같은 분포는 통과, 다른 분포는 탈락.
    ok["mmd_same_not_significant"] = s["mmd_p"] > 0.05
    for k in ("shift_3", "too_wide", "too_narrow", "half_wasted"):
        ok[f"mmd_detects_{k}"] = res[k]["mmd_p"] <= 0.05
    ok["mmd_detects_bimodal"] = res["bimodal_gt_uni_prior"]["mmd_p"] <= 0.05
    ok["mmd_accepts_correct_mixture"] = res["bimodal_gt_mix_prior"]["mmd_p"] > 0.05
    # (2) MMD^2 는 차이가 커질수록 커진다.
    ok["mmd_monotone_shift"] = s["mmd_mmd2"] < res["shift_1"]["mmd_mmd2"] < res["shift_3"]["mmd_mmd2"]
    # (3) NLL: 평균이 어긋나면 오르고, 올바른 혼합이 최선의 단봉보다 낮다.
    ok["nll_monotone_shift"] = s["nll"] < res["shift_1"]["nll"] < res["shift_3"]["nll"]
    ok["nll_mixture_beats_unimodal"] = (res["bimodal_gt_mix_prior"]["nll"]
                                        < res["bimodal_gt_uni_prior"]["nll"])
    # (4) NLL 은 '질량 절반을 버린' prior 를 거의 벌하지 않는다 -> MMD/nn 이 따로 필요하다.
    ok["nll_blind_to_wasted_mass"] = (res["half_wasted"]["nll"] - s["nll"]) < 1.5
    ok["nn_catches_wasted_mass"] = res["half_wasted"]["precision_ratio"] > 1.5 * s["precision_ratio"]
    ok["nll_blind_check_mmd_flags_it"] = res["half_wasted"]["mmd_p"] <= 0.05
    # (5) 최근접거리: 같은 분포면 비율 ~1, 평균이 어긋나면 precision 악화.
    ok["nn_same_near_one"] = 0.7 < s["precision_ratio"] < 1.4 and 0.7 < s["recall_ratio"] < 1.4
    ok["nn_detects_shift"] = res["shift_3"]["precision_ratio"] > s["precision_ratio"]
    ok["nn_detects_wide"] = res["too_wide"]["precision_ratio"] > s["precision_ratio"]
    # (6) 저차원(거리 집중 없음)에서 두 가지 단봉 실패가 각각 다른 지표에 잡히는지.
    #     넓은 단봉 -> 모드 사이를 채운다 -> precision 악화 / 단위분산 단봉 -> 가운데만 -> recall 붕괴.
    ok["nn_lowdim_gap_precision"] = low["uni_fit"]["precision_ratio"] > 1.4 * low["mix"]["precision_ratio"]
    ok["nn_lowdim_unit_recall_bad"] = low["uni_unit"]["recall_ratio"] > 2.0 * low["mix"]["recall_ratio"]
    ok["nll_lowdim_mixture_beats_unimodal"] = (low["mix"]["nll"] < low["uni_fit"]["nll"]
                                               and low["mix"]["nll"] < low["uni_unit"]["nll"])

    keys = ("nll", "mmd_mmd2", "mmd_p", "precision_ratio", "recall_ratio")
    bad = [k for k, v in ok.items() if not v]
    assert not bad, ("지표 자기검증 실패: " + str(bad) + "\n"
                     + str({k: {m: round(v[m], 4) for m in keys} for k, v in res.items()})
                     + "\nlowdim " + str({k: {m: round(v[m], 4) for m in keys} for k, v in low.items()}))
    return {"checks": {k: bool(v) for k, v in ok.items()}, "cases": res, "lowdim_d4": low,
            "dim": dim, "n": n, "seed": seed,
            "note": ("D=64 에서는 거리 집중으로 nn 비율이 눌린다 (다봉이라도 recall_ratio 가 "
                     "1.3배 정도밖에 안 오른다). MMD 와 NLL 을 1차 지표로, nn 은 방향 진단용으로 쓴다.")}
