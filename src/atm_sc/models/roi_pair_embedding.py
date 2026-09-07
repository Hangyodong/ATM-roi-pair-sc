"""ROI-pair conditioning (pipeline §9).

원본 ATM 은 bundle 마다 별도 모델이라 "bundle 조건" 이 곧 어느 .pth 를 쓰느냐였다.
여기서는 하나의 decoder 를 공유하고 ROI pair 를 embedding 으로 조건화한다.

주입 방식 (upstream 무수정):
  ConvVAE 의 FiLM 층은 anatomical_info [N,512] 만 받는다 (model.py:296-298, 316-317).
  따라서 cond = gain * LayerNorm(anatomy_feature) + Proj(Emb(a) + Emb(b)) 를 그 자리에 넣는다.
  Proj 의 마지막 층은 0 초기화라 시작 시점 cond 는 anatomy 항뿐이다. LayerNorm 은 그 항의
  크기를 pair 항과 맞추려고 새로 넣은 것이고(재학습 설계 §2 ②), pretrained decoder 가 보던
  조건 분포가 바뀌므로 **P0 예열 단계가 필요하다** (좌표 박스 변경과 같은 전례).

undirected 이므로 Emb(a) + Emb(b) (합) 으로 순서 불변성을 구조적으로 보장한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def canonical_pairs(pairs: torch.Tensor) -> torch.Tensor:
    """[N,2] -> (min, max)."""
    assert pairs.ndim == 2 and pairs.shape[1] == 2, pairs.shape
    pairs = pairs.long()                     # npz 는 int16 으로 저장, Embedding 은 long 필요
    return torch.stack([pairs.min(dim=1).values, pairs.max(dim=1).values], dim=1)


class ROIPairEmbedding(nn.Module):
    def __init__(self, n_roi: int, emb_dim: int = 64, cond_dim: int = 512, hidden: int = 256,
                 latent_dim: int = 64, n_modes: int = 2, prior_use_anatomy: bool = False,
                 local_dim: int = 0, prior_local_dim: int = 0,
                 prior_local_rank: int = 0, prior_local_n_roi: int = 0):
        super().__init__()
        self.n_roi, self.emb_dim, self.cond_dim = n_roi, emb_dim, cond_dim
        self.emb = nn.Embedding(n_roi, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.proj = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(), nn.Linear(hidden, cond_dim))
        nn.init.zeros_(self.proj[-1].weight)          # 시작 시 cond == anatomy 항만
        nn.init.zeros_(self.proj[-1].bias)
        # pretrained UNet 의 anatomy feature 는 |a| ~ 0.069 인데 pair_vec 은 1.521 로 22배 크다
        # (실측, PIPELINE_06_FINDINGS.md). 그래서 optimizer 는 anatomy 대신 pair 임베딩만 쓰고
        # T1 을 0 으로 바꿔도 SC 상관이 0.8108 -> 0.8067 밖에 안 변한다.
        # LayerNorm 으로 subject 마다 크기를 고정하고(평균/분산 정규화) 가중치를 1/sqrt(C) 로
        # 두어 LN 출력의 L2 를 1.0 으로 맞춘다 -- pair_vec(1.521) 과 같은 자릿수다 (재학습 설계 §2 ②).
        # 이 스케일을 anatomy_gain 이 아니라 **새 파라미터**에 넣는 이유: 구 checkpoint 에는
        # anatomy_norm 이 없어 missing-key 경로로 이 초기값이 그대로 살고, anatomy_gain(=1.0)
        # 을 실어도 크기가 22배 튀지 않는다.
        # eps 가 기본값(1e-5)이면 안 된다: anatomy 는 |a| ~ 0.069 / 512차원이라 원소 분산이
        # 약 9e-6 으로 eps 와 같은 자릿수다 -> 정규화가 절반쯤 먹히고 크기 의존성이 남는다.
        self.anatomy_norm = nn.LayerNorm(cond_dim, eps=1e-8)
        nn.init.constant_(self.anatomy_norm.weight, cond_dim ** -0.5)
        nn.init.zeros_(self.anatomy_norm.bias)
        # 학습 가능한 스칼라 gain. LN 뒤라 이제 "pair 대비 몇 배로 들을지" 만 조절한다.
        self.anatomy_gain = nn.Parameter(torch.ones(1))
        # pair 별 **국소** anatomy (전략 문서 §3.2). 전역 anatomy 는 subject 성분이 2.1% 뿐인데
        # (subject 간 코사인 0.9994) ROI 국소 pooling 은 13.4% 다 -- global average pooling 이
        # 개인차를 지운 뒤의 벡터만 조건으로 들어가고 있었다. count head 쪽에서 이 입력을 주자
        # 잔차 상관이 0.001 -> 0.034 로, self-shuffled 격차가 -0.005 -> +0.031 로 바뀌었다
        # (outputs/eval/a2_residual_trace_D{,_local}.jsonl). 여기는 그 입력을 **디코더 조건**에도
        # 넣는 통로다. 0-init 이라 켜도 시작 cond 는 bit-exact 하다.
        self.local_dim = int(local_dim)
        self.local_proj = None
        if self.local_dim:
            self.local_proj = nn.Sequential(nn.Linear(self.local_dim, hidden), nn.GELU(),
                                            nn.Linear(hidden, cond_dim))
            nn.init.zeros_(self.local_proj[-1].weight); nn.init.zeros_(self.local_proj[-1].bias)
        # 조건부 prior p(z | pair) = N(mu_pair, I). pair 정보가 z 공간에 직접 들어간다.
        # 근거: 같은 decoder 라도 z 를 GT posterior 은행에서 뽑으면 pair 정확도 0.66, N(0,I) 면 0.
        # 즉 decoder 는 z 에 실린 pair 정보를 듣는다. 0-init -> 시작은 N(0,I).
        self.latent_dim = latent_dim
        self.prior_mu = nn.Linear(emb_dim, latent_dim)
        nn.init.zeros_(self.prior_mu.weight); nn.init.zeros_(self.prior_mu.bias)
        # prior 의 **분산도 pair 마다 학습한다**. 지금까지 I 로 고정돼 있었는데 실측은
        # pair 안 GT latent sd 0.180 / prior sd 1.0 -> 차원당 5.6배 과대였다
        # (outputs/eval/w1e_latent_modality.json:latent_geometry). 분산만 고쳐도 오라클
        # precision_ratio 38.5 -> 7.3 (같은 파일 k_ladder).
        # 0-init 라 sigma = exp(0) = 1.0 -> mu + 1.0*eps 는 기존 mu + eps 와 bit-exact 동일하다.
        self.prior_log_sigma = nn.Linear(emb_dim, latent_dim)
        nn.init.zeros_(self.prior_log_sigma.weight); nn.init.zeros_(self.prior_log_sigma.bias)
        # 생성 단위 모드 (0 = full streamline, 1 = SC edge-aligned segment).
        # 같은 decoder 로 두 가지를 만들되 조건에 모드를 더한다 (EDGE_ALIGNED 전략 §13-14 dual representation).
        # 0-init -> 모드를 붙여도 시작 동작은 지금과 완전히 같다.
        self.n_modes = n_modes
        self.mode_emb = nn.Embedding(n_modes, cond_dim)
        self.mode_prior = nn.Embedding(n_modes, latent_dim)
        self.mode_log_sigma = nn.Embedding(n_modes, latent_dim)
        nn.init.zeros_(self.mode_emb.weight); nn.init.zeros_(self.mode_prior.weight)
        nn.init.zeros_(self.mode_log_sigma.weight)
        # anatomy -> prior 통로. 이게 꺼져 있으면 z 는 pair 만 보므로 subject 성분(조건평균 분산의
        # 16.9%, W2-b subject_ceiling)을 **원리적으로** 못 맞춘다. 0-init 이라 켜도 시작 시점은
        # 기존 체크포인트와 bit-exact 동일하다. 꺼져 있으면 forward 에 들어가지 않아 grad 가
        # None 이고 optimizer 가 건드리지도 않는다.
        # D-f: **pair 의존** subject 조건부 prior. 기존 prior_anatomy 는 Linear(cond_dim -> 2D)
        # 라 출력이 pair 와 무관해서 subject 잔차의 3.8% 만 표현할 수 있었다 (W4-a 구조 제약).
        # 여기는 pair 별 국소 anatomy 와 pair 벡터를 함께 받아 나머지 96.2% 를 겨냥한다.
        # 지도는 이미 있다 -- trainer 의 prior 적합항이 mu 를 GT posterior 평균으로 회귀시킨다.
        # 0-init 이라 켜도 시작은 bit-exact.
        self.prior_local_dim = int(prior_local_dim)
        self.prior_local = None
        if self.prior_local_dim:
            self.prior_local = nn.Sequential(
                nn.Linear(self.prior_local_dim + emb_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 2 * latent_dim))
            nn.init.zeros_(self.prior_local[-1].weight); nn.init.zeros_(self.prior_local[-1].bias)
        # pair 인덱스 저랭크 가지. 왜 -- 위의 공유 MLP 는 실측(D-f 2000 step) 으로 실패했다:
        # posterior mu 의 subject 성분이 **0.566** 인데 prior mu 는 **0.000154** 만 가져왔다.
        # 배울 신호가 56.6% 있는데 0.015% 만 배운 것이다. count_head 의 tier1 에서 겪은 것과 같은
        # 실패다 -- 공유 가중치는 pair 마다 다른 방향의 신호를 표현하지 못한다.
        # 전체 pair 별 [512 -> 128] 은 3321*512*128 = 2.2억 개라 못 쓴다. 공유 압축 U(512->rank)
        # 뒤에 pair 별 [rank -> 2*latent] 를 둔다: rank 4 면 3321*4*128 = 1.7M.
        # bias 는 두지 않는다 -- pair 별 상수는 prior_mu 가 이미 갖고 있어 중복이다.
        self.prior_local_rank, self.prior_local_n_roi = int(prior_local_rank), int(prior_local_n_roi)
        self.prior_local_u = self.prior_local_w = None
        if self.prior_local_dim and self.prior_local_rank and self.prior_local_n_roi:
            n_pair = self.prior_local_n_roi * (self.prior_local_n_roi - 1) // 2
            self.prior_local_u = nn.Linear(self.prior_local_dim, self.prior_local_rank, bias=False)
            self.prior_local_w = nn.Parameter(
                torch.zeros(n_pair, self.prior_local_rank, 2 * latent_dim))   # 0-init -> bit-exact 시작
            # 입력 중심화용 train 평균 ROI feature [n_roi, D/2]. 왜 -- local = [f_i, f_j] 는 대부분
            # pair 정체성이라(실측: 가지 출력의 subject 성분 2%, 98% 가 pair 수준) 가지가 pair 평균
            # 드리프트를 맞추는 데 용량을 다 썼다. 평균을 빼면 bias 가 없는 이 가지는 평균 입력에서
            # 출력이 정확히 0 이라 pair 수준을 표현할 수 없고, subject 편차로만 손실을 줄일 수 있다.
            # 0 이면 중심화 없음(구 checkpoint 호환). run.py 가 train subject 로 채운다.
            self.register_buffer("prior_local_roi_mean",
                                 torch.zeros(self.prior_local_n_roi, self.prior_local_dim // 2))
            # 표준화용 train std. 왜 -- 중심화만 하면 입력 크기가 ~0.02/dim (subject 성분이 7% 라)
            # 이라 u = U(lc) 가 극소이고, W 가 |max| 0.97 까지 커져도 출력이 base 의 3% 를 못 넘는다
            # (실측, 22:06 run). 티어1 의 pair_stats 표준화와 같은 처방. 1 이면 표준화 없음.
            self.register_buffer("prior_local_roi_std",
                                 torch.ones(self.prior_local_n_roi, self.prior_local_dim // 2))
        self.prior_use_anatomy = bool(prior_use_anatomy)
        self.prior_anatomy = nn.Linear(cond_dim, 2 * latent_dim)
        nn.init.zeros_(self.prior_anatomy.weight); nn.init.zeros_(self.prior_anatomy.bias)
        # log_sigma 범위 제한. 0 은 안쪽이라 0-init 동작에는 영향이 없다.
        # 하한 -6 (sigma 2.5e-3) 은 실측 GT sd 0.18(log -1.7) 보다 한참 아래라 여유가 있고,
        # 상한 3 은 폭주를 막는다.
        self.prior_log_sigma_range = (-6.0, 3.0)

    def pair_vec(self, pairs: torch.Tensor) -> torch.Tensor:
        """[N,2] -> [N, emb_dim].  Emb(a)+Emb(b): 순서 불변."""
        assert int(pairs.min()) >= 0 and int(pairs.max()) < self.n_roi, (
            f"ROI 인덱스 범위 밖: {int(pairs.min())}..{int(pairs.max())} (n_roi={self.n_roi})")
        return self.emb(pairs[:, 0]) + self.emb(pairs[:, 1])

    def _mode_idx(self, mode, n: int, device) -> torch.Tensor:
        if torch.is_tensor(mode):
            assert mode.shape == (n,), (mode.shape, n)
            return mode.long().to(device)
        return torch.full((n,), int(mode), dtype=torch.long, device=device)

    def prior_params(self, pairs: torch.Tensor, mode=0, anatomy: torch.Tensor | None = None,
                     local: torch.Tensor | None = None):
        """[N,2] -> (mu [N,D], log_sigma [N,D]).  p(z | pair) = N(mu, diag(exp(2*log_sigma))).

        log_sigma 는 전부 0-init 이므로 학습 전에는 sigma == 1.0 이고 `mu + sigma*eps` 가
        기존 `mu + eps` 와 **bit-exact 동일**하다 (1.0 곱은 float 항등).
        """
        m = self._mode_idx(mode, pairs.shape[0], pairs.device)
        v = self.pair_vec(pairs)
        mu = self.prior_mu(v) + self.mode_prior(m)
        ls = self.prior_log_sigma(v) + self.mode_log_sigma(m)
        if anatomy is not None:
            assert self.prior_use_anatomy, "anatomy prior 통로가 꺼져 있다 (prior_use_anatomy=False)"
            if anatomy.shape[0] == 1:
                anatomy = anatomy.expand(pairs.shape[0], -1)
            assert anatomy.shape == (pairs.shape[0], self.cond_dim), anatomy.shape
            d = self.prior_anatomy(self.anatomy_norm(anatomy))
            mu = mu + d[:, :self.latent_dim]
            ls = ls + d[:, self.latent_dim:]
        else:
            assert not self.prior_use_anatomy, "prior_use_anatomy=True 인데 anatomy 가 없다"
        if self.prior_local is not None:
            assert local is not None, "prior_local_dim > 0 인데 local feature 가 안 넘어왔다"
            assert local.shape == (pairs.shape[0], self.prior_local_dim), (
                local.shape, pairs.shape[0], self.prior_local_dim)
            dl = self.prior_local(torch.cat([local, v], dim=-1))
            if self.prior_local_w is not None:
                from .pair_anchor import upper_index
                idx = upper_index(pairs[:, 0].long(), pairs[:, 1].long(), self.prior_local_n_roi)
                lc = local
                if bool(self.prior_local_roi_mean.abs().sum() > 0):
                    rm = self.prior_local_roi_mean
                    sd_ = self.prior_local_roi_std
                    i_, j_ = pairs[:, 0].long(), pairs[:, 1].long()
                    lc = ((local - torch.cat([rm[i_], rm[j_]], dim=-1))
                          / torch.cat([sd_[i_], sd_[j_]], dim=-1))
                u = self.prior_local_u(lc)                          # [K, rank]  (중심화 입력)
                dl = dl + torch.einsum("kr,krd->kd", u, self.prior_local_w[idx])
            mu = mu + dl[:, :self.latent_dim]
            ls = ls + dl[:, self.latent_dim:]
        else:
            assert local is None, "prior_local_dim = 0 인데 local feature 가 넘어왔다"
        lo, hi = self.prior_log_sigma_range
        return mu, ls.clamp(lo, hi)

    def prior_mean(self, pairs: torch.Tensor, mode=0) -> torch.Tensor:
        """[N,2] -> mu_pair [N, latent_dim]. mode 는 int 또는 [N] tensor."""
        m = self._mode_idx(mode, pairs.shape[0], pairs.device)
        return self.prior_mu(self.pair_vec(pairs)) + self.mode_prior(m)

    def prior_log_std(self, pairs: torch.Tensor, mode=0, anatomy=None,
                      local: torch.Tensor | None = None) -> torch.Tensor:
        """[N,2] -> log_sigma [N, latent_dim]. 0-init 상태에서는 전부 0 (sigma=1).

        **진단용**이다. local 을 안 주면 국소 입력을 0 으로 넣는다 -- 국소 가지가 **없는** 것과
        같지 않다 (학습된 공유 MLP 는 0 입력에도 0 이 아닌 값을 낸다). 실제 추론 경로는
        sample_prior/prior_params 를 쓰고 거기서는 local 이 필수다.
        """
        if local is None and self.prior_local is not None:
            local = torch.zeros(pairs.shape[0], self.prior_local_dim,
                                device=pairs.device, dtype=self.prior_mu.weight.dtype)
        return self.prior_params(pairs, mode, anatomy, local)[1]

    def sample_prior(self, pairs: torch.Tensor, mode=0, anatomy=None,
                     generator: torch.Generator | None = None,
                     eps: torch.Tensor | None = None, local: torch.Tensor | None = None) -> torch.Tensor:
        """z ~ N(mu_pair, diag(sigma^2)).  eps 를 주면 재사용(재현/비교용).

        0-init 에서 `mu + exp(0)*eps == mu + eps` 라 기존 `roi_atm.sample_z` 와 bit-exact 같다.
        """
        mu, ls = self.prior_params(pairs, mode, anatomy, local)
        if eps is None:
            eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        assert eps.shape == mu.shape, (eps.shape, mu.shape)
        return mu + torch.exp(ls) * eps

    def anatomy_term(self, anatomy: torch.Tensor) -> torch.Tensor:
        """cond 에 들어가는 anatomy 항. LayerNorm 으로 pair 항과 크기를 대등하게 맞춘다."""
        return self.anatomy_gain * self.anatomy_norm(anatomy)

    def forward(self, anatomy: torch.Tensor, pairs: torch.Tensor, mode=0,
                local: torch.Tensor | None = None) -> torch.Tensor:
        """anatomy [1,C] 또는 [N,C], pairs [N,2] -> cond [N,C]. mode: 0 full / 1 segment.
        local [N, local_dim] 은 local_dim > 0 일 때 pair 별 국소 anatomy."""
        n = pairs.shape[0]
        if anatomy.shape[0] == 1:
            anatomy = anatomy.expand(n, -1)
        assert anatomy.shape == (n, self.cond_dim), (anatomy.shape, n, self.cond_dim)
        m = self._mode_idx(mode, n, pairs.device)
        c = self.anatomy_term(anatomy) + self.proj(self.pair_vec(pairs)) + self.mode_emb(m)
        if self.local_proj is not None:
            assert local is not None, "local_dim > 0 인데 local feature 가 안 넘어왔다"
            assert local.shape == (n, self.local_dim), (local.shape, n, self.local_dim)
            c = c + self.local_proj(local)
        else:
            assert local is None, "local_dim = 0 인데 local feature 가 넘어왔다"
        return c
