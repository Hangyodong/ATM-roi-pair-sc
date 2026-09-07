"""ROI-pair ATM fine-tuning step (최종 전략 §5-§8, 파이프라인 §11-§24).

한 step = subject 하나.

  [A] anatomy   T1 -> (학습 가능한) UNet -> a [1,512]  **step 당 1회**.
                a 를 leaf 로 분리해 아래 세 pass 가 leaf 에 dL/da 를 모으고, 마지막에
                a.backward(dL/da) 로 UNet backward 를 **1회** 만 한다. 따라서 SC/endpoint/recon/edge
                모든 loss 의 gradient 가 T1 encoder 에 도달하면서도 UNet 은 chunk 마다 다시 돌지 않는다.
  [G] 생성 pass  (endpoint + SC). SC 는 subject-level 이라 chunk 마다 loss 를 걸 수 없다:
                pass 1 (no_grad)  부분 SC 합산 -> L_sc -> dL/dSC
                pass 2 (grad)     chunk 재생성 -> (SC_c * dL/dSC).sum() + L_endpoint_c -> backward
  [R] 재구성 pass  recon(mm) + kl(조건부 prior) + geometry(등간격)
  [E] edge pass    양성/음성 pair BCE

recurrent tracking 없음. loss 별 gradient norm 과 dL/da 를 기록한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import losses as L
from ..models.latent_prior import diag_log_prob as prior_log_prob
from ..models.sc_builder import BundleAccumulator, SCBuilder, streamline_lengths
from ..models.roi_pair_embedding import canonical_pairs
from ..data import local_feats as L_local
from ..data.roi_groups import BLOCKS, N_CTX, TIERS, block_masks
from ..data.balanced_pair_sampler import BalanceConfig, BalancedPairSampler
from ..data.segment_sampler import BalancedSegmentSampler, SegmentBalanceConfig


@dataclass
class LossWeights:
    """L = recon + kl*KL + geom*L_geom + endpoint*L_end + edge*L_edge + corr*L_corr + mag*L_mag + length*L_len"""
    recon: float = 1.0
    kl: float = 0.1
    geom: float = 1.0
    endpoint: float = 1.0
    route: float = 1.0            # ROUTE 전략 §23: 복원 streamline 의 통과 ROI (per-streamline GT)
    route_gen: float = 0.5        # 생성 streamline 의 통과 ROI (pair 별 GT 통과 비율이 target)
    gen_length: float = 1.0       # 생성 streamline 길이를 그 pair 의 GT 평균 길이에 맞춘다.
                                  # 없으면 route loss 가 "많은 ROI 를 지나라"고 요구할 때 모델이
                                  # 길게 헤매는 쪽으로 도망간다 (실측: 111 mm -> 323 mm, GT 103 mm).
    presence: float = 0.5         # §20-21: pass-edge 존재 여부 BCE (기본 SUB-SUB)
    # EDGE_ALIGNED 전략 §13-22: SC edge 단위 segment 분기 + edge count head
    seg_recon: float = 1.0
    seg_kl: float = 0.1
    seg_geom: float = 1.0
    seg_endpoint: float = 0.5
    count: float = 1.0
    scale: float = 1.0            # 총합 로그 차이 (전역 배율). 실측: 배율 하나로 CCC 0.024 -> 0.817
    rmse: float = 0.2             # GT 표준편차로 무차원화한 SC RMSE (절대 오차)
    edge: float = 0.5
    corr: float = 0.5
    resid: float = 0.0        # 실험 A: subject 평균(EMA)을 뺀 **개인차** 상관. 0 = 끔
    # 전략 문서 §4. 전부 cfg.resid_stats (train split 전용 log1p 템플릿/표준편차) 를 요구한다.
    res: float = 0.0          # L_res  SmoothL1(정규화 잔차)
    res_corr: float = 0.0     # L_corr 1 - corr(정규화 잔차). EMA 가 아니라 고정 템플릿 기준
    diff: float = 0.0         # L_diff |(pred_a-pred_b) - (gt_a-gt_b)|_1. 2명/step 필요
    var: float = 0.0          # L_var  |log(예측 쌍차 노름 / GT 쌍차 노름)|. 2명/step 필요
    mag: float = 0.5
    length: float = 0.2
    # 조건부 prior p(z|pair) 적합항 (W3-a). **KL 로 prior 를 학습시키면 안 된다** (C11 실측:
    # KL(q||p) 의 p 최적해는 sigma*=1.074 라 지금의 1.0 이 이미 최적점이고, 그 sigma 로
    # 채점하면 precision 36.61 -> 38.74 로 나빠진다). 그래서 KL 에는 detach 한 prior 를 넣고,
    # prior 는 detach 한 posterior 평균에 대한 로그우도로 따로 맞춘다:
    #     L_prior = -log N(mu_q.detach(); mu_p(pair), diag(sigma_p(pair)^2))
    # 오라클 기대 이득 (w2b_prior_ladder.json, 처음 보는 pair 로 일반화):
    #     precision_ratio 36.61 -> 15.60 (arch_additive). K-혼합은 14.40 로 거의 안 붙어서 쓰지 않는다.
    prior: float = 0.0
    seg_prior: float = 0.0


@dataclass
class TrainConfig:
    sc_mode: str = "pass"                # SC loss/평가 정의. 'pass' = .mat GT 정의, 'endpoint' = sc_end
    n_gen_per_pair: int = 4
    max_pairs_per_step: int | None = None
    n_gt_per_pair: int = 8
    gt_pairs_per_step: int = 64
    neg_pairs_per_step: int = 64
    chunk: int = 4096
    endpoint_tau: float = 5.0
    # 파라미터 그룹별 LR (최종 전략 §8)
    lr_t1: float = 1e-5
    lr_vae_enc: float = 3e-5
    lr_dec: float = 3e-5
    lr_heads: float = 1e-4
    # prior 분산(prior_log_sigma / mode_log_sigma) 전용 LR. None 이면 lr_heads 와 같아
    # **기존 config 는 동작이 바뀌지 않는다** (AdamW 는 파라미터별이라 그룹만 나눈 것은 무영향).
    lr_prior: float | None = None
    lr: float | None = None              # 주면 t1/vae_enc/dec 를 이 값으로 덮어씀 (구 스크립트 호환)
    grad_clip: float = 50.0
    amp_dtype: torch.dtype | None = None
    amp: str = ""                        # ""|"bf16"|"fp16" -- config 에서 문자열로 준다 (yaml 이 dtype 을 못 담는다)
    tf32: bool = True                    # TF32 matmul. 실측: RMSE 9.186 -> 9.188 (차이 0.2%), 1.23배 빠름
    cudnn_benchmark: bool = True         # 입력 형상이 고정이라 안전
    use_checkpoint: bool = True          # UNet stage gradient checkpointing
    # ConvVAE 의 BatchNorm1d 5개를 학습 중에 어떻게 다루나 (W1-d 실측, scripts/38_decoder_capacity.py).
    #   'train'      배치 통계로 forward + running 통계 갱신 (기존 동작).
    #                p4_joint 처럼 recon/segment/생성이 같은 BN 을 지나면 running 통계가 서로 다른
    #                분포의 평균이 되어 train 3.55 mm / eval 8.46 mm 로 갈린다.
    #   'eval'       running 통계로 고정 (갱신 안 함) -> 학습과 추론이 같은 함수가 된다.
    #   'recal_eval' 학습 시작 전 running 통계를 1회 재보정하고 그 뒤 'eval' 과 동일 (run.py 가 처리).
    bn_mode: str = "train"
    active: set = field(default_factory=lambda: {"recon", "endpoint", "edge", "corr", "mag", "length"})
    # 소/중/대 · ctx/sub 균형 (docs/ATM_FINAL_FINETUNING_REPORT.md §14)
    pair_sampling: str = "tier"          # recon/edge 양성 pair 샘플링: 'log' | 'uniform' | 'tier'(강도 구간별 같은 개수)
    sc_groups: str | None = "block"      # SC loss 를 ctx-ctx/ctx-sub/sub-sub 로 나눠 평균 (None = whole-brain 하나)
    resid_stats: str | None = None        # train split 전용 SC 통계 npz (data/sc_template.py)
    count_local_dim: int = 0              # >0 이면 count head 가 pair 별 ROI 국소 anatomy 를 받는다
    cond_local_dim: int = 0               # >0 이면 **디코더 조건(FiLM)** 도 같은 국소 anatomy 를 받는다
    pair_anchor: str | bool | None = None # pair 앵커 재매개화 (models/pair_anchor.py). 경로 또는 True
    anchor_alpha: float = 0.0             # 0 = 전역 박스(기존과 bit-exact), 1 = 완전 앵커
    anchor_alpha_steps: int = 0           # >0 이면 alpha 를 0 -> anchor_alpha_end 로 선형 램프업.
                                          # 재매개화는 출력의 의미를 바꾸므로 한 번에 켜면
                                          # 디코더가 무너진다. 램프업이 그걸 막는다.
    anchor_alpha_end: float = 1.0
    local_source: str = "rigid"           # 그 캐시의 프로토콜 (s1b_feats/{sub}_{source}.npz)
    resid_beta: float = 1.0               # SmoothL1 의 beta (정규화 잔차 단위)
    resid_momentum: float = 0.02          # ResidualCorr 의 subject 평균 EMA 계수
    resid_warmup: int = 200               # EMA 가 템플릿 초기값에서 벗어날 때까지 기울기 차단
                                          # (0.98^200 = 0.018 -- 초기값 편향이 2 % 밑으로 내려간다)
    sc_log_corr: bool = True             # L_corr 를 log1p 도메인 Pearson 으로 (raw 는 큰 edge 몇 개가 지배)
    block_weights: tuple = (1.0, 1.0, 1.0)
    sc_global_weight: float = 0.0        # GESTA 전략 §46: L_corr = λ_type·(block 평균) + λ_global·(whole-brain). 0 = block 만
    route_tau: float = 1.0               # 실측(sub-000001 GT streamline): 통과 ROI 확률 중앙값이
                                         # tau 0.5/1/2/5 에서 0.98/0.83/0.58/0.32. tau 가 크면 완벽한 경로도
                                         # p<0.5 라 target 1 에 닿지 못한다. log 공간이라 작은 tau 여도 gradient 는 산다.
    route_mode: str = "bce"              # 'bce' | 'dice' | 'both'
    route_pos_weight: float = 2.0        # 통과 ROI 는 82개 중 ~5개뿐 -> 양성 가중.
                                         # 5.0 은 과했다: 길이가 3배로 늘고 SC 상관이 0.81 -> 0.73 으로 떨어졌다.
    presence_scale: float = 1.0          # P(edge) = 1 - exp(-SC_pred/scale)
    presence_blocks: tuple = ("sub-sub",)
    weight_mode: str = "count"            # 학습 중 SC 를 만들 때 가닥에 붙이는 가중치
                                         #  'head'  : weight head 가 자유롭게 예측 (기존)
                                         #  'count' : w = N_hat(끝점 pair) / n_gen_per_pair
                                         #            -> 학습에서 만드는 SC 가 "GT 개수대로 만들었을 때"의
                                         #               불편추정이 된다. 100만 가닥을 다 만들지 않고도 스케일이 맞는다.
                                         #  'count_head' : 두 값을 곱한다 (개수가 스케일, head 가 가닥별 변조)
    sc_mag_normalize: str = "sum"        # 'sum' 이면 총합을 맞춘 뒤 비교(패턴만), 'none' 이면 절대 크기
    sc_rmse_normalize: str = "gt_std"
    balance: BalanceConfig = field(default_factory=BalanceConfig)   # §35–38 ROI-pair 균형 recon batch (기본 off)
    seg_edges_per_step: int = 64
    n_seg_per_edge: int = 8
    segment_balance: SegmentBalanceConfig = field(default_factory=SegmentBalanceConfig)
    seed: int = 0


class Trainer:
    def __init__(self, model, assigner, cfg: TrainConfig, weights: LossWeights | None = None):
        self.model, self.cfg = model, cfg
        self.w = weights or LossWeights()
        self.builder = SCBuilder(assigner, mode=cfg.sc_mode)
        self.assigner = assigner
        g = model.param_groups()
        lrs = {"t1_encoder": cfg.lr_t1, "vae_encoder": cfg.lr_vae_enc, "decoder": cfg.lr_dec,
               "heads": cfg.lr_heads,
               "prior_scale": cfg.lr_heads if cfg.lr_prior is None else cfg.lr_prior}
        if cfg.lr is not None:
            lrs.update(t1_encoder=cfg.lr, vae_encoder=cfg.lr, decoder=cfg.lr)
        self.groups = [{"params": ps, "lr": lrs[k], "name": k} for k, ps in g.items() if ps]
        self.opt = torch.optim.AdamW(self.groups, weight_decay=0.0)
        self.device = model.device
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(cfg.seed)
        n_roi = getattr(model, "n_roi", None)
        self.masks = None                    # {block: [R,R] bool}. n_roi 가 66 ctx + 피질하 구조일 때만
        if cfg.sc_groups == "block" and n_roi and n_roi > N_CTX:
            self.masks = {k: torch.as_tensor(v, device=self.device) for k, v in block_masks(n_roi).items()}
        elif cfg.sc_groups not in (None, "block"):
            raise ValueError(cfg.sc_groups)
        self.block_w = dict(zip(BLOCKS, cfg.block_weights))
        # 실험 A -- 잔차 타깃. count head 예측에서 subject 평균(EMA)을 빼고 개인차에만 상관을 건다.
        # EMA 는 train 템플릿에서 출발한다: count head 가 템플릿 인수분해라 초기 예측이 정확히
        # log1p(template) 이고(softplus(log t) = log1p(t)) 시작 시점 잔차가 0 이라 편향이 없다.
        # 전략 문서 §1.2-1.4. template/std/mask 는 train split 에서만 온다 (누수 방지).
        # ROI 국소 anatomy (전략 문서 §3.2). global avg pool 이 지운 개인차를 head 에 되돌린다.
        self._local = {}                     # subject -> [R, D] (subject 당 1회 로드)
        self._roi_labels = None              # stage3 격자의 아틀라스 (살아있는 풀링용)
        self._o3_leaf = self._roi_cache = None
        # 실측(A10, stage3): checkpoint 를 끄면 0.97 -> 0.55 s/step 이고 VRAM 은 9.3GB 로 같다.
        # 원래 메모리를 아끼는 기법인데 여기선 안 아껴서 순수 낭비다.
        model.atm.use_checkpoint = bool(cfg.use_checkpoint)
        if cfg.cond_local_dim:
            assert model.pair_emb.local_dim == cfg.cond_local_dim, (
                'pair_emb 의 local_dim 이 config 와 다르다', model.pair_emb.local_dim, cfg.cond_local_dim)
        if cfg.count_local_dim:
            assert model.count_head is not None and model.count_head.local_dim == cfg.count_local_dim, (
                'count head 의 local_dim 이 config 와 다르다 -- 모델을 count_local_dim 으로 만들어야 한다',
                getattr(model.count_head, 'local_dim', None), cfg.count_local_dim)
        self.rstats = None
        if self.w.res > 0 or self.w.res_corr > 0 or self.w.diff > 0 or self.w.var > 0:
            assert cfg.resid_stats, 'res/res_corr/diff 손실은 cfg.resid_stats 가 필요하다'
            z = np.load(cfg.resid_stats, allow_pickle=False)
            tt = lambda k, d=torch.float32: torch.as_tensor(z[k], dtype=d, device=self.device)
            self.rstats = {'tpl': tt('template'), 'std': tt('std'),
                           'mask': tt('mask', torch.bool)}
            e = int(n_roi * (n_roi - 1) // 2)
            for k, v in self.rstats.items():
                assert v.shape == (e,), (k, v.shape, e)
            assert float(self.rstats['std'].min()) > 0, 'std 에 0 이 있다'
            assert bool(self.rstats['mask'].any()), 'edge mask 가 비어 있다'
            print(f"[trainer] 잔차 통계 {cfg.resid_stats} "
                  f"(edge {int(self.rstats['mask'].sum())}/{e})", flush=True)
        assert not ((self.w.diff > 0 or self.w.var > 0) and self.rstats is None)
        self.resid_corr = None
        if self.w.resid > 0:
            assert getattr(model, 'template', None) is not None, (
                'resid 손실은 템플릿 인수분해 count head 가 필요하다 (model.template 이 None)')
            key = 'sc_end' if cfg.sc_mode == 'endpoint' else 'sc_pass'
            tpl = torch.as_tensor(np.asarray(model.template[key], np.float32), device=self.device)
            tu = torch.log1p(L.upper(tpl).clamp(min=1e-2))   # head 의 template_floor 와 같은 바닥값
            assert torch.isfinite(tu).all() and float(tu.var()) > 0, '템플릿이 상수이거나 NaN'
            self.resid_corr = L.ResidualCorr(tu, tu, momentum=cfg.resid_momentum,
                                            warmup=cfg.resid_warmup, log=False)
        # 절대 스케일 항은 "생성한 SC 총합 vs GT 총합" 을 본다. pair 를 일부만 생성하면 분자만 작아져
        # 배율을 과대 학습한다 -> 전체 pair 를 생성할 때만 허용한다.
        absolute = (cfg.sc_mag_normalize == "none" and "mag" in cfg.active) or bool(cfg.active & {"scale", "rmse"})
        assert not (absolute and cfg.max_pairs_per_step is not None), (
            "절대 스케일 항(sc_mag_normalize='none' / scale / rmse)은 max_pairs_per_step=None 에서만 쓴다 "
            f"(지금 {cfg.max_pairs_per_step})")
        if isinstance(cfg.balance, dict):
            cfg.balance = BalanceConfig(**cfg.balance)
        if isinstance(cfg.segment_balance, dict):
            cfg.segment_balance = SegmentBalanceConfig(**cfg.segment_balance)
        self.samplers = {}                   # subject -> BalancedPairSampler (balance.enabled 일 때)
        self.seg_samplers = {}               # subject -> BalancedSegmentSampler ('segment' phase)
        self.presence_mask = None
        if self.masks is not None and cfg.presence_blocks:
            mk = [self.masks[b] for b in cfg.presence_blocks]
            self.presence_mask = torch.stack(mk).any(0)
        assert cfg.bn_mode in ("train", "eval", "recal_eval"), cfg.bn_mode
        self.ae_bns = [b for b in model.atm.net.ae.modules() if isinstance(b, torch.nn.BatchNorm1d)]
        assert self.ae_bns, "ConvVAE 에서 BatchNorm1d 를 하나도 못 찾았다 (구조가 바뀌었나?)"

    def cond_local(self, sub: str, pairs: torch.Tensor) -> torch.Tensor | None:
        """디코더 조건용 [K, cond_local_dim]. cond_local_dim = 0 이면 None."""
        if not self.cfg.cond_local_dim:
            return None
        live = self._roi_from_live()
        if live is not None:
            return L_local.pair_local(live, pairs)
        from ..data.local_feats import load_roi_feats
        if sub not in self._local:
            self._local[sub] = load_roi_feats(sub, self.cfg.local_source, self.device,
                                              n_roi=self.model.n_roi)
        return L_local.pair_local(self._local[sub], pairs)

    def _roi_from_live(self) -> torch.Tensor | None:
        """UNet 을 학습 중이면 **이번 step 의** stage3 에서 ROI 풀링한다 (gradient 포함).
        디스크 캐시는 동결 인코더 전용이다 -- 학습 중에 쓰면 낡은 값으로 조용히 틀린다."""
        o3 = getattr(self, "_o3_leaf", None)
        if o3 is None:
            return None
        if self._roi_cache is not None:
            return self._roi_cache
        if self._roi_labels is None:
            from ..models.roi_pool import atlas_on_feature_grid
            self._roi_labels = atlas_on_feature_grid(feat_shape=tuple(o3.shape[2:]))
        from ..models.roi_pool import roi_pool
        self._roi_cache = roi_pool(o3, self._roi_labels, self.model.n_roi)
        return self._roi_cache

    def local_feat(self, sub: str) -> torch.Tensor | None:
        """subject 의 ROI 국소 anatomy [R, D]. count_local_dim = 0 이면 None."""
        if not self.cfg.count_local_dim:
            return None
        live = self._roi_from_live()
        if live is not None:
            return live
        if sub not in self._local:
            from ..data.local_feats import load_roi_feats
            self._local[sub] = load_roi_feats(sub, self.cfg.local_source, self.device,
                                              n_roi=self.model.n_roi)
        return self._local[sub]

    def _set_ae_bn_eval(self) -> None:
        """ConvVAE 의 BatchNorm 만 eval 로 (running 통계 사용 + 갱신 안 함)."""
        for b in self.ae_bns:
            b.eval()

    def gt_matrices(self, subject):
        w, l = (subject.sc_end, subject.len_end) if self.cfg.sc_mode == "endpoint" else (subject.sc_pass, subject.len_pass)
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)
        return t(w), t(l)

    # ------------------------------------------------------------------------------
    def step(self, subject, anat_input: torch.Tensor, partner=None, step: int | None = None) -> dict:
        """partner = (ROIPairSubject, anat_input) -- L_diff 용 두 번째 subject (§5).
        step 은 pair 앵커 alpha 램프업에만 쓴다 (학습 스케줄)."""
        m, cfg, w = self.model, self.cfg, self.w
        m.train()
        if cfg.anchor_alpha_steps and m.anchor is not None:
            assert step is not None, 'alpha 램프업인데 step 이 안 넘어왔다'
            m.anchor.set_alpha(min(1.0, step / cfg.anchor_alpha_steps) * cfg.anchor_alpha_end)
        if cfg.bn_mode != "train":
            # ConvVAE BN 만 eval 로 고정. 이렇게 해야 학습 중의 복원과 추론의 복원이 같은
            # 함수다 (안 하면 로그의 3.55 mm 가 추론에서 8.46 mm 로 나온다).
            self._set_ae_bn_eval()
        self.opt.zero_grad(set_to_none=True)
        out, t0 = {}, time.time()
        active = {("recon" if k == "atm" else k) for k in cfg.active}

        # [A] anatomy: 1회 forward, leaf 분리 ------------------------------------------
        t1_trainable = m.unet_level != "none" and anat_input.ndim != 2
        a_full = m.anatomy_forward(anat_input)
        a_leaf = a_full.detach().requires_grad_(t1_trainable)
        assert torch.isfinite(a_leaf).all(), "anatomy feature NaN"
        out["anat_norm"] = float(a_leaf.norm())
        # 살아있는 stage3 도 leaf 로 분리한다. 안 하면 count 블록의 backward 가 UNet 그래프를
        # 직접 타고, 마지막 a_full.backward 에서 "backward a second time" 으로 죽는다.
        o3_full = getattr(m, "_live_stage3", None)
        self._o3_leaf = None if o3_full is None else o3_full.detach().requires_grad_(t1_trainable)
        self._roi_cache = None                       # step 안에서 ROI 풀링 1회만
        dLdo3 = None if self._o3_leaf is None else torch.zeros_like(self._o3_leaf)
        dLda = torch.zeros_like(a_leaf)

        def take_dLda(tag):
            if a_leaf.grad is not None:
                out[f"dLda_{tag}"] = float(a_leaf.grad.norm())
                dLda.add_(a_leaf.grad); a_leaf.grad = None
            if self._o3_leaf is not None and self._o3_leaf.grad is not None:
                out[f"dLdo3_{tag}"] = float(self._o3_leaf.grad.norm())
                dLdo3.add_(self._o3_leaf.grad); self._o3_leaf.grad = None

        # [G] ------------------------------------------------------------------------
        gen_needed = bool(active & {"endpoint", "corr", "mag", "length", "presence"}) or \
            ("route" in active and w.route_gen > 0)
        sc_active = bool(active & {"corr", "mag", "length", "presence", "scale", "rmse"})
        pid_all = np.asarray(subject.pair_ids, np.int64)
        sel = np.arange(len(pid_all))
        if cfg.max_pairs_per_step is not None and len(sel) > cfg.max_pairs_per_step:
            sel = subject.sample_pairs(cfg.max_pairs_per_step, self.rng, mode=cfg.pair_sampling)
        pos = pid_all[sel]
        gen_route = "route" in active and w.route_gen > 0 and getattr(subject, "has_visitation", False)
        marg = (torch.as_tensor(np.asarray(subject.pair_marginal, np.float32)[sel], device=self.device)
                .repeat_interleave(cfg.n_gen_per_pair, 0) if gen_route else None)
        pairs = torch.as_tensor(pos, device=self.device).repeat_interleave(cfg.n_gen_per_pair, 0)
        N = pairs.shape[0]
        eps = m.sample_eps(N, self.gen)
        gt_w, gt_l = self.gt_matrices(subject)

        def gen_chunk(i, grad):
            with (torch.enable_grad() if grad else torch.no_grad()):
                pc = pairs[i:i + cfg.chunk]
                c = m.condition(a_leaf, pc, local=self.cond_local(subject.sub, pc))
                # 추론(`roi_atm.sample_z`)과 **같은 분포**에서 뽑아야 생성 제약이 실제로 쓰이는
                # 영역을 학습한다. log_sigma 0-init 이면 exp(0)=1 이라 기존 `prior_mean + eps` 와
                # bit-exact 같다. sigma 는 detach 한다 -- 안 그러면 endpoint/SC 손실이 sigma 를
                # 0 으로 붕괴시켜(결정적 z) 제약을 만족시키는 도피로가 생긴다.
                mu_pc, ls_pc = m.prior_params(pc, anatomy=a_leaf)
                zc = mu_pc + torch.exp(ls_pc.detach()) * eps[i:i + cfg.chunk]
                if cfg.amp_dtype is None:
                    S = m.decode(zc, c, pc)
                else:
                    with torch.autocast(self.device.type, dtype=cfg.amp_dtype):
                        S = m.decode(zc, c, pc)
                    S = S.float()
                w = None if cfg.weight_mode == "count" else m.weights(c, zc)
                if cfg.weight_mode != "head":
                    assert m.count_head_end is not None, "weight_mode 에 count 를 쓰려면 count head 가 필요"
                    # 균등하게 n_gen 개만 만들었지만, GT 는 이 pair 에 N_hat 개가 있다.
                    # N_hat/n_gen 을 곱하면 전체를 만든 것과 같은 기대값이 된다 (중요도 가중).
                    nh = m.edge_log_counts_end(a_leaf, pc).exp() / max(cfg.n_gen_per_pair, 1)
                    w = nh if cfg.weight_mode == "count" else w * nh
                return S, w

        G_sc = G_num = None
        if sc_active and gen_needed:
            acc = BundleAccumulator(m.n_roi, device=self.device)
            for i in range(0, N, cfg.chunk):
                S, wk = gen_chunk(i, grad=False)
                acc.add(*self.builder(S, streamline_lengths(S), wk))

            def sc_loss_fn(sc, num):
                tot = 0.0
                if "corr" in active:
                    if self.masks is not None:
                        v, rs = L.sc_corr_group_loss(sc, gt_w, self.masks, self.block_w, log=cfg.sc_log_corr)
                        out.update({f"corr_r_{k}": r for k, r in rs.items()})
                        if cfg.sc_global_weight > 0:
                            vg = L.sc_corr_loss(sc, gt_w, log=cfg.sc_log_corr); out["corr_r_all"] = 1.0 - float(vg)
                            v = v + cfg.sc_global_weight * vg
                    else:
                        v = L.sc_corr_loss(sc, gt_w, log=cfg.sc_log_corr)
                    out["L_corr"] = float(v); tot = tot + w.corr * v
                if "mag" in active:
                    v = L.sc_magnitude_loss(sc, gt_w, normalize=cfg.sc_mag_normalize, masks=self.masks)
                    out["L_mag"] = float(v); tot = tot + w.mag * v
                if "scale" in active and w.scale > 0:          # 전역 배율 하나 (mag 와 독립)
                    v = L.sc_scale_loss(sc, gt_w, self.masks)
                    out["L_scale"] = float(v); tot = tot + w.scale * v
                if "rmse" in active and w.rmse > 0:
                    v = L.sc_rmse_loss(sc, gt_w, self.masks, cfg.sc_rmse_normalize)
                    out["L_rmse"] = float(v); tot = tot + w.rmse * v
                if active & {"mag", "scale", "rmse"}:
                    out["sc_sum_ratio"] = float(sc.sum() / gt_w.sum().clamp(min=1))
                if "length" in active:
                    v = L.tract_length_loss(num, sc, gt_l, gt_w, masks=self.masks); out["L_length"] = float(v); tot = tot + w.length * v
                if "presence" in active:
                    v = L.pass_presence_loss(sc, gt_w, self.presence_mask, cfg.presence_scale)
                    out["L_presence"] = float(v); tot = tot + w.presence * v
                return tot
            out["L_sc_total"], G_sc, G_num = acc.loss_grads(sc_loss_fn)
            out.update({f"sc_{k}": v for k, v in L.sc_metrics(acc.sc, gt_w).items()})
            for b, mb in (self.masks or {}).items():
                mm_b = L.sc_metrics(acc.sc, gt_w, mask=mb)
                out[f"sc_rlog_{b}"] = mm_b["r_log"]; out[f"sc_ccc_{b}"] = mm_b["ccc"]
            out["sc_pred_sum"] = float(acc.sc.sum())

        l_end_tot, hits, seen, w_vals, l_route_gen, l_genlen, gen_len_mm = 0.0, 0.0, 0, [], 0.0, 0.0, 0.0
        gt_len_pair = (torch.as_tensor(np.asarray(subject.len_end, np.float32), device=self.device)[
            torch.as_tensor(pos[:, 0], device=self.device), torch.as_tensor(pos[:, 1], device=self.device)]
            .repeat_interleave(cfg.n_gen_per_pair, 0) if gen_needed else None)
        for i in (range(0, N, cfg.chunk) if gen_needed else ()):
            S, wk = gen_chunk(i, grad=True)
            w_vals.append(wk.detach())
            frac = S.shape[0] / N
            total_c = 0.0
            if "endpoint" in active:
                pc = pairs[i:i + cfg.chunk]
                lqs, lqe = self.assigner.endpoint_log_probs(S, tau=cfg.endpoint_tau)
                v = L.endpoint_loss(lqs, lqe, pc[:, 0], pc[:, 1], log_input=True)
                l_end_tot += float(v) * frac
                total_c = total_c + w.endpoint * v * frac
                hits += L.endpoint_accuracy(lqs.exp(), lqe.exp(), pc[:, 0], pc[:, 1])["endpoint_pair_acc"] * S.shape[0]
                seen += S.shape[0]
            if w.gen_length > 0 and gt_len_pair is not None:
                Lp = streamline_lengths(S)
                tgt = gt_len_pair[i:i + cfg.chunk]
                m_ok = tgt > 0
                if bool(m_ok.any()):
                    v = (torch.log1p(Lp[m_ok]) - torch.log1p(tgt[m_ok])).abs().mean()
                    l_genlen += float(v) * frac
                    total_c = total_c + w.gen_length * v * frac
                gen_len_mm += float(Lp.mean()) * frac
            if gen_route:
                lu = self.assigner.visit_log_probs(S, tau=cfg.route_tau)
                v = L.route_loss(lu, marg[i:i + cfg.chunk], cfg.route_mode, cfg.route_pos_weight)
                l_route_gen += float(v) * frac
                total_c = total_c + w.route_gen * v * frac
            if sc_active:
                sc_c, num_c = self.builder(S, streamline_lengths(S), wk)
                total_c = total_c + (sc_c * G_sc).sum() + (num_c * G_num).sum()
            if torch.is_tensor(total_c):
                total_c.backward()
        if "endpoint" in active:
            out["L_endpoint"] = l_end_tot; out["endpoint_pair_acc"] = hits / max(seen, 1)
        if gen_route:
            out["L_route_gen"] = l_route_gen
        if gen_needed and w.gen_length > 0:
            out["L_gen_length"] = l_genlen; out["gen_length_mm"] = gen_len_mm
        if sc_active and "presence" in active:
            out.update({f"gen_{k}": v for k, v in L.presence_metrics(acc.sc, gt_w, self.presence_mask).items()})
        if w_vals:
            out["w_mean"] = float(torch.cat(w_vals).mean())
        if gen_needed:
            out["gnorm_after_G"] = self._grad_norm(); take_dLda("G")

        # [R] ------------------------------------------------------------------------
        rec_route = "route" in active and w.route > 0 and getattr(subject, "has_visitation", False)
        if "recon" in active:
            if cfg.balance.enabled:
                bs = self.samplers.get(subject.sub)
                if bs is None:
                    bs = self.samplers[subject.sub] = BalancedPairSampler(subject, cfg.balance)
                S_gt, P_gt, V_gt, syn, binfo = bs.sample_batch(self.rng, cfg.gt_pairs_per_step,
                                                               cfg.n_gt_per_pair, want_visit=rec_route)
                out.update(binfo)
                S_gt, P_gt = S_gt.to(self.device), P_gt.to(self.device)
                V_gt = V_gt.to(self.device) if V_gt is not None else None
                rw = torch.as_tensor(np.where(syn, cfg.balance.lambda_syn, 1.0), dtype=torch.float32, device=self.device)
            else:
                ks = subject.sample_pairs(cfg.gt_pairs_per_step, self.rng, mode=cfg.pair_sampling)
                tiers = np.asarray(subject.pair_tier)[ks]
                out.update({f"recon_tier_{n}": float((tiers == i).mean()) for i, n in enumerate(TIERS)})
                Ss, Ps, Vs = [], [], []
                for k in ks:
                    S_gt, _ = subject.get_pair(int(k))
                    take = min(cfg.n_gt_per_pair, S_gt.shape[0])
                    idx = torch.as_tensor(self.rng.choice(S_gt.shape[0], take, replace=False))
                    Ss.append(S_gt[idx]); Ps.append(torch.as_tensor(subject.pair_ids[k]).repeat(take, 1))
                    if rec_route:
                        Vs.append(subject.visitation(int(k))[idx])
                S_gt = torch.cat(Ss).to(self.device).float(); P_gt = torch.cat(Ps).to(self.device)
                V_gt = torch.cat(Vs).to(self.device) if Vs else None
                rw = None
            if m.anchor is not None and float(m.anchor.alpha) > 0:
                # 앵커가 방향을 고정하므로 GT 도 같은 방향으로 맞춘다 (PairAnchor.orient 주석 참조)
                S_gt = m.anchor.orient(S_gt, canonical_pairs(P_gt))
            c = m.condition(a_leaf, P_gt, local=self.cond_local(subject.sub, P_gt))
            mu, logvar = m.encode_streamlines(S_gt, c)
            rec = m.decode(m.reparameterize(mu, logvar), c, P_gt)
            l_rec = L.stream_recon_loss(rec, S_gt, weights=rw)
            mu_p, ls_p = m.prior_params(P_gt, anatomy=a_leaf)
            # KL 은 prior 를 **고정 목표**로만 쓴다 (detach). C11 참조.
            l_kl = L.kl_loss(mu, logvar, mu_p.detach(), 2.0 * ls_p.detach())
            l_geom = L.adjacency_loss(rec)
            tot_r = w.recon * l_rec + w.kl * l_kl + w.geom * l_geom
            if w.prior > 0:
                l_prior = -prior_log_prob(mu.detach(), mu_p, ls_p).mean()
                tot_r = tot_r + w.prior * l_prior
                out["L_prior"] = float(l_prior)
                out["prior_sigma_mean"] = float(ls_p.detach().exp().mean())
                # posterior 평균이 prior 평균에서 얼마나 떨어져 있나 (잠재 단위). w2a_latent_drift
                # 의 5.27 -> 3.51 과 같은 양이다.
                out["prior_offset"] = float((mu.detach() - mu_p.detach()).norm(dim=1).mean())
            if rec_route and V_gt is not None:
                keep = (V_gt.sum(-1) > 0)            # synthetic streamline 은 GT 통과 정보가 없다 -> 제외
                if bool(keep.any()):
                    lu = self.assigner.visit_log_probs(rec[keep], tau=cfg.route_tau)
                    l_route = L.route_loss(lu, V_gt[keep], cfg.route_mode, cfg.route_pos_weight)
                    tot_r = tot_r + w.route * l_route
                    out["L_route"] = float(l_route)
                    out.update(L.route_metrics(lu.detach(), V_gt[keep]))
            tot_r.backward()
            out.update(L_recon=float(l_rec), L_kl=float(l_kl), L_geom=float(l_geom),
                       # **train 모드(배치 통계) 값**이다 (bn_mode='train' 일 때). 배포되는 것은
                       # eval 모드이고 그 값은 run.recon_rmse_metrics 의 recon_rmse_eval_mm 이다.
                       # 이름에 모드를 안 박아 둔 것이 3.55 vs 8.46 착시의 출발점이었다.
                       recon_rmse_train_mm=float(((rec - S_gt) ** 2).sum(-1).mean().sqrt()))
            out["gnorm_after_R"] = self._grad_norm(); take_dLda("R")

        # [E] ------------------------------------------------------------------------
        if "edge" in active and m.edge_head is not None:
            kp = subject.sample_pairs(cfg.neg_pairs_per_step, self.rng, mode=cfg.pair_sampling)
            pp = np.asarray(subject.pair_ids)[kp]
            nn_ = subject.negative_pairs(cfg.neg_pairs_per_step, self.rng)
            P = torch.as_tensor(np.concatenate([pp, nn_]), device=self.device)
            y = torch.cat([torch.ones(len(pp)), torch.zeros(len(nn_))]).to(self.device)
            logits = m.edge_logits(a_leaf, P)
            l_edge = L.edge_loss(logits, y)
            (w.edge * l_edge).backward()
            out["L_edge"] = float(l_edge); out.update(L.edge_metrics(logits, y))
            out["gnorm_after_E"] = self._grad_norm(); take_dLda("E")

        # [S] SC edge-aligned segment 분기 -------------------------------------------------
        if "segment" in active and getattr(subject, "has_edge_segments", False):
            ss = self.seg_samplers.get(subject.sub)
            if ss is None:
                ss = self.seg_samplers[subject.sub] = BalancedSegmentSampler(subject, cfg.segment_balance)
            S_sg, P_sg, L_sg, sinfo = ss.sample_batch(self.rng, cfg.seg_edges_per_step, cfg.n_seg_per_edge)
            out.update(sinfo)
            S_sg, P_sg = S_sg.to(self.device), P_sg.to(self.device)
            if m.anchor is not None and float(m.anchor.alpha) > 0:
                S_sg = m.anchor.orient(S_sg, canonical_pairs(P_sg))
            c = m.condition(a_leaf, P_sg, mode=1,                     # mode 1 = segment
                            local=self.cond_local(subject.sub, P_sg))
            mu, logvar = m.encode_streamlines(S_sg, c)
            rec = m.decode(m.reparameterize(mu, logvar), c, P_sg)
            l_sr = L.stream_recon_loss(rec, S_sg)
            mu_ps, ls_ps = m.prior_params(P_sg, mode=1, anatomy=a_leaf)
            l_sk = L.kl_loss(mu, logvar, mu_ps.detach(), 2.0 * ls_ps.detach())
            l_sg = L.adjacency_loss(rec)
            tot_s = w.seg_recon * l_sr + w.seg_kl * l_sk + w.seg_geom * l_sg
            out.update(L_seg_recon=float(l_sr), L_seg_kl=float(l_sk), L_seg_geom=float(l_sg))
            if w.seg_prior > 0:
                l_sp = -prior_log_prob(mu.detach(), mu_ps, ls_ps).mean()
                tot_s = tot_s + w.seg_prior * l_sp
                out["L_seg_prior"] = float(l_sp)
            if w.seg_endpoint > 0:
                lqs, lqe = self.assigner.endpoint_log_probs(rec, tau=cfg.endpoint_tau)
                l_se = L.endpoint_loss(lqs, lqe, P_sg[:, 0], P_sg[:, 1], log_input=True)
                tot_s = tot_s + w.seg_endpoint * l_se
                out["L_seg_endpoint"] = float(l_se)
                out["seg_endpoint_pair_acc"] = L.endpoint_accuracy(lqs.exp(), lqe.exp(), P_sg[:, 0], P_sg[:, 1])["endpoint_pair_acc"]
            tot_s.backward()
            out["gnorm_after_S"] = self._grad_norm(); take_dLda("S")

        # [C] edge count head: SC edge 값을 직접 예측 ---------------------------------------
        if "count" in active and m.count_head is not None:
            R = m.n_roi
            iu = torch.triu_indices(R, R, 1, device=self.device)
            P_all = torch.stack([iu[0], iu[1]], 1)
            fr = self.local_feat(subject.sub)
            loc = None if fr is None else L_local.pair_local(fr, P_all)
            logc = m.edge_log_counts(a_leaf, P_all, loc)
            gt_c = gt_w[iu[0], iu[1]]
            mk = ({b: v[iu[0], iu[1]] for b, v in self.masks.items()} if self.masks is not None else None)
            l_cnt = L.edge_count_loss(logc, gt_c, mk)
            tot_c = w.count * l_cnt
            out["L_count"] = float(l_cnt)
            out.update({f"count_{k}": v for k, v in L.edge_count_metrics(logc.detach().exp(), gt_c).items()})
            if self.rstats is not None:
                # ── 전략 문서 §1: 절대 SC 가 아니라 train 템플릿을 뺀 **정규화 잔차**를 맞춘다.
                # softplus(logc) 가 head 의 log1p 예측이다 (softplus(log t) = log1p(t)).
                tpl, sd, msk = self.rstats["tpl"], self.rstats["std"], self.rstats["mask"]
                d_pred = (torch.nn.functional.softplus(logc) - tpl) / sd
                d_gt = (torch.log1p(gt_c) - tpl) / sd
                out["resid_std_pred"] = float(d_pred[msk].std())
                out["resid_std_gt"] = float(d_gt[msk].std())
                out["resid_var_ratio"] = out["resid_std_pred"] / max(out["resid_std_gt"], 1e-8)
                out["resid_r_step"] = float(L.pearson(d_pred[msk].detach(), d_gt[msk]))
                if w.res > 0:
                    l_rs = L.residual_smooth_l1(d_pred, d_gt, msk, beta=cfg.resid_beta)
                    tot_c = tot_c + w.res * l_rs
                    out["L_res"] = float(l_rs)
                if w.res_corr > 0:
                    l_rc = L.residual_corr_loss(d_pred, d_gt, msk)
                    tot_c = tot_c + w.res_corr * l_rc
                    out["L_res_corr"] = float(l_rc)
                if w.diff > 0 or w.var > 0:
                    # 같은 출력을 내면 예측 차이가 0 이라 손실을 피할 수 없다 (§4.3).
                    assert partner is not None, "diff 손실인데 partner subject 가 안 넘어왔다"
                    ps, pa = partner
                    assert ps.sub != subject.sub, "partner 가 같은 subject 다"
                    a2 = m.anatomy_forward(pa).detach()      # UNet 은 이 경로로 학습하지 않는다
                    fr2 = self.local_feat(ps.sub)
                    loc2 = None if fr2 is None else L_local.pair_local(fr2, P_all)
                    logc2 = m.edge_log_counts(a2, P_all, loc2)
                    gt2 = torch.as_tensor(np.asarray(ps.sc_mat, np.float32),
                                          device=self.device)[iu[0], iu[1]]
                    d_pred2 = (torch.nn.functional.softplus(logc2) - tpl) / sd
                    d_gt2 = (torch.log1p(gt2) - tpl) / sd
                    if w.diff > 0:
                        l_df = L.subject_diff_loss(d_pred, d_pred2, d_gt, d_gt2, msk)
                        tot_c = tot_c + w.diff * l_df
                        out["L_diff"] = float(l_df)
                    if w.var > 0:
                        # 정보를 늘리는 항이 아니라 있는 정보를 **출력 진폭으로 내보내는** 항이다.
                        l_vr = L.subject_var_loss(d_pred, d_pred2, d_gt, d_gt2, msk)
                        tot_c = tot_c + w.var * l_vr
                        out["L_var"] = float(l_vr)
                    out["diff_gt_norm"] = float((d_gt - d_gt2)[msk].norm())
                    out["diff_pred_norm"] = float((d_pred - d_pred2)[msk].detach().norm())
                    out["diff_ratio"] = out["diff_pred_norm"] / max(out["diff_gt_norm"], 1e-8)
                    out["partner"] = ps.sub
            if self.resid_corr is not None:
                # softplus(logc) 가 head 의 log1p 예측이다 (edge_count_loss 와 같은 도메인).
                # 여기서만 subject 평균을 빼므로 기울기가 "템플릿을 더 잘 맞춰라" 로 새지 않는다.
                l_res = self.resid_corr.on_upper(torch.nn.functional.softplus(logc), torch.log1p(gt_c))
                tot_c = tot_c + w.resid * l_res
                out["L_resid"] = float(l_res)
            if m.count_head_end is not None:            # 추론에서 pair 별 생성 개수를 정하는 값
                gt_e = torch.as_tensor(np.asarray(subject.sc_end, np.float32), device=self.device)[iu[0], iu[1]]
                lce = m.edge_log_counts_end(a_leaf, P_all)
                l_ce = L.edge_count_loss(lce, gt_e, mk)
                tot_c = tot_c + w.count * l_ce
                out["L_count_end"] = float(l_ce)
                out.update({f"cend_{k}": v for k, v in L.edge_count_metrics(lce.detach().exp(), gt_e).items()})
            tot_c.backward()
            out["gnorm_after_C"] = self._grad_norm(); take_dLda("C")

        # [A'] UNet backward 1회 ---------------------------------------------------------
        out["dLda_total"] = float(dLda.norm())
        if t1_trainable:
            # a512 경로와 stage3 국소 경로의 gradient 를 **한 번에** UNet 으로 흘린다.
            ts, gs = [], []
            if float(dLda.abs().max()) > 0:
                ts.append(a_full); gs.append(dLda)
            if dLdo3 is not None and float(dLdo3.abs().max()) > 0:
                out["dLdo3_total"] = float(dLdo3.norm())
                ts.append(o3_full); gs.append(dLdo3)
            if ts:
                torch.autograd.backward(ts, gs)
        self._o3_leaf = self._roi_cache = None
        out.update({f"gnorm_{g['name']}": self._group_norm(g["params"]) for g in self.groups})

        gn = torch.nn.utils.clip_grad_norm_(m.trainable_parameters(), cfg.grad_clip)
        out["grad_norm_total"] = float(gn)
        assert np.isfinite(out["grad_norm_total"]), "gradient 가 NaN/Inf"
        self.opt.step()
        out["n_generated"] = N if gen_needed else 0
        out["step_sec"] = time.time() - t0
        if self.device.type == "cuda":
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return out

    def _grad_norm(self) -> float:
        return self._group_norm(self.model.trainable_parameters())

    @staticmethod
    def _group_norm(params) -> float:
        s = 0.0
        for p in params:
            if p.grad is not None:
                s += float(p.grad.norm()) ** 2
        return s ** 0.5
