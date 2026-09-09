"""ROI-pair conditioned ATM (pipeline §9, §16, §10 를 하나로 조립).

    T1 ──UNet(동결)──▶ a [1,512]                       subject 당 1회
    (ROI_a, ROI_b) ──Emb──▶ pair_vec [N,64] ──Proj──▶ + a  = cond [N,512]
    z [N,64] + cond ──ConvVAE.decode──▶ [N,3,128] ──역정규화──▶ streamline [N,128,3] mm
    cond + z ──WeightHead──▶ w [N]
    a + pair_vec ──EdgeHead──▶ P(edge) [N]

pretrained 초기화: 30 bundle 모델 중 하나(init_bundle)의 ConvVAE 를 그대로 쓴다.
서로 독립으로 학습된 30개 decoder 를 평균하는 것은 의미가 없으므로 하나를 고른다.

좌표 정규화: pretrained 상수는 bundle 별 bounding box 라 whole-brain 을 못 덮는다
(AF_L 은 x <= -3.7). MNI152NLin6 brain mask 의 bounding box(+margin) 를 쓴다.
따라서 **시작 시점의 출력 geometry 는 pretrained AF_L 을 whole-brain 박스로 늘린 것**이고
L_ATM 재구성이 이를 다시 맞춘다. 이 사실을 Methods 에 적어야 한다.

latent prior: 학습에서 KL 로 N(0,I) 에 맞추므로 inference 도 N(0,I). upstream 의
per-bundle KDE 는 ROI-pair 모델에는 의미가 없다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..spaces import apply_affine
from .atm_adapter import ANATOMICAL_DIM, LATENT_DIM, N_POINTS, ATMBundle, BundleNorm
from .edge_count_head import EdgeCountHead
from .edge_head import EdgeHead
from .roi_pair_embedding import ROIPairEmbedding, canonical_pairs
from .streamline_refiner import StreamlineRefiner
from .streamline_weight_head import StreamlineWeightHead

_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_MASK = _ROOT / "templates" / "tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"
GROUP_TEMPLATE = _ROOT / "outputs" / "inference" / "template.npz"


def load_group_template(src=True) -> dict:
    """train 평균 SC 템플릿 (scripts/34_build_inference_prior.py 산출물) -> dict.

    keys: sc_pass (count head), sc_end (count head end), edge_prob (edge head).
    test 를 보지 않고 train 으로만 만든 값이어야 한다 -- 34번 스크립트가 그것을 보장한다.
    """
    if isinstance(src, dict):
        z = src
    else:
        p = GROUP_TEMPLATE if src is True else Path(src)
        assert p.exists(), f"그룹 템플릿 없음: {p} -> scripts/34_build_inference_prior.py 먼저"
        z = dict(np.load(p, allow_pickle=False))
    for k in ("sc_pass", "sc_end", "edge_prob"):
        assert k in z, (f"템플릿에 '{k}' 가 없다 (있는 키: {sorted(z)}). "
                        "scripts/34_build_inference_prior.py 를 다시 실행해라")
    out = {k: np.asarray(z[k], np.float64) for k in ("sc_pass", "sc_end", "edge_prob")}
    for k, v in out.items():
        assert v.ndim == 2 and v.shape[0] == v.shape[1], (k, v.shape)
        assert np.isfinite(v).all() and (v >= 0).all() and v.sum() > 0, f"{k} 템플릿이 비었거나 NaN"
    assert out["edge_prob"].max() <= 1.0, "edge_prob 가 확률이 아니다"
    return out


def brain_box_mm(margin: float = 5.0):
    """MNI152NLin6 brain mask 의 bounding box (mm). whole-brain 좌표 정규화 상수."""
    import nibabel as nib
    img = nib.load(TEMPLATE_MASK)
    ijk = np.argwhere(np.asanyarray(img.dataobj) > 0)
    mm = apply_affine(img.affine, ijk.astype(np.float64))
    lo, hi = mm.min(0) - margin, mm.max(0) + margin
    assert (hi - lo > 100).all(), (lo, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def from_checkpoint(path, device="cuda", n_roi: int = 82, **kw):
    """checkpoint 메타(unet_level / in_channels / template)로 모델을 만들고 가중치를 싣는다.

    이걸 쓰지 않고 ROIPairATM(...) 을 직접 만들면 **조용히 틀린 모델**이 된다:
    2채널로 학습한 checkpoint 를 1채널 모델에 넣으면 shape 오류로 멈추지만, 반대로
    프로토콜(rigid/syn)이 다르면 오류 없이 결과만 나빠진다 (실측: own r 0.81 -> 0.72).
    -> (model, sd) 를 돌려주고 sd["t1_source"] 로 입력 프로토콜도 맞출 수 있게 한다.
    """
    sd = torch.load(path, map_location=device, weights_only=False)
    lv = sd.get("unet_level", "none")
    kw.setdefault("use_refiner", bool(sd.get("use_refiner", False)))
    kw.setdefault("prior_use_anatomy", bool(sd.get("prior_use_anatomy", False)))
    kw.setdefault("count_local_dim", int(sd.get("count_local_dim", 0)))
    kw.setdefault("cond_local_dim", int(sd.get("cond_local_dim", 0)))
    # 이걸 빼면 추론이 gain 없는 조건 벡터를 만든다 -- 학습과 다른 입력이라 조용히 틀린다
    kw.setdefault("cond_local_gain", float(sd.get("cond_local_gain", 0.0)))
    kw.setdefault("prior_mu_table", bool(sd.get("prior_mu_table", False)))
    kw.setdefault("count_tier1_dim", int(sd.get("count_tier1_dim", 0)))
    kw.setdefault("count_tier1_pair", bool(sd.get("count_tier1_pair", False)))
    kw.setdefault("aux_local_dim", int(sd.get("aux_local_dim", 0)))
    kw.setdefault("aux_tier1_dim", int(sd.get("aux_tier1_dim", 0)))
    kw.setdefault("prior_local_rank", int(sd.get("prior_local_rank", 0)))
    kw.setdefault("prior_local_dim", int(sd.get("prior_local_dim", 0)))
    if sd.get("pair_anchor"):
        kw.setdefault("pair_anchor", sd["pair_anchor"])
        kw.setdefault("anchor_alpha", float(sd.get("anchor_alpha", 0.0)))
    if sd.get("refiner"):
        kw.setdefault("refiner", sd["refiner"])
    m = ROIPairATM(n_roi=n_roi, trainable="full" if lv == "full" else "vae", unet_level=lv,
                   device=device, in_channels=int(sd.get("in_channels", 1)),
                   template=bool(sd.get("template", False)) or None, **kw)
    m.load_checkpoint(sd["model"])
    m.eval()
    return m, sd


class ROIPairATM(nn.Module):
    def __init__(self, n_roi: int = 82, init_bundle: str = "AF_L",
                 coord_min=None, coord_max=None, t1_norm: BundleNorm | None = None,
                 emb_dim: int = 64, use_weight_head: bool = False, use_edge_head: bool = True,
                 cond_local_gain: float = 0.0, prior_mu_table: bool = False,
                 trainable: str = "vae", device="cuda", models_dir=None, unet_level: str | None = None,
                 in_channels: int = 2, template=None, use_refiner: bool = False,
                 refiner: dict | None = None, prior_use_anatomy: bool = False,
                 count_local_dim: int = 0, cond_local_dim: int = 0,
                 pair_anchor: str | bool | None = None, anchor_alpha: float = 0.0,
                 count_tier1_dim: int = 0, count_tier1_pair: bool = False,
                 aux_local_dim: int = 0, aux_tier1_dim: int = 0,
                 prior_local_dim: int = 0, prior_local_rank: int = 0):
        """trainable: 'decoder' | 'vae' | 'vae+unet4' | 'full'.
        unet_level: 'none' | 'stage4' | 'stage3' | 'stage2' | 'full'. 주면 trainable 의 UNet 부분을 덮어쓴다.
        'full' = VAE 인코더/디코더 + UNet 전체 + heads (최종 전략 §2).
        in_channels: UNet 입력 채널. 2 = [rigid T1 (0~1), WM 확률] (재학습 설계 §2 ①). 1 = 구 프로토콜.
        template: count/edge head 의 그룹 템플릿 (재학습 설계 §2 ③). None 이면 기존 동작.
                  True 또는 경로면 outputs/inference/template.npz 를, dict 면 그 값을 쓴다.
        use_refiner: decode 출력 뒤에 StreamlineRefiner 를 끼운다 (PIPELINE_11 §D1).
                  마지막 층이 0-init 이라 **켜기만 해서는 출력이 바뀌지 않는다** (bit-exact).
                  refiner: 그 정련망의 생성 인자 (hidden/layers/kernel/cond_dim)."""
        super().__init__()
        assert trainable in ("decoder", "vae", "vae+unet4", "full"), trainable
        if unet_level is None:
            unet_level = {"vae+unet4": "stage4", "full": "full"}.get(trainable, "none")
        if coord_min is None or coord_max is None:
            coord_min, coord_max = brain_box_mm()
        base = t1_norm or BundleNorm.from_upstream(init_bundle)
        norm = BundleNorm(base.t1_min, base.t1_max, coord_min, coord_max)

        self.n_roi, self.device = n_roi, torch.device(device)
        self.in_channels = int(in_channels)
        self.atm = ATMBundle(init_bundle, norm, device=device, models_dir=models_dir,
                             in_channels=self.in_channels)
        self.atm.set_unet_trainable(unet_level)
        self.unet_level = unet_level
        self.pair_emb = ROIPairEmbedding(n_roi, emb_dim, ANATOMICAL_DIM, latent_dim=LATENT_DIM,
                                         prior_use_anatomy=prior_use_anatomy,
                                         local_dim=int(cond_local_dim),
                                         local_gain=float(cond_local_gain),
                                         prior_mu_table=bool(prior_mu_table),
                                         prior_local_dim=int(prior_local_dim),
                                       prior_local_rank=int(prior_local_rank),
                                       prior_local_n_roi=(n_roi if prior_local_rank else 0)).to(self.device)
        # pair 앵커 재매개화 (models/pair_anchor.py). 전역 박스가 실제 pair 범위의 28배 부피라
        # z/prior 가 "뇌 어디쯤"까지 떠안고 있다. alpha=0 이면 기존 동작과 bit-exact 다.
        self.anchor = None
        if pair_anchor:
            from .pair_anchor import load as _load_anchor
            self.anchor = _load_anchor(None if pair_anchor is True else pair_anchor,
                                       n_roi=n_roi, alpha=anchor_alpha).to(self.device)
        self.weight_head = StreamlineWeightHead(ANATOMICAL_DIM, LATENT_DIM).to(self.device) \
            if use_weight_head else None
        # 그룹 템플릿 인수분해 (재학습 설계 §2 ③): head 는 템플릿 위의 **개인차만** 학습한다.
        # template=False (checkpoint 메타에서 그대로 넘어오는 값) 도 '안 씀' 으로 받는다
        tpl = None if template is None or template is False else load_group_template(template)
        if tpl is not None:
            for k, v in tpl.items():
                assert v.shape == (n_roi, n_roi), f"{k} 템플릿 {v.shape} != ({n_roi},{n_roi})"
        self.template = tpl
        # aux_* 는 edge_head 와 count_head_end 에 국소/티어1 통로를 연다. 이 둘은 추론에서
        # "어떤 pair 를 만들지"(edge_head)와 "몇 가닥 만들지"(count_head_end)를 정하는데, 전역
        # a512 만 받으면 subject 별로 거의 같은 답을 낸다 (실측 edge Jaccard 0.9896).
        self.edge_head = EdgeHead(ANATOMICAL_DIM, emb_dim,
                                  template_prob=None if tpl is None else tpl["edge_prob"],
                                  local_dim=int(aux_local_dim), tier1_dim=int(aux_tier1_dim),
                                  tier1_n_roi=(self.n_roi if (aux_tier1_dim and count_tier1_pair) else 0)
                                  ).to(self.device) if use_edge_head else None
        # SC edge 값을 직접 예측 (EDGE_ALIGNED 전략 §19-22). weight head 는 streamline 별 가중치라
        # 총합 정규화된 magnitude loss 로는 절대 스케일을 못 배운다 (실측 CCC 0.02).
        # count_head = pass 기준(SC 값) 이므로 sc_pass 템플릿을 쓴다.
        self.count_head = EdgeCountHead(ANATOMICAL_DIM, emb_dim,
                                        template=None if tpl is None else tpl["sc_pass"],
                                        local_dim=int(count_local_dim),
                                        tier1_dim=int(count_tier1_dim),
                                        tier1_n_roi=(self.n_roi if count_tier1_pair else 0)
                                        ).to(self.device) if use_edge_head else None
        # pass 기준(SC 값)과 별도로 **끝점 기준** 개수를 예측한다. 추론에서 pair 마다 몇 가닥을 만들지 정하는 값.
        # GT: 100만 가닥 중 49 %가 두 ROI 를 끝점으로 갖고, pair 당 1~10,150 개로 천차만별이다.
        self.count_head_end = EdgeCountHead(ANATOMICAL_DIM, emb_dim, init_log_count=3.0,
                                            template=None if tpl is None else tpl["sc_end"],
                                            local_dim=int(aux_local_dim), tier1_dim=int(aux_tier1_dim),
                                            tier1_n_roi=(self.n_roi if (aux_tier1_dim and count_tier1_pair) else 0)
                                            ).to(self.device) if use_edge_head else None
        # 디코더 정련망 (PIPELINE_11 §D1). 좌표 상수는 디코더와 **같은 것**을 쓴다 -- 다르면
        # 정규화/역정규화가 어긋나 조용히 틀린다.
        self.use_refiner = bool(use_refiner)
        self.refiner = StreamlineRefiner(
            n_points=N_POINTS, cond_dim=ANATOMICAL_DIM,
            coord_min=self.atm.coord_min.detach().cpu(),
            coord_scale=self.atm.coord_scale.detach().cpu(),
            **(refiner or {})).to(self.device) if self.use_refiner else None
        self.trainable = trainable
        self.norm = norm
        self.weight_head_untrained = bool(self.weight_head is not None and self.weight_head.is_untrained())

    # --- 파라미터 -------------------------------------------------------------
    CONV1_KEY = "atm.net.unet.conv1_1.weight"

    def _pad_conv1_in_channels(self, sd: dict) -> dict:
        """WM 채널 도입 이전(1채널)에 학습된 checkpoint 를 2채널 모델에 싣는다.

        새 채널을 0 으로 채우므로 그 checkpoint 의 동작은 그대로 재현된다 (ch1 기여 0).
        다만 그 가중치는 syn + robust 정규화 T1 로 학습된 것이라 rigid + [0,1] 입력과 섞으면
        결과가 달라진다 -- 조용히 넘어가지 않게 찍는다.
        """
        w = sd.get(self.CONV1_KEY)
        if w is None or w.shape[1] == self.in_channels:
            return sd
        assert w.shape[1] < self.in_channels, (
            f"checkpoint conv1_1 이 {w.shape[1]}채널인데 모델은 {self.in_channels}채널이다")
        pad = torch.zeros(w.shape[0], self.in_channels - w.shape[1], *w.shape[2:],
                          dtype=w.dtype, device=w.device)
        sd = dict(sd)
        sd[self.CONV1_KEY] = torch.cat([w, pad], dim=1)
        print(f"[load] conv1_1 을 {w.shape[1]} -> {self.in_channels} 채널로 0-패딩 "
              "(구 1채널 checkpoint). 입력 프로토콜이 다르면 결과가 달라진다.", flush=True)
        return sd

    @staticmethod
    def _drop_dead_keys(sd: dict, have_weight_head: bool) -> dict:
        """더 이상 만들지 않는 모듈의 checkpoint 키를 버린다.

        weight_head 는 by-count 배분(생성 가닥 수 자체가 SC)에서 구조적으로 불필요하고, 이 프로젝트의
        어느 phase 도 학습시킨 적이 없다 (마지막 층 가중치가 정확히 0 -> w = 1.0 상수). 213,761 개
        파라미터를 옵티마이저와 checkpoint 에 계속 싣고 있었다. 옛 checkpoint 는 이 키를 갖고 있으므로
        모델에 그 모듈이 없으면 조용히 버린다 (없으면 "checkpoint 에만 있는 키" assert 로 죽는다)."""
        if have_weight_head:
            return sd
        drop = [k for k in sd if k.startswith("weight_head.")]
        if drop:
            print(f"[load] weight_head 키 {len(drop)}개 버림 (by-count 배분에서 불필요)", flush=True)
        return {k: v for k, v in sd.items() if not k.startswith("weight_head.")}

    def load_checkpoint(self, sd: dict, strict: bool = False) -> dict:
        """checkpoint 이어받기. strict=False 면 **이 모델에만 있는 새 모듈**(나중에 추가한 head)은
        초기값을 유지하고 나머지는 그대로 싣는다. checkpoint 에만 있는 키가 있으면 구조가 바뀐 것이므로 중단한다."""
        sd = self._pad_conv1_in_channels(sd)
        sd = self._drop_dead_keys(sd, self.weight_head is not None)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, f"checkpoint 에만 있는 키 (구조 불일치): {unexpected[:8]}"
        if missing:
            assert not strict, f"checkpoint 에 없는 키: {missing[:8]}"
            print(f"[load] checkpoint 에 없는 새 파라미터 {len(missing)}개는 초기값 사용: "
                  f"{sorted({k.split('.')[0] for k in missing})}", flush=True)
            if any(k.endswith(("template_log", "template_logit")) for k in missing):
                # head 가 이미 그룹 평균을 bias 에 흡수한 상태다. 거기에 템플릿을 또 더하면
                # count 가 exp(init_log_count) 배 어긋난다 -- 조용히 넘어가면 안 된다.
                print("[load] 경고: 템플릿 인수분해 없이 학습된 checkpoint 다. 재학습 전에는 "
                      "template=None 으로 불러야 스케일이 맞는다.", flush=True)
        # 학습이 weight head 를 손실에 연결하지 않으면(trainer.weight_mode != 'head'/'count_head')
        # 마지막 층이 0-init 그대로 남고 출력이 상수 1.0 이 된다. 추론은 그 w 를 그대로 곱하므로
        # sc_w 는 가중 없는 count SC 와 같아진다 -- 조용히 넘어가면 안 되는 상태다.
        self.weight_head_untrained = bool(self.weight_head is not None and self.weight_head.is_untrained())
        if self.weight_head_untrained:
            print("[load] 경고: weight head 가 학습되지 않았다 (net[-1].weight == 0). "
                  "w = 1.0 상수이므로 weighted SC(sc_w) 는 가중 없는 count SC 와 수치적으로 동일하다. "
                  "가중치를 실제로 쓰려면 weight_mode='head'|'count_head' 로 재학습해야 한다.", flush=True)
        return {"missing": missing, "unexpected": unexpected}

    def param_groups(self) -> dict:
        """이름별 학습 파라미터 (최종 전략 §8 의 LR 그룹). 겹치지 않는다."""
        # 정련망은 'decoder' 그룹에 넣는다 (lr_dec 로 학습). 새 키를 만들면
        # Trainer 의 lrs[k] 조회가 KeyError 로 죽는다 -- trainer.py 를 안 고치는 조건.
        dec = list(self.atm.decoder_parameters())
        if self.refiner is not None:
            dec = dec + list(self.refiner.parameters())
        dec_ids = {id(p) for p in dec}
        # prior 의 **분산**만 따로 뗀다. 이 파라미터는 오직 prior 적합항 하나로만 학습되고
        # (KL 과 생성 경로에서는 detach 된다), 목표까지 log 공간으로 0.6 쯤 움직여야 하는데
        # Adam step 은 손실 가중치가 아니라 LR 로만 정해진다 -- 즉 lr_heads(3.3e-5)로는
        # 3,000 step 에 절반도 못 간다 (실측 1.25e-4/step). 별도 LR 그룹이 필요한 이유다.
        prior_scale = [self.pair_emb.prior_log_sigma.weight, self.pair_emb.prior_log_sigma.bias,
                       self.pair_emb.mode_log_sigma.weight]
        ps_ids = {id(p) for p in prior_scale}
        # tier1 pair 가중치는 33,210 개를 144명으로 맞춘다. 프로브 ridge 가 lambda=10000 이라는
        # 강한 L2 아래에서 r=0.1305 를 낸 값이라, 정규화 없이 그대로 두면 재현이 안 된다.
        # 별도 그룹으로 빼서 weight decay 를 따로 건다 (AdamW 의 decoupled decay).
        tier1_pair = [p for h in (self.count_head, self.count_head_end, self.edge_head)
                      if h is not None and getattr(h, "tier1_w", None) is not None
                      for p in (h.tier1_w, h.tier1_b)]
        # prior 의 pair 별 저랭크 가중치도 같은 weight decay 를 받아야 한다 (1.7M / 144명).
        if getattr(self.pair_emb, "prior_local_w", None) is not None:
            tier1_pair.append(self.pair_emb.prior_local_w)
            tier1_pair.extend(self.pair_emb.prior_local_u.parameters())   # U 도 같은 LR (heads 의 3e-5 로는 안 움직인다)
        ps_ids |= {id(p) for p in tier1_pair}
        groups = {"t1_encoder": list(self.atm.unet_trainable_parameters()),
                  "vae_encoder": [] if self.trainable == "decoder"
                  else [p for p in self.atm.net.ae.parameters() if id(p) not in dec_ids],
                  "decoder": dec,
                  "prior_scale": prior_scale,
                  "tier1_pair": tier1_pair,
                  "heads": [p for p in self.pair_emb.parameters() if id(p) not in ps_ids]
                  + (list(self.weight_head.parameters()) if self.weight_head is not None else [])
                  + ([p for p in self.edge_head.parameters() if id(p) not in ps_ids]
                     if self.edge_head is not None else [])
                  + ([p for p in self.count_head.parameters() if id(p) not in ps_ids]
                     if self.count_head is not None else [])
                  + ([p for p in self.count_head_end.parameters() if id(p) not in ps_ids]
                     if self.count_head_end is not None else [])}
        return groups

    def trainable_parameters(self):
        return [p for ps in self.param_groups().values() for p in ps]

    def param_counts(self) -> dict:
        g = {k: sum(p.numel() for p in v) for k, v in self.param_groups().items()}
        g["trainable"] = sum(g.values())
        g["total"] = sum(p.numel() for p in self.parameters())
        g["frozen"] = g["total"] - g["trainable"]
        return g

    def anatomy_forward(self, anat_input: torch.Tensor) -> torch.Tensor:
        """subject 당 step 마다 1회. 입력 종류로 경로를 고른다.
          [1,512]            이미 계산된 feature (동결 인코더)          -> 그대로
          [1,256,49,58,49]   stage3 캐시 (unet_level='stage4')           -> conv4+fc, grad
          [1,C,193,229,193]  T1(+WM) (unet_level in stage3/stage2/full)  -> checkpointed grad 경로
        """
        if anat_input.ndim == 2:
            assert anat_input.shape == (1, 512), anat_input.shape
            return anat_input
        if anat_input.ndim == 5 and anat_input.shape[1] == 256:
            assert self.unet_level == "stage4", f"stage3 캐시는 unet_level='stage4' 전용 (now {self.unet_level})"
            return self.anatomy_from_stage3(anat_input)
        assert anat_input.ndim == 5 and anat_input.shape[1] == self.in_channels, (
            anat_input.shape, self.in_channels)
        if self.unet_level == "none":
            self._live_stage3 = None
            return self.encode_anatomy(anat_input)
        # 인코더를 학습하면 디스크 캐시 국소 feature 가 낡는다 -> 살아있는 stage3 를 보관해
        # trainer 가 여기서 ROI 풀링하게 한다 (data/local_feats.py 캐시는 동결일 때만 유효).
        a, o3 = self.atm.encode_anatomy_grad(anat_input, self.unet_level,
                                             use_checkpoint=getattr(self.atm, "use_checkpoint", True),
                                             return_stage3=True)
        self._live_stage3 = o3
        return a

    def train(self, mode: bool = True):
        """UNet 은 항상 eval (Dropout3d 비활성 -- batch 1 에서 p=0.2 dropout 은 feature 를 실행마다
        바꾼다). conv4 를 학습하더라도 eval 모드로 gradient 만 흘린다. ConvVAE 의 BatchNorm 은
        학습 시 batch 통계."""
        super().train(mode)
        self.atm.net.unet.eval()
        return self

    # --- 학습 가능한 anatomy 경로 (trainable='vae+unet4') ------------------------
    def cache_stage3(self, t1_w: torch.Tensor) -> torch.Tensor:
        return self.atm.cache_stage3(t1_w)

    def anatomy_from_stage3(self, o3: torch.Tensor) -> torch.Tensor:
        return self.atm.anatomy_from_stage3(o3)

    # --- forward --------------------------------------------------------------
    @torch.no_grad()
    def encode_anatomy(self, t1_w: torch.Tensor) -> torch.Tensor:
        return self.atm.encode_anatomy(t1_w)

    def condition(self, anatomy: torch.Tensor, pairs: torch.Tensor, mode=0,
                  local: torch.Tensor | None = None) -> torch.Tensor:
        """mode 0 = full streamline, 1 = SC edge-aligned segment (같은 decoder, 조건만 다름).
        local [N, D] 은 pair_emb 이 local_dim > 0 으로 만들어졌을 때만 준다."""
        return self.pair_emb(anatomy, canonical_pairs(pairs), mode, local)

    def decode(self, z: torch.Tensor, cond: torch.Tensor,
               pairs: torch.Tensor | None = None) -> torch.Tensor:
        """pairs 를 주고 anchor 가 켜져 있으면 좌표를 pair 앵커 기준으로 재매개화한다.
        anchor.alpha = 0 이면 기존 경로와 bit-exact 다."""
        if self.anchor is not None and float(self.anchor.alpha) != 0.0:
            assert pairs is not None, "pair 앵커가 켜져 있는데 decode 에 pairs 가 안 넘어왔다"
            raw = self.atm.decode_raw(z, cond)
            mm_global = (raw + 1.0) * self.atm.coord_scale + self.atm.coord_min
            mm = self.anchor(raw, canonical_pairs(pairs), mm_global)
        else:
            mm = self.atm.decode_mm(z, cond)
        if self.refiner is None:
            return mm
        # 0-init 상태에서는 delta == 0 이라 이 분기는 기존 출력과 bit-exact 동일하다.
        return self.refiner(mm, cond)

    def encode_streamlines(self, mm: torch.Tensor, cond: torch.Tensor):
        return self.atm.encode_streamline(mm, cond)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def weights(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.weight_head is None:
            return torch.ones(z.shape[0], device=z.device)
        return self.weight_head(cond, z)

    def edge_logits(self, anatomy: torch.Tensor, pairs: torch.Tensor,
                    local: torch.Tensor | None = None,
                    tier1: torch.Tensor | None = None) -> torch.Tensor:
        assert self.edge_head is not None, "edge head 비활성"
        cp = canonical_pairs(pairs)
        return self.edge_head(anatomy, self.pair_emb.pair_vec(cp), cp, local, tier1)

    def edge_log_counts(self, anatomy: torch.Tensor, pairs: torch.Tensor,
                        local: torch.Tensor | None = None,
                        tier1: torch.Tensor | None = None) -> torch.Tensor:
        """[K,2] -> 예측 log count [K] (자연로그). exp 하면 그 edge 가 가져야 할 segment/streamline 수.
        local [K, D] 은 count head 가 local_dim > 0 으로 만들어졌을 때만 준다 (pair 별 국소 anatomy)."""
        assert self.count_head is not None, "count head 비활성"
        cp = canonical_pairs(pairs)
        return self.count_head(anatomy, self.pair_emb.pair_vec(cp), cp, local, tier1)

    def edge_log_counts_end(self, anatomy: torch.Tensor, pairs: torch.Tensor,
                            local: torch.Tensor | None = None,
                            tier1: torch.Tensor | None = None) -> torch.Tensor:
        """[K,2] -> 끝점 기준 log count [K]. exp 하면 그 pair 를 끝점으로 갖는 가닥 수."""
        assert self.count_head_end is not None, "count head 비활성"
        cp = canonical_pairs(pairs)
        return self.count_head_end(anatomy, self.pair_emb.pair_vec(cp), cp, local, tier1)

    def edge_count_matrix(self, anatomy: torch.Tensor, pairs: torch.Tensor,
                          local: torch.Tensor | None = None,
                          tier1: torch.Tensor | None = None) -> torch.Tensor:
        """[K,2] -> 대칭 [n_roi,n_roi] 예측 count 행렬 (대각 0).

        local/tier1 은 count head 가 그 통로로 만들어졌으면 **반드시** 줘야 한다 (안 주면 head 의
        assert 가 잡는다). 예전에는 여기서 안 넘겨서 국소 통로를 켜면 추론이 죽었다."""
        assert self.count_head is not None, "count head 비활성"
        cp = canonical_pairs(pairs)
        return self.count_head.matrix(anatomy, self.pair_emb.pair_vec(cp), cp, self.n_roi,
                                      local, tier1)

    def prior_mean(self, pairs: torch.Tensor, mode=0) -> torch.Tensor:
        return self.pair_emb.prior_mean(canonical_pairs(pairs), mode)

    def prior_params(self, pairs: torch.Tensor, mode=0, anatomy: torch.Tensor | None = None,
                     local: torch.Tensor | None = None):
        """(mu [N,D], log_sigma [N,D]) = p(z | pair[, anatomy, local]). log_sigma 0-init 이면 sigma == 1.0.
        local [N, D] 은 pair 별 국소 anatomy (D-f: pair 의존 subject 조건부 prior)."""
        return self.pair_emb.prior_params(canonical_pairs(pairs), mode,
                                          self._prior_anat(anatomy, pairs), local)

    def sample_eps(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(n, LATENT_DIM, device=self.device, generator=generator)

    def _prior_anat(self, anatomy, pairs) -> torch.Tensor | None:
        """prior 통로가 켜져 있을 때만 anatomy 를 [N, cond_dim] 으로 맞춰 넘긴다.

        꺼져 있으면 None 을 넘겨야 한다 (`prior_params` 가 assert 로 강제한다). 켜져 있는데
        호출부가 anatomy 를 안 주면 조용히 pair-only 로 도는 대신 여기서 멈춘다.
        """
        if not self.pair_emb.prior_use_anatomy:
            return None
        assert anatomy is not None, "prior_use_anatomy=True 인데 anatomy 가 안 넘어왔다"
        n = pairs.shape[0]
        if anatomy.shape[0] == 1 and n != 1:
            anatomy = anatomy.expand(n, -1)
        assert anatomy.shape == (n, self.pair_emb.cond_dim), (anatomy.shape, n)
        return anatomy

    def sample_z(self, pairs_or_n, generator: torch.Generator | None = None,
                 anatomy: torch.Tensor | None = None,
                 local: torch.Tensor | None = None) -> torch.Tensor:
        """z ~ N(mu_pair, diag(sigma_pair^2)). 정수를 주면 (하위 호환) N(0, I).

        log_sigma 0-init 인 구 checkpoint 에서는 `mu + exp(0)*eps == mu + eps` 라 예전 경로와
        **bit-exact** 같다 (scripts/48_prior_ladder.py:selfcheck).
        """
        if isinstance(pairs_or_n, int):
            assert not self.pair_emb.prior_use_anatomy, "anatomy prior 인데 pair 없이 z 를 뽑으려 한다"
            return self.sample_eps(pairs_or_n, generator)
        cp = canonical_pairs(pairs_or_n)
        return self.pair_emb.sample_prior(cp, anatomy=self._prior_anat(anatomy, cp),
                                          generator=generator, local=local)

    def generate(self, anatomy: torch.Tensor, pairs: torch.Tensor, n_per_pair: int,
                 chunk: int = 8192, generator=None, amp_dtype=None, z=None,
                 local_roi: torch.Tensor | None = None):
        """pairs [K,2] 각각에 대해 n_per_pair 개. -> (mm [K*n,128,3], w [K*n], pairs_rep [K*n,2]).

        inference 전용 (grad 없음). anatomy 는 이미 계산된 것을 재사용한다.
        z 를 주면 사전분포 대신 그것을 쓴다 (inference.latent_bank).
        local_roi [R, D] 는 pair_emb.local_dim > 0 일 때 필요하다 (subject 의 ROI 국소 anatomy).
        """
        assert not ((self.pair_emb.local_dim or self.pair_emb.prior_local is not None)
                    and local_roi is None), "국소 통로가 켜져 있는데 local_roi 가 안 넘어왔다"
        with torch.inference_mode():
            pr = canonical_pairs(pairs).repeat_interleave(n_per_pair, dim=0)
            if z is None:
                pl = None
                if self.pair_emb.prior_local is not None:
                    from ..data.local_feats import pair_local
                    assert local_roi is not None, "prior 국소 통로가 켜졌는데 local_roi 가 없다"
                    pl = pair_local(local_roi, canonical_pairs(pr))
                z = self.sample_z(pr, generator, anatomy=anatomy, local=pl)
            else:                       # latent bank 등 외부에서 준 latent
                assert z.shape[0] == pr.shape[0], (z.shape, pr.shape)
                z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
            outs, ws = [], []
            for i in range(0, pr.shape[0], chunk):
                pc = pr[i:i + chunk]
                lc = None
                if self.pair_emb.local_dim:
                    from ..data.local_feats import pair_local
                    lc = pair_local(local_roi, canonical_pairs(pc))
                c = self.condition(anatomy, pc, local=lc)
                if amp_dtype is None:
                    mm = self.decode(z[i:i + chunk], c, pc)
                else:
                    with torch.autocast(self.device.type, dtype=amp_dtype):
                        mm = self.decode(z[i:i + chunk], c, pc)
                outs.append(mm.float()); ws.append(self.weights(c, z[i:i + chunk]))
            return torch.cat(outs), torch.cat(ws), pr
