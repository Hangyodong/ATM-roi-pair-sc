"""학습 phase 공통 실행기 (scripts/07~10 이 호출).

phase 별 활성 loss (pipeline §21):
    roi_atm  : atm                                    (Phase 2)
    endpoint : atm + endpoint                         (Phase 3)
    edge     : atm + endpoint + edge                  (Phase 4)
    sc       : atm + endpoint + edge + corr + mag     (Phase 6-7)
    full     : 전부 (+ length)                        (Phase 8-9)

full training 은 이 파일의 책임이 아니다 -- max_steps 로 반드시 제한해서 부른다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import ROIPairSubject
from ..data.paths import ATLAS, CACHE
from ..models.endpoint_assigner import EndpointAssigner
from ..models.roi_atm import ROIPairATM
from .trainer import LossWeights, TrainConfig, Trainer

class Preempted(Exception):
    """다른 실행(예: PBS H100 job)이 이어받을 수 있게 latest 를 저장하고 빠져나온다."""


PHASES = {                                # 최종 전략 §7
    "geometry": {"recon"},                             # Phase 2  (T1 encoder 동결)
    "t1_encoder": {"recon"},                           # Phase 3  (UNet unfreeze, recon 만)
    "endpoint": {"recon", "endpoint"},                 # Phase 4
    "edge": {"recon", "endpoint", "edge"},             # Phase 5
    "sc_corr": {"recon", "endpoint", "edge", "corr"},  # Phase 6
    "sc": {"recon", "endpoint", "edge", "corr", "mag"},           # Phase 7
    "full": {"recon", "endpoint", "edge", "corr", "mag", "length"},  # Phase 8-9
    "roi_atm": {"recon"},                              # 구 이름 호환
    # ROUTE 전략 §25 (geometry -> endpoint -> route -> global SC -> block SC -> mag -> length -> joint)
    "route": {"recon", "endpoint", "route"},
    "route_edge": {"recon", "endpoint", "route", "edge"},
    "route_sc_corr": {"recon", "endpoint", "route", "edge", "corr"},
    "route_sc_presence": {"recon", "endpoint", "route", "edge", "corr", "presence", "scale"},
    "route_sc_mag": {"recon", "endpoint", "route", "edge", "corr", "presence", "mag"},
    "route_full": {"recon", "endpoint", "route", "edge", "corr", "presence", "mag", "length"},
    # EDGE_ALIGNED 전략 §13-22: segment 분기 + edge count head 를 얹은 단계
    "seg_count": {"recon", "endpoint", "route", "edge", "corr", "segment", "count", "scale"},
    # segment(=SC edge 단위) 분기와 edge count 를 처음부터 켜는 구성
    "seg_route": {"recon", "endpoint", "route", "edge", "corr", "segment", "count", "scale"},
    "seg_route_presence": {"recon", "endpoint", "route", "edge", "corr", "presence", "segment", "count", "scale"},
    # 재학습 설계 (PIPELINE_08) P0-P4: 기하를 먼저 안정화하고 손실을 순서대로 켠다.
    # 실측 근거: 같은 디코더가 recon 만 학습하면 2.655 mm, 다른 손실과 경쟁하면 3.55 mm (34 % 차이).
    "p0_warmup": {"recon"},
    "p1_route": {"recon", "endpoint", "route", "edge"},
    "p2_count": {"recon", "endpoint", "route", "edge", "segment", "count"},
    "p3_mag": {"recon", "endpoint", "route", "edge", "segment", "count", "corr", "presence", "scale", "mag"},
    "p4_joint": {"recon", "endpoint", "route", "edge", "segment", "count", "corr", "presence",
                 "scale", "mag", "rmse", "length"},
    # D1 디코더 충실도 (PIPELINE_11 §D1-d): 복원 하나만. 조건화도 SC 도 route 도 없다.
    # BatchNorm train/eval 격차(8.46 vs 3.55 mm)를 고치는 것이 이 phase 의 목적이라
    # 판정은 반드시 eval 모드 `recon_rmse_eval_mm` 로 한다.
    "d1_decoder": {"recon"},
    # A1 잔차 타깃 (실험 A): count head 하나만 학습한다. 생성도 복원도 없어 step 이 싸고,
    # 개인차 supervision 을 직접 거는 유일한 구성이다. 판정은 val 의 `resid_r`.
    "a1_resid": {"count"},
    # D3 joint (W3-a): D1 이 복원만 학습해 디코더를 날카롭게 만든 결과 **생성 경로가 무너졌다**
    # (C13: valid_conn 0.367 -> 0.042). posterior drift 는 아니었고(오프셋 5.27 -> 3.51 로 오히려 감소),
    # prior 가 뽑는 영역이 더 이상 디코더가 정확한 영역이 아니게 된 것이다. 그래서 순차 phase 로
    # 나누지 않고 복원 + 생성 제약(endpoint/route/SC) + prior 적합을 **동시에** 돈다.
    "d3_joint": {"recon", "endpoint", "route", "edge", "segment", "count", "corr", "presence",
                 "scale", "mag", "rmse", "length", "prior"},
    "seg_count_mag": {"recon", "endpoint", "route", "edge", "corr", "presence", "mag", "scale", "rmse", "segment", "count"},
    "seg_full": {"recon", "endpoint", "route", "edge", "corr", "presence", "mag", "scale", "rmse", "length", "segment", "count"},
}


# T1 캐시 종류: (파일 접미사, [0,1] 정규화 여부)
#   rigid  전처리 프로토콜 (rigid 정합 + unit 정규화). WM 채널과 같이 쓴다 (재학습 설계 §2 ①).
#   syn    구 프로토콜 (SyN + robust 정규화). 옛 checkpoint 재현용으로만 남긴다.
T1_SOURCES = {"rigid": ("_T1w_rigid_W.npy", True), "syn": ("_T1w_syn_W.npy", False)}
T1_SOURCE_DEFAULT = "rigid"


def _cache_tag(model: ROIPairATM, source: str) -> str:
    """캐시 파일명 접미사. 구 프로토콜(syn 1채널) 캐시만 옛 이름을 그대로 쓴다.

    프로토콜을 파일명에 박지 않으면 syn/1채널로 만든 anatomy 캐시가 rigid/2채널 학습에
    조용히 섞인다 (숫자는 나오는데 틀린 값이다).
    """
    return "" if (source == "syn" and model.in_channels == 1) else f"_{source}{model.in_channels}ch"


def t1_input(model: ROIPairATM, sub: str, source: str = T1_SOURCE_DEFAULT) -> torch.Tensor:
    """학습 가능한 UNet 경로용 입력 [1, C, 193, 229, 193] (CPU). step 마다 로드 (C x 34 MB).

      ch0  T1 (source='rigid' 면 [0,1] 정규화, 'syn' 이면 구 robust 정규화)
      ch1  WM 확률 0~1 (model.in_channels == 2 일 때)

    WM 은 T1 에서 결정론적으로 계산된 것이라 정보가 새로 생기지는 않는다. 학습을 쉽게 만드는
    유도 편향이다 (PIPELINE_02_MODEL.md §8).
    """
    assert source in T1_SOURCES, f"알 수 없는 T1 source: {source} (가능: {sorted(T1_SOURCES)})"
    suffix, unit = T1_SOURCES[source]
    t1p = CACHE / f"{sub}{suffix}"
    assert t1p.exists(), f"{sub}: T1 W 캐시 없음 ({t1p.name}) -> scripts/01_qc_coordinate_space.py 먼저"
    t1 = np.load(t1p)
    chans = [np.asarray(model.norm.normalize_t1(t1, unit=unit), np.float32)]
    c = int(getattr(model, "in_channels", 1))
    if c >= 2:
        wp = CACHE / f"{sub}_WM_W.npy"
        assert wp.exists(), f"{sub}: WM 확률맵 없음 ({wp.name}) -> scripts/data/wm_segment 경로 먼저"
        wm = np.load(wp).astype(np.float32)
        assert wm.shape == t1.shape, (wm.shape, t1.shape)
        assert np.isfinite(wm).all(), f"{sub}: WM 맵에 NaN/Inf"
        assert 0.0 <= wm.min() and wm.max() <= 1.0, f"{sub}: WM 이 확률이 아니다 ({wm.min()}, {wm.max()})"
        assert wm.sum() > 0, f"{sub}: WM 맵이 전부 0"
        chans.append(wm)
    assert len(chans) == c, f"채널 {len(chans)}개를 만들었는데 모델은 {c}채널을 받는다"
    return torch.tensor(np.stack(chans)[None], dtype=torch.float32)


def anatomy_feature(model: ROIPairATM, sub: str, init_bundle: str,
                    source: str = T1_SOURCE_DEFAULT) -> torch.Tensor:
    """subject 당 1회. 디스크 캐시 (프로토콜별로 파일명이 다르다)."""
    p = CACHE / f"{sub}_anat_{init_bundle}{_cache_tag(model, source)}.npy"
    if p.exists():
        return torch.from_numpy(np.load(p)).to(model.device)
    a = model.encode_anatomy(t1_input(model, sub, source))
    assert float(a.norm()) > 1e-3, f"{sub}: anatomy feature 가 0 에 가까움 ({float(a.norm()):.2e})"
    np.save(p, a.cpu().numpy())
    return a


def stage3_cache(model: ROIPairATM, sub: str, init_bundle: str,
                 source: str = T1_SOURCE_DEFAULT) -> torch.Tensor:
    """trainable='vae+unet4' 용. conv1~3 출력 [1,256,49,58,49] 을 fp16 으로 디스크 캐시 (72 MB)."""
    p = CACHE / f"{sub}_stage3_{init_bundle}{_cache_tag(model, source)}.npy"
    if p.exists():
        return torch.from_numpy(np.load(p)).float().to(model.device)
    o3 = model.cache_stage3(t1_input(model, sub, source))
    assert torch.isfinite(o3).all() and float(o3.abs().max()) > 0
    np.save(p, o3.half().cpu().numpy())
    return o3


def _row_r(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """행 i 끼리의 Pearson 상관 [n]. 학습 초기 count head 는 모든 edge 에 같은 값을 내므로
    (분산 0) nan 이 섞일 수 있다. 부트스트랩 CI 에는 이 subject 단위 값이 필요하다."""
    assert A.shape == B.shape, (A.shape, B.shape)
    return np.array([np.corrcoef(A[i], B[i])[0, 1] if A[i].std() > 1e-12 and B[i].std() > 1e-12 else np.nan
                     for i in range(len(A))], np.float64)


def _mean_r(A: np.ndarray, B: np.ndarray) -> float:
    """_row_r 의 평균. nan 은 빼고 평균."""
    r = _row_r(A, B)
    r = r[np.isfinite(r)]
    return float(r.mean()) if r.size else float("nan")


@torch.no_grad()
def individuality_metrics(model: ROIPairATM, subjects: list[str], n_subj: int = 8, seed: int = 0,
                          pair_dice: bool = False, n_dice_pairs: int = 30, n_dice_subj: int = 3,
                          n_dice_per_pair: int = 128, source: str = T1_SOURCE_DEFAULT) -> dict:
    """개인차 상시 지표 (PIPELINE_08_RETRAIN_DESIGN.md §2 ⑤). 학습이 끝난 뒤에만 재면
    어떤 변경이 효과였는지 알 수 없으므로 검증마다 같이 잰다.

      abl_own_r / abl_shuf_r / abl_zero_r / abl_gap
          같은 모델에 own(자기 T1) / shuffled(한 칸 옆 subject 의 T1) / zero(T1=0) 를 넣었을 때의
          SC 상관. **own > shuf > zero 가 뚜렷해야** T1 을 실제로 읽고 있는 것이다.
          s5_joint/seg_full_step4000 · test 31명 실측: 0.8108 / 0.8101 / 0.8123 -- 사실상 같고
          T1 을 지운 쪽이 오히려 미세하게 낫다 (PIPELINE_06_FINDINGS.md §B-2).
          문서의 zero 값 0.8067 은 anatomy feature 를 0 으로 둔 것이고, 여기서는 **T1 자체를 0**
          으로 넣어 UNet 인코더까지 포함해 잰다 (2채널 전환 후 그쪽이 바뀌므로).
      resid_r        LOO 중심화 잔차 상관. 개인차를 잡았는가. 현재 0.026 (필요값 0.42).
      inter_subj_r   예측 SC 끼리의 평균 쌍 상관. GT 끼리는 0.897 인데 생성은 0.9995 다 (§B-3).
      pair_dice      상위 pair 의 생성 번들 vs 실제 번들 복셀 dice + 실제 표본끼리의 천장.

    SC 는 count head 직접 예측(`edge_log_counts`)을 쓴다 -- tractogram 을 만들지 않아 싸고,
    위 ablation 기준값 0.81 이 바로 이 경로의 값이다. `pair_dice` 만 실제 생성이 필요하므로
    기본으로 꺼져 있다 (n_dice_subj 명 x n_dice_pairs 개 pair 로 비용을 묶는다).
    """
    from ..evaluation.balance_metrics import bundle_geometry_metrics
    from ..evaluation.reproduction_metrics import ablation_gap, subject_specificity

    if model.count_head is None:
        return {}
    suffix = T1_SOURCES[source][0]                    # 실제로 t1_input 이 읽는 캐시로 거른다
    subs = [s for s in subjects if (CACHE / f"{s}{suffix}").exists()][:n_subj]
    assert len(subs) >= 3, f"개인차 지표에는 T1 캐시가 있는 val subject 3명 이상 필요 (now {len(subs)})"
    iu = np.triu_indices(int(model.n_roi), 1)
    Pt = torch.as_tensor(np.stack(iu, 1).astype(np.int64), device=model.device)

    feats, gts, G = [], [], []
    for s in subs:
        a = model.atm.encode_anatomy(t1_input(model, s, source))
        assert float(a.norm()) > 1e-3, f"{s}: anatomy feature 가 0 에 가까움 ({float(a.norm()):.2e})"
        feats.append(a)
        subj = ROIPairSubject(s)
        gts.append(subj)
        G.append(np.asarray(subj.sc_mat, np.float64)[iu])
    G = np.stack(G)
    assert np.isfinite(G).all() and (G.sum(1) > 0).all(), "GT SC 가 비었거나 NaN"

    def count_sc(a):
        v = model.edge_log_counts(a, Pt).exp().double().cpu().numpy()
        assert np.isfinite(v).all(), "count head 예측에 NaN/Inf"
        return v

    # T1=0 의 anatomy 는 subject 와 무관하므로 한 번만 계산한다.
    a_zero = model.atm.encode_anatomy(torch.zeros_like(t1_input(model, subs[0], source)))
    P = np.stack([count_sc(a) for a in feats])
    P_shuf = np.stack([count_sc(feats[(i + 1) % len(subs)]) for i in range(len(subs))])
    P_zero = count_sc(a_zero)
    n = len(subs)

    Z = np.broadcast_to(P_zero, P.shape)
    # subject 별 상관을 버리지 않고 그대로 넘긴다 (abl_gap 부트스트랩 CI 게이트에 필요).
    r_own, r_shuf, r_zero = _row_r(P, G), _row_r(P_shuf, G), _row_r(Z, G)
    assert len(r_own) == len(r_shuf) == len(r_zero) == n, (len(r_own), len(r_shuf), len(r_zero), n)

    def _fm(r):
        f = r[np.isfinite(r)]
        return float(f.mean()) if f.size else float("nan")
    out = {"n_indiv_subj": n,
           **ablation_gap(_fm(r_own), _fm(r_shuf), _fm(r_zero),
                          own_r=r_own, shuf_r=r_shuf, zero_r=r_zero)}
    # 예측 자체가 T1 에 얼마나 반응하는가 (GT 와 무관한 진단값). 1.0 이면 T1 을 전혀 안 쓴다.
    out["abl_pred_own_vs_zero_r"] = _mean_r(P, Z)
    sp = subject_specificity(P, G)                       # LOO 중심화 잔차 + subject 간 유사도
    out.update(resid_r=sp["resid_r"], pred_degenerate=sp["pred_degenerate"],
               inter_subj_r=sp["inter_subj_r_pred"], inter_subj_r_gt=sp["inter_subj_r_gt"])

    if pair_dice:
        rng = np.random.default_rng(seed)
        dices, ceils = [], []
        for si in range(min(n, max(1, n_dice_subj))):
            subj = gts[si]
            pid = np.asarray(subj.pair_ids, np.int64)
            g = torch.Generator(device=model.device); g.manual_seed(seed)
            for k in np.argsort(-np.asarray(subj.pair_count_full))[:n_dice_pairs]:
                gt_b = subj.get_pair(int(k))[0].numpy()
                h = min(len(gt_b) // 2, n_dice_per_pair)
                if h < 2:                               # 두 표본으로 나눌 수 없다
                    continue
                pm = rng.permutation(len(gt_b))
                ref, other = gt_b[pm[:h]], gt_b[pm[h:2 * h]]
                Pk = torch.as_tensor(pid[k][None], device=model.device)
                S, _, _ = model.generate(feats[si], Pk, h, generator=g)
                dices.append(bundle_geometry_metrics(S.cpu().numpy().astype(np.float32), ref)["dice"])
                # 천장: 같은 pair 의 **실제** streamline 두 표본끼리. dice 는 번들 크기에 민감하므로
                # 생성쪽과 같은 조건(h개 vs 같은 ref h개)으로 재야 비교가 된다.
                # tractography 자체의 재현 한계이자 이 지표가 오를 수 있는 상한이다.
                ceils.append(bundle_geometry_metrics(other, ref)["dice"])
        assert dices, "pair dice 를 잴 pair 가 없음 (bundles.npz 확인)"
        out.update(pair_dice=float(np.mean(dices)), pair_dice_ceiling=float(np.mean(ceils)),
                   pair_dice_n=len(dices))
    return out



# --- 복원 충실도 계측 (P4) ------------------------------------------------------
# 배포되는 것은 **eval 모드** 복원이다. trainer 로그의 recon_rmse_train_mm 은 배치 통계 값이라
# 추론에서 성립하지 않는다 (실측: train 3.55 mm / eval 8.46 mm, W1-d). 한쪽만 보면 착시가
# 반복되므로 여기서는 **두 모드를 같이** 낸다.

def _bn_utils():
    """scripts/38_decoder_capacity.py 의 BN 유틸(bn_state/bn_restore/bn_recalibrate)을 재사용.

    중복 구현하면 두 곳이 갈린다. 파일명이 숫자로 시작해 일반 import 가 안 되고, 그 스크립트가
    이 모듈(t1_input)을 import 하므로 **반드시 지연 import** 여야 한다.
    """
    import importlib.util
    p = Path(__file__).resolve().parents[3] / "scripts" / "38_decoder_capacity.py"
    assert p.exists(), f"BN 유틸 원본이 없다: {p}"
    spec = importlib.util.spec_from_file_location("_dc38", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recon_batches(subjects: list[ROIPairSubject], feats: dict, n_pairs: int | None, n_per_pair: int,
                  rng: np.random.Generator, pair_sampling: str = "tier") -> list:
    """trainer 의 [R] pass 와 같은 방식으로 (S[N,128,3], P[N,2], anatomy) 배치를 만든다.

    BN 재보정과 복원 RMSE 가 **학습이 실제로 보는 분포**를 써야 통계가 맞는다.
    n_pairs=None 이면 그 subject 의 pair 전부 (BN 재보정용 -- 통계는 넓게 모을수록 좋다).
    """
    out = []
    for sub in subjects:
        feat = feats[sub.sub]
        assert feat is not None and feat.ndim == 2 and feat.shape == (1, 512), (
            f"{sub.sub}: anatomy feature 가 [1,512] 가 아니다 ({None if feat is None else tuple(feat.shape)})")
        ks = (np.arange(sub.n_pairs) if n_pairs is None
              else sub.sample_pairs(n_pairs, rng, mode=pair_sampling))
        Ss, Ps = [], []
        for k in ks:
            S_gt, _ = sub.get_pair(int(k))
            take = min(n_per_pair, S_gt.shape[0])
            if take <= 0:
                continue
            idx = torch.as_tensor(rng.choice(S_gt.shape[0], take, replace=False))
            Ss.append(S_gt[idx])
            Ps.append(torch.as_tensor(sub.pair_ids[k]).repeat(take, 1))
        assert Ss, f"{sub.sub}: 복원 배치에 쓸 pair 가 없다 (bundles.npz 확인)"
        S = torch.cat(Ss).float()
        P = torch.cat(Ps)
        assert S.ndim == 3 and S.shape[1:] == (128, 3), S.shape
        assert S.shape[0] == P.shape[0] and P.shape[1] == 2, (S.shape, P.shape)
        assert torch.isfinite(S).all(), f"{sub.sub}: GT streamline 에 NaN/Inf"
        out.append((S, P, feat))
    assert out, "복원 배치를 만들 subject 가 없다"
    return out


@torch.no_grad()
def _recon_rmse(model, data, batch: int = 2048) -> float:
    """복원 RMSE (mm) = sqrt(mean_points ||rec - gt||^2). 모드(train/eval)는 호출측이 정한다.
    z 는 mu (샘플링 잡음 없이) -- trainer 의 값은 reparameterize 라 약간 더 크다."""
    tot, n = 0.0, 0
    for S, P, feat in data:
        for i in range(0, len(S), batch):
            s = S[i:i + batch].to(model.device)
            pr = P[i:i + batch].to(model.device)
            c = model.condition(feat, pr)
            mu, _ = model.encode_streamlines(s, c)
            rec = model.decode(mu, c, pr)
            assert rec.shape == s.shape, (rec.shape, s.shape)
            assert torch.isfinite(rec).all(), "복원에 NaN/Inf"
            tot += float(((rec - s) ** 2).sum(-1).sum())
            n += s.shape[0] * s.shape[1]
    assert n > 0, "복원 RMSE 를 잴 점이 하나도 없다"
    return float(np.sqrt(tot / n))


@torch.no_grad()
def recon_rmse_metrics(model, subjects: list[str], n_subj: int = 3, n_pairs: int = 128,
                       n_per_pair: int = 8, seed: int = 0, init_bundle: str = "AF_L",
                       source: str = T1_SOURCE_DEFAULT, batch: int = 2048,
                       pair_sampling: str = "tier") -> dict:
    """held-out 복원 RMSE (mm) 를 **eval 모드와 train 모드 둘 다** 낸다.

      recon_rmse_eval_mm   running 통계. **추론에서 실제로 나오는 값** -> 게이트는 이걸로 건다.
      recon_rmse_train_mm  배치 통계. trainer 로그와 같은 렌즈 (비교용으로만).

    train 모드 forward 는 no_grad 여도 running buffer 를 바꾸므로 재기 전에 스냅샷을 뜨고
    끝나면 되돌린다 (안 그러면 지표를 재는 행위가 모델을 바꾼다).
    """
    dc = _bn_utils()
    suffix = T1_SOURCES[source][0]
    rp = Path(__file__).resolve().parents[3] / "outputs" / "roi_pairs"
    ready = [s for s in subjects
             if (CACHE / f"{s}{suffix}").exists() and (rp / s / "bundles.npz").exists()][:n_subj]
    assert len(ready) >= 1, f"복원 RMSE 를 잴 subject 가 없다 (source={source})"
    subs = [ROIPairSubject(s) for s in ready]
    feats = {s.sub: anatomy_feature(model, s.sub, init_bundle, source=source) for s in subs}
    data = recon_batches(subs, feats, n_pairs, n_per_pair,
                         np.random.default_rng(seed), pair_sampling=pair_sampling)
    was_training = model.training
    st = dc.bn_state(model)                       # train 모드 pass 가 running 통계를 바꾼다
    model.eval()
    out = {"recon_rmse_eval_mm": _recon_rmse(model, data, batch)}
    model.train()
    out["recon_rmse_train_mm"] = _recon_rmse(model, data, batch)
    dc.bn_restore(model, st)
    out["recon_n_subj"] = len(subs)
    out["recon_n_streamlines"] = int(sum(len(S) for S, _, _ in data))
    out["recon_bn_gap_mm"] = out["recon_rmse_eval_mm"] - out["recon_rmse_train_mm"]
    assert all(np.isfinite(v) for v in (out["recon_rmse_eval_mm"], out["recon_rmse_train_mm"])), out
    model.train(was_training)
    return out


def run(phase: str, subjects: list[str], max_steps: int, out_dir: Path, cfg: TrainConfig | None = None,
        weights: LossWeights | None = None, init_bundle: str = "AF_L", device: str = "cuda",
        resume: Path | None = None, log_every: int = 1, trainable: str = "vae",
        unet_level: str | None = None, save_every: int = 200, resume_optimizer: bool = True,
        preempt_file: Path | None = None, in_channels: int = 2, template=None,
        t1_source: str = T1_SOURCE_DEFAULT, heartbeat: Path | None = None,
        step_hook=None, step_hook_every: int = 0) -> Path:
    """resume 가 같은 phase 의 `*_latest.pt` (optimizer/step/RNG 포함) 이면 그 step 부터 이어서 돈다.
    다른 phase 의 checkpoint 면 모델 가중치만 가져오고 step 0 부터 시작한다."""
    assert phase in PHASES, phase
    # 가속 (검증: scripts/38_decoder_capacity.py, base 조건 1500 step)
    #   fp32 RMSE 9.186 / 48s · tf32 9.188 / 39s · bf16 9.168 / 29s -> 수치 영향 0.2 %, 1.66배
    if cfg is not None and getattr(cfg, "tf32", False):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    if cfg is not None and getattr(cfg, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    assert max_steps > 0
    import nibabel as nib
    cfg = cfg or TrainConfig()
    cfg.active = set(PHASES[phase])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    subs = [ROIPairSubject(s) for s in subjects]
    n_roi = subs[0].n_roi
    model = ROIPairATM(n_roi=n_roi, init_bundle=init_bundle, device=device, trainable=trainable,
                       unet_level=unet_level, in_channels=in_channels, template=template,
                       count_local_dim=int(getattr(cfg, "count_local_dim", 0)),
                       cond_local_dim=int(getattr(cfg, "cond_local_dim", 0)),
                       pair_anchor=getattr(cfg, "pair_anchor", None),
                       anchor_alpha=float(getattr(cfg, "anchor_alpha", 0.0)))
    unet_level = model.unet_level
    start_step, opt_state, rng_state = 0, None, None
    if resume is not None:
        sd = torch.load(resume, map_location=device, weights_only=False)
        model.load_checkpoint(sd["model"])
        if resume_optimizer and sd.get("phase") == phase and "optimizer" in sd and sd.get("step", 0) < max_steps:
            start_step, opt_state, rng_state = int(sd["step"]), sd["optimizer"], sd.get("rng")
            print(f"[{phase}] resume: step {start_step} 부터 이어서 ({resume.name})", flush=True)
    img = nib.load(ATLAS)
    ea = EndpointAssigner(np.load(CACHE / "dist_maps.npy"), img.affine, tau=0.5, device=device,
                          d_bg=None if cfg.sc_mode == "endpoint" else 2.0)
    tr = Trainer(model, ea, cfg, weights)
    if opt_state is not None:
        tr.opt.load_state_dict(opt_state)
    if rng_state is not None:
        # RNG 상태 텐서는 map_location 때문에 GPU 로 올라와 있을 수 있다 -> set_* 는 CPU ByteTensor 만 받는다
        tr.rng.bit_generator.state = rng_state["np"]
        tr.gen.set_state(rng_state["torch_gen"].cpu())
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if device == "cuda" and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng_state["torch_cuda"].cpu())
    if unet_level == "none":
        feats = {s.sub: anatomy_feature(model, s.sub, init_bundle, source=t1_source) for s in subs}
    elif unet_level == "stage4":
        feats = {s.sub: stage3_cache(model, s.sub, init_bundle, source=t1_source) for s in subs}
    else:
        feats = {s.sub: None for s in subs}          # T1 을 step 마다 로드
    print(f"[{phase}] trainable={trainable} unet_level={unet_level} params={model.param_counts()}", flush=True)

    # --- P3: BN 재보정 (bn_mode == 'recal_eval') --------------------------------------
    # gradient 도 파라미터 추가도 없다. running 통계만 recon 분포로 다시 쌓는다.
    # 실측(W1-d, val 3명): eval 모드 8.458 -> 4.002 mm. 그 뒤 trainer 가 BN 을 eval 로 고정하므로
    # 학습과 추론이 같은 함수가 된다.
    if cfg.bn_mode == "recal_eval":
        dc = _bn_utils()
        t_bn = time.time()
        n_recal = min(6, len(subs))                       # 스윕과 같은 규모 (n_train=6)
        with torch.no_grad():
            rf = {s.sub: model.anatomy_forward(
                feats[s.sub] if feats[s.sub] is not None else t1_input(model, s.sub, t1_source))
                for s in subs[:n_recal]}
        rdata = recon_batches(subs[:n_recal], rf, None, cfg.n_gt_per_pair,
                              np.random.default_rng(cfg.seed + 7), pair_sampling=cfg.pair_sampling)
        n_bn = dc.bn_recalibrate(model, None, rdata, batch=cfg.chunk)
        n_str = int(sum(len(S) for S, _, _ in rdata))
        assert n_bn > 0 and n_str > 0, (n_bn, n_str)
        print(f"[{phase}] BN 재보정: {n_bn}개 BatchNorm1d, {n_recal}명 x "
              f"{n_str // n_recal} 가닥 ({time.time() - t_bn:.0f}s)", flush=True)

    log = open(out_dir / "log.jsonl", "a")
    rng = np.random.default_rng(cfg.seed + 1000)          # subject 선택용 (Trainer 내부 rng 와 분리)
    if rng_state is not None and "np_subject" in rng_state:
        rng.bit_generator.state = rng_state["np_subject"]

    def save(step, name):
        torch.save({"model": model.state_dict(), "optimizer": tr.opt.state_dict(), "step": step,
                    "rng": {"np": tr.rng.bit_generator.state, "np_subject": rng.bit_generator.state,
                            "torch_gen": tr.gen.get_state(), "torch_cpu": torch.get_rng_state(),
                            "torch_cuda": torch.cuda.get_rng_state() if device == "cuda" else None},
                    "cfg": vars(cfg) | {"active": sorted(cfg.active)}, "phase": phase, "steps": max_steps,
                    "trainable": trainable, "unet_level": unet_level,
                    "in_channels": model.in_channels, "template": model.template is not None,
                    "prior_use_anatomy": model.pair_emb.prior_use_anatomy,
                    "count_local_dim": (model.count_head.local_dim
                                       if model.count_head is not None else 0),
                    "cond_local_dim": model.pair_emb.local_dim,
                    "pair_anchor": bool(model.anchor is not None),
                    "anchor_alpha": (float(model.anchor.alpha)
                                     if model.anchor is not None else 0.0),
                    "t1_source": t1_source},           # 평가/추론이 같은 입력 프로토콜을 쓰게 한다
                   out_dir / name)

    t0 = time.time()
    for step in range(start_step + 1, max_steps + 1):
        s = subs[rng.integers(len(subs))]
        anat = lambda x: feats[x.sub] if feats[x.sub] is not None else t1_input(model, x.sub, t1_source)
        partner = None
        if getattr(tr.w, "diff", 0.0) > 0 or getattr(tr.w, "var", 0.0) > 0:
            # L_diff / L_var 는 한 step 에 서로 다른 subject 2명이 필요 (전략 문서 §5.1)
            assert len(subs) > 1, "diff 손실에는 subject 2명 이상 필요"
            k = int(rng.integers(len(subs) - 1))
            s2 = subs[k] if subs[k].sub != s.sub else subs[-1]
            partner = (s2, anat(s2))
        o = tr.step(s, anat(s), partner=partner, step=step)
        o.update(step=step, subject=s.sub, phase=phase, elapsed=time.time() - t0)
        log.write(json.dumps({k: (v if isinstance(v, (int, float, str)) else float(v)) for k, v in o.items()}) + "\n")
        log.flush()
        if step % log_every == 0:
            keys = [k for k in ("L_recon", "L_endpoint", "L_edge", "L_corr", "L_mag", "L_length",
                                "endpoint_pair_acc", "sc_r", "grad_norm_total") if k in o]
            print(f"[{phase}] step {step}/{max_steps} {s.sub} " +
                  " ".join(f"{k}={o[k]:.4f}" for k in keys) + f" ({o['step_sec']:.1f}s)", flush=True)
        if step_hook is not None and step_hook_every and step % step_hook_every == 0:
            step_hook(step, model)                # 계측 전용. 학습 상태(파라미터/optimizer/RNG)를 바꾸면 안 된다.
        if save_every and step % save_every == 0 and step < max_steps:
            save(step, f"{phase}_latest.pt")
        if preempt_file is not None and step % 10 == 0 and Path(preempt_file).exists():
            save(step, f"{phase}_latest.pt")
            log.close()
            print(f"[{phase}] preempt 요청 감지 -> step {step} 에서 양보", flush=True)
            raise Preempted(step)
        if heartbeat is not None:
            heartbeat.touch()
    ck = out_dir / f"{phase}_step{max_steps}.pt"
    save(max_steps, ck.name)
    latest = out_dir / f"{phase}_latest.pt"
    if latest.exists():
        latest.unlink()                           # 완료된 phase 의 중간 checkpoint 정리
    log.close()
    return ck
