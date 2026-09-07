"""T1 -> TRK -> SC 재현 평가 지표 (docs/T1_TRK_SC_REPRODUCTION_METRICS.md).

기존 `balance_metrics.py` 가 이미 제공하는 것: pair/whole-brain dice, coverage, overreach,
길이 분포 KS, 중복 비율, SC 확장 지표(Pearson/log/Spearman/CCC/RMSE/MAE/log-MAE/edge F1).

여기서는 문서가 요구하는데 없던 것만 채운다:
  §1-5 MDF (Minimum Direct-Flip)      streamline 모양 거리, 방향 뒤집힘 허용
  §1-6 Endpoint distance              양 끝점 위치 차이
  §1-7 Hausdorff                      최악의 국소 이탈 (평균은 비슷한데 일부가 크게 벗어난 경우)
  §1-8 Valid connection rate          의도한 ROI 쌍을 실제로 연결했는가
  §4-4 Wasserstein                    길이 분포 차이 (KS 보완)
  §3-1 LOO 잔차 상관 / §3-2 subject 간 유사도

주의: MDF/Hausdorff 는 N×M 쌍거리라 번들이 크면 비싸다. `max_n` 으로 표본을 제한한다.
"""
from __future__ import annotations

import numpy as np

from ..data.roi_groups import BLOCKS, N_CTX, TIERS, block_masks, tier_masks

EPS = 1e-9


# --- streamline 쌍거리 -------------------------------------------------------
def _pair_point_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A [N,P,3], B [M,P,3] -> [N,M,P] 대응 점 간 거리 (같은 점 수 가정)."""
    assert A.shape[1:] == B.shape[1:], (A.shape, B.shape)
    return np.linalg.norm(A[:, None] - B[None], axis=-1)


def mdf_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Minimum Direct-Flip 거리 [N,M] (mm).

    streamline 방향(시작/끝)은 임의이므로 정방향과 뒤집은 방향 중 작은 쪽을 쓴다.
    128점으로 재샘플된 우리 구조에 그대로 맞는다.
    """
    d = _pair_point_dist(A, B).mean(-1)
    f = _pair_point_dist(A, B[:, ::-1]).mean(-1)
    return np.minimum(d, f)


def hausdorff_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """대칭 Hausdorff 거리 [N,M] (mm). 최악의 국소 이탈을 잡는다."""
    D = np.linalg.norm(A[:, None, :, None, :] - B[None, :, None, :, :], axis=-1)   # [N,M,P,P]
    return np.maximum(D.min(3).max(2), D.min(2).max(2))


def bundle_distance(S_gen: np.ndarray, S_gt: np.ndarray, max_n: int = 48,
                    seed: int = 0, hausdorff: bool = True) -> dict:
    """번들 대 번들 거리. 각 생성 가닥에서 가장 가까운 GT 가닥까지의 거리를 평균한다
    (양방향 대칭). 대조군으로 GT 를 두 표본으로 나눈 값을 함께 재야 해석이 된다."""
    rng = np.random.default_rng(seed)
    a = S_gen[rng.choice(len(S_gen), min(max_n, len(S_gen)), replace=False)].astype(np.float32)
    b = S_gt[rng.choice(len(S_gt), min(max_n, len(S_gt)), replace=False)].astype(np.float32)
    assert len(a) and len(b), "빈 번들"
    M = mdf_matrix(a, b)
    out = {"mdf_mm": float((M.min(1).mean() + M.min(0).mean()) / 2)}
    if hausdorff:
        H = hausdorff_matrix(a, b)
        out["hausdorff_mm"] = float((H.min(1).mean() + H.min(0).mean()) / 2)
    return out


def endpoint_distance(S_gen: np.ndarray, S_gt: np.ndarray, max_n: int = 48, seed: int = 0) -> float:
    """양 끝점 위치 차이 (mm). 방향 뒤집힘을 허용하고 가장 가까운 GT 가닥과 비교한다."""
    rng = np.random.default_rng(seed)
    a = S_gen[rng.choice(len(S_gen), min(max_n, len(S_gen)), replace=False)]
    b = S_gt[rng.choice(len(S_gt), min(max_n, len(S_gt)), replace=False)]
    ea, eb = a[:, [0, -1]], b[:, [0, -1]]                       # [n,2,3]
    d = np.linalg.norm(ea[:, None] - eb[None], axis=-1).mean(-1)          # 정방향
    f = np.linalg.norm(ea[:, None] - eb[None, :, ::-1], axis=-1).mean(-1)  # 뒤집음
    D = np.minimum(d, f)
    return float((D.min(1).mean() + D.min(0).mean()) / 2)


# --- 연결 정확도 -------------------------------------------------------------
def valid_connection_rate(S: np.ndarray, pairs: np.ndarray, atlas, affine, n_roi: int) -> dict:
    """생성 가닥이 **의도한** ROI 쌍을 실제로 연결했는가 (§1-8).

    valid   양 끝점이 정확히 그 쌍
    partial 한쪽만 맞음
    invalid 둘 다 틀리거나 ROI 밖
    """
    from ..data.tt_io import point_labels
    assert len(S) == len(pairs), (len(S), len(pairs))
    P = S.shape[1]
    lab = point_labels(S.reshape(-1, 3).astype(np.float32), atlas, affine)
    a, b = lab[::P], lab[P - 1::P]                              # 1-based, 0 = ROI 밖
    i, j = pairs[:, 0] + 1, pairs[:, 1] + 1
    hit = ((a == i) & (b == j)) | ((a == j) & (b == i))
    one = (~hit) & ((a == i) | (a == j) | (b == i) | (b == j))
    return {"valid_conn": float(hit.mean()), "partial_conn": float(one.mean()),
            "invalid_conn": float((~hit & ~one).mean()),
            "endpoint_in_roi": float(((a > 0) & (b > 0)).mean())}


# --- 길이 분포 ---------------------------------------------------------------
def length_distribution(L_gen: np.ndarray, L_gt: np.ndarray) -> dict:
    """생성/GT tractogram 의 streamline 길이 분포 차이 (§4-4)."""
    from scipy.stats import ks_2samp, wasserstein_distance
    assert len(L_gen) and len(L_gt), "빈 길이 배열"
    return {"len_mean_diff_mm": float(L_gen.mean() - L_gt.mean()),
            "len_median_diff_mm": float(np.median(L_gen) - np.median(L_gt)),
            "len_wasserstein_mm": float(wasserstein_distance(L_gen, L_gt)),
            "len_ks": float(ks_2samp(L_gen, L_gt).statistic)}


# --- subject 특이성 ----------------------------------------------------------
def loo_center(X: np.ndarray) -> np.ndarray:
    """자기 자신을 뺀 나머지 평균으로 중심화 (§3-1).

    train 템플릿을 빼면 예측의 고정 편향(모든 subject 에 동일)이 남아 상관을 오염시킨다.
    실측: 오라클 정보를 다 줘도 -0.11 ~ -0.15 의 음수가 일정하게 나왔다.
    """
    n = len(X)
    assert n >= 3, f"LOO 중심화에는 subject 가 3명 이상 필요하다 (지금 {n})"
    return X - (X.sum(0) - X) / (n - 1)


def _corr(a, b):
    return np.nan if a.std() < 1e-12 or b.std() < 1e-12 else float(np.corrcoef(a, b)[0, 1])


# --- 계층별 상관 (문제 C1) ---------------------------------------------------
def stratified_corr(pred, gt, n_roi: int = 82, n_ctx: int = N_CTX) -> dict:
    """tier/block 내부 상관. 전체 pair 상관은 tier 간 배율차에 지배되므로 주지표가 아니다.

    실측(test 31명, T1 단독 generated 경로): 전체 pair r = 0.713 인데 tier 안으로 들어가면
    small 0.164 / mid 0.189 / large 0.624 로 붕괴한다. 전체 r 은 "큰 연결은 크다" 를 맞춘
    값이지 개인차를 읽은 값이 아니다 -- 그래서 주지표는 tier 내부 r (특히 small/mid) 이다.

    마스크는 data.roi_groups 것을 그대로 쓴다: tier_masks 는 GT>0 만 (0 인 edge 는 어느
    구간에도 없다), block_masks 는 off-diagonal 전부. n_ctx 기본값은 roi_groups.N_CTX(=66)
    이며 DesikanCortexPD25 82 ROI 의 실제 피질 라벨 수다 (ctx-ctx pair 2145 개).

    반환 r 은 선형, r_log 는 log1p 상관 (balance_metrics.sc_metrics 와 같은 정의).
    n 이 작은 셀은 r 이 불안정하므로 각 셀의 pair 수를 항상 함께 낸다.
    """
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    assert p.ndim == 2 and p.shape == g.shape and p.shape[0] == p.shape[1], (p.shape, g.shape)
    assert p.shape[0] == n_roi, f"n_roi 불일치: 배열 {p.shape[0]} vs 인자 {n_roi}"
    n_bad = int((~np.isfinite(p)).sum()) + int((~np.isfinite(g)).sum())
    assert n_bad == 0, f"pred/gt 에 NaN/Inf {n_bad} 개"

    iu = np.triu_indices(n_roi, 1)
    pu, gu = p[iu], g[iu]
    assert pu.size > 0, "off-diagonal 상삼각이 비었다"
    assert (gu > 0).any(), "GT 상삼각이 전부 0 -- 빈 커넥톰이다"
    lp, lg = np.log1p(np.clip(pu, 0, None)), np.log1p(np.clip(gu, 0, None))

    cells = {"all": np.ones(pu.shape, bool)}
    cells.update({k: v[iu] for k, v in tier_masks(g).items()})
    cells.update({k: v[iu] for k, v in block_masks(n_roi, n_ctx).items()})
    assert any(v.any() for v in cells.values()), "모든 마스크가 False"

    r, r_log, n = {}, {}, {}
    for k, m in cells.items():
        n[k] = int(m.sum())
        # 2 개 이하면 상관이 정의되지 않거나 항상 ±1 이라 의미가 없다 -> nan (0 으로 숨기지 않는다)
        r[k] = _corr(pu[m], gu[m]) if n[k] >= 3 else np.nan
        r_log[k] = _corr(lp[m], lg[m]) if n[k] >= 3 else np.nan
    assert n["all"] > 0
    assert sum(n[b] for b in BLOCKS) == n["all"], "block 이 상삼각을 분할하지 않는다"
    assert sum(n[t] for t in TIERS) == int((gu > 0).sum()), "tier 가 GT>0 edge 를 분할하지 않는다"

    return {"all": r["all"], "all_log": r_log["all"],
            "tier": {t: r[t] for t in TIERS}, "tier_log": {t: r_log[t] for t in TIERS},
            "block": {b: r[b] for b in BLOCKS}, "block_log": {b: r_log[b] for b in BLOCKS},
            "n": {k: n[k] for k in ("all", *TIERS, *BLOCKS)}}


def subject_specificity(P: np.ndarray, G: np.ndarray) -> dict:
    """P, G: [n_subj, n_edge]. §3-1 잔차 상관 + §3-2 subject 간 유사도."""
    assert P.shape == G.shape, (P.shape, G.shape)
    n = len(P)
    Pc, Gc = loo_center(P), loo_center(G)
    iu = np.triu_indices(n, 1)
    rs = [_corr(Pc[i], Gc[i]) for i in range(n)]
    # 예측이 모든 subject 에 대해 같으면(그룹 템플릿을 그대로 내는 경우) 중심화 후 분산이 0 이라
    # 상관이 정의되지 않는다. 이것은 "개인 변동이 전혀 없다" 는 뜻이므로 0 으로 보고하고
    # 별도 플래그를 남긴다 -- nan 으로 두면 로그에서 조용히 사라진다.
    degenerate = bool(np.all(~np.isfinite(rs)))
    return {"resid_r": 0.0 if degenerate else float(np.nanmean(rs)),
            "pred_degenerate": degenerate,
            "inter_subj_r_pred": float(np.corrcoef(P)[iu].mean()),
            "inter_subj_r_gt": float(np.corrcoef(G)[iu].mean()),
            "inter_subj_r_pred_log": float(np.corrcoef(np.log1p(P))[iu].mean()),
            "inter_subj_r_gt_log": float(np.corrcoef(np.log1p(G))[iu].mean()),
            "n_subjects": n}


def ablation_gap(r_own: float, r_shuf: float, r_zero: float,
                 own_r=None, shuf_r=None, zero_r=None) -> dict:
    """§3-3. 자기 T1 > 남의 T1 > T1=0 이 성립해야 T1 을 실제로 쓴다는 뜻이다.

    own_r/shuf_r/zero_r 에 subject 별 상관 배열을 주면 subject 단위 키를 **추가로** 낸다
    (부트스트랩 CI 게이트에 필요 -- 스칼라 평균만으로는 표본 분포를 만들 수 없다).
    셋 다 주거나 셋 다 생략한다. 기존 3-스칼라 호출은 반환이 완전히 그대로다.

    분산 0 인 subject 는 상관이 정의되지 않아 nan 이다 (학습 초기 count head 가 모든 edge 에
    같은 값을 내는 경우). nan 은 지우지 않고 그대로 실어 CI 쪽에서 몇 명이 빠졌는지 보이게 한다.
    """
    out = {"abl_own_r": float(r_own), "abl_shuf_r": float(r_shuf), "abl_zero_r": float(r_zero),
           "abl_gap": float(r_own - r_zero), "abl_ordered": bool(r_own > r_shuf > r_zero)}
    given = [x is not None for x in (own_r, shuf_r, zero_r)]
    assert all(given) or not any(given), "own_r/shuf_r/zero_r 은 셋 다 주거나 셋 다 생략한다"
    if not any(given):
        return out
    o, s_, z = (np.asarray(x, np.float64).ravel() for x in (own_r, shuf_r, zero_r))
    assert o.shape == s_.shape == z.shape, (o.shape, s_.shape, z.shape)
    n = int(o.size)
    assert n > 0, "subject 별 상관 배열이 비었다"
    assert np.isfinite(o).any(), "own_r 이 전부 nan -- 예측 분산이 0 이다"
    for name, v, sc in (("own", o, r_own), ("shuf", s_, r_shuf), ("zero", z, r_zero)):
        # 배열 평균(유한값만)이 스칼라와 어긋나면 서로 다른 것을 재고 있다는 뜻이다 -> 시끄럽게 실패
        f = v[np.isfinite(v)]
        m = float(f.mean()) if f.size else float("nan")
        assert (not np.isfinite(sc) and not np.isfinite(m)) or np.isclose(m, sc, atol=1e-9, rtol=0), \
            f"{name}: 배열 평균 {m} != 스칼라 {sc}"
    out.update(abl_n_subjects=n,
               abl_own_r_subj=[float(x) for x in o],
               abl_shuf_r_subj=[float(x) for x in s_],
               abl_zero_r_subj=[float(x) for x in z],
               abl_gap_own_zero_subj=[float(x) for x in (o - z)],
               abl_gap_own_shuf_subj=[float(x) for x in (o - s_)],
               abl_n_finite=int(np.isfinite(o - z).sum()))
    return out
