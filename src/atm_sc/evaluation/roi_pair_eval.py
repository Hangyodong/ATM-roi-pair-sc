"""Phase 2-3 평가 (pipeline §11, §29 ROI pair / TRK 항목).

질문: "ROI_i ↔ ROI_j 조건에서 그 edge 의 streamline 을 만들 수 있는가?"
생성 streamline 의 양 끝점을 **hard** atlas lookup 으로 ROI 에 배정해 (학습에 쓴 soft 가 아니라)
조건으로 준 pair 와 비교한다. GT bundle 과는 끝점 거리·길이를 비교한다.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.tt_io import point_labels


def _route_metrics(S, subject, pair_idx, n_per_pair, atlas, affine, n_roi, thr=0.2):
    """생성 streamline 이 지나간 ROI 집합 vs GT pair marginal (ROUTE 전략 §59, §64).

    pair 마다 GT 에서 thr 이상의 streamline 이 지나는 ROI 를 정답 집합으로 보고,
    생성 streamline 에서 thr 이상이 지나는 ROI 집합과 비교한다.
    """
    if not getattr(subject, "has_visitation", False):
        return {}
    marg = np.asarray(subject.pair_marginal, np.float32)[np.asarray(pair_idx)]
    lab = point_labels(S.reshape(-1, 3), atlas, affine).astype(np.int64).reshape(len(S), -1)
    v = np.zeros((len(S), n_roi), np.float32)
    rows = np.repeat(np.arange(len(S)), lab.shape[1]); keep = lab.ravel() > 0
    v[rows[keep], lab.ravel()[keep] - 1] = 1.0
    pm = v.reshape(len(pair_idx), n_per_pair, n_roi).mean(1)
    p, t = pm >= thr, marg >= thr
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return {"route_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "route_recall": float(tp / max(tp + fn, 1)), "route_precision": float(tp / max(tp + fp, 1)),
            "route_pred_visits": float(p.sum(1).mean()), "route_gt_visits": float(t.sum(1).mean())}


@torch.no_grad()
def evaluate_pairs(model, subject, anatomy, atlas, affine, pair_idx, n_per_pair=16, seed=0):
    """pair_idx: subject.pair_ids 의 인덱스 배열. dict 로 지표 반환."""
    R = subject.n_roi
    pairs = torch.as_tensor(np.asarray(subject.pair_ids)[pair_idx], device=model.device)
    g = torch.Generator(device=model.device); g.manual_seed(seed)
    S, w, pr = model.generate(anatomy, pairs, n_per_pair, generator=g)
    S = S.cpu().numpy(); pr = pr.cpu().numpy()
    start = point_labels(S[:, 0], atlas, affine).astype(int) - 1
    end = point_labels(S[:, -1], atlas, affine).astype(int) - 1
    route = _route_metrics(S, subject, pair_idx, n_per_pair, atlas, affine, R)
    a, b = pr[:, 0], pr[:, 1]
    start_ok = (start == a) | (start == b)
    end_ok = (end == a) | (end == b)
    pair_ok = ((start == a) & (end == b)) | ((start == b) & (end == a))
    bg = (start < 0) | (end < 0)
    L = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)

    # GT 와 비교: pair 별 GT 끝점 중심까지 거리, 길이
    d_end, len_gt = [], []
    for k, gi in enumerate(pair_idx):
        G, Lg = subject.get_pair(int(gi)); G = G.numpy()
        ends_gt = np.concatenate([G[:, 0], G[:, -1]])          # 두 끝점 모두
        sel = S[k * n_per_pair:(k + 1) * n_per_pair]
        for s in sel:
            d0 = np.linalg.norm(ends_gt - s[0], axis=1).min(); d1 = np.linalg.norm(ends_gt - s[-1], axis=1).min()
            d_end.append(0.5 * (d0 + d1))
        len_gt.append(float(Lg.mean()))
    len_gt = np.repeat(len_gt, n_per_pair)
    return {**route, "n_pairs": len(pair_idx), "n_streamlines": len(S),
            "start_roi_acc": float(start_ok.mean()), "end_roi_acc": float(end_ok.mean()),
            "pair_acc_unordered": float(pair_ok.mean()), "endpoint_in_background": float(bg.mean()),
            "endpoint_dist_to_gt_mm": float(np.mean(d_end)),
            "length_mean_mm": float(L.mean()), "gt_length_mean_mm": float(len_gt.mean()),
            "length_abs_err_mm": float(np.abs(L - len_gt).mean()),
            "w_mean": float(w.mean())}
