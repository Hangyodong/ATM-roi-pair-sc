"""Inference (pipeline §22): T1 -> ROI-pair streamline -> whole-brain tractogram -> SC.

GT 가 필요 없는 경로다. pair 후보는 (a) edge head 확률 > thr, (b) 평가용으로 GT 양성 pair 전체
중 하나로 고른다. 생성 tractogram 에서 **hard** 규칙으로 SC 를 만든다 (학습의 soft 와 별개):
  pass  : 통과 ROI 집합의 모든 쌍  (= .mat GT 정의)
  end   : 양 끝점                  (= sc_end 정의)
weight head 의 w_k 를 곱한 버전도 같이 낸다.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.tt_io import hard_sc


@torch.no_grad()
def select_pairs(model, anatomy, n_roi: int, thr: float = 0.5, chunk: int = 8192):
    """edge head 로 양성 pair 선택. (pairs [K,2], prob [K])"""
    iu = np.stack(np.triu_indices(n_roi, 1), 1)
    P = torch.as_tensor(iu, device=model.device)
    probs = torch.cat([torch.sigmoid(model.edge_logits(anatomy, P[i:i + chunk])) for i in range(0, len(P), chunk)])
    keep = probs > thr
    return iu[keep.cpu().numpy()], probs[keep].cpu().numpy()


@torch.no_grad()
def generate_tractogram(model, anatomy, pairs: np.ndarray, n_per_pair: int, chunk: int = 8192, seed: int = 0):
    """-> (streamlines [N,128,3] mm float32, weights [N], pairs_rep [N,2])"""
    g = torch.Generator(device=model.device); g.manual_seed(seed)
    S, w, pr = model.generate(anatomy, torch.as_tensor(pairs, device=model.device), n_per_pair, chunk=chunk, generator=g)
    return S.cpu().numpy().astype(np.float32), w.cpu().numpy(), pr.cpu().numpy()


def tractogram_sc(S: np.ndarray, w: np.ndarray, atlas: np.ndarray, affine: np.ndarray, n_roi: int):
    """생성 tractogram -> {pass, end} x {count, weighted} SC 와 mean length."""
    npts = np.full(len(S), S.shape[1], np.int64)
    flat = S.reshape(-1, 3)
    out = {}
    for mode in ("pass", "end"):
        W, Ssum = hard_sc(flat, npts, atlas, affine, n_roi, mode)
        L = np.divide(Ssum, W, out=np.zeros_like(Ssum), where=W > 0)
        out[mode] = {"sc": W.astype(np.float64), "len": L}
    # weighted count: streamline 마다 w_k 를 곱한 pass/end
    out["pass"]["sc_w"] = _weighted_sc(S, w, atlas, affine, n_roi, "pass")
    out["end"]["sc_w"] = _weighted_sc(S, w, atlas, affine, n_roi, "end")
    return out


def _weighted_sc(S, w, atlas, affine, n_roi, mode):
    """hard_sc 는 count 만 주므로 w 를 곱한 버전은 label 을 직접 계산한다."""
    from ..data.tt_io import point_labels, roi_visit_sets
    npts = np.full(len(S), S.shape[1], np.int64)
    lab = point_labels(S.reshape(-1, 3), atlas, affine)
    M = np.zeros((n_roi, n_roi), np.float64)
    if mode == "end":
        a, b = lab[::S.shape[1]], lab[S.shape[1] - 1::S.shape[1]]
        m = (a > 0) & (b > 0) & (a != b)
        np.add.at(M, (a[m] - 1, b[m] - 1), w[m]); np.add.at(M, (b[m] - 1, a[m] - 1), w[m])
    else:
        pk = roi_visit_sets(lab, npts)
        bnd = np.searchsorted(pk[:, 0], np.arange(len(S) + 1))
        for t in range(len(S)):
            rs = pk[bnd[t]:bnd[t + 1], 1]
            if len(rs) < 2:
                continue
            ii, jj = np.triu_indices(len(rs), 1)
            np.add.at(M, (rs[ii], rs[jj]), w[t]); np.add.at(M, (rs[jj], rs[ii]), w[t])
    return M


def save_trk(S: np.ndarray, path, ref_affine=None):
    """nibabel 만으로 .trk 저장 (mm, RAS). 헤더는 W 격자."""
    import nibabel as nib
    from ..spaces import W_AFFINE, W_SHAPE
    t = nib.streamlines.Tractogram(list(S), affine_to_rasmm=np.eye(4))
    hdr = nib.streamlines.trk.TrkFile.create_empty_header()
    hdr["voxel_to_rasmm"] = W_AFFINE.astype(np.float32); hdr["dimensions"] = np.array(W_SHAPE, np.int16)
    hdr["voxel_sizes"] = np.ones(3, np.float32); hdr["voxel_order"] = b"RAS"
    nib.streamlines.save(t, str(path), header=hdr)

@torch.no_grad()
def allocate_counts(model, anatomy, pairs: np.ndarray, total: int | None = None,
                    min_per_pair: int = 1, max_per_pair: int = 20000) -> np.ndarray:
    """pair 마다 몇 가닥을 만들지 예측한다 (끝점 기준 count head).

    GT 는 100만 가닥 중 49 %가 두 ROI 를 끝점으로 갖고 pair 당 1~10,150 개로 천차만별이다.
    지금까지처럼 pair 마다 같은 개수를 만들면 tractogram 밀도가 GT 와 전혀 다르다.
    total 을 주면 예측 비율을 유지한 채 총합을 그 값으로 맞춘다.
    """
    P = torch.as_tensor(np.asarray(pairs, np.int64), device=model.device)
    n = model.edge_log_counts_end(anatomy, P).exp().cpu().numpy()
    assert np.isfinite(n).all() and (n >= 0).all()
    if total is not None:
        n = n * (float(total) / max(n.sum(), 1e-9))
    n = np.clip(np.round(n), min_per_pair, max_per_pair).astype(np.int64)
    return n


def template_counts(template_end: np.ndarray, pairs: np.ndarray, total: int,
                    min_per_pair: int = 1) -> np.ndarray:
    """train 평균 sc_end 비율로 pair 당 생성 개수를 정한다.

    count_head_end 는 로그 공간에서 1.8배 과분산이라 전체 가닥의 94 %를 상위 10 % pair 에
    몰아주고 그 선택이 거의 무작위다(GT 배분과 상관 0.09). 실측(test 4명, 총 10만 가닥):
      균등 0.679 / 예측 0.567 / **템플릿 0.853** / GT 배분(오라클) 0.863
    즉 학습된 head 보다 train 평균 비율이 낫다. 어느 연결이 굵은지는 개인차가 작기 때문이다.
    """
    t = np.asarray(template_end, np.float64)
    assert t.ndim == 2 and t.shape[0] == t.shape[1], t.shape
    p = np.asarray(pairs, np.int64)
    n = np.maximum(t[p[:, 0], p[:, 1]], 0.0)
    assert n.sum() > 0, "템플릿에서 이 pair 들의 값이 전부 0"
    n = n * (float(total) / n.sum())
    return np.maximum(np.round(n), min_per_pair).astype(np.int64)


@torch.no_grad()
def generate_by_count(model, anatomy, pairs: np.ndarray, counts: np.ndarray, atlas, affine, n_roi: int,
                      batch: int = 20000, seed: int = 0, keep: bool = False, bank=None):
    """pair 별 counts 만큼 생성하고 SC 를 누적한다 (전부 메모리에 올리지 않는다).

    bank (inference.latent_bank.LatentBank) 를 주면 사전분포 대신 거기서 latent 를 뽑는다.
    bank 에 없는 pair 만 N(mu_pair, I) 로 채운다.
    -> (sc dict {pass,end} x {sc,sc_w,len}, n_total, streamlines | None)
    """
    from ..data.tt_io import hard_sc
    rep = np.repeat(np.asarray(pairs, np.int64), np.asarray(counts, np.int64), axis=0)
    assert len(rep) > 0, "생성할 가닥이 없음"
    W = {m: np.zeros((n_roi, n_roi), np.float64) for m in ("pass", "end")}
    S_sum = {m: np.zeros((n_roi, n_roi), np.float64) for m in ("pass", "end")}
    kept = []
    g = torch.Generator(device=model.device); g.manual_seed(seed)
    rs = np.random.default_rng(seed)
    n_hit = 0
    for i in range(0, len(rep), batch):
        blk = rep[i:i + batch]
        P = torch.as_tensor(blk, device=model.device)
        z = None
        if bank is not None:
            zb, hit = bank.sample(blk, rs)
            n_hit += int(hit.sum())
            z = model.sample_z(P, g)
            if hit.any():
                z = torch.where(torch.as_tensor(hit, device=model.device)[:, None],
                                torch.as_tensor(zb, device=model.device), z)
        S, w, _ = model.generate(anatomy, P, 1, generator=g, z=z)
        S = S.cpu().numpy().astype(np.float32)
        npts = np.full(len(S), S.shape[1], np.int64)
        for mode in ("pass", "end"):
            w_, s_ = hard_sc(S.reshape(-1, 3), npts, atlas, affine, n_roi, mode)
            W[mode] += w_; S_sum[mode] += s_
        if keep:
            kept.append(S)
    out = {}
    for mode in ("pass", "end"):
        L = np.divide(S_sum[mode], W[mode], out=np.zeros_like(S_sum[mode]), where=W[mode] > 0)
        out[mode] = {"sc": W[mode], "sc_w": W[mode].copy(), "len": L}     # 개수 자체가 SC (weight head 불필요)
    if bank is not None:
        out["bank_hit"] = n_hit / len(rep)
    return out, int(len(rep)), (np.concatenate(kept) if keep else None)
