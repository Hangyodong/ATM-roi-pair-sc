"""anatomy -> prior 통로(D2-c) 상한 프로브.

질문: T1 anatomy feature 가 latent 조건평균의 **subject 성분**을 예측하는가?

  - 예측된다  -> `prior_use_anatomy` 를 켜면 개인차가 생성 경로로 갈 수 있다.
  - 예측 안 된다 -> 통로를 켜도 0 이다. 구조가 아니라 정보의 문제로 확정된다.

배경. W2-b `subject_ceiling` 은 latent 조건평균 분산의 **16.9%** 가 subject 성분이라고 쟀다
(pair 내 subject 간 분산 6.70 / 조건평균 총분산 39.79). 그런데 지금 prior 는 pair 만 보므로
그 16.9% 를 원리적으로 못 맞춘다. 이 프로브는 그중 **anatomy 로 설명되는 몫**을 subject 단위
교차검증으로 잰다. S1-b 와 같은 사전등록 방식이고 타깃만 SC 잔차 -> latent 잔차로 바뀐 것이다.

핵심 설계
  - pair 평균은 **훈련 fold 의 subject 로만** 계산한다 (테스트 subject 가 pair 평균에 새면 상관이 부풀려진다).
  - 교차검증은 **subject 단위**다. 같은 subject 의 다른 pair 가 훈련/테스트에 갈라지면 누수다.
  - 대조군: 머리 크기 스칼라만. S1-b 에서 이게 학습된 96차 풀링(0.063)을 이겼다(0.102).

stage
  scan    subject 간 공유 pair 목록 확정 (bundles.npz 의 작은 멤버만 읽는다)
  encode  공유 pair 의 posterior mu 조건평균 + anatomy feature 캐시
  probe   ridge (subject 교차검증) 로 subject 잔차 예측 r
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROI_PAIRS = ROOT / "outputs" / "roi_pairs"
OUT_NPZ = ROOT / "outputs" / "eval" / "w4a_prior_anatomy_cache.npz"
OUT_JSON = ROOT / "outputs" / "eval" / "w4a_prior_anatomy.json"
LOCK = ROOT / "outputs" / "checkpoints" / "gpu.lock"


def subjects_of(split_files) -> list[str]:
    subs = []
    for f in split_files:
        subs += [s.strip() for s in open(ROOT / f) if s.strip()]
    assert len(subs) == len(set(subs)), "split 에 중복 subject"
    return subs


# --------------------------------------------------------------------------- scan
def scan(a) -> dict:
    """모든 subject 의 pair 목록을 읽어 공유 pair 를 고른다. streamlines 는 건드리지 않는다."""
    subs = subjects_of(a.splits.split(","))
    cnt: dict[tuple[int, int], int] = {}
    n_ok = 0
    for sub in subs:
        p = ROI_PAIRS / sub / "bundles.npz"
        if not p.exists():
            continue
        z = np.load(p)                       # lazy: 아래에서 부른 멤버만 압축 해제된다
        pid, off = z["pair_ids"], z["pair_offsets"]
        assert pid.ndim == 2 and pid.shape[1] == 2, pid.shape
        assert len(off) == len(pid) + 1, (len(off), len(pid))
        n = np.diff(off)
        for (i, j), c in zip(pid[n >= a.min_n], n[n >= a.min_n]):
            cnt[(int(i), int(j))] = cnt.get((int(i), int(j)), 0) + 1
        n_ok += 1
    assert n_ok >= 20, f"bundles.npz 가 있는 subject 가 너무 적다 ({n_ok})"

    need = int(np.ceil(a.min_frac * n_ok))
    shared = sorted([k for k, v in cnt.items() if v >= need], key=lambda k: -cnt[k])[:a.n_pairs]
    assert len(shared) >= 10, (
        f"공유 pair 가 {len(shared)}개뿐 (min_n={a.min_n}, min_frac={a.min_frac}). 조건을 낮춰라")
    print(f"subject {n_ok}명 / pair {len(cnt)}종 -> 공유 pair {len(shared)}개 "
          f"(>= {a.min_n} 가닥을 가진 subject {need}명 이상)", flush=True)
    print(f"  prevalence: max {cnt[shared[0]]}/{n_ok}  min {cnt[shared[-1]]}/{n_ok}", flush=True)
    return {"pairs": np.asarray(shared, np.int64), "n_scanned": n_ok, "need": need}


# --------------------------------------------------------------------------- encode
def encode(a, shared: np.ndarray) -> dict:
    import torch
    from atm_sc.data.dataset import ROIPairSubject
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.training.run import CACHE, T1_SOURCES, t1_input

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    t1_src = sd.get("t1_source", "rigid")
    suffix = T1_SOURCES[t1_src][0]
    subs = [s for s in subjects_of(a.splits.split(","))
            if (CACHE / f"{s}{suffix}").exists() and (ROI_PAIRS / s / "bundles.npz").exists()]
    assert len(subs) >= 30, f"T1 캐시 + bundles 를 모두 가진 subject 가 {len(subs)}명뿐"
    print(f"ckpt={a.ckpt} t1_source={t1_src} dev={dev} subjects={len(subs)} pairs={len(shared)}",
          flush=True)

    key = {(int(i), int(j)): p for p, (i, j) in enumerate(shared)}
    P, D = len(shared), 64
    M = np.full((len(subs), P, D), np.nan, np.float32)      # posterior mu 조건평균
    NS = np.zeros((len(subs), P), np.int64)                 # 조건별 가닥 수
    A = np.zeros((len(subs), 512), np.float32)              # anatomy feature
    SZ = np.zeros((len(subs), 2), np.float32)               # 머리 크기 대조군 (T1 / WM 부피)
    keep = []
    for si, sub in enumerate(subs):
        subj = ROIPairSubject(sub)
        pid = np.asarray(subj.pair_ids, np.int64)
        with torch.no_grad():
            x = t1_input(m, sub, t1_src)
            assert x.ndim == 5, x.shape
            SZ[si] = [float((x[0, 0] > 0.05).sum()), float(x[0, -1].sum())]
            feat = m.atm.encode_anatomy(x.to(dev))
            assert feat.shape == (1, 512) and torch.isfinite(feat).all(), feat.shape
            A[si] = feat[0].float().cpu().numpy()
            for k in range(len(pid)):
                p = key.get((int(pid[k, 0]), int(pid[k, 1])))
                if p is None:
                    continue
                S, _ = subj.get_pair(int(k))
                if S.shape[0] < a.min_n:
                    continue
                S = S.to(dev)
                pt = torch.as_tensor(pid[k][None], device=dev).repeat(S.shape[0], 1)
                mu, _ = m.encode_streamlines(S, m.condition(feat, pt))
                mu = mu.float().mean(0).cpu().numpy()
                assert mu.shape == (D,) and np.isfinite(mu).all(), f"{sub} pair {k}: {mu.shape}"
                M[si, p], NS[si, p] = mu, S.shape[0]
        got = int((NS[si] > 0).sum())
        keep.append(got == P)
        print(f"  [{si + 1}/{len(subs)}] {sub}: 공유 pair {got}/{P}", flush=True)

    keep = np.asarray(keep)
    assert keep.sum() >= 30, f"공유 pair 를 전부 가진 subject 가 {int(keep.sum())}명뿐"
    assert np.isfinite(A).all() and np.abs(A).max() > 0, "anatomy feature 가 비었다"
    out = {"M": M, "NS": NS, "A": A, "SZ": SZ, "keep": keep,
           "subjects": np.asarray(subs), "pairs": shared,
           "ckpt": np.asarray(a.ckpt), "t1_source": np.asarray(t1_src)}
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    print(f"캐시 저장: {OUT_NPZ}  M{M.shape}  전 pair 보유 subject {int(keep.sum())}명", flush=True)
    return out


# --------------------------------------------------------------------------- probe
def _ridge(X, Y, lam):
    """X [n,d] (절편 포함 안 함), Y [n,m] -> W [d,m]. 중심화는 호출부 책임."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def _fit_predict(Xtr, Ytr, Xte, lam):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    W = _ridge(Xtr - mx, Ytr - my, lam)
    return (Xte - mx) @ W + my


def probe(a, c) -> dict:
    """subject 단위 K-fold. pair 평균 제거도 fold 안에서 한다."""
    keep = c["keep"].astype(bool)
    M, A, SZ, subs = c["M"][keep], c["A"][keep], c["SZ"][keep], c["subjects"][keep]
    S, P, D = M.shape
    assert np.isfinite(M).all(), "선택된 subject 에 결측 pair 가 있다"

    # 예측 변수 집합
    Xs = {"anatomy512": A,
          "headsize2": SZ,
          "anatomy+size": np.concatenate([A, SZ], 1)}
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(S) % a.n_fold

    res = {}
    for name, X0 in Xs.items():
        X0 = (X0 - X0.mean(0)) / (X0.std(0) + 1e-8)
        best = None
        for lam in [float(v) for v in a.lams.split(",")]:
            pred = np.zeros_like(M)
            for f in range(a.n_fold):
                tr, te = fold != f, fold == f
                pm = M[tr].mean(0)                     # pair 평균: 훈련 subject 만
                Ytr = (M[tr] - pm).reshape(tr.sum(), -1)
                pred[te] = _fit_predict(X0[tr], Ytr, X0[te], lam).reshape(te.sum(), P, D) + pm
            # subject 잔차 상관: 각 pair 에서 pair 평균을 뺀 뒤 전체를 이어붙여 잰다
            pm_all = M.mean(0)
            yt, yp = (M - pm_all).ravel(), (pred - pm_all).ravel()
            r = float(np.corrcoef(yt, yp)[0, 1])
            r2 = float(1 - ((yt - yp) ** 2).sum() / (yt ** 2).sum())
            if best is None or r > best["r"]:
                best = {"lam": lam, "r": r, "r2": r2}
        # pair 별 상관 (subject 축으로) 분포
        pm_all = M.mean(0)
        res[name] = best
        print(f"  {name:<14} lam={best['lam']:<8g} 잔차 r={best['r']:+.4f}  R2={best['r2']:+.4f}",
              flush=True)

    # subject 성분이 실제로 얼마나 되는가 (W2-b subject_ceiling 을 175명으로 재측정)
    pm = M.mean(0)
    subj_var = float(((M - pm) ** 2).mean())
    tot_var = float(((M - M.mean((0, 1))) ** 2).mean())
    res["_variance"] = {"subject_component": subj_var, "total_condition_var": tot_var,
                        "subject_share": subj_var / tot_var,
                        "n_subjects": int(S), "n_pairs": int(P)}
    print(f"  subject 성분 비중 {subj_var / tot_var:.4f}  (subject {S}명 x pair {P}개)", flush=True)

    # **구조 제약**: prior_anatomy 는 Linear(cond_dim -> 2*latent_dim) 이라 출력이 pair 와 무관하다.
    # 즉 학습해도 subject 잔차 중 '전 pair 공통 offset' 성분밖에 표현하지 못한다. 그 몫을 잰다.
    R = M - pm
    o = R.mean(1)                                   # [S,D] 공통 offset
    share = float((np.repeat(o[:, None, :], P, 1) ** 2).mean() / (R ** 2).mean())
    ro = {}
    for name, X0 in Xs.items():
        X0 = (X0 - X0.mean(0)) / (X0.std(0) + 1e-8)
        best = -1.0
        for lam in [float(v) for v in a.lams.split(",")]:
            pred = np.zeros_like(o)
            for f in range(a.n_fold):
                tr, te = fold != f, fold == f
                pred[te] = _fit_predict(X0[tr], o[tr], X0[te], lam)
            best = max(best, float(np.corrcoef(o.ravel(), pred.ravel())[0, 1]))
        ro[name] = best
    res["_offset"] = {"common_offset_share_of_residual": share, "r_to_common_offset": ro}
    print(f"  subject 잔차 중 pair 무관 공통 성분 {share:.4f} "
          f"(나머지 {1 - share:.4f} 는 현재 prior_anatomy 구조로 표현 불가)", flush=True)
    for k, v in ro.items():
        print(f"    anatomy -> 공통 offset [{k:<13}] r = {v:+.4f}", flush=True)
    return res


# --------------------------------------------------------------------------- 경로 프로브 (W4-b)
PATHS_NPY = ROOT / "outputs" / "eval" / "w4b_group_paths.npy"
PATH_NPZ = ROOT / "outputs" / "eval" / "w4b_path_feats.npz"


def _flip_to(S: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """[n,128,3] 을 참조 곡선 방향으로 정렬. 뒤집힌 가닥을 평균하면 경로가 뭉개진다."""
    d0 = ((S - ref) ** 2).sum((1, 2))
    d1 = ((S[:, ::-1] - ref) ** 2).sum((1, 2))
    out = S.copy()
    out[d1 < d0] = S[d1 < d0][:, ::-1]
    return out


def build_paths(a) -> np.ndarray:
    """공유 pair 별 **그룹 평균 경로** [P,128,3] (mm). train subject 만 쓴다.

    이 경로는 '어디를 볼지'만 정한다 (subject 정보를 담지 않는다). 그래도 subject 라벨
    순열검정으로 통제한다 — 경로 정의가 샜다면 순열 귀무도 같이 올라간다.
    """
    from atm_sc.data.dataset import ROIPairSubject
    shared = np.load(ROOT / "outputs" / "eval" / "w4a_shared_pairs.npy")
    key = {(int(i), int(j)): p for p, (i, j) in enumerate(shared)}
    P = len(shared)
    acc = np.zeros((P, 128, 3), np.float64)
    cnt = np.zeros(P, np.int64)
    ref: list[np.ndarray | None] = [None] * P
    subs = subjects_of(["outputs/splits/train.txt"])
    for n, sub in enumerate(subs):
        if not (ROI_PAIRS / sub / "bundles.npz").exists():
            continue
        subj = ROIPairSubject(sub)
        pid = np.asarray(subj.pair_ids, np.int64)
        for k in range(len(pid)):
            p = key.get((int(pid[k, 0]), int(pid[k, 1])))
            if p is None:
                continue
            S = subj.get_pair(int(k))[0].numpy()
            if ref[p] is None:
                ref[p] = _flip_to(S, S[0]).mean(0)
            acc[p] += _flip_to(S, ref[p]).mean(0)
            cnt[p] += 1
        if (n + 1) % 20 == 0:
            print(f"  경로 누적 {n + 1}/{len(subs)}", flush=True)
    assert (cnt > 0).all(), f"경로를 못 만든 pair {int((cnt == 0).sum())}개"
    paths = (acc / cnt[:, None, None]).astype(np.float32)
    span = np.linalg.norm(paths[:, -1] - paths[:, 0], axis=1)
    assert np.isfinite(paths).all() and span.min() > 5.0, f"경로가 퇴화했다 (양끝 거리 min {span.min():.1f}mm)"
    print(f"그룹 평균 경로 {paths.shape}  양끝 거리 {span.min():.0f}~{span.max():.0f} mm  "
          f"(subject {int(cnt.min())}~{int(cnt.max())}명 평균)", flush=True)
    np.save(PATHS_NPY, paths)
    return paths


def path_feats(a, paths: np.ndarray) -> dict:
    """subject 별로 stage3 를 경로 위에서 샘플링 -> [S,P,256]. 끝점 ROI 대조군도 같이 만든다."""
    import torch
    from atm_sc.models.roi_atm import from_checkpoint
    from atm_sc.models.roi_pool import atlas_on_feature_grid
    from atm_sc.training.run import CACHE, T1_SOURCES, t1_input

    c0 = dict(np.load(OUT_NPZ, allow_pickle=True))
    subs = [str(s) for s in c0["subjects"]]
    shared = c0["pairs"]
    P = len(shared)
    assert paths.shape[0] == P, (paths.shape, P)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, sd = from_checkpoint(ROOT / a.ckpt, device=dev)
    t1_src = sd.get("t1_source", "rigid")
    _, faff = atlas_on_feature_grid(return_affine=True)
    inv = np.linalg.inv(np.asarray(faff, np.float64))
    fs = None

    # mm -> feature voxel index (nearest). 격자가 4mm 라 경로 128점에는 충분하다.
    hom = np.concatenate([paths.reshape(-1, 3), np.ones((P * 128, 1))], 1)      # [P*128,4]
    idx = np.rint((inv @ hom.T).T[:, :3]).astype(np.int64)

    X = np.zeros((len(subs), P, 256), np.float32)
    E = np.zeros((len(subs), P, 512), np.float32)                # 끝점 ROI 대조군
    for si, sub in enumerate(subs):
        with torch.no_grad():
            o3 = m.atm.cache_stage3(t1_input(m, sub, t1_src))[0]  # [256,49,58,49]
        if fs is None:
            fs = np.asarray(o3.shape[1:])
            assert (idx.min(0) >= -2).all() and (idx.max(0) <= fs + 1).all(), \
                f"경로가 feature 격자를 크게 벗어난다: {idx.min(0)}..{idx.max(0)} vs {fs}"
            print(f"stage3 {tuple(o3.shape)}  경로 인덱스 {idx.min(0)}..{idx.max(0)}", flush=True)
        ii = np.clip(idx, 0, fs - 1)
        f = o3[:, ii[:, 0], ii[:, 1], ii[:, 2]].float().cpu().numpy()          # [256, P*128]
        X[si] = f.reshape(256, P, 128).mean(2).T
        zr = np.load(ROOT / "outputs" / "cache" / "s1b_feats" / f"{sub}_rigid.npz")["f_roi"]
        E[si] = np.concatenate([zr[shared[:, 0]], zr[shared[:, 1]]], 1)
        if (si + 1) % 25 == 0:
            print(f"  [{si + 1}/{len(subs)}] {sub}", flush=True)
    assert np.isfinite(X).all() and np.abs(X).sum(2).min() > 0, "경로 feature 에 값이 전부 0 인 칸이 있다"
    out = {"X": X, "E": E, "subjects": np.asarray(subs), "pairs": shared}
    np.savez_compressed(PATH_NPZ, **out)
    print(f"저장: {PATH_NPZ}  X{X.shape}  E{E.shape}", flush=True)
    return out


def path_probe(a) -> dict:
    """pair 중심화한 feature -> pair 중심화한 latent 잔차. subject 단위 CV + 순열검정.

    S1-b 와 다른 점은 **어디서 풀링하느냐** 하나다: 끝점 ROI(E) vs 경로(X).
    """
    c0 = dict(np.load(OUT_NPZ, allow_pickle=True))
    pf = dict(np.load(PATH_NPZ, allow_pickle=True))
    assert [str(s) for s in c0["subjects"]] == [str(s) for s in pf["subjects"]], "subject 순서 불일치"
    keep = c0["keep"].astype(bool)
    M, X, E = c0["M"][keep], pf["X"][keep], pf["E"][keep]
    S, P, D = M.shape
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(S) % a.n_fold

    def run(F, lam, fold):
        """F [S,P,C] -> M [S,P,D]. pair 평균은 훈련 fold 로만 뺀다."""
        pred = np.zeros_like(M)
        for f in range(a.n_fold):
            tr, te = fold != f, fold == f
            fm, mm = F[tr].mean(0), M[tr].mean(0)                # [P,C], [P,D]
            Xtr = (F[tr] - fm).reshape(-1, F.shape[2])
            Ytr = (M[tr] - mm).reshape(-1, D)
            sd = Xtr.std(0) + 1e-8
            W = _ridge(Xtr / sd, Ytr, lam)
            pred[te] = ((F[te] - fm).reshape(-1, F.shape[2]) / sd @ W).reshape(te.sum(), P, D) + mm
        pm = M.mean(0)
        return float(np.corrcoef((M - pm).ravel(), (pred - pm).ravel())[0, 1])

    res = {}
    for name, F in [("path256", X), ("endpoint_roi512", E), ("path+endpoint", np.concatenate([X, E], 2))]:
        best = max(((run(F, lam, fold), lam) for lam in [float(v) for v in a.lams.split(",")]))
        res[name] = {"r": best[0], "lam": best[1]}
        print(f"  {name:<16} lam={best[1]:<8g} 잔차 r = {best[0]:+.4f}", flush=True)

    # 순열: subject 라벨을 섞어 feature-subject 대응만 끊는다
    lam = res["path256"]["lam"]
    obs = res["path256"]["r"]
    null = np.array([run(X[rng.permutation(S)], lam, fold) for _ in range(a.n_perm)])
    res["path256"]["null"] = {"mean": float(null.mean()), "sd": float(null.std()),
                              "q95": float(np.quantile(null, 0.95)), "max": float(null.max()),
                              "p": float((null >= obs).mean()), "n": int(a.n_perm)}
    print(f"  순열 {a.n_perm}회: mean {null.mean():+.4f} sd {null.std():.4f} "
          f"95%상한 {np.quantile(null, 0.95):+.4f} -> p = {(null >= obs).mean():.4f}", flush=True)
    return res


# --------------------------------------------------------------------------- W4-c 개인 변위
DEV_NPZ = ROOT / "outputs" / "eval" / "w4c_path_dev.npz"


def path_dev(a) -> dict:
    """subject x pair 의 **평균 경로 − 그룹 평균 경로** = 개인 변위 [S,P,128,3] (mm).

    W4-b 는 anatomy 에게 '이 연결이 얼마나 강한가' 를 물어 실패했다. 여기서는 '이 사람의 이
    다발이 그룹 평균에서 어디로 벗어나는가' 를 묻는다 — 국소 해부가 자연스럽게 결정할 법한 양이고,
    실제로 열려 있는 목표(dice)와 직결된다.
    """
    from atm_sc.data.dataset import ROIPairSubject
    c0 = dict(np.load(OUT_NPZ, allow_pickle=True))
    subs = [str(s) for s in c0["subjects"]]
    shared = c0["pairs"]
    gp = np.load(PATHS_NPY)                                  # [P,128,3] 그룹 평균 (방향 기준)
    key = {(int(i), int(j)): p for p, (i, j) in enumerate(shared)}
    P = len(shared)
    Dv = np.full((len(subs), P, 128, 3), np.nan, np.float32)
    for si, sub in enumerate(subs):
        subj = ROIPairSubject(sub)
        pid = np.asarray(subj.pair_ids, np.int64)
        for k in range(len(pid)):
            p = key.get((int(pid[k, 0]), int(pid[k, 1])))
            if p is None:
                continue
            S = subj.get_pair(int(k))[0].numpy()
            Dv[si, p] = _flip_to(S, gp[p]).mean(0) - gp[p]    # 그룹 경로를 방향 기준으로 쓴다
        if (si + 1) % 25 == 0:
            print(f"  [{si + 1}/{len(subs)}] {sub}", flush=True)
    assert np.isfinite(Dv).all(), f"결측 {int(np.isnan(Dv).any((2, 3)).sum())}칸"
    mag = np.linalg.norm(Dv, axis=3)                          # [S,P,128] 점별 변위 크기
    print(f"개인 변위 {Dv.shape}  |변위| 평균 {mag.mean():.2f} mm  중앙값 {np.median(mag):.2f}  "
          f"95분위 {np.quantile(mag, 0.95):.2f}", flush=True)
    np.savez_compressed(DEV_NPZ, D=Dv, subjects=np.asarray(subs), pairs=shared)
    print(f"저장: {DEV_NPZ}", flush=True)
    return {"D": Dv}


def dev_probe(a) -> dict:
    """경로 feature -> 개인 변위. W4-b 와 완전히 같은 프로토콜 (subject CV, fold 내 pair 중심화)."""
    c0 = dict(np.load(OUT_NPZ, allow_pickle=True))
    pf = dict(np.load(PATH_NPZ, allow_pickle=True))
    dv = dict(np.load(DEV_NPZ, allow_pickle=True))
    assert [str(s) for s in c0["subjects"]] == [str(s) for s in dv["subjects"]], "subject 순서 불일치"
    keep = c0["keep"].astype(bool)
    X, E, A = pf["X"][keep], pf["E"][keep], c0["A"][keep]
    Y = dv["D"][keep].reshape(keep.sum(), -1, 128 * 3)        # [S,P,384]
    S, P, _ = Y.shape
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(S) % a.n_fold

    def run(F, lam, fld):
        pred = np.zeros_like(Y)
        for f in range(a.n_fold):
            tr, te = fld != f, fld == f
            fm, ym = F[tr].mean(0), Y[tr].mean(0)
            Xtr = (F[tr] - fm).reshape(-1, F.shape[2])
            sd = Xtr.std(0) + 1e-8
            W = _ridge(Xtr / sd, (Y[tr] - ym).reshape(-1, Y.shape[2]), lam)
            pred[te] = (((F[te] - fm).reshape(-1, F.shape[2]) / sd) @ W).reshape(te.sum(), P, -1) + ym
        pm = Y.mean(0)
        return float(np.corrcoef((Y - pm).ravel(), (pred - pm).ravel())[0, 1])

    lams = [float(v) for v in a.lams.split(",")]
    Ab = np.repeat(A[:, None, :], P, 1)                       # 전역 anatomy 를 pair 축으로 복제
    res = {}
    for name, F in [("path256", X), ("endpoint_roi512", E), ("anatomy512_global", Ab),
                    ("path+endpoint", np.concatenate([X, E], 2))]:
        r, lam = max(((run(F, l, fold), l) for l in lams))
        res[name] = {"r": r, "lam": lam}
        print(f"  {name:<18} lam={lam:<8g} 변위 예측 r = {r:+.4f}", flush=True)

    obs, lam = res["path256"]["r"], res["path256"]["lam"]
    null = np.array([run(X[rng.permutation(S)], lam, fold) for _ in range(a.n_perm)])
    res["path256"]["null"] = {"mean": float(null.mean()), "sd": float(null.std()),
                              "q95": float(np.quantile(null, 0.95)), "max": float(null.max()),
                              "p": float((null >= obs).mean()), "n": int(a.n_perm)}
    print(f"  순열 {a.n_perm}회: mean {null.mean():+.4f} sd {null.std():.4f} "
          f"95%상한 {np.quantile(null, 0.95):+.4f} -> p = {(null >= obs).mean():.4f}", flush=True)
    return res


def main(a):
    if a.stage in ("scan", "all"):
        sc = scan(a)
        np.save(ROOT / "outputs" / "eval" / "w4a_shared_pairs.npy", sc["pairs"])
    if a.stage in ("encode", "all"):
        shared = np.load(ROOT / "outputs" / "eval" / "w4a_shared_pairs.npy")
        encode(a, shared)
    if a.stage in ("probe", "all"):
        c = dict(np.load(OUT_NPZ, allow_pickle=True))
        out = {"cmd": " ".join(sys.argv), "ckpt": str(c["ckpt"]), "probe": probe(a, c)}
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"저장: {OUT_JSON}", flush=True)
    if a.stage == "paths":
        build_paths(a)
    if a.stage == "pathfeat":
        path_feats(a, np.load(PATHS_NPY))
    if a.stage == "pathdev":
        path_dev(a)
    if a.stage == "devprobe":
        out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        out["dev_probe"] = dev_probe(a)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"저장: {OUT_JSON}", flush=True)
    if a.stage == "pathprobe":
        out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        out["path_probe"] = path_probe(a)
        out["path_cmd"] = " ".join(sys.argv)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"저장: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["scan", "encode", "probe", "all", "paths", "pathfeat", "pathprobe", "pathdev", "devprobe"])
    ap.add_argument("--ckpt", default="outputs/checkpoints/retrain/d3_joint/d3_joint_step3000.pt")
    ap.add_argument("--splits", default="outputs/splits/train.txt,outputs/splits/test.txt")
    ap.add_argument("--n-pairs", type=int, default=48, help="공유 pair 상한")
    ap.add_argument("--min-n", type=int, default=40, help="이보다 가닥이 적은 pair 는 제외")
    ap.add_argument("--min-frac", type=float, default=0.98, help="이 비율 이상 subject 가 가진 pair 만")
    ap.add_argument("--n-fold", type=int, default=5)
    ap.add_argument("--lams", default="1,10,100,1000,10000")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
