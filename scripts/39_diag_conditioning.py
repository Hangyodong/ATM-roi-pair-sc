#!/usr/bin/env python
"""왜 test 예측 SC 가 subject 마다 동일한가 — 조건화 경로를 단계별로 잘라 잰다.

  python scripts/39_diag_conditioning.py [--ckpt ...] [--n 12] [--n-pool 6]

측정 (PIPELINE_06_FINDINGS.md §B-6):
  1  anatomy feature a 의 subject 간 상관/코사인
  2  count head 1층에서 anatomy 기여 vs pair 기여 크기 (평균과 **분산**을 나눠 본다)
  3  템플릿 인수분해의 편차항 f = log_count - log_template 의 pair 변동 vs subject 변동
  4  최종 예측 SC 의 subject 간 상관 (GT 와 비교)
  5  개인차가 어디서 사라지는가: stage3 공간맵 -> global_avg_pool -> fc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                        # noqa: E402
from atm_sc.models.roi_atm import from_checkpoint                     # noqa: E402
from atm_sc.training.run import t1_input                              # noqa: E402


def mean_off(M) -> float:
    j = np.triu_indices(len(M), 1)
    return float(np.asarray(M)[j].mean())


def rel_spread(X) -> float:
    """그룹 평균을 뺀 노름 / 그룹 평균 노름. 개인차가 신호의 몇 배인가."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    return float(np.linalg.norm(X - mu, axis=1).mean() / (np.linalg.norm(mu) + 1e-12))


def head_diagnostics(m, subs, src):
    R = int(m.n_roi)
    iu = np.triu_indices(R, 1)
    P = torch.as_tensor(np.stack(iu, 1).astype(np.int64), device=m.device)
    K = P.shape[0]
    assert m.count_head is not None, "count head 없는 checkpoint 는 이 진단 대상이 아니다"

    A, Fdev, LOG = [], [], []
    with torch.no_grad():
        e = m.pair_emb.pair_vec(P)                                   # subject 무관
        for s in subs:
            a = m.atm.encode_anatomy(t1_input(m, s, src))
            assert float(a.norm()) > 1e-3, f"{s}: anatomy 가 0 에 가깝다"
            A.append(a.squeeze(0).double().cpu().numpy())
            f = m.count_head.net(torch.cat([a.expand(K, -1), e], -1)).squeeze(-1)
            Fdev.append(f.double().cpu().numpy())
            LOG.append(m.edge_log_counts(a, P).double().cpu().numpy())
        assert m.count_head.template_log is not None, "템플릿 인수분해가 꺼진 checkpoint"
        tl = m.count_head.template_log[iu[0], iu[1]].double().cpu().numpy()
        W = m.count_head.net[0].weight.detach().double().cpu()
        Wa, We = W[:, : m.count_head.anatomy_dim], W[:, m.count_head.anatomy_dim:]
        He = (e.double().cpu() @ We.T)
        Ha = torch.stack([torch.as_tensor(a) @ Wa.T for a in A]).numpy()

    A, Fdev, LOG = np.stack(A), np.stack(Fdev), np.stack(LOG)
    na, ne = np.linalg.norm(Ha, axis=1), He.norm(dim=1).numpy()
    An = A / np.linalg.norm(A, axis=1, keepdims=True)

    print(f"\n[1] anatomy feature a (S={len(subs)}, dim={A.shape[1]})")
    print(f"  ||a|| mean {np.linalg.norm(A, axis=1).mean():.4f}   "
          f"subject 간 상관 {mean_off(np.corrcoef(A)):.6f}   코사인 {mean_off(An @ An.T):.6f}")
    print("\n[2] count head 1층 기여 (평균은 커졌는데 분산이 안 커진 것이 핵심)")
    print(f"  ||W_a a|| {na.mean():.4f}   ||W_e e|| {ne.mean():.4f}   비율 {na.mean() / ne.mean():.4f}")
    print(f"  anatomy 기여의 subject 변동 ||W_a(a_i - a_mean)|| {np.linalg.norm(Ha - Ha.mean(0), axis=1).mean():.4f}")
    print("\n[3] 편차항 f = log_count - log_template")
    print(f"  pair 간 std {Fdev.mean(0).std():.4f}   같은 pair 의 subject 간 std {Fdev.std(0).mean():.6f}"
          f"   f 의 subject 간 상관 {mean_off(np.corrcoef(Fdev)):.6f}")
    print("\n[4] 최종 예측 log_count = log_template + f")
    print(f"  pair 간 std: template {tl.std():.4f} -> {LOG.mean(0).std():.4f}   "
          f"subject 간 std {LOG.std(0).mean():.6f} (전체의 {100 * LOG.std(0).mean() / LOG.mean(0).std():.3f}%)")
    G = np.stack([np.asarray(ROIPairSubject(s).sc_mat, np.float64)[iu] for s in subs])
    assert np.isfinite(G).all() and (G.sum(1) > 0).all(), "GT SC 가 비었거나 NaN"
    print(f"  예측 SC subject 간 상관 {mean_off(np.corrcoef(np.exp(LOG))):.6f}   "
          f"GT {mean_off(np.corrcoef(G)):.6f} (log1p {mean_off(np.corrcoef(np.log1p(G))):.6f})")


def pooling_diagnostics(m, subs, src):
    """개인차가 stage3 -> pooling -> fc 중 어디서 사라지는가."""
    u = m.atm.net.unet
    O3, POOL, FC = [], [], []
    with torch.no_grad():
        for s in subs:
            x = t1_input(m, s, src).to(m.device)
            o3 = m.atm._unet_stage3(u, x)
            c41 = F.relu(u.conv4_1(o3))
            o4 = c41 + F.relu(u.conv4_3(u.dropout4(F.relu(u.conv4_2(c41)))))
            g = u.global_avg_pool(o4).view(1, -1)
            a = u.fc(g)
            O3.append(o3.float().cpu().numpy().ravel())
            POOL.append(g.double().cpu().numpy().ravel())
            FC.append(a.double().cpu().numpy().ravel())
            del o3, o4, c41, g, a, x
            torch.cuda.empty_cache()
    O3, POOL, FC = np.stack(O3), np.stack(POOL), np.stack(FC)
    print(f"\n[5] 개인차 소멸 지점 (S={len(subs)})")
    for name, X in (("stage3 공간맵", O3), ("global_avg_pool [512]", POOL), ("fc -> anatomy", FC)):
        print(f"  {name:24s} subject 간 상관 {mean_off(np.corrcoef(X)):.6f}   rel_spread {rel_spread(X):.4f}")
    X = O3.reshape(len(subs), 256, -1)
    print(f"  pooling 이 버리는 공간 편차 성분   rel_spread {rel_spread((X - X.mean(2, keepdims=True)).reshape(len(subs), -1)):.4f}")
    print(f"  pooling 이 남기는 채널 평균 성분   rel_spread {rel_spread(X.mean(2)):.4f}")


def main(a):
    m, sd = from_checkpoint(Path(a.ckpt), device="cuda")
    src = sd.get("t1_source", "rigid")
    print(f"ckpt={Path(a.ckpt).name} unet_level={sd.get('unet_level')} in_ch={sd.get('in_channels')} "
          f"template={sd.get('template')} t1_source={src}")
    subs = [s.strip() for s in open(a.split) if s.strip()]
    assert len(subs) >= 3, f"subject 3명 이상 필요 (now {len(subs)})"
    head_diagnostics(m, subs[: a.n], src)
    if a.n_pool:
        pooling_diagnostics(m, subs[: a.n_pool], src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/checkpoints/retrain/p4_joint/p4_joint_step3000.pt"))
    ap.add_argument("--split", default=str(ROOT / "outputs/splits/test.txt"))
    ap.add_argument("--n", type=int, default=12, help="head 진단에 쓸 subject 수")
    ap.add_argument("--n-pool", type=int, default=6, help="pooling 진단 subject 수 (0 이면 생략, subject 당 143 MB)")
    main(ap.parse_args())
