"""Trainer 통합: segment 분기 + edge count head (EDGE_ALIGNED 전략 §13-22)."""
import numpy as np
import torch

from atm_sc.data.segment_sampler import SegmentBalanceConfig
from atm_sc.models.endpoint_assigner import EndpointAssigner
from atm_sc.models.roi_atm import ROIPairATM
from atm_sc.training.synthetic import make_synthetic
from atm_sc.training.trainer import LossWeights, TrainConfig, Trainer


class _WithSegments:
    """make_synthetic 의 subject 에 edge segment 인터페이스를 붙인다 (직선 bundle 을 그대로 segment 로)."""

    def __init__(self, subj, n_points=128):
        self._s = subj
        self.edge_pair_ids = np.asarray(subj.pair_ids, np.int16)
        self._segs = [b[:, ::4] for b in subj._bundles]          # 128 -> 32 점
        self.edge_count_full = np.array([len(b) * 7 for b in subj._bundles], np.int32)
        self.has_edge_segments = True
        self.sub = getattr(subj, "sub", "synthetic")

    def __getattr__(self, k):
        return getattr(self._s, k)

    def edge_segments(self, e):
        return torch.from_numpy(np.ascontiguousarray(self._segs[e])).float()

    def edge_segment_lengths(self, e):
        return torch.full((len(self._segs[e]),), 30.0)


def _setup(active, **kw):
    subj, atlas, affine, dist, lo, hi = make_synthetic(n_roi=6, n_pos_pairs=3, n_per_pair=8, seed=0)
    subj.set_visitation(atlas, affine)
    s = _WithSegments(subj)
    m = ROIPairATM(n_roi=6, coord_min=lo, coord_max=hi, device="cpu")
    ea = EndpointAssigner(dist, affine, tau=0.5, device="cpu", d_bg=2.0)
    cfg = TrainConfig(sc_mode="pass", n_gen_per_pair=2, n_gt_per_pair=4, gt_pairs_per_step=3,
                      neg_pairs_per_step=4, chunk=64, route_tau=1.0, active=set(active),
                      seg_edges_per_step=3, n_seg_per_edge=4,
                      segment_balance=SegmentBalanceConfig(enabled=True, n_points=128, min_segments=2),
                      lr_dec=1e-3, lr_heads=1e-3, seed=0, **kw)
    return s, Trainer(m, ea, cfg, LossWeights())


def test_segment_and_count_losses_present():
    s, tr = _setup({"recon", "endpoint", "corr", "segment", "count"})
    o = tr.step(s, torch.randn(1, 512) * 0.004)
    for k in ("L_seg_recon", "L_seg_kl", "L_seg_geom", "L_seg_endpoint", "seg_endpoint_pair_acc",
              "L_count", "count_r", "count_log_mae", "seg_edges", "gnorm_after_S", "gnorm_after_C"):
        assert k in o, (k, sorted(o))
    bad = [k for k, v in o.items() if isinstance(v, float) and not np.isfinite(v)]
    # count_r 은 step 0 에서 모든 edge 예측이 같아(상수) 정의되지 않는다 -> 지표만 nan 허용
    assert not [k for k in bad if k.startswith(("L_", "gnorm", "seg_", "dLda"))], bad
    assert o["seg_edges"] == 3 and 0.99 <= o["seg_block_ctx-ctx"] <= 1.0   # 합성 atlas 는 전부 피질 취급


def test_count_head_learns_absolute_scale():
    """총합 정규화 magnitude 와 달리 count head 는 절대값을 맞춘다."""
    s, tr = _setup({"count"})
    a = torch.randn(1, 512) * 0.004
    first = tr.step(s, a)["L_count"]
    for _ in range(30):
        o = tr.step(s, a)
    assert o["L_count"] < first * 0.8, (first, o["L_count"])
    assert o["count_log_mae"] < 1.0


def test_segment_recon_improves():
    s, tr = _setup({"segment"})
    a = torch.randn(1, 512) * 0.004
    first = np.mean([tr.step(s, a)["L_seg_recon"] for _ in range(3)])
    last = np.mean([tr.step(s, a)["L_seg_recon"] for _ in range(15)][-3:])
    assert last < first, (first, last)


def test_segment_mode_does_not_change_full_streamline_path_at_init():
    """mode 임베딩은 0-init 이라 처음에는 full/segment 조건이 같아야 한다."""
    s, tr = _setup({"recon"})
    m = tr.model
    a = torch.randn(1, 512) * 0.004
    P = torch.tensor([[0, 1], [2, 3]])
    assert torch.allclose(m.condition(a, P, mode=0), m.condition(a, P, mode=1))
    assert torch.allclose(m.prior_mean(P, mode=0), m.prior_mean(P, mode=1))


def test_count_head_in_heads_group_and_checkpoint_tolerant():
    s, tr = _setup({"count"})
    m = tr.model
    ids = {id(p) for p in m.param_groups()["heads"]}
    assert all(id(p) in ids for p in m.count_head.parameters())
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("count_head")}
    r = m.load_checkpoint(sd)                       # 구 checkpoint 에서 이어받기
    assert r["missing"] and not r["unexpected"]
