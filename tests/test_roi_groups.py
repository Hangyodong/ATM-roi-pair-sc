"""ctx/sub block · 소/중/대 tier 균형 (docs/ATM_FINAL_FINETUNING_REPORT.md §14)."""
import numpy as np
import torch

from atm_sc import losses as L
from atm_sc.data.roi_groups import (BLOCKS, TIER_EDGES, balanced_choice, block_masks, block_of_pairs,
                                    tier_masks, tier_of_strength)


def _gt(R=82, seed=0):
    rng = np.random.default_rng(seed)
    g = np.triu(rng.lognormal(3, 2, (R, R)), 1)
    return torch.as_tensor(g + g.T, dtype=torch.float32)


def _masks(R=82):
    return {k: torch.as_tensor(v) for k, v in block_masks(R).items()}


def test_block_masks_partition_82():
    m = block_masks(82); iu = np.triu_indices(82, 1)
    assert [int(m[b][iu].sum()) for b in BLOCKS] == [2145, 1056, 120]
    assert all((v == v.T).all() and not v.diagonal().any() for v in m.values())


def test_block_and_tier_indexing():
    assert block_of_pairs(np.array([[0, 1], [0, 70], [70, 71]])).tolist() == [0, 1, 2]
    t = tier_of_strength(np.array([1, TIER_EDGES[0], TIER_EDGES[0] + 1, TIER_EDGES[1], TIER_EDGES[1] + 1]))
    assert t.tolist() == [0, 0, 1, 1, 2]
    g = _gt().numpy(); tm = tier_masks(g); iu = np.triu_indices(82, 1)
    assert sum(int(v[iu].sum()) for v in tm.values()) == int((g[iu] > 0).sum())


def test_balanced_choice_equal_share_per_tier():
    rng = np.random.default_rng(0)
    tiers = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = balanced_choice(tiers, 300, rng)
    assert np.bincount(tiers[idx], minlength=3).tolist() == [100, 100, 100]
    assert len(set(idx[tiers[idx] == 2].tolist())) == 10          # 작은 그룹은 복원 추출로 채움


def test_group_corr_loss_sees_small_block():
    G = _gt(); masks = _masks()
    P = G.clone(); sub = masks["sub-sub"]
    vals = P[sub]; P[sub] = vals[torch.randperm(len(vals), generator=torch.Generator().manual_seed(0))]
    whole = L.sc_corr_loss(P, G)                                 # raw whole-brain: sub-sub 를 뒤섞어도 거의 0
    grp, rs = L.sc_corr_group_loss(P, G, masks, log=True)
    assert float(whole) < 0.02
    assert float(grp) > 0.2 and rs["ctx-ctx"] > 0.99 and rs["sub-sub"] < 0.5
    ident, rs2 = L.sc_corr_group_loss(G, G, masks, log=True)
    assert float(ident) < 1e-4 and all(r > 0.999 for r in rs2.values())


def test_group_magnitude_and_length_not_diluted():
    G = _gt(seed=1); masks = _masks()
    P = G.clone(); P[masks["sub-sub"]] *= 10
    assert float(L.sc_magnitude_loss(P, G, masks=masks)) > 3 * float(L.sc_magnitude_loss(P, G))
    lg = torch.full_like(G, 50.0); lp = lg.clone(); lp[masks["sub-sub"]] = 100.0
    assert float(L.tract_length_loss(lp * P, P, lg, G, masks=masks)) > 3 * float(L.tract_length_loss(lp * P, P, lg, G))


def test_group_metrics_masking():
    G = _gt(seed=2); masks = block_masks(82)
    gm = L.sc_group_metrics(G, G, masks)
    assert [gm[b]["n_edges"] for b in BLOCKS] == [2145, 1056, 120]
    assert all(abs(gm[b]["r"] - 1) < 1e-6 and gm[b]["log_mae"] < 1e-6 for b in BLOCKS)
