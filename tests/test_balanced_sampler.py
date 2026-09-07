"""GESTA 전략 smoke test 3/4/5: 큰 bundle downsample, batch 균형, GT SC 불변 (docs §77–79)."""
import numpy as np
import torch

from atm_sc.data.balanced_pair_sampler import BalanceConfig, BalancedPairSampler, exposure_targets


class FakeSubject:
    """pair_count_full = [5000, 500, 20]; real pool 은 cap 256 처럼 min(N, 256) 개, synthetic 은 선택."""
    def __init__(self, counts=(5000, 500, 20), synth=(0, 0, 0), seed=0):
        self.sub = "fake"
        self.pair_ids = np.array([[0, 1], [2, 3], [4, 5]], np.int64)
        self.pair_count_full = np.array(counts, np.int64)
        self.pair_tier = np.array([2, 1, 0])
        g = torch.Generator().manual_seed(seed)
        self._real = [torch.randn(min(int(c), 256), 128, 3, generator=g) for c in counts]
        self._syn = [torch.randn(int(m), 128, 3, generator=g) + 100 for m in synth]   # +100: real 과 구분
        self.sc_mat = np.array([[0, 5000, 0, 0, 0, 0]] * 6, np.float64)

    def get_pair(self, k):
        return self._real[k], torch.ones(self._real[k].shape[0])

    def synthetic_pair(self, k):
        return self._syn[k]


def test_exposure_targets_compress_large_bundles():
    cfg = BalanceConfig(enabled=True)
    t = exposure_targets(np.array([1, 20, 100, 5000, 50000]), cfg)
    assert np.allclose(t, [16.0, np.sqrt(2000), 100.0, 256.0, 256.0])   # sqrt 규칙 + clip [16, 256]
    assert t[3] / t[1] < 6 < 5000 / 20                                   # 250:1 -> 5.7:1


def test_batch_balance_not_proportional_to_counts():
    s = FakeSubject(); bs = BalancedPairSampler(s, BalanceConfig(enabled=True))
    rng = np.random.default_rng(0)
    for _ in range(3000):                                  # step 마다 pair 1개: 선택 확률 p ∝ B_k 가 노출을 결정
        S, P, V, syn, info = bs.sample_batch(rng, n_pairs=1, n_per_pair=8)
        assert S.shape == (8, 128, 3) and P.shape == (8, 2)
    e = bs.exposure / bs.exposure.sum()
    assert 3 < e[0] / e[2] < 12                             # raw 250:1 이 아니라 256/44.7 = 5.7:1 근처
    assert bs.n_synth_used == 0


def test_large_bundle_downsampled_and_subsets_vary():
    s = FakeSubject(); bs = BalancedPairSampler(s, BalanceConfig(enabled=True))
    rng = np.random.default_rng(1)
    S1, *_ = bs.sample_batch(rng, 1, 8); S2, *_ = bs.sample_batch(rng, 1, 8)
    assert S1.shape[0] == 8 and not torch.equal(S1, S2)     # 5000(256) 개 전부가 아니라 8개, step 마다 다른 subset


def test_small_bundle_uses_synthetic_with_real_floor_and_cap():
    s = FakeSubject(counts=(5000, 500, 20), synth=(0, 0, 200))
    bs = BalancedPairSampler(s, BalanceConfig(enabled=True, real_fraction_min=0.5, max_synthetic_ratio=4.0))
    rng = np.random.default_rng(2)
    real, syn = s.get_pair(2)[0], s.synthetic_pair(2)
    S, _, syn, r, y = bs._fill(2, 16, rng)
    assert S.shape[0] == 16 and r == 8 and y == 8                    # real ≥ 50 %, 나머지 synthetic
    assert syn.sum() == 8 and not syn[:8].any()
    assert ((S[:8] - real.mean()).abs().mean() < 50) and ((S[8:] - 100).abs().mean() < 5)
    s2 = FakeSubject(counts=(5000, 500, 2), synth=(0, 0, 200))
    bs2 = BalancedPairSampler(s2, BalanceConfig(enabled=True))
    S, _, syn2, r, y = bs2._fill(2, 16, rng)
    assert r == 8 and y == 8                                          # real 2개뿐: synthetic 은 4×real=8 까지, 나머지 real 복원


def test_gt_sc_untouched_by_sampling():
    s = FakeSubject(synth=(0, 0, 200)); before = s.sc_mat.copy()
    bs = BalancedPairSampler(s, BalanceConfig(enabled=True))
    rng = np.random.default_rng(3)
    for _ in range(50):
        bs.sample_batch(rng, 3, 16)
    assert np.array_equal(s.sc_mat, before) and bs.n_synth_used > 0
