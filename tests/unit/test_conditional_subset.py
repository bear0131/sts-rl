import numpy as np
import pytest
from env.search.conditional_subset import (
    conditional_probabilities,
    multi_select_slots,
    search_subset,
)

MARK = 3 << 29


def test_conditional_supervision_preserves_exclusion_and_real_slots():
    bits = np.array([MARK, MARK | 1, MARK | 4, MARK | 5])
    assert multi_select_slots(bits) == (0, 2)
    assert not multi_select_slots([0, 1, 4 << 29])
    cond = conditional_probabilities(bits, [0.0, 0.5, 0.5, 0.0])
    assert cond[0, 0] == (1.0, 0.5)
    assert cond[2, 0] == (0.5, 1.0)
    assert cond[2, 1] == (0.5, 0.0)
    assert conditional_probabilities(bits, [0.0, 0.5, 0.5, 0.0], order=[2, 0])[
        0, 4
    ] == (0.5, 0.0)
    with pytest.raises(ValueError):
        multi_select_slots([MARK, 0])


def test_prefix_search_commits_one_legal_set_and_bounds_all_simulations():
    bits = np.array([MARK, MARK | 1, MARK | 2, MARK | 4])

    def evaluate(indices, n, nonce):
        worlds = (n + len(indices) - 1) // len(indices)
        out = np.full(worlds * len(indices), np.nan)
        out[:n] = (np.resize(bits[indices], n) == (MARK | 2)).astype(float)
        return out.reshape(worlds, len(indices))

    i, v = search_subset(bits, 127, evaluate, np.random.default_rng(3), np.zeros(4))
    assert bits[i] == MARK | 2 and 0 < v.sum() <= 127 and v[i] > 0


def test_stratified_completion_pairs_are_distinct_for_uneven_branch_sizes():
    bits = np.array([MARK | m for m in range(64) if m.bit_count() <= 2])

    def evaluate(indices, n, nonce):
        assert len(set(indices.tolist())) == len(indices)
        vals = np.zeros(((n + len(indices) - 1) // len(indices), len(indices)))
        vals.ravel()[n:] = np.nan
        return vals

    for seed in range(32):
        search_subset(
            bits, 250, evaluate, np.random.default_rng(seed), np.zeros(len(bits))
        )
