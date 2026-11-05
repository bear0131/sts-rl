import numpy as np
import pytest

from rl.policies.paired_return_target import paired_return_target


def target(x, boss=False, hp=80, seed=1):
    return paired_return_target(
        x, boss=boss, max_hp=hp, rng=np.random.default_rng(seed)
    )


def test_equal_returns_and_effect_units():
    np.testing.assert_allclose(target([[0.5, 0.5], [0.7, 0.7]]), [0.5, 0.5])
    np.testing.assert_allclose(target([[0.5, 0.5 - 1 / 80]]), [2 / 3, 1 / 3])
    np.testing.assert_allclose(target([[0.6, 0.59]], boss=True), [2 / 3, 1 / 3])
    np.testing.assert_allclose(target([[0.6, 0.59]], boss=True, hp=40), [2 / 3, 1 / 3])


def test_shared_random_shocks_preserve_the_action_difference_and_legal_order():
    x = np.array([[0.9, 0.8, 0.7], [0.3, 0.2, 0.1], [0.7, 0.6, 0.5]])
    np.testing.assert_allclose(target(x), target(x[:1]))
    np.testing.assert_allclose(target(x[:, ::-1]), target(x)[::-1])
    np.testing.assert_allclose(target([[0.3, 0.3], [0.3, np.nan]]), [0.5, 0.5])
    with pytest.raises(ValueError, match="every action"):
        target([[1, np.nan]])
    with pytest.raises(ValueError, match="worlds"):
        target([[np.nan, 0.5], [0.1, 0.3]])
