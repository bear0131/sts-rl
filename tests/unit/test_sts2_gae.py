import numpy as np
import pytest

from rl.algorithms.sts2_ppo import gae

pytestmark = pytest.mark.sts2


def test_truncation_bootstraps_but_does_not_cross_reset():
    rewards = np.array([[0., 0.], [1., 1.]], dtype=np.float32)
    values = np.array([[2., 2.], [3., 3.]], dtype=np.float32)
    next_values = np.array([[4., 4.], [9., 9.]], dtype=np.float32)
    terminated = np.array([[False, True], [True, True]])
    boundaries = np.ones((2, 2), dtype=bool)
    advantage, returns = gae(rewards, values, next_values, terminated, boundaries, .5, .9)
    np.testing.assert_array_equal(advantage, [[0, -2], [-2, -2]])
    np.testing.assert_array_equal(returns, [[2, 0], [1, 1]])
