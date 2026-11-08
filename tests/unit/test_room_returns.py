import numpy as np
import pytest

from rl.algorithms.room_returns import room_gae


def calculate(reward, value, room, *, lam=.9, terminal=True, tail=0., starts=None):
    r, v, f = (np.asarray(x, dtype=float).reshape(-1, 1) for x in (reward, value, room))
    return room_gae(r, v, np.zeros_like(r) if starts is None else np.array(starts).reshape(-1, 1),
        np.ones_like(r), f, last_values=np.array([tail]), last_dones=np.array([terminal]),
        last_acts=np.ones(1), last_floors=f[-1], room_lambda=lam)


def test_one_room_exact_mc_despite_bad_intermediate_values():
    a, target, lam = calculate([0, 0, 1.2], [9., -3., 5.], [1, 1, 1])
    np.testing.assert_allclose(target[:, 0], [1.2, 1.2, 1.2], atol=1e-14)
    np.testing.assert_array_equal(lam[:, 0], [1, 1, 0])


def test_room_boundary_and_inserting_zero_reward_menus():
    _, base, _ = calculate([0, .1, 1.], [.4, .8, .3], [1, 2, 2])
    _, extra, _ = calculate([0, 0, 0, .1, 1.], [.4, 7., -8., .8, .3], [1, 1, 1, 2, 2])
    assert base[0, 0] == pytest.approx(.1 * .8 + .9 * 1.1)
    assert extra[0, 0] == pytest.approx(base[0, 0])


def test_lambda_endpoints_tail_and_episode_reset():
    _, mc, _ = calculate([.1, .2, 1], [8, 5, -1], [1, 2, 3], lam=1)
    np.testing.assert_allclose(mc[:, 0], [1.3, 1.2, 1.])
    _, td, _ = calculate([.1, .2, 1], [8, 5, -1], [1, 2, 3], lam=0)
    np.testing.assert_allclose(td[:, 0], [5.1, -.8, 1.])
    _, tail, _ = calculate([.1, .2], [8, 5], [1, 1], terminal=False, tail=.4)
    np.testing.assert_allclose(tail[:, 0], [.7, .6])
    _, reset, _ = calculate([.1, 0, 1], [8, 100, 5], [1, 1, 1], starts=[1, 1, 0])
    np.testing.assert_allclose(reset[:, 0], [.1, 1, 1])


def test_act_change_is_one_boundary_and_envs_do_not_mix():
    _, target, lam = room_gae(np.array([[.1, 0], [1., .2]]), np.array([[8., 8.], [.5, .4]]),
        np.zeros((2, 2)), np.array([[1, 1], [2, 1]]), np.array([[17, 1], [18, 1]]),
        last_values=np.zeros(2), last_dones=np.ones(2, bool),
        last_acts=np.ones(2), last_floors=np.ones(2), room_lambda=.9)
    np.testing.assert_allclose(lam[0], [.9, 1])
    np.testing.assert_allclose(target[0], [1.05, .2])


def test_invalid_lambda_rejected():
    with pytest.raises(ValueError):
        calculate([0], [0], [1], lam=float('nan'))
