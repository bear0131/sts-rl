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


def test_anchor_cuts_combat_only_and_learns_from_future_without_changing_upper():
    from env.core import vocab
    from rl.algorithms.room_returns import combat_anchor_returns
    c, u = (vocab.PHASE_VOCAB_INDEX[k] for k in ('combat', 'map'))
    r = np.array([.01, 0, 0, 0, .02, 0, 0]).reshape(-1, 1)
    v = np.array([8., 9., -6., .4, -4., 3., 4.]).reshape(-1, 1)
    phases = np.array([u, c, c, c, u, c, c]).reshape(-1, 1)
    anchors = np.array([0, 0, 0, 1, 0, 0, 0], bool).reshape(-1, 1)
    original = np.array([.03, .02, .02, .02, .02, 0., 0.]).reshape(-1, 1)
    advantages, target = combat_anchor_returns(r, v, np.zeros_like(r), phases, anchors,
        original, last_values=np.array([99.]), last_dones=np.array([True]),
        last_phases=np.array([u]), last_anchors=np.array([False]))
    np.testing.assert_allclose(target[:, 0], [.03, .4, .4, .02, .02, 0., 0.])
    np.testing.assert_allclose(advantages, target - v)
    assert original[1, 0] == .02  # no in-place rewrite of the upper MC trace


@pytest.mark.parametrize('anchor_at_tail', [False, True])
def test_anchor_collection_cut_uses_old_tail_value(anchor_at_tail):
    from env.core import vocab
    from rl.algorithms.room_returns import combat_anchor_returns
    c = vocab.PHASE_VOCAB_INDEX['combat']
    x = np.zeros((2, 1))
    _, target = combat_anchor_returns(x, x, x, np.full_like(x, c), x.astype(bool), x,
        last_values=np.array([.7]), last_dones=np.array([False]),
        last_phases=np.array([c]), last_anchors=np.array([anchor_at_tail]))
    np.testing.assert_allclose(target, .7)


def test_anchor_missing_boundary_is_an_error_not_future_reward_leak():
    from env.core import vocab
    from rl.algorithms.room_returns import combat_anchor_returns
    x = np.zeros((1, 1))
    with pytest.raises(ValueError, match='without the required'):
        combat_anchor_returns(x, x, x, np.full_like(x, vocab.PHASE_VOCAB_INDEX['combat']),
            x.astype(bool), x, last_values=np.array([.7]), last_dones=np.array([False]),
            last_phases=np.array([vocab.PHASE_VOCAB_INDEX['map']]), last_anchors=np.array([False]))


@pytest.mark.parametrize('precomputed', [False, True])
def test_room_returns_reuses_frozen_tail_without_another_policy_forward(precomputed):
    import torch
    from types import SimpleNamespace
    from env.core import encode, vocab
    from rl.algorithms.room_returns import apply_room_returns
    upper = vocab.PHASE_VOCAB_INDEX['map']
    observations = dict(act=np.full((2,1,1), 1/encode.ACT_SCALE),
                        floor=np.full((2,1,1), 3/encode.FLOOR_SCALE),
                        phase=np.full((2,1,1), upper))
    tail = {k:v[-1].copy() for k,v in observations.items()}
    calls = []
    def predict(obs):
        calls.append(True)
        return torch.tensor([[.4]])
    model = SimpleNamespace(device=torch.device('cpu'), _last_obs=tail,
        _last_episode_starts=np.array([False]), logger=SimpleNamespace(record=lambda *args:None),
        policy=SimpleNamespace(predict_values=predict, room_lambda=1., combat_value_target='whole-run'))
    buffer = SimpleNamespace(observations=observations, rewards=np.array([[.1],[.2]]),
        values=np.array([[3.],[4.]]), episode_starts=np.zeros((2,1)),
        advantages=np.zeros((2,1)), returns=np.zeros((2,1)))
    apply_room_returns(model,buffer,**({'last_values':np.array([.4],np.float32)} if precomputed else {}))
    np.testing.assert_allclose(buffer.returns, [[.7],[.6]], atol=1e-7)
    assert calls == ([] if precomputed else [True])
    with pytest.raises(ValueError,match='tail values'):
        apply_room_returns(model,buffer,last_values=np.array([float('nan')]))
