import numpy as np
import pytest
from gymnasium import spaces

from env.core import native_actions as C
from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer


def make_buffer(capacity=3):
    space = spaces.Dict(combat_legal=spaces.Box(0, 1, (C.JOINT_WIDTH,), bool),
                        player_scalars=spaces.Box(-np.inf, np.inf, (2,), np.float32))
    return NOSLISAZReplayBuffer(capacity=capacity, observation_space=space,
        observation_keys=space.spaces, action_dim=C.JOINT_SIZE)


def sample(index):
    ids = np.array([C.JOINT_END, C.JOINT_PLAY+index, C.JOINT_MULTI+index])
    mask = C.joint_mask(ids)
    target = np.zeros(C.JOINT_SIZE, np.float32)
    target[ids] = [0., 1e-10, 1. - 1e-10]
    return dict(observation=dict(combat_legal=mask[C.JOINT_END:],
                                 player_scalars=np.array([index, 2.], np.float32)),
                action_mask=mask, policy_target=target, value_target=.3)


def test_sparse_joint_replay_preserves_zero_and_tiny_probabilities_and_ring():
    replay = make_buffer()
    replay.add_samples([sample(1), sample(2)])
    replay.add_samples([sample(3), sample(4)])
    assert replay.action_masks is None and replay.policy_targets is None
    assert 'combat_legal' not in replay.observations
    restored = make_buffer()
    restored.load_state_dict(replay.state_dict())
    batch = restored.get_batch(np.array([1, 0, 2, 1]), workers=2)
    for row, index in enumerate([2, 4, 3, 2]):
        expected = sample(index)
        np.testing.assert_array_equal(batch['action_masks'][row], expected['action_mask'])
        np.testing.assert_array_equal(batch['policy_targets'][row], expected['policy_target'])
        np.testing.assert_array_equal(batch['observations']['combat_legal'][row], expected['action_mask'][C.JOINT_END:])
    assert replay.allocated_bytes < 1024
    assert restored.total_added == 4
    state = replay.state_dict()
    state.pop('policy_contract')
    with pytest.raises(ValueError, match='policy contract'):
        make_buffer().load_state_dict(state)


def test_joint_replay_rejects_illegal_probability_and_accepts_value_only():
    replay = make_buffer()
    bad = sample(1)
    bad['policy_target'][0] = .1
    with pytest.raises(ValueError, match='illegal action'):
        replay.add_samples([bad])
    value = sample(1)
    value.update(action_mask=np.zeros(C.JOINT_SIZE, bool),
                 policy_target=np.zeros(C.JOINT_SIZE, np.float32), policy_weight=0.)
    replay.add_samples([value])
    batch = replay.get_batch(np.array([0]))
    assert batch['policy_weights'][0] == 0
    assert batch['value_targets'][0] == pytest.approx(.3)
