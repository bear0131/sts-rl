import numpy as np
import pytest
from gymnasium import spaces

from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer
from env.core.combat_value import VALUE_CONTRACT


@pytest.mark.parametrize('count', [0, 2, 6])
def test_empty_partial_and_wrapped_replay_restore(count):
    def make():
        return NOSLISAZReplayBuffer(capacity=4, action_dim=3,
            observation_keys=['x'], observation_space=spaces.Dict({
                'x':spaces.Box(0, 10, shape=(2,), dtype=np.int64)}))
    first, restored = make(), make()
    def sample(i):
        return dict(observation={'x':np.array([i,i])}, action_mask=np.ones(3,dtype=bool),
            policy_target=np.array([0.2,0.3,0.5]), value_target=i/10)
    first.add_samples([sample(i) for i in range(count)])
    state = first.state_dict()
    if count:
        assert state['observations']['x'].shape[0] == min(count,4)
    restored.load_state_dict(state)
    first.add_samples([sample(7)])
    restored.add_samples([sample(7)])
    assert (first.size,first.position,first.total_added) == (restored.size,restored.position,restored.total_added)
    for name in ('value_targets','policy_targets','action_masks','sample_ids'):
        np.testing.assert_array_equal(getattr(first,name)[:first.size],getattr(restored,name)[:restored.size])
    np.testing.assert_array_equal(first.observations['x'][:first.size],restored.observations['x'][:restored.size])


def test_unbounded_player_scalars_do_not_overflow_float16():
    replay = NOSLISAZReplayBuffer(
        capacity=2,
        action_dim=2,
        observation_keys=['player_scalars', 'normalized'],
        observation_space=spaces.Dict({
            'player_scalars': spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32
            ),
            'normalized': spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        }),
    )
    replay.add_samples([dict(
        observation={
            'player_scalars': np.asarray([100_000.0, -100_000.0], dtype=np.float32),
            'normalized': np.asarray([0.5], dtype=np.float32),
        },
        action_mask=np.ones(2, dtype=bool),
        policy_target=np.asarray([0.5, 0.5], dtype=np.float32),
        value_target=0.25,
    )])

    assert replay.observations['player_scalars'].dtype == np.float32
    assert replay.observations['normalized'].dtype == np.float16
    np.testing.assert_array_equal(
        replay.observations['player_scalars'][0],
        np.asarray([100_000.0, -100_000.0], dtype=np.float32),
    )
    assert np.isfinite(replay.observations['player_scalars'][: replay.size]).all()


def test_old_float16_player_scalars_restore_into_float32_storage():
    space = spaces.Dict({
        'player_scalars': spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32
        ),
    })
    replay = NOSLISAZReplayBuffer(
        capacity=2,
        action_dim=2,
        observation_keys=['player_scalars'],
        observation_space=space,
    )
    state = dict(
        version=2,
        value_contract=VALUE_CONTRACT,
        capacity=2,
        action_dim=2,
        observation_keys=('player_scalars',),
        position=1,
        size=1,
        total_added=1,
        observations={
            'player_scalars': np.asarray([[0.25, 3.0]], dtype=np.float16),
        },
        arrays={
            'action_masks': np.asarray([[True, False]], dtype=bool),
            'policy_targets': np.asarray([[1.0, 0.0]], dtype=np.float16),
            'policy_weights': np.asarray([1.0], dtype=np.float16),
            'value_targets': np.asarray([0.25], dtype=np.float32),
            'sample_ids': np.asarray([0], dtype=np.uint64),
        },
    )

    replay.load_state_dict(state)

    assert replay.observations['player_scalars'].dtype == np.float32
    np.testing.assert_array_equal(
        replay.observations['player_scalars'][0],
        state['observations']['player_scalars'][0].astype(np.float32),
    )
