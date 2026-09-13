import numpy as np
import pytest
from gymnasium import spaces

from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer


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
