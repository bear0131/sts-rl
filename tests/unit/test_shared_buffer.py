from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

from rl.algorithms.shared_buffer import SharedRolloutBuffer


def assert_owned_storage_returns_and_env_major_shuffle(environments, batch_size, prefill, device='cpu'):
    observation_space = spaces.Dict({
        'vector': spaces.Box(-10, 100, (2, 3), dtype=np.float32),
        'category': spaces.Discrete(9),
    })
    buffers = [cls(5, observation_space, spaces.Discrete(4), device=device,
                   gamma=1., gae_lambda=.9, n_envs=environments)
               for cls in (MaskableDictRolloutBuffer, SharedRolloutBuffer)]
    fast = buffers[1]
    allocations = {key: value for key, value in fast.observations.items()}
    arrays = ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
              'advantages', 'returns', 'action_masks')
    retained_samples = []
    for iteration in range(2):
        for buffer in buffers:
            buffer.reset()
        for t in range(5):
            obs = {'vector': np.arange(environments * 6, dtype=np.float32).reshape(
                       environments, 2, 3) + t + iteration,
                   'category': np.arange(environments) + t}
            actions = np.arange(environments) % 4
            reward = np.arange(environments, dtype=np.float32) + t / 10
            starts = np.full(environments, t in (0, 3), dtype=np.float32)
            values = torch.arange(environments, dtype=torch.float32) / 7 + t / 9
            log_prob = torch.full((environments,), -.3 - t / 8)
            if environments == 1:
                log_prob = log_prob[0]
            masks = np.ones((environments, 4), dtype=bool)
            masks[:, (t + 1) % 4] = False
            if prefill:
                fast.prefill_observation(obs)
                assert fast.pos == t and not fast.full
            for buffer in buffers:
                buffer.add(obs, actions, reward, starts, values, log_prob, action_masks=masks)
            # Mutating all caller storage must not corrupt either buffer.
            for value in obs.values():
                value.fill(99)
            actions.fill(3); reward.fill(99); starts.fill(99); masks.fill(False)
            values.fill_(99); log_prob.fill_(99)
        for buffer in buffers:
            buffer.compute_returns_and_advantage(torch.ones(environments),
                                                 np.zeros(environments, dtype=bool))
        for key in buffers[0].observations:
            np.testing.assert_array_equal(buffers[0].observations[key], fast.observations[key])
        for key in arrays:
            np.testing.assert_array_equal(getattr(buffers[0], key), getattr(fast, key))
        batches = []
        for buffer in buffers:
            np.random.seed(20261109 + iteration)
            batches.append(list(buffer.get(batch_size)))
        for expected, actual in zip(*batches, strict=True):
            for key in expected.observations:
                torch.testing.assert_close(actual.observations[key], expected.observations[key],
                                           rtol=0, atol=0)
            for key in expected._fields[1:]:
                torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
            retained_samples.append((actual, deepcopy(actual)))
        for key, allocation in allocations.items():
            assert fast.observations[key] is allocation
    fast.reset()
    assert not fast.full and fast.pos == 0 and not fast.generator_ready
    assert all(not value.any() for value in fast.observations.values())
    assert fast.action_masks.all()
    for actual, saved in retained_samples:
        for key in saved.observations:
            torch.testing.assert_close(actual.observations[key], saved.observations[key], rtol=0, atol=0)


@pytest.mark.parametrize('prefill', [False, True])
@pytest.mark.parametrize('environments', [1, 3])
@pytest.mark.parametrize('batch_size', [4, None])
def test_owned_storage_returns_and_env_major_shuffle_match_sb3(environments, batch_size, prefill):
    assert_owned_storage_returns_and_env_major_shuffle(environments, batch_size, prefill)


def test_prefill_replacement_and_aborted_collection_do_not_reuse_old_observation():
    buffer = SharedRolloutBuffer(2, spaces.Dict({
        'x': spaces.Box(-100, 100, (2,), dtype=np.float32),
    }), spaces.Discrete(2), device='cpu', n_envs=1)

    def add(obs):
        buffer.add(obs, np.array([0]), np.array([1.]), np.array([False]),
                   torch.tensor([.2]), torch.tensor([-.3]))

    stale = {'x': np.array([[1., 2.]], dtype=np.float32)}
    current = {'x': np.array([[3., 4.]], dtype=np.float32)}
    buffer.prefill_observation(stale)
    add(current)
    np.testing.assert_array_equal(buffer.observations['x'][0], current['x'])
    buffer.prefill_observation(stale)
    buffer.reset()  # An early-stop callback can abandon a dispatched step.
    stale['x'][:] = 5
    add(stale)
    np.testing.assert_array_equal(buffer.observations['x'][0], stale['x'])
    stale['x'][:] = 9
    add(stale)
    np.testing.assert_array_equal(buffer.observations['x'][1], stale['x'])
