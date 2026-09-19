from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.training.distributed_sb3 import run_startup_stagger
from rl.training.environments import StartupStaggerWrapper, startup_stagger_delays


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1000, shape=(1,), dtype=np.int32)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.value = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.array([self.value], dtype=np.int32), {
            "run_stats": {"floor": self.value, "act": 1},
        }

    def step(self, action):
        self.value += 1
        return np.array([self.value], dtype=np.int32), 1.0, False, False, {
            "run_stats": {"floor": self.value, "act": 1},
        }


def test_startup_delays_are_reproducible_independent_and_stratified():
    delays = startup_stagger_delays(seed=7, rank=0, n_envs=96, span_steps=2048)

    assert delays == startup_stagger_delays(
        seed=7, rank=0, n_envs=96, span_steps=2048
    )
    assert delays != startup_stagger_delays(
        seed=7, rank=1, n_envs=96, span_steps=2048
    )
    assert min(delays) >= 0
    assert max(delays) <= 2048
    assert abs(float(np.mean(delays)) - 1024.0) < 12.0
    assert len(set(delays)) > 90


def test_stagger_wrapper_releases_as_an_explicit_episode_start():
    env = StartupStaggerWrapper(CounterEnv(), hold_steps=3)
    observation, _ = env.reset(seed=1)
    np.testing.assert_array_equal(observation, [0])

    for remaining in (2, 1):
        observation, reward, terminated, truncated, info = env.step(0)
        np.testing.assert_array_equal(observation, [0])
        assert reward == 0
        assert not terminated and not truncated
        assert info["startup_stagger_remaining"] == remaining

    _observation, _reward, terminated, truncated, _info = env.step(0)
    assert not terminated and truncated
    state = env.startup_stagger_state()
    assert state["released"] is True
    assert state["remaining_hold_steps"] == 0

    env.reset()
    observation, reward, terminated, truncated, _info = env.step(0)
    np.testing.assert_array_equal(observation, [1])
    assert reward == 1
    assert not terminated and not truncated
    assert env.startup_stagger_state()["steps_since_reset"] == 1


class FakeBuffer:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeModel:
    def __init__(self, env):
        self.env = env
        self.num_timesteps = 100
        self.rollout_buffer = FakeBuffer()
        self._last_obs = env.reset()
        self._last_episode_starts = np.ones(env.num_envs, dtype=bool)
        self._nosl_is_az_samples = ["discard"]
        self._nosl_is_az_pending_fights = [["discard"] for _ in range(env.num_envs)]
        self._reasoning_rollout_samples = ["discard"]
        self.ep_info_buffer = deque(["discard"], maxlen=10)
        self.ep_success_buffer = deque(["discard"], maxlen=10)
        self.logger = SimpleNamespace(
            name_to_value={"discard": 1},
            name_to_count={"discard": 1},
            name_to_excluded={"discard": 1},
        )

    def get_env(self):
        return self.env

    def _wait_for_reasoning_samples(self):
        return None

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        rollout_buffer.reset()
        callback.on_rollout_start()
        for _ in range(n_rollout_steps):
            self._last_obs, _rewards, dones, _infos = env.step(
                np.zeros(env.num_envs, dtype=np.int64)
            )
            self._last_episode_starts = dones
            self.num_timesteps += env.num_envs
            callback.update_locals({})
            assert callback.on_step()
        callback.on_rollout_end()
        return True


def test_warmup_discards_counters_samples_and_keeps_staggered_states():
    env = DummyVecEnv([
        lambda: StartupStaggerWrapper(CounterEnv(), hold_steps=0),
        lambda: StartupStaggerWrapper(CounterEnv(), hold_steps=4),
    ])
    model = FakeModel(env)

    result = run_startup_stagger(model, span_steps=4, rollout_steps=2)

    assert model.num_timesteps == 100
    assert result["discarded_transitions_local"] == 8
    assert result["age_min"] == 0
    assert result["age_max"] == 4
    assert model._nosl_is_az_samples == []
    assert model._nosl_is_az_pending_fights == [[], []]
    assert list(model.ep_info_buffer) == []
    assert model.logger.name_to_value == {}
    env.close()
