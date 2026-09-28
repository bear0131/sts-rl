from collections import deque
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.tasks.wholerun import StSMetricsCallback
from rl.training.distributed_sb3 import run_startup_occupancy, run_startup_stagger
from rl.training.environments import (
    StartupOccupancyWrapper, StartupStaggerWrapper, startup_stagger_delays,
)


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


def test_retired_occupancy_arguments_do_not_install_an_unreleased_hold(monkeypatch):
    from rl.training import environments as runtime
    monkeypatch.setattr(runtime, 'observation_flags', lambda model: dict.fromkeys((
        'global_map_view', 'card_language', 'combat_language', 'deck_size_observation',
        'neow_observation', 'event_option_observation', 'interactive_reasoning'), False))
    monkeypatch.setattr(runtime.WholeRunTask, 'make_env_fn', lambda *_: CounterEnv)
    args = SimpleNamespace(ascension=0, interactive_reasoning=False, startup_stagger_steps=0,
        startup_occupancy_budget=160, startup_occupancy_max_age=12, seed=42, rank=0,
        n_envs=2, vec='dummy')
    env = runtime.make_env(SimpleNamespace(), args)
    try:
        env.reset()
        for value in range(1, 5):
            observation, rewards, dones, _ = env.step(np.zeros(2, dtype=int))
            np.testing.assert_array_equal(observation[:, 0], [value, value])
            np.testing.assert_array_equal(rewards, [1.0, 1.0])
            assert not dones.any()
    finally:
        env.close()


class FakeModel:
    def __init__(self, env):
        self.env = env
        self.device = SimpleNamespace(type="cpu")
        self.nosl_is_az = True
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
            self._last_obs, _rewards, dones, infos = env.step(
                np.zeros(env.num_envs, dtype=np.int64)
            )
            self._last_episode_starts = dones
            self.num_timesteps += env.num_envs
            callback.update_locals({"infos": infos})
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
    assert result["completed_episode_count"] == 0
    assert result["synthetic_release_count"] == 1
    assert model._nosl_is_az_samples == []
    assert model._nosl_is_az_pending_fights == [[], []]
    assert list(model.ep_info_buffer) == []
    assert model.logger.name_to_value == {}
    env.close()


def test_active_floor_is_not_the_final_floor_of_completed_episodes():
    env = VecMonitor(DummyVecEnv([
        lambda: StartupStaggerWrapper(
            gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=3),
            hold_steps=0,
        ),
        lambda: StartupStaggerWrapper(
            gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=3),
            hold_steps=4,
        ),
    ]))
    model = FakeModel(env)
    result = run_startup_stagger(model, span_steps=6, rollout_steps=2)

    assert result["active_floor_mean"] == 1.0
    assert result["completed_episode_count"] == 2
    assert result["completed_floor_mean_last_100"] == 3.0
    assert result["completed_win_rate_last_100"] == 0.0
    assert result["synthetic_release_count"] == 1
    assert model.num_timesteps == 100
    env.close()


class RandomLengthEnv(CounterEnv):
    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.length = 1 if self.np_random.integers(2) else 3
        return observation, info

    def step(self, action):
        observation, reward, _, _, info = super().step(action)
        return observation, reward, self.value >= self.length, False, info


def test_occupancy_sampler_weights_episode_length_not_episode_count():
    ages = []
    for seed in range(1500):
        env = StartupOccupancyWrapper(
            RandomLengthEnv(), max_episode_steps=3, seed=seed,
        )
        env.reset(seed=seed + 100_000)
        for _ in range(120):
            if env.startup_occupancy_state()["accepted"]:
                ages.append(env.startup_occupancy_state()["age"])
                break
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                env.reset()
        else:
            pytest.fail("occupancy rejection did not accept within budget")
        env.close()
    proportions = np.bincount(ages, minlength=3) / len(ages)
    # L=1 and L=3 are equiprobable. Across all real decision instants,
    # ages 0,1,2 have expected masses 2/4,1/4,1/4, not 1/3 each.
    np.testing.assert_allclose(proportions, [0.5, 0.25, 0.25], atol=0.045)


def test_occupancy_accepted_state_does_not_advance_before_release():
    env = StartupOccupancyWrapper(CounterEnv(), max_episode_steps=3, seed=4)
    env.reset()
    env._target_age = 1
    env._accepted = False
    observation, reward, *_ = env.step(0)
    assert observation.tolist() == [1]
    assert reward == 1
    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(0)
        assert observation.tolist() == [1]
        assert reward == 0
        assert not terminated and not truncated
        assert info["startup_occupancy_paused"] is True
    assert env.startup_occupancy_state()["real_steps"] == 1
    env.startup_occupancy_release()
    assert env.step(0)[0].tolist() == [2]
    env.close()


def test_occupancy_marks_only_the_incumbent_episode_after_release():
    env = StartupOccupancyWrapper(
        gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=3),
        max_episode_steps=3,
        seed=4,
    )
    env.reset()
    env._target_age = 1
    env._accepted = False
    assert env.step(0)[4]["startup_occupancy_accepted"] is True
    env.startup_occupancy_release()

    _obs, _reward, terminated, truncated, _info = env.step(0)
    assert not terminated and not truncated
    _obs, _reward, _terminated, truncated, info = env.step(0)
    assert truncated is True
    assert info["startup_occupancy_incumbent_episode"] is True

    env.reset()
    env.step(0)
    env.step(0)
    _obs, _reward, _terminated, truncated, info = env.step(0)
    assert truncated is True
    assert "startup_occupancy_incumbent_episode" not in info
    env.close()


def test_metrics_callback_excludes_incumbent_episode_entirely():
    recorded = {}
    callback = StSMetricsCallback(window=100)
    callback.model = SimpleNamespace(
        logger=SimpleNamespace(
            record=lambda key, value: recorded.__setitem__(key, value)
        )
    )
    callback.locals = {"infos": [
        {
            "episode": {"r": 999.0, "l": 999},
            "run_stats": {
                "floor": 55, "act": 4, "deck_size": 99,
                "outcome": "PLAYER_VICTORY",
            },
            "startup_occupancy_incumbent_episode": True,
        },
        {
            "episode": {"r": 1.0, "l": 7},
            "run_stats": {
                "floor": 8, "act": 1, "deck_size": 12,
                "outcome": "UNDECIDED",
            },
        },
    ]}

    assert callback._on_step() is True
    callback._on_rollout_end()

    assert recorded["game/episodes_total"] == 1
    assert recorded["game/episodes_window"] == 1
    assert recorded["game/startup_episodes_excluded_total"] == 1
    assert recorded["game/win_rate"] == 0.0
    assert recorded["game/floor_mean"] == 8
    assert recorded["game/deck_size_mean"] == 12
    assert recorded["rollout/ep_rew_mean"] == 1.0
    assert recorded["rollout/ep_len_mean"] == 7


def test_occupancy_startup_preserves_replay_rng_and_drops_warmup_roots():
    env = VecMonitor(DummyVecEnv([
        lambda: StartupOccupancyWrapper(
            gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=3),
            max_episode_steps=3, seed=10,
        ),
        lambda: StartupOccupancyWrapper(
            gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=3),
            max_episode_steps=3, seed=11,
        ),
    ]))
    model = FakeModel(env)
    model._nosl_is_az_pending_fights = [["live-0"], ["live-1"]]
    model._nosl_is_az_replay = SimpleNamespace(
        position=37, size=524288, total_added=3000000,
    )
    numpy_rng, python_rng, torch_rng = (
        np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    )
    result = run_startup_occupancy(
        model, budget_steps=40, rollout_steps=2, seed=2026,
    )
    assert result["collection_blocks"] >= 1
    assert result["max_age"] == 3
    assert result["discarded_transitions_local"] == 4 * result["collection_blocks"]
    assert model.num_timesteps == 100
    assert model._nosl_is_az_pending_fights == [[], []]
    assert model._nosl_is_az_samples == []
    assert (model._nosl_is_az_replay.position, model._nosl_is_az_replay.size,
            model._nosl_is_az_replay.total_added) == (37, 524288, 3000000)
    assert result["real_transitions_local"] < result["discarded_transitions_local"]
    assert list(env.episode_lengths) == [
        state["age"] for state in env.env_method("startup_occupancy_state")
    ]
    np.testing.assert_allclose(
        env.episode_returns,
        [state["episode_return"] for state in env.env_method("startup_occupancy_state")],
    )
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert list(model.ep_info_buffer) == []
    assert all(state["accepted"] for state in env.env_method("startup_occupancy_state"))
    env.close()


def test_occupancy_budget_exhaustion_does_not_start_formal_update():
    env = DummyVecEnv([lambda: StartupOccupancyWrapper(
        gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=1),
        max_episode_steps=3, seed=0,
    )])
    model = FakeModel(env)
    with pytest.raises(RuntimeError, match="exhausted its budget"):
        run_startup_occupancy(model, budget_steps=2, rollout_steps=2, seed=77)
    assert model.num_timesteps == 102
    env.close()
