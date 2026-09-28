"""Live simulator recovery from a backend seed and committed action journal."""
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit

from env.core.run_env import SlayTheSpireEnv
from rl.training.environments import (
    CheckpointObservationCompat,
    StartupOccupancyWrapper,
    TrainingStateWrapper,
)
from rl.training.cached_mask_vec_env import GroupedCachedMaskSubprocVecEnv
from rl.training.reward import ShapedRewardWrapper


pytestmark = pytest.mark.native


def _assert_observation_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def _advance_to_combat(env):
    for _ in range(64):
        if env.unwrapped._phase() == "combat":
            return
        action = int(np.flatnonzero(env.action_masks())[0])
        _obs, _reward, terminated, truncated, _info = env.step(action)
        assert not terminated and not truncated
    pytest.fail("deterministic recovery fixture did not reach combat")


def _first_action(env):
    mask = env.get_wrapper_attr("action_masks")()
    return int(np.flatnonzero(mask)[0])


def test_action_journal_restores_native_action_and_pending_target():
    env = SlayTheSpireEnv()
    restored = SlayTheSpireEnv()
    try:
        env.reset(seed=123)
        _advance_to_combat(env)

        native_bits = int(env.native_combat_actions()[0])
        thin_action = env._native_action_to_thin(native_bits)
        assert thin_action is not None
        env._nosl_cpp_pending = (thin_action, native_bits)
        pending_state = env.export_training_state()
        pending_observation, _info = restored.restore_training_state(pending_state)
        _assert_observation_equal(env._obs(), pending_observation)
        assert restored._nosl_cpp_pending == (thin_action, native_bits)
        env.step(thin_action)
        target_roots = [
            int(action)
            for action in np.flatnonzero(env.action_masks())
            if not env.combat_action_advances(int(action))
        ]
        assert target_roots
        env.step(target_roots[0])
        assert env._pending is not None

        state = env.export_training_state()
        observation, _info = restored.restore_training_state(state)
        _assert_observation_equal(env._obs(), observation)
        np.testing.assert_array_equal(env.action_masks(), restored.action_masks())
        assert restored._pending == env._pending
        assert any(
            record.get("native_pending") == (thin_action, native_bits)
            for record in state["actions"]
        )

        for _ in range(32):
            action = int(np.flatnonzero(env.action_masks())[0])
            expected = env.step(action)
            actual = restored.step(action)
            _assert_observation_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            np.testing.assert_array_equal(
                expected[4]["action_mask"], actual[4]["action_mask"]
            )
            if expected[2] or expected[3]:
                break
    finally:
        env.close()
        restored.close()


def _training_env():
    base = SlayTheSpireEnv()
    wrapped = ShapedRewardWrapper(base)
    time_limit = TimeLimit(wrapped, max_episode_steps=6000)
    compatible = CheckpointObservationCompat(time_limit, base.observation_space)
    occupancy = StartupOccupancyWrapper(
        compatible, max_episode_steps=1000, seed=456
    )
    return TrainingStateWrapper(occupancy), occupancy, time_limit, wrapped


def test_training_wrapper_restores_occupancy_reward_and_time_limit_state():
    env, occupancy, time_limit, reward = _training_env()
    restored, restored_occupancy, restored_time_limit, restored_reward = (
        _training_env()
    )
    try:
        env.reset(seed=321)
        occupancy._target_age = 2
        occupancy._accepted = False
        for _ in range(2):
            env.step(_first_action(env))
        assert occupancy._accepted
        occupancy.startup_occupancy_release()
        env.step(_first_action(env))

        state = env.export_training_state()
        expected_observation = env.unwrapped._obs()
        actual_observation, _info = restored.restore_training_state(state)
        _assert_observation_equal(expected_observation, actual_observation)
        assert restored_occupancy.startup_occupancy_state() == (
            occupancy.startup_occupancy_state()
        )
        assert restored_time_limit._elapsed_steps == time_limit._elapsed_steps
        assert restored_reward._prev == reward._prev
    finally:
        env.close()
        restored.close()


def _grouped_training_env():
    base = SlayTheSpireEnv()
    return TrainingStateWrapper(TimeLimit(
        ShapedRewardWrapper(base), max_episode_steps=6000,
    ))


def test_grouped_native_environments_restore_in_logical_order():
    env = GroupedCachedMaskSubprocVecEnv(
        [_grouped_training_env] * 3,
        envs_per_worker=2,
        start_method="spawn",
    )
    try:
        observation = env.reset()
        for _ in range(4):
            masks = np.stack(env.env_method("action_masks"))
            actions = np.asarray([
                np.flatnonzero(mask)[-1] for mask in masks
            ], dtype=np.int64)
            observation, _rewards, _dones, _infos = env.step(actions)
        states = env.env_method("export_training_state")
        expected = {key: value.copy() for key, value in observation.items()}
        expected_masks = np.stack(env.env_method("action_masks"))

        env.step(np.asarray([
            np.flatnonzero(mask)[0] for mask in expected_masks
        ], dtype=np.int64))
        actual = env.restore_training_states(states)
        _assert_observation_equal(expected, actual)
        np.testing.assert_array_equal(
            np.stack(env.env_method("action_masks")),
            expected_masks,
        )
    finally:
        env.close()
