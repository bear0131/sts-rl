"""Live simulator recovery from a backend seed and committed action journal."""
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit

from env.core.run_env import SlayTheSpireEnv
from env.core.game_env import SimGameEnv
from env.core import action_space as A
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


@pytest.mark.parametrize('json_history', [False, True])
def test_run_fork_history_isolation_and_replay(json_history):
    import copy
    import json

    env = SlayTheSpireEnv(global_map_view=True)
    restored = SlayTheSpireEnv(global_map_view=True)
    branches = []
    try:
        env.reset(seed=123)
        _advance_to_combat(env)
        bits = int(env.native_combat_actions()[0])
        action = env._native_action_to_thin(bits)
        assert action is not None
        env._nosl_cpp_pending = (action, bits)
        env.step(action)
        state = env.export_training_state()
        if json_history:
            state['actions'] = json.loads(json.dumps(state['actions']))
            env.restore_training_state(state)
        history = copy.deepcopy(env._episode_action_journal)
        parent_obs = env._obs()
        branches = [env.fork_run(), env.fork_run()]
        first, sibling = branches
        for _ in range(8):
            result = first.step(_first_action(first))
            if result[2] or result[3]:
                break
        snapshot = first.export_training_state()
        observation, _ = restored.restore_training_state(snapshot)
        _assert_observation_equal(first._obs(), observation)
        np.testing.assert_array_equal(first.action_masks(), restored.action_masks())
        assert restored._episode_action_journal == first._episode_action_journal
        assert env._episode_action_journal == sibling._episode_action_journal == history
        _assert_observation_equal(env._obs(), parent_obs)
        _assert_observation_equal(sibling._obs(), parent_obs)
        # Mutable records and JSON's pending lists must not alias siblings.
        first._episode_action_journal[0]['action'] = -1
        for record in first._episode_action_journal:
            if isinstance(record.get('native_pending'), list):
                record['native_pending'][0] = -1
        assert env._episode_action_journal == sibling._episode_action_journal == history
        snapshot['actions'][0]['action'] = -2
        assert restored._episode_action_journal[0]['action'] != -2
    finally:
        for branch in branches:
            branch.close()
        env.close()
        restored.close()


def test_backend_step_observation_remains_opt_in_to_omission():
    regular = SimGameEnv()
    transition_only = SimGameEnv(emit_step_observation=False)
    np.testing.assert_equal(regular.reset(seed=123), transition_only.reset(seed=123))
    for _ in range(64):
        action = regular.legal_semantic_actions()[0]
        expected = regular.step(action)
        actual = transition_only.step(action)
        assert isinstance(expected[0], dict)
        assert actual[0] is None
        np.testing.assert_equal(expected[1:], actual[1:])
        np.testing.assert_equal(regular.obs_canonical(), transition_only.obs_canonical())
        if regular.bc is not None:
            break
    assert regular.bc is not None


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('branch', [False, True])
def test_omitted_backend_observation_preserves_terminal_and_boundary(
        short_combat_env_type, native, branch):
    # A custom backend retains the public canonical-returning contract. The
    # built-in Gym backend omits only that return, including in search forks.
    contract = 'native-v1' if native else 'legacy'
    expected_parent = short_combat_env_type(
        backend_factory=SimGameEnv, combat_action_contract=contract)
    actual_parent = short_combat_env_type(combat_action_contract=contract)
    expected = actual = None
    try:
        np.testing.assert_equal(expected_parent.reset(seed=123), actual_parent.reset(seed=123))
        expected = expected_parent.fork_combat() if branch else expected_parent
        actual = actual_parent.fork_combat() if branch else actual_parent
        parent_state = actual_parent.thick.bc.to_dict()
        saw_boundary = saw_terminal = False
        for index in range(64):
            if native:
                legal = expected.native_combat_actions()
                non_end = [int(a) for a in legal if int(a) != (4 << 29)]
                action = (4 << 29) if index == 0 else (non_end or legal)[0]
                assert action in legal
                first = expected.step_native_combat(action)
                second = actual.step_native_combat(action)
            else:
                expected._nosl_value_capture_pending = True
                actual._nosl_value_capture_pending = True
                legal = np.flatnonzero(expected.action_masks())
                non_end = [int(a) for a in legal if int(a) != A.END_TURN]
                action = A.END_TURN if index == 0 else int((non_end or legal)[0])
                first = expected.step(action)
                second = actual.step(action)
            np.testing.assert_equal(first, second)
            saw_boundary |= 'nosl_value_boundary_observation' in first[4]
            saw_terminal |= 'nosl_value_terminal_target' in first[4]
            if saw_terminal:
                assert 'nosl_value_terminal_outcome' in first[4]
                if native or not branch:
                    assert 'nosl_value_terminal_observation' in first[4]
                break
        assert saw_terminal
        if native:
            assert saw_boundary
        if branch:
            np.testing.assert_equal(actual_parent.thick.bc.to_dict(), parent_state)
    finally:
        if branch:
            if expected is not None:
                expected.close()
            if actual is not None:
                actual.close()
        expected_parent.close()
        actual_parent.close()


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


@pytest.mark.parametrize('factory_omits_occupancy', [False, True])
def test_training_wrapper_restores_occupancy_reward_and_time_limit_state(factory_omits_occupancy):
    env, occupancy, time_limit, reward = _training_env()
    restored, restored_occupancy, restored_time_limit, restored_reward = (
        _training_env()
    )
    if factory_omits_occupancy:
        restored.env = restored_occupancy.env
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
        restored_occupancy = restored.env
        _assert_observation_equal(expected_observation, actual_observation)
        assert restored_occupancy.startup_occupancy_state() == (
            occupancy.startup_occupancy_state()
        )
        assert restored_time_limit._elapsed_steps == time_limit._elapsed_steps
        assert restored_reward._prev == reward._prev
        for _ in range(8):
            action = _first_action(env)
            expected = env.step(action)
            actual = restored.step(action)
            _assert_observation_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
    finally:
        env.close()
        restored.close()


def test_retired_unreleased_occupancy_cannot_silently_freeze_restored_factory():
    env, occupancy, *_ = _training_env()
    restored, old_wrapper, *_ = _training_env()
    restored.env = old_wrapper.env
    try:
        env.reset(seed=321)
        with pytest.raises(ValueError, match='environment-reset fork'):
            restored.restore_training_state(env.export_training_state())
        assert not isinstance(restored.env, StartupOccupancyWrapper)
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
