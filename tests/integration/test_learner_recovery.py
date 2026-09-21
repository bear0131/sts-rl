"""Recovery keeps completed replay, Adam and sampling RNG at the same boundary."""
import copy
import hashlib
import json
import random
import shutil

import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.training.checkpoints import read_bundle, restore_learner, save_bundle
from rl.training.environments import TrainingStateWrapper
from rl.training.reward import ShapedRewardWrapper
from tests.integration.test_training_smoke import train_once, assert_nested_equal

pytestmark = [pytest.mark.native, pytest.mark.training]


def test_replay_rng_and_next_update_match_after_recovery(tmp_path, short_combat_env_type):
    model, env = train_once(True, env_type=short_combat_env_type,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_simulations=4, nosl_is_az_replay_capacity=128,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1)
    try:
        model._prepare_nosl_is_az_replay()
        model.configure_nosl_is_az_inner_only()
        model.nosl_is_az_updates = 1
        # Simulate a process killed during the next generation's write, before
        # manifest publication; recovery must not get stuck at this step.
        interrupted = tmp_path / 'rank0' / f'step-{model.num_timesteps:012d}'
        interrupted.mkdir(parents=True)
        (interrupted/'model.zip').write_bytes(b'incomplete previous write')
        path = save_bundle(model, tmp_path, rank=0, world_size=1,
            run_id='test', wandb_id='same', config={})
        assert read_bundle(path, rank=0, world_size=1)['schema_version'] == 2
        assert (path / 'environment.pt').is_file()
        abandoned = list((tmp_path/'rank0').glob('.incomplete-*'))
        assert len(abandoned) == 1
        assert (abandoned[0]/'model.zip').read_bytes() == b'incomplete previous write'
        expected_random = (random.random(), np.random.random(), torch.rand(4))
        model._train_nosl_is_az(prepare_replay=False)
        recovered = AccumulatingMaskablePPO.load(path/'model.zip', device='cpu')
        recovered.configure_nosl_is_az_inner_only()
        restore_learner(recovered, path, rank=0, world_size=1)
        actual_random = (random.random(), np.random.random(), torch.rand(4))
        assert_nested_equal(expected_random, actual_random)
        for key in ('_nosl_is_az_fresh_samples', '_nosl_is_az_replay_prepare_stats'):
            setattr(recovered, key, copy.deepcopy(getattr(model, key)))
        recovered._train_nosl_is_az(prepare_replay=False)
        assert_nested_equal(model.policy.state_dict(), recovered.policy.state_dict())
        assert_nested_equal(model.policy.optimizer.state_dict(), recovered.policy.optimizer.state_dict())
        first, second = model._nosl_is_az_replay, recovered._nosl_is_az_replay
        assert (first.position, first.size, first.total_added) == (second.position, second.size, second.total_added)
        np.testing.assert_array_equal(first.sample_ids[:first.size], second.sample_ids[:second.size])
        with pytest.raises(ValueError, match='rank/world'):
            read_bundle(path, rank=1, world_size=2)
        with (path/'learner.pt').open('ab') as stream:
            stream.write(b'corruption')
        with pytest.raises(ValueError, match='checksum'):
            read_bundle(path, rank=0, world_size=1)
    finally:
        env.close()


def _checkpointable_vec_env():
    return VecMonitor(DummyVecEnv([
        lambda: TrainingStateWrapper(TimeLimit(
            ShapedRewardWrapper(SlayTheSpireEnv()), max_episode_steps=12,
        )),
        lambda: TrainingStateWrapper(TimeLimit(
            ShapedRewardWrapper(SlayTheSpireEnv()), max_episode_steps=12,
        )),
    ]))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema2_restores_live_vec_monitor_and_next_transition(tmp_path):
    env = _checkpointable_vec_env()
    restored_env = _checkpointable_vec_env()
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
        seed=17,
        device="cpu",
    )
    try:
        model.learn(8)
        path = save_bundle(
            model, tmp_path, rank=0, world_size=1,
            run_id="environment", wandb_id="same", config={},
        )
        expected_observation = copy.deepcopy(model._last_obs)
        expected_starts = model._last_episode_starts.copy()
        expected_returns = env.episode_returns.copy()
        expected_lengths = env.episode_lengths.copy()

        recovered = AccumulatingMaskablePPO.load(
            path / "model.zip", env=restored_env, device="cpu"
        )
        recovered._setup_learn(
            total_timesteps=8,
            callback=None,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        manifest = restore_learner(
            recovered, path, rank=0, world_size=1
        )
        assert manifest["schema_version"] == 2
        assert recovered._recovery_environment_restored is True
        assert recovered._last_obs.keys() == expected_observation.keys()
        for key in recovered._last_obs:
            np.testing.assert_array_equal(
                recovered._last_obs[key], expected_observation[key]
            )
        np.testing.assert_array_equal(
            recovered._last_episode_starts, expected_starts
        )
        np.testing.assert_array_equal(restored_env.episode_returns, expected_returns)
        np.testing.assert_array_equal(restored_env.episode_lengths, expected_lengths)

        masks = np.stack(env.env_method("action_masks"))
        actions = np.asarray([
            np.flatnonzero(mask)[0] for mask in masks
        ], dtype=np.int64)
        expected = env.step(actions)
        actual = restored_env.step(actions)
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
    finally:
        env.close()
        restored_env.close()


def test_schema1_bundle_remains_learner_only_compatible(tmp_path):
    env = _checkpointable_vec_env()
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
        seed=9, device="cpu",
    )
    try:
        model.learn(4)
        current = save_bundle(
            model, tmp_path / "current", rank=0, world_size=1,
            run_id="legacy", wandb_id="same", config={},
        )
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        shutil.copy2(current / "model.zip", legacy / "model.zip")
        learner = torch.load(
            current / "learner.pt", map_location="cpu", weights_only=False
        )
        learner["schema_version"] = 1
        for key in (
            "last_episode_starts", "nosl_is_az_pending_fights",
            "ep_info_buffer", "ep_success_buffer", "episode_num",
        ):
            learner.pop(key)
        torch.save(learner, legacy / "learner.pt")
        manifest = {
            "schema_version": 1,
            "rank": 0,
            "world_size": 1,
            "global_step": model.num_timesteps,
            "run_id": "legacy",
            "wandb_id": "same",
            "environment_resume": "restart; unfinished fights discarded",
            "config": {},
            "sha256": {
                name: _sha256(legacy / name)
                for name in ("model.zip", "learner.pt")
            },
        }
        (legacy / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        recovered = AccumulatingMaskablePPO.load(
            legacy / "model.zip", device="cpu"
        )
        loaded = restore_learner(
            recovered, legacy, rank=0, world_size=1
        )
        assert loaded["schema_version"] == 1
        assert recovered._recovery_environment_restored is False
    finally:
        env.close()


def test_schema2_fork_can_reset_environment_geometry(tmp_path):
    source_env = _checkpointable_vec_env()
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, source_env, n_steps=2, batch_size=2, n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
        device="cpu", seed=23,
    )
    try:
        model.learn(4)
        path = save_bundle(
            model, tmp_path, rank=0, world_size=1,
            run_id="source", wandb_id="source", config={},
        )
        target_env = VecMonitor(DummyVecEnv([
            lambda: TrainingStateWrapper(TimeLimit(
                ShapedRewardWrapper(SlayTheSpireEnv()), max_episode_steps=12,
            )) for _ in range(3)
        ]))
        try:
            recovered = AccumulatingMaskablePPO.load(
                path / "model.zip", env=target_env, device="cpu"
            )
            recovered._setup_learn(
                total_timesteps=4, callback=None,
                reset_num_timesteps=False, progress_bar=False,
            )
            manifest = restore_learner(
                recovered, path, rank=0, world_size=1,
                restore_environment=False,
            )
            assert manifest["schema_version"] == 2
            assert recovered._recovery_environment_restored is False
            assert recovered._last_obs["phase"].shape[0] == 3
            assert len(recovered._last_episode_starts) == 3
            assert len(recovered._nosl_is_az_pending_fights) == 3
            np.testing.assert_array_equal(target_env.episode_returns, np.zeros(3))
            assert target_env.episode_count == 0
        finally:
            target_env.close()
    finally:
        source_env.close()


def test_schema2_fork_reset_restores_learner_rng_after_env_reset(tmp_path):
    source_env = _checkpointable_vec_env()
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, source_env, n_steps=2, batch_size=2, n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
        device="cpu", seed=31,
    )
    try:
        model.learn(4)
        path = save_bundle(
            model, tmp_path, rank=0, world_size=1,
            run_id="source", wandb_id="source", config={},
        )
        target_env = VecMonitor(DummyVecEnv([
            lambda: TrainingStateWrapper(TimeLimit(
                ShapedRewardWrapper(SlayTheSpireEnv()), max_episode_steps=12,
            )) for _ in range(3)
        ]))
        try:
            recovered = AccumulatingMaskablePPO.load(
                path / "model.zip", env=target_env, device="cpu"
            )
            recovered._setup_learn(
                total_timesteps=4, callback=None,
                reset_num_timesteps=False, progress_bar=False,
            )
            expected = torch.load(
                path / "learner.pt", map_location="cpu", weights_only=False
            )
            restore_learner(
                recovered, path, rank=0, world_size=1,
                restore_environment=False,
            )
            np.testing.assert_array_equal(
                np.random.get_state()[1], expected["numpy_rng"][1]
            )
            assert random.getstate() == expected["python_rng"]
            assert torch.equal(torch.get_rng_state(), expected["torch_rng"])
        finally:
            target_env.close()
    finally:
        source_env.close()
