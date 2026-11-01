"""Real native failure scenes survive close/reset without rerunning search."""
import json
from types import SimpleNamespace

from gymnasium.wrappers import TimeLimit
import numpy as np
import pytest
import torch

from env.core.run_env import SlayTheSpireEnv
from rl.training.environments import TrainingStateWrapper
from rl.training.failure_snapshot import install_truncation_recorder, save_failure_scene
from rl.training.native_vec_env import NativeSearchVecEnv

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _factory(limit):
    return TrainingStateWrapper(TimeLimit(SlayTheSpireEnv(global_map_view=True,
        upper_public_context=True, event_option_observation=True,
        combat_action_contract='autoregressive-v2'), max_episode_steps=limit))


def test_censored_combat_scene_journal_and_original_exception(tmp_path):
    env = NativeSearchVecEnv([lambda: _factory(1000)], threads=1)
    model = SimpleNamespace(get_env=lambda: env, num_timesteps=123)
    try:
        env.begin_episode_rollout(42)
        for _ in range(64):
            if env.envs[0].unwrapped.thick.bc is not None:
                break
            env.step([int(np.flatnonzero(env.env_method('action_masks')[0])[0])])
        wrapped = env.envs[0]
        assert wrapped.unwrapped.thick.bc is not None
        limit = wrapped.env._elapsed_steps + 1
        wrapped.env._max_episode_steps = limit
        rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match='censored whole-game cohort') as failure:
            try:
                env.step([0])  # End turn; preserve the actual failed final state.
            except RuntimeError as error:
                directory = save_failure_scene(model, tmp_path, rank=2, error=error)
                raise
        assert 'slot 0' in str(failure.value)
        assert not env.episode_rollout_finished.any()
        report = json.loads((directory/'manifest.json').read_text())
        assert report['selected_slots'] == [0] and report['global_step'] == 123
        assert report['context']['truncated']
        scene = json.loads((directory/'slot0-scene.json').read_text())
        assert scene['combat']['monsters'] and 'draw' in scene['combat'] and 'exhaust' in scene['combat']
        state = torch.load(directory/'slot0-environment.pt', weights_only=False)
        before = wrapped.export_training_state()
        assert state['base']['actions'] == before['base']['actions']
        assert state['visible_observation_sha256'] == before['visible_observation_sha256']
        restored = _factory(limit)
        try:
            restored.restore_training_state(state)
            np.testing.assert_array_equal(restored.unwrapped.action_masks(), wrapped.unwrapped.action_masks())
            assert restored.unwrapped.thick.bc.to_dict() == wrapped.unwrapped.thick.bc.to_dict()
        finally:
            restored.close()
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        env.close()


def test_nonfatal_truncation_saved_before_reset_and_bounded(tmp_path):
    env = NativeSearchVecEnv([lambda: _factory(1)], threads=1)
    model = SimpleNamespace(get_env=lambda: env, num_timesteps=0)
    try:
        env.seed(42); env.reset()
        install_truncation_recorder(model, tmp_path, rank=0)
        for _ in range(6):
            _, _, done, _ = env.step([int(np.flatnonzero(env.env_method('action_masks')[0])[0])])
            assert done[0] and env.envs[0].env._elapsed_steps == 0
        manifests = list((tmp_path/'failure-scenes/rank0').glob('*/manifest.json'))
        assert len(manifests) == 4
        for manifest in manifests:
            state = torch.load(manifest.parent/'slot0-environment.pt', weights_only=False)
            assert state['wrappers'][0]['state']['elapsed_steps'] == 1
            assert len(state['base']['actions']) == 1
        blocked = tmp_path/'not-a-directory'; blocked.write_text('occupied')
        assert save_failure_scene(model, blocked, rank=0, error=ValueError('original')) is None
    finally:
        env.close()
