"""The original collector, real PPO updater and ordinary recovery boundary."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(240)]


def test_full480_real_rows_updates_restore_and_cancel(tmp_path, monkeypatch, latent_training_factory):
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    from rl.algorithms.upper_probe import state_digest
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.policies.checkpoint import load_checkpoint
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.training.checkpoints import save_bundle, restore_learner
    from rl.search.full480 import full_seeded

    assert torch.cuda.is_available(), 'full480 lifecycle requires explicit CUDA artifacts'
    import ctypes
    from rl.search.full480.run_device import lib, check
    # A preceding in-process simulator oracle may leave its stack reservation.
    # The real learner owns no simulator; only the child retains the 64 KiB stack.
    torch.cuda.init()
    stack = ctypes.c_size_t()
    lib.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    check(lib.cuCtxGetLimit(ctypes.byref(stack), 0), 'read idle oracle stack')
    check(lib.cuCtxSetLimit(0, 1024), 'release idle oracle stack')
    torch.cuda.empty_cache()
    monkeypatch.setattr(full_seeded, 'P', tmp_path / 'runtime')
    env = VecMonitor(NativeSearchVecEnv([latent_training_factory] * 2, threads=2))
    target = None
    try:
        model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
            n_steps=32, batch_size=32, n_epochs=1, device='cuda', gamma=1., gae_lambda=1.,
            learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                reward_flow='sequential-v1', deck_combat_encoder='transformer64x2',
                combat_value_target='pre-reward-anchor'))
        configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
        model.deck_search_config = asdict(SearchTrainingConfig(backend='full480', fraction=1.,
            normal_budget=16, boss_budget=16, max_depth=8, timeout_seconds=60.,
            action_selection='search-filter-policy-rank-v1'))
        executed = [[] for _ in range(2)]
        original = env.venv.step_subset
        def record(actions):
            for i, action in actions.items():
                executed[i].append(action)
            return original(actions)
        monkeypatch.setattr(env.venv, 'step_subset', record)
        before = state_digest(model.policy.state_dict())
        model.learn(64)
        assert model.num_timesteps == 64
        assert state_digest(model.policy.state_dict()) != before
        assert max(int(s['step']) for s in model.policy.optimizer.state.values()) == 2
        buffer = model.rollout_buffer
        np.testing.assert_array_equal(buffer.actions[:, :, 0], np.asarray(executed).T)
        assert np.all(buffer.row_positions == 32) and np.isfinite(buffer.returns).all()
        assert len(buffer.search_targets) == len(buffer.search_actions) > 0
        for (position, index), (action, logp) in buffer.search_actions.items():
            assert action == buffer.actions[position, index, 0]
            assert buffer.action_masks[position, index, action]
            assert buffer.log_probs[position, index] == pytest.approx(logp, abs=1e-6)
        assert not full_seeded.current_stream.service.process.is_alive()
        rng = deepcopy(model._full480_rng)
        bundle = save_bundle(model, tmp_path / 'first', rank=0, world_size=1,
            run_id='full480-lifecycle', wandb_id=None, config={'ascension': 0})
        restored = load_checkpoint(bundle / 'model.zip', device='cuda')
        target = VecMonitor(NativeSearchVecEnv([latent_training_factory] * 2, threads=2))
        restored.set_env(target, force_reset=False)
        restored._setup_learn(64, reset_num_timesteps=False)
        restore_learner(restored, bundle, rank=0, world_size=1)
        assert restored._recovery_environment_restored and restored._full480_rng == rng
        assert restored.full480_search_rng_version == model.full480_search_rng_version == 'pcg64-bulk-state-v1'
        assert state_digest(restored.policy.state_dict()) == state_digest(model.policy.state_dict())
        restored.learn(64, reset_num_timesteps=False)
        assert restored.num_timesteps == 128
        assert max(int(s['step']) for s in restored.policy.optimizer.state.values()) == 4
        assert restored._full480_rng != rng
        save_bundle(restored, tmp_path / 'second', rank=0, world_size=1,
            run_id='full480-lifecycle', wandb_id=None, config={'ascension': 0})
        assert not full_seeded.current_stream.service.process.is_alive()
        class Stop(BaseCallback):
            def _on_step(self):
                return False
        callback = Stop(); callback.init_callback(restored)
        digest = state_digest(restored.policy.state_dict())
        assert not restored.collect_rollouts(target, callback, restored.rollout_buffer, 32)
        assert state_digest(restored.policy.state_dict()) == digest
        assert not restored.rollout_buffer.full
        assert not full_seeded.current_stream.service.process.is_alive()
    finally:
        if target is not None:
            target.close()
        env.close()
        check(lib.cuCtxSetLimit(0, stack.value), 'restore oracle stack')
