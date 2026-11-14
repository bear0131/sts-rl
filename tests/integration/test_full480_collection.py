"""Original natural-cohort collection with current observations and model."""
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


@pytest.mark.parametrize('dome', [False, True])
def test_full480_natural_rows_cached_dome_and_cancel(
        tmp_path, monkeypatch, latent_training_factory, dome):
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.upper_probe import state_digest
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.search.full480 import full_seeded
    from rl.search.paired_token_rollout import PairedTokenBatch
    from rl.algorithms import full480
    from rl.search.native import observation_as_tensor
    from env.core import vocab

    assert torch.cuda.is_available(), 'full480 lifecycle requires explicit CUDA artifacts'
    import ctypes
    from rl.search.full480.run_device import lib, check
    torch.empty(1, device='cuda')
    stack = ctypes.c_size_t()
    lib.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    lib.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    check(lib.cuCtxGetLimit(ctypes.byref(stack), 0), 'read idle oracle stack')
    check(lib.cuCtxSetLimit(0, 1024), 'release idle oracle stack')
    torch.cuda.empty_cache()
    env = VecMonitor(NativeSearchVecEnv([latent_training_factory] * 2, threads=2))
    monkeypatch.setattr(full_seeded, 'P', tmp_path / 'runtime')
    try:
        model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
            n_steps=2, batch_size=2, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
            learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                reward_flow='sequential-v1', deck_combat_encoder='transformer64x2',
                combat_value_target='pre-reward-anchor'))
        model.nosl_is_az_combat_strategy = 'small-playout'
        model.nosl_is_az_inner_only = True
        model.nosl_is_az_replay_enabled = False
        model.nosl_is_az_upper_strategy = 'model'
        model._outer_return = model._frozen_upper = False
        model.episode_rollout_config = dict(games_per_rank=2, seed=20261111,
            max_steps=64, max_seconds=120., censored_games='error')
        model.episode_rollout_rank = 0
        model.episode_rollout_world_size = 1
        model.episode_rollouts_completed = model.episode_rollout_seed_offset = 0
        model.independent_rollout_config = dict(mode='independent', threads=2,
            upper_batch=64, upper_wait_seconds=0., seed=20261111, max_combat_turns=50)
        model.deck_search_config = dict(backend='full480')
        class Callback(BaseCallback):
            _uses_rollout_observations = False
            stop = False
            def _on_rollout_start(self):
                for i, base in enumerate(env.venv.native_search_batch.envs):
                    if dome:
                        base.thick.gc.set_relics(['Runic Dome'])
                    for _ in range(64):
                        if base.thick.bc is not None:
                            break
                        base.step(int(np.flatnonzero(base.action_masks())[0]))
                    assert base.thick.bc is not None
                    base.thick.bc.apply_custom_state({'player': {'hp': 1, 'powers': [('STRENGTH', -999)]}})
                    base._invalidate_backend_cache()
                    env.venv._save_obs(i, base._obs())
                model._last_obs = env.venv._obs_from_buf()
            def _on_step(self):
                return not self.stop
        _, callback = model._setup_learn(1, callback=Callback())
        def no_cpu_dome(*args):
            raise AssertionError('Dome decisions must reuse the original GPU policy queue')
        monkeypatch.setattr(PairedTokenBatch, 'dome_policy', no_cpu_dome)
        pending, checked = {}, []
        begin, take = full480.upper_begin, full480.upper_take
        def begin_checked(obs, masks):
            ident = begin(obs, masks)
            pending[ident] = obs
            return ident
        def take_checked(ident):
            answer = take(ident)
            if answer is not None:
                obs = pending.pop(ident)
                rows = np.flatnonzero(obs['phase'].reshape(-1) == vocab.PHASE_VOCAB_INDEX['combat'])
                if len(rows):
                    with torch.inference_mode():
                        expected, _ = model.policy.core.actor_critic(observation_as_tensor(
                            {k:v[rows] for k,v in obs.items()}, 'cpu'))
                    np.testing.assert_allclose(answer['logits'][rows], expected.numpy(), atol=3e-5, rtol=1e-5)
                    checked.extend(rows.tolist())
            return answer
        monkeypatch.setattr(full480, 'upper_begin', begin_checked)
        monkeypatch.setattr(full480, 'upper_take', take_checked)
        before = state_digest(model.policy.state_dict())
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        assert model.episode_rollout_stats['completed_games'] == 2
        assert model.episode_rollout_stats['censored_games'] == 0
        assert not full_seeded.current_stream.service.process.is_alive()
        if dome:
            # The terminal-search case can select End Turn and die on its
            # first action; only this case guarantees intermediate live rows.
            assert model._independent_rollout_diagnostic['combat_observation_rows_omitted'] > 0
            assert checked and full_seeded.count['roots'] == 0
            assert model._small_playout_stats['dome_policy_decisions'] > 0
        else:
            assert model._small_playout_samples
            for row in model._small_playout_samples:
                assert len(row['action_bits']) == len(row['visits']) == len(row['values']) == len(row['prior_logits'])
                assert row['observation']['combat_legal'].any()
                assert np.isfinite(row['values']).all()
        callback.stop = True
        assert not model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        assert not full_seeded.current_stream.service.process.is_alive()
        assert state_digest(model.policy.state_dict()) == before
        assert not model.policy.optimizer.state
    finally:
        env.close()
        check(lib.cuCtxSetLimit(0, stack.value), 'restore oracle stack')
