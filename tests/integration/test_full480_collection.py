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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full480_fixed_real_rows_ppo_and_ordinary_resume(tmp_path, monkeypatch, latent_training_factory, device):
    from dataclasses import asdict
    import ctypes
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    from rl.algorithms.upper_probe import state_digest
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.training.checkpoints import save_bundle, restore_learner
    from rl.search.full480 import full_seeded
    from rl.search.full480.run_device import lib, check

    assert torch.cuda.is_available(), 'full480 lifecycle requires explicit CUDA artifacts'
    torch.empty(1, device='cuda')
    stack = ctypes.c_size_t()
    lib.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    lib.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    check(lib.cuCtxGetLimit(ctypes.byref(stack), 0), 'read idle oracle stack')
    check(lib.cuCtxSetLimit(0, 1024), 'release idle oracle stack')
    monkeypatch.setattr(full_seeded, 'P', tmp_path / 'runtime')
    envs = []
    def vector():
        env = VecMonitor(NativeSearchVecEnv([latent_training_factory] * 2, threads=2))
        envs.append(env)
        return env
    class Rows(BaseCallback):
        def _on_rollout_start(self):
            self.positions = np.zeros(2, np.int64)
            self.records = []
            self.digest = state_digest(self.model.policy.state_dict())
            vec = self.training_env.venv
            self.original = vec.step_subset
            def step(actions):
                for i, action in actions.items():
                    self.records.append((int(self.positions[i]), i, action,
                        {k: v[i].copy() for k, v in self.model._last_obs.items()},
                        bool(self.model._last_episode_starts[i])))
                    self.positions[i] += 1
                return self.original(actions)
            vec.step_subset = step
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            self.training_env.venv.step_subset = self.original
            b = self.model.rollout_buffer
            assert b.full and b.pos == 32 and len(self.records) == 64
            np.testing.assert_array_equal(self.positions, [32, 32])
            assert np.isfinite(b.values).all() and np.isfinite(b.returns).all()
            assert np.isfinite(b.log_probs).all() and np.isfinite(b.advantages).all()
            for position, i, action, observation, start in self.records:
                assert b.actions[position, i, 0] == action and b.action_masks[position, i, action]
                assert b.episode_starts[position, i] == start
                for key, value in observation.items():
                    np.testing.assert_array_equal(b.observations[key][position, i], value, err_msg=key)
            assert b.search_targets and b.search_actions
            for (position, i), (action, logp) in b.search_actions.items():
                assert b.actions[position, i, 0] == action
                assert b.log_probs[position, i] == pytest.approx(logp, abs=1e-6)
            for key, target in b.search_targets.items():
                legal = b.action_masks[key].astype(bool)
                assert np.isfinite(target[legal]).all() and np.isneginf(target[~legal]).all()
                assert np.exp(target).sum() == pytest.approx(1., abs=1e-6)
            for position, i, _, observation, _ in self.records[::8]:
                tensors = {k: torch.as_tensor(v[None], device=device) for k, v in observation.items()}
                with torch.inference_mode():value = self.model.policy.predict_values(tensors)
                assert b.values[position, i] == pytest.approx(float(value.flatten()[0]), abs=3e-5)
            assert state_digest(self.model.policy.state_dict()) == self.digest
            assert not full_seeded.current_stream.service.process.is_alive()
            assert full_seeded.count['submitted_cpu'] == 0
            assert self.model.logger.name_to_value['pure_rl/search_simulations'] == self.model.deck_search_last_stats['chains']
    try:
        env = vector()
        model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
            n_steps=32, batch_size=32, n_epochs=1, device=device, gamma=1., gae_lambda=1.,
            learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                reward_flow='sequential-v1', deck_combat_encoder='transformer64x2',
                combat_value_target='pre-reward-anchor'))
        configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
        model.deck_search_config = asdict(SearchTrainingConfig(backend='full480', fraction=1.,
            normal_budget=2, boss_budget=2, max_depth=8, timeout_seconds=120.,
            action_selection='search-filter-policy-rank-v1'))
        before = state_digest(model.policy.state_dict())
        updates = []
        hook = model.policy.optimizer.register_step_post_hook(lambda *args: updates.append(1))
        model.learn(64, callback=Rows())
        hook.remove()
        assert len(updates) == 2
        assert model.num_timesteps == 64 and state_digest(model.policy.state_dict()) != before
        steps = {int(s['step']) for s in model.policy.optimizer.state.values()}
        assert steps == {2}
        source = save_bundle(model, tmp_path/'first', rank=0, world_size=1,
            run_id='full480-fixed-test', wandb_id=None, config={'ascension': 0})
        target = vector()
        restored = AccumulatingMaskablePPO.load(source/'model.zip', env=target, device=device)
        restored._setup_learn(64, reset_num_timesteps=False)
        restore_learner(restored, source, rank=0, world_size=1)
        assert restored._recovery_environment_restored
        assert restored._independent_rollout_rng == model._independent_rollout_rng
        assert state_digest(restored.policy.state_dict()) == state_digest(model.policy.state_dict())
        updates.clear()
        hook = restored.policy.optimizer.register_step_post_hook(lambda *args: updates.append(1))
        restored.learn(64, reset_num_timesteps=False, callback=Rows())
        hook.remove()
        assert len(updates) == 2
        assert restored.num_timesteps == 128
        # Phase-specific parameters can be absent from a physical batch.
        assert {int(s['step']) for s in restored.policy.optimizer.state.values()} <= {3, 4}
        save_bundle(restored, tmp_path/'resumed', rank=0, world_size=1,
            run_id='full480-fixed-test', wandb_id=None, config={'ascension': 0})
    finally:
        for env in envs:env.close()
        check(lib.cuCtxSetLimit(0, stack.value), 'restore oracle stack')
