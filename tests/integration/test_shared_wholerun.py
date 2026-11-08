from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from env.core.run_env import SlayTheSpireEnv
from env.core import vocab
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.shared_wholerun import SharedWholeRunPolicy
from rl.policies.checkpoint import load_checkpoint
from rl.training.reward import StrategicRewardWrapper
from rl.training.environments import TrainingStateWrapper
from rl.training.checkpoints import save_bundle, restore_learner

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def make_env():
    return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
        card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2')),
        max_episode_steps=6000))


def make_model(env, device):
    return AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261108,
        n_steps=32, batch_size=32, n_epochs=1, device=device,
        gamma=1., gae_lambda=1., learning_rate=1e-4,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=.9))


@pytest.mark.parametrize('callback_kind', ['read_only', 'unknown', 'subclass', 'abort'])
def test_prefill_respects_callback_mutation_and_restores_environment_hooks(callback_kind):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class ReadOnly(BaseCallback):
        _uses_rollout_observations = False

        def _on_step(self):
            return callback_kind != 'abort'

    class Mutating(ReadOnly):
        # An undeclared subclass must not inherit permission to prefill early.
        def _on_step(self):
            self.model._last_obs['gold'][:] += 1
            expected.append(self.model._last_obs['gold'].copy())
            return True

    class Unknown(BaseCallback):
        _on_step = Mutating._on_step

    expected, prefills = [], []
    env = DummyVecEnv([make_env, make_env])
    model = make_model(env, 'cpu')
    buffer = model.rollout_buffer
    original_prefill, original_wait, original_async = buffer.prefill_observation, env.step_wait, env.step_async

    def prefill(obs):
        prefills.append(buffer.pos)
        original_prefill(obs)

    buffer.prefill_observation = prefill
    selected = {'read_only': ReadOnly, 'abort': ReadOnly, 'unknown': Unknown, 'subclass': Mutating}[callback_kind]
    _, callback = model._setup_learn(64, callback=CallbackList([selected()]))
    result = model.collect_rollouts(env, callback, buffer, 32)
    assert env.step_async == original_async and env.step_wait == original_wait
    assert result == (callback_kind != 'abort')
    if callback_kind in ('read_only', 'abort'):
        assert prefills == list(range(32 if result else 1))
    else:
        assert not prefills
        np.testing.assert_array_equal(buffer.observations['gold'], np.stack(expected))
    if not result:
        buffer.reset()
        assert buffer._prefilled is None and buffer.pos == 0
    env.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_ppo_two_rounds_and_resume(tmp_path, device):
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    env = DummyVecEnv([make_env, make_env])
    model = make_model(env, device)
    model.set_logger(configure(folder=None, format_strings=[]))
    recorded = {}
    original_record = model.logger.record
    def record(key, value, *args, **kwargs):
        recorded[key] = value
        original_record(key, value, *args, **kwargs)
    model.logger.record = record
    core = model.policy.core
    assert core.combat.transformer is core.overworld.encoder.transformer
    assert core.combat.card is core.upper_tokenizer.card
    params = [p for g in model.policy.optimizer.param_groups for p in g['params']]
    assert len(params) == len({id(p) for p in params})
    before = deepcopy(model.policy.state_dict())
    from rl.tasks.wholerun import StSMetricsCallback
    model.learn(total_timesteps=128, callback=StSMetricsCallback())
    assert model.num_timesteps == 128 and model._n_updates == 2
    totals = deepcopy(model.pure_rl_game_totals)
    assert not hasattr(model.policy.core, '_collection_graphs')
    if device == 'cuda':
        assert not model._shared_collection_graphs.closed
    assert recorded['pure_rl/combat_samples'] > 0
    assert not model.nosl_is_az and not model.hierarchical_combat_hp
    assert not hasattr(model.policy, 'upper_optimizer')
    for prefix in ('core.combat.transformer.', 'core.combat.end.', 'core.overworld.fixed_head.'):
        assert any(not torch.equal(v, before[k]) for k, v in model.policy.state_dict().items()
                   if k.startswith(prefix)), prefix
    assert recorded['pure_rl/search_simulations'] == 0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    bundle = save_bundle(model, tmp_path/'recovery', rank=0, world_size=1,
                         run_id='pure-test', wandb_id=None, config={})
    assert bundle is not None
    model.save(tmp_path/'policy.zip')
    loaded = load_checkpoint(tmp_path/'policy.zip', device=device)
    assert loaded.pure_rl_game_totals == totals
    assert not hasattr(loaded.policy.core, '_collection_graphs')
    assert not hasattr(loaded, '_shared_collection_graphs')
    assert isinstance(loaded, AccumulatingMaskablePPO)
    assert loaded.policy.core.combat.transformer is loaded.policy.core.overworld.encoder.transformer
    obs = model._last_obs
    masks = np.stack(env.env_method('action_masks'))
    for policy in (model.policy, loaded.policy): policy.set_training_mode(False)
    x, _ = model.predict(obs, action_masks=masks, deterministic=True)
    y, _ = loaded.predict(obs, action_masks=masks, deterministic=True)
    np.testing.assert_array_equal(x, y)
    resumed_env = DummyVecEnv([make_env, make_env])
    loaded.set_env(resumed_env, force_reset=False)
    restore_learner(loaded, bundle, rank=0, world_size=1)
    optimizer = loaded.policy.optimizer
    old_steps = {p: int(s['step']) for p, s in optimizer.state.items()}
    shared = next(loaded.policy.core.combat.transformer.parameters())
    loaded.learn(total_timesteps=128, reset_num_timesteps=False, callback=StSMetricsCallback())
    assert loaded.num_timesteps == 256
    assert loaded.pure_rl_game_totals['natural'] >= totals['natural']
    assert loaded.pure_rl_game_totals['censored'] >= totals['censored']
    assert loaded.pure_rl_game_totals['start_step'] == 0
    # A phase-specific head need not occur in both batches. Shared backbone
    # moments must advance; every existing head must retain its prior steps.
    assert int(optimizer.state[shared]['step']) > old_steps[shared]
    assert all(int(optimizer.state[p]['step']) >= step for p, step in old_steps.items())
    env.close(); resumed_env.close()


@pytest.mark.parametrize('grow', [False, True])
@pytest.mark.parametrize('scene,backend', [
    ('ordinary', 'native'), ('campfire', 'native'),
    ('ordinary', 'native-subproc'),
])
def test_native_distributed_payload_and_resume(tmp_path, monkeypatch, capsys, grow,
                                              scene, backend, campfire_env_type):
    import json
    import sys
    from types import SimpleNamespace
    from rl.training import distributed_sb3
    from rl.training import reward_monitor, campfire_monitor

    factory = make_env
    if scene == 'campfire':
        from stable_baselines3.common.vec_env import VecMonitor
        from rl.training.native_vec_env import NativeSearchVecEnv
        def factory():
            return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(campfire_env_type(
                card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2')),
                max_episode_steps=6000))
        monkeypatch.setattr(distributed_sb3, 'make_env', lambda model, args:
            VecMonitor(NativeSearchVecEnv([factory for _ in range(args.n_envs)], threads=2)))

    env = DummyVecEnv([factory, factory])
    model = make_model(env, 'cpu')
    checkpoint = tmp_path/'initial.zip'
    model.save(checkpoint)
    env.close()
    payloads = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=lambda **kw: SimpleNamespace(finish=lambda **kw: None),
        config=SimpleNamespace(update=lambda *a, **kw: None),
        define_metric=lambda *a, **kw: None,
        log=lambda payload: payloads.append(json.loads(json.dumps(payload, allow_nan=False)))))
    argv = ['distributed_sb3', '--rank', '0', '--world-size', '1',
        '--checkpoint', str(checkpoint), '--updates', '2', '--n-envs', '2',
        '--n-steps', '32', '--batch-size', '32', '--n-epochs', '1',
        '--gradient-accumulation-steps', '1', '--device', 'cpu', '--vec', backend,
        '--envs-per-worker', '2', '--start-method', 'spawn',
        '--native-search-threads', '2', '--gamma', '1', '--gae-lambda', '1',
        '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning',
        '--remote-root', str(tmp_path/'transport'), '--ssh-control-dir', str(tmp_path/'ssh'),
        '--run-id', 'shared-pure-lifecycle', '--wandb', '--wandb-id', 'shared-pure-lifecycle',
        '--training-state-dir', str(tmp_path/'state'), '--save-every-updates', '1',
        '--max-checkpoints', '2', '--seed', '20261108']
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    bundle = tmp_path/'state/rank0/step-000000000128'
    assert json.loads((bundle/'manifest.json').read_text())['schema_version'] == 2
    resumed = list(argv)
    resumed[resumed.index('--checkpoint')+1] = str(bundle/'model.zip')
    resumed += ['--resume-state', str(bundle)]
    if grow:
        resumed[resumed.index('--n-envs')+1] = '4'
        resumed += ['--resume-grow-environments']
    monkeypatch.setattr(sys, 'argv', resumed)
    distributed_sb3.main()
    output = capsys.readouterr().out
    assert 'ENVIRONMENT_RESTORED ' in output
    records = [json.loads(line) for line in output.splitlines()
               if line.startswith('{') and '"rollout_sec"' in line]
    assert [p['global_step'] for p in payloads] == ([64, 128, 256, 384] if grow else [64, 128, 192, 256])
    assert ('ENVIRONMENT_EXPANDED_ON_RESUME ' in output) == grow
    assert len(records) == 4
    assert [row['rollout_round'] for row in records] == [1, 2, 3, 4]
    assert [p['legacy_step'] for p in payloads] == [1, 2, 3, 4]
    for row, payload in zip(records, payloads, strict=True):
        for key in ('room_lambda', 'combat_samples', 'upper_samples', 'in_room_edges',
                    'boundary_edges', 'search_simulations', 'return_version',
                    'censored_episodes_since_start'):
            assert row['pure_rl/'+key] == payload['pure_rl/'+key]
        assert row['pure_rl/search_simulations'] == 0
        assert row['pure_rl/room_lambda'] == .9
        count = 128 if grow and row['rollout_round'] > 2 else 64
        assert row['pure_rl/combat_samples'] + row['pure_rl/upper_samples'] == count
        assert row['optimizer_steps'] == count // 32
        assert row['pure_rl/update/count'] == count
        assert row['pure_rl/value_pre/all/count'] == count
        for key in ('value_loss', 'policy_loss', 'entropy', 'approx_kl', 'clip_fraction',
                    'value_pre/all/mse', 'value_pre/combat/mse', 'value_pre/upper/mse'):
            assert np.isfinite(row['pure_rl/'+key])
            assert row['pure_rl/'+key] == payload['pure_rl/'+key]
        assert row['pure_rl/value_loss'] >= 0
        assert 0 <= row['pure_rl/clip_fraction'] <= 1
        assert row['completed_full_games_global'] == row['pure_rl/natural_games_global']
        assert row['completed_full_games_global'] == payload['game/completed_full_games_global']
        for key in {**reward_monitor.counts(None), **campfire_monitor.counts(None)}:
            assert row['upper/'+key] == payload['upper/'+key]
        for key, value in row.items():
            if key.startswith(('upper/campfire_', 'upper/reward_')) and key.endswith(
                    ('choice_rate', 'pickup_rate', 'policy_probability')):
                assert 0 <= value <= 1
                assert value == payload[key]
    if scene == 'campfire':
        assert sum(r['upper/campfire_decisions'] for r in records) > 0
        assert sum(r['upper/reward_gold_resolved'] for r in records) > 0
        assert sum(r['upper/reward_potion_resolved'] for r in records) > 0
    assert [r['completed_full_games_global'] for r in records] == sorted(
        r['completed_full_games_global'] for r in records)


def test_grow_live_pure_ppo_environments_without_resetting_source_slots(tmp_path):
    original_env = DummyVecEnv([make_env, make_env])
    model = make_model(original_env, 'cpu')
    model.learn(total_timesteps=64)
    before_obs = deepcopy(model._last_obs)
    before_starts = model._last_episode_starts.copy()
    model.ppo_rollouts_completed = 1
    bundle = save_bundle(model, tmp_path/'state', rank=0, world_size=1,
        run_id='pure-growth', wandb_id=None, config=dict(n_envs=2, seed=20261108))
    source_rng = torch.get_rng_state().clone()
    grown_env = DummyVecEnv([make_env for _ in range(4)])
    resumed = AccumulatingMaskablePPO.load(bundle/'model.zip', env=grown_env, device='cpu')
    resumed._setup_learn(total_timesteps=128, reset_num_timesteps=False)
    restore_learner(resumed, bundle, rank=0, world_size=1, grow_environments=True)
    for key, expected in before_obs.items():
        np.testing.assert_array_equal(resumed._last_obs[key][:2], expected)
    np.testing.assert_array_equal(resumed._last_episode_starts[:2], before_starts)
    assert resumed._last_episode_starts[2:].all()
    assert torch.equal(source_rng, torch.get_rng_state())
    growth = resumed._recovery_environment_growth
    assert growth['restored_environments'] == 2 and growth['appended_environments'] == 2
    assert growth['appended_first_seed'] == 20261108 + 64 + 2
    shared = next(resumed.policy.core.combat.transformer.parameters())
    old_step = int(resumed.policy.optimizer.state[shared]['step'])
    resumed.learn(total_timesteps=128, reset_num_timesteps=False)
    assert int(resumed.policy.optimizer.state[shared]['step']) > old_step
    original_env.close(); grown_env.close()
