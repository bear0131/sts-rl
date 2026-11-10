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


def make_env(staged=False):
    return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
        card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2',
        combat_exit_anchor=staged, reward_flow='sequential-v1' if staged else 'full')),
        max_episode_steps=6000))


def make_model(env, device, staged=False):
    return AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261108,
        n_steps=32, batch_size=32, n_epochs=1, device=device,
        gamma=1., gae_lambda=1., learning_rate=1e-4,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1. if staged else .9,
            combat_value_target='pre-reward-anchor' if staged else 'whole-run',
            reward_flow='sequential-v1' if staged else 'full'))


def test_anchor_bundle_continues_weights_adam_rng_and_live_games(tmp_path):
    from rl.training.anchor_resume import migrate_anchor_bundle
    def factory(anchor=False):
        return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=anchor, reward_flow='sequential-v1')), max_episode_steps=6000))
    env = DummyVecEnv([factory])
    model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261108,
        n_steps=32, batch_size=32, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
        learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                           reward_flow='sequential-v1'))
    model.learn(64)
    source = save_bundle(model, tmp_path/'original', rank=0, world_size=1,
        run_id='anchor-resume-test', wandb_id=None, config={'ascension': 0})
    dest = tmp_path/'migrated'
    report = migrate_anchor_bundle(source, dest)
    assert report['global_step'] == 64
    assert (source/'learner.pt').read_bytes() == (dest/'learner.pt').read_bytes()
    before = torch.load(source/'environment.pt', weights_only=False)
    after = torch.load(dest/'environment.pt', weights_only=False)
    assert before['states'][0]['base']['actions'] == after['states'][0]['base']['actions']
    assert before['states'][0]['wrappers'] == after['states'][0]['wrappers']
    target_env = DummyVecEnv([lambda: factory(True)])
    restored = AccumulatingMaskablePPO.load(dest/'model.zip', env=target_env, device='cpu')
    restored._setup_learn(32, reset_num_timesteps=False)
    restore_learner(restored, dest, rank=0, world_size=1)
    assert restored._recovery_environment_restored
    name = 'core.combat.player_projection.weight'
    for (key, p), (new_key, q) in zip(model.policy.named_parameters(), restored.policy.named_parameters(), strict=True):
        assert key == new_key
        if key == name:
            q_without_anchor = torch.cat((q[:, :-8], q[:, -6:]), 1)
            torch.testing.assert_close(p, q_without_anchor, rtol=0, atol=0)
            assert not q[:, -8:-6].any()
        else:
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for field, value in model.policy.optimizer.state[p].items():
            new_value = restored.policy.optimizer.state[q][field]
            if key == name and torch.is_tensor(value) and value.ndim:
                assert not new_value[:, -8:-6].any()
                new_value = torch.cat((new_value[:, :-8], new_value[:, -6:]), 1)
            torch.testing.assert_close(value, new_value, rtol=0, atol=0)
    restored.learn(64, reset_num_timesteps=False)
    assert restored.num_timesteps == 128
    second = save_bundle(restored, tmp_path/'continued', rank=0, world_size=1,
        run_id='anchor-resume-test', wandb_id=None, config={'ascension': 0})
    assert second.exists()
    env.close(); target_env.close()


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


@pytest.mark.parametrize('device,precision', [
    ('cpu', None), ('cuda', None), ('cpu', 'fp32'), ('cuda', 'fp16')])
@pytest.mark.parametrize('normalization', ['group256', 'rollout'])
def test_real_ppo_two_rounds_and_resume(tmp_path, device, precision, normalization):
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    env = DummyVecEnv([make_env, make_env])
    model = make_model(env, device)
    from rl.algorithms.ppo import configure_advantage_normalization
    configure_advantage_normalization(model, normalization)
    if precision:
        from rl.algorithms.packed_ppo import configure_packed_updates
        configure_packed_updates(model, dict(token_budget=512, max_rows=7, precision=precision))
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
    assert loaded.ppo_advantage_normalization == normalization
    assert loaded.ppo_update_config == model.ppo_update_config
    if precision:
        assert loaded.ppo_update_loss_scale == model.ppo_update_loss_scale
        assert loaded.policy.core.combat.transformer.update_precision == precision
        cpu = load_checkpoint(tmp_path/'policy.zip', device='cpu', inference_only=True)
        action, _ = cpu.predict(model._last_obs,
            action_masks=np.stack(env.env_method('action_masks')), deterministic=True)
        assert len(action) == env.num_envs
        assert all(p.dtype == torch.float32 for p in cpu.policy.parameters())
        configure_packed_updates(cpu, 'off')
        assert cpu.ppo_update_config is None
        assert cpu.policy.core.combat.transformer.update_precision == 'fp32'
        assert not cpu.policy.core.combat.compact_training_tokens
    assert loaded._ppo_advantage_stats is None
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
@pytest.mark.parametrize('normalization', ['group256', 'rollout'])
@pytest.mark.parametrize('scene,backend', [
    ('ordinary', 'native'), ('campfire', 'native'),
    ('ordinary', 'native-subproc'),
])
def test_native_distributed_payload_and_resume(tmp_path, monkeypatch, capsys, grow,
                                              scene, backend, campfire_env_type, normalization):
    _check_native_distributed_lifecycle(tmp_path, monkeypatch, capsys, grow,
        scene, backend, campfire_env_type, normalization)


@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_packed_distributed_payload_and_resume(tmp_path, monkeypatch, capsys,
                                              campfire_env_type, precision):
    _check_native_distributed_lifecycle(tmp_path, monkeypatch, capsys, False,
        'ordinary', 'native', campfire_env_type, 'rollout', precision)


@pytest.mark.parametrize('precision,backend', [('fp32', 'native-subproc'), ('fp16', 'native')])
def test_anchor_sequential_distributed_payload_and_resume(tmp_path, monkeypatch, capsys,
                                                         campfire_env_type, precision, backend):
    _check_native_distributed_lifecycle(tmp_path, monkeypatch, capsys, False,
        'ordinary', backend, campfire_env_type, 'rollout', precision, staged=True)


def _check_native_distributed_lifecycle(tmp_path, monkeypatch, capsys, grow,
                                       scene, backend, campfire_env_type, normalization,
                                       precision=None, staged=False):
    import json
    import sys
    from types import SimpleNamespace
    from rl.training import distributed_sb3
    from rl.training import reward_monitor, campfire_monitor

    factory = (lambda: make_env(staged=True)) if staged else make_env
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
    model = make_model(env, 'cpu', staged=staged)
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
        '--max-checkpoints', '2', '--seed', '20261108',
        '--ppo-advantage-normalization', normalization]
    if precision:
        update_config = tmp_path/'packed.json'
        update_config.write_text(json.dumps(dict(token_budget=512, max_rows=7, precision=precision)))
        argv += ['--ppo-update-config', str(update_config)]
        argv[argv.index('--device') + 1] = 'cuda' if precision == 'fp16' else 'cpu'
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    bundle = tmp_path/'state/rank0/step-000000000128'
    assert json.loads((bundle/'manifest.json').read_text())['schema_version'] == 2
    resumed = list(argv)
    resumed[resumed.index('--checkpoint')+1] = str(bundle/'model.zip')
    resumed += ['--resume-state', str(bundle)]
    # The selected mode must persist without an explicit CLI override.
    index = resumed.index('--ppo-advantage-normalization')
    del resumed[index:index + 2]
    if precision:
        index = resumed.index('--ppo-update-config')
        del resumed[index:index + 2]
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
        assert row['pure_rl/room_lambda'] == (1. if staged else .9)
        assert row['pure_rl/return_version'] == (2 if staged else 1)
        count = 128 if grow and row['rollout_round'] > 2 else 64
        assert row['pure_rl/advantage_normalization_rollout'] == int(normalization == 'rollout')
        if normalization == 'rollout':
            assert row['pure_rl/advantage_normalization/count'] == count
            assert row['pure_rl/normalization_group_size'] == count
            assert np.isfinite(row['pure_rl/advantage_normalization/std'])
        assert (row['pure_rl/combat_samples'] + row['pure_rl/upper_samples']
                + row.get('pure_rl/combat_anchor_samples', 0)) == count
        assert row['optimizer_steps'] == count // 32
        assert row['pure_rl/update/count'] == count
        if precision:
            assert row['pure_rl/update_loss_scale'] >= 1
            assert row['pure_rl/update_overflow_retries'] == 0
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


@pytest.mark.parametrize('overflow_source', ['local', 'peer'])
def test_packed_overflow_retries_same_adam_batch(tmp_path, overflow_source):
    from rl.algorithms.packed_ppo import configure_packed_updates, packed_ppo_epoch
    from rl.algorithms.ppo import configure_advantage_normalization, prepare_advantage_normalization
    from rl.training.distributed_sb3 import NoopCallback
    assert torch.cuda.is_available(), 'explicit CUDA qualification required'
    env = DummyVecEnv([make_env, make_env])
    model = make_model(env, 'cpu')
    _, callback = model._setup_learn(64, callback=NoopCallback())
    assert model.collect_rollouts(env, callback, model.rollout_buffer, 32)
    model.save(tmp_path/'before.zip')
    reference = AccumulatingMaskablePPO.load(tmp_path/'before.zip', device='cpu')
    reference.rollout_buffer = deepcopy(model.rollout_buffer)
    reference.set_logger(configure(folder=None, format_strings=[]))
    for learner in (reference, model):
        learner.device = torch.device('cuda')
        learner.policy.to('cuda')
        learner.rollout_buffer.device = learner.device
        learner.policy.set_training_mode(True)
        configure_advantage_normalization(learner, 'rollout')
        prepare_advantage_normalization(learner)
        configure_packed_updates(learner, dict(token_budget=512, max_rows=7, precision='fp16'))
        learner._ppo_update_metric_sums = None
        learner._ppo_update_metric_count = 0
    reference.ppo_update_loss_scale = 32768.
    rng = np.random.get_state()
    baseline = packed_ppo_epoch(reference, .2, None)
    np.random.set_state(rng)
    failures = []
    handle = None
    if overflow_source == 'local':
        parameter = model.policy.core.combat.transformer.layers[0].self_attn.in_proj_weight
        def inject(gradient):
            if not failures:
                failures.append('local')
                return torch.full_like(gradient, float('inf'))
            return gradient
        handle = parameter.register_hook(inject)
    finite_checks, synchronizations = [], []
    def average(values):
        if 'nonfinite' in values:
            finite_checks.append(values['nonfinite'])
            if overflow_source == 'peer' and not failures:
                failures.append('peer')
                return dict(nonfinite=.25)
        return values
    def synchronize(step):
        assert len(finite_checks) == step + 2
        synchronizations.append(step)
        return dict(sync_sec=0.,gradient_pack_sec=0.,sync_mbps=0.,grad_bytes=0)
    try:
        actual = packed_ppo_epoch(model, .2, None, synchronize=synchronize, average=average)
    finally:
        if handle is not None:
            handle.remove()
        env.close()
    assert failures == [overflow_source]
    assert synchronizations == [0, 1]
    assert actual['overflow_retries'] == 1
    assert actual['optimizer_steps'] == baseline['optimizer_steps'] == 2
    assert model._ppo_update_metric_count == reference._ppo_update_metric_count == 64
    assert model.ppo_update_loss_scale == 32768.
    for (n, p), (m, q) in zip(model.policy.named_parameters(), reference.policy.named_parameters()):
        assert n == m
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        a, b = model.policy.optimizer.state.get(p, {}), reference.policy.optimizer.state.get(q, {})
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


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
