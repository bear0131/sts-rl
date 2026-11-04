"""Natural native terminal cohorts through production training and recovery."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import VecMonitor

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.entity_joint import EntityJointPolicy
from rl.training import distributed_sb3
from rl.training.checkpoints import read_bundle
from rl.training.environments import TrainingStateWrapper
from rl.training.native_vec_env import NativeSearchVecEnv
from rl.training.reward import ShapedRewardWrapper

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(240)]


def test_finished_slots_wait_and_terminal_journal_fails_closed(short_natural_game_env_type):
    import copy
    env = NativeSearchVecEnv([lambda: short_natural_game_env_type(global_map_view=True,
        upper_public_context=True, event_option_observation=True,
        combat_action_contract='autoregressive-v2')] * 2, threads=1)
    try:
        env.begin_episode_rollout(42)
        seen = np.zeros(2, dtype=int)
        for _ in range(200):
            masks = env.env_method('action_masks')
            actions = [int(np.flatnonzero(mask)[0]) for mask in masks]
            _, _, dones, _ = env.step(actions)
            seen += dones
            if not env.episode_rollout_active.any():
                break
        assert seen.tolist() == [1, 1]
        before = env.env_method('export_training_state')
        _, rewards, dones, infos = env.step([0, 0])
        assert not dones.any() and not rewards.any()
        assert all(info['episode_cohort_hold'] for info in infos)
        assert [s['actions'] for s in env.env_method('export_training_state')] == [s['actions'] for s in before]
        restored = short_natural_game_env_type(global_map_view=True,
            upper_public_context=True, event_option_observation=True,
            combat_action_contract='autoregressive-v2')
        try:
            restored.restore_training_state(before[0])
            malformed = copy.deepcopy(before[0]); malformed['actions'] += malformed['actions'][-1:]
            with pytest.raises(ValueError, match='terminated before live state'):
                restored.restore_training_state(malformed)
            malformed = copy.deepcopy(before[0]); malformed['schema_version'] = 1
            with pytest.raises(ValueError, match='terminated before live state'):
                restored.restore_training_state(malformed)
        finally:
            restored.close()
        env.begin_episode_rollout(44)
        assert env.episode_rollout_active.all() and not env.episode_rollout_finished.any()
    finally:
        env.close()


def test_censored_cohort_is_error_not_a_finished_game(short_natural_game_env_type):
    from gymnasium.wrappers import TimeLimit
    env = NativeSearchVecEnv([lambda: TimeLimit(short_natural_game_env_type(global_map_view=True,
        combat_action_contract='autoregressive-v2'), max_episode_steps=1)], threads=1)
    try:
        env.begin_episode_rollout(42)
        with pytest.raises(RuntimeError, match='censored whole-game cohort'):
            env.step([int(np.flatnonzero(env.env_method('action_masks')[0])[0])])
        assert not env.episode_rollout_finished.any()
    finally:
        env.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('natural_start', [False, True])
@pytest.mark.parametrize('lambda_override', [None, 1.])
def test_complete_cohort_two_rounds_and_ordinary_resume(tmp_path, monkeypatch, short_natural_game_env_type, device, natural_start, lambda_override):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, natural_start, lambda_override)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('censor_case',['mixed','all'])
def test_budget_exclusion_two_rounds_and_ordinary_resume(tmp_path, monkeypatch, short_natural_game_env_type, device, censor_case):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, False, None, censor_case)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_disable_kl_on_resume_preserves_state_and_two_complete_epochs(tmp_path, monkeypatch, short_natural_game_env_type, device):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device,
                      False, None, resume_target_kl=0.)


@pytest.mark.parametrize('fork_resume', [False, True])
def test_round_anchor_retention_survives_real_save_and_resume(tmp_path, monkeypatch, short_natural_game_env_type, fork_resume):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, 'cpu',
                      False, None, resume_target_kl=0., keep_round_multiple=2,
                      fork_resume=fork_resume)


@pytest.mark.parametrize('critic_batch', [None, 3])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_unified_mc_cohort_update_payload_and_resume(tmp_path, monkeypatch, short_natural_game_env_type, device, critic_batch):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, False, None,
                      unified=True, critic_batch=critic_batch)


@pytest.mark.timeout(180)
@pytest.mark.parametrize('device,natural_start', [('cpu', False), ('cpu', True), ('cuda', False)])
def test_independent_cohort_updates_payload_and_plain_resume(tmp_path, monkeypatch, short_natural_game_env_type, device, natural_start):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device,
                      natural_start, None, unified=True, independent=True)


def _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, natural_start, lambda_override, censor_case=None, resume_target_kl=None, keep_round_multiple=0, fork_resume=False, unified=False, critic_batch=None, independent=False, cohort_error=False):
    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit CUDA qualification requires CUDA'
    def make_env(*args):
        def factory(index):
            from env.core.run_env import SlayTheSpireEnv
            kind = SlayTheSpireEnv if natural_start else short_natural_game_env_type
            wrapped=ShapedRewardWrapper(kind(
                global_map_view=True, upper_public_context=True, event_option_observation=True,
                neow_observation=unified,
                card_language=True, combat_action_contract='autoregressive-v2'))
            if censor_case=='mixed' and index==1:
                from gymnasium.wrappers import TimeLimit
                wrapped=TimeLimit(wrapped,max_episode_steps=1)
            return TrainingStateWrapper(wrapped)
        count = getattr(args[-1], 'n_envs', 2) if args else 2
        return VecMonitor(NativeSearchVecEnv([lambda i=i: factory(i) for i in range(count)], threads=1))
    env = make_env()
    model = AccumulatingMaskablePPO(EntityJointPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
        policy_kwargs=dict(entity_config=dict(width=32, layers=1, heads=4, quantiles=8),
            upper_config=dict(width=32, layers=1, heads=4, epochs=2, minibatch_size=4,
                micro_batch_size=2, gae_lambda=.97, optimizer='adamw', learning_rate=3e-5,
                value_learning_rate=1e-4, weight_decay=1e-4, max_grad_norm=5,
                **(dict(architecture='upper-latent-actions-v1', target_kl=None, latents=4,
                        effective_batch_size=8, critic_epochs=3) if unified else {})),
            small_playout_config=dict(epochs=2, batch_size=8, micro_batch_size=4,
                target_mix=.2, optimizer='adamw', learning_rate=.003, weight_decay=.0001, max_grad_norm=0)),
        seed=42, device=device, amp_mode='off', nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_combat_strategy='small-playout',
        nosl_is_az_simulations=4, nosl_is_az_replay_enabled=False, hierarchical_combat_hp=True)
    initial = tmp_path/'initial.zip'
    model.save(initial)
    initial_policy = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
    env.close()
    assert not model.policy.upper_optimizer.state
    assert not model.policy.small_playout_optimizer.state
    payloads = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: SimpleNamespace(finish=lambda **kw: None),
        config=SimpleNamespace(update=lambda *a, **kw: None), define_metric=lambda *a, **kw: None,
        log=lambda p: payloads.append(json.loads(json.dumps(p, allow_nan=False)))))
    monkeypatch.setattr(distributed_sb3, 'make_env', make_env)
    argv = ['train', '--checkpoint', str(initial), '--rank', '0', '--world-size', '1',
        '--updates', '2', '--n-envs', '2', '--n-steps', '2', '--batch-size', '2',
        '--gradient-accumulation-steps', '1', '--n-epochs', '1', '--device', device,
        '--amp', 'off', '--vec', 'native', '--episode-rollout', '--episode-rollout-max-steps', '6000',
        '--episode-rollout-max-seconds', '120', '--seed', '42', '--nosl-is-az-combat-strategy', 'small-playout',
        '--no-nosl-is-az-replay', '--nosl-is-az-train-shared-embeddings', '--reasoning-updates', '0',
        '--reasoning-coef', '0', '--no-interactive-reasoning', '--learning-rate', '.0003',
        '--remote-root', str(tmp_path/'transport'), '--ssh-control-dir', str(tmp_path/'ssh'),
        '--run-id', 'cohort-test', '--wandb', '--wandb-id', 'cohort-test',
        '--training-state-dir', str(tmp_path/'state'), '--save-every-updates', '1', '--max-checkpoints', '4',
        '--stop-file', str(tmp_path/'STOP'), '--output', str(tmp_path/'latest.zip')]
    if independent:
        scheduler_path = tmp_path/'scheduler.json'
        scheduler_config = dict(mode='independent', threads=2, upper_batch=2,
                                upper_wait_seconds=.010, seed=42)
        scheduler_path.write_text(json.dumps(scheduler_config))
        argv += ['--independent-rollout-config', str(scheduler_path), '--rollout-resource-interval', '.1']
    if unified and not censor_case and not independent and not cohort_error:
        argv += ['--episode-rollout-censored', 'exclude']
    if critic_batch is not None:
        argv += ['--upper-critic-effective-batch-size', str(critic_batch)]
    if lambda_override is not None:
        argv += ['--upper-gae-lambda', str(lambda_override)]
    retained_before_save = []
    if keep_round_multiple:
        argv += ['--checkpoint-keep-round-multiple', str(keep_round_multiple)]
        original_save = distributed_sb3.save_bundle
        def save(model, root, **kwargs):
            kept = sorted(json.loads(p.read_text())['rollout_round']
                          for p in (tmp_path/'state/rank0').glob('step-*/manifest.json'))
            retained_before_save.append(kept)
            return original_save(model, root, **kwargs)
        monkeypatch.setattr(distributed_sb3,'save_bundle',save)
    if censor_case:
        argv += ['--episode-rollout-censored','exclude']
        if censor_case=='all':
            argv += ['--episode-rollout-max-steps','1']
    from rl.training import episode_rollout
    original_exclude=episode_rollout.exclude_samples
    excluded_checks=[]
    def exclude(model, env):
        excluded=set(np.flatnonzero(env.episode_rollout_censored).tolist())
        before_upper=[r for r in model._upper_completed if r['env'] in excluded]
        before_lower=[r for r in model._small_playout_samples if r['env'] in excluded]
        original_exclude(model,env)
        assert all(r['env'] not in excluded for r in model._upper_completed)
        assert all(r['env'] not in excluded for r in model._small_playout_samples)
        assert not any(model._upper_pending)
        excluded_checks.append((len(excluded),len(before_upper),len(before_lower)))
    monkeypatch.setattr(episode_rollout,'exclude_samples',exclude)
    natural_count=0 if censor_case=='all' else 1 if censor_case=='mixed' else 2
    original_learn = AccumulatingMaskablePPO.learn
    def learn(self, *args, **kwargs):
        assert self.small_playout_search_contract == 'terminal-hp-boss-win-turn128-v1'
        if not getattr(self, 'episode_rollouts_completed', 0):
            torch.testing.assert_close(self.policy.state_dict(), initial_policy,
                                       rtol=0, atol=0, check_device=False)
            assert not self.policy.upper_optimizer.state and not self.policy.small_playout_optimizer.state
        assert self.policy.core.upper_config.gae_lambda == (lambda_override or .97)
        if resume_target_kl is not None:
            expected_kl = None if getattr(self, 'episode_rollouts_completed', 0) else .02
            assert self.policy.core.upper_config.target_kl == expected_kl
        return original_learn(self, *args, **kwargs)
    monkeypatch.setattr(AccumulatingMaskablePPO, 'learn', learn)
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    first = sorted((tmp_path/'state/rank0').glob('step-*'))[-1]
    saved = AccumulatingMaskablePPO.load(first/'model.zip', device='cpu')
    assert saved.episode_rollouts_completed == 2 and getattr(saved,'small_playout_games_local',0) == 2*natural_count
    assert not any(torch.load(first/'learner.pt', weights_only=False)['upper_pending'])
    original_restore = distributed_sb3.restore_learner
    def restore(model, path, **kwargs):
        expected = AccumulatingMaskablePPO.load(path/'model.zip', device='cpu')
        result = original_restore(model, path, **kwargs)
        assert model._recovery_environment_restored and result['schema_version'] == 2
        for a, b in ((model.policy.state_dict(), expected.policy.state_dict()),
                     (model.policy.upper_optimizer.state_dict(), expected.policy.upper_optimizer.state_dict()),
                     (model.policy.small_playout_optimizer.state_dict(), expected.policy.small_playout_optimizer.state_dict())):
            torch.testing.assert_close(a, b, rtol=0, atol=0, check_device=False)
        learner = torch.load(path/'learner.pt', weights_only=False)
        np.testing.assert_equal(np.random.get_state(), learner['numpy_rng'])
        assert torch.equal(torch.get_rng_state(), learner['torch_rng'])
        assert model.episode_rollouts_completed == 2
        return result
    monkeypatch.setattr(distributed_sb3, 'restore_learner', restore)
    resume_argv = list(argv)
    if independent:
        index = resume_argv.index('--independent-rollout-config')
        del resume_argv[index:index+2]
        scheduler_path.unlink()  # Resume must not depend on a config file.
    if critic_batch is not None:
        index = resume_argv.index('--upper-critic-effective-batch-size')
        del resume_argv[index:index+2]
    if lambda_override is not None:
        index = resume_argv.index('--upper-gae-lambda')
        del resume_argv[index:index+2]
    if resume_target_kl is not None:
        resume_argv += ['--upper-target-kl', str(resume_target_kl)]
    if fork_resume:
        resume_argv += ['--run-id', 'checkpoint-retention-fork', '--wandb-id', 'retention-fork',
                        '--fork-state', str(first)]
    else:
        resume_argv += ['--resume-state', str(first)]
    monkeypatch.setattr(sys, 'argv', resume_argv)
    distributed_sb3.main()
    assert [p['legacy_step'] for p in payloads] == [1, 2, 3, 4]
    assert [p['game/completed_full_games_global'] for p in payloads] == [i*natural_count for i in (1,2,3,4)]
    assert [x[0] for x in excluded_checks]==[2-natural_count]*4
    bundles = sorted((tmp_path/'state/rank0').glob('step-*'))
    assert len(bundles) == (1 if keep_round_multiple else 4)
    if keep_round_multiple:
        assert retained_before_save == [[],[1],[2],[2,3]]
    for bundle_index, bundle in enumerate(bundles):
        manifest = read_bundle(bundle, rank=0, world_size=1)
        if fork_resume:
            assert manifest['run_id'] == 'checkpoint-retention-fork'
            assert manifest['config']['fork_source_global_step'] == saved.num_timesteps
        i = manifest['rollout_round'] - 1
        if not keep_round_multiple:
            assert i == bundle_index
        assert manifest['config']['checkpoint_keep_round_multiple'] == keep_round_multiple
        current = AccumulatingMaskablePPO.load(bundle/'model.zip', device='cpu')
        if independent:
            assert current.independent_rollout_config == scheduler_config
            assert manifest['config']['independent_rollout_config'] == scheduler_config
            assert current._independent_rollout_rng['version'] == 'independent-games-pcg64-v1'
            assert len(current._independent_rollout_rng['upper']) == 2
            assert payloads[i]['speed/rollout_env_steps_per_second'] > 0
            assert payloads[i]['dist/rollout_sec_max'] == payloads[i]['dist/rollout_sec']
            assert payloads[i]['small_playout_train/updates'] > 0
            assert current.policy.upper_optimizer.state and current.policy.small_playout_optimizer.state
        assert current.small_playout_search_contract == 'terminal-hp-boss-win-turn128-v1'
        assert manifest['config']['small_playout_search_contract'] == 'terminal-hp-boss-win-turn128-v1'
        assert current.policy.core.upper_config.gae_lambda == (lambda_override or .97)
        assert manifest['config']['upper_config']['gae_lambda'] == (lambda_override or .97)
        if resume_target_kl is not None:
            expected_kl = .02 if i < 2 else None
            assert current.policy.core.upper_config.target_kl == expected_kl
            assert current.policy.upper_config['target_kl'] == expected_kl
            assert current.policy_kwargs['upper_config']['target_kl'] == expected_kl
            assert manifest['config']['upper_config']['target_kl'] == expected_kl
            assert payloads[i]['upper/kl_early_stop_enabled'] == int(i < 2)
            if i >= 2:
                assert payloads[i]['upper/updates'] == payloads[i]['upper/requested_updates']
                assert payloads[i]['upper/epochs_completed'] == 2
                assert payloads[i]['upper/samples_used'] == 2 * payloads[i]['upper/samples']
                assert payloads[i]['upper/kl_stopped'] == 0
        if unified:
            import math
            cfg = current.policy.core.upper_config
            assert cfg.critic_effective_batch_size == (critic_batch or 0)
            assert manifest['config']['upper_config']['critic_effective_batch_size'] == (critic_batch or 0)
            samples = payloads[i]['upper/samples']
            assert payloads[i]['upper/updates'] == 2 * math.ceil(samples / 8)
            assert payloads[i]['upper/critic_updates'] == 3 * math.ceil(samples / (critic_batch or 8))
            assert payloads[i]['upper/critic_samples_used'] == 3 * samples
            assert payloads[i]['upper/requested_critic_effective_batch_size'] == (critic_batch or 8)
        stats = current.episode_rollout_stats
        assert stats['first_seed'] == 42 + i * 2 and stats['last_seed'] == 43 + i * 2
        assert stats['completed_games'] == natural_count and stats['unfinished_games'] == 0
        assert stats['censored_games']==2-natural_count
        assert stats['transitions'] > 0
        assert stats['vector_steps']==1 if censor_case=='all' else stats['vector_steps']>2
        assert manifest['config']['episode_rollout_config']['games_per_rank'] == 2
        assert payloads[i]['upper/pending'] == payloads[i]['upper/censored'] == payloads[i]['upper/bootstrap_samples'] == 0
        assert payloads[i]['upper/samples_used'] == 2 * payloads[i]['upper/samples'] or payloads[i]['upper/kl_stopped']
        if censor_case=='all':
            assert payloads[i]['upper/updates']==payloads[i]['small_playout_train/updates']==0
            assert not current.policy.upper_optimizer.state and not current.policy.small_playout_optimizer.state
            torch.testing.assert_close(current.policy.state_dict(),initial_policy,rtol=0,atol=0,check_device=False)

    return dict(argv=argv, payloads=payloads, last_bundle=bundles[-1])


@pytest.mark.timeout(180)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_independent_fork_preserves_nonempty_state_and_plain_resume(
        tmp_path, monkeypatch, short_natural_game_env_type, device):
    from rl.training.checkpoints import restore_learner, _observation_sha256
    from tests.integration.test_training_smoke import assert_nested_equal
    setup = _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type,
                              device, False, None, unified=True, cohort_error=True)
    source = setup['last_bundle']
    original_manifest = read_bundle(source, rank=0, world_size=1)
    scheduler_path = tmp_path/'independent.json'
    cfg = dict(mode='independent', threads=2, upper_batch=2,
               upper_wait_seconds=.010, seed=42)
    scheduler_path.write_text(json.dumps(cfg))
    restored = []
    def observed_restore(model, path, **kwargs):
        expected = AccumulatingMaskablePPO.load(path/'model.zip', device=device)
        environment = torch.load(path/'environment.pt', weights_only=False)
        result = restore_learner(model, path, **kwargs)
        assert expected.policy.upper_optimizer.state and expected.policy.small_playout_optimizer.state
        for actual, saved in ((model.policy.state_dict(), expected.policy.state_dict()),
                (model.policy.upper_optimizer.state_dict(), expected.policy.upper_optimizer.state_dict()),
                (model.policy.small_playout_optimizer.state_dict(), expected.policy.small_playout_optimizer.state_dict())):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0, check_device=False)
        assert_nested_equal(model.env.env_method('export_training_state'), environment['states'])
        assert _observation_sha256(model._last_obs) == environment['last_observation_sha256']
        assert model.independent_rollout_config == cfg
        restored.append(model.episode_rollouts_completed)
        return result
    monkeypatch.setattr(distributed_sb3, 'restore_learner', observed_restore)
    args = [*setup['argv'], '--run-id', 'independent-fork', '--wandb-id', 'independent-fork',
            '--training-state-dir', str(tmp_path/'independent-state')]
    monkeypatch.setattr(sys, 'argv', [*args, '--fork-state', str(source),
                        '--independent-rollout-config', str(scheduler_path)])
    distributed_sb3.main()
    boundary = sorted((tmp_path/'independent-state/rank0').glob('step-*'))[-1]
    scheduler_path.unlink()
    monkeypatch.setattr(sys, 'argv', [*args, '--resume-state', str(boundary)])
    distributed_sb3.main()
    assert restored == [4, 6]
    assert read_bundle(source, rank=0, world_size=1) == original_manifest
    assert [p['legacy_step'] for p in setup['payloads']] == list(range(1, 9))
    assert all(p['episode_rollout/completed_games_global'] == 2 for p in setup['payloads'])
    for path in sorted((tmp_path/'independent-state/rank0').glob('step-*')):
        model = AccumulatingMaskablePPO.load(path/'model.zip', device='cpu')
        assert model.independent_rollout_config == cfg
        assert model.independent_rollout_migration['source_round'] == 4
        n = model.episode_rollouts_completed
        assert model.episode_rollout_stats['first_seed'] == 42 + (n-1)*2
        for optimizer, key in ((model.policy.upper_optimizer, 'upper/updates'),
                               (model.policy.small_playout_optimizer, 'small_playout_train/updates')):
            steps = sum(p[key] for p in setup['payloads'][:n])
            if key.startswith('small'):
                assert {int(v['step']) for v in optimizer.state.values()} == {steps}
            else:
                assert optimizer.state


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_lower_batch_fork_preserves_nonempty_state_and_plain_resume(
        tmp_path, monkeypatch, short_natural_game_env_type, device):
    """Real CLI batch override keeps a trained source and survives ordinary resume."""
    import random
    from rl.training.checkpoints import restore_learner, _observation_sha256
    from tests.integration.test_training_smoke import assert_nested_equal
    setup = _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type,
                              device, False, None, unified=True, critic_batch=3)
    source = setup['last_bundle']
    source_manifest = read_bundle(source, rank=0, world_size=1)
    restored_rounds = []
    def observed_restore(model, path, **kwargs):
        expected = AccumulatingMaskablePPO.load(path/'model.zip', device=device)
        learner = torch.load(path/'learner.pt', weights_only=False)
        environment = torch.load(path/'environment.pt', weights_only=False)
        result = restore_learner(model, path, **kwargs)
        assert expected.policy.upper_optimizer.state and expected.policy.small_playout_optimizer.state
        for actual, saved in [(model.policy.state_dict(), expected.policy.state_dict()),
                (model.policy.upper_optimizer.state_dict(), expected.policy.upper_optimizer.state_dict()),
                (model.policy.small_playout_optimizer.state_dict(), expected.policy.small_playout_optimizer.state_dict())]:
            torch.testing.assert_close(actual, saved, rtol=0, atol=0, check_device=False)
        assert model.policy.upper_config == expected.policy.upper_config
        assert model.policy.small_playout_config == dict(expected.policy.small_playout_config, batch_size=4)
        assert model.policy_kwargs['small_playout_config'] == model.policy.small_playout_config
        assert_nested_equal(model.env.env_method('export_training_state'), environment['states'])
        assert _observation_sha256(model._last_obs) == environment['last_observation_sha256']
        assert_nested_equal(model._upper_episode_ids, learner['upper_episode_ids'])
        assert_nested_equal(model._upper_pending, learner['upper_pending'])
        assert random.getstate() == learner['python_rng']
        np.testing.assert_equal(np.random.get_state(), learner['numpy_rng'])
        assert torch.equal(torch.get_rng_state(), learner['torch_rng'])
        if learner['cuda_rng'] is not None:
            assert_nested_equal(torch.cuda.get_rng_state_all(), learner['cuda_rng'])
        restored_rounds.append(model.episode_rollouts_completed)
        return result
    monkeypatch.setattr(distributed_sb3, 'restore_learner', observed_restore)
    argv = [*setup['argv'], '--run-id', 'lower-batch-fork', '--wandb-id', 'lower-batch-fork',
            '--training-state-dir', str(tmp_path/'lower-state'), '--output', str(tmp_path/'lower.zip')]
    monkeypatch.setattr(sys, 'argv', [*argv, '--small-playout-batch-size', '4', '--fork-state', str(source)])
    distributed_sb3.main()
    boundary = sorted((tmp_path/'lower-state/rank0').glob('step-*'))[-1]
    # Omit the override entirely: the checkpoint must carry the new batch.
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume-state', str(boundary)])
    distributed_sb3.main()
    assert restored_rounds == [4, 6]
    assert read_bundle(source, rank=0, world_size=1) == source_manifest
    payloads = setup['payloads']
    assert [p['legacy_step'] for p in payloads] == list(range(1, 9))
    assert all(p['episode_rollout/completed_games_global'] == 2 for p in payloads)
    assert all(p['episode_rollout/censored_games_global'] == 0 for p in payloads)
    assert all(p['small_playout_train/batch_size_per_rank'] == 4 for p in payloads[4:])
    for path in sorted((tmp_path/'lower-state/rank0').glob('step-*')):
        manifest = read_bundle(path, rank=0, world_size=1)
        saved = AccumulatingMaskablePPO.load(path/'model.zip', device='cpu')
        n = saved.episode_rollouts_completed
        assert manifest['config']['small_playout_config']['batch_size'] == 4
        assert saved.policy.small_playout_config['batch_size'] == 4
        assert saved.policy_kwargs['small_playout_config']['batch_size'] == 4
        assert saved.episode_rollout_stats['first_seed'] == 42 + (n-1)*2
        assert saved.episode_rollout_stats['last_seed'] == 43 + (n-1)*2
        assert len(torch.load(path/'environment.pt', weights_only=False)['states']) == 2
        steps = sum(p['small_playout_train/updates'] for p in payloads[:n])
        assert saved.small_playout_optimizer_steps == steps
        assert {int(v['step']) for v in saved.policy.small_playout_optimizer.state.values()} == {steps}


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_complete_cohort_growth_preserves_source_and_plain_resume(
        tmp_path, monkeypatch, short_natural_game_env_type, device):
    import random
    from rl.training.checkpoints import restore_learner, _observation_sha256
    from tests.integration.test_training_smoke import assert_nested_equal
    setup = _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type,
                              device, False, None, unified=True, critic_batch=3)
    source = setup['last_bundle']
    source_manifest = read_bundle(source, rank=0, world_size=1)
    restored_counts = []
    def observed_restore(model, path, **kwargs):
        saved = torch.load(path/'environment.pt', weights_only=False)
        learner = torch.load(path/'learner.pt', weights_only=False)
        expected = AccumulatingMaskablePPO.load(path/'model.zip', device=device)
        source_count = len(saved['states'])
        result = restore_learner(model, path, **kwargs)
        assert_nested_equal(model.policy.state_dict(), expected.policy.state_dict())
        assert_nested_equal(model.policy.upper_optimizer.state_dict(), expected.policy.upper_optimizer.state_dict())
        assert_nested_equal(model.policy.small_playout_optimizer.state_dict(), expected.policy.small_playout_optimizer.state_dict())
        states = model.env.env_method('export_training_state')
        assert_nested_equal(states[:source_count], saved['states'])
        assert _observation_sha256({k:v[:source_count] for k,v in model._last_obs.items()}) == saved['last_observation_sha256']
        np.testing.assert_array_equal(model._last_episode_starts[:source_count], learner['last_episode_starts'])
        np.testing.assert_array_equal(model.env.episode_returns[:source_count], saved['episode_returns'])
        np.testing.assert_array_equal(model.env.episode_lengths[:source_count], saved['episode_lengths'])
        assert model.env.episode_count == saved['episode_count']
        assert model._episode_num == learner['episode_num']
        assert_nested_equal(model._upper_episode_ids[:source_count], learner['upper_episode_ids'])
        assert_nested_equal(model._upper_pending[:source_count], learner['upper_pending'])
        assert_nested_equal(model.ep_info_buffer, expected.ep_info_buffer)
        np.testing.assert_equal(np.random.get_state(), learner['numpy_rng'])
        assert random.getstate() == learner['python_rng']
        assert torch.equal(torch.get_rng_state(), learner['torch_rng'])
        if learner['cuda_rng'] is not None:
            assert_nested_equal(torch.cuda.get_rng_state_all(), learner['cuda_rng'])
        if kwargs['grow_environments']:
            assert model._recovery_environment_growth['appended_first_seed'] == 52
            assert len(states) == 3 and len(learner['upper_pending']) == 2
            assert model._upper_pending[2:] == [None]
            assert model._upper_episode_ids[2:] == [4]
            assert model._last_episode_starts[2:].tolist() == [True]
        restored_counts.append(source_count)
        return result
    monkeypatch.setattr(distributed_sb3, 'restore_learner', observed_restore)
    args = [*setup['argv'], '--n-envs', '3', '--run-id', 'grown', '--wandb-id', 'grown',
            '--training-state-dir', str(tmp_path/'grown-state')]
    # Geometry changes are forbidden without the explicit migration option.
    monkeypatch.setattr(sys, 'argv', [*args, '--fork-state', str(source)])
    with pytest.raises(ValueError, match='ordinary resume'):
        distributed_sb3.main()
    monkeypatch.setattr(sys, 'argv', [*args, '--fork-state', str(source), '--fork-grow-environments'])
    distributed_sb3.main()
    boundary = sorted((tmp_path/'grown-state/rank0').glob('step-*'))[-1]
    monkeypatch.setattr(sys, 'argv', [*args, '--resume-state', str(boundary)])
    distributed_sb3.main()
    assert restored_counts == [2, 3]
    assert read_bundle(source, rank=0, world_size=1) == source_manifest
    assert [p['game/completed_full_games_global'] for p in setup['payloads']] == [2,4,6,8,11,14,17,20]
    for index, path in enumerate(sorted((tmp_path/'grown-state/rank0').glob('step-*'))):
        saved = AccumulatingMaskablePPO.load(path/'model.zip', device=device)
        assert saved.episode_rollouts_completed == 5 + index
        assert saved.episode_rollout_seed_offset == -4
        assert saved.episode_rollout_stats['first_seed'] == 50 + 3 * index
        assert saved.episode_rollout_stats['last_seed'] == 52 + 3 * index
        assert saved.episode_rollout_stats['completed_games'] == 3
        assert saved.episode_rollout_stats['censored_games'] == saved.episode_rollout_stats['unfinished_games'] == 0
        assert len(torch.load(path/'environment.pt', weights_only=False)['states']) == 3
        assert saved.policy.core.upper_config.critic_effective_batch_size == 3


@pytest.mark.parametrize('scheduler', [None, 'barrier', 'independent', 'independent-natural'])
def test_frozen_rollout_diagnostic_uses_real_collector_without_saving_updates(
        tmp_path, monkeypatch, short_natural_game_env_type, scheduler):
    from rl.training.checkpoints import restore_learner
    from rl.algorithms.upper_probe import state_digest
    from tools.analysis.probe_upper_learning import unified_candidates
    from tools.analysis import probe_upper_learning
    from rl.training import reward_monitor, campfire_monitor
    measured_upper, measured_monitors = [], []
    measure = probe_upper_learning.measure_frozen_rollout
    def observed_measure(model, *args, **kwargs):
        result = measure(model, *args, **kwargs)
        measured_upper.append(np.asarray([(r['value'], r['log_prob']) for r in model._upper_completed]))
        measured_monitors.append((reward_monitor.counts(model), campfire_monitor.counts(model)))
        return result
    monkeypatch.setattr(probe_upper_learning, 'measure_frozen_rollout', observed_measure)
    natural_start = scheduler == 'independent-natural'
    if natural_start:
        scheduler = 'independent'
    setup = _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type,
                              'cpu', natural_start, None, unified=True, critic_batch=3)
    source = setup['last_bundle']
    manifest = read_bundle(source, rank=0, world_size=1)
    model = AccumulatingMaskablePPO.load(source/'model.zip', device='cpu')
    expected = dict(policy=state_digest(model.policy.state_dict()),
                    upper=state_digest(model.policy.upper_optimizer.state_dict()),
                    lower=state_digest(model.policy.small_playout_optimizer.state_dict()))
    protocol = tmp_path/'scale.json'
    protocol.write_text(json.dumps(dict(protocol='unified-upper-rollout-scale-v1',
        diagnostic=dict(games_per_rank=3, world_size=1, maximum_wall_seconds=60),
        **(dict(independent_scheduler=dict(mode=scheduler, threads=2, upper_batch=3)) if scheduler else {}))))
    argv, index = [], 1
    while index < len(setup['argv']):
        flag = setup['argv'][index]
        if flag in ('--training-state-dir', '--output', '--wandb-id'):
            index += 2
        elif flag == '--wandb':
            index += 1
        else:
            argv.append(flag)
            index += 1
    output = tmp_path/'rollout-probe'
    monkeypatch.setattr(distributed_sb3, 'restore_learner', restore_learner)
    monkeypatch.setattr(sys, 'argv', ['probe', '--unified-candidate-protocol', str(protocol),
        '--probe-output', str(output), *argv, '--fork-state', str(source), '--fork-grow-environments',
        '--n-envs', '3', '--updates', '1', '--run-id', 'frozen-rollout', '--iteration-timeout-seconds', '60'])
    unified_candidates()
    result = json.loads((output/'rank0/rollout.json').read_text())
    assert result['cohort']['completed_games'] == 3
    assert result['all_rank_completed_games'] == 3
    assert result['completion_sync_seconds'] >= 0
    assert result['cohort']['first_seed'] == 50
    assert result['collector_wall_seconds'] > 0
    assert result['setup_times']['entry_to_collection_seconds'] >= result['setup_times']['environment_learner_restore_seconds'] > 0
    assert result['collector_metrics']['nosl_is_az/collection_seconds'] > 0
    assert result['search_statistics']['common/simulations'] > 0
    assert result['unisolated_timing']['upper_forward_seconds'] is None
    assert result['optimizer_updates'] == 0 and result['policy_and_adam_unchanged']
    if scheduler:
        assert result['independent_scheduler']['random_stream'] == 'independent-games-pcg64-v1'
        assert result['independent_scheduler']['mode'] == scheduler
        assert result['independent_scheduler']['upper_rows'] == result['upper_samples']
        assert result['lower_samples'] > 0
    assert result['state_digests'] == expected
    assert result['semantic_digest_seconds'] >= 0
    assert set(result['semantic_digests']) == {
        'environment', 'upper_samples', 'upper_decisions', 'lower_samples', 'rng'}
    assert result['resources']['interval_seconds'] == 1.
    assert result['torch_max_allocated_bytes'] is None  # CPU is not GPU qualification.
    assert read_bundle(source, rank=0, world_size=1) == manifest
    assert len(setup['payloads']) == 4  # Diagnostic never publishes a training update.
    repeated = list(sys.argv)
    repeat_output = tmp_path/'rollout-repeat'
    repeated[repeated.index('--probe-output') + 1] = str(repeat_output)
    monkeypatch.setattr(sys, 'argv', repeated)
    unified_candidates()
    repeat = json.loads((repeat_output/'rank0/rollout.json').read_text())
    if scheduler == 'independent':
        # Completion timing changes inference batch shapes. Require identical
        # trajectories/labels/RNG here, and bound FP32 reduction differences.
        for key in ('environment', 'upper_decisions', 'lower_samples', 'rng'):
            assert repeat['semantic_digests'][key] == result['semantic_digests'][key]
        assert repeat['game_rng_digest'] == result['game_rng_digest']
        np.testing.assert_allclose(measured_upper[0], measured_upper[1], atol=2e-6, rtol=2e-5)
    else:
        assert repeat['semantic_digests'] == result['semantic_digests']
    if natural_start:
        assert measured_monitors[0][0]['reward_gold_policy_decisions'] > 0
    assert repeat['state_digests'] == expected
    assert len(setup['payloads']) == 4
    config = json.loads(protocol.read_text())
    config['diagnostic']['fresh_cohort_offset'] = 1
    protocol.write_text(json.dumps(config))
    repeated[repeated.index('--probe-output') + 1] = str(tmp_path/'fresh-cohort')
    monkeypatch.setattr(sys, 'argv', repeated)
    unified_candidates()
    fresh = json.loads((tmp_path/'fresh-cohort/rank0/rollout.json').read_text())
    assert fresh['cohort']['first_seed'] == 53
    assert fresh['cohort']['last_seed'] == 55
    assert fresh['state_digests'] == expected
    assert fresh['semantic_digests']['environment'] != result['semantic_digests']['environment']
    assert read_bundle(source, rank=0, world_size=1) == manifest
    assert len(setup['payloads']) == 4
