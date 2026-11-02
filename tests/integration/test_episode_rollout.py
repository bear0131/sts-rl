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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_unified_mc_cohort_update_payload_and_resume(tmp_path, monkeypatch, short_natural_game_env_type, device):
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, False, None, unified=True)


def _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device, natural_start, lambda_override, censor_case=None, resume_target_kl=None, keep_round_multiple=0, fork_resume=False, unified=False):
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
        return VecMonitor(NativeSearchVecEnv([lambda:factory(0),lambda:factory(1)], threads=1))
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

