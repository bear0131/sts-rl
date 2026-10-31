"""Exercise the production collector → learner → logging → checkpoint lifecycle."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.native import NATIVE_TIMING_KEYS
from rl.training import distributed_sb3
from rl.training.environments import StartupStaggerWrapper
from rl.training.reward import ShapedRewardWrapper

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(90)]


@pytest.mark.parametrize('nosl,upper,combat_arch,contract', [
    (False, 'model', 'transformer', 'legacy'), (True, 'model', 'transformer', 'legacy'),
    (True, 'rules', 'transformer', 'legacy'), (True, 'rules', 'mlp', 'legacy'),
    (False, 'model', 'transformer', 'joint-v2'), (True, 'rules', 'transformer', 'joint-v2'),
    (True, 'rules', 'entity', 'joint-v2'), (True, 'rules', 'entity', 'autoregressive-v2')])
def test_training_reaches_wandb_and_checkpoint(tmp_path, monkeypatch, capsys, short_combat_env_type, nosl, upper, combat_arch, contract, device='cpu', amp='off', vector='dummy', merge=True, step_schedule=False, boss_weight=0., joint=False, fixed_lower=False, chests=False, upper_micro_override=None, upper_lambda_override=None, online_small=False):
    fixed_strategy = 'small-playout' if fixed_lower == 'small-playout' else 'turn-planner'
    staggered = nosl and upper == 'rules' and combat_arch == 'mlp'
    occupancy = nosl and upper == 'rules' and combat_arch == 'transformer'
    fixed_calls = []
    if contract == 'autoregressive-v2':
        native_begin = short_combat_env_type.nosl_is_az_cpp_begin
        def observed_begin(self, *args, **kwargs):
            fixed_calls.append(kwargs.get('fixed_worldline'))
            return native_begin(self, *args, **kwargs)
        monkeypatch.setattr(short_combat_env_type, 'nosl_is_az_cpp_begin', observed_begin)

    def make_env(*_):
        def factory(hold_steps):
            def build():
                env = TimeLimit(
                    ShapedRewardWrapper(short_combat_env_type(combat_action_contract=contract,
                        global_map_view=True, upper_public_context=bool(online_small), event_option_observation=bool(online_small), card_language=combat_arch=='entity')),
                    max_episode_steps=12,
                )
                return (
                    StartupStaggerWrapper(env, hold_steps=hold_steps)
                    if staggered else env
                )
            return build

        if vector == 'native':
            from rl.training.native_vec_env import NativeSearchVecEnv
            return NativeSearchVecEnv([factory(0), factory(16)], threads=2)
        return DummyVecEnv([factory(0), factory(16)])

    env = make_env()
    from rl.policies.entity_combat import EntityCombatPolicy
    policy = EntityCombatPolicy if combat_arch == 'entity' else SplitStSPointerPolicy
    kwargs = (dict(entity_config=dict(width=32,layers=1,heads=4,quantiles=8))
              if combat_arch == 'entity' else dict(d_model=32,n_layers=2,n_heads=4,combat_arch=combat_arch))
    if combat_arch == 'entity':
        kwargs['entity_config']['boss_win_weight'] = boss_weight
    if joint:
        from rl.policies.entity_joint import EntityJointPolicy
        policy = EntityJointPolicy
        kwargs['upper_config'] = dict(width=32, layers=1, heads=4, epochs=2, micro_batch_size=8)
    if online_small:
        kwargs['small_playout_config'] = dict(epochs=2, batch_size=8, micro_batch_size=3)
        if online_small == 'reference-ce':
            kwargs['small_playout_config'].update(target_mix=.2, optimizer='adamw',
                weight_decay=1e-4, learning_rate=3e-3, max_grad_norm=0)
    model = AccumulatingMaskablePPO(
        policy, env, n_steps=16, batch_size=8, n_epochs=1,
        policy_kwargs=kwargs, device=device, seed=42, amp_mode=amp,
        nosl_is_az=nosl, nosl_is_az_cpp=nosl, nosl_is_az_inner_only=nosl,
        nosl_is_az_merge_equivalent_actions=merge,
        nosl_is_az_upper_strategy=upper,
        nosl_is_az_combat_strategy=fixed_strategy if fixed_lower else 'model',
        nosl_is_az_simulations=4, nosl_is_az_depth=0, nosl_is_az_updates=3 if step_schedule else 1,
        nosl_is_az_update_schedule=(dict(mode='value-warmup', peak_updates=3,
            hold_rollouts=1, decay_rollouts=2) if combat_arch == 'entity' else None),
        nosl_is_az_replay_enabled=not fixed_lower,
        nosl_is_az_replay_capacity=128, nosl_is_az_batch_size=16,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1,
        hierarchical_combat_hp=nosl)
    if fixed_strategy == 'small-playout' and not online_small:
        from rl.algorithms.small_playout import SmallPlayoutLearner, save_policy_pair
        from rl.training.small_playout import install_policy_pair
        compact = SmallPlayoutLearner(seed=42)
        prior = SmallPlayoutLearner(seed=43).policy
        pair = tmp_path/'pair.pt'
        save_policy_pair(pair, compact.policy, prior, provenance={'fixture':'lifecycle'})
        install_policy_pair(model, pair)
    if fixed_strategy == 'small-playout':
        from rl.policies.entity_joint import EntityJointCore
        monkeypatch.setattr(EntityJointCore, 'combat_actor_critic',
                            lambda *a, **kw: pytest.fail('compact collector called large combat network'))
    initial_combat = (
        {name: value.detach().cpu().clone()
         for name, value in model.policy.core.combat.state_dict().items()}
        if fixed_lower else None
    )
    checkpoint = tmp_path / 'initial.zip'
    initial_upper = ({name: value.detach().cpu().clone() for name, value in model.policy.core.upper_tokenizer.state_dict().items()}
                     if fixed_strategy == 'small-playout' else None)
    initial_small = model.policy.small_playout.export().copy() if online_small else None
    model.save(checkpoint)
    env.close()
    payloads, finished, identities = [], [], []
    stop_file = tmp_path / 'stop'
    def publish(payload):
        payloads.append(json.loads(json.dumps(payload, allow_nan=False)))
        if len(payloads) == 3:
            stop_file.touch()
    def init(**kwargs):
        identities.append(kwargs)
        return SimpleNamespace(
            finish=lambda *, exit_code=0: finished.append(exit_code)
        )
    # Replace only the external service; the production payload construction runs.
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=init, config=SimpleNamespace(update=lambda *a, **kw: None),
        define_metric=lambda *a, **kw: None,
        log=publish))
    monkeypatch.setattr(distributed_sb3, 'make_env', make_env)
    if nosl:
        # Exercise the real main/logging path with an unequal-rank reduction.
        # Only the collective's metric result is injected; collection, learner,
        # payload construction and checkpoint/resume still execute normally.
        from rl.training.monitoring import HP_ORDER_PREFIX as hp_prefix
        reduce_scalars = distributed_sb3.average_scalar_dict
        def unequal_rank_counts(values, transport):
            result = reduce_scalars(values, transport)
            if hp_prefix + '_fraction' in result:
                result.update({hp_prefix + '_count': 50.0,
                               hp_prefix + '_matches': .5,
                               hp_prefix + '_fraction': .5})
            return result
        monkeypatch.setattr(distributed_sb3, 'average_scalar_dict', unequal_rank_counts)
    output = tmp_path / 'trained.zip'
    argv = ['distributed_sb3', '--rank', '0', '--world-size', '1',
        '--checkpoint', str(checkpoint), '--output', str(output), '--updates', '2',
        '--n-envs', '2', '--n-steps', '16', '--batch-size', '8', '--n-epochs', '1',
        '--gradient-accumulation-steps', '1', '--device', device, '--amp', amp, '--vec', vector,
        '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning',
        '--remote-root', str(tmp_path / 'transport'), '--ssh-control-dir', str(tmp_path / 'ssh'),
        '--run-id', 'ci-lifecycle', '--wandb', '--wandb-id', 'ci-existing-run',
        '--training-state-dir', str(tmp_path/'state'), '--save-every-updates', '1',
        '--max-checkpoints', '3', '--stop-file', str(stop_file),
        '--rollout-resource-interval', '0.1']
    if staggered:
        argv += ['--startup-stagger-steps', '16']
    if occupancy:
        argv += [
            '--startup-occupancy-max-age', '12',
            '--startup-occupancy-budget', '160',
        ]
    if contract == 'autoregressive-v2':
        argv.append('--nosl-is-az-fixed-worldline')
    if joint:
        argv.append('--nosl-is-az-train-shared-embeddings')
    if upper_micro_override is not None:
        argv += ['--upper-micro-batch-size', str(upper_micro_override)]
    if upper_lambda_override is not None:
        argv += ['--upper-gae-lambda', str(upper_lambda_override)]
    if online_small == 'reference-ce':
        argv += ['--small-playout-batch-size', '12', '--small-playout-micro-batch-size', '6']
    if fixed_lower:
        argv += [
            '--nosl-is-az-combat-strategy', fixed_strategy,
            '--no-nosl-is-az-replay',
            '--rulebase-planner-worlds', '1',
            '--rulebase-planner-nodes', '8',
            '--rulebase-planner-beam', '2',
            '--rulebase-planner-depth', '4',
            '--rulebase-planner-turns', '1',
            '--rulebase-planner-leaf-profile', 'legacy-v8',
            '--rulebase-planner-nonce', '20260918',
        ]
        monkeypatch.setattr(
            distributed_sb3, 'distributed_nosl_is_az_update',
            lambda *a, **kw: pytest.fail(
                'fixed lower reached distributed NOSL learner'
            ),
        )
    if step_schedule:
        argv += ['--nosl-is-az-update-schedule',
                 '{"mode":"step","hold_rollouts":2,"final_updates":1}']
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    assert len(payloads) == 2
    assert [p['global_step'] for p in payloads] == [32, 64]
    assert [p['legacy_step'] for p in payloads] == [1, 2]
    assert all(p['dist/avg_grad_sec'] == 0 for p in payloads)
    assert all('game/episodes_window' in p for p in payloads)
    if chests:
        assert all(p['game/chest_opportunities_window'] > 0 for p in payloads)
        assert all(p['game/chest_open_rate'] == p['game/chest_opened_window'] /
                   p['game/chest_opportunities_window'] for p in payloads)
        assert any(p.get('game/chest_relic_opportunities_window', 0) > 0 for p in payloads)
    assert all(
        ('game/layer_5_win_rate' in p) == (p['game/episodes_window'] > 0)
        for p in payloads
    )
    from rl.training.monitoring import HP_ORDER_PREFIX, hp_order_fraction
    assert all(p[HP_ORDER_PREFIX + '_fraction'] == pytest.approx(
        hp_order_fraction(p) or 0.0) for p in payloads)
    if nosl:
        assert all(p[HP_ORDER_PREFIX + '_fraction'] == .01 for p in payloads)
        assert all(p['nosl_is_az/selection_prefix_decisions'] +
                   p['nosl_is_az/simulator_transitions'] == 32 for p in payloads)
        assert all('nosl_is_az/overworld_rule_decisions' in p and
                   'nosl_is_az/overworld_fallback_decisions' in p for p in payloads)
        assert all('nosl_is_az/worker_packet_count' in p and
                   'nosl_is_az/worker_native_leaf_seconds' in p and
                   'nosl_is_az/worker_pack_seconds' in p for p in payloads)
    if staggered:
        assert payloads[0]['startup_stagger/span_steps'] == 16
        assert payloads[0]['startup_stagger/discarded_transitions_local'] == 32
    if occupancy:
        # Retired CLI fields remain readable but no discarded warmup runs.
        assert not any(key.startswith('startup_occupancy/') for p in payloads for key in p)
    assert identities[0]['id'] == 'ci-existing-run'
    assert finished == [0]
    restored = AccumulatingMaskablePPO.load(output, device='cpu')
    if upper_micro_override is not None:
        assert restored.policy.core.upper_config.micro_batch_size == upper_micro_override
        assert restored.policy.upper_config['micro_batch_size'] == upper_micro_override
        assert restored.policy_kwargs['upper_config']['micro_batch_size'] == upper_micro_override
    if upper_lambda_override is not None:
        assert restored.policy.core.upper_config.gae_lambda == upper_lambda_override
        assert restored.policy.upper_config['gae_lambda'] == upper_lambda_override
        assert restored.policy_kwargs['upper_config']['gae_lambda'] == upper_lambda_override
    if contract == 'autoregressive-v2':
        assert (not fixed_calls if fixed_lower
                else fixed_calls and all(value is True for value in fixed_calls))
        assert restored.nosl_is_az_fixed_worldline is True
    assert restored.nosl_is_az_upper_strategy == upper
    assert restored.nosl_is_az_merge_equivalent_actions is merge
    if nosl and not merge:
        assert restored.nosl_action_group_contract == 'individual-actions'
    if combat_arch == 'entity':
        assert restored.policy.core.config.quantiles == 8
        for payload in payloads if not fixed_lower else ():
            assert 0 < payload['nosl_is_az/win_probability'] < 1
            assert payload['nosl_is_az/win_bce'] > 0
    else:
        assert restored.policy_kwargs['combat_arch'] == combat_arch
    assert restored.num_timesteps == 64
    if online_small:
        steps_before_resume = restored.small_playout_optimizer_steps
        assert steps_before_resume == sum(p['small_playout_train/updates'] for p in payloads)
        assert {int(s['step']) for s in restored.policy.small_playout_optimizer.state.values()} == {steps_before_resume}
        if online_small == 'reference-ce':
            assert isinstance(restored.policy.small_playout_optimizer, torch.optim.AdamW)
            assert restored.policy.small_playout_config['target_mix'] == .2
            assert all(p['small_playout_train/target_entropy'] >= 0 for p in payloads)
            assert restored.policy.small_playout_config['batch_size'] == 12
            assert restored.policy_kwargs['small_playout_config']['micro_batch_size'] == 6
            assert restored.policy.upper_config == model.policy.upper_config
            assert all(p['small_playout_train/batch_size_per_rank'] == 12 for p in payloads)
            assert all(p['small_playout_train/batch_size_global'] == 12 for p in payloads)
            assert all(p['small_playout_train/micro_batch_size'] == 6 for p in payloads)
            assert all('train/effective_batch_size_global' not in p for p in payloads)
            assert all(p['upper/effective_batch_size_global'] == p['upper/samples'] for p in payloads)
    if not fixed_lower:
        assert restored.policy.optimizer.state
    bundle = tmp_path/'state'/'rank0'/'step-000000000064'
    assert (bundle/'learner.pt').is_file()
    if nosl and not merge:
        state = torch.load(bundle/'learner.pt', map_location='cpu', weights_only=False)['replay']
        assert state['size'] > 0
        import numpy as np
        groups = state['arrays']['action_groups'][:state['size']]
        # Compact replay stores the 60 card/target slots starting at policy id 1.
        np.testing.assert_array_equal(groups, np.broadcast_to(np.arange(1, 61), groups.shape))
    if combat_arch == 'entity':
        manifest=json.loads((bundle/'manifest.json').read_text())
        assert manifest['config']['entity_config']['quantiles']==8
        assert manifest['config']['combat_supervision_contract']=='combat-outcome-quantiles-v1'
        if contract == 'autoregressive-v2':
            assert manifest['config']['nosl_is_az_fixed_worldline'] is True
    resume_argv = list(sys.argv)
    if upper_micro_override is not None:
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['config']['upper_config']['micro_batch_size'] == upper_micro_override
        # Omit the override: ordinary resume must preserve it from the ZIP.
        index = resume_argv.index('--upper-micro-batch-size')
        del resume_argv[index:index + 2]
    if upper_lambda_override is not None:
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['config']['upper_config']['gae_lambda'] == upper_lambda_override
        index = resume_argv.index('--upper-gae-lambda')
        del resume_argv[index:index + 2]
    monkeypatch.setattr(sys, 'argv', [*resume_argv, '--resume-state', str(bundle)])
    distributed_sb3.main()
    assert [p['global_step'] for p in payloads] == [32, 64, 96, 128]
    assert payloads[-1]['legacy_step'] == 4
    if chests:
        assert payloads[-1]['game/chest_opportunities_window'] > 0
        assert 'game/chest_open_rate' in payloads[-1]
    if combat_arch == 'entity' and not fixed_lower:
        assert [p['nosl_is_az/schedule_round'] for p in payloads] == [1, 2, 3, 4]
        assert [p['nosl_is_az/scheduled_updates'] for p in payloads] == ([3, 3, 1, 1] if step_schedule else [3, 2, 1, 1])
        assert [p['nosl_is_az/value_only_updates'] for p in payloads] == ([0, 0, 0, 0] if step_schedule else [2, 1, 0, 0])
        assert [p['nosl_is_az/joint_updates'] for p in payloads] == ([3, 3, 1, 1] if step_schedule else [1, 1, 1, 1])
        final = AccumulatingMaskablePPO.load(output, device='cpu')
        optimizer = final.policy.optimizer
        for parameter in final.policy.core.policy_only_parameters():
            assert optimizer.state[parameter]['step'].item() == (8 if step_schedule else 4)
        assert optimizer.state[final.policy.core.combat.distribution.win.weight]['step'].item() == (8 if step_schedule else 7)
        if step_schedule:
            from rl.algorithms.update_schedule import CombatUpdateSchedule
            assert CombatUpdateSchedule.from_config(final.nosl_is_az_update_schedule).plan(4, 3) == (1, 0)
    assert finished == [0, 0]
    if joint:
        assert all('upper/samples' in p and 'upper/updates' in p for p in payloads)
        for kind in ('gold', 'potion'):
            prefix = 'upper/reward_' + kind + '_'
            assert all(prefix + 'resolved' in p and prefix + 'policy_decisions' in p for p in payloads)
            for p in payloads:
                if p[prefix + 'resolved']:
                    assert p[prefix + 'pickup_rate'] == pytest.approx(
                        p[prefix + 'taken'] / p[prefix + 'resolved'])
                else:
                    assert prefix + 'pickup_rate' not in p
                if p[prefix + 'policy_decisions']:
                    assert p[prefix + 'policy_probability'] == pytest.approx(
                        p[prefix + 'policy_probability_sum'] / p[prefix + 'policy_decisions'])
                else:
                    assert prefix + 'policy_probability' not in p
        assert sum(p['upper/reward_gold_policy_decisions'] for p in payloads) > 0
        assert sum(p['upper/reward_gold_resolved'] for p in payloads) > 0
        assert all(p['upper/bootstrap_precision_bits'] == 32 and p['upper/gae_precision_bits'] == 64
                   and p['upper/target_precision_version'] == 1 for p in payloads)
        assert all(p['upper/bootstrap_samples'] <= p['upper/samples'] for p in payloads)
        assert sum(p['upper/updates'] for p in payloads) > 0
        assert all(p['upper/grad_norm'] > 0 and p['upper/parameter_l2_change'] > 0 for p in payloads)
        assert all('upper/policy_loss_epoch_1' in p and 'upper/clip_fraction' in p for p in payloads)
        final = AccumulatingMaskablePPO.load(output, device='cpu')
        if upper_micro_override is not None:
            assert final.policy.core.upper_config.micro_batch_size == upper_micro_override
            assert final.policy_kwargs['upper_config']['micro_batch_size'] == upper_micro_override
        if upper_lambda_override is not None:
            assert final.policy.core.upper_config.gae_lambda == upper_lambda_override
            assert final.policy_kwargs['upper_config']['gae_lambda'] == upper_lambda_override
        assert final.policy.upper_optimizer.state_dict()['state']
        if fixed_lower:
            assert final.nosl_is_az_combat_strategy == fixed_strategy
            if fixed_strategy != 'small-playout':
                assert all(p['train/optimizer_steps'] == 0 for p in payloads)
            if fixed_strategy == 'small-playout':
                assert all('train/optimizer_steps' not in p for p in payloads)
                assert all(p['small_playout_rank0/common/simulations'] > 0 for p in payloads)
                assert all('game/completed_full_games_global' in p for p in payloads)
                if online_small:
                    import numpy as np
                    assert not hasattr(final, 'small_playout_pair')
                    assert not np.array_equal(final.policy.small_playout.export(), initial_small)
                    assert all(p['small_playout_train/parameter_l2_change'] > 0 for p in payloads)
                    total = sum(p['small_playout_train/updates'] for p in payloads)
                    assert final.small_playout_optimizer_steps == total > steps_before_resume
                    assert {int(s['step']) for s in final.policy.small_playout_optimizer.state.values()} == {total}
                    assert all(p.get('small_playout_rank0/common/prior_calls', 0) == 0 for p in payloads)
                else:
                    assert final.small_playout_pair['sha256'] == model.small_playout_pair['sha256']
                    for key in ('policy', 'prior'):
                        import numpy as np
                        np.testing.assert_array_equal(final.small_playout_pair[key], model.small_playout_pair[key])
                assert any(not torch.equal(value.cpu(), initial_upper[name])
                           for name, value in final.policy.core.upper_tokenizer.state_dict().items()
                           if name.startswith('card.'))
            else:
                assert all(p['nosl_is_az/rulebase_lower_decisions'] > 0 for p in payloads)
            assert all(p['nosl_is_az/rulebase_lower_batch_seconds'] > 0
                       for p in payloads)
            for name, value in final.policy.core.combat.state_dict().items():
                torch.testing.assert_close(value, initial_combat[name], rtol=0, atol=0)
    if contract == 'autoregressive-v2':
        assert not fixed_calls if fixed_lower else all(
            value is True for value in fixed_calls
        )
    assert (tmp_path/'state'/'rank0'/'step-000000000128'/'manifest.json').is_file()
    assert all(p['dist/iteration_wall_sec'] >= p['dist/checkpoint_sec'] >= 0 for p in payloads)
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith('{') and 'rollout_resources' in line]
    assert len(records)==4
    if online_small:
        import hashlib
        hashes = [r['small_playout_collection_sha256'] for r in records]
        assert hashes[0] == hashlib.sha256(initial_small.tobytes()).hexdigest()
        assert len(set(hashes)) == 4
        assert hashes[2] == hashlib.sha256(restored.policy.small_playout.export().tobytes()).hexdigest()
    # The collector must carry actual native work through reduction and both
    # reporting sinks, including the two rounds after checkpoint recovery.
    # Pin the public metric contract so deleting a key cannot weaken this test.
    assert set(NATIVE_TIMING_KEYS) == {
        'native_begin_seconds', 'native_advance_seconds', 'native_finish_seconds',
        'native_tree_seconds', 'native_pack_seconds',
    }
    for record, payload in zip(records, payloads, strict=True):
        for name in NATIVE_TIMING_KEYS:
            key = f'nosl_is_az/{name}'
            assert record[key] == payload[key], key
            if nosl and vector == 'native' and not fixed_lower:
                assert record[key] > 0, key
            else:
                assert record[key] == 0, key
    assert all(r['rollout_resources']['sample_count']>=2 for r in records)
    assert all(r['rollout_resources']['tree_rss_peak_sampled_bytes']>0 for r in records)
    assert all((r['rollout_resources']['gpu_devices'] is None) == (device == 'cpu') for r in records)
    for record in records:
        collection, iteration = record['rollout_resources'], record['iteration_resources']
        assert 0 < collection['elapsed_seconds'] < iteration['elapsed_seconds'] <= record['iteration_wall_sec']
        assert iteration['sample_count'] > collection['sample_count']
        assert iteration['tree_cpu_seconds'] >= collection['tree_cpu_seconds']
        assert iteration['tree_rss_peak_sampled_bytes'] >= collection['tree_rss_peak_sampled_bytes']
        assert (iteration['gpu_devices'] is None) == (device == 'cpu')
        for resources in (collection, iteration):
            for name in ('allocated', 'reserved'):
                current = resources[f'torch_{name}_bytes']
                peak = resources[f'torch_{name}_peak_sampled_bytes']
                if device == 'cpu':
                    assert current is None and peak is None
                    assert resources['torch_memory_unavailable_reason']
                else:
                    assert 0 < current <= peak
                    assert resources['torch_memory_unavailable_reason'] is None
                    assert resources['torch_device'].startswith('cuda:')

    if staggered:
        stop_file.unlink()
        fork_output = tmp_path / 'forked.zip'
        fork_state = tmp_path / 'fork-state'
        fork_stop = tmp_path / 'fork-stop'
        fork_source = tmp_path/'state'/'rank0'/'step-000000000128'
        monkeypatch.setattr(sys, 'argv', [
            *argv,
            '--run-id', 'ci-fork',
            '--wandb-id', 'ci-fork-run',
            '--wandb-name', 'ci-fork',
            '--fork-state', str(fork_source),
            '--updates', '1',
            '--output', str(fork_output),
            '--training-state-dir', str(fork_state),
            '--stop-file', str(fork_stop),
        ])
        distributed_sb3.main()
        assert payloads[-1]['global_step'] == 160
        assert identities[-1]['id'] == 'ci-fork-run'
        fork_bundle = fork_state/'rank0'/'step-000000000160'
        manifest = json.loads((fork_bundle/'manifest.json').read_text())
        assert manifest['run_id'] == 'ci-fork'
        assert manifest['wandb_id'] == 'ci-fork-run'
        assert manifest['config']['fork_source_global_step'] == 128

    if (nosl and combat_arch == 'entity' and contract == 'autoregressive-v2'
            and not step_schedule and not fixed_lower):
        stop_file.unlink()
        fork_root = tmp_path / 'hyperparameter-fork'
        source = tmp_path / 'state' / 'rank0' / 'step-000000000128'
        source_learner = torch.load(source / 'learner.pt', weights_only=False)
        restore = distributed_sb3.restore_learner
        restored_states = []

        def observe_restore(model, *args, **kwargs):
            result = restore(model, *args, **kwargs)
            restored_states.append((
                model._recovery_environment_restored,
                model._nosl_is_az_replay.total_added,
                model._nosl_is_az_replay.size,
            ))
            return result

        monkeypatch.setattr(distributed_sb3, 'restore_learner', observe_restore)
        fork_argv = [*argv, '--run-id', 'ci-hyperparameters',
            '--wandb-id', 'ci-hyperparameters', '--fork-state', str(source),
            '--training-state-dir', str(fork_root), '--updates', '2',
            '--learning-rate', '0.0001', '--max-grad-norm', '1',
            '--nosl-is-az-updates', '2', '--nosl-is-az-update-schedule', 'constant',
            '--nosl-is-az-batch-size', '16', '--nosl-is-az-micro-batch-size', '4',
            '--nosl-is-az-replay-max-old-fraction', '0.75',
            '--nosl-is-az-policy-coef', '0.75', '--nosl-is-az-value-coef', '1.25']
        monkeypatch.setattr(sys, 'argv', fork_argv)
        distributed_sb3.main()
        # The short scenario fixture may emit schema 1 (no game journal).
        assert restored_states == [(source_learner['schema_version'] >= 2,
                                    source_learner['replay']['total_added'],
                                    source_learner['replay']['size'])]
        assert [p['global_step'] for p in payloads[-2:]] == [160, 192]
        assert [p['legacy_step'] for p in payloads[-2:]] == [5, 6]
        for payload in payloads[-2:]:
            assert payload['nosl_is_az/learning_rate'] == pytest.approx(0.0001)
            assert payload['nosl_is_az/updates'] == 2
            assert payload['nosl_is_az/replay_old_fraction'] == pytest.approx(0.75)
        forked = AccumulatingMaskablePPO.load(output, device='cpu')
        assert forked.nosl_is_az_policy_coef == 0.75
        assert forked.nosl_is_az_value_coef == 1.25
        assert forked.max_grad_norm == 1
        assert forked.n_steps == 16 and forked.n_envs == 2
        assert forked.nosl_is_az_simulations == 4
        assert forked.nosl_is_az_fixed_worldline is True
        for parameter in forked.policy.core.policy_only_parameters():
            assert forked.policy.optimizer.state[parameter]['step'].item() == 8
        assert forked.policy.optimizer.state[
            forked.policy.core.combat.distribution.win.weight]['step'].item() == 11
        manifest = json.loads((fork_root / 'rank0' / 'step-000000000192' /
                               'manifest.json').read_text())
        assert manifest['run_id'] == manifest['wandb_id'] == 'ci-hyperparameters'
        assert manifest['config']['fork_source_global_step'] == 128
        assert manifest['config']['nosl_is_az_policy_coef'] == 0.75
        assert manifest['config']['nosl_is_az_value_coef'] == 1.25


@pytest.mark.parametrize('nosl', [False, True], ids=['ppo', 'entity-nosl'])
def test_cpu_native_lifecycle(tmp_path, monkeypatch, capsys, short_combat_env_type, nosl):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        nosl, 'rules' if nosl else 'model', 'entity' if nosl else 'transformer',
        'autoregressive-v2' if nosl else 'legacy', vector='native')


def test_joint_step_schedule_survives_resume_at_boundary(tmp_path, monkeypatch, capsys, short_combat_env_type):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'rules', 'entity', 'autoregressive-v2', vector='native',
        merge=False, step_schedule=True)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='boss BF16 lifecycle requires CUDA'))])
def test_boss_utility_lifecycle(tmp_path, monkeypatch, capsys, short_combat_env_type, device):
    import slaythespire as sts
    class BossEnv(short_combat_env_type):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.thick.gc.cur_room = sts.Room.BOSS
            self._invalidate_backend_cache()
            return self._obs(), self._info()
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, BossEnv, True, 'rules', 'entity',
        'autoregressive-v2', vector='native', boss_weight=.9, device=device,
        amp='bf16' if device == 'cuda' else 'off')
    bundle=tmp_path/'state'/'rank0'/'step-000000000128'
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert manifest['config']['entity_config']['boss_win_weight']==.9
    replay=torch.load(bundle/'learner.pt',map_location='cpu',weights_only=False)['replay']
    assert (replay['observations']['combat_is_boss'] > 0).any()


def test_blue_individual_search_lifecycle(tmp_path, monkeypatch, capsys, short_combat_env_type):
    class VisiblePowerEnv(short_combat_env_type):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.thick.bc.set_player_status('BARRICADE', 1)
            self._invalidate_backend_cache()
            return self._obs(), self._info()
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, VisiblePowerEnv,
        True, 'rules', 'entity', 'autoregressive-v2', vector='native', merge=False)
    from env.core import encode
    bundle = tmp_path / 'state' / 'rank0' / 'step-000000000128'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['observation_version'] == encode.OBS_VERSION == 20
    replay = torch.load(bundle / 'learner.pt', map_location='cpu', weights_only=False)['replay']
    index = encode.V.PLAYER_BUFF_VOCAB_INDEX['Barricade']
    assert (replay['observations']['player_buffs'][:, index] > 0).any()


@pytest.mark.parametrize('failure_stage', ['learner', 'checkpoint'])
@pytest.mark.parametrize('monitor_exit_fails', [False, True])
def test_iteration_monitor_closes_on_training_failure(tmp_path, monkeypatch, capsys, short_combat_env_type, failure_stage, monitor_exit_fails):
    from rl.training import resource_monitor
    from rl.training.native_vec_env import NativeSearchVecEnv

    monitors, closed_environments = [], []
    class ObservedMonitor(resource_monitor.ResourceMonitor):
        def __enter__(self):
            monitors.append(self)
            return super().__enter__()

        def __exit__(self, *args):
            super().__exit__(*args)
            if monitor_exit_fails:
                raise RuntimeError('injected monitor exit failure')

    monkeypatch.setattr(resource_monitor, 'ResourceMonitor', ObservedMonitor)
    close = NativeSearchVecEnv.close
    def observed_close(env):
        if monitors:
            closed_environments.append(env)
        return close(env)
    monkeypatch.setattr(NativeSearchVecEnv, 'close', observed_close)
    def fail():
        assert len(monitors) == 1
        assert monitors[0]._thread.is_alive(), 'sampling must cover the learner and checkpoint'
        raise RuntimeError(f'injected {failure_stage} failure')

    if failure_stage == 'learner':
        original = distributed_sb3.distributed_nosl_is_az_update
        def failed_update(*args, **kwargs):
            original(*args, **kwargs)
            fail()
        monkeypatch.setattr(distributed_sb3, 'distributed_nosl_is_az_update', failed_update)
    else:
        original = torch.save
        generation = tmp_path / 'state' / 'rank0' / 'step-000000000032'
        def failed_save(value, destination, *args, **kwargs):
            if destination == generation / 'learner.pt':
                assert (generation / 'model.zip').is_file()
                fail()
            return original(value, destination, *args, **kwargs)
        monkeypatch.setattr(torch, 'save', failed_save)
    expected_error = 'injected monitor exit failure' if monitor_exit_fails else f'injected {failure_stage} failure'
    with pytest.raises(RuntimeError, match=expected_error):
        test_training_reaches_wandb_and_checkpoint(
            tmp_path, monkeypatch, capsys, short_combat_env_type,
            True, 'rules', 'entity', 'autoregressive-v2', vector='native')
    assert len(monitors) == 1
    assert not monitors[0]._thread.is_alive()
    assert closed_environments, 'telemetry cleanup failure must not bypass environment cleanup'
    assert monitors[0].summary['sample_count'] >= 2
    assert not (tmp_path / 'state' / 'rank0' / 'latest').exists()
    assert not (tmp_path / 'state' / 'rank0' / 'step-000000000032').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='BF16 learner lifecycle requires CUDA')
@pytest.mark.parametrize('nosl,merge', [(False, True), (True, True), (True, False)],
                         ids=['ppo', 'nosl-grouped', 'nosl-blue-individual'])
def test_cuda_native_lifecycle(tmp_path, monkeypatch, capsys, short_combat_env_type, nosl, merge):
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        test_training_reaches_wandb_and_checkpoint(
            tmp_path, monkeypatch, capsys, short_combat_env_type,
            nosl, 'rules' if nosl else 'model', 'entity' if nosl else 'transformer',
            'autoregressive-v2' if nosl else 'legacy', device='cuda', amp='bf16', vector='native', merge=merge)
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='BF16 joint lifecycle requires CUDA'))])
def test_joint_wholerun_lifecycle(tmp_path, monkeypatch, capsys, short_combat_env_type, device):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'model', 'entity', 'autoregressive-v2', device=device,
        amp='bf16' if device == 'cuda' else 'off', vector='native', joint=True)


def test_joint_wholerun_fixed_rulebase_lower_lifecycle(
        tmp_path, monkeypatch, capsys, short_combat_env_type):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'model', 'entity', 'autoregressive-v2', vector='native',
        joint=True, fixed_lower=True)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='BF16 small-playout lifecycle requires CUDA'))])
@pytest.mark.parametrize('online', [False, True, 'reference-ce'])
def test_joint_wholerun_small_playout_lifecycle(
        tmp_path, monkeypatch, capsys, short_combat_env_type, device, online):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'model', 'entity', 'autoregressive-v2', vector='native',
        device=device, amp='bf16' if device == 'cuda' else 'off',
        joint=True, fixed_lower='small-playout', online_small=online)


def test_upper_microbatch_override_survives_plain_resume(
        tmp_path, monkeypatch, capsys, short_combat_env_type):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'model', 'entity', 'autoregressive-v2', vector='native',
        joint=True, fixed_lower=True, upper_micro_override=5)


@pytest.mark.parametrize('mode', ['ppo', 'nosl', 'fixed-lower'])
def test_chest_metrics_training_lifecycle(tmp_path, monkeypatch, capsys, chest_env_type, mode):
    fixed = mode == 'fixed-lower'
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, chest_env_type,
        mode != 'ppo', 'model', 'entity' if fixed else 'transformer',
        'autoregressive-v2' if fixed else 'legacy', chests=True,
        vector='native' if fixed else 'dummy', joint=fixed, fixed_lower=fixed)


def test_upper_lambda_override_survives_plain_resume(
        tmp_path, monkeypatch, capsys, short_combat_env_type):
    test_training_reaches_wandb_and_checkpoint(
        tmp_path, monkeypatch, capsys, short_combat_env_type,
        True, 'model', 'entity', 'autoregressive-v2', vector='native',
        joint=True, fixed_lower=True, upper_lambda_override=1.)


@pytest.mark.parametrize('value', ['-0.01', '1.01', 'nan', 'inf'])
def test_upper_lambda_rejects_invalid_override(monkeypatch, value):
    monkeypatch.setattr(sys, 'argv', ['distributed_sb3', '--rank', '0',
        '--world-size', '1', '--checkpoint', 'unused.zip', '--upper-gae-lambda', value])
    with pytest.raises(ValueError, match='upper-gae-lambda must be finite'):
        distributed_sb3.main()
