"""Exercise the production collector → learner → logging → checkpoint lifecycle."""
import json
import sys
from types import SimpleNamespace

import pytest
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.training import distributed_sb3
from rl.training.environments import StartupStaggerWrapper
from rl.training.reward import ShapedRewardWrapper

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(90)]


@pytest.mark.parametrize('nosl,upper,combat_arch,contract', [
    (False, 'model', 'transformer', 'legacy'), (True, 'model', 'transformer', 'legacy'),
    (True, 'rules', 'transformer', 'legacy'), (True, 'rules', 'mlp', 'legacy'),
    (False, 'model', 'transformer', 'joint-v2'), (True, 'rules', 'transformer', 'joint-v2'),
    (True, 'rules', 'entity', 'joint-v2'), (True, 'rules', 'entity', 'autoregressive-v1')])
def test_training_reaches_wandb_and_checkpoint(tmp_path, monkeypatch, capsys, short_combat_env_type, nosl, upper, combat_arch, contract):
    staggered = nosl and upper == 'rules' and combat_arch == 'mlp'
    occupancy = nosl and upper == 'rules' and combat_arch == 'transformer'

    def make_env(*_):
        def factory(hold_steps):
            def build():
                env = TimeLimit(
                    ShapedRewardWrapper(short_combat_env_type(combat_action_contract=contract,card_language=combat_arch=='entity')),
                    max_episode_steps=12,
                )
                return (
                    StartupStaggerWrapper(env, hold_steps=hold_steps)
                    if staggered else env
                )
            return build

        return DummyVecEnv([factory(0), factory(16)])

    env = make_env()
    from rl.policies.entity_combat import EntityCombatPolicy
    policy = EntityCombatPolicy if combat_arch == 'entity' else SplitStSPointerPolicy
    kwargs = (dict(entity_config=dict(width=32,layers=1,heads=4,quantiles=8))
              if combat_arch == 'entity' else dict(d_model=32,n_layers=2,n_heads=4,combat_arch=combat_arch))
    model = AccumulatingMaskablePPO(
        policy, env, n_steps=16, batch_size=8, n_epochs=1,
        policy_kwargs=kwargs, device='cpu', seed=42,
        nosl_is_az=nosl, nosl_is_az_cpp=nosl, nosl_is_az_inner_only=nosl,
        nosl_is_az_upper_strategy=upper,
        nosl_is_az_simulations=4, nosl_is_az_depth=0, nosl_is_az_updates=1,
        nosl_is_az_update_schedule=(dict(mode='value-warmup', peak_updates=3,
            hold_rollouts=1, decay_rollouts=2) if combat_arch == 'entity' else None),
        nosl_is_az_replay_capacity=128, nosl_is_az_batch_size=16,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1,
        hierarchical_combat_hp=nosl)
    checkpoint = tmp_path / 'initial.zip'
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
        '--gradient-accumulation-steps', '1', '--device', 'cpu', '--vec', 'dummy',
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
    monkeypatch.setattr(sys, 'argv', argv)
    distributed_sb3.main()
    assert len(payloads) == 2
    assert [p['global_step'] for p in payloads] == [32, 64]
    assert [p['legacy_step'] for p in payloads] == [1, 2]
    assert all(p['dist/avg_grad_sec'] == 0 for p in payloads)
    assert all('game/episodes_window' in p for p in payloads)
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
    if staggered:
        assert payloads[0]['startup_stagger/span_steps'] == 16
        assert payloads[0]['startup_stagger/discarded_transitions_local'] == 32
    if occupancy:
        # Retired CLI fields remain readable but no discarded warmup runs.
        assert not any(key.startswith('startup_occupancy/') for p in payloads for key in p)
    assert identities[0]['id'] == 'ci-existing-run'
    assert finished == [0]
    restored = AccumulatingMaskablePPO.load(output, device='cpu')
    assert restored.nosl_is_az_upper_strategy == upper
    if combat_arch == 'entity':
        assert restored.policy.core.config.quantiles == 8
        for payload in payloads:
            assert 0 < payload['nosl_is_az/win_probability'] < 1
            assert payload['nosl_is_az/win_bce'] > 0
    else:
        assert restored.policy_kwargs['combat_arch'] == combat_arch
    assert restored.num_timesteps == 64
    assert restored.policy.optimizer.state
    bundle = tmp_path/'state'/'rank0'/'step-000000000064'
    assert (bundle/'learner.pt').is_file()
    if combat_arch == 'entity':
        manifest=json.loads((bundle/'manifest.json').read_text())
        assert manifest['config']['entity_config']['quantiles']==8
        assert manifest['config']['combat_supervision_contract']=='combat-outcome-quantiles-v1'
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--resume-state', str(bundle)])
    distributed_sb3.main()
    assert [p['global_step'] for p in payloads] == [32, 64, 96, 128]
    assert payloads[-1]['legacy_step'] == 4
    if combat_arch == 'entity':
        assert [p['nosl_is_az/schedule_round'] for p in payloads] == [1, 2, 3, 4]
        assert [p['nosl_is_az/scheduled_updates'] for p in payloads] == [3, 2, 1, 1]
        assert [p['nosl_is_az/value_only_updates'] for p in payloads] == [2, 1, 0, 0]
        assert [p['nosl_is_az/joint_updates'] for p in payloads] == [1, 1, 1, 1]
        final = AccumulatingMaskablePPO.load(output, device='cpu')
        optimizer = final.policy.optimizer
        for parameter in final.policy.core.policy_only_parameters():
            assert optimizer.state[parameter]['step'].item() == 4
        assert optimizer.state[final.policy.core.combat.distribution.win.weight]['step'].item() == 7
    assert finished == [0, 0]
    assert (tmp_path/'state'/'rank0'/'step-000000000128'/'manifest.json').is_file()
    assert all(p['dist/iteration_wall_sec'] >= p['dist/checkpoint_sec'] >= 0 for p in payloads)
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith('{') and 'rollout_resources' in line]
    assert len(records)==4
    assert all(r['rollout_resources']['sample_count']>=2 for r in records)
    assert all(r['rollout_resources']['tree_rss_peak_sampled_bytes']>0 for r in records)
    assert all(r['rollout_resources']['gpu_devices'] is None for r in records)

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
