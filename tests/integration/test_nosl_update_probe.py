"""A diagnostic may update its clone, but must never modify a source bundle."""
import pytest
from types import SimpleNamespace
import sys

from rl.training.checkpoints import read_bundle, save_bundle
from tests.integration.test_training_smoke import train_once
from tools.analysis.probe_nosl_updates import run_probe
from tools.analysis import probe_nosl_updates

pytestmark = [pytest.mark.native, pytest.mark.training]


def test_update_probe_uses_real_learner_and_preserves_source(tmp_path, short_combat_env_type, monkeypatch):
    model, env = train_once(True, env_type=short_combat_env_type,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_simulations=4, nosl_is_az_replay_capacity=128,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1)
    try:
        model._prepare_nosl_is_az_replay()
        source = save_bundle(model, tmp_path/'source', rank=0, world_size=1,
                             run_id='do-not-restart', wandb_id='same', config={})
        original = read_bundle(source, rank=0, world_size=1)
        output = tmp_path/'probe.json'
        result = run_probe(source, output, batch_sizes=[8, 16, 8], updates=2, per_stratum=4)
        assert result['state'] == 'complete'
        assert [len(b['rows']) for b in result['branches']] == [3, 3, 3]
        def without_timing(branch):
            return {**branch, 'rows': [{k: v for k, v in r.items() if k != 'fit_seconds'}
                                      for r in branch['rows']]}
        assert without_timing(result['branches'][0]) == without_timing(result['branches'][2])
        assert all(r['optimizer_steps'] == 1 for b in result['branches'] for r in b['rows'][1:])
        assert all(b['rows'][0]['policy_kl_from_start'] == 0 for b in result['branches'])
        assert read_bundle(source, rank=0, world_size=1) == original
        # Later audit labels must not enter optimization of the original clone.
        replay = model._ensure_nosl_is_az_replay()
        replay.sample_ids[:replay.size] += replay.total_added
        replay.total_added += replay.size
        replay.value_targets[:replay.size] = 0
        model.num_timesteps += 1
        later = save_bundle(model, tmp_path/'later', rank=0, world_size=1,
                            run_id='do-not-restart', wandb_id='same', config={})
        from rl.evaluation.nosl.hp_sensitivity import build_terminal_corpus, save_corpus
        terminal_path = tmp_path/'terminals.npz'
        save_corpus(build_terminal_corpus(model, ['Cultist'], [8], hps=[1, 80]), terminal_path)
        audited = run_probe(source, tmp_path/'later-audit.json', batch_sizes=[8], updates=1,
                            per_stratum=4, audit_bundle=later, terminal_corpus=terminal_path)
        assert audited['audit']['source_step'] == model.num_timesteps
        assert audited['branches'][0]['rows'][1]['fit_value_mae'] == result['branches'][0]['rows'][1]['fit_value_mae']
        assert audited['branches'][0]['rows'][0]['strata'] != result['branches'][0]['rows'][0]['strata']
        assert [h['update'] for h in audited['branches'][0]['health']] == [0, 1]
        assert all(h['terminal']['summary']['states'] == 2 for h in audited['branches'][0]['health'])
        assert read_bundle(source, rank=0, world_size=1) == original
        rates = run_probe(source, tmp_path/'rate-controls.json', batch_sizes=[8], updates=2,
                          per_stratum=4, learning_rate_scales=[0., .5, 1.], terminal_corpus=terminal_path)
        zero, half, full = rates['branches']
        assert zero['initial_learning_rates'] == [0.]*len(full['initial_learning_rates'])
        assert half['initial_learning_rates'] == [lr*.5 for lr in full['initial_learning_rates']]
        for branch in rates['branches']:
            assert all(r['learning_rates'] == branch['initial_learning_rates'] for r in branch['rows'][1:])
            assert branch['rows'][0]['strata'] == full['rows'][0]['strata']
        assert all(r['strata'] == zero['rows'][0]['strata'] and r['policy_kl_from_start'] == 0
                   for r in zero['rows'])
        assert zero['health'][0]['terminal'] == zero['health'][-1]['terminal']
        assert full['rows'][-1]['policy_kl_from_start'] > 0
        assert read_bundle(source, rank=0, world_size=1) == original
        with pytest.raises(ValueError, match='later in the same run'):
            run_probe(source, tmp_path/'not-later.json', audit_bundle=source)
        with pytest.raises(ValueError, match='bf16 on CUDA'):
            run_probe(source, tmp_path/'cpu-bf16.json', amp='bf16')
        # The short native fixture need not visit low HP. Inject a synthetic
        # quota partition only in each private clone to exercise real updates.
        configure = probe_nosl_updates.configure_probe_sampling
        def synthetic_partition(replay, fraction):
            replay.observations['player_scalars'][0, 0] = 1/200
            replay.observations['player_scalars'][-1, 0] = .4
            return configure(replay, fraction)
        monkeypatch.setattr(probe_nosl_updates, 'configure_probe_sampling', synthetic_partition)
        balanced = run_probe(source, tmp_path/'balanced.json', batch_sizes=[8],
            low_hp_fractions=[.25], updates=1, per_stratum=4)
        assert balanced['state'] == 'complete'
        assert balanced['branches'][0]['rows'][1]['optimizer_steps'] == 1
        assert balanced['branches'][0]['sampling']['low_hp_fraction'] == .25
        assert read_bundle(source, rank=0, world_size=1) == original
        with pytest.raises(ValueError, match='new output'):
            run_probe(source, output)
        with pytest.raises(ValueError, match='outside'):
            run_probe(source, source/'probe.json')
        ticks = iter([0., 1., 2.])
        monkeypatch.setattr(probe_nosl_updates, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
        monkeypatch.setattr(sys, 'argv', ['probe', str(source), '--max-seconds', '.5',
            '--output', str(tmp_path/'partial.json')])
        with pytest.raises(SystemExit) as stopped:
            probe_nosl_updates.main()
        assert stopped.value.code == 2  # partial evidence is not a complete trial
        assert (tmp_path/'partial.json').is_file()
    finally:
        env.close()
