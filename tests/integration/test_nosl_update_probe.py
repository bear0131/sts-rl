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
        assert result['branches'][0] == result['branches'][2]
        assert all(r['optimizer_steps'] == 1 for b in result['branches'] for r in b['rows'][1:])
        assert all(b['rows'][0]['policy_kl_from_start'] == 0 for b in result['branches'])
        assert read_bundle(source, rank=0, world_size=1) == original
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
