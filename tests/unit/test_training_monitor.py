from rl.training.monitoring import assess
from rl.training.monitoring import assess_relic_collapse
from tools.operations.monitor_nosl import relic_guard_check
import json
from pathlib import Path
import subprocess
import sys
import pytest


def records(values):
    return [dict(global_timesteps=(i+1)*12288, **{'game/floor_won/05':v}) for i,v in enumerate(values)]


def test_rising_noisy_and_single_bad_point_do_not_trigger():
    for values in ([.28,.51,.67,.6,.72,.7], [.7,.72,.69,.7,.68,.1], [.1]*8):
        assert assess(records(values),now=0)['proposal'] is None


def test_sustained_collapse_proposes_noise_first():
    result=assess(records([.2,.2,.19,.04,0,0]),now=0)
    assert result['status']=='collapse'
    assert result['proposal']==dict(parameter='nosl_is_az_gumbel_scale',value=1.0)


def test_interventions_wait_for_new_samples_and_time_then_move_one_coordinate():
    history=records([.7]*3+[.1]*10)
    changes=[dict(global_step=6*12288,time=0)]
    assert assess(history,changes,now=1800)['status']=='intervention_grace'
    assert assess(history[:9],changes,now=4000)['status']=='intervention_grace'
    result=assess(history,changes,now=4000)
    assert result['proposal']==dict(parameter='nosl_is_az_gumbel_value_scale',value=.1)


def test_invalid_metrics_are_not_search_parameter_evidence():
    assert assess(records([float('nan')]),now=0)['status']=='invalid_metrics'


def relic_records(values):
    return [dict(global_timesteps=(i + 1) * 786432, **{
        'game/chest_relic_take_rate': value,
        'game/chest_relic_taken_window': value * 400,
        'game/chest_relic_opportunities_window': 400,
    }) for i, value in enumerate(values)]


@pytest.mark.parametrize('values,stop', [
    ([.8] * 5 + [.39] * 3, True),
    ([.6] * 5 + [.3] * 3, True),
    ([.8] * 5 + [.1, .5, .1], False),
    ([.3] * 5 + [.14] * 3, False),  # halved but not a 20-point loss
    ([.1] * 8, False),
    ([.9] + [.1] * 7, False),  # a single high point is not a stable baseline
    ([.8] * 5 + [.1] * 2, False),
])
def test_relic_guard_needs_stable_reference_and_sustained_large_loss(values, stop):
    result = assess_relic_collapse(relic_records(values), iteration_steps=786432)
    assert result['stop'] is stop


@pytest.mark.parametrize('change', ['missing', 'small', 'zero', 'nan', 'inconsistent', 'gap'])
def test_relic_guard_does_not_treat_invalid_or_missing_windows_as_zero(change):
    rows = relic_records([.8] * 5 + [.1] * 3)
    if change == 'missing':
        rows[-2].pop('game/chest_relic_take_rate')
    elif change in ('small', 'zero'):
        rows[-2]['game/chest_relic_opportunities_window'] = 100 if change == 'small' else 0
    elif change == 'nan':
        rows[-2]['game/chest_relic_take_rate'] = float('nan')
    elif change == 'inconsistent':
        rows[-2]['game/chest_relic_taken_window'] = 399
    else:
        rows[-1]['global_timesteps'] += 786432
    result = assess_relic_collapse(rows, iteration_steps=786432)
    assert not result['stop']
    assert result['status'] == 'incomplete_recent_windows'


def test_relic_guard_duplicate_steps_are_not_three_rounds():
    rows = relic_records([.8] * 5 + [.1])
    rows += [rows[-1]] * 4
    result = assess_relic_collapse(rows, iteration_steps=786432)
    assert not result['stop'] and result['points'] == 6


def test_relic_guard_persists_reference_across_log_rotation_and_never_overwrites_stop(tmp_path):
    log, state, stop = (tmp_path / n for n in ('rank0.log', 'guard.json', 'STOP'))
    rows = relic_records([.8] * 5 + [.1] * 3)
    log.write_text('\n'.join(map(json.dumps, rows[:5])))
    assert not relic_guard_check(log, state, stop, iteration_steps=786432)['stop']
    log.write_text('\n'.join(map(json.dumps, rows[5:])))
    result = relic_guard_check(log, state, stop, iteration_steps=786432)
    assert result['stop'] and result['reference'] == pytest.approx(.8)
    saved = stop.read_bytes()
    assert json.loads(saved)['reason'] == 'sustained_chest_relic_take_collapse'
    assert relic_guard_check(log, state, stop, iteration_steps=786432)['status'] == 'intentional_stop'
    assert stop.read_bytes() == saved
    with pytest.raises(ValueError, match='another run or configuration'):
        relic_guard_check(log, state, stop, iteration_steps=1)


def test_relic_guard_cli_writes_real_stop_only_after_three_distinct_low_rounds(tmp_path):
    log, state, stop = (tmp_path / n for n in ('rank0.log', 'guard.json', 'STOP'))
    rows = relic_records([.8] * 5 + [.1] * 3)
    command = [sys.executable, '-m', 'tools.operations.monitor_nosl', '--log', str(log),
               '--state', str(state), '--relic-guard', '--stop-file', str(stop),
               '--iteration-steps', '786432']
    root = Path(__file__).resolve().parents[2]
    log.write_text('\n'.join(map(json.dumps, rows[:7])) + '\n{"partial":')
    for _ in range(2):
        first = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10)
        assert first.returncode == 0, first.stderr
        assert not stop.exists()
    log.write_text('\n'.join(map(json.dumps, rows)))
    final = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10)
    assert final.returncode == 0, final.stderr
    assert json.loads(final.stdout)['stop']
    assert stop.exists()
