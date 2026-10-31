import json
from types import SimpleNamespace
from unittest.mock import patch

from apps.training_monitor import Monitor, clean, merge_rows, allowed_client


def test_lan_access_is_explicit_and_loopback_stays_available():
    import ipaddress
    networks = [ipaddress.ip_network('192.168.20.0/24')]
    assert allowed_client('127.0.0.1', [])
    assert allowed_client('::1', [])
    assert allowed_client('192.168.20.15', networks)
    assert not allowed_client('192.168.21.15', networks)
    assert not allowed_client('8.8.8.8', networks)
    assert not allowed_client('192.168.20.15', [])


def test_resume_rows_replace_duplicate_steps_without_losing_history():
    before = [{'global_timesteps': 10, 'loss': 1}, {'global_timesteps': 20, 'loss': 2}]
    incoming = [{'global_timesteps': 20, 'loss': 3}, {'global_timesteps': 30, 'loss': 4}]
    assert merge_rows(before, incoming) == [before[0], incoming[0], incoming[1]]


def test_nonfinite_metrics_are_missing_not_zero():
    assert clean({'loss': float('nan'), 'values': [float('inf'), -0.1]}) == {
        'loss': None, 'values': [None, -0.1]}


def test_new_campfire_metrics_do_not_backfill_historical_rows():
    old = {'global_timesteps': 10}
    new = {'global_timesteps': 20, 'upper/campfire_rest_choice_rate': .25,
           'upper/campfire_rest_policy_probability': .3}
    rows = clean(merge_rows([old], [new]))
    assert rows == [old, new]
    assert 'upper/campfire_rest_choice_rate' not in rows[0]
    assert 'upper/campfire_dig_choice_rate' not in rows[1]


def test_connection_failure_preserves_cache_and_marks_stale(tmp_path):
    config = dict(run_id='example', iteration_steps=10, ports=[2200], ssh_key='key',
                  root='/repo', host='root@example.com')
    output = tmp_path / 'state.json'
    output.write_text(json.dumps(dict(run_id='example', rows=[{'global_timesteps': 10}],
                                     ranks=[dict(rank=0, checked_at=100, alive=True,
                                                 cursor=dict(offset=20, inode=1))])))
    m = Monitor(config, output, 30)
    with patch('apps.training_monitor.subprocess.run', side_effect=OSError('unreachable')):
        m.poll()
    snapshot = json.loads(m.snapshot())
    assert snapshot['rows'] == [{'global_timesteps': 10}]
    assert snapshot['ranks'][0]['checked_at'] == 100
    assert snapshot['ranks'][0]['cursor']['offset'] == 20
    assert snapshot['ranks'][0]['connection_error'] == 'unreachable'
    assert json.loads(output.read_text()) == snapshot
    other = Monitor(dict(config, run_id='another'), output, 30)
    assert other.state['rows'] == []


def test_ssh_stdin_uses_literal_lf_on_windows(tmp_path):
    config = dict(run_id='example', iteration_steps=10, ports=[2200], ssh_key='key',
                  root='/repo with space', host='root@example.com')
    monitor = Monitor(config, tmp_path / 'state.json', 30)
    with patch('apps.training_monitor.subprocess.run', return_value=SimpleNamespace(
            returncode=0, stdout=b'{"rank": 0, "alive": true}', stderr=b'')) as run:
        assert monitor.read_rank(0)['alive']
    script = run.call_args.kwargs['input']
    assert isinstance(script, bytes) and b'\r' not in script
    assert script.startswith(b"cd '/repo with space'\n")
    assert script.endswith(b'\nPY_MONITOR\n')


def test_comparison_keeps_same_steps_separate_and_survives_restart(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text(json.dumps(dict(run_id='old', iteration_steps=10,
                                  rows=[dict(global_timesteps=20, loss=1.)])))
    config = dict(run_id='new', iteration_steps=10, ports=[2200], branch_step=20,
                  comparisons=[dict(run_id='old', history_file=str(old), branch_step=10)])
    output = tmp_path / 'new.json'
    output.write_text(json.dumps(dict(run_id='new', rows=[dict(global_timesteps=20, loss=2.)])))
    m = Monitor(config, output, 30)
    snapshot = json.loads(m.snapshot())
    assert snapshot['rows'][0]['loss'] == 2.
    assert snapshot['comparisons'][0]['rows'][0]['loss'] == 1.
    assert snapshot['branch_step'] == 20
    assert snapshot['comparisons'][0]['branch_step'] == 10
    assert json.loads(old.read_text())['rows'][0]['loss'] == 1.
