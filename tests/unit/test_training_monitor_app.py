import json
from types import SimpleNamespace
from unittest.mock import patch

from apps.training_monitor import Monitor, clean, merge_rows


def test_resume_rows_replace_duplicate_steps_without_losing_history():
    before = [{'global_timesteps': 10, 'loss': 1}, {'global_timesteps': 20, 'loss': 2}]
    incoming = [{'global_timesteps': 20, 'loss': 3}, {'global_timesteps': 30, 'loss': 4}]
    assert merge_rows(before, incoming) == [before[0], incoming[0], incoming[1]]


def test_nonfinite_metrics_are_missing_not_zero():
    assert clean({'loss': float('nan'), 'values': [float('inf'), -0.1]}) == {
        'loss': None, 'values': [None, -0.1]}


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
